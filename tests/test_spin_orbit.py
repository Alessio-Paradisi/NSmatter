"""Spin-orbit force v_ls L.S + v_lst L.S tau.tau.

Interface (src/nsmatter/hamiltonian/momentum.py):
    S_PAIR                                   total pair spin (sigma_i + sigma_j)/2, (3, 4, 4)
    make_ls_potential(system, radial, tol)   radial = {"ls": terms, "lst": terms}, Gaussians
Run with:  uv run pytest -q tests/test_spin_orbit.py
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import minimum_image
from nsmatter.hamiltonian.momentum import make_ls_potential
from nsmatter.hamiltonian.pair_operators import DOT, PAULI
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated

RADIAL = {"ls": ((-30.0, 0.9),), "lst": ((12.0, 1.1),)}


def random_state(system, seed):
    """Operator-correlated state (complex, mixes S_z) with random parameters."""
    params, log_psi = make_correlated(system, n_terms=3)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    params = {"jastrow": 0.3 * jax.random.normal(k1, params["jastrow"].shape),
              "corr": 0.3 * jax.random.normal(k2, params["corr"].shape)}
    r = jax.random.uniform(k3, (system.n_particles, 3), maxval=system.box_length)
    z = jax.random.normal(k4, (system.n_particles, 2, 2))
    xi = z[..., 0] + 1j * z[..., 1]
    xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
    return params, log_psi, r, xi, jnp.asarray(system.isospins)


# =============================================================================
# Test 1 - Against explicit matrices and finite differences
# =============================================================================
def brute_force_ls(system, params, log_psi, r, xi, t, h=1e-4):
    """(V_LS Psi)(S) / Psi(S) the long way:
      - Psi_b at every discrete spin-isospin state b and its position gradient
        by central finite differences (no automatic differentiation);
      - L.S as explicit matrices: S_a from Pauli matrices, (r x p)_a by hand;
      - images summed in R = sum_n v(r_n) r_n over a large explicit cube;
      - contracted with the spinor components c_b(S).
    """
    N, L = system.n_particles, system.box_length
    t_cur = tuple(int(x) for x in t)
    spins = list(itertools.product([1, -1], repeat=N))
    isos = sorted(set(itertools.permutations(t_cur)))
    # Psi at pure up/down states can hit an exact zero of the Slater part
    # (where the fast updates are undefined), so evaluate Psi in a random
    # generic spinor basis w_q0, w_q1 for each particle and convert by
    # linearity: e_b = sum_k X[b, k] w_k with X = W^-1, so
    #   Psi(.., e_b, ..) = sum_k X[b, k] Psi(.., w_k, ..).
    rng = np.random.default_rng(0)
    W = []
    for _ in range(N):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        W.append(q)  # rows = basis spin coordinates w_0, w_1
    X = [np.linalg.inv(w) for w in W]
    ks = list(itertools.product([0, 1], repeat=N))
    S_gen = jnp.array([[W[q][k[q]] for q in range(N)] for k in ks])  # (2^N, N, 2)
    batch = jax.jit(jax.vmap(lambda x, s, tt: jnp.exp(log_psi(params, x, s, tt)), in_axes=(None, 0, None)))

    def in_up_down(values):
        """Generic-basis values (2^N, ...) -> up/down basis tensor (2,)*N + (...)."""
        v = values.reshape((2,) * N + values.shape[1:])
        for q in range(N):
            v = np.moveaxis(np.tensordot(X[q], v, axes=([1], [q])), 0, q)
        return v

    psi, grad = {}, {}
    for tb in isos:
        tt = jnp.array(tb)
        base = in_up_down(np.asarray(batch(r, S_gen, tt)))
        g = np.zeros((2,) * N + (N, 3), complex)
        for q in range(N):
            for c in range(3):
                plus = in_up_down(np.asarray(batch(r.at[q, c].add(h), S_gen, tt)))
                minus = in_up_down(np.asarray(batch(r.at[q, c].add(-h), S_gen, tt)))
                g[..., q, c] = (plus - minus) / (2 * h)
        for sb in spins:
            idx = tuple(0 if x == 1 else 1 for x in sb)
            psi[(sb, tb)] = base[idx]
            grad[(sb, tb)] = g[idx]

    P, D = np.asarray(PAULI), np.asarray(DOT)
    one2 = np.eye(2)
    S_a = [(np.kron(P[a], one2) + np.kron(one2, P[a])) / 2 for a in range(3)]
    g = np.arange(-6, 7)
    nn = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    B = {1: 0, -1: 1}
    xi = np.asarray(xi)
    total = 0.0
    for i, j in itertools.combinations(range(N), 2):
        rn = np.asarray(minimum_image(r[i] - r[j], L))[None] + L * nn
        r2 = (rn**2).sum(-1)
        R = {ch: sum(V * (np.exp(-k * r2)[:, None] * rn).sum(0) for V, k in RADIAL[ch]) for ch in RADIAL}
        iso_op = {"ls": np.eye(4), "lst": D}
        for sb in spins:
            c_b = np.prod([xi[q, B[sb[q]]] for q in range(N)])
            row_s = 2 * B[sb[i]] + B[sb[j]]
            row_t = 2 * B[t_cur[i]] + B[t_cur[j]]
            for col_s, col_t in itertools.product(range(4), range(4)):
                s2, t2 = list(sb), list(t_cur)
                s2[i], s2[j] = 1 - 2 * (col_s // 2), 1 - 2 * (col_s % 2)
                t2[i], t2[j] = 1 - 2 * (col_t // 2), 1 - 2 * (col_t % 2)
                key = (tuple(s2), tuple(t2))
                if key not in psi:
                    continue
                dg = grad[key][i] - grad[key][j]  # (grad_i - grad_j) Psi_target
                for ch in RADIAL:
                    iso = iso_op[ch][row_t, col_t]
                    if iso == 0:
                        continue
                    Lvec = -0.5j * np.cross(R[ch], dg)  # (L_a Psi)_target, images included
                    total += c_b * iso * sum(S_a[a][row_s, col_s] * Lvec[a] for a in range(3))
    return total / complex(jnp.exp(log_psi(params, r, jnp.asarray(xi), t)))


def test_spin_orbit_against_brute_force():
    # A correlated, complex wavefunction with random spinors: every S_z sector
    # and both isospin channels contribute.
    for counts, seed in [((1, 1, 1, 1), 1), ((1, 1, 1, 0), 2)]:
        system = System(*counts, 0.08)
        params, log_psi, r, xi, t = random_state(system, seed)
        e = make_ls_potential(system, RADIAL)(log_psi, params, r, xi, t)
        e_brute = brute_force_ls(system, params, log_psi, r, xi, t)
        assert np.isclose(e, e_brute, rtol=1e-6, atol=1e-6), (counts, e, e_brute)


# =============================================================================
# Test 2 - Consistency: tau.tau = 1 in neutron matter
# =============================================================================
def test_lst_equals_ls_for_neutrons():
    # For neutrons only, tau_i.tau_j = 1: an "lst" radial function must act
    # exactly like the same function in the "ls" channel.
    system = System(7, 1, 0, 0, 0.08)
    params, log_psi, r, xi, t = random_state(system, 3)
    a = make_ls_potential(system, {"lst": RADIAL["lst"]})(log_psi, params, r, xi, t)
    b = make_ls_potential(system, {"ls": RADIAL["lst"]})(log_psi, params, r, xi, t)
    assert np.isclose(a, b, rtol=1e-10)
