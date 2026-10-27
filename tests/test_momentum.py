"""Momentum-dependent operators (L^2, (L.S)^2 families) and tabulated radial functions.

Interfaces:
    src/nsmatter/hamiltonian/momentum.py
        CHANNELS: ls, lst, l2, l2t, l2s, l2st, ls2, ls2t
        make_momentum_potential(system, radial) -> term(log_psi, params, r, s, t)
    src/nsmatter/hamiltonian/radial.py
        Gaussians(terms), Table(r_grid, values), tensor_table(r_grid, v_T),
        as_radial(f), image_vectors(L, functions)
Run with:  uv run pytest -q tests/test_momentum.py
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import minimum_image
from nsmatter.hamiltonian.momentum import make_momentum_potential
from nsmatter.hamiltonian.pair_operators import DOT, PAULI, make_v6_potential
from nsmatter.hamiltonian.radial import Gaussians, Table, tensor_table
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated

V0, KAPPA = -20.0, 0.5  # radial function used in the tests: v(r) = V0 exp(-KAPPA r^2)


def random_spinors(n, seed):
    z = jax.random.normal(jax.random.PRNGKey(seed), (n, 2, 2))
    xi = z[..., 0] + 1j * z[..., 1]
    return xi / jnp.linalg.norm(xi, axis=1, keepdims=True)


# =============================================================================
# Test 1 - Exact eigenvalues of L^2, L.S, (L.S)^2
# =============================================================================
@pytest.mark.parametrize("l", [1, 2])
def test_eigenvalues_on_angular_momentum_states(l):
    # Two neutrons in the relative state (x + i y)^l exp(-r^2) |up up>:
    # orbital l, m = l, spin S = 1, M = 1, so J = l + 1 and
    #   L^2 = l(l+1),  sigma.sigma = 1,  L.S = [J(J+1) - l(l+1) - 2]/2 = l,  (L.S)^2 = l^2.
    # tau.tau = 1 for two neutrons. A short-range v in a large box makes the
    # periodic images negligible, so E_L = v(r) * eigenvalue for any spinors.
    system = System(1, 1, 0, 0, 0.001)
    L = system.box_length

    def log_psi(params, r, s, t):
        d = minimum_image(r[0] - r[1], L)
        return l * jnp.log(d[0] + 1j * d[1]) - jnp.sum(d**2) + jnp.log(s[0, 0]) + jnp.log(s[1, 0])

    r = jnp.array([[0.3, -0.2, 0.5], [1.1, 0.4, -0.3]])
    xi = random_spinors(2, l)
    t = jnp.array([-1, -1])
    v = V0 * np.exp(-KAPPA * float(jnp.sum((r[0] - r[1]) ** 2)))
    expected = {"ls": l, "lst": l, "l2": l * (l + 1), "l2t": l * (l + 1), "l2s": l * (l + 1),
                "l2st": l * (l + 1), "ls2": l**2, "ls2t": l**2}
    for ch, eig in expected.items():
        e = make_momentum_potential(system, {ch: ((V0, KAPPA),)})(log_psi, {}, r, xi, t)
        assert np.isclose(e, v * eig, rtol=1e-9, atol=1e-9), (ch, complex(e), v * eig)


# =============================================================================
# Test 2 - L^2 and (L.S)^2 families against matrices and finite differences
# =============================================================================
def brute_force(system, params, log_psi, r, xi, t, radial, h=2.5e-4):
    """(V Psi)(S)/Psi(S) with explicit matrices, finite differences, explicit images.

    For each pair, Psi at every discrete spin-isospin state b is obtained by
    linearity from a generic spinor basis (where nothing vanishes), with
    derivatives in the relative displacement u: f(u) = Psi(r_i + u, r_j - u),
    d = (1/2) d/du. L_a and L_a L_b are applied image by image from their
    definition (eps tensors), not from the moment formulas of the module.
    """
    N, L = system.n_particles, system.box_length
    t_cur = tuple(int(x) for x in t)
    spins = list(itertools.product([1, -1], repeat=N))
    isos = sorted(set(itertools.permutations(t_cur)))
    rng = np.random.default_rng(0)
    W = [np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0] for _ in range(N)]
    X = [np.linalg.inv(w) for w in W]
    ks = list(itertools.product([0, 1], repeat=N))
    S_gen = jnp.array([[W[q][k[q]] for q in range(N)] for k in ks])
    batch = jax.jit(jax.vmap(lambda x, s, tt: jnp.exp(log_psi(params, x, s, tt)), in_axes=(None, 0, None)))

    def up_down(values):
        v = values.reshape((2,) * N)
        for q in range(N):
            v = np.moveaxis(np.tensordot(X[q], v, axes=([1], [q])), 0, q)
        return v

    P, D = np.asarray(PAULI), np.asarray(DOT)
    one2, one4 = np.eye(2), np.eye(4)
    S = [(np.kron(P[a], one2) + np.kron(one2, P[a])) / 2 for a in range(3)]
    eps = np.zeros((3, 3, 3))
    for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[a, b, c], eps[a, c, b] = 1, -1
    g = np.arange(-6, 7)
    nn = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    B = {1: 0, -1: 1}
    E = np.eye(3)
    xi = np.asarray(xi)
    total = 0.0
    for i, j in itertools.combinations(range(N), 2):
        # Psi, grad and Hessian (in r = r_i - r_j) at every discrete state, by finite differences.
        def psi_at(u, tt):
            return up_down(np.asarray(batch(r.at[i].add(u).at[j].add(-u), S_gen, jnp.array(tt))))

        data = {}
        for tb in isos:
            p0 = psi_at(jnp.zeros(3), tb)
            grad = np.stack([(psi_at(h * E[a], tb) - psi_at(-h * E[a], tb)) / (2 * h) for a in range(3)], -1) / 2
            hess = np.zeros((2,) * N + (3, 3), complex)
            for a in range(3):
                for c in range(3):
                    hess[..., a, c] = (
                        psi_at(h * (E[a] + E[c]), tb) - psi_at(h * (E[a] - E[c]), tb)
                        - psi_at(h * (-E[a] + E[c]), tb) + psi_at(-h * (E[a] + E[c]), tb)
                    ) / (4 * h * h) / 4
            data[tb] = (p0, grad, hess)

        rn = np.asarray(minimum_image(r[i] - r[j], L))[None] + L * nn
        v = {ch: sum(V * np.exp(-k * (rn**2).sum(-1)) for V, k in terms) for ch, terms in radial.items()}
        for sb in spins:
            c_b = np.prod([xi[q, B[sb[q]]] for q in range(N)])
            row_s = 2 * B[sb[i]] + B[sb[j]]
            row_t = 2 * B[t_cur[i]] + B[t_cur[j]]
            for col_s, col_t in itertools.product(range(4), range(4)):
                s2, t2 = list(sb), list(t_cur)
                s2[i], s2[j] = 1 - 2 * (col_s // 2), 1 - 2 * (col_s % 2)
                t2[i], t2[j] = 1 - 2 * (col_t // 2), 1 - 2 * (col_t % 2)
                if tuple(t2) not in data:
                    continue
                idx = tuple(0 if x == 1 else 1 for x in s2)
                p0, grad, hess = (arr[idx] for arr in data[tuple(t2)])
                for ch in radial:
                    iso = (D if ch.endswith("t") else one4)[row_t, col_t]
                    if iso == 0:
                        continue
                    for n in range(len(rn)):  # image by image, from the definitions
                        if abs(v[ch][n]) < 1e-14:
                            continue
                        rv = rn[n]
                        La = -1j * np.einsum("acd,c,d->a", eps, rv, grad)  # L_a Psi
                        LaLb = -(np.einsum("acd,bdf,c,f->ab", eps, eps, rv, grad)
                                 + np.einsum("acd,bef,c,e,df->ab", eps, eps, rv, rv, hess))
                        if ch.startswith("ls2"):
                            val = sum((S[a] @ S[b])[row_s, col_s] * LaLb[a, b] for a in range(3) for b in range(3))
                        elif ch.startswith("l2"):
                            spin = D if ch in ("l2s", "l2st") else one4
                            val = spin[row_s, col_s] * np.trace(LaLb)
                        else:
                            val = sum(S[a][row_s, col_s] * La[a] for a in range(3))
                        total += c_b * iso * v[ch][n] * val
    return total / complex(jnp.exp(log_psi(params, r, jnp.asarray(xi), t)))


def test_against_brute_force():
    # Correlated complex state, random spinors, neutrons and protons:
    # all L^2 and (L.S)^2 channels at once.
    system = System(1, 1, 1, 1, 0.08)
    params, log_psi = make_correlated(system, n_terms=3)
    k = jax.random.split(jax.random.PRNGKey(7), 3)
    params = {"jastrow": 0.3 * jax.random.normal(k[0], params["jastrow"].shape),
              "corr": 0.3 * jax.random.normal(k[1], params["corr"].shape)}
    r = jax.random.uniform(k[2], (4, 3), maxval=system.box_length)
    xi = random_spinors(4, 8)
    t = jnp.asarray(system.isospins)
    radial = {"l2": ((-8.0, 0.9),), "l2t": ((3.0, 1.1),), "l2s": ((2.0, 0.8),),
              "l2st": ((-1.5, 1.2),), "ls2": ((5.0, 1.0),), "ls2t": ((-2.0, 0.7),)}
    e = make_momentum_potential(system, radial)(log_psi, params, r, xi, t)
    # Finite differences carry an O(h^2) error: with h = 2.5e-4 it is about
    # 5e-6 relative (it shrinks by 4 each time h is halved, onto the module's value).
    e_brute = brute_force(system, params, log_psi, r, xi, t, radial)
    assert np.isclose(e, e_brute, rtol=2e-5, atol=1e-6), (complex(e), e_brute)


# =============================================================================
# Test 3 - Tabulated radial functions reproduce analytic ones
# =============================================================================
def test_tables_reproduce_gaussians():
    # A spline on a fine grid reproduces the Gaussian, and a v6 potential
    # built from tables (tensor via tensor_table) equals the Gaussian one.
    grid = np.linspace(0.0, 12.0, 2401)  # dr = 0.005 fm
    g = Gaussians(((200.0, 1.487), (-30.0, 0.465)))
    tab = Table(grid, np.asarray(g(jnp.asarray(grid**2))))
    r2 = jnp.linspace(0.0, 100.0, 777)
    assert jnp.allclose(tab(r2), g(r2), atol=1e-7)

    system = System(7, 7, 1, 1, 0.16)
    params, log_psi = make_correlated(system, n_terms=3)
    params = {"jastrow": 0.2 * jnp.ones_like(params["jastrow"]), "corr": 0.1 * jnp.ones_like(params["corr"])}
    r = jax.random.uniform(jax.random.PRNGKey(9), (16, 3), maxval=system.box_length)
    xi = random_spinors(16, 10)
    t = jnp.asarray(system.isospins)
    gauss = {"c": ((200.0, 1.487), (-30.0, 0.465)), "s": ((-15.0, 0.7),), "tnt": ((10.0, 0.6),)}
    tables = {
        "c": tab,
        "s": Table(grid, -15.0 * np.exp(-0.7 * grid**2)),
        "tnt": tensor_table(grid, 10.0 * grid**2 * np.exp(-0.6 * grid**2)),  # v_T itself
    }
    a = make_v6_potential(system, gauss)(log_psi, params, r, xi, t)
    b = make_v6_potential(system, tables)(log_psi, params, r, xi, t)
    assert np.isclose(a, b, rtol=1e-6)
