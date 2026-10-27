"""Operator-correlated wavefunction exp(J) [1 + sum U_ij] Phi.

Interface (src/nsmatter/wavefunctions/correlations.py):
    OPERATORS = ("s", "t", "st", "tn", "tnt")
    make_correlated(system, n_terms=3, cutoff=None) -> (params, log_psi)
        params = {"jastrow": (2, n_terms), "corr": (5, n_terms)}, zeros at start
Run with:  uv run pytest -q tests/test_correlations.py
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import minimum_image
from nsmatter.hamiltonian.pair_operators import DOT, PAULI
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated
from nsmatter.wavefunctions.jastrow import jastrow_basis, make_jastrow_slater
from nsmatter.wavefunctions.slater import make_slater


def random_state(system, seed, n_terms=3):
    params, log_psi = make_correlated(system, n_terms=n_terms)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    params = {
        "jastrow": 0.3 * jax.random.normal(k1, params["jastrow"].shape),
        "corr": 0.3 * jax.random.normal(k2, params["corr"].shape),
    }
    r = jax.random.uniform(k3, (system.n_particles, 3), maxval=system.box_length)
    z = jax.random.normal(k4, (system.n_particles, 2, 2))
    xi = z[..., 0] + 1j * z[..., 1]
    xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
    return params, log_psi, r, xi, jnp.asarray(system.isospins)


# =============================================================================
# Test 1 - Zero correlation parameters give the Jastrow-Slater state
# =============================================================================
def test_zero_correlations_reduce_to_jastrow_slater():
    system = System(7, 7, 1, 1, 0.16)
    params, log_psi, r, xi, t = random_state(system, 0)
    params = {**params, "corr": jnp.zeros_like(params["corr"])}
    _, jas = make_jastrow_slater(system, n_terms=3)
    d = log_psi(params, r, xi, t) - jas({"jastrow": params["jastrow"]}, r, xi, t)
    assert np.isclose(d.real, 0.0, atol=1e-10) and np.isclose(np.cos(d.imag), 1.0, atol=1e-10)


# =============================================================================
# Test 2 - The fast 2x2 update against a brute-force construction
# =============================================================================
def brute_force_psi(system, params, r, xi, t, n_terms=3):
    """Psi(R, S) built the long way, sharing no code with correlations.py
    except the radial basis:
      - Phi_b = Slater determinant at every discrete spin state b (full slogdet);
      - F = 1 + sum_ij U_ij as an explicit matrix on spin (x) isospin states,
        from Pauli matrices, pair by pair;
      - Psi(S) = exp(J) sum_b c_b(S) (F Phi)_b, with c_b = product of spinor components.
    """
    N, L = system.n_particles, system.box_length
    rc = 0.5 * L
    _, slater = make_slater(system)
    t_cur = tuple(int(x) for x in t)
    spins = list(itertools.product([1, -1], repeat=N))
    isos = set(itertools.permutations(t_cur))
    phi = {
        (sb, tb): complex(jnp.exp(slater({}, r, jnp.array(sb), jnp.array(tb)))) for sb in spins for tb in isos
    }
    P, D, one = np.asarray(PAULI), np.asarray(DOT), np.eye(4)
    B = {1: 0, -1: 1}
    a = np.asarray(params["corr"])
    xi = np.asarray(xi)
    total = 0.0
    for sb in spins:
        c_b = np.prod([xi[q, B[sb[q]]] for q in range(N)])
        f_phi = phi[(sb, t_cur)]  # the "1" of F
        for i, j in itertools.combinations(range(N), 2):
            d = np.asarray(minimum_image(r[i] - r[j], L))
            r2 = (d**2).sum()
            basis = np.asarray(jastrow_basis(jnp.asarray(r2), rc, n_terms))
            u = a @ basis  # u_s, u_t, u_st, w_tn, w_tnt
            sd = np.einsum("a,aij->ij", d, P)
            s12_u = (3 * np.kron(sd, sd) - r2 * D) / rc**2  # = S12 * x^2 (the tensor's extra x^2)
            spin_A = u[0] * D + u[3] * s12_u
            spin_B = u[1] * one + u[2] * D + u[4] * s12_u
            U = np.kron(spin_A, one) + np.kron(spin_B, D)  # index 4*spin + iso
            row = 4 * (2 * B[sb[i]] + B[sb[j]]) + 2 * B[t_cur[i]] + B[t_cur[j]]
            for col in range(16):
                if U[row, col] == 0:
                    continue
                a2, b2 = divmod(col, 4)
                s2, t2 = list(sb), list(t_cur)
                s2[i], s2[j] = 1 - 2 * (a2 // 2), 1 - 2 * (a2 % 2)
                t2[i], t2[j] = 1 - 2 * (b2 // 2), 1 - 2 * (b2 % 2)
                f_phi += U[row, col] * phi[(tuple(s2), tuple(t2))]
        total += c_b * f_phi
    # Central Jastrow, computed directly.
    J = 0.0
    for i, j in itertools.combinations(range(N), 2):
        d = np.asarray(minimum_image(r[i] - r[j], L))
        ch = 0 if t_cur[i] == t_cur[j] else 1
        J += float(np.asarray(jastrow_basis(jnp.asarray((d**2).sum()), rc, n_terms)) @ np.asarray(params["jastrow"][ch]))
    return np.exp(J) * total


def test_fast_update_against_brute_force():
    # Random parameters for every operator, random spinors: every part of the
    # construction (2x2 determinant ratios, tensor directions, isospin swaps,
    # orientation <current|U|target>) is compared with the explicit sum.
    for counts, seed in [((1, 1, 1, 1), 1), ((7, 1, 0, 0), 2), ((1, 1, 1, 0), 3)]:
        system = System(*counts, 0.08)
        params, log_psi, r, xi, t = random_state(system, seed)
        psi_fast = complex(jnp.exp(log_psi(params, r, xi, t)))
        psi_brute = brute_force_psi(system, params, r, xi, t)
        assert np.isclose(psi_fast, psi_brute, rtol=1e-8), (counts, psi_fast, psi_brute)


# =============================================================================
# Test 3 - Antisymmetry under exchange of complete particles
# =============================================================================
def test_antisymmetry_with_correlations():
    # sum_ij U_ij is symmetric under permutations, so Psi stays antisymmetric
    # under the exchange of (r, xi, t) of any two nucleons.
    system = System(7, 7, 7, 1, 0.08)
    params, log_psi, r, xi, t = random_state(system, 4)
    ref = log_psi(params, r, xi, t)
    for i, j in [(0, 7), (3, 15), (8, 21), (14, 20)]:
        idx, rev = jnp.array([i, j]), jnp.array([j, i])
        d = log_psi(params, r.at[idx].set(r[rev]), xi.at[idx].set(xi[rev]), t.at[idx].set(t[rev])) - ref
        assert np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), -1.0, atol=1e-9)


# =============================================================================
# Test 4 - The local energy stays finite where a basis-spinor target would vanish
# =============================================================================
def test_local_energy_finite_with_degenerate_orbitals():
    # One nucleon per species, all in k = 0: setting two neutrons to the same
    # basis spinor makes Phi exactly zero while Psi is not. The potential must
    # evaluate its targets in the adapted basis {xi, xi_perp}, so E_L stays finite.
    from nsmatter.hamiltonian.pair_operators import make_v6_potential
    from nsmatter.local_energy import make_local_energy

    system = System(1, 1, 1, 1, 0.04)
    params, log_psi, r, xi, t = random_state(system, 5)
    radial = {"c": ((200.0, 1.487), (-30.0, 0.465)), "tnt": ((10.0, 0.6),)}
    e = make_local_energy(log_psi, (make_v6_potential(system, radial),))(params, r, xi, t)
    assert np.isfinite(e.real) and np.isfinite(e.imag)
