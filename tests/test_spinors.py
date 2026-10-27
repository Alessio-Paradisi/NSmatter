"""Spinor spin coordinates: wavefunction and pair-operator local energy.

Interface:
    src/nsmatter/spin.py: as_spinors(s), UP, DOWN, PAIR_BASIS
    make_slater / make_jastrow_slater accept s as labels (N,) or spinors (N, 2)
    make_v6_potential works with spinors, using <current|v|target>
Run with:  uv run pytest -q tests/test_spinors.py
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import minimum_image
from nsmatter.hamiltonian.exchange import make_v4_potential
from nsmatter.hamiltonian.pair_operators import DOT, PAULI, make_v6_potential
from nsmatter.spin import as_spinors
from nsmatter.system import System
from nsmatter.wavefunctions.jastrow import make_jastrow_slater

RADIAL = {
    "c": ((200.0, 1.487), (-30.0, 0.465)),
    "s": ((-15.0, 0.7),),
    "t": ((10.0, 0.9),),
    "st": ((-5.0, 1.2),),
    "tn": ((-20.0, 0.8),),
    "tnt": ((8.0, 1.0),),
}


def state(system, seed):
    """Jastrow-Slater state with random parameters, random positions and spinors."""
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = {"jastrow": 0.3 * jax.random.normal(k1, params["jastrow"].shape)}
    r = jax.random.uniform(k2, (system.n_particles, 3), maxval=system.box_length)
    z = jax.random.normal(k3, (system.n_particles, 2, 2))
    xi = z[..., 0] + 1j * z[..., 1]
    xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)  # random unit spinors
    return params, log_psi, r, xi, jnp.asarray(system.isospins)


# =============================================================================
# Test 1 - Psi is linear in every spinor, and labels are basis spinors
# =============================================================================
def test_multilinear_in_spinors():
    # Psi(.., xi_j, ..) = xi_j,up Psi(.., up, ..) + xi_j,down Psi(.., down, ..).
    # Every operator acting on spins relies on this property.
    system = System(7, 7, 1, 1, 0.16)
    params, log_psi, r, xi, t = state(system, 0)
    psi = lambda s: jnp.exp(log_psi(params, r, s, t))  # noqa: E731
    for j in (0, 9, 15):
        up = xi.at[j].set(jnp.array([1.0, 0.0]))
        down = xi.at[j].set(jnp.array([0.0, 1.0]))
        assert np.isclose(psi(xi), xi[j, 0] * psi(up) + xi[j, 1] * psi(down), rtol=1e-9)
    # Discrete labels give the same Psi as the corresponding basis spinors.
    s = jnp.asarray(system.spins)
    assert np.isclose(log_psi(params, r, s, t), log_psi(params, r, as_spinors(s), t))


# =============================================================================
# Test 2 - v6 local energy against an explicit matrix-vector product
# =============================================================================
def full_local_energy(system, params, log_psi, r, xi, t):
    """E_pot,L = (V Psi)(S) / Psi(S) the long way, for a few particles.

    Expand Psi(S) = sum_b c_b(S) Psi_b over all 2^N discrete spin states b
    (c_b = product of spinor components), build V as a matrix on those
    states pair by pair (images summed explicitly), and contract:
        (V Psi)(S) = sum_b c_b(S) sum_b' V[b, b'] Psi_b'.
    Isospin: discrete; tau.tau only permutes the labels, so the isospin states
    needed are the distinct permutations of the current labels.
    This is an ordinary matrix-vector product: <current|V|target> by construction.
    """
    N, L = system.n_particles, system.box_length
    t_cur = tuple(int(x) for x in t)
    spins = list(itertools.product([1, -1], repeat=N))
    isos = set(itertools.permutations(t_cur))
    psi = {
        (sb, tb): complex(jnp.exp(log_psi(params, r, jnp.array(sb), jnp.array(tb))))
        for sb in spins
        for tb in isos
    }
    g = np.arange(-6, 7)
    nn = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    P, D, one = np.asarray(PAULI), np.asarray(DOT), np.eye(4)
    B = {1: 0, -1: 1}

    def pair_matrix(d):
        rn = d[None] + L * nn
        r2 = (rn**2).sum(-1)
        sc = {ch: sum(V * np.exp(-k * r2).sum() for V, k in RADIAL[ch]) for ch in ("c", "s", "t", "st")}

        def tens(ch):
            m = np.zeros((4, 4), complex)
            for V, k in RADIAL[ch]:
                w = V * np.exp(-k * r2)
                sr = np.einsum("na,aij->nij", rn, P)
                kr = np.einsum("nij,nkl->nikjl", sr, sr).reshape(-1, 4, 4)
                m += np.einsum("n,nij->ij", w, 3 * kr) - (w * r2).sum() * D
            return m

        spin_A = sc["c"] * one + sc["s"] * D + tens("tn")
        spin_B = sc["t"] * one + sc["st"] * D + tens("tnt")
        return np.kron(spin_A, one) + np.kron(spin_B, D)  # index 4*spin_pair + iso_pair

    xi = np.asarray(xi)
    v_psi = 0.0
    for i, j in itertools.combinations(range(N), 2):
        M = pair_matrix(np.asarray(minimum_image(r[i] - r[j], L)))
        for sb in spins:
            c_b = np.prod([xi[k, B[sb[k]]] for k in range(N)])
            row = 4 * (2 * B[sb[i]] + B[sb[j]]) + 2 * B[t_cur[i]] + B[t_cur[j]]
            for col in range(16):
                if M[row, col] == 0:
                    continue
                a2, b2 = divmod(col, 4)
                s2, t2 = list(sb), list(t_cur)
                s2[i], s2[j] = 1 - 2 * (a2 // 2), 1 - 2 * (a2 % 2)
                t2[i], t2[j] = 1 - 2 * (b2 // 2), 1 - 2 * (b2 % 2)
                v_psi += c_b * M[row, col] * psi[(tuple(s2), tuple(t2))]
    return v_psi / complex(jnp.exp(log_psi(params, r, jnp.asarray(xi), t)))


def test_v6_against_matrix_vector_product():
    # Random spinors overlap all S_z sectors, so the S_z-changing (complex)
    # tensor matrix elements now matter: a transposed matrix fails here.
    # Two cases: neutrons and protons (isospin swaps), and 8 neutrons with
    # plane waves k != 0 (a complex Psi).
    for counts, seed in [((1, 1, 1, 1), 1), ((7, 1, 0, 0), 2)]:
        system = System(*counts, 0.08)
        params, log_psi, r, xi, t = state(system, seed)
        e = make_v6_potential(system, RADIAL)(log_psi, params, r, xi, t)
        assert np.isclose(e, full_local_energy(system, params, log_psi, r, xi, t), rtol=1e-8, atol=1e-8)


# =============================================================================
# Test 3 - Two independent routes agree for the v4 operators
# =============================================================================
def test_v4_routes_agree_with_spinors():
    # exchange.py (sigma.sigma = 2 P - 1, swapping spinors) and
    # pair_operators.py (explicit 4 x 4 matrices) must agree for any spinors.
    system = System(7, 7, 1, 0, 0.16)
    params, log_psi, r, xi, t = state(system, 3)
    v4 = {ch: RADIAL[ch] for ch in ("c", "s", "t", "st")}
    a = make_v6_potential(system, v4)(log_psi, params, r, xi, t)
    b = make_v4_potential(system, v4)(log_psi, params, r, xi, t)
    assert np.isclose(a, b, rtol=1e-10)
