"""NV3 three-nucleon force in neutron matter against a brute-force three-body calculation.

8 neutrons (7 up + 1 down shells), correlated wavefunction with random parameters.
Brute force: the full spin space (2^8 = 256 states); every term of
    V = sum_{i<j<k} sum_cyclic V(ijk)
built as an explicit 256 x 256 matrix, with {X_ij, X_jk} computed as a matrix
anticommutator (no pair reduction); then
    E_L = sum_targets <current|V|target> Psi(target) / Psi
over the 256 product states of a generic spin basis. Large box (density 1e-4):
only the minimum image of each pair is within range, so the geometry is unambiguous.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401
from nsmatter.hamiltonian.pair_operators import PAULI
from nsmatter.hamiltonian.three_body import CHANNELS, make_three_body_pnm
from nsmatter.potentials.nv3 import nv3_couplings, nv3_radial
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated

SYSTEM = System(7, 1, 0, 0, 1e-4)
N = SYSTEM.n_particles


def one_body(op, k):
    """op acting on particle k in the 2^N spin space."""
    mats = [np.eye(2)] * N
    mats[k] = np.asarray(op)
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


@pytest.fixture(scope="module")
def setup():
    params, log_psi = make_correlated(SYSTEM, n_terms=3)
    key = jax.random.PRNGKey(4)
    params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), params)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(-1.6, 1.6, (N, 3)))  # a compact cluster: all triples interact
    xi = rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2))
    xi = jnp.asarray(xi / np.linalg.norm(xi, axis=1, keepdims=True))
    t = jnp.asarray(SYSTEM.isospins)
    return params, log_psi, r, xi, t


def brute_force(log_psi, params, r, xi, t, radial, cpl):
    sig = [[one_body(PAULI[a], k) for a in range(3)] for k in range(N)]
    rv = np.asarray(r)
    f = {}

    def leg(p, q):
        d = rv[p] - rv[q]
        dist = np.linalg.norm(d)
        vals = {k: float(v) for k, v in radial(jnp.asarray(dist)).items()}
        rhat = d / dist
        A = (vals["Y"] - vals["T"]) * np.eye(3) + 3 * vals["T"] * np.outer(rhat, rhat)
        return A, vals["Z"] * rhat, vals["C"]

    for p in range(N):
        for q in range(N):
            if p != q:
                f[p, q] = leg(p, q)

    def X(p, q):
        A = f[p, q][0]
        return sum(A[a, b] * sig[p][a] @ sig[q][b] for a in range(3) for b in range(3))

    dim = 2**N
    V = {ch: np.zeros((dim, dim), dtype=complex) for ch in CHANNELS}
    for i, j, k in itertools.combinations(range(N), 3):
        for a, m, b in ((i, j, k), (j, k, i), (k, i, j)):  # cyclic: middle particle m
            Xam, Xmb = X(a, m), X(m, b)
            V["3b_c3"] += cpl["a_c3"] * (Xam @ Xmb + Xmb @ Xam)
            V["3b_Delta"] += cpl["a_Delta"] * (Xam @ Xmb + Xmb @ Xam)
            za, zb = f[a, m][1], f[m, b][1]
            sa = sum(za[c] * sig[a][c] for c in range(3))
            sb = sum(zb[c] * sig[b][c] for c in range(3))
            V["3b_c1"] += cpl["a_c1"] * sa @ sb
            V["3b_D"] += cpl["a_D"] * X(a, b) * (f[a, m][2] + f[m, b][2])
            V["3b_E"] += cpl["a_E"] * f[a, m][2] * f[m, b][2] * np.eye(dim)

    # generic orthonormal spin basis per particle (rows), targets = all 2^N products
    rng = np.random.default_rng(7)
    bases = [np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0].T for _ in range(N)]
    bra = np.asarray(xi[0])
    for k in range(1, N):
        bra = np.kron(bra, np.asarray(xi[k]))
    lp0 = complex(log_psi(params, r, xi, t))
    labels = list(itertools.product((0, 1), repeat=N))
    targets = jnp.asarray(np.array([[bases[k][lab[k]] for k in range(N)] for lab in labels]))
    ratios = np.exp(np.asarray(jax.vmap(lambda s: log_psi(params, r, s, t))(targets)) - lp0)
    kets = []
    for lab in labels:
        ket = np.conj(bases[0][lab[0]])
        for k in range(1, N):
            ket = np.kron(ket, np.conj(bases[k][lab[k]]))
        kets.append(ket)
    kets = np.array(kets)  # (dim targets, dim)
    return {ch: np.sum((bra @ V[ch] @ kets.T) * ratios) for ch in CHANNELS}


@pytest.mark.parametrize("model", ["Ib*", "Ia"])
def test_three_body_against_brute_force(setup, model):
    params, log_psi, r, xi, t = setup
    cpl, (R_S, R_L) = nv3_couplings(model)
    radial = nv3_radial(R_S, R_L, r_max=12.0)
    term = make_three_body_pnm(SYSTEM, radial, cpl, r_max=12.0)
    fast = {k: complex(v) for k, v in term.by_channel(log_psi, params, r, xi, t).items()}
    ref = brute_force(log_psi, params, r, xi, t, radial, cpl)
    for ch in CHANNELS:
        assert np.isclose(fast[ch], ref[ch], rtol=1e-9, atol=1e-12), (ch, fast[ch], ref[ch])
    assert abs(sum(fast.values())) > 1e-3  # the test is not trivially zero
