"""NV2 inside the VMC: tables, isospin operators (T12, tau_z), and the potential term.

References:
- the tables against nv2_radial at off-grid points;
- isospin matrix elements against explicit 4 x 4 isospin matrices;
- the static NV2 term for an np pair against a brute-force 16 x 16 spin-isospin
  matrix acting on Psi evaluated in a generic spin basis;
- for momentum operators, the identity L.S T12 = -3 L.S - L.S tau.tau (np pair).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401
from nsmatter.hamiltonian.momentum import make_momentum_potential
from nsmatter.hamiltonian.pair_operators import DOT, PAULI, isospin_factors, pair_index, tensor_operator
from nsmatter.potentials.nv2 import nv2_radial
from nsmatter.potentials.nv2_vmc import STATIC, make_nv2_potential, nv2_tables
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated

TZ = jnp.diag(jnp.array([1.0, -1.0]))  # tau_z on (proton, neutron)
I2 = jnp.eye(2)
ISO = {
    "1": jnp.eye(4),
    "tt": DOT.real,
    "T": 3 * jnp.kron(TZ, TZ) - DOT.real,
    "tz": jnp.kron(TZ, I2) + jnp.kron(I2, TZ),
}


@pytest.fixture(scope="module")
def tables():
    return nv2_tables("Ib")


def test_tables_match_radial(tables):
    r = np.array([0.0021, 0.137, 0.613, 1.0071, 1.77, 2.913, 4.41])
    v = nv2_radial(r, "b")
    for ch in STATIC:
        got = np.asarray(tables[ch](jnp.asarray(r**2)))
        if ch in ("tn", "tnt", "tnT"):
            got = got * r**2
        assert np.allclose(got, v[ch], rtol=1e-5, atol=1e-3), ch  # atol: the 1e-3 MeV cut


def test_isospin_factors():
    labels = jnp.array([1, -1])
    for a in labels:
        for b in labels:
            t = jnp.array([a, b])
            f = isospin_factors(t, jnp.array([0]), jnp.array([1]))
            cur, sw = pair_index(a, b), pair_index(b, a)
            for k, M in ISO.items():
                assert np.isclose(f[k][0][0], M[cur, cur]), (k, a, b)
                expected_swap = M[cur, sw] if a != b else 0.0
                assert np.isclose(f[k][1][0], expected_swap), (k, a, b)


def _random_basis(key):
    """A random orthonormal spinor basis (rows), used as spin coordinates."""
    z = jax.random.normal(key, (2, 2)) + 1j * jax.random.normal(jax.random.split(key)[0], (2, 2))
    q, _ = jnp.linalg.qr(z)
    return q.T


def test_static_nv2_np_pair_bruteforce(tables):
    system = System(1, 0, 1, 0, 1e-4)  # one n up, one p up; L ~ 27 fm > 2 r_max
    params, log_psi = make_correlated(system, n_terms=3)
    key = jax.random.PRNGKey(3)
    params = jax.tree_util.tree_map(lambda p: p + 0.2 * jax.random.normal(key, p.shape), params)
    r = jnp.array([[0.3, -0.2, 0.1], [0.9, 0.5, -0.6]])
    k1, k2, k3 = jax.random.split(key, 3)
    xi = jax.random.normal(k1, (2, 2)) + 1j * jax.random.normal(k2, (2, 2))
    xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
    t = jnp.asarray(system.isospins)

    term = make_nv2_potential(system, "Ib", tables=tables, momentum_terms=False)
    got = term(log_psi, params, r, xi, t)

    # Brute force: V = sum_ch v_ch(r) O_spin (x) O_iso on |s_i s_j> (x) |t_i t_j>.
    from nsmatter.hamiltonian.pair_operators import CHANNELS

    d = r[0] - r[1]
    dist = jnp.linalg.norm(d)
    spin_ops = {"1": jnp.eye(4, dtype=complex), "ss": DOT, "S12": tensor_operator(d)}
    V = jnp.zeros((16, 16), dtype=complex)
    for ch, (s_op, i_op) in CHANNELS.items():
        val = tables[ch](dist**2) * (dist**2 if s_op == "S12" else 1.0)
        V = V + val * jnp.kron(spin_ops[s_op], ISO[i_op])
    bases = [_random_basis(k) for k in jax.random.split(k3, 2)]
    bra = jnp.kron(jnp.kron(xi[0], xi[1]), jnp.eye(4)[pair_index(t[0], t[1])])
    lp0 = log_psi(params, r, xi, t)
    total = 0.0
    for a in range(2):
        for b in range(2):
            s_target = jnp.stack([bases[0][a], bases[1][b]])
            spin_ket = jnp.kron(jnp.conj(bases[0][a]), jnp.conj(bases[1][b]))
            for t_target in (t, t[::-1]):  # all operators conserve total charge
                ket = jnp.kron(spin_ket, jnp.eye(4)[pair_index(t_target[0], t_target[1])])
                total = total + (bra @ V @ ket) * jnp.exp(log_psi(params, r, s_target, t_target) - lp0)
    assert np.isclose(got, total, rtol=1e-9, atol=1e-9)


def test_momentum_lsT_identity():
    """np pair: T12 = 3 tau_z tau_z - tau.tau = -3 - tau.tau, so lsT = -3 ls - lst."""
    system = System(1, 0, 1, 0, 0.01)
    params, log_psi = make_correlated(system, n_terms=3)
    key = jax.random.PRNGKey(5)
    params = jax.tree_util.tree_map(lambda p: p + 0.2 * jax.random.normal(key, p.shape), params)
    r = jnp.array([[0.3, -0.2, 0.1], [0.9, 0.5, -0.6]])
    xi = jnp.array([[0.6, 0.8j], [0.8, -0.6]], dtype=complex)
    t = jnp.asarray(system.isospins)
    f = ((5.0, 1.3),)
    e = lambda rad: make_momentum_potential(system, rad)(log_psi, params, r, xi, t)  # noqa: E731
    lhs = e({"lsT": f})
    rhs = e({"ls": ((-15.0, 1.3),)}) - e({"lst": f})
    assert np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10)
