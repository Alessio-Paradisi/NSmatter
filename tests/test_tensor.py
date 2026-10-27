"""Tests for general pair operators and the tensor force (v6).

Interface defined here (module src/nsmatter/hamiltonian/pair_operators.py):
    PAULI, DOT                      Pauli matrices; sigma.sigma as a 4 x 4 matrix
    pair_index(a, b)                basis index of |a b>, a, b in {+1, -1}
    tensor_operator(rvec)           S12 = 3 (sigma_1.r^)(sigma_2.r^) - sigma_1.sigma_2, 4 x 4
    make_v6_potential(system, radial, tol=1e-10) -> term(log_psi, params, r, s, t)
        channels "c", "s", "t", "st": v(r) = sum V exp(-kappa r^2)
        channels "tn", "tnt":         v(r) = sum V r^2 exp(-kappa r^2)
        v_ij = v_c + v_s s.s + v_tn S12 + (v_t + v_st s.s + v_tnt S12) t.t
Run with:  uv run pytest -q tests/test_tensor.py
"""

# =============================================================================
# Block 0 - Imports and helpers
# =============================================================================
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import minimum_image, occupied_kvectors
from nsmatter.hamiltonian.central import TOY
from nsmatter.hamiltonian.exchange import make_v4_potential
from nsmatter.hamiltonian.pair_operators import DOT, PAULI, make_v6_potential, pair_index, tensor_operator
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking_error
from nsmatter.system import System
from nsmatter.wavefunctions.jastrow import make_jastrow_slater
from nsmatter.wavefunctions.slater import make_slater

# Test interaction: all six channels non-zero.
V4_PART = {"c": TOY, "s": ((-15.0, 0.7),), "t": ((10.0, 0.9),), "st": ((-5.0, 1.2),)}
RADIAL = {**V4_PART, "tn": ((-20.0, 0.8),), "tnt": ((8.0, 1.0),)}


def config(system, seed=0):
    key = jax.random.PRNGKey(seed)
    r = jax.random.uniform(key, (system.n_particles, 3), maxval=system.box_length)
    return r, jnp.asarray(system.spins), jnp.asarray(system.isospins)


def jastrow_state(system, seed=0):
    """Jastrow-Slater state with random non-zero parameters."""
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    a = 0.3 * jax.random.normal(jax.random.PRNGKey(200 + seed), params["jastrow"].shape)
    return {"jastrow": a}, log_psi


# =============================================================================
# Block 1 - Algebra of the tensor operator
# =============================================================================
DIRECTIONS = [jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]), jnp.array([0.3, -1.2, 0.7])]


def test_pair_index_basis_order():
    # |up up> = 0, |up down> = 1, |down up> = 2, |down down> = 3.
    assert [int(pair_index(a, b)) for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]] == [0, 1, 2, 3]


def test_dot_matrix():
    # sigma.sigma = 2 P - 1: eigenvalue +1 on the triplet, -3 on the singlet.
    expected = jnp.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])
    assert jnp.allclose(DOT, expected)


def test_tensor_along_z():
    # For r^ = z the matrix is known by hand: diagonal (2, -2, -2, 2) from
    # 3 sigma_z sigma_z - sigma.sigma, plus -2 between |up down> and |down up>.
    expected = jnp.array([[2, 0, 0, 0], [0, -2, -2, 0], [0, -2, -2, 0], [0, 0, 0, 2]])
    assert jnp.allclose(tensor_operator(jnp.array([0.0, 0.0, 2.5])), expected)


@pytest.mark.parametrize("rvec", DIRECTIONS)
def test_tensor_properties(rvec):
    S = tensor_operator(rvec)
    # Hermitian and traceless (its average over spin states is zero).
    assert jnp.allclose(S, S.conj().T)
    assert np.isclose(jnp.trace(S), 0.0)
    # It annihilates the spin singlet (|up down> - |down up>)/sqrt 2: the
    # tensor force acts only in S = 1, as in the deuteron.
    singlet = jnp.array([0, 1, -1, 0]) / jnp.sqrt(2)
    assert jnp.allclose(S @ singlet, 0.0)
    # Textbook identity: S12^2 = 6 + 2 sigma.sigma - 2 S12.
    assert jnp.allclose(S @ S, 6 * jnp.eye(4) + 2 * DOT - 2 * S)


def test_tensor_depends_only_on_direction():
    # S12 depends on r^ only: rescaling or reversing r changes nothing.
    r = DIRECTIONS[2]
    assert jnp.allclose(tensor_operator(r), tensor_operator(3.7 * r))
    assert jnp.allclose(tensor_operator(r), tensor_operator(-r))


def test_tensor_flips_both_spins():
    # Unlike sigma.sigma, S12 couples |up up> to |down down> (total S_z
    # changes by 2) unless r^ is along z.
    S = tensor_operator(DIRECTIONS[1])
    assert abs(S[3, 0]) > 1.0
    assert abs(tensor_operator(DIRECTIONS[0])[3, 0]) < 1e-12


# =============================================================================
# Block 2 - The local energy against a brute-force calculation
# =============================================================================
PAULI_NP = np.asarray(PAULI)
DOT_NP = np.asarray(DOT)
LAB = {0: +1, 1: -1}


def pair_operator_brute(d, L):
    """The full 16 x 16 operator v_ij on |s_i s_j> (x) |t_i t_j>, summed over a
    large explicit cube of images. Tensor parts built from r_n directly."""
    g = np.arange(-6, 7)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    rn = d[None, :] + L * n
    r2 = (rn**2).sum(-1)
    one = np.eye(4)
    op = np.zeros((16, 16), dtype=complex)

    def scalar(ch):
        return sum(V * np.exp(-k * r2).sum() for V, k in RADIAL[ch])

    def tensor(ch):
        # sum_n V r_n^2 exp(-k r_n^2) S12(r^_n) = sum_n V exp(-k r_n^2) (3 (s.r)(s.r) - r^2 s.s)
        m = np.zeros((4, 4), dtype=complex)
        for V, k in RADIAL[ch]:
            w = V * np.exp(-k * r2)
            sr = np.einsum("na,aij->nij", rn, PAULI_NP)  # sigma . r_n, shape (n, 2, 2)
            kron = np.einsum("nij,nkl->nikjl", sr, sr).reshape(-1, 4, 4)
            m += np.einsum("n,nij->ij", w, 3 * kron) - (w * r2).sum() * DOT_NP
        return m

    spin_A = scalar("c") * one + scalar("s") * DOT_NP + tensor("tn")
    spin_B = scalar("t") * one + scalar("st") * DOT_NP + tensor("tnt")
    op = np.kron(spin_A, one) + np.kron(spin_B, DOT_NP)  # spin (x) isospin, index 4*spin + iso
    return op


@pytest.mark.parametrize("counts", [(7, 1, 1, 0), (1, 1, 1, 1)])
def test_local_energy_against_brute_force(counts):
    # All 16 spin-isospin target states of every pair, with explicit matrices:
    #   E_pot,L = sum_pairs sum_targets <current|v_ij|target> Psi(target)/Psi.
    # (With discrete labels of fixed S_z this cannot tell <current|v|target>
    # from its transpose; test_spinors.py checks that with random spinors.)
    system = System(*counts, 0.08)
    L = system.box_length
    params, log_psi = jastrow_state(system, 1)
    r, s, t = config(system, 2)
    psi0 = complex(jnp.exp(log_psi(params, r, s, t)))
    e_brute = 0.0
    for i, j in itertools.combinations(range(system.n_particles), 2):
        op = pair_operator_brute(np.asarray(minimum_image(r[i] - r[j], L)), L)
        col = 4 * int(pair_index(s[i], s[j])) + int(pair_index(t[i], t[j]))
        for row in range(16):
            amp = op[col, row]  # <current|v|target>
            if abs(amp) < 1e-14:
                continue
            a2, b2 = divmod(row, 4)
            s2 = s.at[i].set(LAB[a2 // 2]).at[j].set(LAB[a2 % 2])
            t2 = t.at[i].set(LAB[b2 // 2]).at[j].set(LAB[b2 % 2])
            e_brute += amp * complex(jnp.exp(log_psi(params, r, s2, t2))) / psi0
    e = make_v6_potential(system, RADIAL)(log_psi, params, r, s, t)
    assert np.isclose(e, e_brute, rtol=1e-8, atol=1e-8)


def test_v4_channels_match_exchange_module():
    # Without tensor channels, the general code must agree with exchange.py,
    # which uses a completely different route (sigma.sigma = 2P - 1).
    system = System(7, 7, 1, 0, 0.16)
    params, log_psi = jastrow_state(system, 3)
    r, s, t = config(system, 4)
    a = make_v6_potential(system, V4_PART)(log_psi, params, r, s, t)
    b = make_v4_potential(system, V4_PART)(log_psi, params, r, s, t)
    assert np.isclose(a, b, rtol=1e-10)


def test_tensor_local_energy_is_nonzero():
    # The tensor part of E_L does not vanish configuration by configuration
    # (it averages to zero in a Slater determinant, see block 3): this makes
    # sure the zero in block 3 is not just a term that is never computed.
    system = System(7, 7, 1, 1, 0.16)
    params, log_psi = jastrow_state(system, 5)
    r, s, t = config(system, 6)
    e = make_v6_potential(system, {"tn": RADIAL["tn"], "tnt": RADIAL["tnt"]})(log_psi, params, r, s, t)
    assert abs(e) > 1e-3


# =============================================================================
# Block 3 - Exact expectation value in the Slater determinant
# =============================================================================
IDX = {+1: 0, -1: 1}


def hartree_fock_v4(system, radial):
    """Exact <V> of the v4 channels for plane-wave spin-isospin orbitals (MeV, total)."""
    L = system.box_length
    k, sl, tl = [], [], []
    for count, sz, tz in zip(system.counts, (1, -1, 1, -1), (-1, -1, 1, 1)):
        if count:
            k.append(occupied_kvectors(count, L))
            sl += [sz] * count
            tl += [tz] * count
    k = np.concatenate(k)
    one = np.eye(4)
    ops = {"c": (one, one), "s": (DOT_NP, one), "t": (one, DOT_NP), "st": (DOT_NP, DOT_NP)}
    total = 0.0
    for a in range(len(k)):
        for b in range(len(k)):
            sa, sb, ta, tb = IDX[sl[a]], IDX[sl[b]], IDX[tl[a]], IDX[tl[b]]
            q2 = ((k[a] - k[b]) ** 2).sum()
            for op in ("c", "s", "t", "st"):
                ms, mt = ops[op]
                vq0 = sum(V * (np.pi / kk) ** 1.5 for V, kk in radial[op])
                vq = sum(V * (np.pi / kk) ** 1.5 * np.exp(-q2 / (4 * kk)) for V, kk in radial[op])
                direct = (ms[2 * sa + sb, 2 * sa + sb] * mt[2 * ta + tb, 2 * ta + tb]).real
                exchange = (ms[2 * sa + sb, 2 * sb + sa] * mt[2 * ta + tb, 2 * tb + ta]).real
                total += vq0 * direct - vq * exchange
    return total / (2 * L**3)


def test_vmc_tensor_averages_to_zero_in_slater():
    # In a closed-shell plane-wave Slater determinant the tensor force gives
    # exactly zero: the direct term vanishes (S12 averages to zero over
    # directions) and so does the exchange term (a closed shell is cubic
    # symmetric, and S12 is an l = 2 operator with no cubic-invariant part).
    # So the VMC average of the full v6 must equal the v4 Hartree-Fock value.
    # Physically: the tensor force binds only through tensor CORRELATIONS
    # in the wavefunction, which the Jastrow factor does not have yet.
    system = System(7, 1, 1, 1, 0.16)
    N, L = system.n_particles, system.box_length
    params, log_psi = make_slater(system)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    term = make_v6_potential(system, RADIAL)
    v_batch = jax.jit(jax.vmap(lambda r: term(log_psi, params, r, s, t).real))
    log_prob = make_log_prob(log_psi, s, t)
    key = jax.random.PRNGKey(12)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, 128, N, L)
    r, step, key = thermalize(log_prob, params, key, r, 0.1 * L / np.sqrt(N), L)
    series = []
    for _ in range(150):
        key, sub = jax.random.split(key)
        r, _ = metropolis(log_prob, params, sub, r, step, L, 5)
        series.append(float(v_batch(r).mean()) / N)
    mean, err = blocking_error(series)
    exact = hartree_fock_v4(system, RADIAL) / N
    assert abs(mean - exact) < 5 * err
    assert err < 0.3
