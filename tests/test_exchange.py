"""Tests for spin-isospin operators: label-aware Slater and the v4 potential.

Interfaces defined here:
    src/nsmatter/wavefunctions/slater.py  (REWRITTEN)
        make_slater(system) -> (params, log_psi), now a single N x N determinant
        of spin-isospin orbitals, so that Psi depends on the labels s, t.
    src/nsmatter/hamiltonian/exchange.py  (NEW)
        OPERATORS = ("c", "s", "t", "st")
        swap(labels, i, j) -> labels with entries i and j exchanged
        make_v4_potential(system, radial, tol=1e-10) -> term(log_psi, params, r, s, t)
            v_ij = v_c + v_s sigma.sigma + v_t tau.tau + v_st (sigma.sigma)(tau.tau)
            radial = {"c": terms, "s": terms, ...}, each a tuple of Gaussians (V, kappa)
            as in central.py; missing keys mean zero.
Run with:  uv run pytest -q tests/test_exchange.py
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
from nsmatter.hamiltonian.central import TOY, make_central_potential
from nsmatter.hamiltonian.exchange import make_v4_potential, swap
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking_error
from nsmatter.system import System, pure_neutron_matter
from nsmatter.wavefunctions.jastrow import make_jastrow_slater
from nsmatter.wavefunctions.slater import make_slater

# A test v4 interaction: arbitrary Gaussians in every operator channel.
RADIAL = {
    "c": TOY,
    "s": ((-15.0, 0.7),),
    "t": ((10.0, 0.9),),
    "st": ((-5.0, 1.2),),
}


def config(system, seed=0):
    key = jax.random.PRNGKey(seed)
    r = jax.random.uniform(key, (system.n_particles, 3), maxval=system.box_length)
    return r, jnp.asarray(system.spins), jnp.asarray(system.isospins)


def jastrow_state(system, seed=0):
    """A Jastrow-Slater state with random (non-zero) parameters."""
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    a = 0.3 * jax.random.normal(jax.random.PRNGKey(100 + seed), params["jastrow"].shape)
    return {"jastrow": a}, log_psi


def swap_triples(r, s, t, i, j):
    """Exchange particles i and j completely: position, spin and isospin."""
    idx = jnp.array([i, j])
    rev = jnp.array([j, i])
    return r.at[idx].set(r[rev]), s.at[idx].set(s[rev]), t.at[idx].set(t[rev])


# =============================================================================
# Block 1 - The label-aware Slater determinant
# =============================================================================
@pytest.mark.parametrize("counts", [(7, 7, 0, 0), (7, 1, 7, 1), (19, 7, 7, 1)])
def test_equals_product_of_species_determinants(counts):
    # With the labels in species order, the N x N matrix is block diagonal
    # and its determinant is the product of the per-species determinants
    # (the old implementation, recomputed here by hand).
    system = System(*counts, 0.08)
    params, log_psi = make_slater(system)
    r, s, t = config(system)
    expected = 0.0
    for sl, count in zip(system.species_slices, system.counts):
        if count:
            k = occupied_kvectors(count, system.box_length)
            sign, logabs = np.linalg.slogdet(np.exp(1j * (k @ np.asarray(r[sl]).T)))
            expected += logabs + np.log(sign)
    d = log_psi(params, r, s, t) - expected
    assert np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), 1.0, atol=1e-9)


def test_full_antisymmetry_across_species():
    # Psi is antisymmetric under the exchange of ANY two nucleons, also of
    # different species, provided position, spin and isospin move together.
    system = System(7, 7, 7, 1, 0.08)
    params, log_psi = jastrow_state(system)
    r, s, t = config(system, 1)
    ref = log_psi(params, r, s, t)
    for i, j in [(0, 7), (3, 15), (8, 21), (14, 20)]:  # n_up-n_down, n_up-p_up, n_down-p_down, ...
        d = log_psi(params, *swap_triples(r, s, t, i, j)) - ref
        assert np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), -1.0, atol=1e-9)


def test_wrong_composition_gives_zero():
    # Flipping one spin changes (N_up, N_down): no orbital matches the new
    # label set, the determinant has a zero block, and Psi = 0.
    system = System(7, 7, 0, 0, 0.16)
    params, log_psi = make_slater(system)
    r, s, t = config(system)
    value = log_psi(params, r, s.at[0].set(-1), t)
    assert value.real == -np.inf  # ln 0


def test_spin_isospin_exchange_equals_minus_space_exchange():
    # For an antisymmetric state, P_sigma P_tau P_r = -1. So swapping only the
    # spins AND isospins of i, j equals minus swapping only their positions.
    system = System(7, 7, 7, 1, 0.08)
    params, log_psi = jastrow_state(system, 2)
    r, s, t = config(system, 3)
    for i, j in [(0, 9), (2, 16), (10, 20)]:
        a = log_psi(params, r, swap(s, i, j), swap(t, i, j))
        idx = jnp.array([i, j])
        b = log_psi(params, r.at[idx].set(r[idx[::-1]]), s, t)
        d = a - b
        assert np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), -1.0, atol=1e-9)


# =============================================================================
# Block 2 - The v4 local energy against a brute-force calculation
# =============================================================================
# Pauli matrices; basis index 0 = up (+1), 1 = down (-1). Same for isospin
# with 0 = proton (+1), 1 = neutron (-1).
PAULI = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
DOT = sum(np.kron(p, p) for p in PAULI)  # sigma_1 . sigma_2 as a 4 x 4 matrix on |a b>
IDX = {+1: 0, -1: 1}
LAB = {0: +1, 1: -1}


def radial_brute(d, L, terms):
    """v(|d + L n|) summed over a large explicit cube of images."""
    g = np.arange(-6, 7)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    r2 = ((d[None, :] + L * n) ** 2).sum(-1)
    return sum(v * np.exp(-k * r2).sum() for v, k in terms)


@pytest.mark.parametrize("counts", [(7, 1, 1, 0), (1, 1, 1, 1)])
def test_local_energy_against_pauli_matrices(counts):
    # Independent route: apply the operators with explicit Pauli matrices.
    # For a pair (i, j) the two-body operator maps |s_i s_j t_i t_j> to a
    # combination of the 16 two-body states |s'_i s'_j t'_i t'_j>, so
    #   E_pot,L = sum_pairs sum_{primed} <primed|v_ij|labels> Psi(primed)/Psi.
    # No exchange identity is used here (sigma.sigma = 2 P - 1 is NOT assumed):
    # this checks the identity, the coefficients and the swapped evaluations.
    system = System(*counts, 0.08)
    L = system.box_length
    params, log_psi = jastrow_state(system, 4)
    r, s, t = config(system, 5)
    psi0 = complex(jnp.exp(log_psi(params, r, s, t)))
    e_brute = 0.0
    for i, j in itertools.combinations(range(system.n_particles), 2):
        d = np.asarray(minimum_image(r[i] - r[j], L))
        v = {op: radial_brute(d, L, RADIAL[op]) for op in RADIAL}
        a = IDX[int(s[i])] * 2 + IDX[int(s[j])]  # column index of |s_i s_j>
        b = IDX[int(t[i])] * 2 + IDX[int(t[j])]  # column index of |t_i t_j>
        for a2, b2 in itertools.product(range(4), range(4)):
            spin_c = 1.0 if a2 == a else 0.0
            iso_c = 1.0 if b2 == b else 0.0
            amp = (
                v["c"] * spin_c * iso_c
                + v["s"] * DOT[a2, a] * iso_c
                + v["t"] * spin_c * DOT[b2, b]
                + v["st"] * DOT[a2, a] * DOT[b2, b]
            )
            if amp == 0:
                continue
            s2 = s.at[i].set(LAB[a2 // 2]).at[j].set(LAB[a2 % 2])
            t2 = t.at[i].set(LAB[b2 // 2]).at[j].set(LAB[b2 % 2])
            psi2 = complex(jnp.exp(log_psi(params, r, s2, t2)))
            e_brute += amp * psi2 / psi0
    term = make_v4_potential(system, RADIAL)
    e = term(log_psi, params, r, s, t)
    assert np.isclose(e, e_brute, rtol=1e-8, atol=1e-8)


def test_central_part_matches_central_potential():
    # With only the "c" channel, the v4 term is the central potential.
    system = System(7, 7, 1, 0, 0.16)
    params, log_psi = jastrow_state(system, 6)
    r, s, t = config(system, 7)
    a = make_v4_potential(system, {"c": TOY})(log_psi, params, r, s, t)
    b = make_central_potential(system, TOY)(log_psi, params, r, s, t)
    assert np.isclose(a, b, rtol=1e-10)


def test_tau_tau_is_one_for_neutrons():
    # In pure neutron matter tau_i . tau_j = +1 for every pair: a "t" channel
    # acts exactly like the same radial function in the "c" channel.
    system = pure_neutron_matter(7, 0.16)
    params, log_psi = jastrow_state(system, 8)
    r, s, t = config(system, 9)
    a = make_v4_potential(system, {"t": RADIAL["t"]})(log_psi, params, r, s, t)
    b = make_v4_potential(system, {"c": RADIAL["t"]})(log_psi, params, r, s, t)
    assert np.isclose(a, b, rtol=1e-10)


# =============================================================================
# Block 3 - Exact expectation value in the Slater determinant
# =============================================================================
def hartree_fock_v4(system, radial):
    """Exact <V> for plane-wave spin-isospin orbitals (MeV, total).

    <V> = 1/(2V) sum_{a,b} sum_op [ v~_op(0)       <ab|O|ab>     (direct)
                                  - v~_op(k_a-k_b) <ab|O|ba> ]   (exchange)
    with <ab|O|ab> and <ab|O|ba> spin-isospin matrix elements of the
    product states of orbitals a and b.
    """
    L = system.box_length
    k, sl, tl = [], [], []
    for count, sz, tz in zip(system.counts, (1, -1, 1, -1), (-1, -1, 1, 1)):
        if count:
            k.append(occupied_kvectors(count, L))
            sl += [sz] * count
            tl += [tz] * count
    k = np.concatenate(k)
    one = np.eye(4)
    ops = {"c": (one, one), "s": (DOT, one), "t": (one, DOT), "st": (DOT, DOT)}
    total = 0.0
    for a in range(len(k)):
        for b in range(len(k)):
            sa, sb, ta, tb = IDX[sl[a]], IDX[sl[b]], IDX[tl[a]], IDX[tl[b]]
            ab_s, ba_s = sa * 2 + sb, sb * 2 + sa
            ab_t, ba_t = ta * 2 + tb, tb * 2 + ta
            q2 = ((k[a] - k[b]) ** 2).sum()
            for op, terms in radial.items():
                ms, mt = ops[op]
                vq0 = sum(v * (np.pi / kk) ** 1.5 for v, kk in terms)
                vq = sum(v * (np.pi / kk) ** 1.5 * np.exp(-q2 / (4 * kk)) for v, kk in terms)
                direct = (ms[ab_s, ab_s] * mt[ab_t, ab_t]).real
                exchange = (ms[ab_s, ba_s] * mt[ab_t, ba_t]).real
                total += vq0 * direct - vq * exchange
    return total / (2 * L**3)


def test_vmc_reproduces_hartree_fock_v4():
    # As for the central potential: the exact Slater expectation value,
    # now with spin-isospin exchange matrix elements, against sampling.
    # Neutrons and protons, both spins: every operator channel contributes.
    # The slowest test so far (about 1 minute on a CPU, much less on the GPU):
    # every sample needs Psi at 3 x 120 label-swapped configurations.
    system = System(7, 7, 1, 1, 0.16)
    N, L = system.n_particles, system.box_length
    params, log_psi = make_slater(system)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    term = make_v4_potential(system, RADIAL)
    v_batch = jax.jit(jax.vmap(lambda r: term(log_psi, params, r, s, t).real))
    log_prob = make_log_prob(log_psi, s, t)
    key = jax.random.PRNGKey(11)
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
    assert err < 0.15
