"""Tests for the Jastrow factor (Jastrow x plane-wave Slater wavefunction).

Interface defined here (module src/nsmatter/wavefunctions/jastrow.py):
    N_CHANNELS = 2   (0: same isospin, 1: opposite isospin; spin dependence
                      goes into the operator correlations, since spins are spinors)
    jastrow_basis(r2, rc, n_terms) -> basis functions phi_m, shape (..., n_terms)
    pair_channel(t, i, j)          -> channel 0..1 of the pairs (i, j)
    make_jastrow_slater(system, n_terms=4, cutoff=None) -> (params, log_psi)
        params = {"jastrow": array of shape (N_CHANNELS, n_terms)}, zeros at start
Run with:  uv run pytest -q tests/test_jastrow.py
"""

# =============================================================================
# Block 0 - Imports and helpers
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401  (float64)
from nsmatter.constants import HBAR2_2M
from nsmatter.geometry import minimum_image
from nsmatter.hamiltonian.kinetic import kinetic_local_energy
from nsmatter.system import System, pure_neutron_matter, symmetric_matter
from nsmatter.wavefunctions.jastrow import N_CHANNELS, jastrow_basis, make_jastrow_slater, pair_channel
from nsmatter.wavefunctions.slater import make_slater

SYSTEMS = [
    pure_neutron_matter(7, 0.16),
    symmetric_matter(7, 0.16),
    System(19, 7, 7, 1, 0.08),
]


def random_config(system, seed=0):
    """Random positions in the box plus the fixed spin/isospin labels."""
    key = jax.random.PRNGKey(seed)
    r = jax.random.uniform(key, (system.n_particles, 3), maxval=system.box_length)
    return r, jnp.asarray(system.spins), jnp.asarray(system.isospins)


def random_params(params, seed=0, scale=0.3):
    """Same structure as params, filled with random numbers (a non-trivial Jastrow)."""
    key = jax.random.PRNGKey(seed)
    return {"jastrow": scale * jax.random.normal(key, params["jastrow"].shape)}


def same_psi(a, b):
    """Two values of ln Psi describe the same Psi (phases compared modulo 2 pi)."""
    d = a - b
    return np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), 1.0, atol=1e-9)


# =============================================================================
# Block 1 - The radial basis functions
# =============================================================================
RC = 3.0  # cutoff used in the basis tests (fm)


def u_of_r(m):
    """The m-th basis function as a function of the distance r (not r^2)."""
    return lambda r: jastrow_basis(r**2, RC, 4)[m]


def test_basis_shape():
    r2 = jnp.linspace(0.0, 2 * RC**2, 7)
    assert jastrow_basis(r2, RC, 4).shape == (7, 4)


@pytest.mark.parametrize("m", range(4))
def test_basis_smooth_at_cutoff(m):
    # u, u' and u'' must all vanish at r = rc. Otherwise ln Psi has a kink
    # (or a jump in curvature) where pairs cross the cutoff, and the kinetic
    # energy silently misses a delta-function contribution there.
    # Evaluated just INSIDE the cutoff: exactly at rc the code takes the
    # "zero" branch, and that would pass even with a non-smooth envelope.
    # With the (1 - x^2)^3 envelope, at r = rc (1 - 1e-5) the values are
    # about 1e-14, 1e-9 and 5e-5; a (1 - x^2)^2 envelope would give u'' ~ 1.
    f = u_of_r(m)
    d1 = jax.grad(f)
    d2 = jax.grad(d1)
    r = jnp.asarray(RC * (1.0 - 1e-5))
    assert abs(f(r)) < 1e-12 and abs(d1(r)) < 1e-8 and abs(d2(r)) < 1e-3


@pytest.mark.parametrize("m", range(4))
def test_basis_zero_beyond_cutoff(m):
    # Beyond the cutoff the pair is uncorrelated: u = 0 exactly.
    f = u_of_r(m)
    for r in (RC * 1.001, RC * 1.3, RC * 2.0):
        assert f(jnp.asarray(r)) == 0.0


@pytest.mark.parametrize("m", range(4))
def test_basis_smooth_at_origin(m):
    # The NV2 potentials are regular at r = 0, so the wavefunction has no
    # cusp: u'(0) = 0. Using powers of r^2 guarantees it, and also avoids
    # the derivative of |d| at d = 0, which is NaN in automatic differentiation.
    d1 = jax.grad(u_of_r(m))
    assert abs(d1(jnp.asarray(0.0))) < 1e-12


# =============================================================================
# Block 2 - Pair channels
# =============================================================================
def test_pair_channels():
    # Isospin labels of four particles: n, n, p, p.
    t = jnp.array([-1, -1, +1, +1])
    i = jnp.array([0, 0, 0, 2, 1])
    j = jnp.array([0, 1, 2, 3, 2])
    # (0,0) and (0,1): two neutrons -> 0;  (0,2): n-p -> 1;
    # (2,3): two protons -> 0;  (1,2): n-p -> 1
    assert pair_channel(t, i, j).tolist() == [0, 0, 1, 0, 1]


# =============================================================================
# Block 3 - Properties of the full wavefunction
# =============================================================================
@pytest.mark.parametrize("system", SYSTEMS)
def test_zero_parameters_reduce_to_slater(system):
    # The parameters start at zero: J = 0 and Psi is exactly the Slater
    # determinant. The optimization then starts from the free gas.
    params, log_psi = make_jastrow_slater(system)
    sp, slater = make_slater(system)
    r, s, t = random_config(system)
    assert params["jastrow"].shape == (N_CHANNELS, 4)
    assert np.allclose(params["jastrow"], 0.0)
    assert same_psi(log_psi(params, r, s, t), slater(sp, r, s, t))


@pytest.mark.parametrize("system", SYSTEMS)
def test_antisymmetry_preserved(system):
    # J is symmetric under exchange, so exp(J) * Slater is still antisymmetric.
    params, log_psi = make_jastrow_slater(system)
    params = random_params(params)
    r, s, t = random_config(system)
    idx = jnp.array([0, 1])
    r_swapped = r.at[idx].set(r[idx[::-1]])
    d = log_psi(params, r_swapped, s, t) - log_psi(params, r, s, t)
    assert np.isclose(d.real, 0.0, atol=1e-9)
    assert np.isclose(np.cos(d.imag), -1.0, atol=1e-9)


@pytest.mark.parametrize("system", SYSTEMS)
def test_periodicity_and_translation(system):
    # J depends only on minimum-image distances: periodic in each particle,
    # and unchanged when all particles move together.
    params, log_psi = make_jastrow_slater(system)
    params = random_params(params, seed=1)
    r, s, t = random_config(system, seed=1)
    ref = log_psi(params, r, s, t)
    for axis in range(3):
        assert same_psi(log_psi(params, r.at[2, axis].add(system.box_length), s, t), ref)
    assert same_psi(log_psi(params, r + jnp.array([0.4, -1.1, 2.3]), s, t), ref)


def test_cutoff_larger_than_half_box_rejected():
    # A cutoff above L/2 would make a pair interact with two images at once
    # and break the minimum-image construction.
    system = SYSTEMS[0]
    with pytest.raises(ValueError):
        make_jastrow_slater(system, cutoff=0.6 * system.box_length)


# =============================================================================
# Block 4 - J against a hand calculation
# =============================================================================
def test_two_particle_value_by_hand():
    # Two neutrons (channel 0, same isospin), each alone in its species,
    # so the Slater part is 1 (only k = 0 is occupied). Then
    #   ln Psi = u_0(r) = (1 - x^2)^3 * sum_m a_{0,m} x^(2m),  x = r / rc,
    # with r the minimum-image distance and rc = L/2 by default.
    system = System(1, 1, 0, 0, 0.01)
    L = system.box_length
    params, log_psi = make_jastrow_slater(system)
    a = jnp.zeros((N_CHANNELS, 4)).at[0].set(jnp.array([0.5, -0.2, 0.1, 0.05]))
    r = jnp.array([[0.3, 0.2, 0.1], [L - 0.9, 1.5, 0.4]])  # close through the boundary
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    d = np.asarray(minimum_image(r[0] - r[1], L))
    x2 = (d**2).sum() / (0.5 * L) ** 2
    expected = (1 - x2) ** 3 * sum(float(a[0, m]) * x2**m for m in range(4))
    value = log_psi({"jastrow": a}, r, s, t)
    assert np.isclose(value.real, expected, rtol=1e-12)
    assert np.isclose(np.cos(value.imag), 1.0)


# =============================================================================
# Block 5 - Derivatives through the Jastrow factor
# =============================================================================
def test_kinetic_energy_against_finite_differences():
    # With a Jastrow, Psi is no longer an eigenstate. Check the autodiff
    # kinetic local energy against -hbar^2/2m * (lap Psi) / Psi, with the
    # Laplacian of Psi itself (not ln Psi) taken by finite differences.
    system = System(7, 1, 1, 0, 0.16)
    params, log_psi = make_jastrow_slater(system)
    params = random_params(params, seed=2)
    r, s, t = random_config(system, seed=3)

    def psi(x):
        return jnp.exp(log_psi(params, x.reshape(r.shape), s, t))

    x0 = r.reshape(-1)
    h = 1e-4
    lap = 0.0
    for k in range(x0.size):
        e = jnp.zeros_like(x0).at[k].set(h)
        lap += (psi(x0 + e) - 2 * psi(x0) + psi(x0 - e)) / h**2
    e_fd = -HBAR2_2M * lap / psi(x0)
    e_ad = kinetic_local_energy(log_psi, params, r, s, t)
    assert np.isclose(e_ad, e_fd, rtol=1e-5, atol=1e-4)


def test_local_energy_varies():
    # Not an eigenstate any more: E_L differs between configurations.
    # (This is what will give the sampler and the statistics real work.)
    system = SYSTEMS[0]
    params, log_psi = make_jastrow_slater(system)
    params = random_params(params, seed=4)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    e = [kinetic_local_energy(log_psi, params, random_config(system, k)[0], s, t).real for k in range(2)]
    assert abs(e[0] - e[1]) > 1e-3


def test_parameter_gradient_is_the_basis_sum():
    # J is linear in the parameters, so
    #   d ln Psi / d a_{c,m} = sum over pairs in channel c of phi_m(r_ij).
    # These log-derivatives are exactly what the optimizer will need.
    system = System(1, 1, 1, 1, 0.08)  # one particle per species: 6 pairs, all channels
    L = system.box_length
    params, log_psi = make_jastrow_slater(system)
    r, s, t = random_config(system, seed=5)
    grad = jax.grad(lambda p: log_psi(p, r, s, t).real)(params)["jastrow"]
    expected = np.zeros((N_CHANNELS, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            d = minimum_image(r[i] - r[j], L)
            phi = jastrow_basis(jnp.sum(d**2), 0.5 * L, 4)
            c = int(pair_channel(t, jnp.array(i), jnp.array(j)))
            expected[c] += np.asarray(phi)
    assert np.allclose(grad, expected, atol=1e-12)
