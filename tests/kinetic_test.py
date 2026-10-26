"""Tests for the kinetic local energy.

Interface defined here:
    src/nsmatter/hamiltonian/kinetic.py
        grad_and_laplacian(f, x) -> (gradient vector, Laplacian scalar)
        kinetic_local_energy(log_psi, params, r, s, t, hbar2_2m=HBAR2_2M) -> complex
    src/nsmatter/local_energy.py
        make_local_energy(log_psi, potential_terms=()) -> local_energy(params, r, s, t)
Run with:  uv run pytest -q tests/test_kinetic.py
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
from nsmatter.hamiltonian.kinetic import grad_and_laplacian, kinetic_local_energy
from nsmatter.local_energy import make_local_energy
from nsmatter.system import System, pure_neutron_matter, symmetric_matter
from nsmatter.wavefunctions.slater import free_gas_energy, make_slater

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


# =============================================================================
# Block 1 - Derivatives: autodiff against finite differences
# =============================================================================
def test_grad_and_laplacian_finite_differences():
    # A complex function of real variables, like ln Psi. Central finite
    # differences with step h are accurate to O(h^2): ~1e-8 for the first
    # derivative and ~1e-6 for the second (rounding grows as 1/h^2).
    def f(x):
        return jnp.sin(x[0]) * jnp.exp(1j * x[1] * x[2]) + x[3] ** 3

    x = jnp.array([0.3, -0.7, 1.1, 0.5])
    grad, lap = grad_and_laplacian(f, x)
    h = 1e-4
    eye = np.eye(x.size)
    g_fd = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in eye])
    l_fd = sum((f(x + h * e) - 2 * f(x) + f(x - h * e)) / h**2 for e in eye)
    assert grad.shape == x.shape
    assert np.allclose(grad, g_fd, atol=1e-7)
    assert np.isclose(lap, l_fd, atol=1e-5)


# =============================================================================
# Block 2 - Analytic case where E_L is NOT constant
# =============================================================================
def test_gaussian_local_energy_analytic():
    # One particle (no box) with ln Psi = -alpha |r|^2. Then
    #   grad ln Psi = -2 alpha r,   lap ln Psi = -6 alpha,
    #   E_kin,L = -hbar^2/2m * (-6 alpha + 4 alpha^2 |r|^2).
    # Unlike the Slater case, E_L depends on r: this checks that both the
    # Laplacian and the (grad)^2 term are there, with the right signs.
    alpha = 0.3

    def log_psi(params, r, s, t):
        return (-alpha * jnp.sum(r**2)).astype(jnp.complex128)

    r = jnp.array([[0.4, -1.2, 0.7]])
    s = t = jnp.array([1])
    e = kinetic_local_energy(log_psi, {}, r, s, t)
    expected = -HBAR2_2M * (-6 * alpha + 4 * alpha**2 * jnp.sum(r**2))
    assert np.isclose(e.real, expected, rtol=1e-12)
    assert np.isclose(e.imag, 0.0, atol=1e-12)


def test_plane_wave_phase_contributes():
    # One particle with ln Psi = i k.r (a single plane wave):
    #   grad ln Psi = i k,  lap = 0,  (grad)^2 = -|k|^2,
    #   E_kin,L = +hbar^2/2m * |k|^2.
    # The energy comes entirely from the SQUARE of a purely imaginary
    # gradient: this fails if the code takes the modulus or the real part too early.
    k = jnp.array([0.5, -1.0, 2.0])

    def log_psi(params, r, s, t):
        return 1j * jnp.dot(k, r[0])

    r = jnp.array([[0.1, 0.2, 0.3]])
    s = t = jnp.array([1])
    e = kinetic_local_energy(log_psi, {}, r, s, t)
    assert np.isclose(e.real, HBAR2_2M * jnp.sum(k**2), rtol=1e-12)


# =============================================================================
# Block 3 - The Slater determinant is an exact eigenstate
# =============================================================================
@pytest.mark.parametrize("system", SYSTEMS)
def test_slater_local_energy_is_exact(system):
    # For an eigenstate H Psi = E Psi, so E_L(R) = E at EVERY configuration:
    # check several random ones. The imaginary part must vanish too.
    params, log_psi = make_slater(system)
    e_loc = jax.jit(make_local_energy(log_psi))
    exact = free_gas_energy(system)
    for seed in range(3):
        r, s, t = random_config(system, seed)
        e = e_loc(params, r, s, t)
        assert np.isclose(e.real, exact, rtol=1e-9)
        assert abs(e.imag) < 1e-7 * exact


# =============================================================================
# Block 4 - Local energy assembly
# =============================================================================
def test_potential_terms_are_added():
    # make_local_energy adds every potential term to the kinetic part.
    # A constant fake "potential" of 5 MeV must shift E_L by exactly 5.
    system = SYSTEMS[0]
    params, log_psi = make_slater(system)
    r, s, t = random_config(system)

    def constant_term(log_psi, params, r, s, t):
        return jnp.asarray(5.0 + 0.0j)

    e0 = make_local_energy(log_psi)(params, r, s, t)
    e1 = make_local_energy(log_psi, potential_terms=(constant_term,))(params, r, s, t)
    assert np.isclose(e1 - e0, 5.0, atol=1e-9)


def test_vmap_over_walkers():
    # The same function works on a batch of walkers via jax.vmap:
    # positions get a leading walker axis, params and labels are shared.
    system = SYSTEMS[0]
    params, log_psi = make_slater(system)
    batched = jax.jit(jax.vmap(make_local_energy(log_psi), in_axes=(None, 0, None, None)))
    key = jax.random.PRNGKey(7)
    r = jax.random.uniform(key, (4, system.n_particles, 3), maxval=system.box_length)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    e = batched(params, r, s, t)
    assert e.shape == (4,)
    assert np.allclose(e.real, free_gas_energy(system), rtol=1e-9)
