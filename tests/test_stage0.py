"""Stage 0 tests: geometry, Slater determinant, kinetic energy, statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401  (enables float64)
from nsmatter.geometry import closed_shells, minimum_image, occupied_kvectors
from nsmatter.hamiltonian.kinetic import grad_and_laplacian
from nsmatter.local_energy import make_local_energy
from nsmatter.stats import blocking_error
from nsmatter.system import System, pure_neutron_matter, symmetric_matter
from nsmatter.wavefunctions.slater import free_gas_energy, free_gas_energy_infinite, make_slater

SYSTEMS = [
    pure_neutron_matter(7, 0.16),
    symmetric_matter(7, 0.16),
    System(19, 7, 7, 1, 0.08),  # four different populations
]


def random_config(system, seed=0):
    key = jax.random.PRNGKey(seed)
    r = jax.random.uniform(key, (system.n_particles, 3), maxval=system.box_length)
    return r, jnp.asarray(system.spins), jnp.asarray(system.isospins)


def test_closed_shells():
    assert closed_shells()[:8] == [1, 7, 19, 27, 33, 57, 81, 93]


def test_open_shell_rejected():
    with pytest.raises(ValueError):
        occupied_kvectors(8, 5.0)


def test_minimum_image_range():
    d = jnp.array([[0.6, -0.6, 2.3]])
    m = minimum_image(d, 1.0)
    assert jnp.all(jnp.abs(m) <= 0.5)


def test_composition():
    s = System(19, 7, 7, 1, 0.08)
    assert s.n_particles == 34
    assert np.isclose(s.box_length**3 * s.density, 34)
    assert s.spins.sum() == (19 - 7) + (7 - 1)
    assert s.isospins.sum() == 8 - 26


@pytest.mark.parametrize("system", SYSTEMS)
def test_antisymmetry_within_species(system):
    """Swapping two particles of the same species flips the sign: ln Psi shifts by i*pi."""
    params, log_psi = make_slater(system)
    r, s, t = random_config(system)
    i, j = 0, 1  # both in the first species (n_up)
    r_swapped = r.at[jnp.array([i, j])].set(r[jnp.array([j, i])])
    diff = log_psi(params, r_swapped, s, t) - log_psi(params, r, s, t)
    assert np.isclose(diff.real, 0.0, atol=1e-9)
    assert np.isclose(np.cos(diff.imag), -1.0, atol=1e-9)


@pytest.mark.parametrize("system", SYSTEMS)
def test_periodicity(system):
    """Moving any particle by L along any axis leaves Psi unchanged."""
    params, log_psi = make_slater(system)
    r, s, t = random_config(system, seed=1)
    L = system.box_length
    r_shift = r.at[3, 1].add(L)
    diff = log_psi(params, r_shift, s, t) - log_psi(params, r, s, t)
    assert np.isclose(diff.real, 0.0, atol=1e-9)
    assert np.isclose(np.cos(diff.imag), 1.0, atol=1e-9)


def test_laplacian_against_finite_differences():
    """Autodiff gradient and Laplacian of a complex test function vs central differences."""

    def f(x):
        return jnp.sin(x[0]) * jnp.exp(1j * x[1] * x[2]) + x[3] ** 3

    x = jnp.array([0.3, -0.7, 1.1, 0.5])
    grad, lap = grad_and_laplacian(f, x)
    h = 1e-4
    eye = np.eye(x.size)
    g_fd = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in eye])
    l_fd = sum((f(x + h * e) - 2 * f(x) + f(x - h * e)) / h**2 for e in eye)
    assert np.allclose(grad, g_fd, atol=1e-7)
    assert np.isclose(lap, l_fd, atol=1e-5)


@pytest.mark.parametrize("system", SYSTEMS)
def test_local_energy_is_exact_constant(system):
    """The plane-wave Slater is an eigenstate: E_L equals the exact energy at every configuration."""
    params, log_psi = make_slater(system)
    e_loc = jax.jit(make_local_energy(log_psi))
    exact = free_gas_energy(system)
    for seed in range(3):
        r, s, t = random_config(system, seed)
        e = e_loc(params, r, s, t)
        assert np.isclose(e.real, exact, rtol=1e-9)
        assert np.isclose(e.imag, 0.0, atol=1e-7 * exact)


def test_finite_size_approaches_infinite_limit():
    """Shell effects shrink as N grows: |E_N/E_inf - 1| decreases from N = 14 to 66."""
    gaps = []
    for n in (7, 19, 33):
        s = pure_neutron_matter(n, 0.16)
        gaps.append(abs(free_gas_energy(s) / free_gas_energy_infinite(s) - 1))
    assert gaps[-1] < gaps[0]


def test_blocking_white_noise():
    """For uncorrelated data the blocking error equals sigma / sqrt(n)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2**14)
    mean, err = blocking_error(x)
    assert abs(mean) < 4 * err
    assert np.isclose(err, 1 / np.sqrt(x.size), rtol=0.2)


def test_blocking_correlated_data():
    """For an AR(1) series the blocking error recovers the larger, correlated error."""
    rng = np.random.default_rng(1)
    phi, n = 0.9, 2**16
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    _, err = blocking_error(x)
    sigma2 = 1 / (1 - phi**2)
    tau_factor = (1 + phi) / (1 - phi)  # 2 * tau_int
    expected = np.sqrt(sigma2 * tau_factor / n)
    assert np.isclose(err, expected, rtol=0.25)
