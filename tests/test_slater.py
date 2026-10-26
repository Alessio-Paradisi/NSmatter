"""Tests for the plane-wave Slater determinant (the free Fermi gas in the box).

Interface defined here (module src/nsmatter/wavefunctions/slater.py):
    make_slater(system) -> (params, log_psi)
        log_psi(params, r, s, t) -> complex scalar ln Psi
    free_gas_energy(system)          -> exact box energy, total (MeV)
    free_gas_energy_infinite(system) -> infinite free gas at the same densities, total (MeV)
Run with:  uv run pytest -q tests/test_slater.py
"""

# =============================================================================
# Block 0 - Imports and helpers
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401  (float64)
from nsmatter.system import System, pure_neutron_matter, symmetric_matter
from nsmatter.wavefunctions.slater import free_gas_energy, free_gas_energy_infinite, make_slater

# Three compositions used by most tests: unpolarized neutron matter,
# symmetric matter, and four different populations (catches index mix-ups).
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


def same_psi(a, b):
    """True if two values of ln Psi describe the same Psi.

    ln Psi is complex: real part = ln|Psi|, imaginary part = phase.
    Phases are only defined modulo 2 pi, so compare exp(i * phase difference).
    """
    d = a - b
    return np.isclose(d.real, 0.0, atol=1e-9) and np.isclose(np.cos(d.imag), 1.0, atol=1e-9)


# =============================================================================
# Block 1 - Interface
# =============================================================================
def test_returns_complex_scalar():
    # log_psi takes one configuration (N, 3) and returns a single complex
    # number. Batching over walkers is done outside, with jax.vmap.
    system = SYSTEMS[0]
    params, log_psi = make_slater(system)
    r, s, t = random_config(system)
    value = log_psi(params, r, s, t)
    assert value.shape == ()
    assert jnp.iscomplexobj(value)


# =============================================================================
# Block 2 - Antisymmetry
# =============================================================================
@pytest.mark.parametrize("system", SYSTEMS)
def test_antisymmetry_within_species(system):
    # Exchanging two identical fermions (same species) flips the sign of Psi:
    # ln Psi keeps its real part and its phase shifts by pi.
    params, log_psi = make_slater(system)
    r, s, t = random_config(system)
    idx = jnp.array([0, 1])  # particles 0 and 1 are both in the first species
    r_swapped = r.at[idx].set(r[idx[::-1]])
    d = log_psi(params, r_swapped, s, t) - log_psi(params, r, s, t)
    assert np.isclose(d.real, 0.0, atol=1e-9)
    assert np.isclose(np.cos(d.imag), -1.0, atol=1e-9)


def test_coincident_particles_give_a_node():
    # Pauli principle: two identical fermions at the same point -> Psi = 0,
    # i.e. ln|Psi| -> -infinity (or a very large negative number).
    system = SYSTEMS[0]
    params, log_psi = make_slater(system)
    r, s, t = random_config(system)
    r = r.at[1].set(r[0])
    assert log_psi(params, r, s, t).real < -20.0


# =============================================================================
# Block 3 - Symmetries of the box
# =============================================================================
@pytest.mark.parametrize("system", SYSTEMS)
def test_periodicity(system):
    # Moving one particle by L along any axis must leave Psi unchanged:
    # this is the periodic boundary condition, k = 2 pi n / L guarantees it.
    params, log_psi = make_slater(system)
    r, s, t = random_config(system, seed=1)
    for axis in range(3):
        r_shift = r.at[3, axis].add(system.box_length)
        assert same_psi(log_psi(params, r_shift, s, t), log_psi(params, r, s, t))


@pytest.mark.parametrize("system", SYSTEMS)
def test_translation_invariance(system):
    # Moving ALL particles by the same vector a multiplies each species'
    # determinant by exp(i a . sum k). In a closed shell sum k = 0,
    # so Psi does not change: the state has zero total momentum.
    params, log_psi = make_slater(system)
    r, s, t = random_config(system, seed=2)
    a = jnp.array([0.37, -1.21, 2.05])
    assert same_psi(log_psi(params, r + a, s, t), log_psi(params, r, s, t))


# =============================================================================
# Block 4 - Free-gas energies (reference numbers)
# =============================================================================
def test_free_gas_energy_reference_values():
    # Exact energy of the box Slater determinant, E = hbar^2/2m * sum |k|^2,
    # and of the infinite free gas, E/N = 3/5 hbar^2 kF^2 / 2m per species.
    # Reference numbers for N = 14 neutrons at 0.16 fm^-3 (MeV per particle).
    system = pure_neutron_matter(7, 0.16)
    N = system.n_particles
    assert np.isclose(free_gas_energy(system) / N, 35.60047909, atol=1e-6)
    assert np.isclose(free_gas_energy_infinite(system) / N, 35.09351229, atol=1e-6)


def test_finite_size_effect_shrinks():
    # The box energy approaches the infinite-gas energy as N grows:
    # the relative gap for N = 66 is smaller than for N = 14.
    gaps = []
    for n in (7, 19, 33):
        s = pure_neutron_matter(n, 0.16)
        gaps.append(abs(free_gas_energy(s) / free_gas_energy_infinite(s) - 1))
    assert gaps[-1] < gaps[0]


def test_empty_species_contribute_nothing():
    # Fully polarized neutron matter: only one determinant, no crash,
    # and the energy equals that of a single species.
    s = System(7, 0, 0, 0, 0.16)
    params, log_psi = make_slater(s)
    r, sp, t = random_config(s)
    assert np.isfinite(log_psi(params, r, sp, t).real)
    assert free_gas_energy(s) > 0
