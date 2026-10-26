"""Tests for the foundation: constants, periodic geometry, system composition.

These tests are written before the code: they define the interface
(module and function names) and what "correct" means for each piece.
Run with:  uv run pytest -q tests/test_foundation.py
"""

# =============================================================================
# Block 0 - Imports and setup
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Importing the package also turns on float64 (via src/nsmatter/__init__.py).
import nsmatter  # noqa: F401

# The names below are the interface the three modules must provide.
from nsmatter.constants import HBAR2_2M
from nsmatter.geometry import closed_shells, minimum_image, occupied_kvectors, wrap
from nsmatter.system import System, pure_neutron_matter, symmetric_matter

# JAX defaults to float32; every 1e-12 comparison below needs float64.
# Repeated here so this file does not depend on the package __init__.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Block 1 - Constants
# =============================================================================
def test_hbar2_2m():
    # hbar^2 / 2m with the average nucleon mass is about 20.7355 MeV fm^2.
    # Loose tolerance: catches a wrong formula (missing 2, hbar*c instead of
    # (hbar*c)^2), not the exact choice of mass values.
    assert np.isclose(HBAR2_2M, 20.7355, atol=1e-3)


# =============================================================================
# Block 2 - Closed momentum shells
# =============================================================================
def test_closed_shells():
    # Cumulative number of integer vectors n with |n|^2 <= 0, 1, 2, 3, 4, 5, 6, 8.
    # |n|^2 = 7 is missing (7 is not a sum of three squares), hence 81 -> 93.
    assert closed_shells()[:8] == [1, 7, 19, 27, 33, 57, 81, 93]


# =============================================================================
# Block 3 - Occupied plane-wave momenta k = 2 pi n / L
# =============================================================================
L_TEST = 5.0  # box side used by the geometry tests (fm); any value works


def test_kvectors_shape_and_scale():
    # One k-vector per particle, three components each.
    k = occupied_kvectors(19, L_TEST)
    assert k.shape == (19, 3)
    # Converting back to n = k L / (2 pi) must give integers:
    # catches a missing 2 pi or a wrong box side.
    n = k * L_TEST / (2 * np.pi)
    assert np.allclose(n, np.round(n))


def test_kvectors_sorted_by_energy():
    # States are filled from the lowest |k|^2 up (non-decreasing order).
    # The small negative tolerance absorbs floating-point rounding.
    k2 = (occupied_kvectors(33, L_TEST) ** 2).sum(axis=1)
    assert np.all(np.diff(k2) >= -1e-12)


def test_kvectors_parity():
    # In a closed shell every k comes with -k: total momentum is zero.
    # This is what makes the free ground state parity-symmetric.
    for count in (1, 7, 19, 27, 33):
        assert np.allclose(occupied_kvectors(count, L_TEST).sum(axis=0), 0.0)


def test_kvectors_open_shell_and_empty():
    # 8 particles would half-fill the |n|^2 = 2 shell: must be refused.
    with pytest.raises(ValueError):
        occupied_kvectors(8, L_TEST)
    # An empty species (e.g. no protons in neutron matter) must still work.
    assert occupied_kvectors(0, L_TEST).shape == (0, 3)


def test_kvectors_deterministic():
    # The order inside a shell is arbitrary but must be the same every call,
    # so that runs are reproducible.
    assert np.array_equal(occupied_kvectors(27, L_TEST), occupied_kvectors(27, L_TEST))


# =============================================================================
# Block 4 - Wrapping and minimum image
# =============================================================================
def test_wrap():
    # JAX has no global random state: a key is created and passed explicitly.
    key = jax.random.PRNGKey(0)
    # Positions spread over several boxes, including negative values.
    r = jax.random.uniform(key, (100, 3), minval=-3 * L_TEST, maxval=3 * L_TEST)
    w = wrap(r, L_TEST)
    # Wrapped positions lie in the box. The upper bound is <= L, not < L:
    # jnp.mod(-1e-17, 5.0) returns exactly 5.0 because of rounding, and a
    # position exactly at L is physically the same point as 0.
    assert jnp.all((w >= 0.0) & (w <= L_TEST))
    # Wrapping only shifts by whole multiples of L: no physical displacement.
    shift = (w - r) / L_TEST
    assert jnp.allclose(shift, jnp.round(shift), atol=1e-12)


def test_minimum_image_range():
    # Separation vectors several boxes long, in all directions.
    key = jax.random.PRNGKey(1)
    d = jax.random.uniform(key, (1000, 3), minval=-4 * L_TEST, maxval=4 * L_TEST)
    m = minimum_image(d, L_TEST)
    # Every component of the minimum image lies within +-L/2.
    assert jnp.all(jnp.abs(m) <= 0.5 * L_TEST + 1e-12)


def test_minimum_image_periodic_invariance():
    # Two random particles in the box.
    key = jax.random.PRNGKey(2)
    r = jax.random.uniform(key, (2, 3), maxval=L_TEST)
    d0 = minimum_image(r[0] - r[1], L_TEST)
    for axis in range(3):
        # Move particle 0 by one box along `axis`. JAX arrays are immutable:
        # .at[...].add(...) returns a modified copy.
        r_shifted = r.at[0, axis].add(L_TEST)
        d1 = minimum_image(r_shifted[0] - r_shifted[1], L_TEST)
        # The separation the physics sees must not change.
        assert jnp.allclose(d0, d1, atol=1e-12)


def test_minimum_image_is_shortest():
    # Check the definition directly: the minimum image must be the shortest
    # of the 27 copies d + L*(i, j, k) with i, j, k in {-1, 0, 1}.
    key = jax.random.PRNGKey(3)
    d = jax.random.uniform(key, (3,), minval=-L_TEST, maxval=L_TEST)
    m = minimum_image(d, L_TEST)
    images = [d + L_TEST * jnp.array([i, j, k]) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    shortest = min(float(jnp.linalg.norm(x)) for x in images)
    assert np.isclose(float(jnp.linalg.norm(m)), shortest)


# =============================================================================
# Block 5 - System composition
# =============================================================================
# The test system (19, 7, 7, 1) has four different populations on purpose:
# a swapped index (protons/neutrons, up/down) cannot hide behind symmetry.
# Order of the arguments: n_up, n_down, p_up, p_down, density (fm^-3).


def test_system_counts_and_box():
    s = System(19, 7, 7, 1, 0.08)
    assert s.n_particles == 34
    # Box side from the density: N / L^3 = rho.
    assert np.isclose(s.box_length**3 * s.density, 34)
    # Proton fraction x = N_p / N.
    assert np.isclose(s.proton_fraction, 8 / 34)
    # Spin polarization Delta = (N_up - N_down) / N, neutrons and protons together.
    assert np.isclose(s.polarization, ((19 + 7) - (7 + 1)) / 34)


def test_system_labels():
    s = System(19, 7, 7, 1, 0.08)
    spins, isospins = s.spins, s.isospins
    # One label per particle.
    assert spins.shape == isospins.shape == (34,)
    # Only +1 / -1 values are allowed.
    assert set(np.unique(spins)) <= {-1, 1} and set(np.unique(isospins)) <= {-1, 1}
    # Conventions: spin up = +1; proton = +1, neutron = -1.
    assert spins.sum() == (19 + 7) - (7 + 1)
    assert isospins.sum() == (7 + 1) - (19 + 7)


def test_system_slices_partition_particles():
    s = System(19, 7, 7, 1, 0.08)
    # The four species slices together cover indices 0..N-1 exactly once, in order.
    covered = np.concatenate([np.arange(34)[sl] for sl in s.species_slices])
    assert np.array_equal(covered, np.arange(34))
    # Order n_up, n_down, p_up, p_down, with (spin, isospin) labels:
    expected = [(+1, -1), (-1, -1), (+1, +1), (-1, +1)]
    for sl, count, (sz, tz) in zip(s.species_slices, s.counts, expected):
        # Each slice has as many particles as its species...
        assert len(range(34)[sl]) == count
        # ...and every particle in it carries that species' labels.
        assert np.all(s.spins[sl] == sz) and np.all(s.isospins[sl] == tz)


def test_empty_species_allowed():
    # Fully polarized neutron matter: three empty species.
    s = System(7, 0, 0, 0, 0.16)
    assert s.n_particles == 7
    # Empty slices must exist and must not break the indexing.
    assert [len(range(7)[sl]) for sl in s.species_slices] == [7, 0, 0, 0]


def test_helpers():
    # Shortcuts: the argument is the number of particles per species.
    pnm = pure_neutron_matter(7, 0.16)
    assert pnm.counts == (7, 7, 0, 0) and pnm.proton_fraction == 0.0
    snm = symmetric_matter(7, 0.16)
    assert snm.counts == (7, 7, 7, 7) and snm.proton_fraction == 0.5 and snm.polarization == 0.0


def test_system_is_hashable():
    # Equal content -> equal objects with equal hash. JAX needs this to accept
    # a System as a static (compile-time) argument; a frozen dataclass gives it.
    a, b = System(7, 7, 0, 0, 0.16), System(7, 7, 0, 0, 0.16)
    assert hash(a) == hash(b) and a == b
