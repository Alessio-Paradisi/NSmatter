"""Tests for the central test potential (sum of Gaussians, periodic in the box).

Interface defined here (module src/nsmatter/hamiltonian/central.py):
    TOY = ((V_1, kappa_1), (V_2, kappa_2), ...)    v(r) = sum_k V_k exp(-kappa_k r^2)
    gaussian_sum(r2, terms)                -> v at squared distances r2
    image_shifts(L, terms, tol=1e-10)      -> periodic image vectors L*n, shape (M, 3)
    make_central_potential(system, terms=TOY, tol=1e-10)
        -> term(log_psi, params, r, s, t), the potential part of E_L (complex)
    hartree_fock_energy(system, terms=TOY) -> exact <V> in the Slater determinant (MeV, total)
Run with:  uv run pytest -q tests/test_central.py
"""

# =============================================================================
# Block 0 - Imports and helpers
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.hamiltonian.central import TOY, gaussian_sum, hartree_fock_energy, image_shifts, make_central_potential
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking_error
from nsmatter.system import System, pure_neutron_matter
from nsmatter.wavefunctions.slater import make_slater


def labels(system):
    return jnp.asarray(system.spins), jnp.asarray(system.isospins)


# =============================================================================
# Block 1 - The radial function
# =============================================================================
def test_gaussian_sum_values():
    # v(r) = sum_k V_k exp(-kappa_k r^2), evaluated at r^2.
    # At r = 0 it is the sum of the strengths; far away it vanishes.
    assert np.isclose(gaussian_sum(jnp.asarray(0.0), TOY), sum(v for v, _ in TOY))
    r2 = jnp.asarray(1.3**2)
    expected = sum(v * np.exp(-k * 1.3**2) for v, k in TOY)
    assert np.isclose(gaussian_sum(r2, TOY), expected)
    assert abs(gaussian_sum(jnp.asarray(30.0**2), TOY)) < 1e-12


def test_toy_potential_is_stable():
    # A central potential with integral  v~(0) = int v(r) d^3r < 0  makes matter
    # collapse (the Hartree energy -> -infinity as the density grows).
    # The toy has a repulsive core and a weak attraction with v~(0) > 0.
    vq0 = sum(v * (np.pi / k) ** 1.5 for v, k in TOY)
    assert vq0 > 0


# =============================================================================
# Block 2 - Periodic images
# =============================================================================
def test_image_shifts_structure():
    # Image vectors L*n on a cube of integer vectors, containing n = 0.
    L = 4.0
    sh = image_shifts(L, TOY)
    assert sh.shape[1] == 3
    n = sh / L
    assert np.allclose(n, np.round(n))
    assert np.any(np.all(sh == 0.0, axis=1))
    # A cube (2m+1)^3 of images.
    side = round(sh.shape[0] ** (1 / 3))
    assert side**3 == sh.shape[0] and side % 2 == 1


def test_two_particles_against_brute_force_images():
    # Two particles: the periodic potential is sum over ALL images n of
    # v(|r1 - r2 + L n|). Compare with an explicit, much larger image sum.
    system = System(1, 1, 0, 0, 0.16)  # small box (L = 2.3 fm): images matter a lot
    L = system.box_length
    term = make_central_potential(system)
    r = jnp.array([[0.2, 0.4, 1.9], [2.1, 0.1, 0.3]])
    s, t = labels(system)
    g = np.arange(-8, 9)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    d = np.asarray(r[0] - r[1])[None, :] + L * n
    r2 = (d**2).sum(-1)
    brute = sum(v * np.exp(-k * r2).sum() for v, k in TOY)
    assert np.isclose(term(None, None, r, s, t).real, brute, rtol=1e-10)


def test_periodicity_and_symmetry():
    # Moving a particle by L, or exchanging two particles, changes nothing.
    system = pure_neutron_matter(7, 0.16)
    term = make_central_potential(system)
    s, t = labels(system)
    r = jax.random.uniform(jax.random.PRNGKey(0), (14, 3), maxval=system.box_length)
    ref = term(None, None, r, s, t)
    assert np.isclose(term(None, None, r.at[4, 2].add(system.box_length), s, t), ref)
    idx = jnp.array([3, 9])
    assert np.isclose(term(None, None, r.at[idx].set(r[idx[::-1]]), s, t), ref)


def test_image_sum_converged():
    # The default tolerance must give the same energy as a much tighter one.
    system = pure_neutron_matter(7, 0.16)
    s, t = labels(system)
    r = jax.random.uniform(jax.random.PRNGKey(1), (14, 3), maxval=system.box_length)
    a = make_central_potential(system)(None, None, r, s, t)
    b = make_central_potential(system, tol=1e-16)(None, None, r, s, t)
    assert np.isclose(a, b, rtol=1e-9)


# =============================================================================
# Block 3 - Exact expectation value in the Slater determinant
# =============================================================================
def test_vmc_reproduces_hartree_fock():
    # For plane waves the expectation value of a periodic central potential
    # is known exactly (direct minus exchange, through the Fourier transform
    # v~(q) = sum_k V_k (pi/kappa_k)^(3/2) exp(-q^2 / 4 kappa_k)):
    #   <V> = 1/(2V) [ N^2 v~(0) - sum_species sum_{a,b} v~(k_a - k_b) ].
    # The same number is computed here a second, independent way: sample
    # |Slater|^2 and average the potential term in real space. They must
    # agree within 5 error bars. This checks the potential, the image sum,
    # the sampler with a FLUCTUATING estimator, and the error bars together.
    # 5 rather than 3-4 sigma: with only 150 measurements the blocking error
    # is itself uncertain (by ~20%), and the test must not fail by chance
    # (a GPU run produces a different, equally valid chain).
    system = pure_neutron_matter(7, 0.16)
    N, L = system.n_particles, system.box_length
    params, log_psi = make_slater(system)
    s, t = labels(system)
    term = make_central_potential(system)
    log_prob = make_log_prob(log_psi, s, t)
    v_batch = jax.jit(jax.vmap(lambda r: term(log_psi, params, r, s, t).real))

    key = jax.random.PRNGKey(2)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, 256, N, L)
    r, step, key = thermalize(log_prob, params, key, r, 0.1 * L / np.sqrt(N), L)
    series = []
    for _ in range(150):
        key, sub = jax.random.split(key)
        r, _ = metropolis(log_prob, params, sub, r, step, L, 5)
        series.append(float(v_batch(r).mean()) / N)
    mean, err = blocking_error(series)
    exact = hartree_fock_energy(system) / N
    assert abs(mean - exact) < 5 * err
    assert err < 0.1  # the comparison must be precise enough to mean something
