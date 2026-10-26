"""Tests for the Metropolis sampler and the error analysis.

Interface defined here:
    src/nsmatter/sampler.py
        make_log_prob(log_psi, s, t) -> log_prob(params, r) = ln |Psi(r)|^2
        metropolis(log_prob, params, key, r, step, L, n_steps) -> (r, acceptance)
        thermalize(log_prob, params, key, r, step, L, ...) -> (r, step, key)
        uniform_walkers(key, n_walkers, n_particles, L) -> r, shape (W, N, 3)
    src/nsmatter/stats.py
        blocking(series) -> (mean, errors per level, blocks per level)
        blocking_error(series, min_blocks=32) -> (mean, error)
        integrated_autocorrelation_time(series) -> tau_int
Run with:  uv run pytest -q tests/test_sampling.py
"""

# =============================================================================
# Block 0 - Imports
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0, i1

import nsmatter  # noqa: F401  (float64)
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking, blocking_error, integrated_autocorrelation_time


# =============================================================================
# Block 1 - Error analysis on series with known statistics
# =============================================================================
def test_blocking_white_noise():
    # Uncorrelated Gaussian data: the error of the mean is sigma / sqrt(n),
    # and blocking must not inflate it.
    rng = np.random.default_rng(0)
    x = rng.normal(size=2**14)
    mean, err = blocking_error(x)
    assert abs(mean) < 4 * err
    assert np.isclose(err, 1 / np.sqrt(x.size), rtol=0.2)


def ar1_series(phi, n, seed):
    """AR(1) process x_i = phi x_{i-1} + noise: a model of a correlated Markov chain.

    Exact results: variance 1/(1 - phi^2), tau_int = (1 + phi) / (2 (1 - phi)).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_blocking_correlated_data():
    # For correlated data the naive error is too small by sqrt(2 tau_int).
    # Blocking must recover the true error: sqrt(variance * 2 tau_int / n).
    phi, n = 0.9, 2**16
    x = ar1_series(phi, n, seed=1)
    _, err = blocking_error(x)
    expected = np.sqrt(1 / (1 - phi**2) * (1 + phi) / (1 - phi) / n)
    assert np.isclose(err, expected, rtol=0.25)
    # The naive error (level 0) is much smaller: the plateau rises above it.
    _, errors, _ = blocking(x)
    assert errors.max() > 3 * errors[0]


def test_autocorrelation_time():
    # tau_int for phi = 0.9 is 9.5: one independent sample every ~19 steps.
    phi = 0.9
    x = ar1_series(phi, 2**16, seed=2)
    tau = integrated_autocorrelation_time(x)
    assert np.isclose(tau, (1 + phi) / (2 * (1 - phi)), rtol=0.2)


# =============================================================================
# Block 2 - Sampler basics
# =============================================================================
L = 4.0  # box side for the sampler tests (fm)


def test_uniform_walkers_shape_and_range():
    r = uniform_walkers(jax.random.PRNGKey(0), 16, 5, L)
    assert r.shape == (16, 5, 3)
    assert jnp.all((r >= 0) & (r <= L))


def flat_log_psi(params, r, s, t):
    # Psi = 1: every configuration has the same probability.
    return jnp.zeros((), dtype=jnp.complex128)


def test_flat_distribution_accepts_everything_and_stays_uniform():
    # With |Psi|^2 constant every proposal is accepted (acceptance = 1), and
    # the walkers stay uniform in the box: mean L/2, variance L^2/12 per axis.
    s = t = jnp.ones(2, dtype=jnp.int32)
    log_prob = make_log_prob(flat_log_psi, s, t)
    r = uniform_walkers(jax.random.PRNGKey(1), 4096, 2, L)
    r, acc = metropolis(log_prob, {}, jax.random.PRNGKey(2), r, 0.7, L, 20)
    assert np.isclose(acc, 1.0)
    assert jnp.all((r >= 0) & (r <= L))
    x = np.asarray(r).ravel()
    assert np.isclose(x.mean(), L / 2, atol=4 * L / np.sqrt(12 * x.size))
    assert np.isclose(x.var(), L**2 / 12, rtol=0.03)


# =============================================================================
# Block 3 - Sampling a non-trivial periodic distribution
# =============================================================================
def test_von_mises_distribution():
    # ln Psi = a * sum over all coordinates of cos(2 pi x / L), so each
    # coordinate is independent with |Psi|^2 ~ exp(2a cos(2 pi x / L))
    # (a von Mises distribution). Exact result:
    #     < cos(2 pi x / L) > = I1(2a) / I0(2a)
    # with I0, I1 modified Bessel functions. This checks the acceptance rule
    # (a wrong ratio, a missing factor 2 in |Psi|^2, or broken wrapping all
    # move the answer). The estimate must agree within 4 error bars.
    a = 0.8

    def log_psi(params, r, s, t):
        return (a * jnp.sum(jnp.cos(2 * jnp.pi * r / L))).astype(jnp.complex128)

    s = t = jnp.ones(2, dtype=jnp.int32)
    log_prob = make_log_prob(log_psi, s, t)
    key = jax.random.PRNGKey(3)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, 512, 2, L)

    # Equilibrate and tune the step; acceptance should end near 50%.
    r, step, key = thermalize(log_prob, {}, key, r, 0.5, L)
    key, sub = jax.random.split(key)
    _, acc = metropolis(log_prob, {}, sub, r, step, L, 50)
    assert 0.35 < acc < 0.65

    # Production: one walker-averaged measurement every 5 steps.
    series = []
    for _ in range(400):
        key, sub = jax.random.split(key)
        r, _ = metropolis(log_prob, {}, sub, r, step, L, 5)
        series.append(float(jnp.cos(2 * jnp.pi * r / L).mean()))
    mean, err = blocking_error(series)
    exact = i1(2 * a) / i0(2 * a)
    assert abs(mean - exact) < 4 * err
    assert err < 0.01  # enough statistics for the test to mean something


def test_reproducible_with_same_key():
    # Same key -> identical chain. Different key -> different chain.
    s = t = jnp.ones(2, dtype=jnp.int32)
    log_prob = make_log_prob(flat_log_psi, s, t)
    r0 = uniform_walkers(jax.random.PRNGKey(4), 8, 2, L)
    a, _ = metropolis(log_prob, {}, jax.random.PRNGKey(5), r0, 0.3, L, 10)
    b, _ = metropolis(log_prob, {}, jax.random.PRNGKey(5), r0, 0.3, L, 10)
    c, _ = metropolis(log_prob, {}, jax.random.PRNGKey(6), r0, 0.3, L, 10)
    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)
