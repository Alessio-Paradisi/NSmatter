"""Tests for stochastic reconfiguration (SR).

Interface defined here (module src/nsmatter/optimize.py):
    log_derivatives(log_psi, params, r, s, t) -> O, complex, shape (W, P)
        O[w, k] = d ln Psi(R_w) / d p_k for every walker w and parameter k
        (parameters flattened in jax.flatten_util.ravel_pytree order)
    sr_direction(O, e_loc, diag_shift=...) -> (delta, grad, S)
        grad_k = 2 Re < (O_k - <O_k>)^* (E_L - <E_L>) >
        S_kl   =   Re < (O_k - <O_k>)^* (O_l - <O_l>) >
        delta  = (S + diag_shift * 1)^-1 grad
    apply_update(params, delta, lr) -> params - lr * delta (same structure)
Run with:  uv run pytest -q tests/test_optimize.py
"""

# =============================================================================
# Block 0 - Imports and helpers
# =============================================================================
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import nsmatter  # noqa: F401  (float64)
from nsmatter.hamiltonian.central import hartree_fock_energy, make_central_potential
from nsmatter.local_energy import make_local_energy
from nsmatter.optimize import apply_update, log_derivatives, sr_direction
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.system import pure_neutron_matter
from nsmatter.wavefunctions.jastrow import make_jastrow_slater
from nsmatter.wavefunctions.slater import free_gas_energy


def synthetic_O(n_walkers=4000, n_params=5, seed=0):
    """Random real log-derivatives with different scales per parameter."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(n_walkers, n_params)) * np.array([1.0, 0.1, 3.0, 0.5, 2.0])[:n_params])


# =============================================================================
# Block 1 - Log-derivatives
# =============================================================================
def test_log_derivatives_shape_and_values():
    # One row per walker, one column per parameter. For the Jastrow-Slater
    # state, ln Psi = J + ln Slater with J real and linear in the parameters,
    # and the Slater part does not depend on them: O must be real and equal
    # to the gradient of Re ln Psi, computed here walker by walker.
    system = pure_neutron_matter(7, 0.16)
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    params = {"jastrow": 0.2 * jax.random.normal(jax.random.PRNGKey(0), params["jastrow"].shape)}
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    r = jax.random.uniform(jax.random.PRNGKey(1), (6, 14, 3), maxval=system.box_length)
    O = log_derivatives(log_psi, params, r, s, t)
    assert O.shape == (6, params["jastrow"].size)
    assert jnp.allclose(O.imag, 0.0, atol=1e-12)
    for w in range(6):
        g = jax.grad(lambda p: log_psi(p, r[w], s, t).real)(params)
        assert jnp.allclose(O[w].real, ravel_pytree(g)[0], atol=1e-12)


def test_unused_channels_have_zero_derivative():
    # Pure neutron matter has no neutron-proton pairs: the parameters of the
    # opposite-isospin channels (2 and 3) do not affect Psi at all.
    # SR must leave them untouched (the diagonal shift keeps S invertible).
    system = pure_neutron_matter(7, 0.16)
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    r = jax.random.uniform(jax.random.PRNGKey(2), (50, 14, 3), maxval=system.box_length)
    O = log_derivatives(log_psi, params, r, s, t).reshape(50, 4, 3)
    assert jnp.allclose(O[:, 2:, :], 0.0)
    e = jax.random.normal(jax.random.PRNGKey(3), (50,))
    delta, _, _ = sr_direction(O.reshape(50, 12), e, diag_shift=0.05)
    assert jnp.allclose(delta.reshape(4, 3)[2:], 0.0)


# =============================================================================
# Block 2 - The SR linear algebra on data with a known answer
# =============================================================================
def test_sr_recovers_linear_model():
    # If E_L depends linearly on the log-derivatives, E_L = c + sum_k b_k O_k,
    # then grad = 2 S b exactly, and with a vanishing diagonal shift the SR
    # direction is delta = S^-1 grad = 2 b.
    O = synthetic_O()
    b = jnp.array([0.7, -2.0, 0.1, 1.5, -0.3])
    e = 3.0 + O @ b
    delta, grad, S = sr_direction(O, e, diag_shift=0.0)
    assert jnp.allclose(grad, 2 * S @ b, rtol=1e-10)
    assert jnp.allclose(delta, 2 * b, rtol=1e-8)


def test_constant_local_energy_gives_no_update():
    # Zero variance (an eigenstate within the ansatz): nothing to improve.
    O = synthetic_O()
    delta, grad, _ = sr_direction(O, jnp.full(O.shape[0], 12.5), diag_shift=1e-3)
    assert jnp.allclose(grad, 0.0) and jnp.allclose(delta, 0.0)


def test_s_matrix_properties():
    # S is the covariance matrix of the log-derivatives: symmetric and
    # positive semi-definite. With real O it is just the sample covariance.
    O = synthetic_O()
    _, _, S = sr_direction(O, jnp.zeros(O.shape[0]), diag_shift=1e-3)
    assert jnp.allclose(S, S.T)
    assert jnp.all(jnp.linalg.eigvalsh(S) > -1e-12)
    assert jnp.allclose(S, np.cov(np.asarray(O).T, bias=True), rtol=1e-10)


def test_sr_is_invariant_under_rescaling_parameters():
    # The key property of SR (a "natural gradient"): if parameter k is
    # rescaled, p_k = alpha_k p'_k, then O'_k = alpha_k O_k and the SR step
    # in p' is delta'_k = delta_k / alpha_k -- the SAME physical change of the
    # wavefunction. Plain gradient descent would change by alpha_k^2 instead,
    # which is why it struggles when parameters have very different scales.
    O = synthetic_O()
    e = jnp.sin(O[:, 0]) + O[:, 2] ** 2 - 0.5 * O[:, 4]
    alpha = jnp.array([10.0, 0.01, 1.0, 3.0, 0.2])
    # No diagonal shift here: a shift is NOT scale invariant (it is the price
    # paid for stability), so the exact property only holds without it.
    d, _, _ = sr_direction(O, e, diag_shift=0.0)
    d_scaled, _, _ = sr_direction(O * alpha, e, diag_shift=0.0)
    assert jnp.allclose(d_scaled * alpha, d, rtol=1e-8)


def test_apply_update_keeps_structure():
    # The update works on the flat vector but returns the original structure.
    params = {"a": jnp.ones((2, 3)), "b": jnp.array([5.0])}
    flat, _ = ravel_pytree(params)
    delta = jnp.arange(flat.size, dtype=float)
    new = apply_update(params, delta, 0.1)
    assert new["a"].shape == (2, 3) and new["b"].shape == (1,)
    assert jnp.allclose(ravel_pytree(new)[0], flat - 0.1 * delta)


# =============================================================================
# Block 3 - A real optimization
# =============================================================================
def test_optimization_lowers_energy_and_variance():
    # Jastrow-Slater for 14 neutrons at 0.16 fm^-3 with the toy potential.
    # Start (J = 0): E = free gas + Hartree-Fock = 33.87 MeV per particle,
    # with a large variance of E_L (the repulsive core is felt in full).
    # After 30 SR steps the Jastrow suppresses close pairs: the energy must
    # drop clearly below the starting value (by > 1.5 MeV per particle; it
    # reaches ~31.3) and the variance of E_L by more than a factor 3.
    system = pure_neutron_matter(7, 0.16)
    N, L = system.n_particles, system.box_length
    params, log_psi = make_jastrow_slater(system, n_terms=3)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    e_loc = jax.jit(jax.vmap(make_local_energy(log_psi, (make_central_potential(system),)), in_axes=(None, 0, None, None)))
    log_derivs = jax.jit(lambda p, r: log_derivatives(log_psi, p, r, s, t))
    log_prob = make_log_prob(log_psi, s, t)

    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, 256, N, L)
    r, step, key = thermalize(log_prob, params, key, r, 0.1 * L / np.sqrt(N), L)

    energies, variances = [], []
    for _ in range(30):
        key, sub = jax.random.split(key)
        r, _ = metropolis(log_prob, params, sub, r, step, L, 10)
        e = e_loc(params, r, s, t).real
        O = log_derivs(params, r)
        delta, _, _ = sr_direction(O, e, diag_shift=0.05)
        params = apply_update(params, delta, 0.002)
        energies.append(float(e.mean()) / N)
        variances.append(float(e.var()) / N**2)

    start = (free_gas_energy(system) + hartree_fock_energy(system)) / N
    assert np.mean(energies[-10:]) < start - 1.5
    assert np.mean(variances[-10:]) < variances[0] / 3
