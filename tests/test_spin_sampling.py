"""Sampling positions and spinors together.

Interface (added to src/nsmatter/sampler.py):
    random_spinors(key, shape)                    -> uniform unit spinors, shape + (2,)
    rotate_spinors(key, xi, spin_step)            -> random SU(2) rotation of each spinor
    make_log_prob_spin(log_psi, t)                -> log_prob(params, r, xi)
    metropolis_spin(log_prob, params, key, r, xi, step, spin_step, L, n_steps)
        -> (r, xi, position acceptance, spinor acceptance)
    thermalize_spin(log_prob, params, key, r, xi, step, spin_step, L, ...)
        -> (r, xi, step, spin_step, key)
Run with:  uv run pytest -q tests/test_spin_sampling.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.geometry import occupied_kvectors
from nsmatter.hamiltonian.exchange import make_v4_potential
from nsmatter.sampler import make_log_prob_spin, metropolis_spin, random_spinors, thermalize_spin, uniform_walkers
from nsmatter.stats import blocking_error
from nsmatter.system import pure_neutron_matter
from nsmatter.wavefunctions.slater import make_slater

L = 4.0


def run(log_psi, n_particles, n_walkers, n_meas, observable, seed):
    """Sample |Psi|^2 and return the blocking mean and error of an observable."""
    t = -jnp.ones(n_particles, dtype=jnp.int32)
    log_prob = make_log_prob_spin(log_psi, t)
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)
    r = uniform_walkers(k1, n_walkers, n_particles, L)
    xi = random_spinors(k2, (n_walkers, n_particles))
    r, xi, step, spin_step, key = thermalize_spin(log_prob, {}, key, r, xi, 0.5, 0.5, L)
    series = []
    for _ in range(n_meas):
        key, sub = jax.random.split(key)
        r, xi, _, _ = metropolis_spin(log_prob, {}, sub, r, xi, step, spin_step, L, 5)
        series.append(float(observable(xi)))
    return blocking_error(series), xi


# =============================================================================
# Test 1 - Spinors stay uniform on the sphere when |Psi|^2 does not depend on them
# =============================================================================
def test_flat_distribution_stays_uniform():
    # For spinors uniform on the unit sphere of C^2, u = |xi_up|^2 is
    # uniform on [0, 1]: mean 1/2, variance 1/12. Rotations must also keep
    # every spinor normalized.
    flat = lambda params, r, xi, t: jnp.zeros((), dtype=jnp.complex128)  # noqa: E731
    (mean, err), xi = run(flat, 2, 2048, 50, lambda xi: jnp.mean(jnp.abs(xi[..., 0]) ** 2), 0)
    assert jnp.allclose(jnp.linalg.norm(xi, axis=-1), 1.0, atol=1e-12)
    assert abs(mean - 0.5) < 5 * err
    u = np.asarray(jnp.abs(xi[..., 0]) ** 2).ravel()
    assert np.isclose(u.var(), 1 / 12, rtol=0.1)


# =============================================================================
# Test 2 - A spin-dependent |Psi|^2 with an exact answer
# =============================================================================
def test_weighted_spinor_distribution():
    # One particle with Psi = xi_up: |Psi|^2 = u, so the sampled u has density
    # 2u on [0, 1] and <u> = 2/3 exactly. This checks the spinor acceptance
    # rule (and that the proposal is symmetric: an asymmetric one biases <u>).
    up_only = lambda params, r, xi, t: jnp.log(xi[0, 0])  # noqa: E731
    (mean, err), _ = run(up_only, 1, 2048, 200, lambda xi: jnp.mean(jnp.abs(xi[..., 0]) ** 2), 1)
    assert abs(mean - 2 / 3) < 5 * err
    assert err < 0.003


# =============================================================================
# Test 3 - Spin-dependent forces with sampled spinors: exact Hartree-Fock value
# =============================================================================
DOT = np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])


def hartree_fock_sigma(system, radial):
    """Exact <V> of v_c + v_s sigma.sigma in the plane-wave Slater determinant (MeV, total).

    Pure neutron matter: tau.tau = 1, so only the spin matrix elements matter.
    <V> = 1/(2V) sum_{a,b} [ v~(0) <ab|O|ab> - v~(k_a - k_b) <ab|O|ba> ].
    """
    L = system.box_length
    k, sl = [], []
    for count, sz in zip(system.counts[:2], (0, 1)):  # 0 = up, 1 = down
        k.append(occupied_kvectors(count, L))
        sl += [sz] * count
    k = np.concatenate(k)
    ops = {"c": np.eye(4), "s": DOT}
    total = 0.0
    for a in range(len(k)):
        for b in range(len(k)):
            q2 = ((k[a] - k[b]) ** 2).sum()
            ab, ba = 2 * sl[a] + sl[b], 2 * sl[b] + sl[a]
            for op, terms in radial.items():
                vq0 = sum(V * (np.pi / kk) ** 1.5 for V, kk in terms)
                vq = sum(V * (np.pi / kk) ** 1.5 * np.exp(-q2 / (4 * kk)) for V, kk in terms)
                total += vq0 * ops[op][ab, ab] - vq * ops[op][ab, ba]
    return total / (2 * L**3)


def test_spin_forces_with_sampled_spinors():
    # Same state as with fixed labels, now with the spins sampled as spinors:
    # the expectation value of a spin-dependent potential must still be the
    # exact Hartree-Fock value. This tests the spinor measure, the spinor
    # moves and the spin operators together.
    system = pure_neutron_matter(7, 0.16)
    N, Lb = system.n_particles, system.box_length
    radial = {"c": ((200.0, 1.487), (-30.0, 0.465)), "s": ((-25.0, 0.7),)}
    params, log_psi = make_slater(system)
    t = jnp.asarray(system.isospins)
    term = make_v4_potential(system, radial)
    v_batch = jax.jit(jax.vmap(lambda r, xi: term(log_psi, params, r, xi, t).real))
    log_prob = make_log_prob_spin(log_psi, t)
    key = jax.random.PRNGKey(2)
    k1, k2, key = jax.random.split(key, 3)
    r = uniform_walkers(k1, 128, N, Lb)
    xi = random_spinors(k2, (128, N))
    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, 0.1 * Lb / np.sqrt(N), 0.3, Lb)
    series = []
    for _ in range(150):
        key, sub = jax.random.split(key)
        r, xi, _, _ = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, Lb, 5)
        series.append(float(v_batch(r, xi).mean()) / N)
    mean, err = blocking_error(series)
    exact = hartree_fock_sigma(system, radial) / N
    assert abs(mean - exact) < 5 * err
    assert err < 0.3
