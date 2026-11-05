"""Metropolis sampling of |Psi|^2 over particle positions.

Many independent chains (walkers) run in parallel with vmap; each step
proposes a Gaussian displacement of all particles of a walker at once and
accepts it with probability min(1, |Psi(R')|^2 / |Psi(R)|^2).
Spins and isospins stay fixed (fixed-label convention, see system.py).
"""

from functools import partial

import jax
import jax.numpy as jnp

from .geometry import wrap


def make_log_prob(log_psi, s, t):
    """ln |Psi(R)|^2 for one walker."""

    def log_prob(params, r):
        return 2.0 * jnp.real(log_psi(params, r, s, t))

    return log_prob


@partial(jax.jit, static_argnames=("log_prob", "n_steps"))
def metropolis(log_prob, params, key, r, step, L, n_steps):
    """Run n_steps Metropolis steps on a batch of walkers r (W, N, 3).

    Returns (new positions, mean acceptance rate).
    """
    batched = jax.vmap(log_prob, in_axes=(None, 0))
    lp = batched(params, r)

    def one_step(carry, key):
        r, lp = carry
        k_move, k_acc = jax.random.split(key)
        proposal = wrap(r + step * jax.random.normal(k_move, r.shape), L)
        lp_new = batched(params, proposal)
        log_u = jnp.log(jax.random.uniform(k_acc, lp.shape))
        accept = log_u < lp_new - lp
        r = jnp.where(accept[:, None, None], proposal, r)
        lp = jnp.where(accept, lp_new, lp)
        return (r, lp), accept.mean()

    (r, _), acc = jax.lax.scan(one_step, (r, lp), jax.random.split(key, n_steps))
    return r, acc.mean()


def thermalize(log_prob, params, key, r, step, L, n_rounds=20, steps_per_round=50, target=0.5):
    """Equilibrate the walkers while tuning the step size towards a target acceptance."""
    for _ in range(n_rounds):
        key, sub = jax.random.split(key)
        r, acc = metropolis(log_prob, params, sub, r, step, L, steps_per_round)
        step = step * float(jnp.clip(acc / target, 0.5, 2.0))
    return r, step, key


def uniform_walkers(key, n_walkers, n_particles, L):
    """Random initial positions, uniform in the box."""
    return jax.random.uniform(key, (n_walkers, n_particles, 3), maxval=L)


# =============================================================================
# Sampling positions AND spinors
# =============================================================================
# Pauli matrices, for the SU(2) rotations of the spinors.
_PAULI = jnp.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=jnp.complex128)


def random_spinors(key, shape):
    """Spinors uniform on the unit sphere of C^2, shape shape + (2,).

    A normalized complex Gaussian 2-vector is uniform on that sphere.
    """
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, shape + (2,)) + 1j * jax.random.normal(k2, shape + (2,))
    return z / jnp.linalg.norm(z, axis=-1, keepdims=True)


def rotate_spinors(key, xi, spin_step):
    """Apply an independent random SU(2) rotation to every spinor.

    U = cos(theta/2) 1 - i sin(theta/2) n.sigma, with n a random unit vector
    and theta ~ spin_step * N(0, 1). theta and -theta are equally likely, so
    U and its inverse are equally likely: the proposal is symmetric with
    respect to the uniform measure on the sphere, as Metropolis requires.
    """
    k1, k2 = jax.random.split(key)
    n = jax.random.normal(k1, xi.shape[:-1] + (3,))
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    theta = spin_step * jax.random.normal(k2, xi.shape[:-1])
    n_sigma = jnp.einsum("...a,aij->...ij", n, _PAULI)
    eye = jnp.eye(2, dtype=jnp.complex128)
    U = jnp.cos(theta / 2)[..., None, None] * eye - 1j * jnp.sin(theta / 2)[..., None, None] * n_sigma
    return jnp.einsum("...ij,...j->...i", U, xi)


def make_log_prob_spin(log_psi, t):
    """log_prob(params, r, xi) = ln |Psi(r, xi)|^2; the isospin labels t are fixed."""

    def log_prob(params, r, xi):
        return 2.0 * jnp.real(log_psi(params, r, xi, t))

    return log_prob


@partial(jax.jit, static_argnames=("log_prob", "n_steps"))
def metropolis_spin(log_prob, params, key, r, xi, step, spin_step, L, n_steps):
    """n_steps Metropolis steps on walkers with positions r (W, N, 3) and spinors xi (W, N, 2).

    Each step is two moves, each with its own accept/reject:
    first all positions of a walker (Gaussian displacement), then all its
    spinors (random SU(2) rotations).
    Returns (r, xi, position acceptance, spinor acceptance).
    """
    batched = jax.vmap(log_prob, in_axes=(None, 0, 0))
    lp = batched(params, r, xi)

    def accept_or_keep(key, new, old, lp_new, lp_old):
        log_u = jnp.log(jax.random.uniform(key, lp_old.shape))
        acc = log_u < lp_new - lp_old
        shape = (-1,) + (1,) * (old.ndim - 1)
        return jnp.where(acc.reshape(shape), new, old), jnp.where(acc, lp_new, lp_old), acc.mean()

    def one_step(carry, key):
        r, xi, lp = carry
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # Position move.
        r_new = wrap(r + step * jax.random.normal(k1, r.shape), L)
        r, lp, acc_r = accept_or_keep(k2, r_new, r, batched(params, r_new, xi), lp)
        # Spinor move.
        xi_new = rotate_spinors(k3, xi, spin_step)
        xi, lp, acc_s = accept_or_keep(k4, xi_new, xi, batched(params, r, xi_new), lp)
        return (r, xi, lp), (acc_r, acc_s)

    (r, xi, _), (acc_r, acc_s) = jax.lax.scan(one_step, (r, xi, lp), jax.random.split(key, n_steps))
    return r, xi, acc_r.mean(), acc_s.mean()


def thermalize_spin(log_prob, params, key, r, xi, step, spin_step, L, n_rounds=20, steps_per_round=50, target=0.5):
    """Equilibrate and tune both step sizes towards `target` acceptance.

    The spinor step is capped at pi: larger rotation angles add nothing.
    Returns (r, xi, step, spin_step, key).
    """
    for _ in range(n_rounds):
        key, sub = jax.random.split(key)
        r, xi, acc_r, acc_s = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, steps_per_round)
        step = step * float(jnp.clip(acc_r / target, 0.5, 2.0))
        spin_step = min(float(jnp.pi), spin_step * float(jnp.clip(acc_s / target, 0.5, 2.0)))
    return r, xi, step, spin_step, key
