"""Metropolis sampling of |Psi|^2 over particle positions.

Many independent chains (walkers) run in parallel with vmap. Each step
proposes a Gaussian displacement of all particles of a walker at once,
wraps them back into the box, and accepts with probability
min(1, |Psi(R')|^2 / |Psi(R)|^2). The proposal is symmetric (the box is a
torus), so no correction factor is needed. Spins and isospins stay fixed
(fixed-label convention, see system.py).
"""

from functools import partial

import jax
import jax.numpy as jnp

from .geometry import wrap


def make_log_prob(log_psi, s, t):
    """Return log_prob(params, r) = ln |Psi(r)|^2 = 2 Re ln Psi for one walker.

    The labels s, t are fixed, so they are captured here once.
    """

    def log_prob(params, r):
        return 2.0 * jnp.real(log_psi(params, r, s, t))

    return log_prob


# jax.jit compiles the function. `log_prob` (a Python function) and
# `n_steps` (it sets the loop length) must be known at compile time,
# so they are declared static: a new value triggers a recompilation.
@partial(jax.jit, static_argnames=("log_prob", "n_steps"))
def metropolis(log_prob, params, key, r, step, L, n_steps):
    """Run n_steps Metropolis steps on a batch of walkers.

    r: positions, shape (W, N, 3) for W walkers.
    step: standard deviation of the Gaussian displacement (fm).
    Returns (new positions, acceptance rate averaged over steps and walkers).
    """
    # log_prob for all walkers at once: map over the walker axis of r,
    # share params (in_axes=None).
    batched = jax.vmap(log_prob, in_axes=(None, 0))
    lp = batched(params, r)

    def one_step(carry, key):
        r, lp = carry
        # Two independent random streams: one for the move, one for acceptance.
        k_move, k_acc = jax.random.split(key)
        proposal = wrap(r + step * jax.random.normal(k_move, r.shape), L)
        lp_new = batched(params, proposal)
        # Accept if ln u < ln|Psi'|^2 - ln|Psi|^2, u uniform in (0, 1).
        # Working with logs avoids overflow in the ratio.
        log_u = jnp.log(jax.random.uniform(k_acc, lp.shape))
        accept = log_u < lp_new - lp
        # Per walker: keep the proposal where accepted, the old one otherwise.
        r = jnp.where(accept[:, None, None], proposal, r)
        lp = jnp.where(accept, lp_new, lp)
        return (r, lp), accept.mean()

    # lax.scan compiles the whole loop at once (no return to Python per step).
    # One fresh key per step, split from the input key: reproducible chains.
    (r, _), acc = jax.lax.scan(one_step, (r, lp), jax.random.split(key, n_steps))
    return r, acc.mean()


def thermalize(log_prob, params, key, r, step, L, n_rounds=20, steps_per_round=50, target=0.5):
    """Equilibrate the walkers and tune the step size towards `target` acceptance.

    The step is rescaled after each round by acceptance / target, limited to
    a factor 2 per round. Tuning happens ONLY here: changing the step during
    production would break detailed balance.
    Returns (positions, tuned step, updated key).
    """
    for _ in range(n_rounds):
        key, sub = jax.random.split(key)
        r, acc = metropolis(log_prob, params, sub, r, step, L, steps_per_round)
        step = step * float(jnp.clip(acc / target, 0.5, 2.0))
    return r, step, key


def uniform_walkers(key, n_walkers, n_particles, L):
    """Random initial positions, uniform in the box, shape (W, N, 3)."""
    return jax.random.uniform(key, (n_walkers, n_particles, 3), maxval=L)
