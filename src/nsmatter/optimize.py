"""Stochastic reconfiguration (SR)."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def log_derivatives(log_psi, params, r, s, t):
    flat, unravel = ravel_pytree(params)

    def re(p, x):
        return jnp.real(log_psi(unravel(p), x, s, t))

    def im(p, x):
        return jnp.imag(log_psi(unravel(p), x, s, t))

    g_re = jax.vmap(jax.grad(re), in_axes=(None, 0))(flat, r)
    g_im = jax.vmap(jax.grad(im), in_axes=(None, 0))(flat, r)
    return g_re + 1j * g_im


def sr_direction(O, e_loc, diag_shift=1e-3):
    dO = O - O.mean(axis=0)
    dE = e_loc - e_loc.mean()
    W = O.shape[0]
    grad = 2.0 * jnp.real(jnp.conj(dO).T @ dE) / W
    S = jnp.real(jnp.conj(dO).T @ dO) / W
    S_reg = S + diag_shift * jnp.eye(S.shape[0])
    delta = jnp.linalg.solve(S_reg, grad)
    return delta, grad, S


def apply_update(params, delta, lr):
    flat, unravel = ravel_pytree(params)
    return unravel(flat - lr * delta)


def log_derivatives_spin(log_psi, params, r, xi, t):
    """O for walkers that carry their own spinors: r (W, N, 3), xi (W, N, 2).

    Same as log_derivatives, but maps over the spinors too.
    """
    flat, unravel = ravel_pytree(params)

    def re(p, x, s):
        return jnp.real(log_psi(unravel(p), x, s, t))

    def im(p, x, s):
        return jnp.imag(log_psi(unravel(p), x, s, t))

    g_re = jax.vmap(jax.grad(re), in_axes=(None, 0, 0))(flat, r, xi)
    g_im = jax.vmap(jax.grad(im), in_axes=(None, 0, 0))(flat, r, xi)
    return g_re + 1j * g_im
