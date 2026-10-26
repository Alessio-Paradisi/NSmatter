"""Stochastic reconfiguration (SR): natural-gradient optimization of Psi.

Everything is built from the log-derivatives
    O_k(R) = d ln Psi(R) / d p_k,
one value per walker and per (flattened) parameter. With averages <.> over
the walkers (samples of |Psi|^2):
    grad_k = 2 Re < (O_k - <O_k>)^* (E_L - <E_L>) >    energy gradient
    S_kl   =   Re < (O_k - <O_k>)^* (O_l - <O_l>) >    metric (covariance of O)
    p  ->  p - lr * (S + diag_shift * 1)^-1 grad
lr plays the role of an imaginary-time step (units MeV^-1, since E_L is the
total energy): it must shrink when N, and with it the spread of E_L, grows.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def log_derivatives(log_psi, params, r, s, t):
    """O for a batch of walkers r (W, N, 3). Returns complex array (W, P).

    The parameters (any nested structure of arrays) are flattened into one
    vector of length P with ravel_pytree; `unravel` rebuilds the structure.
    """
    flat, unravel = ravel_pytree(params)

    # jax.grad needs a real scalar, and ln Psi is complex: differentiate the
    # real and imaginary parts separately, then recombine. (Re ln Psi changes
    # |Psi|; Im ln Psi changes the phase.)
    def re(p, x):
        return jnp.real(log_psi(unravel(p), x, s, t))

    def im(p, x):
        return jnp.imag(log_psi(unravel(p), x, s, t))

    # One gradient per walker: map over positions, share the parameters.
    g_re = jax.vmap(jax.grad(re), in_axes=(None, 0))(flat, r)
    g_im = jax.vmap(jax.grad(im), in_axes=(None, 0))(flat, r)
    return g_re + 1j * g_im


def sr_direction(O, e_loc, diag_shift=0.05):
    """SR step direction from log-derivatives O (W, P) and local energies (W,).

    Returns (delta, grad, S): the parameters should move by -lr * delta.
    """
    W = O.shape[0]
    # Fluctuations around the walker average.
    dO = O - O.mean(axis=0)
    dE = e_loc - e_loc.mean()
    # Energy gradient: covariance of O with E_L. Vanishes if E_L is constant.
    grad = 2.0 * jnp.real(jnp.conj(dO).T @ dE) / W
    # Metric: covariance matrix of the log-derivatives (symmetric, >= 0).
    S = jnp.real(jnp.conj(dO).T @ dO) / W
    # S is noisy and can be singular (parameters that do not change Psi):
    # the diagonal shift damps directions the samples cannot resolve.
    S_reg = S + diag_shift * jnp.eye(S.shape[0])
    # Solve (S + shift) delta = grad instead of inverting S explicitly.
    delta = jnp.linalg.solve(S_reg, grad)
    return delta, grad, S


def apply_update(params, delta, lr):
    """params - lr * delta, with delta a flat vector in ravel_pytree order."""
    flat, unravel = ravel_pytree(params)
    return unravel(flat - lr * delta)
