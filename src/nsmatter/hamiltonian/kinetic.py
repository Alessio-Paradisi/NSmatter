"""Kinetic part of the local energy.

    E_kin(R) = -hbar^2/2m * sum_i [ lap_i ln Psi + (grad_i ln Psi)^2 ]

ln Psi is complex and depends on real coordinates, so the derivatives are
taken in forward mode (jvp), which handles complex outputs directly. For
each Cartesian direction e of the 3N coordinates we take a jvp of a jvp:
the inner one gives the first derivative, the outer one the second.
Cost: 3N forward passes, vectorized with vmap. A faster forward-Laplacian
(e.g. the folx library) can replace this later without changing the interface.
"""

import jax
import jax.numpy as jnp

from ..constants import HBAR2_2M


def grad_and_laplacian(f, x):
    """Gradient (vector) and Laplacian (scalar) of a scalar function f at flat x."""

    def along(e):
        def first(y):
            return jax.jvp(f, (y,), (e,))[1]

        d1, d2 = jax.jvp(first, (x,), (e,))
        return d1, d2

    basis = jnp.eye(x.size, dtype=x.dtype)
    d1, d2 = jax.vmap(along)(basis)
    return d1, d2.sum()


def kinetic_local_energy(log_psi, params, r, s, t, hbar2_2m: float = HBAR2_2M):
    """Kinetic local energy (complex; its real part enters the energy)."""
    shape = r.shape

    def f(x):
        return log_psi(params, x.reshape(shape), s, t)

    grad, lap = grad_and_laplacian(f, r.reshape(-1))
    return -hbar2_2m * (lap + jnp.sum(grad**2))
