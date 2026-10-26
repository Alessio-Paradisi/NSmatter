"""Kinetic part of the local energy.

    E_kin,L(R) = -hbar^2/2m * sum_i [ lap_i ln Psi + (grad_i ln Psi)^2 ]

ln Psi is complex and depends on real coordinates. Forward-mode
differentiation (jax.jvp) handles complex outputs directly: for each of the
3N coordinate directions e, a jvp of a jvp gives the first and second
derivative along e. Cost: 3N forward passes, run in parallel with vmap.
A faster forward Laplacian can replace this later with the same interface.
"""

import jax
import jax.numpy as jnp

from ..constants import HBAR2_2M


def grad_and_laplacian(f, x):
    """Gradient and Laplacian of a scalar function f at the flat point x.

    f: function from a 1D real array (length D) to a (possibly complex) scalar.
    Returns (gradient, shape (D,); Laplacian, scalar).
    """

    def along(e):
        # Directional derivative of f along e, as a function of the point y.
        def first(y):
            return jax.jvp(f, (y,), (e,))[1]

        # jvp of `first` along the same e: d1 = df/de, d2 = d^2f/de^2.
        d1, d2 = jax.jvp(first, (x,), (e,))
        return d1, d2

    # Unit vectors along every coordinate direction, one per row.
    basis = jnp.eye(x.size, dtype=x.dtype)
    # Evaluate all directions at once: d1 = gradient components,
    # d2 = diagonal of the Hessian, whose sum is the Laplacian.
    d1, d2 = jax.vmap(along)(basis)
    return d1, d2.sum()


def kinetic_local_energy(log_psi, params, r, s, t, hbar2_2m: float = HBAR2_2M):
    """Kinetic local energy (MeV) for one configuration.

    Returns a complex number; its real part enters the energy, and its
    imaginary part averages to zero over |Psi|^2.
    """
    shape = r.shape  # (N, 3)

    # Flatten positions to one vector of 3N coordinates, as grad_and_laplacian expects.
    def f(x):
        return log_psi(params, x.reshape(shape), s, t)

    grad, lap = grad_and_laplacian(f, r.reshape(-1))
    # (grad ln Psi)^2 is a plain square, NOT |.|^2: for a plane wave
    # grad ln Psi = i k and the square gives -|k|^2, the free kinetic energy.
    return -hbar2_2m * (lap + jnp.sum(grad**2))
