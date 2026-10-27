"""Spin coordinates: discrete labels or continuous spinors.

A spin coordinate is a complex 2-vector xi = (xi_up, xi_down) with
|xi_up|^2 + |xi_down|^2 = 1; Psi(R, S) is linear in each xi_j:
    Psi(..., xi_j, ...) = xi_up Psi(..., up, ...) + xi_down Psi(..., down, ...).
The discrete labels are the special cases up = (1, 0), down = (0, 1).
Sampling the spinors uniformly on the sphere (with weight |Psi|^2) gives
correct expectation values, because the spinor states resolve the identity.
"""

import jax.numpy as jnp


def as_spinors(s):
    """Spin coordinates as an (N, 2) complex array.

    Accepts discrete labels, shape (N,), with +1 = up and -1 = down,
    or spinors, shape (N, 2), which are returned unchanged. (The branch
    depends only on the shape, so it is fixed at compile time under jit.)
    """
    s = jnp.asarray(s)
    if s.ndim == 1:
        return jnp.stack([(1 + s) / 2, (1 - s) / 2], axis=-1).astype(jnp.complex128)
    return s.astype(jnp.complex128)


# Basis spinors: up = (1, 0), down = (0, 1).
UP = jnp.array([1.0, 0.0], dtype=jnp.complex128)
DOWN = jnp.array([0.0, 1.0], dtype=jnp.complex128)
# The 4 two-body basis states |a b> in index order 2*idx(a) + idx(b):
# |up up>, |up down>, |down up>, |down down>.
PAIR_BASIS = jnp.array([[UP, UP], [UP, DOWN], [DOWN, UP], [DOWN, DOWN]])  # (4, 2, 2)
