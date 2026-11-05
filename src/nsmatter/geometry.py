"""Periodic cubic box: wrapping, minimum image and plane-wave shells."""

import jax.numpy as jnp
import numpy as np


def wrap(r, L):
    """Map positions back into [0, L)."""
    return jnp.mod(r, L)


def minimum_image(d, L):
    """Shortest periodic image of a separation vector, components in [-L/2, L/2)."""
    return d - L * jnp.round(d / L)


def integer_vectors(n2_max: int) -> np.ndarray:
    """All integer vectors n with |n|^2 <= n2_max, sorted by |n|^2 then lexicographically."""
    m = int(np.ceil(np.sqrt(n2_max)))
    grid = np.arange(-m, m + 1)
    n = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    n2 = (n**2).sum(axis=1)
    n = n[n2 <= n2_max]
    n2 = (n**2).sum(axis=1)
    order = np.lexsort((n[:, 2], n[:, 1], n[:, 0], n2))
    return n[order]


def closed_shells(n2_max: int = 20) -> list:
    """Particle numbers that fill complete momentum shells for one species.

    Returns [1, 7, 19, 27, 33, 57, 81, 93, ...].
    """
    n = integer_vectors(n2_max)
    n2 = (n**2).sum(axis=1)
    return [int((n2 <= v).sum()) for v in np.unique(n2)]


def occupied_kvectors(count: int, L: float) -> np.ndarray:
    """The `count` lowest plane-wave momenta k = 2 pi n / L, shape (count, 3).

    `count` must be a closed-shell number (or 0), otherwise the state is not
    unique and not translation/parity symmetric.
    """
    if count == 0:
        return np.zeros((0, 3))
    shells = closed_shells(n2_max=50)
    if count not in shells:
        raise ValueError(f"{count} is not a closed shell; allowed: {shells[:10]} ...")
    n = integer_vectors(50)[:count]
    return 2.0 * np.pi / L * n.astype(float)
