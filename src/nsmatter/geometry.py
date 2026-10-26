"""Periodic cubic box: wrapping, minimum image and plane-wave momentum shells.

Two kinds of functions live here:
- wrap and minimum_image run inside the simulation loop, so they use
  jax.numpy (compiled, vectorized, differentiable);
- the shell functions run once at setup, so they use plain NumPy.
"""

import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Periodic boundary conditions (used inside the simulation: jax.numpy)
# -----------------------------------------------------------------------------
def wrap(r, L):
    """Bring positions back into the box [0, L) along each axis.

    r: array of positions, any shape ending in 3 (e.g. (N, 3)).
    Moving a particle by a whole multiple of L changes nothing physically,
    so this only keeps the numbers bounded.
    """
    return jnp.mod(r, L)


def minimum_image(d, L):
    """Shortest periodic copy of a separation vector d = r_i - r_j.

    Each component is shifted by the nearest whole multiple of L, so the
    result has components in [-L/2, L/2]. This is the separation the
    pair potential and the correlations must see.
    """
    return d - L * jnp.round(d / L)


# -----------------------------------------------------------------------------
# Plane-wave momenta k = (2 pi / L) n (setup only: NumPy)
# -----------------------------------------------------------------------------
def integer_vectors(n2_max: int) -> np.ndarray:
    """All integer vectors n with |n|^2 <= n2_max, shape (M, 3).

    Sorted by |n|^2 (lowest kinetic energy first); vectors with the same
    |n|^2 are ordered lexicographically so the order is always the same.
    """
    # Every component satisfies |n_i| <= sqrt(n2_max): build that cube.
    m = int(np.ceil(np.sqrt(n2_max)))
    grid = np.arange(-m, m + 1)
    # All (nx, ny, nz) combinations in the cube, as rows of a (M, 3) array.
    n = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    # Keep only those inside the sphere |n|^2 <= n2_max.
    n = n[(n**2).sum(axis=1) <= n2_max]
    n2 = (n**2).sum(axis=1)
    # np.lexsort sorts by the LAST key first: |n|^2, then nx, ny, nz.
    order = np.lexsort((n[:, 2], n[:, 1], n[:, 0], n2))
    return n[order]


def closed_shells(n2_max: int = 20) -> list:
    """Particle numbers that fill complete momentum shells for one species.

    A shell is the set of n with the same |n|^2 (degenerate in energy).
    Returns the cumulative counts [1, 7, 19, 27, 33, 57, 81, 93, ...].
    """
    n2 = (integer_vectors(n2_max) ** 2).sum(axis=1)
    # For each |n|^2 that actually occurs, count the vectors up to it.
    # (|n|^2 = 7 never occurs: 7 is not a sum of three squares.)
    return [int((n2 <= v).sum()) for v in np.unique(n2)]


def occupied_kvectors(count: int, L: float) -> np.ndarray:
    """The `count` lowest plane-wave momenta for one species, shape (count, 3), in fm^-1.

    `count` must be a closed-shell number (or 0): an open shell would leave
    an arbitrary choice of which degenerate states to fill and would break
    the parity and cubic symmetry of the free ground state.
    """
    # An empty species (e.g. no protons) has no occupied states.
    if count == 0:
        return np.zeros((0, 3))
    # n2_max = 50 covers several hundred particles per species: plenty.
    shells = closed_shells(n2_max=50)
    if count not in shells:
        raise ValueError(f"{count} is not a closed shell; allowed: {shells[:10]} ...")
    # Take the first `count` vectors in energy order and convert n -> k.
    n = integer_vectors(50)[:count]
    return 2.0 * np.pi / L * n.astype(float)
