"""Radial functions of the potential: Gaussian sums or tabulated splines.

Every potential module takes, for each operator channel, a radial function
object f with
    f(r2)   -> value at squared distance r2 (jax.numpy, any shape)
    f.r_max -> distance beyond which f is zero (sets the periodic image sum)
Two kinds:
- Gaussians(((V_1, kappa_1), ...)):  f(r) = sum_k V_k exp(-kappa_k r^2)  (test potentials)
- Table(r_grid, values): cubic spline on a uniform grid starting at r = 0,
  zero beyond the last grid point (realistic potentials, e.g. NV2).
A plain tuple of (V, kappa) pairs is accepted wherever a radial function is,
and converted with as_radial.

Tensor channels take w(r) = v_T(r) / r^2 (finite at r = 0 for physical tensor
forces), so that the code can use v_T r^_a r^_b = w r_a r_b without dividing
by r. For Gaussians this is the convention v_T(r) = sum_k V_k r^2 exp(-kappa_k r^2);
for tables use tensor_table(r_grid, v_T).
"""

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline


class Gaussians:
    """f(r) = sum_k V_k exp(-kappa_k r^2)."""

    def __init__(self, terms, tol=1e-10):
        self.terms = tuple((float(V), float(k)) for V, k in terms)
        # r_max: where the sum of |V_k| times the slowest Gaussian drops below tol.
        # A factor r^2 (tensor, L^2) or r (spin-orbit) lengthens the tail a little:
        # the extra 1e-3 keeps those below tol as well.
        vmax = sum(abs(V) for V, _ in self.terms)
        kmin = min(k for _, k in self.terms)
        self.r_max = float(np.sqrt(np.log(vmax / (tol * 1e-3)) / kmin))

    def __call__(self, r2):
        return sum(V * jnp.exp(-k * r2) for V, k in self.terms)


class Table:
    """Cubic spline through (r_grid, values) on a uniform grid with r_grid[0] = 0; zero beyond."""

    def __init__(self, r_grid, values):
        r = np.asarray(r_grid, dtype=float)
        v = np.asarray(values, dtype=float)
        dr = r[1] - r[0]
        if r[0] != 0.0 or not np.allclose(np.diff(r), dr, rtol=1e-10, atol=0):
            raise ValueError("Table needs a uniform grid starting at r = 0")
        spline = CubicSpline(r, v)
        self.dr = dr
        self.r_max = float(r[-1])
        self.n = len(r)
        self.coeffs = jnp.asarray(spline.c)  # (4, n-1): cubic coefficients per interval
        self.knots = jnp.asarray(r[:-1])

    def __call__(self, r2):
        r = jnp.sqrt(r2)
        k = jnp.clip(jnp.floor(r / self.dr).astype(jnp.int32), 0, self.n - 2)
        x = r - self.knots[k]
        c = self.coeffs[:, k]
        value = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
        return jnp.where(r < self.r_max, value, 0.0)


def tensor_table(r_grid, v_tensor):
    """Table of w(r) = v_T(r) / r^2 for a tensor channel.

    At r = 0 (where both vanish) the value is extrapolated quadratically from
    the next three grid points.
    """
    r = np.asarray(r_grid, dtype=float)
    w = np.zeros_like(r)
    w[1:] = np.asarray(v_tensor, dtype=float)[1:] / r[1:] ** 2
    w[0] = 3 * w[1] - 3 * w[2] + w[3]
    return Table(r, w)


def as_radial(f):
    """Accept a radial function object or a tuple of (V, kappa) Gaussian terms."""
    if f is None or (isinstance(f, (tuple, list)) and len(f) == 0):
        return None
    if isinstance(f, (tuple, list)):
        return Gaussians(f)
    return f


def image_vectors(L, functions):
    """Periodic image vectors L*n for all n in [-m, m]^3, enough to cover the longest range.

    Separations are minimum images (components within +-L/2), so every image
    with |n_i| > m is farther than (m + 1/2) L >= r_max.
    """
    r_max = max([f.r_max for f in functions if f is not None], default=0.0)
    m = int(np.ceil(r_max / L + 0.5))
    g = np.arange(-m, m + 1)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    return L * n.astype(float)
