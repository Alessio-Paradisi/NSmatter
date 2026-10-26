"""Jastrow factor times the plane-wave Slater determinant.

    Psi(R) = exp( J(R) ) * Slater(R),    J(R) = sum_{i<j} u_c(r_ij)

r_ij is the minimum-image distance and c the channel of the pair:
    0: same spin, same isospin          1: opposite spin, same isospin
    2: same spin, opposite isospin      3: opposite spin, opposite isospin

Each channel has its own correlation function
    u_c(r) = (1 - x^2)^3 * sum_m a_{c,m} x^(2m),   x = r / r_c < 1,
    u_c(r) = 0                                      for r >= r_c.
- the envelope (1 - x^2)^3 makes u, u', u'' vanish at the cutoff, so ln Psi
  has no kink where pairs cross r_c;
- only even powers of r: u'(0) = 0 (no cusp), and the code only needs r^2,
  whose derivative is finite even at r = 0;
- J is linear in the parameters a_{c,m}.
J is symmetric under exchange of particles, so Psi stays antisymmetric.
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..system import System
from .slater import make_slater

# Number of pair channels (same/opposite spin x same/opposite isospin).
N_CHANNELS = 4


def jastrow_basis(r2, rc, n_terms):
    """Basis functions phi_m(r) = (1 - x^2)^3 x^(2m) for m = 0..n_terms-1.

    r2: squared distances, any shape. Returns shape r2.shape + (n_terms,).
    Zero for r >= rc.
    """
    x2 = r2 / rc**2
    # x^(2m) = (x^2)^m for every m; the new last axis runs over m.
    powers = x2[..., None] ** jnp.arange(n_terms)
    # Envelope inside the cutoff, exactly zero outside.
    envelope = jnp.where(x2 < 1.0, (1.0 - x2) ** 3, 0.0)
    return envelope[..., None] * powers


def pair_channel(s, t, i, j):
    """Channel (0..3) of the pairs (i, j), from the spin and isospin labels.

    s_i s_j = +1 (same) or -1 (opposite), so (1 - s_i s_j) / 2 is 0 or 1.
    Channel = opposite_spin + 2 * opposite_isospin.
    Computed from the labels (not from the species order), so it keeps
    working when spins are sampled.
    """
    opposite_spin = (1 - s[i] * s[j]) // 2
    opposite_isospin = (1 - t[i] * t[j]) // 2
    return opposite_spin + 2 * opposite_isospin


def make_jastrow_slater(system: System, n_terms: int = 4, cutoff: float = None):
    """Build exp(J) * Slater for `system`.

    n_terms: number of basis functions per channel.
    cutoff: r_c in fm; default L/2, the largest allowed by the minimum image.
    Returns (params, log_psi) with params = {"jastrow": zeros (N_CHANNELS, n_terms)}.
    """
    L = system.box_length
    rc = 0.5 * L if cutoff is None else cutoff
    # Beyond L/2 a particle would see two images of the same partner.
    if rc > 0.5 * L + 1e-12:
        raise ValueError("cutoff must not exceed L/2")

    # The Slater part is reused unchanged; it has no parameters.
    slater_params, slater_log_psi = make_slater(system)

    # All pairs i < j as two index arrays, computed once (setup: NumPy).
    # For N particles there are N(N-1)/2 pairs.
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)

    # Start from zero: J = 0, Psi = Slater (the free gas).
    params = {"jastrow": jnp.zeros((N_CHANNELS, n_terms))}

    def jastrow(a, r, s, t):
        """J(R) for coefficients a, shape (N_CHANNELS, n_terms)."""
        # Minimum-image separations and squared distances of all pairs.
        d = minimum_image(r[i] - r[j], L)
        r2 = jnp.sum(d**2, axis=-1)
        # Basis functions for every pair: shape (n_pairs, n_terms).
        phi = jastrow_basis(r2, rc, n_terms)
        # Coefficients of each pair's channel: shape (n_pairs, n_terms).
        coeff = a[pair_channel(s, t, i, j)]
        # Sum over pairs and basis functions.
        return jnp.sum(phi * coeff)

    def log_psi(params, r, s, t):
        # ln Psi = J + ln Slater (J is real, the Slater part carries the phase).
        return jastrow(params["jastrow"], r, s, t) + slater_log_psi(slater_params, r, s, t)

    return params, log_psi
