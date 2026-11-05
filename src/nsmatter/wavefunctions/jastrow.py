"""Jastrow factor times the plane-wave Slater determinant.

    Psi(R) = exp( J(R) ) * Slater(R),   J(R) = sum_{i<j} u_c(r_ij)

c = channel of the pair, from the labels:
    0: same spin, same isospin      1: opposite spin, same isospin
    2: same spin, opposite isospin  3: opposite spin, opposite isospin
u_c(r) = (1 - x^2)^3 * sum_m a_{c,m} x^(2m)   for x = r / r_c < 1, else 0.
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..system import System
from .slater import make_slater

N_CHANNELS = 2


def jastrow_basis(r2, rc, n_terms):
    x2 = r2 / rc**2
    powers = x2[..., None] ** jnp.arange(n_terms)
    envelope = jnp.where(x2 < 1.0, (1.0 - x2) ** 3, 0.0)
    return envelope[..., None] * powers


def pair_channel(t, i, j):
    return (1 - t[i] * t[j]) // 2


def make_jastrow_slater(system: System, n_terms: int = 4, cutoff: float = None):
    L = system.box_length
    rc = 0.5 * L if cutoff is None else cutoff
    if rc > 0.5 * L + 1e-12:
        raise ValueError("cutoff must not exceed L/2")
    slater_params, slater_log_psi = make_slater(system)
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)

    params = {"jastrow": jnp.zeros((N_CHANNELS, n_terms))}

    def jastrow(a, r, s, t):
        d = minimum_image(r[i] - r[j], L)
        r2 = jnp.sum(d**2, axis=-1)
        phi = jastrow_basis(r2, rc, n_terms)
        coeff = a[pair_channel(t, i, j)]
        return jnp.sum(phi * coeff)

    def log_psi(params, r, s, t):
        return jastrow(params["jastrow"], r, s, t) + slater_log_psi(slater_params, r, s, t)

    return params, log_psi
