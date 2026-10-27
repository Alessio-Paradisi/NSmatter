"""Spin-isospin dependent two-body potential: the v4 operator set.

    v_ij = v_c(r) + v_s(r) sigma_i.sigma_j + v_t(r) tau_i.tau_j
         + v_st(r) (sigma_i.sigma_j)(tau_i.tau_j)

The dot products are exchange operators:
    sigma_i.sigma_j = 2 P^sigma_ij - 1,    tau_i.tau_j = 2 P^tau_ij - 1,
where P^sigma_ij swaps the spin labels of i and j (P^tau the isospins).
Expanding the products, each pair contributes to the local energy
    a0 + a_s rho_s + a_t rho_t + a_st rho_st
with the wavefunction ratios
    rho_s  = Psi(spins of i, j swapped) / Psi
    rho_t  = Psi(isospins of i, j swapped) / Psi
    rho_st = Psi(both swapped) / Psi
and the coefficients
    a0 = v_c - v_s - v_t + v_st,   a_s = 2 (v_s - v_st),
    a_t = 2 (v_t - v_st),          a_st = 4 v_st.
Radial functions are sums of Gaussians, as in central.py, summed over
periodic images; the NV2 radial functions will replace them later.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..system import System
from .central import gaussian_sum, image_shifts

# Operator channels, in a fixed order: central, sigma.sigma, tau.tau, product.
OPERATORS = ("c", "s", "t", "st")


def swap(labels, i, j):
    """Copy of `labels` with entries i and j exchanged (JAX arrays are immutable)."""
    li, lj = labels[i], labels[j]
    return labels.at[i].set(lj).at[j].set(li)


def make_v4_potential(system: System, radial: dict, tol=1e-10):
    """Potential term for make_local_energy: term(log_psi, params, r, s, t).

    radial: {"c": terms, "s": terms, "t": terms, "st": terms}, each a tuple
    of Gaussians ((V_k, kappa_k), ...); a missing key means that channel is zero.
    """
    L = system.box_length

    # One set of image vectors, large enough for the longest-range Gaussian
    # of any channel.
    all_terms = tuple(term for op in OPERATORS for term in radial.get(op, ()))
    shifts = jnp.asarray(image_shifts(L, all_terms, tol))

    # All pairs i < j, computed once.
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    n_pairs = i.shape[0]

    def radial_values(r):
        """Image-summed radial function of every channel, each of shape (n_pairs,)."""
        d = minimum_image(r[i] - r[j], L)
        r2 = jnp.sum((d[:, None, :] + shifts[None]) ** 2, axis=-1)  # (n_pairs, n_images)
        out = {}
        for op in OPERATORS:
            if radial.get(op):
                out[op] = jnp.sum(gaussian_sum(r2, radial[op]), axis=1)
            else:
                out[op] = jnp.zeros(n_pairs)
        return out

    # Swap the labels of pair p, for all pairs at once: result (n_pairs, N).
    swap_all_pairs = jax.vmap(swap, in_axes=(None, 0, 0))

    def term(log_psi, params, r, s, t):
        v = radial_values(r)
        lp0 = log_psi(params, r, s, t)

        # Label sets for every pair: swapped and unswapped copies.
        s_sw = swap_all_pairs(s, i, j)
        t_sw = swap_all_pairs(t, i, j)
        s_same = jnp.broadcast_to(s, (n_pairs,) + s.shape)
        t_same = jnp.broadcast_to(t, (n_pairs,) + t.shape)

        # Evaluate Psi at all swapped configurations in one batched call.
        many = jax.vmap(log_psi, in_axes=(None, None, 0, 0))
        rho_s = jnp.exp(many(params, r, s_sw, t_same) - lp0)
        rho_t = jnp.exp(many(params, r, s_same, t_sw) - lp0)
        rho_st = jnp.exp(many(params, r, s_sw, t_sw) - lp0)
        # Note: if the two labels are equal, the swap changes nothing and the
        # ratio is exactly 1 (computed anyway here; skipping it is an
        # optimization for later). If the swap leads to Psi = 0 (e.g. a
        # wrong composition), exp(-inf) = 0 and the term drops out.

        # Coefficients from sigma.sigma = 2 P - 1 and tau.tau = 2 P - 1.
        a0 = v["c"] - v["s"] - v["t"] + v["st"]
        a_s = 2.0 * (v["s"] - v["st"])
        a_t = 2.0 * (v["t"] - v["st"])
        a_st = 4.0 * v["st"]
        e = a0 + a_s * rho_s + a_t * rho_t + a_st * rho_st
        return jnp.sum(e)

    return term
