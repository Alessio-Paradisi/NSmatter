"""Structure observables: isospin-resolved pair distributions and spin-isospin correlations.

For every walker configuration, pairs are binned by their minimum-image distance
r_ij < L/2 (bins of width dr), separately for nn, np and pp pairs:
    counts[alpha, k]  number of alpha pairs in bin k
    sigma[alpha, k]   sum over those pairs of the local estimator of sigma_i . sigma_j
    tau[alpha, k]     sum over those pairs of the local estimator of tau_i . tau_j
    sigma_total, tau_total: the same summed over all pairs (any distance)
The local estimators are computed like the potential energy,
    O_L = sum_targets <current|O|target> Psi(target) / Psi
(pair_operators.pair_targets), so they include the spin-isospin exchange
correlations of the wavefunction. Their averages give:

    pair distribution (Fore et al., Commun. Phys. 8, 108 (2025), Eq. 4)
        g_alpha(r) = < sum_{i<j} delta(r - r_ij) P_alpha > / (2 pi r^2 rho N)
                   = counts[alpha, k] / (2 pi r_k^2 dr rho N)
      (for uncorrelated particles g_nn + g_np + g_pp -> (N - 1) / N at large r)
    spin correlation   C_sigma,alpha(r) = sigma[alpha, k] / counts[alpha, k]
                       (+1: aligned spins, S = 1; -3: spin singlet; 0: uncorrelated)
    isospin correlation C_tau,alpha(r)  = tau[alpha, k] / counts[alpha, k]
                       (nn, pp: always 1; np: -3 for T = 0, +1 for T = 1)
normalize() turns accumulated sums into these functions.
"""

import jax.numpy as jnp
import numpy as np

from .geometry import minimum_image
from .hamiltonian.pair_operators import DOT, PAULI, has_both_species, pair_targets
from .system import System

CLASSES = ("nn", "np", "pp")


def make_pair_observables(system: System, n_bins: int = 40):
    """Per-walker function (log_psi, params, r, s, t) -> dict of counts, sigma, tau, each (3, n_bins)."""
    L = system.box_length
    dr = 0.5 * L / n_bins
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    needs_swap = has_both_species(system)

    def observe(log_psi, params, r, s, t):
        d = minimum_image(r[i] - r[j], L)
        dist = jnp.sqrt(jnp.sum(d**2, axis=-1))
        k = jnp.floor(dist / dr).astype(jnp.int32)
        in_range = dist < 0.5 * L
        cls = jnp.where(t[i] == t[j], jnp.where(t[i] < 0, 0, 2), 1)  # 0 nn, 1 np, 2 pp
        onehot = ((cls[:, None, None] == jnp.arange(3)[None, :, None])
                  & (k[:, None, None] == jnp.arange(n_bins)[None, None, :])
                  & in_range[:, None, None]).astype(jnp.float64)  # (n_pairs, 3, n_bins)

        T = pair_targets(log_psi, params, r, s, t, i, j, needs_swap)
        # sigma.sigma: spin targets, isospins unchanged
        sig = jnp.einsum("pc,cd,ptd->pt", T["bra"], DOT, T["kets"])
        sigma_local = jnp.sum(sig * T["rho_same"], axis=1).real
        # tau.tau: spin identity (only the unchanged spin target), isospin same + swapped
        one = jnp.einsum("pc,ptc->pt", T["bra"], T["kets"])
        tt_same = jnp.where(t[i] == t[j], 1.0, -1.0)
        tt_swap = jnp.where(t[i] == t[j], 0.0, 2.0)
        tau_local = jnp.sum(one * (tt_same[:, None] * T["rho_same"] + tt_swap[:, None] * T["rho_swap"]), axis=1).real

        return {
            "counts": jnp.sum(onehot, axis=0),
            "sigma": jnp.einsum("pck,p->ck", onehot, sigma_local),
            "tau": jnp.einsum("pck,p->ck", onehot, tau_local),
            # all pairs, also beyond L/2: sum_{i<j} sigma_i.sigma_j = 2 S^2 - 3N/2 (same for tau)
            "sigma_total": jnp.sum(sigma_local),
            "tau_total": jnp.sum(tau_local),
        }

    observe.bins = (np.arange(n_bins) + 0.5) * dr  # bin centres, fm
    observe.dr = dr
    return observe


def normalize(sums, n_samples, system: System, dr):
    """g_alpha(r), C_sigma(r), C_tau(r) from sums over n_samples walker configurations.

    sums: dict of arrays (..., 3, n_bins) (counts, sigma, tau). Returns dict with
    g, C_sigma, C_tau of the same shape (C is NaN where a bin has no pairs).
    """
    counts, sigma, tau = (np.asarray(sums[k], dtype=float) for k in ("counts", "sigma", "tau"))
    n_bins = counts.shape[-1]
    r = (np.arange(n_bins) + 0.5) * dr
    shell = 2 * np.pi * r**2 * dr * system.density * system.n_particles
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "r": r,
            "g": counts / n_samples / shell,
            "C_sigma": np.where(counts > 0, sigma / counts, np.nan),
            "C_tau": np.where(counts > 0, tau / counts, np.nan),
        }


def spin_vectors(xi):
    """Spin direction <sigma> of each sampled spinor, shape (..., N, 3), unit length.

    A spin coordinate xi enters Psi as a bra, so the spin state is the ket conj(xi):
    n_a = conj(xi)^dagger sigma_a conj(xi) = xi^T sigma_a conj(xi).
    """
    xi = jnp.asarray(xi)
    return jnp.einsum("...a,kab,...b->...k", xi, PAULI, jnp.conj(xi)).real
