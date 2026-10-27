"""Operator-correlated wavefunction.

    Psi = exp(J) * [ 1 + sum_{i<j} U_ij ] * Phi
    U_ij = u_s sigma.sigma + u_t tau.tau + u_st (sigma.sigma)(tau.tau)
         + u_tn S12 + u_tnt S12 tau.tau

exp(J): central Jastrow (jastrow.py); Phi: spin-isospin Slater determinant.
U_ij acts on Phi like the potential acts on Psi in the local energy:
    (U_ij Phi)(X) = sum_targets <current|U_ij|target> Phi(target),
targets = 4 basis spin states of the pair x 2 isospin states.
A target changes only columns i and j of the Slater matrix M, so
    Phi(target) / Phi(X) = det [[G_i . c_i', G_i . c_j'], [G_j . c_i', G_j . c_j']]
with G = M^-1 (rows i, j) and c_i', c_j' the new columns: one inverse for
all pairs and targets.

Radial functions (minimum image, cutoff r_c <= L/2, as the Jastrow):
    u_p(r)  = (1 - x^2)^3 sum_m a_{p,m} x^(2m)          for s, t, st
    u_p(r)  = x^2 (1 - x^2)^3 sum_m a_{p,m} x^(2m)      for tn, tnt
The extra x^2 makes u_T S12 smooth at r = 0; the code uses
u_T r^_a r^_b = w r_a r_b / r_c^2 with w = (1 - x^2)^3 sum_m a x^(2m).
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image, occupied_kvectors
from ..hamiltonian.pair_operators import DOT, SIGMA_SIGMA, pair_index
from ..spin import as_spinors
from ..system import System
from .jastrow import N_CHANNELS, jastrow_basis, pair_channel

OPERATORS = ("s", "t", "st", "tn", "tnt")


def make_correlated(system: System, n_terms: int = 3, cutoff: float = None):
    """Build exp(J) [1 + sum U_ij] Phi. Returns (params, log_psi).

    params = {"jastrow": (2, n_terms), "corr": (5, n_terms)}, all zero at start
    (then Psi is exactly the Jastrow-Slater state).
    """
    L = system.box_length
    rc = 0.5 * L if cutoff is None else cutoff
    if rc > 0.5 * L + 1e-12:
        raise ValueError("cutoff must not exceed L/2")

    # Orbitals: momentum, spin index (0 up, 1 down) and isospin of each.
    ks, so, to = [], [], []
    for count, sz, tz in zip(system.counts, (0, 1, 0, 1), (-1, -1, 1, 1)):
        if count > 0:
            ks.append(occupied_kvectors(count, L))
            so += [sz] * count
            to += [tz] * count
    k = jnp.asarray(np.concatenate(ks))
    spin_orb = jnp.asarray(so)
    iso_orb = jnp.asarray(to)

    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    n_pairs = i.shape[0]

    params = {
        "jastrow": jnp.zeros((N_CHANNELS, n_terms)),
        "corr": jnp.zeros((len(OPERATORS), n_terms)),
    }
    eye4 = jnp.eye(4, dtype=jnp.complex128)

    def operator_matrices(a, d):
        """Spin matrices of U for every pair: A (isospin identity), B (times tau.tau)."""
        r2 = jnp.sum(d**2, axis=-1)
        phi = jastrow_basis(r2, rc, n_terms)  # (n_pairs, n_terms), zero beyond rc
        u = phi @ a.T  # (n_pairs, 5): u_s, u_t, u_st, w_tn, w_tnt

        def tensor(w):
            # u_T S12 = w [3 r r^T . (sigma x sigma) - r^2 DOT] / rc^2
            rr = 3.0 * jnp.einsum("p,pa,pb->pab", w, d, d)
            return (jnp.einsum("pab,abij->pij", rr, SIGMA_SIGMA) - (w * r2)[:, None, None] * DOT) / rc**2

        A = u[:, 0, None, None] * DOT + tensor(u[:, 3])
        B = u[:, 1, None, None] * eye4 + u[:, 2, None, None] * DOT + tensor(u[:, 4])
        return A, B

    def log_psi(params, r, s, t):
        xi = as_spinors(s)
        # --- Slater matrix and its inverse -------------------------------------
        plane = jnp.exp(1j * (k @ r.T))  # (orbitals, particles)
        spin = xi[:, spin_orb].T
        iso = iso_orb[:, None] == t[None, :]
        M = jnp.where(iso, plane * spin, 0.0)
        sign, logabs = jnp.linalg.slogdet(M)
        G = jnp.linalg.inv(M)  # (particles, orbitals)

        # --- all possible new columns, contracted with every row of G ---------
        # Column of particle q with basis spin b and isospin c:
        #   C[a] = plane[a, q] * [spin_orb[a] == b] * [iso_orb[a] == c]
        # D[p, q, b, c] = sum_a G[p, a] C[a]   (all at once, O(N^3))
        spin_sel = jnp.stack([spin_orb == 0, spin_orb == 1], axis=-1)  # (orb, 2)
        iso_sel = jnp.stack([iso_orb == 1, iso_orb == -1], axis=-1)  # (orb, 2): c=0 proton, 1 neutron
        sel = (spin_sel[:, :, None] & iso_sel[:, None, :]).astype(jnp.complex128)  # (orb, 2, 2)
        D = jnp.einsum("pa,aq,abc->pqbc", G, plane, sel)

        # --- determinant ratios for every pair, spin target, isospin choice ----
        # Target spin states |b_i b_j> in basis order, isospin index c = (1 - t)/2.
        bi = jnp.array([0, 0, 1, 1])
        bj = jnp.array([0, 1, 0, 1])
        ci = (1 - t[i]) // 2
        cj = (1 - t[j]) // 2

        def ratios(ci_new, cj_new):
            # 2x2 determinant for each pair (n_pairs) and spin target (4).
            d_ii = D[i, i][:, bi, :][jnp.arange(n_pairs), :, ci_new]  # (n_pairs, 4)
            d_ij = D[i, j][:, bj, :][jnp.arange(n_pairs), :, cj_new]
            d_ji = D[j, i][:, bi, :][jnp.arange(n_pairs), :, ci_new]
            d_jj = D[j, j][:, bj, :][jnp.arange(n_pairs), :, cj_new]
            return d_ii * d_jj - d_ij * d_ji

        rho_same = ratios(ci, cj)  # isospins unchanged
        rho_swap = ratios(cj, ci)  # isospins of i and j exchanged

        # --- amplitudes <current|U_ij|target> ----------------------------------
        d = minimum_image(r[i] - r[j], L)
        A, B = operator_matrices(params["corr"], d)
        bra = jnp.einsum("pa,pb->pab", xi[i], xi[j]).reshape(n_pairs, 4)
        A_row = jnp.einsum("pc,pct->pt", bra, A)
        B_row = jnp.einsum("pc,pct->pt", bra, B)
        b_cur = pair_index(t[i], t[j])
        b_sw = pair_index(t[j], t[i])
        differ = (t[i] != t[j]).astype(jnp.float64)
        amp_same = A_row + B_row * DOT[b_cur, b_cur][:, None]
        amp_swap = B_row * (DOT[b_cur, b_sw] * differ)[:, None]

        correlation = 1.0 + jnp.sum(amp_same * rho_same + amp_swap * rho_swap)

        # --- central Jastrow J (same as jastrow.py) ----------------------------
        phi = jastrow_basis(jnp.sum(d**2, axis=-1), rc, n_terms)
        J = jnp.sum(phi * params["jastrow"][pair_channel(t, i, j)])

        return J + logabs + jnp.log(sign) + jnp.log(correlation)

    return params, log_psi
