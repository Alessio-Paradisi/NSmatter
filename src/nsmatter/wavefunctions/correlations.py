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

Spin-orbit correlations (spin_orbit=True): two more operators,
    U_ij += u_ls(r) L_ij.S_ij + u_lst(r) L_ij.S_ij tau_i.tau_j,
L = r x p (r = r_i - r_j, p = -i d/dr), S = (sigma_i + sigma_j)/2, acting on Phi:
    (L_a Phi)(target) = -i (r x d_r)_a Phi(target),   d_r = (grad_i - grad_j)/2.
Moving r_i by +u and r_j by -u changes only columns i and j of the Slater matrix,
whose derivatives are analytic for plane waves (d_u e^{ik.r} = i k e^{ik.r}), so
    d_u [Phi(target)/Phi] = Dk_ii R_jj - R_ii Dk_jj + Dk_ij R_ji - R_ij Dk_ji
with Dk[p, q] = G_p . (i k) c_q' and R the 2 x 2 matrix of ratios above.

Radial functions (minimum image, cutoff r_c <= L/2, as the Jastrow):
    u_p(r)  = (1 - x^2)^3 sum_m a_{p,m} x^(2m)          for s, t, st
    u_p(r)  = x^2 (1 - x^2)^3 sum_m a_{p,m} x^(2m)      for tn, tnt
    u_p(r)  = (1 - x^2)^3 sum_m a_{p,m} x^(2m)          for ls, lst (L is already O(r))
The extra x^2 makes u_T S12 smooth at r = 0; the code uses
u_T r^_a r^_b = w r_a r_b / r_c^2 with w = (1 - x^2)^3 sum_m a x^(2m).

Pair moves (log_psi.pair_move = (prepare, move)), used by the momentum operators:
the configuration X' with particles (p, q) moved to r_p + u, r_q - u and given
new spinors and isospins, relative to the current X. Only columns p, q of M
change, M' = M + U V^T (rank 2), so with R = G[p,q] C'[:, p,q] (2 x 2):
    Phi' / Phi = det R,   G' = G - (G U) R^-1 G[p,q]
and all pair ratios of X' follow from D' = G' C' = E - (G U) R^-1 E[p,q], E = G C'.
Only the 2N - 3 amplitudes of pairs containing p or q are recomputed.
Cost O(N^2) instead of O(N^3) per evaluation (and per derivative direction).
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image, occupied_kvectors
from ..hamiltonian.pair_operators import DOT, PAULI, SIGMA_SIGMA, pair_index
from ..spin import as_spinors
from ..system import System
from .jastrow import N_CHANNELS, jastrow_basis, pair_channel

OPERATORS = ("s", "t", "st", "tn", "tnt")
OPERATORS_SO = OPERATORS + ("ls", "lst")
# Total pair spin S_a = (sigma_a (x) 1 + 1 (x) sigma_a) / 2 on |s_i s_j>, shape (3, 4, 4).
_EYE2 = jnp.eye(2, dtype=jnp.complex128)
S_PAIR = jnp.array([(jnp.kron(PAULI[a], _EYE2) + jnp.kron(_EYE2, PAULI[a])) / 2 for a in range(3)])


def make_correlated(system: System, n_terms: int = 3, cutoff: float = None, spin_orbit: bool = False):
    """Build exp(J) [1 + sum U_ij] Phi. Returns (params, log_psi).

    params = {"jastrow": (2, n_terms), "corr": (5, n_terms)}, or (7, n_terms) with
    spin_orbit=True (rows: s, t, st, tn, tnt[, ls, lst]); all zero at start
    (then Psi is exactly the Jastrow-Slater state).
    """
    L = system.box_length
    N = system.n_particles
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
    spin_sel = jnp.stack([spin_orb == 0, spin_orb == 1], axis=-1)  # (orb, 2)
    iso_sel = jnp.stack([iso_orb == 1, iso_orb == -1], axis=-1)  # (orb, 2): c=0 proton, 1 neutron
    sel = (spin_sel[:, :, None] & iso_sel[:, None, :]).astype(jnp.complex128)  # (orb, 2, 2)

    i_np, j_np = np.triu_indices(N, k=1)
    i, j = jnp.asarray(i_np), jnp.asarray(j_np)
    n_pairs = i.shape[0]
    # For a move of pair (p, q): the 2N - 3 distinct pairs containing p or q,
    # indexed by the pair number of (p, q) (pair_number[p, q], p < q or p > q).
    pair_number = np.zeros((N, N), dtype=int)
    pair_number[i_np, j_np] = pair_number[j_np, i_np] = np.arange(len(i_np))
    touched = jnp.asarray(np.array([np.nonzero((i_np == a) | (j_np == a) | (i_np == b) | (j_np == b))[0]
                                    for a, b in zip(i_np, j_np)]))
    pair_number = jnp.asarray(pair_number)

    params = {
        "jastrow": jnp.zeros((N_CHANNELS, n_terms)),
        "corr": jnp.zeros((len(OPERATORS_SO if spin_orbit else OPERATORS), n_terms)),
    }
    ik = 1j * k  # (orb, 3): d/du of the plane waves
    # Target spin states |b_i b_j> in basis order.
    bi = jnp.array([0, 0, 1, 1])
    bj = jnp.array([0, 1, 0, 1])

    # --- building blocks ---------------------------------------------------------
    def columns(rq, xq, tq):
        """Slater columns for particles at rq (n,3) with spinors xq (n,2), isospins tq: (orb, n), plane waves."""
        plane = jnp.exp(1j * (k @ rq.T))  # (orbitals, n)
        spin = xq[:, spin_orb].T
        iso = iso_orb[:, None] == tq[None, :]
        return jnp.where(iso, plane * spin, 0.0), plane

    def contract(G, plane):
        """D[p, q, b, c] = sum_a G[p, a] plane[a, q] sel[a, b, c]: rows of G with every possible new column."""
        return jnp.einsum("pa,aq,abc->pqbc", G, plane, sel)

    def contract_k(G, plane):
        """Dk[p, q, b, c, x] = sum_a G[p, a] (i k_x) plane[a, q] sel[a, b, c]: derivatives of the columns."""
        return jnp.einsum("pa,aq,abc,ax->pqbcx", G, plane, sel, ik)

    def ratio_derivatives(D, Dk, t):
        """d_u [Phi(target)/Phi] for every pair and spin target (moving i by +u, j by -u): (same, swapped)."""
        ci = (1 - t[i]) // 2
        cj = (1 - t[j]) // 2
        P = jnp.arange(n_pairs)

        def ddet2(ci_new, cj_new):
            d_ii = D[i, i][:, bi, :][P, :, ci_new][..., None]  # (n_pairs, 4, 1)
            d_ij = D[i, j][:, bj, :][P, :, cj_new][..., None]
            d_ji = D[j, i][:, bi, :][P, :, ci_new][..., None]
            d_jj = D[j, j][:, bj, :][P, :, cj_new][..., None]
            k_ii = Dk[i, i][:, bi][P, :, ci_new]  # (n_pairs, 4, 3)
            k_ij = Dk[i, j][:, bj][P, :, cj_new]
            k_ji = Dk[j, i][:, bi][P, :, ci_new]
            k_jj = Dk[j, j][:, bj][P, :, cj_new]
            return k_ii * d_jj - d_ii * k_jj + k_ij * d_ji - d_ij * k_ji

        return ddet2(ci, cj), ddet2(cj, ci)

    def ratios(D, t):
        """Phi(target) / Phi for every pair and spin target; isospins unchanged and swapped."""
        ci = (1 - t[i]) // 2
        cj = (1 - t[j]) // 2

        def det2(ci_new, cj_new):
            d_ii = D[i, i][:, bi, :][jnp.arange(n_pairs), :, ci_new]  # (n_pairs, 4)
            d_ij = D[i, j][:, bj, :][jnp.arange(n_pairs), :, cj_new]
            d_ji = D[j, i][:, bi, :][jnp.arange(n_pairs), :, ci_new]
            d_jj = D[j, j][:, bj, :][jnp.arange(n_pairs), :, cj_new]
            return d_ii * d_jj - d_ij * d_ji

        return det2(ci, cj), det2(cj, ci)

    def amplitudes(a, d, xa, xb, ta, tb):
        """<current|U|target> for pairs with separations d, spinors xa, xb and isospins ta, tb: (same, swapped).

        The bra is contracted with the spin operators first (no dependence on the
        positions), so derivatives with respect to positions only pass through
        the radial functions and the tensor direction.
        """
        bra = jnp.einsum("pa,pb->pab", xa, xb).reshape(-1, 4)
        s1 = bra  # bra . identity
        ss = bra @ DOT  # bra . sigma.sigma, (pairs, 4)
        sab = jnp.einsum("pc,abct->pabt", bra, SIGMA_SIGMA)  # bra . sigma_a sigma_b, (pairs, 3, 3, 4)
        r2 = jnp.sum(d**2, axis=-1)
        u = jastrow_basis(r2, rc, n_terms) @ a.T  # (pairs, 5): u_s, u_t, u_st, w_tn, w_tnt
        dsd = jnp.einsum("pa,pb,pabt->pt", d, d, sab)  # d_a d_b bra.sigma_a sigma_b

        def tensor(w):  # bra . u_T S12 = w [3 d.(sigma x sigma).d - r^2 sigma.sigma] / rc^2
            return w[:, None] * (3.0 * dsd - r2[:, None] * ss) / rc**2

        A_row = u[:, 0, None] * ss + tensor(u[:, 3])
        B_row = u[:, 1, None] * s1 + u[:, 2, None] * ss + tensor(u[:, 4])
        b_cur = pair_index(ta, tb)
        b_sw = pair_index(tb, ta)
        differ = (ta != tb).astype(jnp.float64)
        tt_same = DOT[b_cur, b_cur][:, None]
        tt_swap = (DOT[b_cur, b_sw] * differ)[:, None]
        amp_same = A_row + B_row * tt_same
        amp_swap = B_row * tt_swap
        if not spin_orbit:
            return amp_same, amp_swap, None, None
        # u L.S with L_a = -i (r x d_r)_a, d_r = d_u / 2: sum_a s_a L_a -> w . d_u, w = -i/2 u (s x r)
        s_vec = jnp.einsum("pc,act->pta", bra, S_PAIR)  # <current|S_a|target>, (pairs, 4, 3)
        sxr = jnp.cross(s_vec, d[:, None, :])  # (pairs, 4, 3)
        w_ls = -0.5j * u[:, 5, None, None] * sxr
        w_lst = -0.5j * u[:, 6, None, None] * sxr
        w_same = w_ls + w_lst * tt_same[..., None]
        w_swap = w_lst * tt_swap[..., None]
        return amp_same, amp_swap, w_same, w_swap

    def jastrow(params, r, t):
        d = minimum_image(r[i] - r[j], L)
        phi = jastrow_basis(jnp.sum(d**2, axis=-1), rc, n_terms)
        return jnp.sum(phi * params["jastrow"][pair_channel(t, i, j)])

    # --- full evaluation ---------------------------------------------------------
    def log_psi(params, r, s, t):
        xi = as_spinors(s)
        M, plane = columns(r, xi, t)
        sign, logabs = jnp.linalg.slogdet(M)
        G = jnp.linalg.inv(M)  # (particles, orbitals)
        D = contract(G, plane)
        rho_same, rho_swap = ratios(D, t)
        d = minimum_image(r[i] - r[j], L)
        amp_same, amp_swap, w_same, w_swap = amplitudes(params["corr"], d, xi[i], xi[j], t[i], t[j])
        correlation = 1.0 + jnp.sum(amp_same * rho_same + amp_swap * rho_swap)
        if spin_orbit:
            drho_same, drho_swap = ratio_derivatives(D, contract_k(G, plane), t)
            correlation = correlation + jnp.sum(w_same * drho_same + w_swap * drho_swap)
        return jastrow(params, r, t) + logabs + jnp.log(sign) + jnp.log(correlation)

    # --- pair moves ----------------------------------------------------------------
    def prepare(params, r, s, t):
        """Everything about the current configuration that pair moves reuse."""
        xi = as_spinors(s)
        M, plane = columns(r, xi, t)
        G = jnp.linalg.inv(M)
        D = contract(G, plane)
        rho_same, rho_swap = ratios(D, t)
        d = minimum_image(r[i] - r[j], L)
        amp_same, amp_swap, w_same, w_swap = amplitudes(params["corr"], d, xi[i], xi[j], t[i], t[j])
        correlation = 1.0 + jnp.sum(amp_same * rho_same + amp_swap * rho_swap)
        cache = {"r": r, "xi": xi, "t": t, "M": M, "G": G, "D": D, "amp_same": amp_same,
                 "amp_swap": amp_swap, "J": jastrow(params, r, t)}
        if spin_orbit:
            Dk = contract_k(G, plane)
            drho_same, drho_swap = ratio_derivatives(D, Dk, t)
            correlation = correlation + jnp.sum(w_same * drho_same + w_swap * drho_swap)
            cache.update(Dk=Dk, w_same=w_same, w_swap=w_swap)
        cache["log_corr"] = jnp.log(correlation)
        return cache

    def move(params, cache, p, q, u, xp, xq, tp, tq):
        """ln Psi(X') - ln Psi(X), X': r_p + u, r_q - u, spinors xp, xq, isospins tp, tq.

        Written with masks and gathers only (no scatters): under vmap and forward-mode
        derivatives, XLA scatters are slow on CPU.
        """
        on_p = jnp.arange(N) == p
        on_q = jnp.arange(N) == q
        r = cache["r"] + on_p[:, None] * u - on_q[:, None] * u
        xi = jnp.where(on_p[:, None], xp, jnp.where(on_q[:, None], xq, cache["xi"]))
        t = jnp.where(on_p, tp, jnp.where(on_q, tq, cache["t"]))
        G, M, D = cache["G"], cache["M"], cache["D"]
        pq = jnp.stack([p, q])

        # Slater determinant: rank-2 update of columns p, q.
        cols, plane = columns(jnp.stack([r[p], r[q]]), jnp.stack([xp, xq]), jnp.stack([tp, tq]))  # (orb, 2)
        R = G[pq] @ cols  # (2, 2)
        det_R = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
        R_inv = jnp.array([[R[1, 1], -R[0, 1]], [-R[1, 0], R[0, 0]]]) / det_R
        GU = G @ (cols - M[:, pq])  # (N, 2)
        E_new = contract(G, plane)  # (N, 2, 2, 2): G times the new columns of p and q
        mp, mq = on_p[None, :, None, None], on_q[None, :, None, None]
        E = jnp.where(mp, E_new[:, 0][:, None], jnp.where(mq, E_new[:, 1][:, None], D))
        X = GU @ R_inv  # (N, 2); rank-2 correction as broadcast products (fast in XLA)
        D_new = E - X[:, 0, None, None, None] * E[p][None] - X[:, 1, None, None, None] * E[q][None]
        rho_same, rho_swap = ratios(D_new, t)
        if spin_orbit:  # the same rank-2 update for the derivative columns
            Ek_new = contract_k(G, plane)  # (N, 2, 2, 2, 3)
            Ek = jnp.where(mp[..., None], Ek_new[:, 0][:, None], jnp.where(mq[..., None], Ek_new[:, 1][:, None],
                                                                           cache["Dk"]))
            Dk_new = Ek - X[:, 0, None, None, None, None] * Ek[p][None] - X[:, 1, None, None, None, None] * Ek[q][None]
            drho_same, drho_swap = ratio_derivatives(D_new, Dk_new, t)

        # Amplitudes: only the 2N - 3 pairs containing p or q change. Their old
        # contribution is replaced by the new one.
        idx = touched[pair_number[p, q]]
        d = minimum_image(r[i[idx]] - r[j[idx]], L)
        new_same, new_swap, nw_same, nw_swap = amplitudes(params["corr"], d, xi[i[idx]], xi[j[idx]],
                                                          t[i[idx]], t[j[idx]])
        old_same, old_swap = cache["amp_same"][idx], cache["amp_swap"][idx]
        correlation = (1.0 + jnp.sum(cache["amp_same"] * rho_same + cache["amp_swap"] * rho_swap)
                       + jnp.sum((new_same - old_same) * rho_same[idx] + (new_swap - old_swap) * rho_swap[idx]))
        if spin_orbit:
            correlation = (correlation + jnp.sum(cache["w_same"] * drho_same + cache["w_swap"] * drho_swap)
                           + jnp.sum((nw_same - cache["w_same"][idx]) * drho_same[idx]
                                     + (nw_swap - cache["w_swap"][idx]) * drho_swap[idx]))

        return jastrow(params, r, t) - cache["J"] + jnp.log(det_R) + jnp.log(correlation) - cache["log_corr"]

    log_psi.pair_move = (prepare, move)
    return params, log_psi
