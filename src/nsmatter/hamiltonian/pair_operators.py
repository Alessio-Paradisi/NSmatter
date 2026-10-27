"""General two-body spin-isospin operators, including the tensor force (v6).

    v_ij = [ v_c + v_s sigma_i.sigma_j + v_tn S12 ]
         + [ v_t + v_st sigma_i.sigma_j + v_tnt S12 ] tau_i.tau_j

with the tensor operator
    S12(r) = 3 (sigma_i . r^)(sigma_j . r^) - sigma_i.sigma_j.

Unlike sigma.sigma, S12 is not an exchange: it can flip both spins of a pair
(e.g. |up up> -> |down down>). So each pair operator is written as explicit
4 x 4 matrices on the two-body spin states |s_i s_j>, and
    E_pot,L = sum_pairs sum_targets <current|v_ij|target> Psi(target) / Psi,
the ordinary matrix-vector product (v Psi)(X) = sum_X' <X|v|X'> Psi(X').
Spins are spinors (labels are the special case of basis spinors): the
current pair state is xi_i (x) xi_j, and each target replaces xi_i, xi_j
by the basis spinors of one of the 4 two-body states.
Targets: 4 spin states of the pair, times 2 isospin states (unchanged,
or swapped if t_i != t_j); tau.tau never changes the total isospin
projection, so no other isospin state appears. That is 8 evaluations of
Psi per pair.

Two-body basis index of |a b>: 2 * idx(a) + idx(b), idx(+1) = 0, idx(-1) = 1.

Radial functions, all summed over periodic images:
- "c", "s", "t", "st": v(r) = sum_k V_k exp(-kappa_k r^2)            (as central.py)
- "tn", "tnt":         v(r) = sum_k V_k r^2 exp(-kappa_k r^2)
  The r^2 factor makes v_T vanish at the origin, as physical tensor forces
  do, and lets the code use v(r) r^_a r^_b = V exp(-kappa r^2) r_a r_b
  without dividing by r (no singularity when two particles meet).
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..spin import as_spinors
from ..system import System
from .central import gaussian_sum, image_shifts

# Pauli matrices sigma_x, sigma_y, sigma_z; basis (up, down).
PAULI = jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=jnp.complex128,
)

# sigma_i . sigma_j as a 4 x 4 matrix on |s_i s_j> (it is real).
DOT = sum(jnp.kron(PAULI[a], PAULI[a]) for a in range(3))

# All 9 products sigma_a (x) sigma_b, shape (3, 3, 4, 4), for the tensor part.
SIGMA_SIGMA = jnp.array([[jnp.kron(PAULI[a], PAULI[b]) for b in range(3)] for a in range(3)])

CHANNELS = ("c", "s", "t", "st", "tn", "tnt")


def pair_index(a, b):
    """Two-body basis index of |a b> for labels a, b in {+1, -1}."""
    return 2 * ((1 - a) // 2) + (1 - b) // 2


def tensor_operator(rvec):
    """S12 for the direction of rvec, as a 4 x 4 matrix (for tests and checks)."""
    rhat = rvec / jnp.linalg.norm(rvec)
    return 3.0 * jnp.einsum("a,b,abij->ij", rhat, rhat, SIGMA_SIGMA) - DOT


def make_v6_potential(system: System, radial: dict, tol=1e-10):
    """Potential term for make_local_energy: term(log_psi, params, r, s, t)."""
    L = system.box_length

    # Image vectors. The r^2 factor of the tensor channels lengthens their
    # tail, so the tolerance is tightened when they are present.
    all_terms = tuple(term for ch in CHANNELS for term in radial.get(ch, ()))
    has_tensor = bool(radial.get("tn") or radial.get("tnt"))
    shifts = jnp.asarray(image_shifts(L, all_terms, tol * (1e-3 if has_tensor else 1.0)))

    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    n_pairs = i.shape[0]

    def scalar_sum(r2, ch):
        """Image-summed scalar radial function of a channel, shape (n_pairs,)."""
        if not radial.get(ch):
            return jnp.zeros(n_pairs)
        return jnp.sum(gaussian_sum(r2, radial[ch]), axis=1)

    def tensor_matrix(dd, r2, ch):
        """Image-summed v_T S12 as 4 x 4 spin matrices, shape (n_pairs, 4, 4).

        sum_n v(r_n) S12(r^_n) = sum_n w_n [3 r_n r_n^T . (sigma x sigma) - r_n^2 DOT]
        with v(r) = r^2 w(r), w(r) = sum_k V_k exp(-kappa_k r^2).
        """
        if not radial.get(ch):
            return jnp.zeros((n_pairs, 4, 4), dtype=jnp.complex128)
        w = gaussian_sum(r2, radial[ch])  # (n_pairs, n_images)
        rr = 3.0 * jnp.einsum("pn,pna,pnb->pab", w, dd, dd)  # (n_pairs, 3, 3)
        trace_part = jnp.sum(w * r2, axis=1)  # sum_n v(r_n)
        return jnp.einsum("pab,abij->pij", rr, SIGMA_SIGMA) - trace_part[:, None, None] * DOT

    def term(log_psi, params, r, s, t):
        # --- geometry: all images of all pairs ---------------------------------
        d = minimum_image(r[i] - r[j], L)
        dd = d[:, None, :] + shifts[None]  # (n_pairs, n_images, 3)
        r2 = jnp.sum(dd**2, axis=-1)

        # --- spin operators of each pair, split by isospin dependence --------
        # A: acts with isospin identity; B: multiplies tau.tau.
        eye = jnp.eye(4, dtype=jnp.complex128)
        A = (
            scalar_sum(r2, "c")[:, None, None] * eye
            + scalar_sum(r2, "s")[:, None, None] * DOT
            + tensor_matrix(dd, r2, "tn")
        )
        B = (
            scalar_sum(r2, "t")[:, None, None] * eye
            + scalar_sum(r2, "st")[:, None, None] * DOT
            + tensor_matrix(dd, r2, "tnt")
        )

        # --- spin targets in an adapted basis ----------------------------------
        # The identity (v Psi)(S) = sum_targets <S|v|target> <target|Psi> holds
        # for ANY orthonormal spin basis of each particle. We use {xi, xi_perp}:
        # the particle's own spinor and the one orthogonal to it. Then targets
        # never carry two identical basis spinors, and Phi(target) never vanishes
        # structurally (with basis spinors it can, e.g. two neutrons both "up" in
        # the same orbital, which breaks the fast updates of correlated states).
        # For discrete labels {xi, xi_perp} is {up, down} up to a phase.
        # Convention: a spin coordinate eta enters Psi as a bra, so the ket of
        # a target is conj(eta). Spinors must be normalized.
        xi = as_spinors(s)  # (N, 2)
        perp = jnp.stack([-jnp.conj(xi[:, 1]), jnp.conj(xi[:, 0])], axis=-1)
        eta = jnp.stack([xi, perp], axis=1)  # (N, 2 basis states, 2 components)
        # Target index 2*a + b: particle i in eta_i[a], particle j in eta_j[b].
        eta_i = eta[i][:, jnp.array([0, 0, 1, 1])]  # (n_pairs, 4, 2)
        eta_j = eta[j][:, jnp.array([0, 1, 0, 1])]
        kets = jnp.einsum("pta,ptb->ptab", jnp.conj(eta_i), jnp.conj(eta_j)).reshape(n_pairs, 4, 4)

        # --- matrix elements <current|v|target> -------------------------------
        # (v Psi)(X) = sum_X' <X|v|X'> Psi(X'): the current state is the bra
        # xi_i (x) xi_j, the target the ket.
        bra = jnp.einsum("pa,pb->pab", xi[i], xi[j]).reshape(n_pairs, 4)
        A_row = jnp.einsum("pc,pcd,ptd->pt", bra, A, kets)  # (n_pairs, 4 targets)
        B_row = jnp.einsum("pc,pcd,ptd->pt", bra, B, kets)
        # Isospin: discrete labels, tau.tau real and symmetric.
        b = pair_index(t[i], t[j])  # current isospin state
        b_sw = pair_index(t[j], t[i])  # isospin state after swapping
        differ = (t[i] != t[j]).astype(jnp.float64)
        tt_same = DOT[b, b]  # <t|tau.tau|t>: +1 if equal labels, -1 if different
        tt_swap = DOT[b, b_sw] * differ  # <t|tau.tau|swapped>: 2 if different, else 0
        amp_same = A_row + B_row * tt_same[:, None]  # isospins unchanged
        amp_swap = B_row * tt_swap[:, None]  # isospins swapped

        # --- target configurations --------------------------------------------
        def spin_targets(p_i, p_j, e_i, e_j):
            set_pair = lambda a, b: xi.at[p_i].set(a).at[p_j].set(b)  # noqa: E731
            return jax.vmap(set_pair)(e_i, e_j)

        s_new = jax.vmap(spin_targets)(i, j, eta_i, eta_j)  # (n_pairs, 4, N, 2)
        t_same = jnp.broadcast_to(t, (n_pairs, t.shape[0]))
        t_swap = jax.vmap(lambda p_i, p_j: t.at[p_i].set(t[p_j]).at[p_j].set(t[p_i]))(i, j)

        # --- wavefunction ratios at all targets --------------------------------
        lp0 = log_psi(params, r, xi, t)
        over_targets = jax.vmap(log_psi, in_axes=(None, None, 0, None))  # 4 spin targets
        over_pairs = jax.vmap(over_targets, in_axes=(None, None, 0, 0))  # all pairs
        rho_same = jnp.exp(over_pairs(params, r, s_new, t_same) - lp0)  # (n_pairs, 4)
        rho_swap = jnp.exp(over_pairs(params, r, s_new, t_swap) - lp0)

        return jnp.sum(amp_same * rho_same + amp_swap * rho_swap)

    return term
