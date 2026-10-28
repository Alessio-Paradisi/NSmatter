"""General static two-body spin-isospin operators, including the tensor force.

    v_ij = sum_channels v_ch(r) O_spin(ch) O_iso(ch)

spin operators:    1, sigma_i.sigma_j, S12(r) = 3 (sigma_i.r^)(sigma_j.r^) - sigma_i.sigma_j
isospin operators: 1, tau_i.tau_j,
                   T12 = 3 tau_iz tau_jz - tau_i.tau_j  (charge-independence breaking),
                   tau_iz + tau_jz                      (charge-symmetry breaking)
Channels (CHANNELS): c, s, tn (isospin 1); t, st, tnt (tau.tau);
T, sT, tnT (T12); tz (tau_z + tau_z).

Unlike sigma.sigma, S12 is not an exchange: it can flip both spins of a pair
(e.g. |up up> -> |down down>). So each pair operator is written as explicit
4 x 4 matrices on the two-body spin states |s_i s_j>, and
    E_pot,L = sum_pairs sum_targets <current|v_ij|target> Psi(target) / Psi,
the ordinary matrix-vector product (v Psi)(X) = sum_X' <X|v|X'> Psi(X').
Spins are spinors (labels are the special case of basis spinors): the
current pair state is xi_i (x) xi_j, and each target replaces xi_i, xi_j
by the basis spinors of one of the 4 two-body states.
Isospins are fixed labels. All isospin operators above conserve the total
projection, so the only isospin targets are "unchanged" and "swapped"
(t_i <-> t_j, non-zero only if t_i != t_j); isospin_factors gives both
matrix elements for each operator. Swapped targets are evaluated only when
the system has both protons and neutrons and a tau.tau or T12 channel is active.

Two-body basis index of |a b>: 2 * idx(a) + idx(b), idx(+1) = 0, idx(-1) = 1.

Radial functions (radial.py: Gaussian sums or spline tables), summed over
periodic images:
- scalar channels:       v(r)
- tensor channels (S12): w(r) = v_T(r) / r^2, so the code uses v_T r^_a r^_b = w r_a r_b
  without dividing by r (no singularity when two particles meet). For Gaussian
  terms this means v_T(r) = sum_k V_k r^2 exp(-kappa_k r^2).
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..spin import as_spinors
from ..system import System
from .radial import as_radial, image_vectors

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

# channel -> (spin operator, isospin operator)
CHANNELS = {
    "c": ("1", "1"), "s": ("ss", "1"), "tn": ("S12", "1"),
    "t": ("1", "tt"), "st": ("ss", "tt"), "tnt": ("S12", "tt"),
    "T": ("1", "T"), "sT": ("ss", "T"), "tnT": ("S12", "T"),
    "tz": ("1", "tz"),
}
ISOSPIN_OPERATORS = ("1", "tt", "T", "tz")
SWAPPING = ("tt", "T")  # isospin operators with a swapped target


def pair_index(a, b):
    """Two-body basis index of |a b> for labels a, b in {+1, -1}."""
    return 2 * ((1 - a) // 2) + (1 - b) // 2


def isospin_factors(t, i, j):
    """Isospin matrix elements for pairs (i, j): operator -> (<t|O|t>, <t|O|t swapped>).

    tau.tau = 2 P_exchange - 1: diagonal +1 (equal labels) or -1, swapped 2 (if different).
    T12 = 3 tau_iz tau_jz - tau.tau, tau_iz + tau_jz diagonal.
    """
    ti, tj = t[i], t[j]
    differ = (ti != tj).astype(jnp.float64)
    tt_same = jnp.where(ti == tj, 1.0, -1.0)
    tt_swap = 2.0 * differ
    zero = jnp.zeros_like(tt_same)
    return {
        "1": (jnp.ones_like(tt_same), zero),
        "tt": (tt_same, tt_swap),
        "T": (3.0 * ti * tj - tt_same, -tt_swap),
        "tz": ((ti + tj).astype(jnp.float64), zero),
    }


def has_both_species(system: System):
    """True if the system has protons and neutrons (swapped isospin targets can be non-zero)."""
    n_p = system.p_up + system.p_down
    return 0 < n_p < system.n_particles


def tensor_operator(rvec):
    """S12 for the direction of rvec, as a 4 x 4 matrix (for tests and checks)."""
    rhat = rvec / jnp.linalg.norm(rvec)
    return 3.0 * jnp.einsum("a,b,abij->ij", rhat, rhat, SIGMA_SIGMA) - DOT


def make_static_potential(system: System, radial: dict):
    """Potential term for make_local_energy, term(log_psi, params, r, s, t), for any CHANNELS."""
    L = system.box_length
    # Radial function objects (tuples of Gaussian terms are converted).
    radial = {ch: as_radial(radial.get(ch)) for ch in CHANNELS}
    active = [ch for ch in CHANNELS if radial[ch] is not None]
    iso_active = sorted({CHANNELS[ch][1] for ch in active}, key=ISOSPIN_OPERATORS.index)
    needs_swap = has_both_species(system) and any(k in SWAPPING for k in iso_active)
    # Image vectors covering the longest-range channel.
    shifts = jnp.asarray(image_vectors(L, radial.values()))

    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    n_pairs = i.shape[0]

    def scalar_sum(r2, ch):
        """Image-summed scalar radial function of a channel, shape (n_pairs,)."""
        if radial[ch] is None:
            return jnp.zeros(n_pairs)
        return jnp.sum(radial[ch](r2), axis=1)

    def tensor_matrix(dd, r2, ch):
        """Image-summed v_T S12 as 4 x 4 spin matrices, shape (n_pairs, 4, 4).

        sum_n v(r_n) S12(r^_n) = sum_n w_n [3 r_n r_n^T . (sigma x sigma) - r_n^2 DOT]
        with w(r) = v_T(r) / r^2 the radial function of the channel.
        """
        if radial[ch] is None:
            return jnp.zeros((n_pairs, 4, 4), dtype=jnp.complex128)
        w = radial[ch](r2)  # (n_pairs, n_images)
        rr = 3.0 * jnp.einsum("pn,pna,pnb->pab", w, dd, dd)  # (n_pairs, 3, 3)
        trace_part = jnp.sum(w * r2, axis=1)  # sum_n v(r_n)
        return jnp.einsum("pab,abij->pij", rr, SIGMA_SIGMA) - trace_part[:, None, None] * DOT

    def term(log_psi, params, r, s, t):
        # --- geometry: all images of all pairs ---------------------------------
        d = minimum_image(r[i] - r[j], L)
        dd = d[:, None, :] + shifts[None]  # (n_pairs, n_images, 3)
        r2 = jnp.sum(dd**2, axis=-1)

        # --- spin operators of each pair, one 4 x 4 matrix per isospin operator ---
        eye = jnp.eye(4, dtype=jnp.complex128)
        spin = {k: jnp.zeros((n_pairs, 4, 4), dtype=jnp.complex128) for k in iso_active}
        for ch in active:
            spin_op, iso_op = CHANNELS[ch]
            if spin_op == "S12":
                spin[iso_op] = spin[iso_op] + tensor_matrix(dd, r2, ch)
            else:
                mat = eye if spin_op == "1" else DOT
                spin[iso_op] = spin[iso_op] + scalar_sum(r2, ch)[:, None, None] * mat

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
        iso = isospin_factors(t, i, j)
        amp_same = 0.0
        amp_swap = 0.0
        for k in iso_active:
            row = jnp.einsum("pc,pcd,ptd->pt", bra, spin[k], kets)  # (n_pairs, 4 targets)
            amp_same = amp_same + row * iso[k][0][:, None]  # isospins unchanged
            amp_swap = amp_swap + row * iso[k][1][:, None]  # isospins swapped

        # --- target configurations --------------------------------------------
        def spin_targets(p_i, p_j, e_i, e_j):
            set_pair = lambda a, b: xi.at[p_i].set(a).at[p_j].set(b)  # noqa: E731
            return jax.vmap(set_pair)(e_i, e_j)

        s_new = jax.vmap(spin_targets)(i, j, eta_i, eta_j)  # (n_pairs, 4, N, 2)
        t_same = jnp.broadcast_to(t, (n_pairs, t.shape[0]))
        t_swap = jax.vmap(lambda p_i, p_j: t.at[p_i].set(t[p_j]).at[p_j].set(t[p_i]))(i, j)

        # --- wavefunction ratios at all targets --------------------------------
        fast = getattr(log_psi, "pair_move", None)  # rank-2 updates (correlations.py)
        if fast is not None:
            prepare, move = fast
            cache = prepare(params, r, xi, t)
            u0 = jnp.zeros(3)
            ratio = lambda p_i, p_j, e_i, e_j, t_i, t_j: move(params, cache, p_i, p_j, u0, e_i, e_j, t_i, t_j)  # noqa: E731
            over_pairs = jax.vmap(jax.vmap(ratio, in_axes=(None, None, 0, 0, None, None)))
            rho_same = jnp.exp(over_pairs(i, j, eta_i, eta_j, t[i], t[j]))
            energy = jnp.sum(amp_same * rho_same)
            if needs_swap:
                rho_swap = jnp.exp(over_pairs(i, j, eta_i, eta_j, t[j], t[i]))
                energy = energy + jnp.sum(amp_swap * rho_swap)
            return energy

        lp0 = log_psi(params, r, xi, t)
        over_targets = jax.vmap(log_psi, in_axes=(None, None, 0, None))  # 4 spin targets
        over_pairs = jax.vmap(over_targets, in_axes=(None, None, 0, 0))  # all pairs
        rho_same = jnp.exp(over_pairs(params, r, s_new, t_same) - lp0)  # (n_pairs, 4)
        energy = jnp.sum(amp_same * rho_same)
        if needs_swap:
            rho_swap = jnp.exp(over_pairs(params, r, s_new, t_swap) - lp0)
            energy = energy + jnp.sum(amp_swap * rho_swap)
        return energy

    return term


def make_v6_potential(system: System, radial: dict, tol=None):
    """The v6 channels (c, s, t, st, tn, tnt) only; kept for compatibility."""
    return make_static_potential(system, {ch: radial.get(ch) for ch in ("c", "s", "t", "st", "tn", "tnt")})
