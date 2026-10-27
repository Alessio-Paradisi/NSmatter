"""Momentum-dependent two-body operators: L.S, L^2, (L.S)^2 and their spin-isospin forms.

Channels (each with a radial function, see radial.py):
    "ls"  : v L.S              "lst" : v L.S tau.tau
    "l2"  : v L^2              "l2t" : v L^2 tau.tau
    "l2s" : v L^2 sigma.sigma  "l2st": v L^2 (sigma.sigma)(tau.tau)
    "ls2" : v (L.S)^2          "ls2t": v (L.S)^2 tau.tau

L = r x p for the pair, r = r_i - r_j, p = -i d, d = (grad_i - grad_j) / 2 (the
derivative with respect to r), and S = (sigma_i + sigma_j) / 2. L acts on
positions, the spin-isospin operators on spins and isospins; they commute, so
    (O Psi)(X) = sum_targets <current|O_spin-iso|target> (O_L Psi)(target).
The L parts need d ln Psi = g and d d Psi / Psi = K = H + g g^T at each target,
H the Hessian of ln Psi in r. Derivatives are taken with respect to the
relative coordinate only: f(u) = ln Psi(r_i + u, r_j - u) has df/du = 2 g,
d2f/du2 = 4 H (3 variables instead of 3N).

Periodic images: L_n = r_n x p is linear in r_n, L_n L_n quadratic, so the
image sums reduce to the moments
    R = sum_n v(r_n) r_n,   M = sum_n v(r_n) r_n r_n^T,   s = tr M,
and per pair and target (eps = Levi-Civita):
    L_a Psi / Psi        = -i (R x g)_a
    L^2 Psi / Psi        = -s tr K + M : K + 2 R . g
    L_a L_b Psi / Psi    = -[eps_acd eps_bdf R_c g_f + eps_acd eps_bef M_ce K_df]
Spin targets use the adapted basis {xi, xi_perp} of pair_operators.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..spin import as_spinors
from ..system import System
from .pair_operators import DOT, PAULI, pair_index
from .radial import as_radial, image_vectors

EYE2 = jnp.eye(2, dtype=jnp.complex128)
EYE4 = jnp.eye(4, dtype=jnp.complex128)
# Total pair spin S_a = (sigma_a (x) 1 + 1 (x) sigma_a) / 2, shape (3, 4, 4).
S_PAIR = jnp.array([(jnp.kron(PAULI[a], EYE2) + jnp.kron(EYE2, PAULI[a])) / 2 for a in range(3)])
# S_a S_b, shape (3, 3, 4, 4).
SS_PAIR = jnp.einsum("aij,bjk->abik", S_PAIR, S_PAIR)
# Levi-Civita symbol.
EPS = np.zeros((3, 3, 3))
for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[a, b, c], EPS[a, c, b] = 1.0, -1.0
EPS = jnp.asarray(EPS)

# channel -> (kind of L operator, spin operator, multiplies tau.tau?)
CHANNELS = {
    "ls": ("ls", "S", False),
    "lst": ("ls", "S", True),
    "l2": ("l2", "1", False),
    "l2t": ("l2", "1", True),
    "l2s": ("l2", "ss", False),
    "l2st": ("l2", "ss", True),
    "ls2": ("ls2", "SS", False),
    "ls2t": ("ls2", "SS", True),
}


def make_momentum_potential(system: System, radial: dict):
    """Potential term for make_local_energy with any of the CHANNELS."""
    L = system.box_length
    radial = {ch: as_radial(radial.get(ch)) for ch in CHANNELS}
    active = [ch for ch in CHANNELS if radial[ch] is not None]
    shifts = jnp.asarray(image_vectors(L, radial.values()))
    needs_swap = any(CHANNELS[ch][2] for ch in active)  # isospin-swapped targets only for tau.tau

    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    n_pairs = i.shape[0]

    def term(log_psi, params, r, s, t):
        # --- image moments of every active channel -------------------------------
        d = minimum_image(r[i] - r[j], L)
        dd = d[:, None, :] + shifts[None]  # (n_pairs, n_images, 3)
        r2 = jnp.sum(dd**2, axis=-1)
        moments = {}
        for ch in active:
            w = radial[ch](r2)
            moments[ch] = (jnp.einsum("pn,pna->pa", w, dd), jnp.einsum("pn,pna,pnb->pab", w, dd, dd))

        # --- spin targets (adapted basis) and spin matrix elements ------------
        xi = as_spinors(s)
        perp = jnp.stack([-jnp.conj(xi[:, 1]), jnp.conj(xi[:, 0])], axis=-1)
        eta = jnp.stack([xi, perp], axis=1)
        eta_i = eta[i][:, jnp.array([0, 0, 1, 1])]
        eta_j = eta[j][:, jnp.array([0, 1, 0, 1])]
        kets = jnp.einsum("pta,ptb->ptab", jnp.conj(eta_i), jnp.conj(eta_j)).reshape(n_pairs, 4, 4)
        bra = jnp.einsum("pa,pb->pab", xi[i], xi[j]).reshape(n_pairs, 4)
        spin_amp = {
            "1": jnp.einsum("pc,cd,ptd->pt", bra, EYE4, kets),
            "ss": jnp.einsum("pc,cd,ptd->pt", bra, DOT, kets),
            "S": jnp.einsum("pc,acd,ptd->pat", bra, S_PAIR, kets),
            "SS": jnp.einsum("pc,abcd,ptd->pabt", bra, SS_PAIR, kets),
        }

        # --- isospin factors ----------------------------------------------------
        b = pair_index(t[i], t[j])
        b_sw = pair_index(t[j], t[i])
        differ = (t[i] != t[j]).astype(jnp.float64)
        iso = {  # (target isospins unchanged, swapped) for (identity, tau.tau)
            False: (jnp.ones(n_pairs), jnp.zeros(n_pairs)),
            True: (DOT[b, b], DOT[b, b_sw] * differ),
        }

        # --- ln Psi, g and K at every target --------------------------------------
        def spin_targets(p_i, p_j, e_i, e_j):
            return jax.vmap(lambda a, c: xi.at[p_i].set(a).at[p_j].set(c))(e_i, e_j)

        s_new = jax.vmap(spin_targets)(i, j, eta_i, eta_j)  # (n_pairs, 4, N, 2)
        t_same = jnp.broadcast_to(t, (n_pairs, t.shape[0]))
        t_swap = jax.vmap(lambda p_i, p_j: t.at[p_i].set(t[p_j]).at[p_j].set(t[p_i]))(i, j)
        lp0 = log_psi(params, r, xi, t)

        def derivatives(p_i, p_j, s_x, t_x):
            def f(u):
                x = r.at[p_i].add(u).at[p_j].add(-u)
                return log_psi(params, x, s_x, t_x)

            u0 = jnp.zeros(3)
            g = jax.jacfwd(f)(u0) / 2.0
            H = jax.jacfwd(jax.jacfwd(f))(u0) / 4.0
            return f(u0), g, H + jnp.outer(g, g)

        over_targets = jax.vmap(derivatives, in_axes=(None, None, 0, None))
        over_pairs = jax.vmap(over_targets, in_axes=(0, 0, 0, 0))

        def l_parts(lp, g, K):
            """Ratio and L-operator values at the targets, per active channel."""
            rho = jnp.exp(lp - lp0)  # (n_pairs, 4)
            out = {}
            trK = jnp.trace(K, axis1=-2, axis2=-1)
            for ch in active:
                R, M = moments[ch]
                kind = CHANNELS[ch][0]
                if kind == "ls":
                    out[ch] = -1j * jnp.cross(R[:, None, :], g)  # (n_pairs, 4, 3)
                elif kind == "l2":
                    s0 = jnp.trace(M, axis1=-2, axis2=-1)
                    out[ch] = (
                        -s0[:, None] * trK
                        + jnp.einsum("pbc,ptbc->pt", M, K)
                        + 2 * jnp.einsum("pb,ptb->pt", R, g)
                    )
                else:  # ls2: L_a L_b, shape (n_pairs, 4, 3, 3)
                    first = jnp.einsum("acd,bdf,pc,ptf->ptab", EPS, EPS, R, g)
                    second = jnp.einsum("acd,bef,pce,ptdf->ptab", EPS, EPS, M, K)
                    out[ch] = -(first + second)
            return rho, out

        def channel_sum(rho, lval, which):
            total = 0.0
            for ch in active:
                kind, spin_op, with_tt = CHANNELS[ch]
                factor = iso[with_tt][which]  # (n_pairs,)
                if kind == "ls":
                    val = jnp.einsum("pat,pta->pt", spin_amp["S"], lval[ch])
                elif kind == "l2":
                    val = spin_amp[spin_op] * lval[ch]
                else:
                    val = jnp.einsum("pabt,ptab->pt", spin_amp["SS"], lval[ch])
                total = total + jnp.sum(factor[:, None] * rho * val)
            return total

        rho, lval = l_parts(*over_pairs(i, j, s_new, t_same))
        energy = channel_sum(rho, lval, 0)
        if needs_swap:
            rho, lval = l_parts(*over_pairs(i, j, s_new, t_swap))
            energy = energy + channel_sum(rho, lval, 1)
        return energy

    return term


def make_ls_potential(system: System, radial: dict, tol=None):
    """Spin-orbit channels only ("ls", "lst"); kept for compatibility."""
    return make_momentum_potential(system, {ch: radial.get(ch) for ch in ("ls", "lst")})
