"""Three-nucleon force of the NV3 (Norfolk) form in pure neutron matter, as pair operators.

The force (Piarulli et al., PRL 120, 052503 (2018); in neutron matter as in Lovato
et al., PRC 105, 055808 (2022), Eqs. 9-13):
    V = sum_{i<j<k} sum_cyclic V(i j k),
    V(ijk) = a_X {X_ij, X_jk}                                       two-pion: c3 and Delta
           + a_1 Z(r_ij) Z(r_jk) (sigma_i . r^_ij)(sigma_k . r^_jk)  two-pion: c1
           + a_D X_ik [C(r_ij) + C(r_jk)]                           one-pion + contact: cD
           + a_E C(r_ij) C(r_jk)                                    contact: cE
    X_ij = Y(r_ij) sigma_i . sigma_j + T(r_ij) S_ij = sigma_i^a A^ab(r_ij) sigma_j^b,
    A(r) = Y(r) 1 + T(r) (3 r^ r^ - 1),     r_ij = r_i - r_j.
In neutron matter tau_i . tau_j = 1: the isospin anticommutator {tau.tau, tau.tau}
is 2 (included in a_X, a_1) and the commutator terms (c4) vanish.

Reduction to pair operators: sigma_j enters {X_ij, X_jk} only through
{sigma_j^b, sigma_j^c} = 2 delta^bc, so
    {X_ij, X_jk} = 2 sigma_i^a [A(r_ij) A(r_jk)]^ac sigma_k^c,
an operator on the spins of the OUTER pair (i, k) only. Every term is of this
form, sigma_i^a M^ac sigma_k^c (+ a scalar), and V(ijk) is symmetric in i <-> k, so
    V = sum_{pairs (a,b)} sum_{m != a,b} V(a m b)
is a sum of pair operators with matrices M(a,b) collected over the middle
particle m. Their local energy is computed like the static two-body potential
(pair_operators.pair_targets: 4 spin targets per pair, isospins unchanged).

Periodic images: with all images of m and b summed independently (infinite
lattice), every term factorizes into image sums of two-body quantities,
    sum_images A(r_am) A(r_mb) = Abar(a,m) Abar(m,b),   Abar(p,q) = sum_n A(d_pq + n L)
(likewise zbar = sum_n Z r^ and cbar = sum_n C), with d_pq the minimum image of
r_p - r_q. Triples in which a particle meets its own periodic copy are excluded.

Only pure (possibly spin-polarized) neutron matter: with protons the isospin
structure is genuinely three-body (commutator terms) and needs triple targets.
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image
from ..system import System
from .pair_operators import SIGMA_SIGMA, pair_targets
from .radial import image_vectors

CHANNELS = ("3b_c1", "3b_c3", "3b_Delta", "3b_D", "3b_E")


class _Range:
    def __init__(self, r_max):
        self.r_max = r_max


def make_three_body_pnm(system: System, radial, couplings, r_max):
    """Potential term (with by_channel) for the NV3-form force in neutron matter.

    radial: callable r (fm, jax array) -> dict with Y, T, Z (dimensionless) and C (MeV^3),
            all zero beyond r_max.
    couplings: dict a_c1, a_c3, a_Delta (MeV, multiply the operators above), a_D (MeV^-2),
               a_E (MeV^-5), see potentials/nv3.py.
    """
    if system.p_up + system.p_down > 0:
        raise NotImplementedError("the three-nucleon force is implemented for pure neutron matter only")
    N, L = system.n_particles, system.box_length
    shifts = jnp.asarray(image_vectors(L, [_Range(r_max)]))
    i, j = np.triu_indices(N, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)
    off = ~np.eye(N, dtype=bool)
    # middle particles of each pair: m != a, b
    middle = jnp.asarray((np.arange(N)[None, :] != np.asarray(i)[:, None])
                         & (np.arange(N)[None, :] != np.asarray(j)[:, None])).astype(jnp.float64)
    eye3 = jnp.eye(3)

    def leg_sums(r):
        """Image sums Abar (N,N,3,3), zbar (N,N,3), cbar (N,N) for all ordered pairs (p, q), p != q."""
        d = minimum_image(r[:, None, :] - r[None, :, :], L)  # d[p, q] = r_p - r_q
        dd = d[:, :, None, :] + shifts[None, None]  # (N, N, n_img, 3)
        r2 = jnp.sum(dd**2, axis=-1)
        valid = (r2 > 1e-12) & jnp.asarray(off)[:, :, None]  # no self pairs
        rr = jnp.sqrt(jnp.where(valid, r2, 1.0))
        f = radial(rr)
        Y, T, Z, C = (jnp.where(valid, f[k], 0.0) for k in ("Y", "T", "Z", "C"))
        rhat = dd / rr[..., None]
        outer = jnp.einsum("pqna,pqnb->pqnab", rhat, rhat)
        A = (Y - T)[..., None, None] * eye3 + 3.0 * T[..., None, None] * outer
        return A.sum(axis=2), jnp.einsum("pqn,pqna->pqa", Z, rhat), C.sum(axis=2)

    def by_channel(log_psi, params, r, s, t):
        Abar, zbar, cbar = leg_sums(r)
        A_am, A_mb = Abar[i], jnp.swapaxes(Abar[:, j], 0, 1)  # (P, N, 3, 3): [a, m] and [m, b]
        z_am, z_mb = zbar[i], jnp.swapaxes(zbar[:, j], 0, 1)  # (P, N, 3)
        c_am, c_mb = cbar[i], cbar[:, j].T  # (P, N)
        w = middle  # (P, N)

        M_X = 2.0 * jnp.einsum("pm,pmab,pmbc->pac", w, A_am, A_mb)  # {X_am, X_mb} -> sigma_a M sigma_b
        M_1 = jnp.einsum("pm,pma,pmb->pab", w, z_am, z_mb)
        M_D = Abar[i, j] * jnp.sum(w * (c_am + c_mb), axis=1)[:, None, None]
        S_E = jnp.sum(w * c_am * c_mb, axis=1)

        T_ = pair_targets(log_psi, params, r, s, t, i, j, needs_swap=False)

        def expect(M):
            op = jnp.einsum("pab,abst->pst", M, SIGMA_SIGMA)  # sigma_a^a M^ab sigma_b^b as 4 x 4
            row = jnp.einsum("pc,pcd,ptd->pt", T_["bra"], op, T_["kets"])
            return jnp.sum(row * T_["rho_same"])

        X = expect(M_X)
        return {
            "3b_c1": couplings["a_c1"] * expect(M_1),
            "3b_c3": couplings["a_c3"] * X,
            "3b_Delta": couplings["a_Delta"] * X,
            "3b_D": couplings["a_D"] * expect(M_D),
            "3b_E": couplings["a_E"] * jnp.sum(S_E) + 0.0j,  # scalar: target = current, ratio 1
        }

    def term(log_psi, params, r, s, t):
        return sum(by_channel(log_psi, params, r, s, t).values())

    term.by_channel = by_channel
    return term
