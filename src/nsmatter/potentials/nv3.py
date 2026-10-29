"""NV3: the local chiral three-nucleon force that goes with NV2 (Norfolk models).

References: Piarulli et al., PRL 120, 052503 (2018) [operators, Eqs. 3-11];
Baroni et al., PRC 98, 044003 (2018) [starred fits]; in pure neutron matter as
written in Lovato et al., PRC 105, 055808 (2022), Eqs. (9)-(13) and Table I.

    V^Delta(ijk) = -g_A^2 h_A^2 / (72 * 144 pi^2) * m_pi^6 / (m_DN f_pi^4) * 2 {X_ij, X_jk}
    V^2pi(ijk)   =  g_A^2 / (256 pi^2) * m_pi^6 / f_pi^4
                    * [ -16 c1 Z(r_ij) Z(r_jk) (sigma_i.r^_ij)(sigma_k.r^_jk) + 4/9 c3 {X_ij, X_jk} ]
    V^CT(ijk)    =  g_A c_D / (96 pi) * m_pi^3 / (Lambda_chi f_pi^4) * X_ik [C_RS(r_ij) + C_RS(r_jk)]
                  + c_E / (Lambda_chi f_pi^4) * C_RS(r_ij) C_RS(r_jk)
with (tau.tau = 1 in neutron matter already used)
    Y(r) = exp(-x)/x C_RL(r),  T(r) = (1 + 3/x + 3/x^2) Y(r),  Z(r) = -(1 + 1/x) Y(r),  x = m_pi r,
    C_RL(r) = 1 - 1/((r/R_L)^6 exp((r - R_L)/a_L) + 1), a_L = R_L/2,
    C_RS(r) = exp(-(r/R_S)^2) / (pi^(3/2) R_S^3),   Lambda_chi = 1 GeV.
g_A, m_pi (average), h_A, m_DN, c1, c3 are those of NV2 (nv2.py); cutoffs (R_S, R_L)
as in the NV2 model.
f_pi here is the pion decay constant F_pi = 92.4 MeV, i.e. HALF the f_pi = 184.80 MeV
parameter of the NV2 code (nv2.F_PI). This is the convention in which c_D and c_E are
defined (the contact term E = c_E / (Lambda_chi F_pi^4) of chiral EFT), and with it
the Fujita-Miyazawa strength, about -0.04 MeV for {X,X}{tau.tau, tau.tau}, is
comparable to the Urbana IX value (-0.029 MeV); with 184.80 every term would be 16x
smaller than any realistic 3N force. To be confirmed against the published NV2+3
neutron-matter energies (Lovato et al. 2022).

Models: "Ia", "Ib", "IIa", "IIb" (c_D, c_E fitted to trinucleon energies and the
nd doublet scattering length) and "Ia*", ..., "IIb*" (also fitted to the tritium
Gamow-Teller matrix element). Lovato et al. find that c_E below about -0.1 makes
pure neutron matter collapse: for neutron matter use the starred models.
"""

import jax.numpy as jnp
import numpy as np

from ..hamiltonian.three_body import make_three_body_pnm
from ..system import System
from .nv2 import C1, C3, DELTA_M, F_PI, G_A, H_A, HBARC, M_PI, MODELS
from .nv2_vmc import model_key

LAMBDA_CHI = 1000.0  # MeV
F_PI_3N = F_PI / 2  # 92.4 MeV, see the module docstring
# (c_D, c_E), Lovato et al. PRC 105, 055808 (2022), Table I.
CD_CE = {
    "Ia": (3.666, -1.638), "Ib": (-2.061, -0.982), "IIa": (1.278, -1.029), "IIb": (-4.480, -0.412),
    "Ia*": (-0.635, -0.090), "Ib*": (-4.71, 0.55), "IIa*": (-0.61, -0.35), "IIb*": (-5.25, 0.05),
}


def nv3_couplings(model):
    """Coefficients of the operators in hamiltonian/three_body.py (MeV units), and (R_S, R_L) in fm."""
    cD, cE = CD_CE[model]
    R_L, R_S = MODELS[model_key(model.rstrip("*"))]
    f4 = F_PI_3N**4
    K = G_A**2 * M_PI**6 / (256 * np.pi**2 * f4)
    return {
        "a_c1": -16 * C1 * K,
        "a_c3": 4 / 9 * C3 * K,
        "a_Delta": -2 * G_A**2 * H_A**2 * M_PI**6 / (72 * 144 * np.pi**2 * DELTA_M * f4),
        "a_D": G_A * cD / (96 * np.pi) * M_PI**3 / (LAMBDA_CHI * f4),
        "a_E": cE / (LAMBDA_CHI * f4),
    }, (R_S, R_L)


def nv3_radial(R_S, R_L, r_max):
    """r (fm) -> dict Y, T, Z (dimensionless) and C (C_RS in MeV^3), zero beyond r_max."""
    a_L = R_L / 2
    m = M_PI / HBARC  # fm^-1

    def f(r):
        x = m * r
        z = (r / R_L) ** 6 * jnp.exp((r - R_L) / a_L)
        C_L = z / (1 + z)  # = 1 - 1/(z + 1), stable for large z
        inside = r < r_max
        Y = jnp.where(inside, jnp.exp(-x) / x * C_L, 0.0)
        C_S = jnp.where(inside, jnp.exp(-((r / R_S) ** 2)) / (np.pi**1.5 * R_S**3), 0.0) * HBARC**3
        return {"Y": Y, "T": (1 + 3 / x + 3 / x**2) * Y, "Z": -(1 + 1 / x) * Y, "C": C_S}

    return f


def make_nv3_potential(system: System, model, r_max=12.0):
    """NV3 potential term for pure neutron matter (model "Ia*", "Ib*", ... or unstarred)."""
    couplings, (R_S, R_L) = nv3_couplings(model)
    return make_three_body_pnm(system, nv3_radial(R_S, R_L, r_max), couplings, r_max)
