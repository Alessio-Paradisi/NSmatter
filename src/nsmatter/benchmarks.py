"""Published reference results for neutron matter with the Norfolk interactions.

Lovato, Bombaci, Logoteta, Piarulli, Wiringa, PRC 105, 055808 (2022):
  AFDMC (66 neutrons, constraint released with 14) fits, Table IV,
      E/N = a_{2/3} u^{2/3} + a_1 u + a_2 u^2,   u = rho / 0.16 fm^-3   (MeV)
  and Table II, E/N at rho = 0.16 fm^-3 with 66 neutrons for NV2+3-Ia*:
      VMC with plane-wave (Jastrow) Slater: 20.40(1), with backflow: 18.42(1),
      DMC constrained path PW: 17.31(4), BF: 16.32(2).
Keys are (NV2 model, NV3 model).
"""

import numpy as np

RHO0 = 0.16

AFDMC_FITS = {  # (a_2/3, a_1, a_2) with errors
    ("Ia", "Ia*"): ((24.23, 0.44), (-15.09, 0.99), (6.02, 0.13)),
    ("Ib", "Ib*"): ((26.17, 0.18), (-18.71, 0.40), (10.85, 0.05)),
    ("IIb", "IIb*"): ((24.35, 0.46), (-15.11, 0.90), (6.49, 0.06)),
}

POINTS_RHO0 = {  # label -> (E/N, error), 66 neutrons, rho = 0.16
    ("Ia", "Ia*"): {"VMC plane waves": (20.40, 0.01), "VMC backflow": (18.42, 0.01),
                    "DMC-CP plane waves": (17.31, 0.04), "DMC-CP backflow": (16.32, 0.02)},
}

REFERENCE = "Lovato et al., PRC 105, 055808 (2022)"


def afdmc_energy(model, three_body, rho):
    """AFDMC fit E/N(rho) (MeV) for NV2+3, or None if not published."""
    fit = AFDMC_FITS.get((model, three_body))
    if fit is None:
        return None
    u = np.asarray(rho, dtype=float) / RHO0
    (a23, _), (a1, _), (a2, _) = fit
    return a23 * u ** (2 / 3) + a1 * u + a2 * u**2
