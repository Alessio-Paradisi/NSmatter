"""EOS thermodynamics, beta equilibrium and TOV against exact references.

- TOV: constant-density star (Schwarzschild interior solution, exact in GR);
  P = K eps^2 at low compactness -> Newtonian n = 1 polytrope:
  R = sqrt(pi K / 2) (G = c = 1) and Love number k2 = (15 - pi^2) / (2 pi^2).
- EOS: P = rho^2 dE/drho and mu = deps/drho by finite differences;
  beta equilibrium: charge neutrality, mu_e = 4 S (1 - 2x) + m_n - m_p, and
  thermodynamic consistency P = rho^2 d(eps/rho)/drho along the equilibrium.
"""

import numpy as np

from nsmatter.eos import DM, M_ELECTRON, M_MUON, PNMFit, SNMFit, beta_equilibrium, eos_table, lepton_gas, thermodynamics
from nsmatter.tov import MEV_FM3_TO_KM2, MSUN_KM, TabulatedEOS, mass_radius, solve_tov

PNM = PNMFit(12.7, 0.49, 3.3, 2.4)  # roughly the AV8'-like PNM of QMC studies
SNM = SNMFit(-16.0, 240.0, -300.0)


class ConstantDensity:
    def __init__(self, eps0, p_min):
        self.e0, self.p_min, self.p_max = eps0, p_min, np.inf

    def eps(self, P):
        return self.e0


class Polytrope:
    """P = K eps^2 (MeV fm^-3)."""

    def __init__(self, K, p_min):
        self.K, self.p_min, self.p_max = K, p_min, np.inf

    def eps(self, P):
        return np.sqrt(P / self.K)

    def cs2(self, P):
        return 2 * self.K * self.eps(P)


def test_constant_density_star():
    eps0 = 500.0  # MeV fm^-3
    R = 9.0  # km
    e = eps0 * MEV_FM3_TO_KM2
    M = 4 / 3 * np.pi * e * R**3
    s = np.sqrt(1 - 2 * M / R)
    pc = eps0 * (1 - s) / (3 * s - 1)
    star = solve_tov(ConstantDensity(eps0, 1e-12 * pc), pc, tidal=False)
    assert abs(star["R"] - R) < 1e-5
    assert abs(star["M"] - M / MSUN_KM) < 1e-5


def test_newtonian_polytrope_radius_and_love_number():
    R_target = 10.0
    K = 2 * R_target**2 / np.pi * MEV_FM3_TO_KM2  # K_geo = 2 R^2 / pi, back to MeV fm^-3 units
    eps_c = 0.05  # MeV fm^-3: very low compactness
    eos = Polytrope(K, 1e-14 * K * eps_c**2)
    star = solve_tov(eos, K * eps_c**2)
    assert star["C"] < 1e-4
    assert abs(star["R"] - R_target) / R_target < 2e-3
    assert abs(star["k2"] - (15 - np.pi**2) / (2 * np.pi**2)) < 2e-3


def test_pnm_thermodynamics():
    rho = np.linspace(0.05, 0.5, 7)
    h = 1e-6
    th = thermodynamics(PNM, rho)
    E = lambda r: PNM(r)[0]  # noqa: E731
    assert np.allclose(th["P"], rho**2 * (E(rho + h) - E(rho - h)) / (2 * h), rtol=1e-6)
    eps = lambda r: thermodynamics(PNM, r)["eps"]  # noqa: E731
    assert np.allclose(th["mu"], (eps(rho + h) - eps(rho - h)) / (2 * h), rtol=1e-8)


def test_beta_equilibrium():
    rho = np.array([0.08, 0.16, 0.32, 0.5])
    b = beta_equilibrium(PNM, SNM, rho)
    n_l = lepton_gas(b["mu_e"], M_ELECTRON)[0] + lepton_gas(b["mu_e"], M_MUON)[0]
    assert np.allclose(b["x"] * rho, n_l, rtol=1e-8)
    assert np.allclose(b["mu_e"], 4 * b["S"] * (1 - 2 * b["x"]) + DM, rtol=1e-10)
    assert np.all((0 < b["x"]) & (b["x"] < 0.25))
    h = 1e-6
    e_over_rho = lambda r: beta_equilibrium(PNM, SNM, np.atleast_1d(r))["eps"] / r  # noqa: E731
    dP = rho**2 * (e_over_rho(rho + h) - e_over_rho(rho - h)) / (2 * h)
    assert np.allclose(b["P"], dP, rtol=1e-5)


def test_pnm_star_is_reasonable():
    """Smoke test of the full chain fit -> table -> M(R): numbers in the expected range."""
    eps, P = eos_table(PNM)
    mr = mass_radius(TabulatedEOS(eps, P), n=40)
    assert 1.5 < mr["M_max"] < 3.0
    assert 9.0 < mr["R_1.4"] < 15.0
    assert 50 < mr["Lambda_1.4"] < 2000


def test_causal_cap():
    """After the cap, dP/deps <= 1 everywhere and P is continuous at the transition."""
    from nsmatter.eos import causal_cap

    eps, P = eos_table(PNM, rho_max=1.5)
    assert (np.diff(P) / np.diff(eps)).max() > 1  # this fit is acausal at high density
    e2, P2, e_t = causal_cap(eps, P)
    assert e_t is not None
    assert (np.diff(P2) / np.diff(e2)).max() <= 1 + 1e-9
    k = np.searchsorted(e2, e_t)
    assert np.allclose(P2[: k + 1], P[: k + 1])


def test_sly_reproduces_published_stars():
    """The analytic SLy fit through our TOV: M_max = 2.05 M_sun, R_1.4 = 11.7 km (Douchin & Haensel 2001)."""
    from nsmatter.eos import sly_table

    eps, P, _ = sly_table(eps_min=1e-8)
    mr = mass_radius(TabulatedEOS(eps, P))
    assert abs(mr["M_max"] - 2.05) < 0.02
    assert abs(mr["R_1.4"] - 11.7) < 0.1


def test_attach_crust():
    """Crust below the junction, core above it, eps(P) increasing, core untouched."""
    from nsmatter.eos import CRUST_CORE_DENSITY, attach_crust

    th = thermodynamics(PNM, np.geomspace(1e-4, 1.0, 400))
    eps, P = eos_table(th)
    rho = th["rho"][: len(eps)]
    e2, P2, r2, P_t = attach_crust(eps, P, rho)
    assert np.all(np.diff(P2) > 0) and np.all(np.diff(e2) > 0)
    assert np.allclose(P2[P2 >= P_t], P[P >= P_t])
    assert np.isclose(P_t, np.interp(CRUST_CORE_DENSITY, rho, P), rtol=1e-3)
    assert P2[0] < 1e-6 * P_t  # the crust reaches far below the core


def test_fit_powers_recovers_coefficients():
    from nsmatter.benchmarks import afdmc_energy, afdmc_fit
    from nsmatter.eos import fit_powers

    rho = np.linspace(0.04, 0.48, 9)
    fit = fit_powers(rho, afdmc_energy("Ib", "Ib*", rho), np.full(9, 0.1))
    assert np.allclose(fit.params, afdmc_fit("Ib", "Ib*").params, rtol=1e-8)
    E, dE, _ = fit(0.2)
    h = 1e-6
    assert np.isclose(dE, (fit(0.2 + h)[0] - fit(0.2 - h)[0]) / (2 * h), rtol=1e-6)
