"""Equation of state from VMC energies: fits, thermodynamics, beta equilibrium.

Units: densities rho in fm^-3, energies per particle E/N in MeV, energy
densities and pressures in MeV fm^-3. "energy density" eps includes the rest mass.

1. Fit E/N(rho) to the VMC points:
     pure neutron matter (PNM):  E/N = a u^alpha + b u^beta,  u = rho / rho0
         (the standard QMC parametrization; a, alpha: low density, b, beta: stiffness)
     symmetric matter (SNM):     E/N = E0 + K/2 x^2 + Q/6 x^3,  x = (u - 1) / 3
         (expansion around saturation; E0 at u = 1, incompressibility K, skewness Q;
          rho0 is fixed, not fitted, which is enough for a first look)
   Each fit object returns E, dE/drho and d2E/drho2 analytically.
2. Thermodynamics at T = 0 for one component (PNM):
     P = rho^2 dE/drho,  eps = rho (m + E),  mu = (eps + P) / rho,
     c_s^2 = dP/deps = (2 rho E' + rho^2 E'') / (m + E + rho E')
3. Beta equilibrium (n, p, e, mu) with the quadratic isospin expansion
     E(rho, x) = E_SNM(rho) + S(rho) (1 - 2x)^2,  S = E_PNM - E_SNM,
   so mu_n - mu_p = 4 S (1 - 2x) + (m_n - m_p) = mu_e = mu_mu, charge x = x_e + x_mu.
4. Tabulate (eps, P) on a density grid for the TOV solver (tov.py).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, curve_fit

from .constants import HBARC, M_NEUTRON, M_PROTON

RHO0 = 0.16  # fm^-3, reference density of the fits
M_ELECTRON = 0.51099895  # MeV
M_MUON = 105.6583755  # MeV
DM = M_NEUTRON - M_PROTON  # MeV


# =============================================================================
# Fits of E/N(rho)
# =============================================================================
@dataclass
class PNMFit:
    """E/N = a u^alpha + b u^beta, u = rho / rho0."""

    a: float
    alpha: float
    b: float
    beta: float
    rho0: float = RHO0

    @staticmethod
    def form(rho, a, alpha, b, beta, rho0=RHO0):
        u = rho / rho0
        return a * u**alpha + b * u**beta

    def __call__(self, rho):
        """(E, dE/drho, d2E/drho2)."""
        u = np.asarray(rho, dtype=float) / self.rho0
        terms = [(self.a, self.alpha), (self.b, self.beta)]
        E = sum(c * u**p for c, p in terms)
        dE = sum(c * p * u ** (p - 1) for c, p in terms) / self.rho0
        d2E = sum(c * p * (p - 1) * u ** (p - 2) for c, p in terms) / self.rho0**2
        return E, dE, d2E

    @property
    def params(self):
        return np.array([self.a, self.alpha, self.b, self.beta])


@dataclass
class SNMFit:
    """E/N = E0 + K/2 x^2 + Q/6 x^3, x = (rho - rho0) / (3 rho0)."""

    E0: float
    K: float
    Q: float
    rho0: float = RHO0

    @staticmethod
    def form(rho, E0, K, Q, rho0=RHO0):
        x = (rho / rho0 - 1) / 3
        return E0 + K / 2 * x**2 + Q / 6 * x**3

    def __call__(self, rho):
        x = (np.asarray(rho, dtype=float) / self.rho0 - 1) / 3
        dx = 1 / (3 * self.rho0)
        E = self.E0 + self.K / 2 * x**2 + self.Q / 6 * x**3
        dE = (self.K * x + self.Q / 2 * x**2) * dx
        d2E = (self.K + self.Q * x) * dx**2
        return E, dE, d2E

    @property
    def params(self):
        return np.array([self.E0, self.K, self.Q])


def fit_pnm(rho, E, err, p0=(13.0, 0.5, 3.0, 2.3)):
    """Weighted least-squares PNM fit; returns (PNMFit, covariance of (a, alpha, b, beta))."""
    popt, pcov = curve_fit(PNMFit.form, np.asarray(rho), np.asarray(E), p0=p0, sigma=np.asarray(err),
                           absolute_sigma=True, bounds=([-np.inf, 0.0, -np.inf, 1.0], [np.inf, 2.0, np.inf, 6.0]),
                           maxfev=20000)
    return PNMFit(*popt), pcov


def fit_snm(rho, E, err, p0=(-16.0, 240.0, -300.0)):
    """Weighted least-squares SNM fit; returns (SNMFit, covariance of (E0, K, Q))."""
    popt, pcov = curve_fit(SNMFit.form, np.asarray(rho), np.asarray(E), p0=p0, sigma=np.asarray(err),
                           absolute_sigma=True, maxfev=20000)
    return SNMFit(*popt), pcov


# =============================================================================
# One-component thermodynamics (pure neutron matter)
# =============================================================================
def thermodynamics(fit, rho, mass=M_NEUTRON):
    """Dict of E, P, eps, mu, cs2 on the density array rho for a single-component fit."""
    rho = np.asarray(rho, dtype=float)
    E, dE, d2E = fit(rho)
    P = rho**2 * dE
    eps = rho * (mass + E)
    mu = mass + E + rho * dE
    cs2 = (2 * rho * dE + rho**2 * d2E) / mu
    return {"rho": rho, "E": E, "P": P, "eps": eps, "mu": mu, "cs2": cs2}


# =============================================================================
# Leptons and beta equilibrium
# =============================================================================
def lepton_gas(mu, mass):
    """Free relativistic Fermi gas at chemical potential mu (MeV): (n fm^-3, eps MeV fm^-3, P MeV fm^-3)."""
    mu = np.asarray(mu, dtype=float)
    kf = np.sqrt(np.clip(mu**2 - mass**2, 0.0, None))  # MeV
    e = np.sqrt(kf**2 + mass**2)
    log = np.log((kf + e) / mass) if mass > 0 else 0.0
    n = kf**3 / (3 * np.pi**2) / HBARC**3
    eps = (kf * e * (2 * kf**2 + mass**2) - mass**4 * log) / (8 * np.pi**2) / HBARC**3
    P = (kf * e * (2 * kf**2 - 3 * mass**2) + 3 * mass**4 * log) / (24 * np.pi**2) / HBARC**3
    return n, eps, P


def beta_equilibrium(pnm, snm, rho):
    """n p e mu matter in beta equilibrium. Returns dict of rho, x (proton fraction), P, eps, ...

    Uses E(rho, x) = E_SNM + S (1 - 2x)^2. At each rho, solves
    x rho = n_e(mu_e) + n_mu(mu_e) with mu_e = 4 S (1 - 2x) + (m_n - m_p).
    """
    rho = np.asarray(rho, dtype=float)
    out = {k: np.zeros_like(rho) for k in ("x", "mu_e", "P", "eps", "E", "S")}
    for k, r in enumerate(rho):
        Ep, dEp, _ = pnm(r)
        Es, dEs, _ = snm(r)
        S, dS = Ep - Es, dEp - dEs

        def charge(x):
            mu_e = 4 * S * (1 - 2 * x) + DM
            n_l = lepton_gas(mu_e, M_ELECTRON)[0] + lepton_gas(mu_e, M_MUON)[0]
            return x * r - n_l

        x = brentq(charge, 0.0, 0.5) if S > 0 else 0.0
        mu_e = 4 * S * (1 - 2 * x) + DM
        _, eps_e, P_e = lepton_gas(mu_e, M_ELECTRON)
        _, eps_m, P_m = lepton_gas(mu_e, M_MUON)
        d = (1 - 2 * x) ** 2
        E = Es + S * d
        dE = dEs + dS * d  # at fixed x
        out["x"][k], out["mu_e"][k], out["S"][k], out["E"][k] = x, mu_e, S, E
        out["eps"][k] = r * ((1 - x) * M_NEUTRON + x * M_PROTON + E) + eps_e + eps_m
        out["P"][k] = r**2 * dE + P_e + P_m
    out["rho"] = rho
    return out


# =============================================================================
# Table for the TOV solver
# =============================================================================
def eos_table(fit_or_table, rho_min=1e-4, rho_max=1.0, n=400, mass=M_NEUTRON):
    """(eps, P) arrays, MeV fm^-3, on a log grid in rho, cut where P stops increasing.

    fit_or_table: a PNMFit (pure neutron matter) or a dict with "eps" and "P"
    (e.g. from beta_equilibrium on a grid).
    """
    if isinstance(fit_or_table, dict):
        eps, P = np.asarray(fit_or_table["eps"]), np.asarray(fit_or_table["P"])
    else:
        th = thermodynamics(fit_or_table, np.geomspace(rho_min, rho_max, n), mass)
        eps, P = th["eps"], th["P"]
    start = int(np.argmax(P > 0))  # first point with positive pressure
    rising = np.diff(P[start:]) > 0
    end = start + 1 + (int(np.argmin(rising)) if not rising.all() else len(rising))
    return eps[start:end], P[start:end]


def causal_cap(eps, P):
    """Replace the EOS beyond the first point with dP/deps > 1 by P = P_t + (eps - eps_t) (c_s^2 = 1).

    The stiffest causal continuation: an upper bound on M_max when a fit becomes
    acausal at densities beyond the computed ones. Returns (eps, P, eps_t or None).
    """
    eps, P = np.asarray(eps, dtype=float), np.asarray(P, dtype=float)
    cs2 = np.diff(P) / np.diff(eps)  # slope between neighbouring points
    bad = np.nonzero(cs2 > 1)[0]
    if not len(bad):
        return eps, P, None
    k = bad[0]
    P = P.copy()
    P[k:] = P[k] + (eps[k:] - eps[k])
    return eps, P, float(eps[k])
