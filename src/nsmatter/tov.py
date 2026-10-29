"""Neutron-star structure: Tolman-Oppenheimer-Volkoff equations and tidal deformability.

Geometric units inside (G = c = 1, lengths in km): energy density and pressure
in km^-2, mass in km. Input and output in MeV fm^-3, km and solar masses.

    dm/dr = 4 pi r^2 eps
    dP/dr = -(eps + P)(m + 4 pi r^3 P) / (r (r - 2m))

Tidal deformability (Hinderer 2008; Postnikov et al. 2010): y = r H'/H of the
static l = 2 perturbation obeys
    r dy/dr + y^2 + y F + r^2 Q = 0,  y(0) = 2,
    F = [1 - 4 pi r^2 (eps - P)] / (1 - 2m/r)
    Q = 4 pi [5 eps + 9 P + (eps + P) / c_s^2] / (1 - 2m/r) - 6 / (r^2 (1 - 2m/r))
        - 4 (m + 4 pi r^3 P)^2 / (r^4 (1 - 2m/r)^2)
If the EOS ends with eps(R-) > 0 at the surface, y(R) -> y(R) - 4 pi R^3 eps(R-) / M.
Then, with C = M/R,
    k2 = 8/5 C^5 (1-2C)^2 [2 + 2C(y-1) - y] / { 2C [6 - 3y + 3C(5y - 8)]
         + 4C^3 [13 - 11y + C(3y - 2) + 2C^2 (1 + y)] + 3 (1-2C)^2 [2 - y + 2C(y - 1)] ln(1 - 2C) }
    Lambda = 2/3 k2 / C^5.
For C < 1e-3 this formula cancels catastrophically (numerator and denominator
are both O(C^5)); there its Newtonian limit k2 = (2 - y) / (2 (3 + y)) is used.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

# G / c^4 * (1 MeV fm^-3) in km^-2, and G M_sun / c^2 in km.
_G, _C, _MEV, _MSUN = 6.67430e-11, 2.99792458e8, 1.602176634e-13, 1.98840987e30
MEV_FM3_TO_KM2 = _G / _C**4 * _MEV * 1e45 * 1e6
MSUN_KM = _G * _MSUN / _C**2 / 1e3


class TabulatedEOS:
    """eps(P) and c_s^2(P) from (eps, P) arrays (MeV fm^-3), monotone interpolation in log-log."""

    def __init__(self, eps, P):
        eps, P = np.asarray(eps, dtype=float), np.asarray(P, dtype=float)
        if np.any(np.diff(P) <= 0) or np.any(np.diff(eps) <= 0):
            raise ValueError("eps and P must increase together")
        self.log_eps = PchipInterpolator(np.log(P), np.log(eps))
        self.dlog = self.log_eps.derivative()
        self.p_min, self.p_max = P[0], P[-1]

    def eps(self, P):
        return np.exp(self.log_eps(np.log(P)))

    def cs2(self, P):
        """dP/deps = (P / eps) / (dlog eps / dlog P)."""
        return P / self.eps(P) / self.dlog(np.log(P))


def solve_tov(eos, p_central, tidal=True, rtol=1e-7):
    """One star with central pressure p_central (MeV fm^-3). Returns dict R (km), M (M_sun), C, k2, Lambda.

    `eos` needs eps(P) (and cs2(P) for the tidal part) in MeV fm^-3, P >= eos.p_min.
    The surface is where P drops to eos.p_min.
    """
    k = MEV_FM3_TO_KM2
    pc = p_central * k
    p_surf = eos.p_min * k * (1 + 1e-12)

    def eps(P):
        return eos.eps(P / k) * k

    def rhs(r, u):
        m, P, y = u
        P = max(P, p_surf)
        e = eps(P)
        f = 1 - 2 * m / r
        dm = 4 * np.pi * r**2 * e
        dP = -(e + P) * (m + 4 * np.pi * r**3 * P) / (r**2 * f)
        if tidal:
            cs2 = eos.cs2(P / k)
            F = (1 - 4 * np.pi * r**2 * (e - P)) / f
            Q = (4 * np.pi * (5 * e + 9 * P + (e + P) / cs2) / f - 6 / (r**2 * f)
                 - 4 * (m + 4 * np.pi * r**3 * P) ** 2 / (r**4 * f**2))
            dy = -(y**2 + y * F + r**2 * Q) / r
        else:
            dy = 0.0
        return [dm, dP, dy]

    def surface(r, u):
        return u[1] - p_surf

    surface.terminal, surface.direction = True, -1

    # Start slightly off-centre with the series m = 4/3 pi eps_c r^3, P = P_c - O(r^2).
    e_c = eps(pc)
    r0 = 1e-4  # km
    m0 = 4 / 3 * np.pi * e_c * r0**3
    p0 = pc - 2 * np.pi * (e_c + pc) * (pc + e_c / 3) * r0**2
    sol = solve_ivp(rhs, (r0, 100.0), [m0, p0, 2.0], events=surface, rtol=rtol, atol=1e-30, method="LSODA")
    if not sol.t_events[0].size:
        raise RuntimeError("no surface found within 100 km")
    R = sol.t_events[0][0]
    M, _, y = sol.y_events[0][0]
    out = {"R": R, "M": M / MSUN_KM, "C": M / R}
    if tidal:
        y = y - 4 * np.pi * R**3 * eps(p_surf) / M  # surface density discontinuity
        C = M / R
        if C < 1e-3:
            out["k2"] = (2 - y) / (2 * (3 + y))
            out["Lambda"] = 2 / 3 * out["k2"] / C**5
            return out
        num = 8 / 5 * C**5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (y - 1) - y)
        den = (2 * C * (6 - 3 * y + 3 * C * (5 * y - 8))
               + 4 * C**3 * (13 - 11 * y + C * (3 * y - 2) + 2 * C**2 * (1 + y))
               + 3 * (1 - 2 * C) ** 2 * (2 - y + 2 * C * (y - 1)) * np.log(1 - 2 * C))
        out["k2"] = num / den
        out["Lambda"] = 2 / 3 * out["k2"] / C**5
    return out


def mass_radius(eos, n=25, p_low=None, p_high=None, tidal=True, refine=6, rtol=1e-7):
    """Sequence of stars on a log grid of central pressures, stopped after the maximum mass.

    n stars on the coarse grid, then `refine` more between the neighbours of the
    heaviest one and two around 1.4 M_sun, so M_max, R_1.4 and Lambda_1.4 are precise
    without a fine grid everywhere (about 15x faster than a uniform fine grid).
    Returns dict of arrays p_c, R, M, Lambda (if tidal), plus M_max, R_at_Mmax,
    p_c_max and R_1.4 / Lambda_1.4 (interpolated on the stable branch, NaN if not reached).
    """
    p_low = 1.0 if p_low is None else p_low  # MeV fm^-3: ~0.1 M_sun stars
    p_high = eos.p_max if p_high is None else min(p_high, eos.p_max)
    rows = []

    def star(pc):
        s = solve_tov(eos, pc, tidal=tidal, rtol=rtol)
        rows.append((pc, s["R"], s["M"], s.get("Lambda", np.nan)))

    for pc in np.geomspace(p_low, p_high, n):
        star(pc)
        if len(rows) > 3 and rows[-1][2] < rows[-2][2] < rows[-3][2]:
            break  # past the maximum: unstable branch
    k = int(np.argmax([row[2] for row in rows]))
    if refine and 0 < k < len(rows) - 1:
        for pc in np.geomspace(rows[k - 1][0], rows[k + 1][0], refine + 2)[1:-1]:
            star(pc)
    rows.sort()
    if refine:  # two more stars around 1.4 M_sun on the stable branch, for R_1.4 and Lambda_1.4
        k = int(np.argmax([row[2] for row in rows]))
        for a, b in zip(rows[:k], rows[1:k + 1]):
            if a[2] < 1.4 <= b[2]:
                for pc in np.geomspace(a[0], b[0], 4)[1:-1]:
                    star(pc)
                break
        rows.sort()
    pc, R, M, Lam = (np.array(c) for c in zip(*rows))
    k = int(np.argmax(M))
    out = {"p_c": pc, "R": R, "M": M, "Lambda": Lam, "M_max": M[k], "R_at_Mmax": R[k], "p_c_max": pc[k]}
    stable = slice(0, k + 1)
    if M[0] < 1.4 < M[k]:
        out["R_1.4"] = float(np.interp(1.4, M[stable], R[stable]))
        out["Lambda_1.4"] = float(np.exp(np.interp(1.4, M[stable], np.log(Lam[stable])))) if tidal else np.nan
    else:
        out["R_1.4"] = out["Lambda_1.4"] = np.nan
    return out
