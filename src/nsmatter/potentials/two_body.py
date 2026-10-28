"""Two-nucleon checks of a local potential: deuteron and 1S0 scattering length.

Partial-wave matrix elements of the operators for a pair with orbital L,
spin S, total J, isospin T and projection M_T (proton = +1):
    sigma.sigma = 2S(S+1) - 3,   tau.tau = 2T(T+1) - 3
    L.S = [J(J+1) - L(L+1) - S(S+1)] / 2,   L^2 = L(L+1)
    T12 = 3 tau1z tau2z - tau.tau, with tau1z tau2z = +1 (|M_T| = 1) or -1 (M_T = 0)
    tau1z + tau2z = 2 M_T
    S12: 2 (J = L, S = 1); coupled L = J -+ 1:
         [[-2(J-1)/(2J+1), 6 sqrt(J(J+1))/(2J+1)], [6 sqrt(J(J+1))/(2J+1), -2(J+2)/(2J+1)]]
The radial functions are the dict of nv2.nv2_radial (tensor channels as v_T itself).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .nv2 import HBARC, M_N, M_P


def channel_potential(v, L, S, J, T, MT):
    """Central (scalar) and tensor-coefficient parts of v in one partial wave."""
    ss = 2 * S * (S + 1) - 3
    tt = 2 * T * (T + 1) - 3
    ls = (J * (J + 1) - L * (L + 1) - S * (S + 1)) / 2
    l2 = L * (L + 1)
    zz = 1 if abs(MT) == 1 else -1
    T12 = 3 * zz - tt
    tz = 2 * MT
    g = lambda k: v.get(k, 0.0)  # noqa: E731
    central = (g("c") + tt * g("t") + ss * (g("s") + tt * g("st")) + ls * (g("ls") + tt * g("lst"))
               + ls**2 * g("ls2") + l2 * (g("l2") + ss * g("l2s"))
               + T12 * (g("T") + ss * g("sT") + ls * g("lsT")) + tz * g("tz"))
    tensor = g("tn") + tt * g("tnt") + T12 * g("tnT")
    return central, tensor


def deuteron(v, r, mu=None):
    """Deuteron binding energy (MeV) and D-state probability from radial functions on grid r.

    Coupled 3S1-3D1 equations, finite differences on the grid (r[0] = 0 excluded,
    u = 0 at both ends), lowest eigenvalue of the symmetric matrix.
    """
    mu = M_N * M_P / (M_N + M_P) if mu is None else mu
    k = HBARC**2 / (2 * mu)  # hbar^2 / 2 mu, MeV fm^2
    h = r[1] - r[0]
    x = r[1:-1]
    vv = {key: np.asarray(val)[1:-1] for key, val in v.items()}
    cS, tS = channel_potential(vv, 0, 1, 1, 0, 0)
    cD, tD = channel_potential(vv, 2, 1, 1, 0, 0)
    # S12 in the coupled J = 1 channel: <S|S12|S> = 0, <S|S12|D> = sqrt(8), <D|S12|D> = -2.
    one = np.ones_like(x)
    VSS = cS * one
    VDD = (cD - 2 * tD) * one + k * 6 / x**2
    VSD = np.sqrt(8) * tS * one
    n = len(x)
    kin = diags([-k / h**2 * np.ones(n - 1), 2 * k / h**2 * np.ones(n), -k / h**2 * np.ones(n - 1)], [-1, 0, 1])
    from scipy.sparse import bmat
    H = bmat([[kin + diags(VSS), diags(VSD)], [diags(VSD), kin + diags(VDD)]]).tocsc()
    E, vec = eigsh(H, k=1, sigma=-3.0, which="LM")
    u, w = vec[:n, 0], vec[n:, 0]
    return -E[0], float((w**2).sum() / ((u**2).sum() + (w**2).sum()))


def scattering_length_1s0(v, r, MT=-1):
    """1S0 scattering length (fm) for nn (MT = -1), np (0) or pp (+1, strong part only).

    Zero-energy radial equation u'' = (2 mu / hbar^2) V u, integrated outwards;
    beyond the range of V, u = C (r - a).
    """
    mu = {-1: M_N / 2, 0: M_N * M_P / (M_N + M_P), 1: M_P / 2}[MT]
    k = HBARC**2 / (2 * mu)
    central, _ = channel_potential(v, 0, 0, 0, 1, MT)
    from scipy.interpolate import CubicSpline
    V = CubicSpline(r, central)

    def rhs(x, y):
        return [y[1], V(x) / k * y[0]]

    sol = solve_ivp(rhs, (r[0], r[-1]), [0.0, 1.0], rtol=1e-11, atol=1e-13, method="DOP853")
    u, du = sol.y[0, -1], sol.y[1, -1]
    return r[-1] - u / du
