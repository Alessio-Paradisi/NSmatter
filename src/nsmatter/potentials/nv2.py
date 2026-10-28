"""Radial functions of the local chiral potentials NV2 (Norfolk models).

References
    [P15] M. Piarulli et al., Phys. Rev. C 91, 024003 (2015): long-range part
          (Appendix A), short-range radial functions (Appendix B), constants (Tables I, II).
    [P16] M. Piarulli et al., Phys. Rev. C 94, 054007 (2016): local operator
          structure (Eqs. 1-7) and the LECs of the local models (Tables II, III).

Models: (R_L, R_S) = a (1.2, 0.8), b (1.0, 0.7), c (0.8, 0.6) fm, fitted up to
125 MeV; "at", "bt", "ct" (a-tilde, ...) fitted up to 200 MeV. In the usual names:
NV2-Ia = a, NV2-Ib = b, NV2-IIa = at, NV2-IIb = bt.

The potential (strong part; the electromagnetic part of AV18 is not included):
    v12 = v_L + v_S
    v_L = sum_{l=1..6} v_L^l O^l + v_L^{sigma T} sigma.sigma T12 + v_L^{tT} S12 T12
    v_S = sum_{l=1..16} v_S^l O^l
    O^{1..6}  = [1, sigma.sigma, S12] x [1, tau.tau]
    O^{7..11} = L.S, L.S tau.tau, (L.S)^2, L^2, L^2 sigma.sigma
    O^{12..16}= T12, tau1z + tau2z, sigma.sigma T12, S12 T12, L.S T12
with T12 = 3 tau1z tau2z - tau1.tau2.

This module is setup code (NumPy/SciPy): it evaluates the radial functions on
a grid, in MeV, with r in fm. All functions are returned as a dict keyed by
the channel names used by the simulation:
    c, t (tau.tau), s (sigma.sigma), st, tn (S12), tnt (S12 tau.tau),
    ls, lst, ls2, l2, l2s, T, tz, sT, tnT, lsT
Tensor channels (tn, tnt, tnT) hold v_T(r) itself; convert to v_T / r^2 with
radial.tensor_table when building the simulation tables.
"""

import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.special import k0, k1

HBARC = 197.32697  # MeV fm, as in [P15]

# --- constants of [P15], Tables I and II -------------------------------------
G_A = 1.29
H_A = 3 * G_A / np.sqrt(2)  # = 2.7365 (Table I of [P15] rounds it to 2.74; the reference code uses the exact value)
F_PI = 184.80  # MeV (F_pi = 2 f_pi)
C1, C2, C3, C4 = -0.57e-3, -0.25e-3, -0.79e-3, 1.33e-3  # GeV^-1 -> MeV^-1
B3_B8 = 1.40e-3  # MeV^-1
M_PI0 = 134.9766
M_PIC = 139.5702
M_PI = (2 * M_PIC + M_PI0) / 3  # average, used in the TPE terms
M_N, M_P = 939.56524, 938.27192
DELTA_M = 293.1

MODELS = {"a": (1.2, 0.8), "b": (1.0, 0.7), "c": (0.8, 0.6), "at": (1.2, 0.8), "bt": (1.0, 0.7), "ct": (0.8, 0.6)}

# --- LECs of [P16], Tables II and III (C in fm^2 or fm^4, D in fm^6) ----------
_LEC_NAMES = ["CS", "CT", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
              "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
              "C0IV", "C0IT", "C1IT", "C2IT", "C3IT", "C4IT"]
_LEC_VALUES = {
    "a": [0.2726141e1, -0.5228448, -0.6992838e-1, -0.1496013, -0.2502401e-1, -0.2728396e-1,
          -0.6530008e-2, -0.7554924e-1, -0.1017206e1,
          -0.4251199e-1, -0.5567938e-2, -0.1666607e-1, 0.1054347e-2, 0.5383828e-2, -0.8012050e-2,
          -0.2309392e-1, 0.1383136e-1, 0.4797012e-1, -0.1156876, -0.1453295e-1,
          0.9325477e-2, 0.1578240e-1, -0.2179452e-2, -0.6288540e-2, -0.5799803e-2, 0.2250167e-1],
    "b": [0.8038124e1, -0.1203741e1, -0.2280422, -0.2249889, -0.4007665e-1, 0.1243960e-1,
          -0.1870727e-1, -0.7406609e-1, -0.1197452e1,
          -0.3820959e-1, -0.5343034e-2, -0.1601394e-1, 0.4219347e-2, 0.8971752e-2, -0.5986245e-2,
          -0.6180197e-2, 0.1782567e-1, 0.3094851e-1, -0.8073891e-1, -0.1162060e-1,
          0.1018989e-1, 0.2416591e-1, -0.3707396e-2, -0.3601899e-2, -0.4559006e-2, 0.1859997e-1],
    "c": [0.1858356e2, -0.6118406e1, -0.5624246, -0.3529711, -0.2225345, 0.3381613e-1,
          -0.2881762e-1, -0.6535759e-1, -0.1464748e1,
          -0.2163208e-1, 0.2866318e-2, -0.1472287e-1, 0.1052796e-2, 0.7477159e-2, -0.2247046e-2,
          0.3616700e-1, 0.2903320e-1, 0.9175910e-1, -0.1229688, -0.2671576e-1,
          0.1357818e-1, 0.2195881e-1, -0.2698274e-2, -0.1288174e-2, -0.3126089e-3, 0.8987538e-2],
    "at": [0.2936041e1, -0.4933897, -0.1013462, -0.1444844, -0.3647634e-1, -0.1630825e-1,
           -0.6658100e-2, -0.6176835e-1, -0.9578191,
           -0.3102824e-1, -0.4438695e-2, -0.1351171e-1, -0.7084459e-3, 0.1110108e-1, -0.8598857e-2,
           -0.5367908e-1, 0.3119241e-1, 0.3281636e-1, -0.8647128e-1, -0.1167788e-1,
           0.9575695e-2, 0.2194758e-1, -0.1550501e-2, -0.8354679e-2, -0.6682746e-2, 0.1276971e-1],
    "bt": [0.8398499e1, -0.1207696e1, -0.2324413, -0.2108143, -0.3461629e-1, 0.8748772e-2,
           -0.3614304e-1, -0.5542581e-1, -0.1019849e1,
           -0.1193597e-1, -0.4450346e-2, -0.9542801e-2, 0.3976205e-2, 0.7809205e-2, -0.7362895e-2,
           -0.4158494e-2, 0.1090986e-1, 0.6095858e-3, -0.5432144e-1, -0.5186422e-2,
           0.1077541e-1, 0.2102140e-1, 0.1152693e-3, -0.1391786e-2, -0.3194459e-3, 0.2879873e-2],
    "ct": [0.1858331e2, -0.6116424e1, -0.5565484, -0.3574422, -0.2266117, 0.3921168e-1,
           -0.2661419e-1, -0.6532432e-1, -0.1465875e1,
           -0.2144023e-1, 0.1386494e-2, -0.1620926e-1, 0.2071219e-2, 0.7238077e-2, -0.2323562e-2,
           0.3065351e-1, 0.2957488e-1, 0.9135194e-1, -0.1196465, -0.3065569e-1,
           0.1312712e-1, 0.1394723e-1, -0.8965197e-2, -0.3079018e-2, 0.3905867e-4, 0.8844043e-3],
}
LECS = {m: dict(zip(_LEC_NAMES, v)) for m, v in _LEC_VALUES.items()}


# =============================================================================
# Long-range part: one- and two-pion exchange, [P15] Appendix A
# =============================================================================
def _mu_integrals(x, y):
    """The parametric mu-integrals of Eqs. (A9)-(A34), for arrays x, y (one per r).

    With s = sqrt(mu^2 + 4 x^2) and E = exp(-s):
        I[f] = int_0^inf dmu mu^2 / s E f     ("first" kind)
        J[g] = int_0^inf dmu mu   / s E g arctan(mu / 2y)   ("second" kind)
    Returns a dict name -> array over r.
    """
    x2, y2 = x**2, y**2

    def integrand(mu):
        s = np.sqrt(mu**2 + 4 * x2)
        E = np.exp(-s)
        a = mu**2 / s * E  # weight of the I integrals
        b = mu / s * E * np.arctan(mu / (2 * y))  # weight of the J integrals
        m2 = mu**2
        P = m2 + 4 * x2  # mu^2 + 4x^2
        Q = 3 + 3 * s + m2 + 4 * x2  # 3 + 3s + mu^2 + 4x^2
        A = 2 * x2 + m2 + 2 * y2  # 2x^2 + mu^2 + 2y^2
        out = [
            a * (12 * x2 + 5 * m2 + 12 * y2),  # 0  I: A9 first
            b * A,  # 1  J: A9 (times -12y)
            a * (24 * x2 + 11 * m2 + 12 * y2),  # 2  I: A9 second
            b * A**2,  # 3  J: A9 second, A24 second
            a * P,  # 4  I: (mu^2+4x^2)
            b * P * (m2 + 4 * y2),  # 5  J
            a * Q,  # 6  I
            b * (m2 + 4 * y2) * Q,  # 7  J
            a * (4 * y2 + 2 * A**2 / (m2 + 4 * y2)),  # 8  I: A14
            b * A * (2 * x2 + m2 - 6 * y2),  # 9  J: A14
            a * ((24 * x2 + 11 * m2 + 24 * y2) + 6 * A**2 / (m2 + 4 * y2)),  # 10 I: A15
            b * A * (2 * x2 + m2 + 10 * y2),  # 11 J: A15, A29, A30
            b * P * (m2 + 12 * y2),  # 12 J: A16, A31, A32
            b * P * (-m2 + 4 * y2),  # 13 J: A17
            b * Q * (m2 + 12 * y2),  # 14 J: A18, A33, A34
            b * Q * (-m2 + 4 * y2),  # 15 J: A19
            a * (-24 * C1 * x2 + C2 * (5 * m2 + 12 * x2 + 12 * y2) - 6 * C3 * (m2 + 2 * x2)),  # 16 I: A23
            b * A * (4 * C1 * x2 - 2 * C2 * y2 + C3 * (m2 + 2 * x2)),  # 17 J: A23
            a * ((24 * x2 + 11 * m2 + 12 * y2) + 6 * A**2 / (m2 + 4 * y2)),  # 18 I: A29, A30
        ]
        return np.stack(out)

    vals, _ = quad_vec(integrand, 0.0, np.inf, epsrel=1e-11, epsabs=0.0, limit=500)
    return vals


def regulator_long(r, R_L):
    """C_RL(r) = 1 - 1/[(r/R_L)^6 exp((r - R_L)/a_L) + 1], a_L = R_L / 2,
    written as z / (1 + z) to avoid cancellation at small r."""
    z = (r / R_L) ** 6 * np.exp((r - R_L) / (R_L / 2))
    return z / (1 + z)


def long_range(r, R_L):
    """Regularized v_L^l(r) in MeV for r > 0 (fm). Keys: c, t, s, st, tn, tnt, sT, tnT."""
    r = np.asarray(r, dtype=float)
    rn = r / HBARC  # r in MeV^-1
    gA2, gA4, hA2, hA4 = G_A**2, G_A**4, H_A**2, H_A**4
    F4 = F_PI**4
    pi, pi2, pi3 = np.pi, np.pi**2, np.pi**3

    # --- LO, one-pion exchange (A1-A4, A41, A42) ---
    def Y(m):
        xa = m * rn
        return gA2 / (12 * pi) * m**3 / F_PI**2 * np.exp(-xa) / xa

    def T(m):
        xa = m * rn
        return Y(m) * (1 + 3 / xa + 3 / xa**2)

    Y0, Yc, T0, Tc = Y(M_PI0), Y(M_PIC), T(M_PI0), T(M_PIC)
    lo_st = (Y0 + 2 * Yc) / 3
    lo_tnt = (T0 + 2 * Tc) / 3

    # --- TPE ---
    x = M_PI * rn
    y = DELTA_M * rn
    e2x = np.exp(-2 * x)
    K0, K1 = k0(2 * x), k1(2 * x)
    I = _mu_integrals(x, y)

    # NLO without Delta (A5-A7)
    nlo_t_nd = 1 / (8 * pi3 * rn**4) * M_PI / F4 * (
        x * (1 + 10 * gA2 - gA4 * (23 + 4 * x**2)) * K0
        + (1 + 2 * gA2 * (5 + 2 * x**2) - gA4 * (23 + 12 * x**2)) * K1)
    nlo_s_nd = 1 / (2 * pi3 * rn**4) * gA4 / F4 * M_PI * (3 * x * K0 + (3 + 2 * x**2) * K1)
    nlo_tn_nd = -1 / (8 * pi3 * rn**4) * gA4 / F4 * M_PI * (12 * x * K0 + (15 + 4 * x**2) * K1)

    # NLO, one Delta (A8-A13)
    r5 = rn**5
    nlo_c_d = -1 / (6 * pi2 * r5 * y) * gA2 * hA2 / F4 * e2x * (6 + 12 * x + 10 * x**2 + 4 * x**3 + x**4)
    nlo_t_d = (-1 / (216 * pi3 * r5) * hA2 / F4 * (I[0] - 12 * y * I[1])
               - 1 / (216 * pi3 * r5) * gA2 * hA2 / F4 * (-I[2] + 6 / y * I[3]))
    nlo_s_d = -1 / (72 * pi3 * r5) * gA2 * hA2 / F4 * (2 * I[4] - I[5] / y)
    # (A11) and (A13): the polynomials (3 + 3x + x^2) and (3 + 3x + 2x^2) are printed
    # swapped in [P15]. The reference implementation and the heavy-Delta limit (which
    # must reproduce the c4 terms (A21), (A22)) give the assignment used here.
    nlo_st_d = 1 / (54 * pi2 * r5 * y) * gA2 * hA2 / F4 * e2x * (1 + x) * (3 + 3 * x + 2 * x**2)
    nlo_tn_d = 1 / (144 * pi3 * r5) * gA2 * hA2 / F4 * (2 * I[6] - I[7] / y)
    nlo_tnt_d = -1 / (54 * pi2 * r5 * y) * gA2 * hA2 / F4 * e2x * (1 + x) * (3 + 3 * x + x**2)

    # NLO, two Deltas (A14-A19)
    nlo_c_dd = -1 / (108 * pi3 * r5) * hA4 / F4 * (I[8] + I[9] / y)
    nlo_t_dd = -1 / (1944 * pi3 * r5) * hA4 / F4 * (I[10] - 3 / y * I[11])
    nlo_s_dd = -1 / (1296 * pi3 * r5) * hA4 / F4 * (-6 * I[4] + I[12] / y)
    nlo_st_dd = -1 / (7776 * pi3 * r5) * hA4 / F4 * (-2 * I[4] + I[13] / y)
    nlo_tn_dd = 1 / (2592 * pi3 * r5) * hA4 / F4 * (-6 * I[6] + I[14] / y)
    nlo_tnt_dd = 1 / (15552 * pi3 * r5) * hA4 / F4 * (-2 * I[6] + I[15] / y)

    # N2LO without Delta (A20-A22)
    r6 = rn**6
    n2_c_nd = 3 / (2 * pi2 * r6) * gA2 / F4 * e2x * (
        2 * C1 * x**2 * (1 + x) ** 2 + C3 * (6 + 12 * x + 10 * x**2 + 4 * x**3 + x**4))
    n2_st_nd = 1 / (3 * pi2 * r6) * gA2 / F4 * C4 * e2x * (1 + x) * (3 + 3 * x + 2 * x**2)
    n2_tnt_nd = -1 / (3 * pi2 * r6) * gA2 / F4 * C4 * e2x * (1 + x) * (3 + 3 * x + x**2)

    # N2LO, one Delta (A23-A28)
    n2_c_d = 1 / (18 * pi3 * r6) * hA2 * y / F4 * (I[16] + 6 / y * I[17])
    n2_t_d = (-1 / (54 * pi3 * r6) * B3_B8 * H_A * y / F4 * (I[0] - 12 * y * I[1])
              - 1 / (54 * pi3 * r6) * B3_B8 * H_A * gA2 * y / F4 * (-I[2] + 6 / y * I[3]))
    n2_s_d = -1 / (18 * pi3 * r6) * B3_B8 * H_A * gA2 * y / F4 * (2 * I[4] - I[5] / y)
    n2_st_d = -1 / (108 * pi3 * r6) * C4 * hA2 * y / F4 * (2 * I[4] - I[5] / y)
    n2_tn_d = 1 / (36 * pi3 * r6) * B3_B8 * H_A * gA2 * y / F4 * (2 * I[6] - I[7] / y)
    n2_tnt_d = 1 / (216 * pi3 * r6) * C4 * hA2 * y / F4 * (2 * I[6] - I[7] / y)

    # N2LO, two Deltas (A29-A34)
    hA3 = H_A**3
    n2_c_dd = -2 / (81 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (I[18] - 3 / y * I[11])
    n2_t_dd = -1 / (243 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (I[18] - 3 / y * I[11])
    n2_s_dd = -1 / (162 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (-6 * I[4] + I[12] / y)
    n2_st_dd = -1 / (972 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (-6 * I[4] + I[12] / y)
    n2_tn_dd = 1 / (324 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (-6 * I[6] + I[14] / y)
    n2_tnt_dd = 1 / (1944 * pi3 * r6) * B3_B8 * hA3 * y / F4 * (-6 * I[6] + I[14] / y)

    # Sums (A35-A40), CD part (A41-A42), regulator (A43)
    C = regulator_long(r, R_L)
    out = {
        "c": nlo_c_d + nlo_c_dd + n2_c_nd + n2_c_d + n2_c_dd,
        "t": nlo_t_nd + nlo_t_d + nlo_t_dd + n2_t_d + n2_t_dd,
        "s": nlo_s_nd + nlo_s_d + nlo_s_dd + n2_s_d + n2_s_dd,
        "st": lo_st + nlo_st_d + nlo_st_dd + n2_st_nd + n2_st_d + n2_st_dd,
        "tn": nlo_tn_nd + nlo_tn_d + nlo_tn_dd + n2_tn_d + n2_tn_dd,
        "tnt": lo_tnt + nlo_tnt_d + nlo_tnt_dd + n2_tnt_nd + n2_tnt_d + n2_tnt_dd,
        "sT": (Y0 - Yc) / 3,
        "tnT": (T0 - Tc) / 3,
    }
    return {k: C * v for k, v in out.items()}


# =============================================================================
# Short-range part: contact terms, [P15] Appendix B with [P16] operators/LECs
# =============================================================================
def _short_range_functions(R_S):
    """Exact (sympy) radial building blocks of Eqs. (B11)-(B33), as numpy functions of r (fm).

    Returns dict name -> callable(r) giving the bracket that multiplies each LEC, in fm^-(3+n).
    Simplifying symbolically removes the apparent 1/r^n singularities, so every
    function is finite at r = 0.
    """
    r = sp.symbols("r", positive=True)
    R = sp.Rational(str(R_S))
    Cr = sp.exp(-(r / R) ** 2) / (sp.pi ** sp.Rational(3, 2) * R**3)
    d = [Cr] + [sp.diff(Cr, r, n) for n in range(1, 5)]
    k2 = -d[2] - 2 / r * d[1]  # k^2 -> (B3)
    k4 = d[4] + 4 / r * d[3]  # k^4 -> (B4)
    blocks = {
        "C": d[0],
        "k2": k2,
        "k4": k4,
        "S12": d[2] - d[1] / r,  # bracket of (B5), (B15)-(B16), (B24)-(B25), (B30)-(B31)
        "S12k2": d[4] + d[3] / r - 6 / r**2 * d[2] + 6 / r**3 * d[1],  # (B15)-(B16), D terms
        "LS": d[1] / r,  # (B6), (B17), (B32)-(B33)
        "LSk2": d[3] / r + 2 / r**2 * d[2] - 2 / r**3 * d[1],  # (B17)-(B18), D terms
        "L2": (d[2] - d[1] / r) / r**2,  # (B19)-(B21)
    }
    return {k: sp.lambdify(r, sp.simplify(v), "numpy") for k, v in blocks.items()}


def short_range(r, model):
    """v_S^l(r) in MeV (r in fm), channels c, t, s, st, tn, tnt, ls, lst, ls2, l2, l2s, T, tz, sT, tnT, lsT."""
    R_S = MODELS[model][1]
    L = LECS[model]
    f = {k: (lambda g: (lambda x: np.broadcast_to(g(x), np.shape(x)).astype(float)))(g)
         for k, g in _short_range_functions(R_S).items()}
    r = np.asarray(r, dtype=float)
    B = {k: g(r) for k, g in f.items()}
    v = {
        "c": L["CS"] * B["C"] + L["C1"] * B["k2"] + L["D1"] * B["k4"],  # (B11)
        "t": L["C2"] * B["k2"] + L["D2"] * B["k4"],  # (B12)
        "s": L["CT"] * B["C"] + L["C3"] * B["k2"] + L["D3"] * B["k4"],  # (B13)
        "st": L["C4"] * B["k2"] + L["D4"] * B["k4"],  # (B14)
        "tn": -L["C5"] * B["S12"] + L["D5"] * B["S12k2"],  # (B15)
        "tnt": -L["C6"] * B["S12"] + L["D6"] * B["S12k2"],  # (B16)
        "ls": -L["C7"] * B["LS"] + L["D7"] * B["LSk2"],  # (B17)
        "lst": L["D8"] * B["LSk2"],  # (B18)
        "ls2": -L["D9"] * B["L2"],  # (B19)
        "l2": -L["D10"] * B["L2"],  # (B20)
        "l2s": -L["D11"] * B["L2"],  # (B21)
        "T": L["C0IT"] * B["C"] + L["C1IT"] * B["k2"],  # (B26)
        "tz": L["C0IV"] * B["C"],  # (B27), LO only in [P16]
        "sT": L["C2IT"] * B["k2"],  # (B28)
        "tnT": -L["C3IT"] * B["S12"],  # (B30)
        "lsT": -L["C4IT"] * B["LS"],  # (B32)
    }
    # LECs in fm^k times functions in fm^-(k+1): fm^-1 -> MeV.
    return {k: HBARC * val for k, val in v.items()}


# =============================================================================
# Full potential on a grid
# =============================================================================
CHANNELS = ("c", "t", "s", "st", "tn", "tnt", "ls", "lst", "ls2", "l2", "l2s", "T", "tz", "sT", "tnT", "lsT")


def nv2_radial(r, model):
    """All radial functions v^l(r) of NV2 model `model` on the grid r (fm), in MeV.

    The long-range part is singular at r = 0 before regularization; at r = 0 it
    is set by quadratic extrapolation from the next three grid points.
    """
    r = np.asarray(r, dtype=float)
    R_L = MODELS[model][0]
    pos = r > 0
    lr = long_range(r[pos], R_L)
    sr = short_range(r, model)
    out = {}
    for ch in CHANNELS:
        v = np.array(sr[ch], dtype=float)
        if ch in lr:
            full = np.zeros_like(r)
            full[pos] = lr[ch]
            if not pos.all():  # r = 0 is the first point
                full[0] = 3 * full[1] - 3 * full[2] + full[3]
            v = v + full
        out[ch] = v
    return out
