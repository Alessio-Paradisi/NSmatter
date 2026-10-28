"""NV2 two-nucleon checks against the published fits (strong interaction only, no EM).

Deuteron and 1S0 nn scattering length of model b. EM terms (absent here) shift
B_d by ~0.02 MeV and a_nn by ~+0.3 fm, so the tolerances are set accordingly.
"""

import numpy as np
import pytest

from nsmatter.potentials import nv2, two_body as tb

R = np.linspace(0.0, 25.0, 5001)


@pytest.fixture(scope="module")
def model_b():
    return nv2.nv2_radial(R, "b")


def test_deuteron(model_b):
    """B_d = 2.2246 MeV, P_D ~ 5-6 %."""
    B, PD = tb.deuteron(model_b, R)
    assert abs(B - 2.2246) < 0.04
    assert 0.04 < PD < 0.07


def test_nn_scattering_length(model_b):
    """a_nn = -18.9 fm with EM; strong-only about -19.2."""
    a = tb.scattering_length_1s0(model_b, R, MT=-1)
    assert -19.6 < a < -18.8


def test_long_range_tail(model_b):
    """Far from the regulators only OPE survives: v_st -> Y_pi, v_s, v_t ~ 0."""
    k = np.searchsorted(R, 8.0)
    assert abs(model_b["c"][k]) < 1e-3 * abs(model_b["st"][k]) + 1e-6
    assert model_b["st"][k] > 0
