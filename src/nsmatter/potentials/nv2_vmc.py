"""NV2 as a VMC potential term: spline tables of the radial functions + operator terms.

Model keys (thesis names): "a" = NV2-Ia, "b" = NV2-Ib, "at" = NV2-IIa, "bt" = NV2-IIb
(also "c", "ct"). See nv2.py for the radial functions and their validation.

Each channel is tabulated once on a uniform grid r = 0, dr, ..., and cut where
|v(r)| stays below `tol` (MeV) for good; only one-pion exchange (st, tnt, tnT)
reaches beyond ~5 fm. Tensor channels are stored as w = v_T / r^2 (radial.py).
Static channels go to pair_operators.make_static_potential, momentum-dependent
ones (L.S, L^2, (L.S)^2 forms) to momentum.make_momentum_potential.
"""

import numpy as np

from ..hamiltonian import momentum, pair_operators
from ..hamiltonian.radial import Table, tensor_table
from ..system import System
from .nv2 import MODELS, nv2_radial

THESIS_NAMES = {"Ia": "a", "Ib": "b", "IIa": "at", "IIb": "bt"}
STATIC = tuple(pair_operators.CHANNELS)  # c s tn t st tnt T sT tnT tz
MOMENTUM = tuple(ch for ch in momentum.CHANNELS if ch in ("ls", "lst", "lsT", "l2", "l2s", "ls2"))


def model_key(model):
    """Accept "Ib" or "b" style names."""
    key = THESIS_NAMES.get(model, model)
    if key not in MODELS:
        raise ValueError(f"unknown NV2 model {model!r}; use one of {list(THESIS_NAMES)} or {list(MODELS)}")
    return key


def nv2_tables(model, dr=0.005, r_end=15.0, tol=1e-3):
    """Dict channel -> Table (tensor channels as w = v_T / r^2), each cut at its own range."""
    r = np.arange(0.0, r_end + dr / 2, dr)
    v = nv2_radial(r, model_key(model))
    tables = {}
    for ch in STATIC + MOMENTUM:
        big = np.nonzero(np.abs(v[ch]) > tol)[0]
        last = big[-1] if len(big) else 0
        n = min(len(r), max(last + 2, 4))  # one point past the last |v| > tol
        rr, vv = r[:n], v[ch][:n].copy()
        vv[-1] = 0.0  # end the spline at zero: no jump at r_max
        if pair_operators.CHANNELS.get(ch, ("",))[0] == "S12":
            tables[ch] = tensor_table(rr, vv)
        else:
            tables[ch] = Table(rr, vv)
    return tables


def make_nv2_potential(system: System, model, tables=None, momentum_terms=True):
    """Single potential term (log_psi, params, r, s, t) -> complex with the full NV2.

    momentum_terms=False keeps only the static channels (much cheaper; for tests
    and quick looks, not for physics).
    """
    tables = nv2_tables(model) if tables is None else tables
    static = pair_operators.make_static_potential(system, {ch: tables[ch] for ch in STATIC})
    if not momentum_terms:
        return static
    mom = momentum.make_momentum_potential(system, {ch: tables[ch] for ch in MOMENTUM})

    def term(log_psi, params, r, s, t):
        return static(log_psi, params, r, s, t) + mom(log_psi, params, r, s, t)

    return term
