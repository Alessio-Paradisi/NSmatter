"""Load and select VMC run records (the JSON lines written by scripts/run_nv2.py).

    from nsmatter.results import load, select
    runs = load("results/vmc_Ib.jsonl")                        # list of dicts
    pnm = select(runs, model="Ib", populations=(7, 7, 0, 0))    # sorted by density
    rho = [r["density"] for r in pnm]; E = [r["E_per_N"] for r in pnm]
    T = [r["components"]["kinetic"][0] for r in pnm]            # [mean, error] per particle

Several files can be loaded at once: load("results/vmc_*.jsonl") (a glob pattern).
Older records (before the energy decomposition) have no "components", "structure"
or "configs"; component() returns NaN for them.
"""

import glob
import json
import pathlib

import numpy as np


def load(pattern):
    """All records of the files matching pattern (a path or a glob), in file order."""
    records = []
    for path in sorted(glob.glob(str(pattern))) or [pattern]:
        for line in pathlib.Path(path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_file"] = str(path)
                records.append(rec)
    return records


def select(records, model=None, populations=None, density=None, static_only=False, three_body=None,
           latest=True):
    """Records matching the filters, sorted by density.

    three_body: None selects NV2-only runs, a model name (e.g. "Ib*") NV2+3 runs with
    that NV3 model, "any" both. latest=True keeps only the last record of each
    (model, density, populations, three_body), so a repeated point replaces the older one.
    """
    out = []
    for rec in records:
        if model is not None and rec["model"] != model:
            continue
        if populations is not None and tuple(rec["populations"]) != tuple(populations):
            continue
        if density is not None and not np.isclose(rec["density"], density):
            continue
        if static_only is not None and bool(rec.get("static_only")) != static_only:
            continue
        if three_body != "any" and rec.get("three_body") != three_body:
            continue
        out.append(rec)
    if latest:
        last = {}
        for rec in out:
            last[(rec["model"], round(rec["density"], 6), tuple(rec["populations"]), rec.get("three_body"))] = rec
        out = list(last.values())
    return sorted(out, key=lambda r: r["density"])


def component(records, name):
    """(mean, error) arrays of one energy component per particle, e.g. "kinetic", "v_tn", "group_tensor"."""
    vals = [rec.get("components", {}).get(name, [np.nan, np.nan]) for rec in records]
    arr = np.array(vals, dtype=float).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def structure(record, quantity="g"):
    """r and {class: (values, errors)} for "g", "C_sigma" or "C_tau" of one record (None -> NaN)."""
    st = record["structure"]
    to = lambda x: np.array([np.nan if v is None else v for v in x], dtype=float)  # noqa: E731
    return np.array(st["r"]), {c: (to(st[quantity][c]), to(st[quantity + "_err"][c])) for c in st[quantity]}


def configs(record):
    """The saved configurations of a run (numpy NpzFile), or None."""
    path = record.get("configs")
    if path is None:
        return None
    p = pathlib.Path(path)
    if not p.exists():  # relative to the results file
        p = pathlib.Path(record["_file"]).parent / "configs" / p.name
    return np.load(p)
