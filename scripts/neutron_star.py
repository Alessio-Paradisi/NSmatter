"""From VMC energies to neutron stars: EOS fit, mass-radius curve, tidal deformability.

Reads results/nv2_vmc.jsonl (written by run_nv2.py / eos_scan.py), and for each model:
  1. fits E/N(rho) of pure neutron matter (and of symmetric matter, for --beta);
  2. builds the EOS: pure neutron matter, or n p e mu in beta equilibrium (--beta);
  3. solves TOV + tidal equations: M(R), M_max, R_1.4, Lambda_1.4;
  4. propagates the VMC error bars by a parametric bootstrap (refit on
     E_i + err_i * gaussian noise, --bootstrap samples).
Outputs: a summary table on screen, results/summary/neutron_star.json, and the figure
results/figures/neutron_star.pdf (E/N, P(eps), M(R); --format png for a bitmap).

Comparison data (e.g. your BHF results) as text files with columns rho E/N [err]:
    --compare "BHF Ib:bhf/pnm_Ib.dat" "BHF IIb:bhf/pnm_IIb.dat"
Symmetric matter for --beta, if not computed with VMC (populations 7 7 7 7):
    --snm-file bhf/snm_Ib.dat

Two-body NV2 only (no three-nucleon force): expect stars that are too soft and too
small compared with observations; this is the baseline that NV3 will correct.

Usage (from ~/projects/nsmatter, venv active):
    python scripts/neutron_star.py --models Ib IIb
    python scripts/neutron_star.py --models Ib --beta --snm-file bhf/snm_Ib.dat
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nsmatter.eos import RHO0, beta_equilibrium, causal_cap, eos_table, fit_pnm, fit_snm, thermodynamics  # noqa: E402
from nsmatter.tov import TabulatedEOS, mass_radius  # noqa: E402


def load_points(path, model, populations, three_body=None, wavefunction="basic"):
    """{density: (E/N, error)} for one model and population set (last run wins).

    three_body: None for NV2-only runs, or the NV3 model (e.g. "Ib*") for NV2+3 runs.
    """
    pts = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec["model"] == model and tuple(rec["populations"]) == populations and not rec.get("static_only")
                and rec.get("three_body") == three_body and rec.get("wavefunction", "basic") == wavefunction):
            pts[round(rec["density"], 6)] = (rec["E_per_N"], rec["error"])
    rho = np.array(sorted(pts))
    E = np.array([pts[r][0] for r in rho])
    err = np.array([max(pts[r][1], 1e-3) for r in rho])
    return rho, E, err


def load_table(path):
    d = np.loadtxt(path, ndmin=2)
    err = d[:, 2] if d.shape[1] > 2 else np.full(len(d), 0.1)
    return d[:, 0], d[:, 1], err


def star_from_fits(pnm, snm, beta, cap=True):
    """M(R) sequence for the fitted EOS, plus the central density of the heaviest star."""
    rho_grid = np.geomspace(1e-4, 1.2, 400)
    table = beta_equilibrium(pnm, snm, rho_grid) if beta else thermodynamics(pnm, rho_grid)
    eps, P = eos_table(table)
    start = int(np.searchsorted(table["eps"], eps[0]))
    rho = rho_grid[start:start + len(eps)]
    eps_t = None
    if cap:
        eps, P, eps_t = causal_cap(eps, P)
        if eps_t is not None:
            # beyond the cap, rho follows from d ln rho = d eps / (eps + P)
            k = int(np.searchsorted(eps, eps_t))
            dlog = np.diff(eps[k:]) / (0.5 * (eps[k:-1] + eps[k + 1:]) + 0.5 * (P[k:-1] + P[k + 1:]))
            rho = rho.copy()
            rho[k + 1:] = rho[k] * np.exp(np.cumsum(dlog))
    star = mass_radius(TabulatedEOS(eps, P))
    star["rho_c_max"] = float(np.exp(np.interp(np.log(star["p_c_max"]), np.log(P), np.log(rho))))
    star["capped"] = eps_t is not None and star["p_c_max"] > float(np.interp(eps_t, eps, P))
    return star


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/nv2_vmc.jsonl")
    p.add_argument("--models", nargs="+", default=["Ia", "Ib", "IIa", "IIb"])
    p.add_argument("--beta", action="store_true", help="n p e mu in beta equilibrium (needs SNM)")
    p.add_argument("--three-body", nargs="*", default=None,
                   help="NV3 model per --models entry (e.g. Ia* Ib*) to use NV2+3 runs; default NV2 only")
    p.add_argument("--snm-file", default=None, help="SNM E/N table if not from VMC")
    p.add_argument("--compare", nargs="*", default=[], help='"label:file" with columns rho E/N [err]')
    p.add_argument("--bootstrap", type=int, default=50)
    p.add_argument("--wavefunction", default="basic", choices=("basic", "LS"),
                   help="runs with the basic trial wavefunction or with spin-orbit correlations")
    p.add_argument("--no-causal-cap", action="store_true", help="keep the fit even where c_s^2 > 1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="neutron_star", help="name of the figure and the summary file")
    p.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    summary = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tb_models = a.three_body if a.three_body else [None] * len(a.models)
    if len(tb_models) != len(a.models):
        raise SystemExit("--three-body needs one NV3 model per --models entry")
    for c, model, tb in zip(colors, a.models, tb_models):
        rho, E, err = load_points(a.results, model, (7, 7, 0, 0), tb, a.wavefunction)
        label = f"NV2-{model}" + (f"+3-{tb}" if tb else "")
        if len(rho) < 5:
            print(f"{label}: only {len(rho)} PNM points (need >= 5 for the 4-parameter fit), skipped")
            continue
        pnm, _ = fit_pnm(rho, E, err)
        snm = None
        if a.beta:
            if a.snm_file:
                rs, Es, es = load_table(a.snm_file)
            else:
                rs, Es, es = load_points(a.results, model, (7, 7, 7, 7))
            if len(rs) < 3:
                print(f"{label}: --beta needs >= 3 SNM points, skipped")
                continue
            snm, _ = fit_snm(rs, Es, es)

        star = star_from_fits(pnm, snm, a.beta, cap=not a.no_causal_cap)
        boots = []
        for _ in range(a.bootstrap):
            try:
                pb, _ = fit_pnm(rho, E + err * rng.standard_normal(len(E)), err, p0=pnm.params)
                sb = None
                if a.beta:
                    sb, _ = fit_snm(rs, Es + es * rng.standard_normal(len(Es)), es, p0=snm.params)
                boots.append(star_from_fits(pb, sb, a.beta, cap=not a.no_causal_cap))
            except (RuntimeError, ValueError):
                continue  # failed refit or no star: dropped from the band
        keys = ("M_max", "R_at_Mmax", "R_1.4", "Lambda_1.4", "rho_c_max")
        spread = {k: float(np.nanstd([b[k] for b in boots])) if boots else np.nan for k in keys}
        th0 = thermodynamics(pnm, np.array([RHO0]))
        acausal = thermodynamics(pnm, np.linspace(0.05, 1.2, 500))
        rho_acausal = acausal["rho"][np.argmax(acausal["cs2"] > 1)] if np.any(acausal["cs2"] > 1) else None
        summary[label] = {
            "pnm_fit": dict(zip(("a", "alpha", "b", "beta"), pnm.params.tolist())),
            "snm_fit": dict(zip(("E0", "K", "Q"), snm.params.tolist())) if snm else None,
            "E_pnm_rho0": float(th0["E"][0]), "P_pnm_rho0": float(th0["P"][0]),
            "rho_cs2_gt_1": rho_acausal, "rho_max_computed": float(rho.max()), "M_max_uses_cap": bool(star["capped"]),
            **{k: float(star[k]) for k in keys}, **{k + "_err": spread[k] for k in keys},
        }

        # --- plots ----------------------------------------------------------------
        r_plot = np.linspace(0.02, max(0.4, rho.max() * 1.2), 200)
        ax[0].errorbar(rho, E, err, fmt="o", color=c, ms=4, label=f"VMC {label}")
        ax[0].plot(r_plot, pnm(r_plot)[0], color=c, lw=1)
        th = thermodynamics(pnm, r_plot)
        ax[1].plot(th["eps"], th["P"], color=c, label=label)
        for b in boots[:30]:
            ax[2].plot(b["R"], b["M"], color=c, alpha=0.08, lw=1)
        ax[2].plot(star["R"], star["M"], color=c, lw=2, label=label)

    for k, item in enumerate(a.compare):
        label, path = item.split(":", 1)
        r, e, s = load_table(path)
        ax[0].errorbar(r, e, s, fmt="s", mfc="none", color="k", alpha=0.4 + 0.6 * k / max(1, len(a.compare)),
                       ms=5, label=label)

    ax[0].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel="E/N (MeV)", title="neutron matter")
    ax[1].set(xlabel=r"$\varepsilon$ (MeV fm$^{-3}$)", ylabel="P (MeV fm$^{-3}$)", xscale="log", yscale="log",
              title="EOS" + (" (beta equilibrium)" if a.beta else " (PNM)") + "; curves: PNM fit")
    ax[2].axhline(2.0, color="gray", ls=":", lw=1)
    ax[2].set(xlabel="R (km)", ylabel=r"M (M$_\odot$)", xlim=(8, 16), ylim=(0, 2.6), title="mass-radius")
    for x in ax:
        x.legend(fontsize=8)
    fig.tight_layout()
    fig_path = pathlib.Path("results/figures") / f"{a.out}.{a.format}"
    sum_path = pathlib.Path("results/summary") / f"{a.out}.json"
    for path in (fig_path, sum_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    sum_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'model':14s} {'M_max':>12s} {'R(M_max)':>12s} {'R_1.4':>12s} {'Lambda_1.4':>14s}  E(rho0)  rho_c(Mmax)  cs2>1 at")
    for m, s in summary.items():
        f = lambda k, d=2: f"{s[k]:.{d}f}+-{s[k + '_err']:.{d}f}"  # noqa: E731
        acs = f"{s['rho_cs2_gt_1']:.2f}" if s["rho_cs2_gt_1"] else "-"
        print(f"{m:14s} {f('M_max'):>12s} {f('R_at_Mmax'):>12s} {f('R_1.4'):>12s} {f('Lambda_1.4', 0):>14s}"
              f"  {s['E_pnm_rho0']:6.2f}  {s['rho_c_max']:6.2f}       {acs}")
    print("(densities in fm^-3; beyond the largest computed density the EOS is the fit's extrapolation;")
    print(" where c_s^2 would exceed 1 the EOS is continued with c_s^2 = 1 unless --no-causal-cap)")
    print(f"\nsaved {fig_path} and {sum_path}")


if __name__ == "__main__":
    main()
