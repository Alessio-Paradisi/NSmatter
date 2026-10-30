"""Complete report of the VMC neutron-matter results in one PDF.

Pages:
  1  overview        what was computed (models, densities, settings) and a table of all runs:
                     E/N, kinetic, potential, three-body energy (per particle) and quality flags
  2  EOS             E/N(rho) with fit, free gas and published benchmarks (Lovato et al. 2022);
                     fit residuals; pressure P(rho); speed of sound c_s^2(rho)
  3  decomposition   kinetic / potential vs rho, potential by operator group vs rho,
                     every channel at the density closest to rho0, table at rho0
  4  structure       g_nn(r) and <sigma.sigma>(r) at all densities, vs r and vs k_F r
  5  neutron star    M(R) with bootstrap band, M(rho_c), Lambda(M), table of star properties
  6  snapshots       3D walker configurations at the lowest density and near rho0
  7+ quality         summary table and one page per run (see below)
Each "variant" is an NV2 model with or without an NV3 force (e.g. Ib, Ib+3-Ib*); all
variants present in the results are compared on the same pages (pure neutron matter).

Finite-size correction (--fs-correct): E/N -> E/N - (T_FG,box - T_FG,inf), the shell
effect of the free Fermi gas of the same N in the same box (stored in every record);
shown on the EOS page in any case, used for the fit and the star with --fs-correct.

Quality pages, per run: optimization history and drift, measurement series with the
integrated autocorrelation time tau_int and the corrected error sigma/sqrt(n) sqrt(2 tau_int),
variance and acceptances, <S^2>.
Flags (also listed on the summary page):
  DRIFT   last two optimization quarters differ by > max(0.5 MeV, 3 sigma): not converged
  OFFSET  measured E/N differs from the end of the optimization by > max(1 MeV, 4 sigma)
  ACC     an acceptance outside [0.25, 0.75]: step sizes badly tuned
  TAU     tau_int > 2: the recorded error is underestimated (use the corrected one)
  FEW     fewer than 30 measurements or 20 iterations
  S2      <S^2> deviates by more than 1 from S(S+1)
  OLD     record without the measurement series (run with the earlier code)

Usage (from ~/projects/nsmatter, venv active):
    python scripts/report.py --results results/vmc_Ib.jsonl
    python scripts/report.py --results "results/vmc_*.jsonl" --bootstrap 30 --fs-correct
    python scripts/report.py --results results/vmc_Ib.jsonl --quality-only   # seconds: quality pages only
Output: results/reports/report.pdf (quality-only: results/reports/quality.pdf), or --out.
The full report takes about a minute (bootstrap of the TOV solutions), longer while a scan runs.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.special import spherical_jn  # noqa: E402

from nsmatter.benchmarks import POINTS_RHO0, REFERENCE, afdmc_energy  # noqa: E402
from nsmatter.eos import RHO0, causal_cap, eos_table, fit_pnm, thermodynamics  # noqa: E402
from nsmatter.plotting import draw_walker  # noqa: E402
from nsmatter.potentials.nv2_vmc import GROUPS  # noqa: E402
from nsmatter.results import component, configs, load, select, structure  # noqa: E402
from nsmatter.stats import integrated_autocorrelation_time  # noqa: E402
from nsmatter.tov import TabulatedEOS, mass_radius  # noqa: E402

PNM = (7, 7, 0, 0)
A4 = (11.7, 8.3)


# =============================================================================
# data
# =============================================================================
class Variant:
    """Runs of one NV2 model with or without an NV3 force, with fit and stars."""

    def __init__(self, runs, model, three_body, fs_correct, n_boot, rng):
        self.model, self.three_body = model, three_body
        self.name = f"NV2-{model}" + (f"+3-{three_body}" if three_body else "")
        self.recs = select(runs, model=model, populations=PNM, three_body=three_body)
        self.rho = np.array([r["density"] for r in self.recs])
        self.E = np.array([r["E_per_N"] for r in self.recs])
        self.err = np.array([max(r["error"], 1e-3) for r in self.recs])
        self.fs_shift = np.array([r.get("free_gas_box", np.nan) - r.get("free_gas_infinite", np.nan)
                                  for r in self.recs])
        self.E_fit_data = self.E - np.nan_to_num(self.fs_shift) if fs_correct else self.E
        self.fit = self.star = None
        self.boots = []
        if len(self.rho) >= 5:
            self.fit, _ = fit_pnm(self.rho, self.E_fit_data, self.err)
            self.star = build_star(self.fit)
            for _ in range(n_boot):
                try:
                    f, _ = fit_pnm(self.rho, self.E_fit_data + self.err * rng.standard_normal(len(self.E)),
                                   self.err, p0=self.fit.params)
                    self.boots.append(build_star(f))
                except (RuntimeError, ValueError):
                    continue

    def spread(self, key):
        vals = [b[key] for b in self.boots if np.isfinite(b.get(key, np.nan))]
        return float(np.std(vals)) if len(vals) > 1 else np.nan


def build_star(fit):
    """M(R) sequence of pure neutron matter for a fit, with c_s^2 <= 1 continuation; adds rho_c per star."""
    rho_grid = np.geomspace(1e-4, 1.2, 400)
    th = thermodynamics(fit, rho_grid)
    eps, P = eos_table(th)
    start = int(np.searchsorted(th["eps"], eps[0]))
    rho = rho_grid[start:start + len(eps)].copy()
    eps, P, eps_t = causal_cap(eps, P)
    if eps_t is not None:
        k = int(np.searchsorted(eps, eps_t))
        dlog = np.diff(eps[k:]) / (0.5 * (eps[k:-1] + eps[k + 1:]) + 0.5 * (P[k:-1] + P[k + 1:]))
        rho[k + 1:] = rho[k] * np.exp(np.cumsum(dlog))
    star = mass_radius(TabulatedEOS(eps, P))
    star["rho_c"] = np.exp(np.interp(np.log(star["p_c"]), np.log(P), np.log(rho)))
    star["rho_c_max"] = float(star["rho_c"][int(np.argmax(star["M"]))])
    star["rho_cap"] = float(np.exp(np.interp(np.log(eps_t), np.log(eps), np.log(rho)))) if eps_t else np.nan
    return star


# =============================================================================
# pages
# =============================================================================
def page_overview(pdf, variants, runs, args):
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.94, "VMC neutron matter with the Norfolk (NV2 / NV2+3) interactions", fontsize=16, weight="bold")
    s = next((r.get("settings") for v in variants for r in v.recs if r.get("settings")), {}) or {}
    commit = next((r.get("commit") for v in variants for r in v.recs if r.get("commit")), None)
    lines = [
        f"variants: {', '.join(v.name for v in variants)}   (pure neutron matter, 7 up + 7 down neutrons, periodic box)",
        f"wavefunction: exp(J) [1 + sum U_ij] Phi (central Jastrow + spin-isospin-tensor correlations, plane-wave Slater)",
        f"settings: {s.get('walkers', '?')} walkers, {s.get('iterations', '?')} SR iterations, "
        f"{s.get('measurements', '?')} measurements, n_terms {s.get('n_terms', '?')}; code {commit or '?'}",
        f"finite-size correction in fits: {'yes' if args.fs_correct else 'no'};  bootstrap samples: {args.bootstrap}",
        "energies per particle in MeV; errors are statistical (VMC); E/N is a variational upper bound",
    ]
    for k, ln in enumerate(lines):
        fig.text(0.05, 0.88 - 0.035 * k, ln, fontsize=9.5)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.62])
    ax.axis("off")
    head = ["variant", "rho", "E/N", "kinetic", "potential", "three-body", "FG box - inf", "flags"]
    cells, bad = [], []
    for v in variants:
        for r in v.recs:
            c = r.get("components", {})
            f = lambda k: f"{c[k][0]:.2f} +- {c[k][1]:.2f}" if k in c else "-"  # noqa: E731
            flags = analyse(r)[0]["flags"]
            cells.append([v.name, f"{r['density']:.3f}", f"{r['E_per_N']:.3f} +- {r['error']:.3f}", f("kinetic"),
                          f("potential"), f("group_three-body"),
                          f"{r.get('free_gas_box', np.nan) - r.get('free_gas_infinite', np.nan):+.2f}",
                          ", ".join(flags) or "ok"])
            bad.append(bool([x for x in flags if x != "OLD"]))
    if cells:
        t = ax.table(cellText=cells, colLabels=head, loc="upper center", cellLoc="center",
                     colWidths=[0.14, 0.06, 0.13, 0.13, 0.13, 0.13, 0.1, 0.12])
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.3)
        for k, b in enumerate(bad, start=1):
            if b:
                for col in range(len(head)):
                    t[k, col].set_facecolor("#fde2e2")
    pdf.savefig(fig)
    plt.close(fig)


def page_eos(pdf, variants, colors):
    fig, ax = plt.subplots(2, 2, figsize=A4)
    rmax = max([v.rho.max() for v in variants if len(v.rho)] + [0.24])
    r_plot = np.linspace(0.01, max(0.5, 1.3 * rmax), 300)
    for v, c in zip(variants, colors):
        if not len(v.rho):
            continue
        ax[0, 0].errorbar(v.rho, v.E, v.err, fmt="o", color=c, ms=4, label=f"VMC {v.name}")
        if np.isfinite(v.fs_shift).any():
            ax[0, 0].plot(v.rho, v.E - v.fs_shift, "o", mfc="none", color=c, ms=5, label=f"{v.name}, FS-corrected")
        if v.fit is not None:
            ax[0, 0].plot(r_plot, v.fit(r_plot)[0], color=c, lw=1)
            ax[0, 1].errorbar(v.rho, (v.E_fit_data - v.fit(v.rho)[0]) / v.err, 1, fmt="o", color=c, label=v.name)
            th = thermodynamics(v.fit, r_plot)
            ax[1, 0].plot(r_plot, th["P"], color=c, label=v.name)
            ax[1, 1].plot(r_plot, th["cs2"], color=c, label=v.name)
        bench = afdmc_energy(v.model, v.three_body, r_plot)
        if bench is not None:
            ax[0, 0].plot(r_plot, bench, color=c, ls="--", lw=1, label=f"AFDMC {v.name} ({REFERENCE.split(',')[0]})")
        for k, (e, de) in enumerate(POINTS_RHO0.get((v.model, v.three_body), {}).values()):
            ax[0, 0].errorbar([RHO0], [e], [de], fmt="s", mfc="none", ms=6, color=c, alpha=0.7,
                              label=f"published 66-n values, {v.name}" if k == 0 else None)
    fg = thermodynamics_free_gas(r_plot)
    ax[0, 0].plot(r_plot, fg, color="gray", ls="-.", lw=1, label="free Fermi gas")
    top = max([v.E.max() for v in variants if len(v.E)] + [10.0])
    ax[0, 0].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel="E/N (MeV)", title="energy per neutron",
                 xlim=(0, 1.35 * rmax), ylim=(0, 1.5 * top))
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel="(data - fit) / error", title="fit residuals")
    for x in (ax[1, 0], ax[1, 1]):
        x.axvspan(rmax, r_plot[-1], color="gray", alpha=0.12)
        x.text(0.98, 0.04, "extrapolated", transform=x.transAxes, ha="right", fontsize=8, color="gray")
    ax[1, 0].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel=r"P (MeV fm$^{-3}$)", yscale="log", title="pressure (fit)")
    ax[1, 1].axhline(1, color="r", lw=0.8, ls=":")
    ax[1, 1].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel=r"$c_s^2$", ylim=(0, 1.3), title="speed of sound (fit)")
    for x in ax.flat:
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.suptitle("Equation of state of pure neutron matter", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def thermodynamics_free_gas(rho):
    from nsmatter.constants import HBAR2_2M
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    return 0.6 * HBAR2_2M * kf**2


def page_decomposition(pdf, variants, colors):
    fig, ax = plt.subplots(2, 2, figsize=A4)
    at_rho0 = []
    labelled = set()  # one legend entry per operator group
    for v, c in zip(variants, colors):
        recs = [r for r in v.recs if "components" in r]
        if not recs:
            continue
        rho = np.array([r["density"] for r in recs])
        for key, ls in (("kinetic", "--"), ("potential", ":"), ("total", "-")):
            m, e = component(recs, key)
            ax[0, 0].errorbar(rho, m, e, fmt="o" + ls, color=c, ms=4, lw=1, label=f"{key} {v.name}")
        ls = ["-", "--", ":", "-."][colors.index(c) % 4]
        for k, g in enumerate(GROUPS):
            m, e = component(recs, f"group_{g}")
            if np.all(np.isnan(m)):
                continue
            ax[0, 1].errorbar(rho, m, e, fmt="os^vDP"[k % 6], ls=ls, color=f"C{k}", ms=4, lw=1,
                              label=g if g not in labelled else None)
            labelled.add(g)
        ax[0, 1].plot([], [], ls=ls, color="k", label=f"line: {v.name}")
        at_rho0.append((v, c, recs[int(np.argmin(np.abs(rho - RHO0)))]))
    ax[0, 0].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel="MeV per neutron", title="kinetic and potential energy")
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel=r"$\langle V\rangle/N$ (MeV)", title="potential by operator group")

    chans = [c for g in GROUPS.values() for c in g]
    width = 0.8 / max(1, len(at_rho0))
    for n, (v, c, rec) in enumerate(at_rho0):
        vals = np.array([rec["components"].get(f"v_{ch}", [np.nan, np.nan]) for ch in chans], dtype=float)
        ax[1, 0].barh(np.arange(len(chans)) + n * width, vals[:, 0], width, xerr=vals[:, 1], color=c,
                      label=f"{v.name}, rho = {rec['density']}")
    ax[1, 0].set_yticks(np.arange(len(chans)) + 0.4 - width / 2, chans, fontsize=7)
    ax[1, 0].invert_yaxis()
    ax[1, 0].axvline(0, color="k", lw=0.5)
    ax[1, 0].set(xlabel=r"$\langle V_p\rangle/N$ (MeV)", title="every channel near rho0")

    ax[1, 1].axis("off")
    if at_rho0:
        keys = ["total", "kinetic", "potential"] + [f"group_{g}" for g in GROUPS]
        cells = []
        for k in keys:
            row = [k.replace("group_", "")]
            for v, c, rec in at_rho0:
                m = rec["components"].get(k)
                row.append(f"{m[0]:.2f} +- {m[1]:.2f}" if m else "-")
            cells.append(row)
        t = ax[1, 1].table(cellText=cells, colLabels=["per neutron (MeV)"] + [v.name for v, _, _ in at_rho0],
                           loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.3)
        ax[1, 1].set_title(f"values near rho0 = {RHO0}", fontsize=10)
    for x in (ax[0, 0], ax[0, 1], ax[1, 0]):
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=6.5)
    fig.suptitle("Energy decomposition (in neutron matter only the group sums of c/t and s/st are basis-independent)",
                 fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_structure(pdf, variants):
    fig, ax = plt.subplots(2, 2, figsize=A4)
    v = variants[0]
    recs = [r for r in v.recs if "structure" in r]
    cmap = plt.get_cmap("viridis")
    for n, rec in enumerate(recs):
        col = cmap(n / max(1, len(recs) - 1))
        r, g = structure(rec, "g")
        _, cs = structure(rec, "C_sigma")
        kf = (3 * np.pi**2 * rec["density"]) ** (1 / 3)
        lab = f"rho = {rec['density']}"
        ax[0, 0].errorbar(r, g["nn"][0], g["nn"][1], color=col, lw=1.2, label=lab)
        ax[0, 1].plot(kf * r, g["nn"][0] * rec["N"] / (rec["N"] - 1), color=col, lw=1.2, label=lab)
        ax[1, 0].errorbar(r, cs["nn"][0], cs["nn"][1], color=col, lw=1.2, label=lab)
        ax[1, 1].plot(kf * r, cs["nn"][0], color=col, lw=1.2, label=lab)
    x = np.linspace(0.05, 6, 200)
    ax[0, 1].plot(x, 1 - 0.5 * (3 * spherical_jn(1, x) / x) ** 2, "k--", lw=1, label="free Fermi gas")
    ax[0, 0].set(xlabel="r (fm)", ylabel=r"$g_{nn}(r)$", title=f"pair distribution ({v.name})")
    ax[0, 1].set(xlabel=r"$k_F r$", ylabel=r"$g_{nn}\,N/(N-1)$", title="scaled with the Fermi momentum")
    for a in (ax[1, 0], ax[1, 1]):
        a.axhline(0, color="k", lw=0.5)
        a.axhline(-3, color="gray", lw=0.5, ls=":")
    ax[1, 0].set(xlabel="r (fm)", ylabel=r"$\langle\sigma_i\cdot\sigma_j\rangle(r)$",
                 title="spin correlation (-3: singlet, +1: aligned)")
    ax[1, 1].set(xlabel=r"$k_F r$", ylabel=r"$\langle\sigma_i\cdot\sigma_j\rangle$", title="scaled with the Fermi momentum")
    for a in ax.flat:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.suptitle("Structure of neutron matter: correlation hole and spin pairing", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_star(pdf, variants, colors):
    fig, ax = plt.subplots(2, 2, figsize=A4)
    rows = []
    for v, c in zip(variants, colors):
        if v.star is None:
            continue
        for b in v.boots[:40]:
            ax[0, 0].plot(b["R"], b["M"], color=c, alpha=0.07, lw=1)
        s = v.star
        ax[0, 0].plot(s["R"], s["M"], color=c, lw=2, label=v.name)
        ax[0, 1].plot(s["rho_c"], s["M"], color=c, lw=2, label=v.name)
        ax[0, 1].axvline(v.rho.max(), color=c, ls=":", lw=1)
        if np.isfinite(s["rho_cap"]):
            ax[0, 1].axvline(s["rho_cap"], color=c, ls="--", lw=0.8)
        ok = np.isfinite(s["Lambda"]) & (s["M"] > 0.8)
        ax[1, 0].plot(s["M"][ok], s["Lambda"][ok], color=c, lw=2, label=v.name)
        f = lambda k, d=2: f"{s[k]:.{d}f} +- {v.spread(k):.{d}f}"  # noqa: E731
        rows.append([v.name, f("M_max"), f("R_at_Mmax"), f("R_1.4"), f("Lambda_1.4", 0), f"{s['rho_c_max']:.2f}",
                     f"{s['rho_cap']:.2f}" if np.isfinite(s["rho_cap"]) else "-", f"{v.rho.max():.2f}"])
    ax[0, 0].axhline(2.0, color="gray", ls=":", lw=1)
    ax[0, 0].set(xlabel="R (km)", ylabel=r"M (M$_\odot$)", xlim=(8, 16), ylim=(0, 2.8), title="mass-radius")
    ax[0, 1].set(xlabel=r"central density $\rho_c$ (fm$^{-3}$)", ylabel=r"M (M$_\odot$)", xscale="log",
                 title="mass vs central density")
    ax[0, 1].text(0.02, 0.95, "dotted: largest computed density\ndashed: start of c_s^2 = 1 continuation",
                  transform=ax[0, 1].transAxes, fontsize=7, va="top")
    ax[1, 0].axvline(1.4, color="gray", ls=":", lw=1)
    ax[1, 0].axhline(720, color="r", ls=":", lw=1)
    ax[1, 0].text(0.02, 0.04, "red dotted: GW170817 bound Lambda_1.4 < ~720", transform=ax[1, 0].transAxes,
                  fontsize=7, color="r")
    ax[1, 0].set(xlabel=r"M (M$_\odot$)", ylabel=r"$\Lambda$", yscale="log", title="tidal deformability")
    ax[1, 1].axis("off")
    if rows:
        t = ax[1, 1].table(cellText=rows, colLabels=["variant", "M_max", "R(M_max)", "R_1.4", "Lambda_1.4",
                                                     "rho_c\n(M_max)", "c_s^2 = 1\nfrom", "largest\nrun rho"],
                           loc="center", cellLoc="center",
                           colWidths=[0.2, 0.13, 0.13, 0.13, 0.12, 0.1, 0.1, 0.1])
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1.15, 2.4)
        ax[1, 1].set_title("pure neutron stars (masses in M_sun, radii km, densities fm^-3)", fontsize=9)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle("Neutron stars from the VMC equation of state\n(pure neutron matter, no crust; beyond the largest "
                 "computed density the EOS is the fit's extrapolation)", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_snapshots(pdf, variants):
    recs = [r for r in variants[0].recs if r.get("configs")]
    if not recs:
        return
    picks = [recs[0], min(recs, key=lambda r: abs(r["density"] - RHO0))]
    fig = plt.figure(figsize=A4)
    for k, rec in enumerate(picks):
        try:
            data = configs(rec)
        except (FileNotFoundError, OSError):
            continue
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        draw_walker(ax, data["r"][0], data["spin"][0], np.asarray(data["t"]), float(data["L"]))
        ax.set_title(f"{variants[0].name}, rho = {rec['density']} fm$^{{-3}}$, L = {float(data['L']):.2f} fm", fontsize=10)
    fig.suptitle("Monte Carlo configurations (one walker; arrows: spin direction of each neutron)", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# quality of each run
# =============================================================================
def analyse(rec):
    """Numbers and flags of one run."""
    out = {"flags": []}
    h = np.asarray(rec.get("history", []), dtype=float)
    out["history"] = h
    if len(h) >= 8:
        q = len(h) // 4
        last, prev = h[-q:], h[-2 * q:-q]
        out["opt_end"] = last.mean()
        out["drift"] = last.mean() - prev.mean()
        sig = np.hypot(last.std(ddof=1) / np.sqrt(q), prev.std(ddof=1) / np.sqrt(q))
        if abs(out["drift"]) > max(0.5, 3 * sig):
            out["flags"].append("DRIFT")
        if abs(rec["E_per_N"] - out["opt_end"]) > max(1.0, 4 * np.hypot(rec["error"], last.std(ddof=1) / np.sqrt(q))):
            out["flags"].append("OFFSET")
    else:
        out["opt_end"] = out["drift"] = np.nan
    if len(h) < 20:
        out["flags"].append("FEW")

    data = None
    try:
        data = configs(rec)
    except (FileNotFoundError, OSError):
        data = None
    if data is None or "series_total" not in data:
        out["flags"].append("OLD")
        out["tau"] = out["err_corr"] = np.nan
        return out, None
    e = np.asarray(data["series_total"])
    out["series"] = {k[7:]: np.asarray(data[k]) for k in data.files if k.startswith("series_")}
    out["tau"] = max(integrated_autocorrelation_time(e), 0.5)  # 0.5 = uncorrelated
    out["err_corr"] = e.std(ddof=1) / np.sqrt(len(e)) * np.sqrt(max(2 * out["tau"], 1.0))
    if out["tau"] > 2:
        out["flags"].append("TAU")
    if len(e) < 30:
        out["flags"].append("FEW")
    for k in ("acc_r", "acc_s"):
        a = out["series"].get(k)
        if a is not None and not (0.25 <= a.mean() <= 0.75):
            out["flags"].append("ACC")
            break
    nu, nd, pu, pd = rec["populations"]
    S = abs(nu + pu - nd - pd) / 2
    out["S2_expected"] = S * (S + 1)
    if "S2" in out["series"] and abs(out["series"]["S2"].mean() - out["S2_expected"]) > 1:
        out["flags"].append("S2")
    out["flags"] = sorted(set(out["flags"]))
    return out, data


def label(rec):
    tb = f"+3-{rec['three_body']}" if rec.get("three_body") else ""
    return f"NV2-{rec['model']}{tb}  rho = {rec['density']:.3f}  pops {''.join(map(str, rec['populations']))}"


def run_page(pdf, rec, a):
    fig, ax = plt.subplots(2, 2, figsize=(11.7, 8.3))
    h = a["history"]
    ax[0, 0].plot(h, lw=1, color="C0")
    if len(h) >= 8:
        q = len(h) // 4
        ax[0, 0].axhline(a["opt_end"], color="C1", ls="--", lw=1, label=f"last quarter {a['opt_end']:.2f}")
        ax[0, 0].axvspan(len(h) - 2 * q, len(h) - q, color="gray", alpha=0.15)
        ax[0, 0].axvspan(len(h) - q, len(h), color="C1", alpha=0.12)
    ax[0, 0].axhline(rec["E_per_N"], color="C3", lw=1, label=f"measured {rec['E_per_N']:.3f}")
    ax[0, 0].set(xlabel="SR iteration", ylabel="E/N (MeV)", title="optimization")
    ax[0, 0].legend(fontsize=8)

    s = a.get("series")
    if s is not None:
        e = s["total"]
        ax[0, 1].plot(e, "o-", ms=3, lw=0.8)
        ax[0, 1].axhline(e.mean(), color="C3", lw=1)
        ax[0, 1].axhspan(e.mean() - a["err_corr"], e.mean() + a["err_corr"], color="C3", alpha=0.15,
                         label=f"mean +- corrected error ({a['err_corr']:.3f})")
        ax[0, 1].set(xlabel="measurement", ylabel="E/N (MeV)", title=f"measurement (tau_int = {a['tau']:.2f})")
        ax[0, 1].legend(fontsize=8)
        ax[1, 0].plot(s["variance"], color="C2", lw=1, label="var(E_L)/N^2")
        ax[1, 0].set(xlabel="measurement", ylabel=r"var$(E_L)/N^2$ (MeV$^2$)", title="sampling")
        ax2 = ax[1, 0].twinx()
        ax2.plot(s["acc_r"], color="C0", lw=1, label="acc positions")
        ax2.plot(s["acc_s"], color="C1", lw=1, label="acc spins")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("acceptance")
        lines = ax[1, 0].get_lines() + ax2.get_lines()
        ax[1, 0].legend(lines, [ln.get_label() for ln in lines], fontsize=8)
        if "S2" in s:
            ax[1, 1].plot(s["S2"], color="C4", lw=1, label=r"$\langle S^2\rangle$")
            ax[1, 1].axhline(a["S2_expected"], color="k", ls=":", lw=1, label=f"S(S+1) = {a['S2_expected']:g}")
            ax[1, 1].set(xlabel="measurement", ylabel=r"$\langle S^2 \rangle$", title="total spin")
            ax[1, 1].legend(fontsize=8)
    else:
        for x in (ax[0, 1], ax[1, 0], ax[1, 1]):
            x.text(0.5, 0.5, "no measurement series\n(record from the earlier code)", ha="center", va="center",
                   transform=x.transAxes)
    flags = ", ".join(a["flags"]) or "none"
    fig.suptitle(f"{label(rec)}\nE/N = {rec['E_per_N']:.3f} +- {rec['error']:.3f} (recorded)"
                 f"   corrected error {a['err_corr']:.3f}   drift {a['drift']:+.2f}   flags: {flags}", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, rows):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    head = ["run", "E/N", "err", "err corr.", "tau", "opt. end", "drift", "flags"]
    cells = [[label(r).replace("  ", " "), f"{r['E_per_N']:.3f}", f"{r['error']:.3f}",
              f"{a['err_corr']:.3f}", f"{a['tau']:.2f}", f"{a['opt_end']:.2f}", f"{a['drift']:+.2f}",
              ", ".join(a["flags"]) or "ok"] for r, a in rows]
    table = ax.table(cellText=cells, colLabels=head, loc="upper center", cellLoc="center",
                     colWidths=[0.36, 0.08, 0.07, 0.08, 0.06, 0.08, 0.07, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for k, (_, a) in enumerate(rows, start=1):
        bad = [f for f in a["flags"] if f != "OLD"]
        if bad:
            for c in range(len(head)):
                table[k, c].set_facecolor("#fde2e2")
    ax.set_title("VMC run quality (flags: see the report.py docstring)", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/vmc_*.jsonl")
    p.add_argument("--bootstrap", type=int, default=20)
    p.add_argument("--fs-correct", action="store_true", help="fit the finite-size corrected energies")
    p.add_argument("--no-quality", action="store_true", help="skip the per-run quality pages")
    p.add_argument("--quality-only", action="store_true", help="only the quality table and pages (fast)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default results/reports/report.pdf (or quality.pdf)")
    args = p.parse_args()
    out = pathlib.Path(args.out or ("results/reports/quality.pdf" if args.quality_only else "results/reports/report.pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = load(args.results)
    if args.quality_only:
        recs = sorted(select(runs, three_body="any", static_only=None),
                      key=lambda r: (r.get("three_body") or "", r["model"], tuple(r["populations"]), r["density"]))
        rows = [(r, analyse(r)[0]) for r in recs]
        with PdfPages(out) as pdf:
            summary_page(pdf, rows)
            for r, a in rows:
                run_page(pdf, r, a)
        print(f"{'run':52s} {'E/N':>8s} {'err':>6s} {'corr':>6s} {'tau':>5s} {'drift':>6s}  flags")
        for r, a in rows:
            print(f"{label(r):52s} {r['E_per_N']:8.3f} {r['error']:6.3f} {a['err_corr']:6.3f} {a['tau']:5.2f} "
                  f"{a['drift']:+6.2f}  {', '.join(a['flags']) or 'ok'}")
        print(f"\nsaved {out}")
        return
    keys = sorted({(r["model"], r.get("three_body")) for r in runs
                   if tuple(r["populations"]) == PNM and not r.get("static_only")},
                  key=lambda k: (k[1] is not None, k[0], k[1] or ""))
    if not keys:
        raise SystemExit("no pure neutron matter runs found")
    rng = np.random.default_rng(args.seed)
    print("variants:", ", ".join(f"{m}{'+3-' + t if t else ''}" for m, t in keys))
    variants = []
    for m, t in keys:
        print(f"  fitting and solving TOV for NV2-{m}{'+3-' + t if t else ''} ...", flush=True)
        variants.append(Variant(runs, m, t, args.fs_correct, args.bootstrap, rng))
    colors = [f"C{k}" for k in range(len(variants))]

    with PdfPages(out) as pdf:
        page_overview(pdf, variants, runs, args)
        page_eos(pdf, variants, colors)
        page_decomposition(pdf, variants, colors)
        page_structure(pdf, variants)
        page_star(pdf, variants, colors)
        page_snapshots(pdf, variants)
        if not args.no_quality:
            rows = [(r, analyse(r)[0]) for v in variants for r in v.recs]
            summary_page(pdf, rows)
            for r, a in rows:
                run_page(pdf, r, a)
    for v in variants:
        if v.star is not None:
            s = v.star
            print(f"{v.name:16s} M_max {s['M_max']:.2f}  R_1.4 {s['R_1.4']:.2f} km  Lambda_1.4 {s['Lambda_1.4']:.0f}  "
                  f"rho_c(Mmax) {s['rho_c_max']:.2f}")
        else:
            print(f"{v.name:16s} {len(v.rho)} densities (>= 5 needed for the EOS fit and the star)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
