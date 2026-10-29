"""Quality report of VMC runs: one summary page plus one page per run, in a single PDF.

For every run (record of run_nv2.py, with its configuration file if present):
  optimization  E/N per SR iteration, mean of the last quarter, drift between the
                last two quarters (convergence of the wavefunction optimization)
  measurement   E/N per measurement with mean and error; integrated autocorrelation
                time tau_int and the corrected error sigma/sqrt(n) * sqrt(2 tau_int)
                (the recorded error assumes independent measurements)
  sampling      variance of E_L, acceptance of position and spin moves
  wavefunction  <S^2> (closed shells: S(S+1) with S = |N_up - N_down|/2; tensor
                correlations mix spins, so small deviations are expected)
Flags (also listed on the summary page):
  DRIFT   last two optimization quarters differ by > max(0.5 MeV, 3 sigma): not converged
  OFFSET  measured E/N differs from the end of the optimization by > max(1 MeV, 4 sigma)
  ACC     an acceptance outside [0.25, 0.75]: step sizes badly tuned
  TAU     tau_int > 2: the recorded error is underestimated (use the corrected one)
  FEW     fewer than 30 measurements or 20 iterations
  S2      <S^2> deviates by more than 1 from S(S+1)
  OLD     record without the measurement series (run with the earlier code)

Usage (from ~/projects/nsmatter, venv active):
    python scripts/run_report.py --results results/vmc_Ib.jsonl
    python scripts/run_report.py --results "results/vmc_*.jsonl" --model Ib --three-body any
Output: results/figures/run_report.pdf (or --out).
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from nsmatter.results import configs, load, select  # noqa: E402
from nsmatter.stats import integrated_autocorrelation_time  # noqa: E402


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
    ax.set_title("VMC run quality (flags: see run_report.py docstring)", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/vmc_*.jsonl")
    p.add_argument("--model", default=None)
    p.add_argument("--three-body", default="any", help='"any" (default), None-like "none", or e.g. Ib*')
    p.add_argument("--out", default="results/figures/run_report.pdf")
    a = p.parse_args()
    tb = None if a.three_body.lower() == "none" else a.three_body
    recs = select(load(a.results), model=a.model, three_body=tb, static_only=None)
    recs = sorted(recs, key=lambda r: (r.get("three_body") or "", r["model"], tuple(r["populations"]), r["density"]))
    if not recs:
        raise SystemExit("no records")
    rows = [(r, analyse(r)[0]) for r in recs]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        summary_page(pdf, rows)
        for r, an in rows:
            run_page(pdf, r, an)
    print(f"{'run':52s} {'E/N':>8s} {'err':>6s} {'corr':>6s} {'tau':>5s} {'drift':>6s}  flags")
    for r, an in rows:
        print(f"{label(r):52s} {r['E_per_N']:8.3f} {r['error']:6.3f} {an['err_corr']:6.3f} {an['tau']:5.2f} "
              f"{an['drift']:+6.2f}  {', '.join(an['flags']) or 'ok'}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
