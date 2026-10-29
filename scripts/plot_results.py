"""Figures from VMC runs (records of run_nv2.py / eos_scan.py and their configuration files).

Three modes:
  energy  E/N, kinetic and potential energy vs density; potential split by operator
          group; bar chart of every NV2 channel at one density.
  pairs   pair distributions g_nn, g_np, g_pp(r) (with the free Fermi gas for nn in
          neutron matter), spin correlation C_sigma(r) and isospin correlation C_tau(r).
  config  3D snapshot of one walker in the periodic box: neutrons blue, protons orange,
          spin directions as arrows; --images draws the neighbouring periodic copies;
          --animate saves a GIF of a tracked walker's trajectory (one frame per measurement).

Usage (from ~/projects/nsmatter, venv active):
    python scripts/plot_results.py energy --results "results/vmc_*.jsonl" --models Ib IIb --density 0.16
    python scripts/plot_results.py pairs  --results results/vmc_Ib.jsonl --model Ib --densities 0.04 0.16 0.24
    python scripts/plot_results.py config --results results/vmc_Ib.jsonl --model Ib --density 0.16 --images
    python scripts/plot_results.py config --results results/vmc_Ib.jsonl --model Ib --density 0.16 --animate
Figures are written to results/figures/ as PDF (--format png for bitmaps); animations as GIF.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from scipy.special import spherical_jn  # noqa: E402

from nsmatter.potentials.nv2_vmc import GROUPS  # noqa: E402
from nsmatter.results import component, configs, load, select, structure  # noqa: E402

NEUTRON, PROTON = "#1f77b4", "#ff7f0e"
CLASS_STYLE = {"nn": ("nn", NEUTRON), "np": ("np", "#2ca02c"), "pp": ("pp", PROTON)}
FIG = pathlib.Path("results/figures")
FORMAT = ["pdf"]  # set from --format


def pops_arg(p):
    p.add_argument("--populations", nargs=4, type=int, default=[7, 7, 0, 0], metavar=("NU", "ND", "PU", "PD"))
    p.add_argument("--three-body", default=None, help="NV3 model of NV2+3 runs (e.g. Ib*); default NV2 only")


# =============================================================================
# energy
# =============================================================================
def plot_energy(a):
    runs = load(a.results)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_data = {}
    for c, model in zip(colors, a.models):
        recs = [r for r in select(runs, model=model, populations=a.populations, three_body=a.three_body) if "components" in r]
        name = f"NV2-{model}" + (f"+3-{a.three_body}" if a.three_body else "")
        if not recs:
            print(f"{name}: no records with an energy decomposition")
            continue
        rho = np.array([r["density"] for r in recs])
        for key, ls, lab in (("total", "-", "E/N"), ("kinetic", "--", "kinetic"), ("potential", ":", "potential")):
            m, e = component(recs, key)
            ax[0].errorbar(rho, m, e, fmt="o" + ls, color=c, ms=4, lw=1.2, label=f"{lab} {name}")
        fg = [r["free_gas_infinite"] for r in recs]
        ax[0].plot(rho, fg, color="gray", lw=0.8, ls="-.", label="free gas" if model == a.models[0] else None)
        for k, g in enumerate(GROUPS):
            m, e = component(recs, f"group_{g}")
            if np.all(np.isnan(m)):
                continue
            ax[1].errorbar(rho, m, e, fmt="os^vDP"[k % 6] + "-", color=c, ms=4, lw=1, alpha=0.4 + 0.6 * (k + 1) / len(GROUPS),
                           label=f"{g} ({model})")
        k = int(np.argmin(np.abs(rho - a.density)))
        bar_data[model] = recs[k]
    ax[0].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel="energy per particle (MeV)", title="energy decomposition")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel=r"$\rho$ (fm$^{-3}$)", ylabel=r"$\langle V\rangle/N$ (MeV)", title="potential by operator group")

    # bar chart: every channel at the chosen density
    chans = [c for g in GROUPS.values() for c in g]
    width = 0.8 / max(1, len(bar_data))
    for n, (model, rec) in enumerate(bar_data.items()):
        vals = [rec["components"].get(f"v_{ch}", [np.nan, np.nan]) for ch in chans]
        m, e = np.array(vals, dtype=float).T
        y = np.arange(len(chans)) + n * width
        ax[2].barh(y, m, width, xerr=e, color=colors[a.models.index(model)], label=f"NV2-{model}{'+3-' + a.three_body if a.three_body else ''}, rho = {rec['density']}")
    ax[2].set_yticks(np.arange(len(chans)) + 0.4 - width / 2, chans)
    ax[2].invert_yaxis()
    ax[2].axvline(0, color="k", lw=0.5)
    ax[2].set(xlabel=r"$\langle V_p\rangle/N$ (MeV)", title="interaction channels")
    for x in ax:
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"energy_{'_'.join(a.models)}")


# =============================================================================
# pairs
# =============================================================================
def free_gas_g_same_species(r, rho_species):
    """g for pairs of the same spin-isospin species in a free Fermi gas: 1 - [3 j1(x)/x]^2, x = kF r."""
    kf = (6 * np.pi**2 * rho_species) ** (1 / 3)
    x = kf * r
    return 1 - (3 * spherical_jn(1, x) / x) ** 2


def plot_pairs(a):
    runs = load(a.results)
    recs = [r for d in a.densities for r in select(runs, model=a.model, populations=a.populations, density=d, three_body=a.three_body)
            if "structure" in r]
    if not recs:
        raise SystemExit("no records with pair observables for these settings")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cmap = plt.get_cmap("viridis")
    for n, rec in enumerate(recs):
        shade = cmap(n / max(1, len(recs) - 1))
        r, g = structure(rec, "g")
        _, cs = structure(rec, "C_sigma")
        _, ct = structure(rec, "C_tau")
        for cls, (vals, errs) in g.items():
            if np.all(np.isnan(vals)) or np.nansum(vals) == 0:
                continue
            ls = {"nn": "-", "np": "--", "pp": ":"}[cls]
            ax[0].errorbar(r, vals, errs, fmt=ls, color=shade, lw=1.3, label=f"{cls}, rho = {rec['density']}")
            ax[1].errorbar(r, cs[cls][0], cs[cls][1], fmt=ls, color=shade, lw=1.3, label=f"{cls}, rho = {rec['density']}")
            if cls == "np":
                ax[2].errorbar(r, ct[cls][0], ct[cls][1], fmt=ls, color=shade, lw=1.3, label=f"np, rho = {rec['density']}")
        nu, nd, pu, pd = rec["populations"]
        if pu + pd == 0 and nu == nd:  # unpolarized neutron matter: same-spin pairs are half of all
            N = rec["N"]
            g_fg = 0.5 * (1 + free_gas_g_same_species(r, rec["density"] / 2)) * (N - 1) / N
            ax[0].plot(r, g_fg, color=shade, lw=0.8, ls="-.", alpha=0.8)
    ax[0].set(xlabel="r (fm)", ylabel=r"$g_\alpha(r)$", title="pair distribution (dash-dot: free gas, nn)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="r (fm)", ylabel=r"$\langle\sigma_i\cdot\sigma_j\rangle(r)$",
              title="spin correlation (+1 aligned, -3 singlet)")
    if not ax[2].lines:
        ax[2].text(0.5, 0.5, "no np pairs\n(nn, pp: tau.tau = 1)", ha="center", va="center", transform=ax[2].transAxes)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set(xlabel="r (fm)", ylabel=r"$\langle\tau_i\cdot\tau_j\rangle(r)$",
              title="isospin correlation, np pairs (+1 T=1, -3 T=0)")
    for x in ax:
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"pairs_{a.model}_{''.join(map(str, a.populations))}")


# =============================================================================
# config
# =============================================================================
def draw_box(ax, L, offset=(0, 0, 0), **kw):
    o = np.asarray(offset) - L / 2
    for s in range(3):
        for a_ in (0, L):
            for b_ in (0, L):
                p0, p1 = np.zeros(3), np.zeros(3)
                u, v = [k for k in range(3) if k != s]
                p0[u] = p1[u] = a_
                p0[v] = p1[v] = b_
                p1[s] = L
                ax.plot(*zip(o + p0, o + p1), **kw)


def draw_walker(ax, r, spin, t, L, images=False, spins=True, arrow=None):
    """Particles wrapped into [-L/2, L/2); neighbouring copies faded with --images."""
    arrow = 0.12 * L if arrow is None else arrow
    r = (r + L / 2) % L - L / 2
    shifts = [np.zeros(3)]
    if images:
        g = (-1, 0, 1)
        shifts = [L * np.array([x, y, z]) for x in g for y in g for z in g]
    for sh in shifts:
        central = not np.any(sh)
        alpha = 1.0 if central else 0.35
        for tz, col, lab in ((-1, NEUTRON, "neutron"), (1, PROTON, "proton")):
            m = t == tz
            if not m.any():
                continue
            p = r[m] + sh
            ax.scatter(*p.T, s=60 if central else 30, color=col, alpha=alpha, edgecolors="k" if central else "none",
                       linewidths=0.4, depthshade=central, label=lab if central else None)
            if spins and central:
                ax.quiver(*p.T, *(arrow * spin[m]).T, color="k", lw=1.0, arrow_length_ratio=0.3, alpha=0.8)
        if not central:
            draw_box(ax, L, offset=sh, color="gray", lw=0.4, alpha=0.25)
    draw_box(ax, L, color="k", lw=0.9, alpha=0.8)
    lim = 1.5 * L if images else 0.6 * L
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xlabel="x (fm)", ylabel="y (fm)", zlabel="z (fm)")
    ax.set_box_aspect((1, 1, 1))


def plot_config(a):
    if a.npz:
        data = np.load(a.npz)
        title = pathlib.Path(a.npz).stem
    else:
        recs = [r for r in select(load(a.results), model=a.model, populations=a.populations, density=a.density,
                              three_body=a.three_body)
                if r.get("configs")]
        if not recs:
            raise SystemExit("no record with saved configurations for these settings")
        data = configs(recs[-1])
        title = f"NV2-{a.model}, rho = {a.density} fm$^{{-3}}$, N = {len(data['t'])}"
    name = (pathlib.Path(a.npz).stem if a.npz
            else f"{a.model}_rho{a.density:.3f}_{''.join(map(str, a.populations))}") + f"_w{a.walker}"
    L, t = float(data["L"]), np.asarray(data["t"])

    if not a.animate:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        draw_walker(ax, data["r"][a.walker], data["spin"][a.walker], t, L, images=a.images, spins=not a.no_spins)
        ax.set_title(f"{title}\nwalker {a.walker}" + (" (with periodic images)" if a.images else ""), fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        save(fig, f"config_{name}" + ("_images" if a.images else ""))
        return

    traj_r, traj_s = data["traj_r"][:, a.walker], data["traj_spin"][:, a.walker]  # (frames, N, 3)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="3d")

    def frame(n):
        ax.clear()
        draw_walker(ax, traj_r[n], traj_s[n], t, L, images=a.images, spins=not a.no_spins)
        ax.view_init(elev=20, azim=30 + 2 * n)
        ax.set_title(f"{title}\nwalker {a.walker}, measurement {n + 1}/{len(traj_r)}", fontsize=10)

    anim = FuncAnimation(fig, frame, frames=len(traj_r))
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"trajectory_{name}{'_images' if a.images else ''}.gif"
    anim.save(path, writer=PillowWriter(fps=a.fps))
    print(f"saved {path}")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.{FORMAT[0]}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("energy")
    e.add_argument("--results", default="results/vmc_*.jsonl")
    e.add_argument("--models", nargs="+", default=["Ib"])
    e.add_argument("--density", type=float, default=0.16, help="density of the channel bar chart")
    pops_arg(e)

    q = sub.add_parser("pairs")
    q.add_argument("--results", default="results/vmc_*.jsonl")
    q.add_argument("--model", default="Ib")
    q.add_argument("--densities", nargs="+", type=float, default=[0.04, 0.16])
    pops_arg(q)

    c = sub.add_parser("config")
    c.add_argument("--results", default="results/vmc_*.jsonl")
    c.add_argument("--npz", default=None, help="a configs/*.npz file directly, instead of model/density")
    c.add_argument("--model", default="Ib")
    c.add_argument("--density", type=float, default=0.16)
    c.add_argument("--walker", type=int, default=0, help="walker index (for --animate: one of the tracked ones)")
    c.add_argument("--images", action="store_true", help="draw the 26 neighbouring periodic copies")
    c.add_argument("--no-spins", action="store_true")
    c.add_argument("--animate", action="store_true", help="GIF of the walker's trajectory")
    c.add_argument("--fps", type=int, default=4)
    pops_arg(c)

    for sp in (e, q, c):
        sp.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    a = p.parse_args()
    FORMAT[0] = a.format
    {"energy": plot_energy, "pairs": plot_pairs, "config": plot_config}[a.mode](a)


if __name__ == "__main__":
    main()
