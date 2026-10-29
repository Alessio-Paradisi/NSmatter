"""Plotting helpers shared by the scripts (scripts/plot_results.py, scripts/report.py).

draw_walker draws one Monte Carlo configuration in the periodic box: neutrons blue,
protons orange, spin directions as arrows, optionally the 26 neighbouring copies.
"""

import numpy as np

NEUTRON, PROTON = "#1f77b4", "#ff7f0e"


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
