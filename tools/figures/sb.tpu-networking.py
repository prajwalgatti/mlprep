"""Figures for sb.tpu-networking (Scaling Book ch. 2, networking)."""
import numpy as np

from figures.style import arrow, blank, figure, register


def _grid(ax, x0, p, wrap, title, path, wrap_path=()):
    n = 4
    s = 1.0
    pts = {(i, j): (x0 + j * s, (n - 1 - i) * s) for i in range(n) for j in range(n)}
    for i in range(n):
        for j in range(n):
            x, y = pts[(i, j)]
            if j < n - 1:
                ax.plot([x, x + s], [y, y], color=p.faint, lw=1.6, zorder=1)
            if i < n - 1:
                ax.plot([x, x], [y, y - s], color=p.faint, lw=1.6, zorder=1)
    stub = 0.5
    if wrap:  # wraparound links drawn as outward stubs on every edge node
        for k in range(n):
            for (i, j), (dx, dy) in [((k, 0), (-stub, 0)), ((k, n - 1), (stub, 0)),
                                     ((0, k), (0, stub)), ((n - 1, k), (0, -stub))]:
                x, y = pts[(i, j)]
                ax.plot([x, x + dx], [y, y + dy], color=p.muted, lw=1.3, ls=(0, (2, 1.5)), zorder=1)
    for a, b in zip(path[:-1], path[1:]):
        (xa, ya), (xb, yb) = pts[a], pts[b]
        ax.plot([xa, xb], [ya, yb], color=p.label, lw=2.4, zorder=2)
    for a, (dx, dy) in wrap_path:  # highlighted wraparound stubs
        xa, ya = pts[a]
        ax.plot([xa, xa + dx * stub], [ya, ya + dy * stub], color=p.label, lw=2.4, zorder=2)
    for (i, j), (x, y) in pts.items():
        c = p.accent if (i, j) in (path[0], path[-1]) or (path and False) else p.surface
        ax.plot([x], [y], "o", ms=9, color=c, mec=p.muted, mew=0.8, zorder=3)
    ax.text(x0 + 1.5, -1.25, title, ha="center", fontsize=8, color=p.fg)


@register("sb.tpu-networking", "torus-hops")
def torus_hops(p):
    """Hop count from chip (0,0) to (3,3) on a 4x4 mesh (no wraparound) vs a 4x4 torus."""
    fig, ax = figure(2.3)
    blank(ax, (-0.8, 9.9), (-1.7, 4.4))
    _grid(ax, 0, p, False, "4×4 mesh: 6 hops", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)])
    # torus: (0,0) -> wrap left -> (0,3) -> wrap up -> (3,3)
    _grid(ax, 5.5, p, True, "4×4 torus: 2 hops", [(0, 0), (0, 0)],
          wrap_path=[((0, 0), (-1, 0)), ((0, 3), (1, 0)), ((0, 3), (0, 1)), ((3, 3), (0, -1))])
    pts = {(i, j): (5.5 + j, 3 - i) for i in range(4) for j in range(4)}
    for key in [(0, 3), (3, 3)]:
        x, y = pts[key]
        ax.plot([x], [y], "o", ms=9, color=p.accent if key == (3, 3) else p.surface, mec=p.label, mew=1.4, zorder=4)
    ax.text(4.95, 3.3, "hop 1\n(wrap ↔)", fontsize=6.3, color=p.label, ha="right", va="bottom")
    ax.text(8.62, -0.55, "hop 2 (wrap ↕)", fontsize=6.3, color=p.label, ha="left", va="center")
    ax.text(7.0, 4.15, "dashed stubs = wraparound links", ha="center", fontsize=6.8, color=p.muted)
    return fig


@register("sb.tpu-networking", "bandwidth-ladder")
def bandwidth_ladder(p):
    """Bandwidth available to one TPU v5e chip over each channel, on a log scale."""
    items = [("VMEM ↔ MXU (≈22× HBM)", 22 * 8.2e11),
             ("HBM ↔ TensorCore", 8.2e11),
             ("ICI, all 4 links out", 4 * 4.5e10),
             ("ICI, one link one way", 4.5e10),
             ("PCIe to host", 1.6e10),
             ("DCN (per chip)", 3.125e9)]
    fig, ax = figure(2.5)
    y = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    cols = [p.c(2), p.c(0), p.c(1), p.c(1), p.c(3), p.c(4)]
    ax.barh(y, vals, color=cols, height=0.6, alpha=0.9)
    ax.set_xscale("log")
    for yi, (lab, v) in zip(y, items):
        ax.text(v * 1.15, yi, f"{v / 1e9:,.0f} GB/s" if v < 1e12 else f"{v / 1e12:.1f} TB/s",
                va="center", fontsize=7, color=p.fg)
        ax.text(1.1e9, yi + 0.42, lab, va="center", fontsize=7, color=p.muted)
    ax.set_yticks([])
    ax.set_xlim(1e9, 3e14)
    ax.set_ylim(-0.6, len(items) - 0.1)
    ax.set_xlabel("bytes / s (TPU v5e, log scale)")
    ax.spines["left"].set_visible(False)
    return fig


@register("sb.tpu-networking", "challenge-stages")
def challenge_stages(p):
    """Stage times for gathering a 16 GiB int8 matrix from host DRAM onto TPU{0,0} of a v5e 4x4 and using it."""
    V = (128 * 1024) ** 2
    stages = [("PCIe: 16 links into HBM", V / (16 * 1.6e10)),
              ("ICI: 15/16 of it into chip (0,0)", V * 15 / 16 / (2 * 4.5e10)),
              ("HBM → MXU", V / 8.2e11),
              ("FLOPs", 2 * 8 * (128 * 1024) ** 2 / 1.97e14)]
    fig, ax = figure(2.0)
    y = np.arange(len(stages))[::-1]
    t = [s[1] * 1e3 for s in stages]
    cols = [p.c(3), p.bad, p.c(1), p.c(2)]
    ax.barh(y, t, color=cols, height=0.55, alpha=0.9)
    for yi, (lab, _), ti in zip(y, stages, t):
        ax.text(ti + 3, yi, f"{ti:.1f} ms" if ti < 10 else f"{ti:.0f} ms", va="center", fontsize=7.5, color=p.fg)
        ax.text(0, yi + 0.42, lab, va="center", fontsize=7, color=p.muted)
    ax.set_yticks([])
    ax.set_xlim(0, 215)
    ax.set_ylim(-0.5, len(stages) - 0.05)
    ax.set_xlabel("time (ms); overlapped, the slowest stage sets the pace")
    ax.spines["left"].set_visible(False)
    return fig
