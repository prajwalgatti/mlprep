"""Figures for sb.collective-costs (Scaling Book ch. 3, AllGather / ReduceScatter / AllReduce costs)."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register


def _held(d, step, n, bidi):
    if not bidi:
        return {(d - k) % n for k in range(step + 1)}
    out = {d}
    for k in range(1, step + 1):
        out |= {(d - k) % n, (d + k) % n}
    return out


@register("sb.collective-costs", "ring-allgather")
def ring_allgather(p):
    """Which shards each of 4 ring devices holds after each step of a unidirectional vs bidirectional AllGather."""
    n = 4
    fig, ax = figure(2.5)
    blank(ax, (-1.2, 10.4), (-1.0, 5.4))
    s = 0.32
    for panel, (bidi, steps, x0, title) in enumerate([(False, 4, 0.0, "one way: 3 steps"),
                                                       (True, 3, 6.1, "both ways: 2 steps")]):
        ax.text(x0 + steps * 1.45 / 2 - 0.2, 5.05, title, ha="center", fontsize=7.6, color=p.fg)
        for st in range(steps):
            ax.text(x0 + st * 1.45 + 2 * s - 0.15, 4.45, f"step {st}", ha="center", fontsize=6.5, color=p.muted)
            for d in range(n):
                held = _held(d, st, n, bidi)
                y = 3.6 - d * 1.0
                for k in range(n):
                    ax.add_patch(Rectangle((x0 + st * 1.45 + k * s, y), s * 0.9, s * 1.6,
                                           facecolor=p.c(k) if k in held else "none",
                                           edgecolor=p.c(k) if k in held else p.faint, lw=0.8))
        if panel == 0:
            for d in range(n):
                ax.text(-0.25, 3.6 - d * 1.0 + 0.25, f"dev {d}", ha="right", va="center", fontsize=6.5,
                        color=p.muted)
    ax.text(4.6, -0.55, "colour = whose shard; each step moves\nV/N bytes per link per direction",
            ha="center", fontsize=6.6, color=p.label)
    return fig


@register("sb.collective-costs", "latency-vs-bandwidth")
def latency_vs_bandwidth(p):
    """Bidirectional ring AllGather time on TPU v5e vs total bytes, for 4 and 16 devices: latency floor then V/W."""
    W1, tmin = 4.5e10, 1e-6
    V = np.logspace(3, 9, 400)
    fig, ax = figure(2.5)
    for n, col in [(4, p.c(0)), (16, p.c(1))]:
        t = np.maximum(tmin * n / 2, V / (2 * W1))
        ax.loglog(V, t * 1e6, color=col, lw=1.8)
        vx = tmin * n / 2 * 2 * W1
        ax.plot([vx], [tmin * n / 2 * 1e6], "o", color=col, ms=4)
        ax.text(2e3, tmin * n / 2 * 1e6 * 1.25, f"N = {n}: {n // 2} µs floor", color=col,
                fontsize=7, va="bottom")
        ax.annotate(f"V ≈ {vx / 1e3:.0f} kB", (vx, tmin * n / 2 * 1e6), xytext=(vx * 2.5, tmin * n / 2 * 1e6 * 0.3),
                    color=col, fontsize=7, arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    ax.text(1.5e3, 300, "bandwidth-bound:\nT = V/W for any N", color=p.fg, fontsize=7, ha="left")
    ax.set_xlabel("total array size V (bytes)")
    ax.set_ylabel("AllGather time (µs)")
    ax.set_ylim(0.3, 3e4)
    return fig
