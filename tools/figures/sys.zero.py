"""Figures for sys.zero: per-rank model-state bars by stage, memory vs GPU count, ZeRO-3 per-layer dataflow."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register

P = 4
PARTS = [("bf16 params", 2), ("bf16 grads", 2), ("fp32 optimizer state", 12)]   # bytes per parameter


def _canvas(h):
    fig, ax = figure(h)
    fig.set_layout_engine(None)
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    return fig, ax


@register("sys.zero", "zero-stages")
def zero_stages(p):
    """Per-rank model-state memory (bytes per parameter) for DDP and ZeRO-1/2/3 with P = 4 ranks.
    A sharded part keeps only the 1/P slice the rank owns; the dashed outline is what was removed."""
    stages = [("DDP", 0), ("ZeRO-1", 1), ("ZeRO-2", 2), ("ZeRO-3", 3)]
    sharded = {0: set(), 1: {2}, 2: {1, 2}, 3: {0, 1, 2}}
    fig, ax = _canvas(3.6)
    scale = 0.36                   # data units per byte/param
    blank(ax, (-1.9, 7.4), (-3.5, 13.0))
    rowh, gap = 0.55, 0.09
    y = 12.0
    for name, s in stages:
        total = sum(b if i not in sharded[s] else b / P for i, (_, b) in enumerate(PARTS))
        ax.text(-1.85, y + 0.25, name, fontsize=7.8, color=p.fg, va="center", fontweight="bold")
        ax.text(7.35, y + 0.25, f"{total:g} B/param", fontsize=7.2, color=p.label, va="center", ha="right")
        y -= 0.62
        for r in range(P):
            x = 0.0
            for i, (lab, b) in enumerate(PARTS):
                w = b * scale
                col = p.c(i)
                if i in sharded[s]:
                    ax.add_patch(Rectangle((x, y), w, rowh, facecolor="none", edgecolor=p.faint, lw=0.7, ls="--"))
                    ax.add_patch(Rectangle((x + r * w / P, y), w / P, rowh, facecolor=col, edgecolor="none"))
                else:
                    ax.add_patch(Rectangle((x, y), w, rowh, facecolor=col, edgecolor=p.surface, lw=0.5))
                x += w
            ax.text(-0.12, y + rowh / 2, f"r{r}", fontsize=6.6, color=p.muted, ha="right", va="center")
            y -= rowh + gap
        y -= 0.35
    for i, (lab, b) in enumerate(PARTS):
        ax.add_patch(Rectangle((0.0, -2.05 - 0.48 * i), 0.3, 0.3, facecolor=p.c(i), edgecolor="none"))
        ax.text(0.45, -1.9 - 0.48 * i, f"{lab}: {b} bytes/param", va="center", fontsize=6.8, color=p.fg)
    return fig


@register("sys.zero", "zero-memory-vs-gpus")
def zero_memory_vs_gpus(p):
    """Per-GPU model-state memory for a 7B model vs data-parallel degree P, each ZeRO stage, with 80 GB line."""
    N = 7e9
    Pg = np.logspace(0, 10, 200, base=2)
    curves = [("DDP: 16N", np.full_like(Pg, 16 * N)),
              ("ZeRO-1: 4N + 12N/P", 4 * N + 12 * N / Pg),
              ("ZeRO-2: 2N + 14N/P", 2 * N + 14 * N / Pg),
              ("ZeRO-3: 16N/P", 16 * N / Pg)]
    fig, ax = figure(2.6)
    ypos = [150, 33, 17, 2.6]
    for i, ((lab, y), yl) in enumerate(zip(curves, ypos)):
        col = p.fg if i == 0 else p.c(i)
        ax.loglog(Pg, y / 1e9, color=col, lw=1.8)
        ax.text(1.15 if i == 0 else (40 if i < 3 else 120), yl, lab, color=col, fontsize=7.2)
    ax.axhline(80, color=p.bad, lw=1.0, ls="--")
    ax.text(40, 82, "80 GB H100", color=p.bad, fontsize=6.9, va="bottom")
    for i, (lab, y) in enumerate(curves[1:3], start=1):
        ax.axhline(y[-1] / 1e9, color=p.c(i), lw=0.6, ls=":")
    ax.set_xlabel("data-parallel degree P (log₂)")
    ax.set_ylabel("model-state memory per GPU (GB)")
    ax.set_xticks([1, 4, 16, 64, 256, 1024])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
    ax.set_ylim(0.5, 300)
    ax.set_xlim(1, 1024)
    return fig


@register("sys.zero", "zero3-dataflow")
def zero3_dataflow(p):
    """ZeRO-3 on one rank, 3 layers (illustrative units). Each layer's all-gather is issued one layer ahead, so it
    overlaps the previous layer's compute; gathered weights are freed after use; the backward re-gathers them and
    reduce-scatters each layer's gradients as soon as they exist."""
    fig, ax = _canvas(2.7)
    blank(ax, (-4.4, 19.4), (-3.6, 6.2))
    h = 0.85
    yc, ya, yr = 3.4, 2.2, 1.0
    for y, lab in [(yc, "compute"), (ya, "AG stream"), (yr, "RS stream")]:
        ax.text(-0.3, y + h / 2, lab, ha="right", va="center", fontsize=6.8, color=p.muted)
    f, b, g = 1.6, 3.2, 1.2      # forward, backward, collective durations

    def blk(x, y, w, lab, col, hatch=None, tc=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor=p.surface, lw=0.8, hatch=hatch))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=6.6, color=tc or p.surface)

    # forward
    ag_start, ag_end = {0: 0.0}, {0: g}
    blk(0.0, ya, g, "AG1", p.c(0))
    t = 0.0
    for l in range(3):
        st = max(t, ag_end[l])
        blk(st, yc, f, f"F{l + 1}", p.c(l))
        if l < 2:                                    # prefetch next layer's weights while computing this one
            ag_start[l + 1] = max(st, ag_end[l])
            ag_end[l + 1] = ag_start[l + 1] + g
            blk(ag_start[l + 1], ya, g, f"AG{l + 2}", p.c(l + 1))
        t = st + f
    fwd_end = t
    ax.plot([fwd_end + 0.2] * 2, [yr - 0.2, yc + h + 0.9], color=p.muted, lw=0.7, ls=":")
    # backward: layers 3, 2, 1
    order = [2, 1, 0]
    ag_e = {2: fwd_end + 0.4 + g}
    blk(fwd_end + 0.4, ya, g, "AG3", p.c(2))
    t, rs_free = ag_e[2], 0.0
    for i, l in enumerate(order):
        st = max(t, ag_e[l])
        blk(st, yc, b, f"B{l + 1}", p.c(l))
        if i < 2:
            nl = order[i + 1]
            blk(st, ya, g, f"AG{nl + 1}", p.c(nl))
            ag_e[nl] = st + g
        t = st + b
        rs_st = max(t, rs_free)
        blk(rs_st, yr, g, f"RS{l + 1}", p.c(l))
        rs_free = rs_st + g
    ax.text(fwd_end / 2, yc + h + 0.65, "forward", ha="center", fontsize=7.4, color=p.fg)
    ax.text((fwd_end + rs_free) / 2, yc + h + 0.65, "backward", ha="center", fontsize=7.4, color=p.fg)
    ax.text(8.2, -2.2, "AG = all-gather a layer's weights, issued one layer ahead\n"
            "RS = reduce-scatter that layer's gradients\ngathered weights are freed right after use",
            ha="center", va="center", fontsize=6.6, color=p.label, linespacing=1.35)
    return fig
