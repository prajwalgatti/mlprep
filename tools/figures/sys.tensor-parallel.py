"""Figures for sys.tensor-parallel (Megatron-style tensor parallelism)."""
import numpy as np
from matplotlib.patches import Circle

from figures.style import arrow, blank, box, figure, register

# Hardware constants used in the lesson (Scaling Book Part 12 spec tables).
C_PEAK = 989e12        # H100 dense bf16 FLOP/s
W_NVLINK = 450e9       # per-GPU NVLink egress, bytes/s per direction
W_NIC = 50e9           # one 400 Gb/s NIC, bytes/s


def _op(ax, xy, label, p, color):
    """Small circle for a communication operator (f or g)."""
    ax.add_patch(Circle(xy, 0.36, facecolor=p.surface, edgecolor=color, lw=1.3))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=8.5, color=color, style="italic")


@register("sys.tensor-parallel", "megatron-mlp")
def megatron_mlp(p):
    """2-way TP MLP: column-parallel A, local GeLU, row-parallel B, one all-reduce (g) in forward."""
    fig, ax = figure(2.75)
    blank(ax, (-0.1, 10.65), (-0.5, 6.6))
    # input
    box(ax, (-0.05, 2.45), 1.25, 1.5, "$X$\n$[n,h]$", p, fontsize=7.5, color=p.fg)
    _op(ax, (1.75, 3.2), "f", p, p.c(2))
    lanes = {0: 4.55, 1: 0.55}
    for r, y in lanes.items():
        ax.text(4.25, y + 1.45, f"rank {r}", fontsize=7, color=p.muted)
        box(ax, (2.3, y), 1.95, 1.25, f"$XA_{r}$\n$A_{r}:[h,2h]$", p, fontsize=6.8, color=p.c(0))
        box(ax, (4.55, y), 1.2, 1.25, "GeLU\n(local)", p, fontsize=6.6, color=p.c(0))
        box(ax, (6.05, y), 1.95, 1.25, f"$Y_{r}B_{r}$\n$B_{r}:[2h,h]$", p, fontsize=6.8, color=p.c(1))
        arrow(ax, (2.05, 3.2 + (0.25 if r == 0 else -0.25)), (2.3, y + 0.62), p)
        arrow(ax, (4.25, y + 0.62), (4.55, y + 0.62), p)
        ax.text(4.4, y - 0.32, f"$Y_{r}:[n,2h]$", fontsize=6.3, color=p.muted, ha="center")
        arrow(ax, (5.75, y + 0.62), (6.05, y + 0.62), p)
        ax.text(7.03, y - 0.32, f"$Z_{r}:[n,h]$ partial", fontsize=6.3, color=p.muted, ha="center")
        arrow(ax, (8.0, y + 0.62), (8.62, 3.2 + (0.25 if r == 0 else -0.25)), p)
    arrow(ax, (1.2, 3.2), (1.39, 3.2), p)
    _op(ax, (8.95, 3.2), "g", p, p.label)
    box(ax, (9.45, 2.6), 0.95, 1.2, "$Z$\n$[n,h]$", p, fontsize=7, color=p.fg)
    arrow(ax, (9.31, 3.2), (9.45, 3.2), p)
    ax.text(1.75, 2.35, "fwd: identity\nbwd: all-red.", fontsize=6, color=p.c(2), ha="center", va="top")
    ax.text(8.95, 2.35, "fwd: all-red.\nbwd: identity", fontsize=6, color=p.label, ha="center", va="top")
    ax.text(5.15, 6.35, r"$Z=Y_0B_0+Y_1B_1$: in forward, the only communication",
            fontsize=6.8, color=p.fg, ha="center")
    return fig


@register("sys.tensor-parallel", "megatron-attention")
def megatron_attention(p):
    """2-way TP attention with a=4 heads: QKV columns split by heads, per-head attention local, W_O rows split."""
    fig, ax = figure(2.75)
    blank(ax, (-0.1, 10.65), (-0.5, 6.6))
    box(ax, (-0.05, 2.45), 1.25, 1.5, "$X$\n$[n,h]$", p, fontsize=7.5, color=p.fg)
    _op(ax, (1.75, 3.2), "f", p, p.c(2))
    arrow(ax, (1.2, 3.2), (1.39, 3.2), p)
    lanes = {0: 4.55, 1: 0.55}
    heads = {0: "heads 1,2", 1: "heads 3,4"}
    for r, y in lanes.items():
        ax.text(4.25, y + 1.45, f"rank {r}: {heads[r]}", fontsize=7, color=p.muted)
        box(ax, (2.3, y), 1.95, 1.25, "$Q,K,V$\nits heads", p, fontsize=6.6, color=p.c(0))
        box(ax, (4.55, y), 1.45, 1.25, "softmax\n$(QK^\\top)V$\nper head", p, fontsize=6.3, color=p.c(0))
        box(ax, (6.3, y), 1.7, 1.25, "$\\cdot W_O$ rows\n$[h/2,h]$", p, fontsize=6.6, color=p.c(1))
        arrow(ax, (2.05, 3.2 + (0.25 if r == 0 else -0.25)), (2.3, y + 0.62), p)
        arrow(ax, (4.25, y + 0.62), (4.55, y + 0.62), p)
        arrow(ax, (6.0, y + 0.62), (6.3, y + 0.62), p)
        ax.text(5.28, y - 0.32, "$[n,h/2]$", fontsize=6.3, color=p.muted, ha="center")
        ax.text(7.15, y - 0.32, "$[n,h]$ partial", fontsize=6.3, color=p.muted, ha="center")
        arrow(ax, (8.0, y + 0.62), (8.62, 3.2 + (0.25 if r == 0 else -0.25)), p)
    _op(ax, (8.95, 3.2), "g", p, p.label)
    box(ax, (9.45, 2.6), 0.95, 1.2, "out\n$[n,h]$", p, fontsize=7, color=p.fg)
    arrow(ax, (9.31, 3.2), (9.45, 3.2), p)
    ax.text(5.15, 6.35, "heads never mix inside attention, so no comm until $W_O$",
            fontsize=6.8, color=p.fg, ha="center")
    return fig


@register("sys.tensor-parallel", "layer-comm")
def layer_comm(p):
    """One transformer layer under TP: where the 2 forward and 2 backward all-reduces sit."""
    fig, ax = figure(2.5)
    blank(ax, (-0.1, 10.9), (-1.6, 3.6))
    items = [
        (0.0, 0.9, "LN", None),
        (1.15, 0.45, "f", "f"),
        (1.85, 1.75, "attention\n(TP)", "tp"),
        (3.85, 0.45, "g", "g"),
        (4.55, 0.6, "+", None),
        (5.4, 0.9, "LN", None),
        (6.55, 0.45, "f", "f"),
        (7.25, 1.75, "MLP\n(TP)", "tp"),
        (9.25, 0.45, "g", "g"),
        (9.95, 0.6, "+", None),
    ]
    y0, hgt = 1.0, 1.2
    for x, w, label, kind in items:
        if kind in ("f", "g"):
            col = p.c(2) if kind == "f" else p.label
            ax.add_patch(Circle((x + w / 2, y0 + hgt / 2), 0.3, facecolor=p.surface, edgecolor=col, lw=1.3))
            ax.text(x + w / 2, y0 + hgt / 2, label, ha="center", va="center", fontsize=8.5, color=col,
                    style="italic")
        else:
            box(ax, (x, y0), w, hgt, label, p, fontsize=6.8,
                color=p.c(0) if kind == "tp" else p.muted)
    xs = [0.9, 1.6, 3.6, 4.3, 5.15, 6.3, 7.0, 9.0]
    xe = [1.1, 1.8, 3.85, 4.55, 5.4, 6.55, 7.25, 9.25]
    for a, b in zip(xs, xe):
        arrow(ax, (a, y0 + hgt / 2), (b, y0 + hgt / 2), p)
    arrow(ax, (9.7, y0 + hgt / 2), (9.95, y0 + hgt / 2), p)
    arrow(ax, (10.55, y0 + hgt / 2), (10.85, y0 + hgt / 2), p)
    # forward / backward annotations
    for xc in (4.075, 9.475):
        ax.text(xc, 2.65, "fwd\nall-reduce", fontsize=6.5, color=p.label, ha="center", va="bottom")
    for xc in (1.375, 6.775):
        ax.text(xc, 2.65, "fwd\nno-op", fontsize=6.5, color=p.muted, ha="center", va="bottom")
    for xc in (1.375, 6.775):
        ax.text(xc, 0.75, "bwd\nall-reduce", fontsize=6.5, color=p.c(2), ha="center", va="top")
    for xc in (4.075, 9.475):
        ax.text(xc, 0.75, "bwd\nno-op", fontsize=6.5, color=p.muted, ha="center", va="top")
    ax.text(5.4, -1.5, "per layer: 2 fwd + 2 bwd all-reduces\nLN, residual add, dropout: replicated",
            fontsize=6.5, color=p.fg, ha="center")
    return fig


@register("sys.tensor-parallel", "comm-vs-compute")
def comm_vs_compute(p):
    """Per-layer forward compute vs TP all-reduce time as t grows (b=1, s=4096, h=8192, bf16, H100)."""
    b, s, h = 1, 4096, 8192
    ts = np.array([1, 2, 4, 8])
    flops = 24 * b * s * h**2 + 4 * b * s**2 * h
    comp = flops / ts / C_PEAK * 1e3
    act = 2 * b * s * h
    ar = 2 * 2 * (ts - 1) / ts * act          # two all-reduces, ring: 2(t-1)/t * bytes each
    nvl = ar / W_NVLINK * 1e3
    nic = ar / W_NIC * 1e3
    fig, ax = figure(2.5)
    ax.plot(ts, comp, "o-", color=p.c(0), ms=4)
    ax.plot(ts[1:], nvl[1:], "s-", color=p.c(2), ms=4)
    ax.plot(ts[1:], nic[1:], "^-", color=p.bad, ms=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ts)
    ax.set_xticklabels([str(t) for t in ts])
    ax.set_yticks([0.3, 1, 3, 10])
    ax.set_yticklabels(["0.3", "1", "3", "10"])
    ax.set_ylim(0.18, 20)
    ax.set_xlim(0.85, 10.5)
    ax.set_xlabel("tensor-parallel degree $t$")
    ax.set_ylabel("ms per layer (forward)")
    ax.text(1.0, 1.9, "compute $\\propto 1/t$", color=p.c(0), fontsize=7.5)
    ax.text(2.0, 6.4, "comm: one GPU per node\n(own 50 GB/s NIC)",
            color=p.bad, fontsize=6.8, va="bottom")
    ax.text(2.9, 0.2, "comm: inside one node\n(NVLink 450 GB/s)", color=p.c(2), fontsize=6.8, va="bottom")
    ax.annotate(f"{comp[3]:.2f} vs {nvl[3]:.2f} ms", (8, nvl[3]), xytext=(3.0, 0.75), fontsize=7,
                color=p.label, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.grid(True, which="major", axis="y")
    return fig
