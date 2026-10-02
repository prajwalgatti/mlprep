"""Figures for sb.tpu-chip (Scaling Book ch. 2, chip internals and appendices)."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


@register("sb.tpu-chip", "chip-diagram")
def chip_diagram(p):
    """Block diagram of one TPU chip: TensorCore (MXUs, VPU, VMEM, scalar unit) next to HBM."""
    fig, ax = figure(2.9)
    blank(ax, (0, 10), (0, 7.2))
    # TensorCore outline
    box(ax, (0.1, 0.3), 6.6, 6.5, "", p, color=p.muted)
    ax.text(3.4, 6.45, "TensorCore", ha="center", fontsize=8.5, color=p.fg)
    for i in range(4):
        box(ax, (0.4 + i * 1.55, 4.3), 1.35, 1.6, f"MXU\n128×128", p, fontsize=6.5, color=p.c(0))
    box(ax, (0.4, 2.55), 3.0, 1.3, "VPU (8×128 lanes)\n+ VREGs", p, fontsize=6.5, color=p.c(2))
    box(ax, (3.65, 2.55), 2.8, 1.3, "scalar unit\n+ SMEM", p, fontsize=6.5, color=p.muted)
    box(ax, (0.4, 0.6), 6.05, 1.5, "VMEM (128 MiB on v5e)\nprogrammer-managed scratchpad", p, fontsize=6.8, color=p.c(1))
    # HBM
    box(ax, (7.6, 0.6), 2.2, 6.2, "HBM\n\n16 GiB (v5e)\n96 GB (v5p)\n\nBW to core:\n0.8 TB/s (v5e)\n2.8 TB/s (v5p)", p, fontsize=7, color=p.c(3))
    arrow(ax, (7.55, 1.35), (6.5, 1.35), p, color=p.label, style="<|-|>")
    arrow(ax, (3.4, 2.15), (3.4, 2.5), p, color=p.label, style="<|-|>")
    arrow(ax, (1.9, 3.9), (1.9, 4.25), p, color=p.label, style="<|-|>")
    ax.text(4.9, 0.25, "VMEM ↔ compute ≈ 22× HBM BW", ha="center", fontsize=6.3, color=p.label, va="top")
    return fig


@register("sb.tpu-chip", "systolic")
def systolic(p):
    """Weight-stationary systolic array: weights preloaded, activations step right, partial sums step down, inputs skewed."""
    fig, ax = figure(2.75)
    n, g, w = 4, 1.35, 0.9          # cells, pitch, cell size
    blank(ax, (-4.4, 6.9), (-2.4, 5.6))
    X = lambda c: c * g
    Y = lambda r: (n - 1 - r) * g
    for r in range(n):
        for c in range(n):
            box(ax, (X(c), Y(r)), w, w, f"$w_{{{r}{c}}}$", p, fontsize=7.5, color=p.c(0))
            if c < n - 1:   # activation moves right to the next cell
                arrow(ax, (X(c) + w, Y(r) + 0.62), (X(c + 1), Y(r) + 0.62), p, color=p.c(2), lw=1.1)
            if r < n - 1:   # partial sum moves down to the next cell
                arrow(ax, (X(c) + 0.3, Y(r)), (X(c) + 0.3, Y(r + 1) + w), p, color=p.c(1), lw=1.1)
    # skewed input streams: row r starts r cycles later, drawn as r empty slots before its first value
    for r in range(n):
        y = Y(r) + 0.45
        for k in range(3):
            xk = -0.75 - (k + r) * 0.62
            if xk < -4.2:
                continue
            ax.text(xk, y, f"$x^{{({k + 1})}}_{{{r}}}$", ha="center", va="center", fontsize=6.8, color=p.c(2))
        arrow(ax, (-0.45, y), (-0.03, y), p, color=p.c(2), lw=1.1)
    for c in range(n):
        x = X(c) + 0.3
        arrow(ax, (x, Y(n - 1)), (x, Y(n - 1) - 0.75), p, color=p.c(1), lw=1.1)
        ax.text(x, -0.95, f"$y_{{{c}}}$", ha="center", va="top", fontsize=7.5, color=p.c(1))
    ax.text(2.2, 5.3, "weights preloaded, one per cell (stationary)", ha="center", fontsize=7.3, color=p.c(0))
    ax.text(5.55, 2.6, "each cycle a cell\nadds $x\cdot w$ to the\nsum from above\n\n$x$ steps right →\nsum steps down ↓",
            fontsize=6.6, color=p.fg, va="center")
    ax.text(2.2, -2.0, "column $c$ outputs $y_c = \\Sigma_r\\, x_r w_{rc}$", ha="center", fontsize=7.3, color=p.c(1))
    return fig


@register("sb.tpu-chip", "vmem-vs-hbm")
def vmem_vs_hbm(p):
    """Time of an int8 [B,4096]x[4096,16384] matmul on TPU v5e: compute time vs HBM- and VMEM-load time."""
    D, F = 4096, 16384
    B = np.logspace(0, 3.3, 300)
    t_math = 2 * B * D * F / 3.94e14 * 1e6
    t_hbm = (D * F + B * D + B * F) / 8.2e11 * 1e6
    t_vmem = (D * F + B * D + B * F) / (22 * 8.2e11) * 1e6
    fig, ax = figure(2.5)
    ax.loglog(B, t_math, color=p.fg, lw=1.8)
    ax.loglog(B, t_hbm, color=p.c(0), lw=1.8)
    ax.loglog(B, t_vmem, color=p.c(2), lw=1.8, ls="--")
    bh = D * F / 8.2e11 / (2 * D * F / 3.94e14 - (D + F) / 8.2e11)
    bv = D * F / (22 * 8.2e11) / (2 * D * F / 3.94e14 - (D + F) / (22 * 8.2e11))
    for b, col in [(bh, p.c(0)), (bv, p.c(2))]:
        t = 2 * b * D * F / 3.94e14 * 1e6
        ax.plot([b], [t], "o", color=col, ms=4.5, zorder=5)
        ax.annotate(f"B ≈ {b:.0f}", (b, t), xytext=(b * 1.4, t * 0.35), color=col, fontsize=7.5)
    ax.text(1.3, 100, "load from HBM", color=p.c(0), fontsize=7.5)
    ax.text(1.3, 5.5, "load from VMEM", color=p.c(2), fontsize=7.5)
    ax.text(3.0, 0.35, "compute (int8 MXU)", color=p.fg, fontsize=7.5)
    ax.set_xlabel("token batch size B")
    ax.set_ylabel("time (µs)")
    ax.set_ylim(0.2, 3000)
    return fig
