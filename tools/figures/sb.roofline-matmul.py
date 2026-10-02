"""Figures for sb.roofline-matmul (Scaling Book ch. 1, matmul and network rooflines)."""
import numpy as np

from figures.style import arrow, blank, box, figure, register

PEAK = 1.97e14   # TPU v5e bf16 FLOPs/s
HBM = 8.2e11     # TPU v5e HBM bytes/s


def attainable(B, D, F, wbytes):
    flops = 2 * B * D * F
    t = np.maximum(flops / PEAK, (2 * B * D + wbytes * D * F + 2 * B * F) / HBM)
    return flops / t / 1e12


def crossover(D, F, wbytes):
    """Exact B where compute time equals HBM time."""
    return wbytes * D * F / HBM / (2 * D * F / PEAK - (2 * D + 2 * F) / HBM)


@register("sb.roofline-matmul", "batch-roofline")
def batch_roofline(p):
    """Exact attainable FLOPs/s of [B,D]x[D,F] on TPU v5e as a function of token batch B."""
    B = np.arange(1, 601)
    fig, ax = figure(2.6)
    cases = [(4096, 2, p.c(0), "-", "bf16 W, D=F=4096"),
             (1024, 2, p.c(1), "-", "bf16 W, D=F=1024"),
             (4096, 1, p.c(2), "--", "int8 W, D=F=4096"),
             (1024, 1, p.c(3), "--", "int8 W, D=F=1024")]
    for D, wb, col, ls, lab in cases:
        ax.plot(B, attainable(B, D, D, wb), color=col, ls=ls, lw=1.8)
    # crossover markers and direct labels
    pos = {0: (320, 125), 1: (355, 70), 2: (250, 32), 3: (370, 163)}
    for k, (D, wb, col, ls, lab) in enumerate(cases):
        b = crossover(D, D, wb)
        ax.plot([b], [PEAK / 1e12], "o", color=col, ms=4, zorder=5)
        x, y = pos[k]
        ax.text(x, y, f"{lab}: B ≈ {b:.0f}", color=col, fontsize=7, ha="left", va="center")
    ax.axvline(240, color=p.muted, lw=0.7, ls=":")
    ax.text(234, 206, "B = 240", color=p.muted, fontsize=7, ha="right", va="center")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 222)
    ax.set_xlabel("token batch size B")
    ax.set_ylabel("attainable TFLOP/s (v5e)")
    return fig


@register("sb.roofline-matmul", "two-chip")
def two_chip(p):
    """A matmul split along the contracting dim D over two chips: each computes a partial sum of Z."""
    fig, ax = figure(2.3)
    blank(ax, (0, 10), (0, 5.2))
    for c, x0 in enumerate([0.1, 5.1]):
        ax.text(x0 + 2.4, 5.0, f"chip {c}", ha="center", fontsize=8.5, color=p.fg)
        box(ax, (x0, 3.0), 1.35, 1.5, f"$X_{c}$\n[B, D/2]", p, fontsize=7, color=p.c(0))
        box(ax, (x0 + 1.7, 3.0), 1.35, 1.5, f"$Y_{c}$\n[D/2, F]", p, fontsize=7, color=p.c(0))
        box(ax, (x0 + 3.4, 3.0), 1.3, 1.5, f"$Z_{c}$\n[B, F]\npartial", p, fontsize=7, color=p.c(1))
        ax.text(x0 + 1.52, 3.75, "·", ha="center", va="center", fontsize=12, color=p.fg)
        ax.text(x0 + 3.22, 3.75, "=", ha="center", va="center", fontsize=9, color=p.fg)
    arrow(ax, (4.15, 2.9), (8.75, 2.05), p, color=p.label)
    arrow(ax, (8.75, 2.9), (4.15, 2.05), p, color=p.label)
    ax.text(5.0, 1.45, "swap partial sums: 2BF bytes each way", ha="center", fontsize=7.5, color=p.label)
    ax.text(5.0, 0.55, "$Z = Z_0 + Z_1$ on both chips", ha="center", fontsize=8, color=p.fg)
    return fig
