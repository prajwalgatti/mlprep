"""Figures for sb.sharding-notation (Scaling Book ch. 3, notation and Case 1)."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register


def _panel(ax, x0, y0, owner, title, p, w=2.0, h=1.4):
    """Draw a 4x4-block array; owner(i, j) -> list of device ids holding block (i, j)."""
    n = 4
    bw, bh = w / n, h / n
    for i in range(n):
        for j in range(n):
            devs = owner(i, j)
            k = len(devs)
            if k == 4:   # fully replicated: one neutral cell, labelled once below
                ax.add_patch(Rectangle((x0 + j * bw, y0 + (n - 1 - i) * bh), bw, bh,
                                       facecolor=p.faint, edgecolor=p.surface, lw=0.6))
                continue
            for t, d in enumerate(devs):   # split the cell if several devices hold it
                ax.add_patch(Rectangle((x0 + j * bw + t * bw / k, y0 + (n - 1 - i) * bh), bw / k, bh,
                                       facecolor=p.c(d), edgecolor=p.surface, lw=0.6, alpha=0.85))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=p.muted, lw=0.8))
    if all(len(owner(i, j)) == 4 for i in range(n) for j in range(n)):
        ax.text(x0 + w / 2, y0 + h / 2, "all 4 devices\nhold every block", ha="center", va="center", fontsize=7,
                color=p.fg)
    ax.text(x0 + w / 2, y0 + h + 0.12, title, ha="center", va="bottom", fontsize=8, color=p.fg)
    ax.text(x0 - 0.08, y0 + h / 2, "I", ha="right", va="center", fontsize=7, color=p.muted)
    ax.text(x0 + w / 2, y0 - 0.08, "J", ha="center", va="top", fontsize=7, color=p.muted)


@register("sb.sharding-notation", "sharding-patterns")
def sharding_patterns(p):
    """Which of 4 devices (mesh X=2 by Y=2; device = 2x + y) holds each block of A[I, J] under four shardings."""
    fig, ax = figure(3.1)
    blank(ax, (-0.3, 5.3), (-1.25, 4.3))
    _panel(ax, 0.2, 2.5, lambda i, j: [0, 1, 2, 3], "$A[I, J]$: every device\nholds everything", p)
    _panel(ax, 2.9, 2.5, lambda i, j: [2 * (i // 2), 2 * (i // 2) + 1], "$A[I_X, J]$: rows split\non X, copied across Y", p)
    _panel(ax, 0.2, 0.0, lambda i, j: [2 * (i // 2) + (j // 2)], "$A[I_X, J_Y]$: one\nquarter each", p)
    _panel(ax, 2.9, 0.0, lambda i, j: [i], "$A[I_{XY}, J]$: rows split\nover all 4 devices", p)
    # legend
    for d in range(4):
        x = 0.6 + (d % 2) * 2.4
        yy = -0.7 - (d // 2) * 0.32
        ax.add_patch(Rectangle((x, yy - 0.07), 0.18, 0.15, facecolor=p.c(d), edgecolor="none"))
        ax.text(x + 0.24, yy, f"device {d} = (x={d // 2}, y={d % 2})", fontsize=6.0, color=p.muted, va="center")
    return fig


@register("sb.sharding-notation", "case1-blocks")
def case1_blocks(p):
    """Case 1: device (x, y) holds row block x of A and column block y of B, so it can form block (x, y) of C alone."""
    fig, ax = figure(2.2)
    blank(ax, (-0.2, 9.6), (-1.5, 3.2))
    # A: rows split in 2 (X)
    for x in range(2):
        ax.add_patch(Rectangle((0, 1.25 - x * 1.25), 1.6, 1.25, facecolor=p.surface,
                               edgecolor=p.muted, lw=1))
        ax.text(0.8, 1.875 - x * 1.25, f"rows\nx={x}", ha="center", va="center", fontsize=7, color=p.fg)
    ax.text(0.8, 2.7, "$A[I_X, J]$", ha="center", fontsize=8, color=p.fg)
    ax.text(2.0, 1.25, "·", ha="center", va="center", fontsize=14, color=p.fg)
    # B: columns split in 2 (Y)
    for y in range(2):
        ax.add_patch(Rectangle((2.5 + y * 1.25, 0.15), 1.25, 2.2, facecolor=p.surface,
                               edgecolor=p.muted, lw=1))
        ax.text(3.125 + y * 1.25, 1.25, f"cols\ny={y}", ha="center", va="center", fontsize=7, color=p.fg)
    ax.text(3.75, 2.7, "$B[J, K_Y]$", ha="center", fontsize=8, color=p.fg)
    ax.text(5.4, 1.25, "=", ha="center", va="center", fontsize=11, color=p.fg)
    # C: 2x2 blocks, each on its own device
    for x in range(2):
        for y in range(2):
            d = 2 * x + y
            ax.add_patch(Rectangle((6.0 + y * 1.25, 1.25 - x * 1.25), 1.25, 1.25, facecolor=p.c(d), alpha=0.85,
                                   edgecolor=p.surface, lw=1))
            ax.text(6.625 + y * 1.25, 1.875 - x * 1.25, f"({x},{y})", ha="center", va="center", fontsize=7,
                    color=p.surface)
    ax.text(7.25, 2.7, "$C[I_X, K_Y]$", ha="center", fontsize=8, color=p.fg)
    ax.text(4.7, -0.75, "device (x, y) multiplies row block x by column block y;\nJ is whole everywhere, so no communication",
            ha="center", fontsize=7.2, color=p.label)
    return fig
