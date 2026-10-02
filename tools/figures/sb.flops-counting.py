"""Figures for sb.flops-counting (Scaling Book ch. 4, counting dots and forward/backward FLOPs)."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register


@register("sb.flops-counting", "einsum-dims")
def einsum_dims(p):
    """Classifying the dims of an attention-score contraction Q[B,T,N,H]·K[B,S,N,H] -> [B,T,S,N]."""
    fig, ax = figure(2.3)
    blank(ax, (0, 10), (-1.6, 3.4))
    kinds = {"B": "batch", "N": "batch", "H": "contract", "T": "free", "S": "free"}
    col = {"batch": p.c(0), "contract": p.c(3), "free": p.c(2)}
    rows = [("Q", ["B", "T", "N", "H"], 2.4), ("K", ["B", "S", "N", "H"], 1.3), ("out", ["B", "T", "S", "N"], 0.2)]
    for name, dims, y in rows:
        ax.text(1.2, y + 0.35, name, ha="right", va="center", fontsize=8.5, color=p.fg)
        for i, d in enumerate(dims):
            k = kinds[d]
            ax.add_patch(Rectangle((1.5 + i * 0.95, y), 0.8, 0.7, facecolor=col[k], alpha=0.85, edgecolor=p.surface))
            ax.text(1.9 + i * 0.95, y + 0.35, d, ha="center", va="center", fontsize=9, color=p.surface)
    for i, (k, lab) in enumerate([("batch", "batching: in both inputs\nand the output"),
                                  ("contract", "contracting: in both\ninputs, summed away"),
                                  ("free", "free: in one input\nand the output")]):
        y = 2.45 - i * 1.1
        ax.add_patch(Rectangle((6.0, y), 0.35, 0.35, facecolor=col[k], edgecolor="none"))
        ax.text(6.5, y + 0.17, lab, va="center", fontsize=6.8, color=p.fg)
    ax.text(5.0, -0.9, "FLOPs = 2 × B·N·H·T·S (each dim once)", ha="center", fontsize=7.8, color=p.label)
    return fig


@register("sb.flops-counting", "fwd-bwd")
def fwd_bwd(p):
    """The three matmuls per weight matrix in training: forward 2NPM, dL/dW 2NPM, dL/dX 2NPM."""
    fig, ax = figure(1.9)
    items = [("forward $C = AB$", "contract P", p.c(0)),
             ("$\\partial L/\\partial B = A^\\top \\partial L/\\partial C$", "contract N", p.c(1)),
             ("$\\partial L/\\partial A = \\partial L/\\partial C\\, B^\\top$", "contract M", p.c(2))]
    left = 0
    for name, c, colr in items:
        ax.barh(0, 2, left=left, height=0.5, color=colr, alpha=0.85, edgecolor=p.surface)
        ax.text(left + 1, 0, "2NPM", ha="center", va="center", fontsize=7.5, color=p.surface)
        ax.text(left + 1, 0.45, name, ha="center", va="bottom", fontsize=6.6, color=p.fg)
        ax.text(left + 1, -0.42, c, ha="center", va="top", fontsize=6.5, color=p.muted)
        left += 2
    ax.annotate("", xy=(2, -0.95), xytext=(0, -0.95), arrowprops=dict(arrowstyle="<->", color=p.muted, lw=0.8))
    ax.annotate("", xy=(6, -0.95), xytext=(2, -0.95), arrowprops=dict(arrowstyle="<->", color=p.muted, lw=0.8))
    ax.text(1, -1.1, "forward: 2NPM", ha="center", va="top", fontsize=7, color=p.fg)
    ax.text(4, -1.1, "backward: 4NPM", ha="center", va="top", fontsize=7, color=p.fg)
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(-1.5, 1.1)
    ax.axis("off")
    return fig


@register("sb.flops-counting", "matmul-vs-matvec")
def matmul_vs_matvec(p):
    """FLOPs per byte of square bf16 products as the size n grows: matmul grows like n, mat-vec stays flat."""
    n = np.logspace(1, 4.3, 200)
    mm = 2 * n ** 3 / (2 * 3 * n ** 2)
    mv = 2 * n ** 2 / (2 * (n ** 2 + 2 * n))
    fig, ax = figure(2.3)
    ax.loglog(n, mm, color=p.c(0))
    ax.loglog(n, mv, color=p.c(1))
    ax.axhline(240, color=p.muted, lw=0.8, ls=":")
    ax.text(12, 290, "v5e ridge ≈ 240", fontsize=7, color=p.muted)
    ax.text(80, 2500, "matmul $[n,n]\\cdot[n,n]$\nintensity $= n/3$", fontsize=7.2, color=p.c(0), ha="center")
    ax.text(1500, 2.2, "mat-vec $[n,n]\\cdot[n]$: ≈ 1", fontsize=7.2, color=p.c(1), ha="center")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("FLOPs per byte")
    ax.set_ylim(0.3, 2e4)
    return fig
