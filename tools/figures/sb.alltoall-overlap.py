"""Figures for sb.alltoall-overlap (Scaling Book ch. 3, AllToAll, transposes, collective matmul)."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import arrow, blank, figure, register


@register("sb.alltoall-overlap", "alltoall-blocks")
def alltoall_blocks(p):
    """AllToAll on 4 devices: A[I_X, J] -> A[I, J_X]. Device d starts with row block d and ends with column block d."""
    n = 4
    fig, ax = figure(2.3)
    blank(ax, (-0.8, 10.2), (-1.0, 4.6))
    c = 0.62
    for side, x0 in enumerate([0.0, 6.2]):
        for i in range(n):
            for j in range(n):
                dev = i if side == 0 else j     # who holds block (i, j)
                ax.add_patch(Rectangle((x0 + j * c, 3.0 - i * c - c + 0.62), c, c, facecolor=p.c(dev), alpha=0.85,
                                       edgecolor=p.surface, lw=0.8))
                ax.text(x0 + j * c + c / 2, 3.0 - i * c + 0.31, f"{i}{j}", ha="center", va="center", fontsize=5.5,
                        color=p.surface)
    ax.text(1.24, 3.95, "before: $A[I_X, J]$\nrow block d on dev d", ha="center", fontsize=7.2, color=p.fg)
    ax.text(7.44, 3.95, "after: $A[I, J_X]$\ncol block d on dev d", ha="center", fontsize=7.2, color=p.fg)
    arrow(ax, (3.0, 2.0), (5.9, 2.0), p, color=p.label, lw=1.4)
    ax.text(4.45, 2.25, "AllToAll", ha="center", fontsize=7.5, color=p.label)
    ax.text(4.6, -0.55, "block ij moves from device i to device j",
            ha="center", fontsize=6.6, color=p.muted)
    return fig


@register("sb.alltoall-overlap", "chunk-hops")
def chunk_hops(p):
    """Busiest-link load (in shard-sized units) for AllGather vs AllToAll on a ring of N devices, one-way and both ways."""
    N = np.array([4, 8, 16, 32, 64])
    ag_uni = (N - 1) / N              # per link, units of V
    ag_bi = (N // 2) / N
    a2a_uni = (N - 1) / (2 * N)
    a2a_bi = (N ** 2 / 8) / N ** 2   # antipodal piece split across both directions
    fig, ax = figure(2.5)
    for y, col, lab in [(ag_uni, p.c(0), "AllGather, one way"), (ag_bi, p.c(0), "AllGather, both ways"),
                        (a2a_uni, p.c(1), "AllToAll, one way"), (a2a_bi, p.c(1), "AllToAll, both ways")]:
        ls = "-" if "both" in lab else "--"
        ax.plot(N, y, color=col, ls=ls, marker="o", ms=3.5, lw=1.6)
        off = {"AllGather, both ways": 0.045, "AllToAll, one way": -0.045}.get(lab, 0)
        ax.text(70, y[-1] + off, lab, color=col, fontsize=6.8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlim(3.5, 300)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("ring size N")
    ax.set_ylabel("busiest link load / V")
    ax.text(5, 0.04, "antipodal piece split both ways", fontsize=6.8, color=p.muted)
    return fig


@register("sb.alltoall-overlap", "collective-matmul")
def collective_matmul(p):
    """Blocking AllGather then matmul vs a collective matmul that overlaps ring steps with chunk matmuls."""
    fig, ax = figure(1.9)
    rows = [("blocking", [(0, 87, p.c(1), "gather"), (87, 224, p.c(0), "matmul")]),
            ("overlapped", [(0, 56, p.c(0), "1"), (56, 56, p.c(0), "2"), (112, 56, p.c(0), "3"),
                            (168, 56, p.c(0), "4")])]
    for r, (name, segs) in enumerate(rows):
        y = 1 - r
        for x0, w, col, lab in segs:
            ax.barh(y, w, left=x0, height=0.38, color=col, alpha=0.85, edgecolor=p.surface)
            ax.text(x0 + w / 2, y, lab, ha="center", va="center", fontsize=6.8, color=p.surface)
        ax.text(-6, y, name, ha="right", va="center", fontsize=7.5, color=p.fg)
    for k in range(3):   # ring transfers hidden under compute
        ax.barh(-0.33, 29, left=k * 56 + 5, height=0.16, color=p.c(1), alpha=0.85)
    ax.text(175, -0.33, "ring shifts (hidden)", fontsize=6.5, color=p.c(1), va="center")
    ax.text(311, 1, " 311 µs", fontsize=7, color=p.muted, va="center")
    ax.text(224, 0, " 224 µs ideal,\n 244 measured", fontsize=7, color=p.muted, va="center")
    ax.set_xlim(-75, 420)
    ax.set_ylim(-0.6, 1.4)
    ax.axis("off")
    return fig


@register("sb.alltoall-overlap", "transpose-matrices")
def transpose_matrices(p):
    """AllGather and ReduceScatter as matrices for p = 2 devices, n = 1 value per shard: one is the other's transpose."""
    ag = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    fig, ax = figure(2.2)
    blank(ax, (-0.5, 10.0), (-1.3, 4.6))
    c = 0.7

    def draw(M, x0, y0, title, sub):
        r, k = M.shape
        for i in range(r):
            for j in range(k):
                ax.add_patch(Rectangle((x0 + j * c, y0 + (r - 1 - i) * c), c, c,
                                       facecolor=p.c(0) if M[i, j] else p.surface, alpha=0.85 if M[i, j] else 1,
                                       edgecolor=p.muted, lw=0.7))
                ax.text(x0 + j * c + c / 2, y0 + (r - 1 - i) * c + c / 2, str(M[i, j]), ha="center", va="center",
                        fontsize=7.5, color=p.surface if M[i, j] else p.muted)
        ax.text(x0 + k * c / 2, y0 + r * c + 0.25, title, ha="center", fontsize=7.8, color=p.fg)
        ax.text(x0 + k * c / 2, y0 - 0.3, sub, ha="center", va="top", fontsize=6.6, color=p.muted)

    draw(ag, 1.0, 0.6, "AllGather (4×2)", "shards (a, b) →\n(a, b) on dev 0,\n(a, b) on dev 1")
    draw(ag.T, 5.6, 1.3, "ReduceScatter (2×4)", "partials (a0,b0,a1,b1)\n→ (a0+a1) on dev 0,\n(b0+b1) on dev 1")
    ax.text(4.3, 1.9, "transpose", ha="center", fontsize=7.5, color=p.label)
    arrow(ax, (3.2, 1.6), (5.4, 1.6), p, color=p.label, style="<|-|>")
    return fig
