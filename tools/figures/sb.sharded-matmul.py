"""Figures for sb.sharded-matmul (Scaling Book ch. 3, Cases 1-4)."""
from matplotlib.patches import Rectangle

from figures.style import arrow, blank, box, figure, register


@register("sb.sharded-matmul", "four-cases")
def four_cases(p):
    """Decision flow for A[...]·B[...] with contracted dim J: which communication is needed."""
    fig, ax = figure(3.2)
    blank(ax, (-0.2, 10.3), (0, 8.4))
    box(ax, (2.6, 7.0), 4.8, 1.0, "Is the contracted dim J\nsharded in A or B?", p, fontsize=7.5, color=p.fg)
    box(ax, (0.1, 4.5), 4.2, 1.4, "No. Do two\nnon-contracted dims\nshare a mesh axis?", p, fontsize=6.8, color=p.fg)
    box(ax, (5.7, 4.5), 4.2, 1.4, "Yes. Is J sharded\nthe same way in\nboth inputs?", p, fontsize=6.8, color=p.fg)
    arrow(ax, (4.0, 7.0), (2.4, 5.85), p)
    arrow(ax, (6.0, 7.0), (7.6, 5.85), p)
    # leaves
    box(ax, (0.0, 1.2), 2.2, 2.3, "Case 1\nlocal\nmatmul,\nno comms", p, fontsize=6.5, color=p.good)
    box(ax, (2.45, 1.2), 2.2, 2.3, "Case 4\nAllGather\none input\nfirst", p, fontsize=6.5, color=p.bad)
    box(ax, (5.35, 1.2), 2.2, 2.3, "Case 2\nAllGather\nthe sharded\ninput", p, fontsize=6.5, color=p.c(1))
    box(ax, (7.8, 1.2), 2.2, 2.3, "Case 3\npartial sums,\nthen AllReduce\nor Reduce-\nScatter", p, fontsize=6.3,
        color=p.c(2))
    arrow(ax, (1.4, 4.6), (1.1, 3.55), p)
    arrow(ax, (3.0, 4.6), (3.5, 3.55), p)
    arrow(ax, (6.9, 4.6), (6.45, 3.55), p)
    arrow(ax, (8.6, 4.6), (8.9, 3.55), p)
    for x, y, t in [(0.75, 4.05, "no"), (3.65, 4.05, "yes"), (6.15, 4.05, "no"), (9.3, 4.05, "yes")]:
        ax.text(x, y, t, fontsize=7, color=p.muted, ha="center")
    ax.text(5.0, 0.45, "Case 4 can co-occur with 2 or 3: fix it first", ha="center", fontsize=6.8,
            color=p.muted)
    return fig


@register("sb.sharded-matmul", "partial-sums")
def partial_sums(p):
    """Case 3 on two devices: each multiplies its slice of the contracted dim into a full-size partial sum {U_X}."""
    fig, ax = figure(2.2)
    blank(ax, (0, 10), (-0.9, 3.4))
    for d, x0 in enumerate([0.2, 5.3]):
        ax.text(x0 + 2.0, 3.1, f"device x={d}", ha="center", fontsize=8, color=p.fg)
        ax.add_patch(Rectangle((x0, 0.6), 0.65, 2.0, facecolor=p.surface, edgecolor=p.c(0), lw=1.2))
        ax.text(x0 + 0.33, 1.6, f"$A_{{:,{d}}}$", ha="center", va="center", fontsize=7.5, color=p.fg)
        ax.text(x0 + 0.85, 1.6, "·", ha="center", va="center", fontsize=13, color=p.fg)
        ax.add_patch(Rectangle((x0 + 1.05, 1.25), 1.65, 0.7, facecolor=p.surface, edgecolor=p.c(0), lw=1.2))
        ax.text(x0 + 1.88, 1.6, f"$B_{{{d},:}}$", ha="center", va="center", fontsize=7.5, color=p.fg)
        ax.text(x0 + 2.95, 1.6, "=", ha="center", va="center", fontsize=10, color=p.fg)
        ax.add_patch(Rectangle((x0 + 3.2, 0.6), 0.95, 2.0, facecolor=p.surface, edgecolor=p.c(1), lw=1.2))
        ax.text(x0 + 3.67, 1.6, f"$C^{{({d})}}$", ha="center", va="center", fontsize=7.5, color=p.fg)
    ax.text(5.0, -0.1, "$C = C^{(0)} + C^{(1)}$; each device holds $C\\,\\{U_X\\}$",
            ha="center", fontsize=7.3, color=p.label)
    ax.text(5.0, -0.65, "AllReduce or ReduceScatter over X sums them", ha="center", fontsize=7,
            color=p.muted)
    return fig


@register("sb.sharded-matmul", "case4-diagonal")
def case4_diagonal(p):
    """Case 4: A[I_X, J]·B[J, K_X] on 2 devices. Device x holds row block x and column block x, so only C's diagonal blocks are computable."""
    fig, ax = figure(2.0)
    blank(ax, (0, 10), (-0.6, 3.2))
    cols = [p.c(0), p.c(1)]
    for x in range(2):
        ax.add_patch(Rectangle((0.3, 1.4 - x * 1.2), 1.8, 1.2, facecolor=cols[x], alpha=0.75, edgecolor=p.surface))
        ax.text(1.2, 2.0 - x * 1.2, f"rows\ndev {x}", ha="center", va="center", fontsize=6.8, color=p.surface)
        ax.add_patch(Rectangle((3.0 + x * 1.2, 0.2), 1.2, 2.4, facecolor=cols[x], alpha=0.75, edgecolor=p.surface))
        ax.text(3.6 + x * 1.2, 1.4, f"cols\non\ndev {x}", ha="center", va="center", fontsize=6.8, color=p.surface)
    ax.text(1.2, 2.85, "$A[I_X, J]$", ha="center", fontsize=8, color=p.fg)
    ax.text(4.2, 2.85, "$B[J, K_X]$", ha="center", fontsize=8, color=p.fg)
    for x in range(2):
        for y in range(2):
            ok = x == y
            ax.add_patch(Rectangle((6.4 + y * 1.2, 1.4 - x * 1.2), 1.2, 1.2,
                                   facecolor=cols[x] if ok else p.surface, alpha=0.75 if ok else 1,
                                   edgecolor=p.muted, ls="-" if ok else "--"))
            ax.text(7.0 + y * 1.2, 2.0 - x * 1.2, f"dev {x}" if ok else "no one\nhas\nboth", ha="center", va="center",
                    fontsize=6.8, color=p.surface if ok else p.bad)
    ax.text(7.6, 2.85, "$C$ blocks", ha="center", fontsize=8, color=p.fg)
    ax.text(5.0, -0.25, "only the diagonal blocks of C are computable",
            ha="center", fontsize=6.8, color=p.label)
    return fig
