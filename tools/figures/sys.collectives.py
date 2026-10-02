"""Figures for sys.collectives: what each collective does to 4 ranks' buffers, the RS+AG decomposition,
the ring all-reduce schedule (simulated), and an unlabelled before/after puzzle for a quiz question."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import arrow, blank, figure, register

P = 4          # ranks
CELL = 0.36    # cell size in data units


def _canvas(h):
    """Diagram figure without constrained layout (it collapses when wide text is present)."""
    fig, ax = figure(h)
    fig.set_layout_engine(None)
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    return fig, ax


def _cell(ax, x, y, p, kind, k=None, s=CELL):
    """Draw one buffer cell. kind: None (empty slot), ('own', r) = rank r's data,
    'sum' = reduced over all ranks (four colour stripes), ('stale', r) = left untouched (faded)."""
    if kind is None:
        ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92, facecolor="none", edgecolor=p.faint, lw=0.7))
        return
    if kind == "sum":
        w = s * 0.92 / P
        for r in range(P):
            ax.add_patch(Rectangle((x + r * w, y), w, s * 0.92, facecolor=p.c(r), edgecolor="none"))
        ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92, facecolor="none", edgecolor=p.fg, lw=0.8))
        return
    tag, r = kind[0], kind[1]
    if tag == "own":
        ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92, facecolor=p.c(r), edgecolor="none"))
        if len(kind) > 2:   # chunk index label
            ax.text(x + s * 0.46, y + s * 0.44, str(kind[2]), ha="center", va="center", fontsize=6.4,
                    color=p.surface)
    else:  # stale
        ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92, facecolor=p.c(r), edgecolor="none", alpha=0.25))


def _grid(ax, x0, y0, rows, p):
    """rows[r] = list of 4 cell kinds for rank r (rank 0 at top)."""
    for r in range(P):
        for k in range(P):
            _cell(ax, x0 + k * CELL, y0 + (P - 1 - r) * CELL, p, rows[r][k], k)


def _panel(ax, x0, y0, title, before, after, p, rank_labels=False):
    ax.text(x0 + 1.55, y0 + P * CELL + 0.12, title, ha="center", fontsize=7.8, color=p.fg)
    _grid(ax, x0, y0, before, p)
    _grid(ax, x0 + 1.75, y0, after, p)
    arrow(ax, (x0 + P * CELL + 0.02, y0 + P * CELL / 2), (x0 + 1.73, y0 + P * CELL / 2), p, lw=0.9)
    if rank_labels:
        for r in range(P):
            ax.text(x0 - 0.08, y0 + (P - 1 - r) * CELL + CELL * 0.45, f"r{r}", ha="right", va="center",
                    fontsize=6.8, color=p.muted)


def _own_full(r):
    return [("own", r)] * P


@register("sys.collectives", "collectives-grid")
def collectives_grid(p):
    """Before → after for six collectives on 4 ranks. Rows = ranks, columns = chunks of the buffer."""
    fig, ax = _canvas(3.55)
    blank(ax, (-0.35, 7.55), (-1.25, 6.15))
    E = None
    full = [_own_full(r) for r in range(P)]
    panels = [
        ("broadcast (root r0)", [_own_full(0)] + [[E] * P for _ in range(3)], [_own_full(0)] * P),
        ("reduce (root r0)", full, [["sum"] * P] + [[("stale", r)] * P for r in range(1, P)]),
        ("all-reduce", full, [["sum"] * P for _ in range(P)]),
        ("all-gather", [[("own", r, k) if k == r else E for k in range(P)] for r in range(P)],
         [[("own", k, k) for k in range(P)] for _ in range(P)]),
        ("reduce-scatter", full, [["sum" if k == r else E for k in range(P)] for r in range(P)]),
        ("all-to-all", [[("own", r, k) for k in range(P)] for r in range(P)],
         [[("own", k, j) for k in range(P)] for j in range(P)]),
    ]
    for i, (title, before, after) in enumerate(panels):
        col, row = i % 2, i // 2
        _panel(ax, 0.05 + col * 3.85, 4.15 - row * 2.1, title, before, after, p, rank_labels=(col == 0))
    # all-to-all: show that rank j's k-th chunk is the j-th chunk of rank k (transpose) with a small marker
    ax.text(3.6, -0.2, "colour = rank the data came from; digit = chunk index\nstriped = summed over all 4 ranks\n"
            "all-to-all: chunk k of rank r goes to rank k", ha="center", va="top", fontsize=6.9,
            color=p.label, linespacing=1.35)
    return fig


@register("sys.collectives", "allreduce-decomposition")
def allreduce_decomposition(p):
    """All-reduce = reduce-scatter (each rank ends with one fully summed chunk) then all-gather."""
    fig, ax = _canvas(1.75)
    blank(ax, (-0.4, 7.6), (-0.55, 2.0))
    E = None
    full = [_own_full(r) for r in range(P)]
    mid = [["sum" if k == r else E for k in range(P)] for r in range(P)]
    end = [["sum"] * P for _ in range(P)]
    xs = [0.1, 2.85, 5.6]
    for x, rows, lab in zip(xs, [full, mid, end], ["start: 4 different\nbuffers of S bytes", "after reduce-scatter",
                                                     "after all-gather"]):
        _grid(ax, x, 0.0, rows, p)
        ax.text(x + P * CELL / 2, -0.12, lab, ha="center", va="top", fontsize=6.9, color=p.muted)
    for r in range(P):
        ax.text(-0.02, (P - 1 - r) * CELL + CELL * 0.45, f"r{r}", ha="right", va="center", fontsize=6.8,
                color=p.muted)
    arrow(ax, (xs[0] + P * CELL + 0.05, 0.72), (xs[1] - 0.08, 0.72), p, lw=0.9)
    arrow(ax, (xs[1] + P * CELL + 0.05, 0.72), (xs[2] - 0.08, 0.72), p, lw=0.9)
    ax.text((xs[0] + xs[1] + P * CELL) / 2, 0.85, "RS", ha="center", fontsize=7.5, color=p.label)
    ax.text((xs[1] + xs[2] + P * CELL) / 2, 0.85, "AG", ha="center", fontsize=7.5, color=p.label)
    ax.text(3.6, 1.72, r"each phase: $\frac{P-1}{P}S$ per rank; total $2\frac{P-1}{P}S$",
            ha="center", fontsize=7.5, color=p.fg)
    return fig


def _ring_schedule(n=P):
    """Simulate the ring all-reduce (Patarasuk & Yuan's indexing). Returns a list of snapshots
    (counts[rank, chunk] = how many ranks' contributions are in that copy, sent[(rank, chunk)])."""
    counts = np.ones((n, n), dtype=int)
    snaps = [(counts.copy(), set(), "start")]
    for j in range(1, n):                       # reduce-scatter: rank i sends chunk (i - j) mod n to i+1
        sends = [(i, (i - j) % n) for i in range(n)]
        new = counts.copy()
        for i, c in sends:
            new[(i + 1) % n, c] = counts[(i + 1) % n, c] + counts[i, c]
        counts = new
        snaps.append((counts.copy(), set(sends), f"RS step {j}"))
    for j in range(1, n):                       # all-gather: rank i forwards chunk (i - j + 1) mod n
        sends = [(i, (i - j + 1) % n) for i in range(n)]
        new = counts.copy()
        for i, c in sends:
            new[(i + 1) % n, c] = counts[i, c]
        counts = new
        snaps.append((counts.copy(), set(sends), f"AG step {j}"))
    return snaps


@register("sys.collectives", "ring-allreduce")
def ring_allreduce(p):
    """Ring all-reduce on 4 ranks. Number = how many ranks' contributions a copy of that chunk holds.
    Outlined = the copy each rank sends to its right neighbour during this step."""
    snaps = _ring_schedule()
    assert (snaps[-1][0] == P).all()
    fig, ax = _canvas(3.0)
    blank(ax, (-0.45, 7.6), (-0.95, 4.85))
    s = 0.4
    pos = [(0.15, 2.65), (2.05, 2.65), (3.95, 2.65), (5.85, 2.65), (2.05, 0.15), (3.95, 0.15), (5.85, 0.15)]
    for (counts, sent, lab), (x0, y0) in zip(snaps, pos):
        ax.text(x0 + 2 * s, y0 + 4 * s + 0.12, lab, ha="center", fontsize=7.4, color=p.fg)
        for r in range(P):
            for k in range(P):
                x, y = x0 + k * s, y0 + (P - 1 - r) * s
                n = counts[r, k]
                done = n == P
                ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92,
                                       facecolor=p.accent if done else p.surface,
                                       edgecolor=p.label if (r, k) in sent else p.faint,
                                       lw=1.5 if (r, k) in sent else 0.6))
                ax.text(x + s * 0.46, y + s * 0.44, str(n), ha="center", va="center", fontsize=7.2,
                        color=p.surface if done else p.fg)
    for r in range(P):
        for x0 in (0.15, 2.05):
            y0 = 2.65 if x0 == 0.15 else 0.15
            ax.text(x0 - 0.06, y0 + (P - 1 - r) * s + s * 0.45, f"r{r}", ha="right", va="center", fontsize=6.6,
                    color=p.muted)
    # mini ring diagram in the empty slot
    cx, cy, R = 0.95, 0.95, 0.62
    ang = np.pi / 2 - np.arange(P) * np.pi / 2
    pts = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]
    for i in range(P):
        a, b = pts[i], pts[(i + 1) % P]
        arrow(ax, a, b, p, color=p.label, lw=0.9, connectionstyle="arc3,rad=-0.25")
        ax.text(*a, f"r{i}", ha="center", va="center", fontsize=7.2, color=p.fg,
                bbox=dict(boxstyle="circle,pad=0.18", fc=p.surface, ec=p.muted, lw=0.6))
    ax.text(3.6, -0.55, "columns = chunks of S/4; number = ranks summed in\n"
            "outlined = sent right this step; filled = complete",
            ha="center", va="center", fontsize=6.8, color=p.label, linespacing=1.35)
    return fig


@register("sys.collectives", "mystery")
def mystery(p):
    """Quiz figure: numeric buffers on 4 ranks before and after an unnamed collective (it is an all-to-all)."""
    fig, ax = _canvas(1.6)
    blank(ax, (-0.6, 7.6), (-0.15, 2.05))
    s = 0.42
    before = np.arange(1, 17).reshape(P, P)
    after = before.T
    for x0, M, lab in [(0.3, before, "before"), (4.6, after, "after")]:
        ax.text(x0 + 2 * s, 4 * s + 0.12, lab, ha="center", fontsize=7.8, color=p.fg)
        for r in range(P):
            for k in range(P):
                x, y = x0 + k * s, (P - 1 - r) * s
                ax.add_patch(Rectangle((x, y), s * 0.92, s * 0.92, facecolor=p.surface, edgecolor=p.muted, lw=0.6))
                ax.text(x + s * 0.46, y + s * 0.44, str(M[r, k]), ha="center", va="center", fontsize=7.6,
                        color=p.fg)
            ax.text(x0 - 0.08, (P - 1 - r) * s + s * 0.45, f"rank {r}", ha="right", va="center", fontsize=6.8,
                    color=p.muted)
    arrow(ax, (2.15, 0.84), (3.75, 0.84), p, lw=1.0)
    ax.text(2.95, 0.98, "?", ha="center", fontsize=9, color=p.label)
    return fig
