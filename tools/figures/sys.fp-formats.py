"""Figures for sys.fp-formats: bit layouts, representable ranges, and ulp spacing."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register

# name, exponent bits, mantissa bits, note
FORMATS = [
    ("fp32", 8, 23, ""),
    ("tf32", 8, 10, "(19 bits used, in-core only)"),
    ("bf16", 8, 7, ""),
    ("fp16", 5, 10, ""),
    ("E5M2", 5, 2, ""),
    ("E4M3", 4, 3, ""),
]


@register("sys.fp-formats", "bit-layouts")
def bit_layouts(p):
    """One cell per bit, aligned on the exponent/mantissa boundary (x = 0)."""
    fig, ax = figure(2.75)
    row_h, gap = 0.62, 0.42
    n = len(FORMATS)
    blank(ax, (-20.5, 23.6), (-0.9, n * (row_h + gap) + 0.1))
    ax.set_aspect("auto")
    for i, (name, e, m, note) in enumerate(FORMATS):
        y = (n - 1 - i) * (row_h + gap)
        # sign
        ax.add_patch(Rectangle((-e - 1, y), 1, row_h, facecolor=p.c(3), edgecolor=p.faint, lw=0.4))
        for k in range(e):
            ax.add_patch(Rectangle((-e + k, y), 1, row_h, facecolor=p.c(0), edgecolor=p.faint, lw=0.4))
        for k in range(m):
            ax.add_patch(Rectangle((k, y), 1, row_h, facecolor=p.c(2), edgecolor=p.faint, lw=0.4))
        ax.text(-20.3, y + row_h / 2, name, fontsize=8, color=p.fg, va="center", ha="left", weight="bold")
        ax.text(-15.9, y + row_h / 2, f"1·{e}·{m}", fontsize=7, color=p.muted, va="center", ha="left",
                family=p.mono)
        if note:
            ax.text(m + 0.6, y + row_h / 2, note, fontsize=6.6, color=p.muted, va="center", ha="left")
    top = n * (row_h + gap) - gap
    ax.plot([0, 0], [-0.15, top + 0.05], color=p.label, lw=0.9, ls="--")
    ax.text(0, -0.55, "binary point (after the hidden 1)", fontsize=6.8, color=p.label, ha="center", va="center")
    # key
    ax.set_ylim(-1.55, top + 0.15)
    for x, c, lab in [(-19.8, p.c(3), "sign"), (-12.6, p.c(0), "exponent: range"), (5.2, p.c(2), "mantissa: precision")]:
        ax.add_patch(Rectangle((x, -1.35), 0.9, 0.42, facecolor=c, edgecolor="none"))
        ax.text(x + 1.3, -1.14, lab, fontsize=7, color=p.fg, va="center")
    return fig


def _ranges():
    # (name, min subnormal, min normal, max) as log10 values
    out = []
    for name, e, m, mx in [("fp32", 8, 23, None), ("bf16", 8, 7, None), ("fp16", 5, 10, None),
                           ("E5M2", 5, 2, None), ("E4M3", 4, 3, 448.0)]:
        bias = 2 ** (e - 1) - 1
        emax = 2 ** e - 2 - bias
        top = mx if mx else (2 - 2.0 ** -m) * 2.0 ** emax
        out.append((name, np.log10(2.0 ** (1 - bias - m)), np.log10(2.0 ** (1 - bias)), np.log10(top)))
    return out


@register("sys.fp-formats", "number-line")
def number_line(p):
    """Representable magnitudes per format on a log axis, with an illustrative gradient band."""
    fig, ax = figure(2.6)
    rows = _ranges()
    n = len(rows)
    # illustrative band of small gradient magnitudes
    ax.axvspan(-11, -3, color=p.label, alpha=0.13, lw=0)
    ax.text(-7, n - 0.25, "typical small gradients\n(illustrative)", fontsize=6.8, color=p.label,
            ha="center", va="center")
    for i, (name, lsub, lnorm, ltop) in enumerate(rows):
        y = n - 1 - i - 0.35
        ax.plot([lsub, lnorm], [y, y], color=p.c(i), lw=5, alpha=0.4, solid_capstyle="butt")
        ax.plot([lnorm, ltop], [y, y], color=p.c(i), lw=5, solid_capstyle="butt")
        ax.text(-59.5, y, name, fontsize=8, color=p.fg, va="center", ha="left", weight="bold")
    # annotate fp16 underflow limit
    y16 = n - 1 - 2 - 0.35
    ax.annotate("fp16 floor ≈ 6e−8", (np.log10(2.0 ** -24), y16), xytext=(-31, y16 - 0.62),
                fontsize=6.8, color=p.fg, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.7))
    ax.annotate("max 65504", (np.log10(65504), y16), xytext=(20, y16 - 0.62), fontsize=6.8, color=p.fg,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.7))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-60, 40)
    ax.set_ylim(-0.9, n + 0.1)
    ticks = [-40, -30, -20, -10, 0, 10, 20, 30]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_xlabel("magnitude (log scale); faded = subnormal")
    return fig


def _ulp(x, e, m):
    bias = 2 ** (e - 1) - 1
    emin = 1 - bias
    ex = np.floor(np.log2(x))
    ex = np.maximum(ex, emin)
    return 2.0 ** (ex - m)


@register("sys.fp-formats", "spacing")
def spacing(p):
    """Gap between neighbouring representable numbers against magnitude (log-log staircase)."""
    fig, ax = figure(2.6)
    x = np.logspace(-9, 6, 6000)
    for i, (name, e, m, xmax, col) in enumerate([("bf16", 8, 7, 3.39e38, p.c(1)),
                                                 ("fp16", 5, 10, 65504.0, p.c(0))]):
        xs = x[x <= xmax]
        ax.loglog(xs, _ulp(xs, e, m), color=col, lw=1.5, drawstyle="steps-post")
    ax.plot([65504], [_ulp(np.array([65504.0]), 5, 10)[0]], "x", color=p.bad, ms=6, mew=1.5)
    ax.text(3.0e4, 1.6e-3, "fp16\noverflows", fontsize=6.8, color=p.bad, ha="center")
    ax.text(3e-9, 1.2e-7, "fp16 subnormals:\nfixed gap $2^{-24}$", fontsize=6.8, color=p.c(0))
    ax.text(1.2e-4, 1.6e-9, "fp16 normals: gap $\\approx x\\cdot2^{-10}$", fontsize=6.8, color=p.c(0))
    ax.text(1.3e-6, 3e-4, "bf16: gap $\\approx x\\cdot2^{-7}$\n(8× coarser)", fontsize=6.8, color=p.c(1))
    ax.axvline(1, color=p.faint, lw=0.8)
    ax.text(1.15, 2e-12, "x = 1", fontsize=6.8, color=p.muted)
    ax.set_xlabel("magnitude $x$")
    ax.set_ylabel("gap to next number (ulp)")
    ax.set_xlim(1e-9, 1e6)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
    return fig
