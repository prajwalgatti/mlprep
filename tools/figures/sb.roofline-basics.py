"""Figures for sb.roofline-basics (Scaling Book ch. 1)."""
import numpy as np

from figures.style import figure, register

PEAK = 1.97e14      # TPU v5e bf16 FLOPs/s
HBM = 8.2e11        # TPU v5e HBM bytes/s
VMEM = 22 * HBM     # book's assumption: VMEM ~22x HBM bandwidth


@register("sb.roofline-basics", "roofline")
def roofline(p):
    """Log-log roofline for TPU v5e with two memory bandwidths and a few example algorithms."""
    I = np.logspace(-1, 4, 400)
    fig, ax = figure(2.7)
    hbm = np.minimum(PEAK, I * HBM) / 1e12
    vmem = np.minimum(PEAK, I * VMEM) / 1e12
    ax.loglog(I, vmem, color=p.c(2), lw=1.4, ls="--")
    ax.loglog(I, hbm, color=p.c(0), lw=2)
    ax.axhline(PEAK / 1e12, color=p.faint, lw=0.6, zorder=0)
    r1, r2 = PEAK / HBM, PEAK / VMEM
    ax.plot([r1], [PEAK / 1e12], "o", color=p.c(0), ms=4)
    ax.plot([r2], [PEAK / 1e12], "o", color=p.c(2), ms=4)
    ax.annotate(f"ridge ≈ {r1:.0f}", (r1, PEAK / 1e12), xytext=(380, 10), color=p.c(0), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=p.c(0), lw=0.7))
    ax.annotate(f"ridge ≈ {r2:.0f}", (r2, PEAK / 1e12), xytext=(22, 380), color=p.c(2), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=p.c(2), lw=0.7))
    ax.text(0.13, 90, "— HBM bandwidth", color=p.c(0), fontsize=7.5, va="center")
    ax.text(0.13, 300, "-- VMEM (22× HBM)", color=p.c(2), fontsize=7.5, va="center")
    # example algorithms, evaluated against HBM
    for x, lab, dy in [(0.5, "I = 0.5", 0.3), (60, "I = 60", 0.35), (1000, "I = 1000", 1.55)]:
        y = min(PEAK, x * HBM) / 1e12
        good = x * HBM >= PEAK
        ax.plot([x], [y], "s", color=p.good if good else p.bad, ms=4.5, zorder=5)
        ax.text(x, y * dy, lab, color=p.good if good else p.bad, fontsize=7, ha="center",
                va="top" if dy < 1 else "bottom")
    ax.text(4.5e3, PEAK / 1e12 * 0.7, "peak\n197", color=p.muted, fontsize=7, ha="center", va="top")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.03, 900)
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("attainable TFLOP/s")
    return fig


@register("sb.roofline-basics", "overlap")
def overlap(p):
    """Serial vs perfectly overlapped execution of compute and communication."""
    fig, ax = figure(1.8)
    tm, tc = 3.0, 2.0
    rows = [("no overlap\n(upper bound)", [(0, tc, p.c(1), "comms 2 ms"), (tc, tm, p.c(0), "math 3 ms")]),
            ("full overlap\n(lower bound)", [(0, tm, p.c(0), "math 3 ms"), (0, tc, p.c(1), "comms 2 ms")])]
    for r, (name, segs) in enumerate(rows):
        y = 1 - r
        for k, (x0, w, col, lab) in enumerate(segs):
            off = 0.0 if r == 0 else (0.17 if k == 0 else -0.17)
            h = 0.36 if r == 0 else 0.3
            ax.barh(y + off, w, left=x0, height=h, color=col, alpha=0.85, edgecolor="none")
            ax.text(x0 + w / 2, y + off, lab, ha="center", va="center", fontsize=7, color=p.surface)
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=7.5, color=p.fg)
    ax.axvline(5, color=p.muted, lw=0.7, ls=":")
    ax.axvline(3, color=p.muted, lw=0.7, ls=":")
    ax.text(5.08, 1.42, "sum = 5 ms", fontsize=7, color=p.muted, ha="left")
    ax.text(3.08, -0.48, "max = 3 ms", fontsize=7, color=p.muted, ha="left")
    ax.set_xlim(-2.3, 6.4)
    ax.set_ylim(-0.6, 1.55)
    ax.axis("off")
    return fig
