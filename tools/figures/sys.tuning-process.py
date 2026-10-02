"""Figures for sys.tuning-process (Tuning Playbook: the incremental strategy)."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


def _err_shallow(x):
    return 0.300 + 0.05 * (x + 2.3) ** 2


def _err_deep(x):
    return 0.260 + 0.06 * (x + 3.0) ** 2


@register("sys.tuning-process", "fixed-vs-tuned")
def fixed_vs_tuned(p):
    """Validation error vs log LR for two depths: a fixed LR reverses the tuned conclusion."""
    rng = np.random.default_rng(3)
    x = np.linspace(-4.2, -1.2, 300)
    fig, ax = figure(2.5)
    for f, col, name, lab_x in [(_err_shallow, p.c(1), "4 layers", -1.75), (_err_deep, p.c(0), "8 layers", -4.05)]:
        ax.plot(x, f(x), color=col, lw=1.8)
        xs = rng.uniform(-4.2, -1.2, 14)
        ax.plot(xs, f(xs) + rng.normal(0, 0.004, xs.size), "o", ms=3, color=col, alpha=0.75)
        i = f(x).argmin()
        ax.plot([x[i]], [f(x)[i]], "*", ms=10, color=col, zorder=5)
        ax.text(lab_x, f(np.array(lab_x)) + 0.012, name, color=col, fontsize=8,
                ha="left")
    ax.axvline(-2.0, color=p.muted, lw=0.9, ls="--")
    ax.text(-1.95, 0.232, "default LR", color=p.muted, fontsize=7.5, va="bottom")
    ax.annotate("at the default LR,\n4 layers looks better", (-2.0, _err_shallow(-2.0)),
                xytext=(-2.45, 0.368), fontsize=7.5, color=p.c(1), ha="center",
                arrowprops=dict(arrowstyle="-", color=p.c(1), lw=0.8))
    ax.annotate("tuned: 8 layers wins", (-3.0, 0.260), xytext=(-3.75, 0.236), fontsize=7.5,
                color=p.c(0), ha="left")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("validation error")
    ax.set_xlim(-4.25, -1.15)
    ax.set_ylim(0.225, 0.40)
    ax.set_xticks([-4, -3, -2])
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    return fig


@register("sys.tuning-process", "incremental-loop")
def incremental_loop(p):
    """The four-step incremental tuning loop."""
    fig, ax = figure(2.9)
    blank(ax, (0, 10), (0, 8))
    w, h = 4.3, 2.5
    pos = {1: (0.2, 5.2), 2: (5.5, 5.2), 3: (5.5, 0.3), 4: (0.2, 0.3)}
    text = {
        1: "1  Goal\none narrow question\n(does dropout help?)",
        2: "2  Design\nscientific, nuisance,\nfixed → studies",
        3: "3  Learn\nboundaries, density,\ncurves, best per value",
        4: "4  Launch?\nbeats retrain noise\nand worth complexity",
    }
    for k, (x0, y0) in pos.items():
        box(ax, (x0, y0), w, h, text[k], p, color=p.accent if k == 4 else None, fontsize=7.3)
    arrow(ax, (4.55, 6.45), (5.45, 6.45), p, lw=1.1)
    arrow(ax, (7.65, 5.15), (7.65, 2.85), p, lw=1.1)
    arrow(ax, (5.45, 1.55), (4.55, 1.55), p, lw=1.1)
    arrow(ax, (2.35, 2.85), (2.35, 5.15), p, lw=1.1)
    ax.text(5.0, 4.0, "explore for insight first;\nexploit (greedy) at the end",
            ha="center", va="center", fontsize=7.2, color=p.label)
    return fig
