"""Figures for llm.transformer-block (norm placement, residual-stream growth, FFN activations)."""
import numpy as np
from matplotlib.patches import Circle

from figures.style import arrow, blank, box, figure, register


def _plus(ax, xy, p, color):
    ax.add_patch(Circle(xy, 0.22, facecolor=p.surface, edgecolor=color, lw=1.1, zorder=4))
    ax.text(xy[0], xy[1] - 0.01, "+", ha="center", va="center", fontsize=8.5, color=color, zorder=5)


def _panel(ax, x0, title, branch, post_norm, p, highlight=True, parallel=False):
    """One block drawn bottom (input) to top (output).

    branch: list of box labels on the sublayer branch, bottom to top.
    post_norm: True if a Norm sits on the main (residual) line after the add (Post-LN).
    """
    cx = x0 + 1.25          # main residual line
    bx = x0 + 0.55          # branch column (left of main line)
    good = p.good if highlight else p.muted
    bad = p.bad if highlight else p.muted
    y_in, y_add = 0.35, 5.0
    ax.text(cx - 0.35, 7.75, title, ha="center", fontsize=7.4, color=p.fg)
    ax.text(cx, y_in - 0.25, "$x_l$", ha="center", va="top", fontsize=7.5, color=p.muted)
    # skip (identity) line
    lw_skip = 2.0 if highlight else 1.0
    skip_col = bad if post_norm else good
    ax.plot([cx, cx], [y_in, y_add - 0.22], color=skip_col, lw=lw_skip, zorder=1, solid_capstyle="butt")
    # branch boxes
    ax.plot([cx, bx], [0.85, 0.85], color=p.muted, lw=0.9)
    if parallel:
        box(ax, (bx - 0.45, 1.1), 0.9, 0.55, branch[0], p, fontsize=6.2, color=p.c(2))
        arrow(ax, (bx, 0.85), (bx, 1.1), p)
        for dx, lab in [(-0.42, "Attn"), (0.42, "MLP")]:
            box(ax, (bx + dx - 0.36, 2.4), 0.72, 0.9, lab, p, fontsize=5.8, color=p.c(0))
            arrow(ax, (bx, 1.65), (bx + dx, 2.4), p)
            arrow(ax, (bx + dx, 3.3), (cx - 0.2, y_add - 0.1), p)
    else:
        ys = np.linspace(1.15, 3.7, len(branch)) if len(branch) > 1 else [2.2]
        prev = 0.85
        for y, lab in zip(ys, branch):
            col = p.c(2) if lab == "Norm" else p.c(0)
            box(ax, (bx - 0.45, y), 0.9, 0.62, lab, p, fontsize=6.2, color=col)
            arrow(ax, (bx, prev), (bx, y), p)
            prev = y + 0.62
        arrow(ax, (bx, prev), (cx - 0.2, y_add - 0.08), p)
    _plus(ax, (cx, y_add), p, p.fg)
    if post_norm:
        box(ax, (cx - 0.45, 5.55), 0.9, 0.62, "Norm", p, fontsize=6.2, color=p.c(2))
        ax.plot([cx, cx], [y_add + 0.22, 5.55], color=bad, lw=lw_skip)
        ax.plot([cx, cx], [6.17, 6.75], color=bad, lw=lw_skip)
    else:
        ax.plot([cx, cx], [y_add + 0.22, 6.75], color=good, lw=lw_skip)
    ax.text(cx, 6.85, "$x_{l+1}$", ha="center", va="bottom", fontsize=7.5, color=p.muted)


@register("llm.transformer-block", "norm-placement")
def norm_placement(p):
    """Four block layouts; the residual (identity) line is green where it is clean, red where a Norm interrupts it."""
    fig, ax = figure(2.75)
    blank(ax, (-0.1, 10.1), (-0.5, 8.2))
    _panel(ax, 0.0, "Post-LN", ["F"], True, p)
    _panel(ax, 2.55, "Pre-LN", ["Norm", "F"], False, p)
    _panel(ax, 5.1, "Sandwich", ["Norm", "F", "Norm"], False, p)
    _panel(ax, 7.65, "Parallel", ["Norm"], False, p, parallel=True)
    return fig


@register("llm.transformer-block", "norm-quiz")
def norm_quiz(p):
    """[fig-Q] Four unlabelled layouts A-D (Pre-LN, parallel, Post-LN, norm-on-output), no highlighting."""
    fig, ax = figure(2.75)
    blank(ax, (-0.1, 10.1), (-0.5, 8.2))
    _panel(ax, 0.0, "A", ["Norm", "F"], False, p, highlight=False)
    _panel(ax, 2.55, "B", ["Norm"], False, p, highlight=False, parallel=True)
    _panel(ax, 5.1, "C", ["F"], True, p, highlight=False)
    _panel(ax, 7.65, "D", ["F", "Norm"], False, p, highlight=False)
    return fig


def _ln(x):
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / s


def _simulate(L=32, d=512, n=32, seeds=8):
    """Xiong et al. (2020) init setting: W_Q = W_K = 0 (uniform attention), Xavier W_V, W1, W2, ReLU FFN.

    Returns E||x_l||^2 / d after each full layer for Pre-LN, and the relative size of each layer's update.
    """
    sq = np.zeros(L + 1)
    rel = np.zeros(L)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        x = rng.normal(0, 1, (n, d))
        sq[0] += (x ** 2).sum(-1).mean() / d
        for l in range(L):
            Wv, W1, W2 = (rng.normal(0, np.sqrt(1 / d), (d, d)) for _ in range(3))
            a = _ln(x) @ Wv
            att = np.repeat(a.mean(0, keepdims=True), n, axis=0)   # uniform attention over n tokens
            x3 = x + att
            ffn = np.maximum(_ln(x3) @ W1, 0) @ W2
            upd = att + ffn
            rel[l] += np.linalg.norm(upd, axis=-1).mean() / np.linalg.norm(x, axis=-1).mean()
            x = x3 + ffn
            sq[l + 1] += (x ** 2).sum(-1).mean() / d
    return sq / seeds, rel / seeds


@register("llm.transformer-block", "residual-growth")
def residual_growth(p):
    """Simulated Pre-LN residual stream at init: squared norm grows linearly in depth, each layer's relative update shrinks."""
    L = 32
    sq, rel = _simulate(L)
    ls = np.arange(L + 1)
    fig, (a1, a2) = figure(2.3, ncols=2)
    a1.fill_between(ls, 1 + ls / 2, 1 + 1.5 * ls, color=p.faint, lw=0)
    a1.plot(ls, sq, color=p.c(0), label="Pre-LN (sim.)")
    a1.axhline(1, color=p.c(1), lw=1.4)
    a1.text(L, 2.5, "Post-LN: 1", color=p.c(1), fontsize=7, ha="right")
    a1.text(3, 40, "Xiong bounds", color=p.muted, fontsize=6.8)
    a1.text(14, 9, "Pre-LN", color=p.c(0), fontsize=7.2)
    a1.set_xlabel("layer $l$")
    a1.set_ylabel(r"$\mathbb{E}\|x_l\|^2 / d$")
    a1.set_xlim(0, L)
    a1.set_ylim(0, 50)
    l1 = np.arange(1, L + 1)
    a2.plot(l1, rel, color=p.c(0))
    ref = rel[-1] * np.sqrt(L) / np.sqrt(l1)
    a2.plot(l1, ref, color=p.muted, lw=1, ls="--")
    a2.text(21, 0.31, r"$\propto 1/\sqrt{l}$", color=p.muted, fontsize=7.2)
    a2.set_xlabel("layer $l$")
    a2.set_ylabel(r"$\|\Delta x_l\| / \|x_l\|$")
    a2.set_xlim(0, L)
    a2.set_ylim(0, 0.8)
    return fig


@register("llm.transformer-block", "activations")
def activations(p):
    """ReLU, GELU and SiLU (Swish-1), plus the SiLU derivative."""
    from math import erf, sqrt
    x = np.linspace(-4, 3, 400)
    relu = np.maximum(x, 0)
    gelu = x * 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))
    sig = 1 / (1 + np.exp(-x))
    silu = x * sig
    dsilu = sig * (1 + x * (1 - sig))
    fig, ax = figure(2.2)
    ax.axhline(0, color=p.faint, lw=0.8)
    ax.axvline(0, color=p.faint, lw=0.8)
    ax.plot(x, relu, color=p.muted, lw=1.3)
    ax.plot(x, gelu, color=p.c(2), lw=1.6)
    ax.plot(x, silu, color=p.c(0), lw=1.8)
    ax.plot(x, dsilu, color=p.c(1), lw=1.3, ls="--")
    i = silu.argmin()
    ax.plot([x[i]], [silu[i]], "o", color=p.c(0), ms=3.5)
    ax.annotate(f"min {silu[i]:.3f} at {x[i]:.2f}", (x[i], silu[i]), xytext=(-3.9, -0.75),
                fontsize=6.8, color=p.c(0), arrowprops=dict(arrowstyle="-", color=p.c(0), lw=0.7))
    ax.text(1.55, 2.95, "ReLU", color=p.muted, fontsize=7.2)
    ax.text(0.75, 2.35, "GELU", color=p.c(2), fontsize=7.2)
    ax.text(2.05, 1.55, "SiLU", color=p.c(0), fontsize=7.2)
    ax.text(-3.9, 1.25, "SiLU$'$ (overshoots 1)", color=p.c(1), fontsize=7.2)
    ax.set_xlim(-4, 3)
    ax.set_ylim(-0.9, 3.1)
    ax.set_xlabel("$x$")
    return fig
