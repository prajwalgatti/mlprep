"""Figures for fund.classification-metrics.

The classifier is a synthetic binormal scorer: negative scores ~ N(0, 1), positive scores ~ N(MU, 1).
Every curve below is computed exactly from that model (no sampling), so the shapes are illustrative.
"""
import numpy as np
from scipy.stats import norm

from figures.style import figure, register

MU = 1.5                            # separation of the positive score distribution
T = np.linspace(-5, 7, 2000)[::-1]  # thresholds, high to low, so recall increases along the arrays


def _rates(mu=MU):
    tpr = norm.sf(T, loc=mu)
    fpr = norm.sf(T)
    return fpr, tpr


def _precision(tpr, fpr, pi):
    return pi * tpr / (pi * tpr + (1 - pi) * fpr)


@register("fund.classification-metrics", "score-histograms")
def score_histograms(p):
    """One threshold on the score densities gives one (FPR, TPR) point; sweeping it traces the ROC."""
    t0 = 1.0
    s = np.linspace(-3.5, 5, 400)
    fig, (ax, ax2) = figure(2.0, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
    neg, pos = norm.pdf(s), norm.pdf(s, loc=MU)
    ax.plot(s, neg, color=p.c(1), lw=1.6, label="negatives")
    ax.plot(s, pos, color=p.c(0), lw=1.6, label="positives")
    ax.fill_between(s, pos, where=s >= t0, color=p.c(0), alpha=0.3, lw=0)
    ax.fill_between(s, neg, where=s >= t0, color=p.c(1), alpha=0.45, lw=0)
    ax.axvline(t0, color=p.label, lw=1.2)
    ax.text(t0 + 0.12, 0.43, "threshold $t$", color=p.label, fontsize=7.5)
    ax.text(2.6, 0.2, "TP", color=p.c(0), fontsize=7.5)
    ax.text(1.25, 0.03, "FP", color=p.fg, fontsize=7.5)
    ax.set_xlabel("score")
    ax.set_yticks([])
    ax.set_ylim(0, 0.48)
    ax.legend(loc="upper left", handlelength=1.2, fontsize=7)
    fpr, tpr = _rates()
    ax2.plot(fpr, tpr, color=p.fg, lw=1.6)
    ax2.plot([0, 1], [0, 1], color=p.muted, lw=0.8, ls=":")
    f0, r0 = norm.sf(t0), norm.sf(t0, loc=MU)
    ax2.plot([f0], [r0], "o", color=p.label, ms=5, zorder=5)
    ax2.annotate(f"$t$: ({f0:.2f}, {r0:.2f})", (f0, r0), xytext=(0.22, 0.45), fontsize=7.2, color=p.label,
                 arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR", labelpad=1)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_aspect("equal")
    return fig


def _roc_pr(p, names, baselines=True):
    """Same scorer, two prevalences: identical ROC, very different PR. `names` labels the two test sets."""
    fpr, tpr = _rates()
    fig, (ax, ax2) = figure(2.15, ncols=2)
    ax.plot(fpr, tpr, color=p.c(0), lw=2.6)
    ax.plot(fpr, tpr, color=p.c(1), lw=1.2, ls="--")
    ax.plot([0, 1], [0, 1], color=p.muted, lw=0.8, ls=":")
    ax.text(0.35, 0.3, "identical for both", color=p.fg, fontsize=7.2)
    ax.set_title("ROC", fontsize=8.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR", labelpad=1)
    for pi, color, ls, name, xy in [(0.5, p.c(0), "-", names[0], (0.3, 0.72)), (0.01, p.c(1), "--", names[1], (0.3, 0.3))]:
        prec = _precision(tpr, fpr, pi)
        ax2.plot(tpr, prec, color=color, lw=1.8, ls=ls)
        if baselines:   # a random scorer has precision = prevalence
            ax2.axhline(pi, color=color, lw=0.7, ls=":")
        ax2.text(*xy, name, color=color, fontsize=7.5)
    ax2.set_title("precision–recall", fontsize=8.5)
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision", labelpad=1)
    for a in (ax, ax2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_xticks([0, 0.5, 1])
        a.set_yticks([0, 0.5, 1])
    return fig


@register("fund.classification-metrics", "roc-pr-pair")
def roc_pr_pair(p):
    return _roc_pr(p, [r"$\pi=50\%$", r"$\pi=1\%$"])


@register("fund.classification-metrics", "roc-pr-mystery")
def roc_pr_mystery(p):
    """[fig-Q] The same pair with neutral labels and no baselines (they would give the answer away)."""
    return _roc_pr(p, ["test set A", "test set B"], baselines=False)


@register("fund.classification-metrics", "pr-interpolation")
def pr_interpolation(p):
    """Between two operating points, interpolate counts (TP, FP), not precision."""
    pos = 100
    tp_a, fp_a, tp_b, fp_b = 10, 2, 80, 120
    tp = np.linspace(tp_a, tp_b, 200)
    fp = fp_a + (fp_b - fp_a) / (tp_b - tp_a) * (tp - tp_a)
    rec, prec = tp / pos, tp / (tp + fp)
    ra, pa, rb, pb = tp_a / pos, tp_a / (tp_a + fp_a), tp_b / pos, tp_b / (tp_b + fp_b)
    fig, ax = figure(2.3)
    ax.plot([ra, rb], [pa, pb], color=p.bad, lw=1.4, ls="--", label="straight line (wrong)")
    ax.plot(rec, prec, color=p.good, lw=2, label="interpolate TP and FP (right)")
    ax.fill_between(rec, prec, np.interp(rec, [ra, rb], [pa, pb]), color=p.bad, alpha=0.15, lw=0)
    for r, pr, label in [(ra, pa, f"A: TP={tp_a}, FP={fp_a}"), (rb, pb, f"B: TP={tp_b}, FP={fp_b}")]:
        ax.plot([r], [pr], "o", color=p.fg, ms=5, zorder=5)
        ax.annotate(label, (r, pr), xytext=(6, 4), textcoords="offset points", fontsize=7.2, color=p.fg)
    ax.text(0.33, 0.73, "area overstated", color=p.bad, fontsize=7.2)
    ax.set_xlabel("recall (100 positives)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", handlelength=1.6)
    return fig


@register("fund.classification-metrics", "f1-contours")
def f1_contours(p):
    """Iso-F1 curves bend toward the axes: F1 stays low unless both P and R are high."""
    r = np.linspace(0.005, 1, 400)
    R, P = np.meshgrid(r, r)
    F = 2 * P * R / (P + R)
    fig, ax = figure(2.5)
    ax.contour(R, P, F, levels=[0.2, 0.4, 0.6, 0.8], colors=[p.c(0)], linewidths=1.2)
    for f in (0.2, 0.4, 0.6, 0.8):   # label each curve where it meets recall = 1, i.e. P = F/(2 - F)
        ax.text(1.03, f / (2 - f), f"$F_1$={f:.1f}", color=p.c(0), fontsize=7, va="center")
    ax.plot(r, 1 - r, color=p.muted, lw=1, ls="--")
    ax.text(0.36, 0.69, "arithmetic mean = 0.5", color=p.muted, fontsize=7, rotation=-45,
            rotation_mode="anchor")
    ax.plot([1.0], [0.02], "o", color=p.label, ms=5, clip_on=False, zorder=5)
    ax.text(1.03, -0.06, "flag all, π = 2%:\nmean 0.51, $F_1$ 0.04", color=p.label, fontsize=7, va="center")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    return fig
