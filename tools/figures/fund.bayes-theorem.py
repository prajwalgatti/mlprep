"""Figures for fund.bayes-theorem. All numbers are computed from Bayes' rule with the stated rates."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


def _grid(p, counts, labels, title):
    """1,000 people as dots, 40 columns x 25 rows, filled in order TP, FN, FP, TN."""
    cols, rows = 40, 25
    colors = [p.bad, p.label, p.c(2), p.faint]
    seq = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    fig, ax = figure(2.85)
    xs, ys, cs = [], [], []
    for k, cat in enumerate(seq):
        r, c = divmod(k, cols)
        xs.append(c)
        ys.append(rows - 1 - r)
        cs.append(colors[cat])
    ax.scatter(xs, ys, s=7.5, c=cs, linewidths=0)
    ax.set_xlim(-0.8, cols - 0.2)
    ax.set_ylim(-6.6, rows + 0.4)
    ax.axis("off")
    # 2x2 legend underneath
    for i, lab in enumerate(labels):
        x0 = 0 if i % 2 == 0 else 20
        y0 = -2.2 if i < 2 else -4.9
        ax.scatter([x0], [y0], s=22, c=[colors[i]], linewidths=0.6 if i == 3 else 0,
                   edgecolors=p.muted if i == 3 else "none")
        ax.text(x0 + 0.9, y0, lab, va="center", fontsize=7.6, color=p.fg)
    ax.text(cols / 2 - 0.5, rows + 0.1, title, ha="center", va="bottom", fontsize=7.8, color=p.muted)
    return fig


@register("fund.bayes-theorem", "base-rate-grid")
def base_rate_grid(p):
    """Prevalence 1%, sensitivity 99%, FPR 5%, rounded to whole people: 10 TP, 0 FN, 50 FP, 940 TN."""
    return _grid(p, [10, 0, 50, 940], ["sick, +", "sick, −", "healthy, +", "healthy, −"],
                 "1,000 people · prevalence 1% · sensitivity 99% · FPR 5%")


@register("fund.bayes-theorem", "base-rate-grid-q")
def base_rate_grid_q(p):
    """Prevalence 5%, sensitivity 90%, FPR 10%: 45 TP, 5 FN, 95 FP, 855 TN (exact)."""
    return _grid(p, [45, 5, 95, 855], ["sick, + (45)", "sick, − (5)", "healthy, + (95)", "healthy, − (855)"],
                 "1,000 people tested")


@register("fund.bayes-theorem", "sequential-update")
def sequential_update(p):
    """Posterior after k independent positive tests: odds multiply by LR = 0.99/0.05 = 19.8 each time."""
    prior, lr = 0.01, 0.99 / 0.05
    k = np.arange(4)
    odds = prior / (1 - prior) * lr ** k
    post = odds / (1 + odds)
    fig, ax = figure(2.35)
    bars = ax.bar(k, post, width=0.55, color=[p.muted, p.c(0), p.c(0), p.c(0)])
    for b, v in zip(bars, post):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=7.8, color=p.fg)
    for i in range(3):
        ax.annotate("", xy=(i + 0.72, 1.17), xytext=(i + 0.28, 1.17),
                    arrowprops=dict(arrowstyle="-|>", color=p.label, lw=0.9))
        ax.text(i + 0.5, 1.22, r"odds $\times19.8$", ha="center", fontsize=7, color=p.label)
    ax.set_xticks(k)
    ax.set_xticklabels(["prior", "1 test +", "2 tests +", "3 tests +"])
    ax.set_ylabel("P(disease | data)")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.5, 1])
    return fig


@register("fund.bayes-theorem", "explaining-away")
def explaining_away(p):
    """Burglary (B) and earthquake (E) independent; alarm A depends on both. P(B), P(B|A), P(B|A,E)."""
    pB, pE = 0.01, 0.02
    pA = {(0, 0): 0.001, (1, 0): 0.95, (0, 1): 0.30, (1, 1): 0.97}
    J = {(b, e): (pB if b else 1 - pB) * (pE if e else 1 - pE) * pA[(b, e)] for b in (0, 1) for e in (0, 1)}
    Z = sum(J.values())
    pBA = (J[(1, 0)] + J[(1, 1)]) / Z
    pBAE = J[(1, 1)] / (J[(0, 1)] + J[(1, 1)])
    fig, (ax0, ax1) = figure(2.3, ncols=2, gridspec_kw=dict(width_ratios=[1.1, 1]))
    blank(ax0, (0, 3.2), (0.1, 2.75))
    box(ax0, (0.0, 1.55), 1.4, 0.8, "Burglary\nB", p, fontsize=7.2)
    box(ax0, (1.8, 1.55), 1.4, 0.8, "Earth-\nquake E", p, fontsize=7.2)
    box(ax0, (0.88, 0.3), 1.45, 0.62, "Alarm A", p, color=p.accent, fontsize=7.4)
    arrow(ax0, (0.72, 1.53), (1.35, 0.95), p)
    arrow(ax0, (2.5, 1.53), (1.85, 0.95), p)
    ax0.text(1.6, 2.5, r"$B \perp E$ a priori", ha="center", fontsize=7.4, color=p.muted)
    vals = [pB, pBA, pBAE]
    labs = ["P(B)", "P(B|A)", "P(B|A,E)"]
    bars = ax1.bar(range(3), vals, width=0.6, color=[p.muted, p.c(0), p.c(1)])
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=7.8, color=p.fg)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labs, fontsize=7)
    ax1.set_ylim(0, 0.75)
    ax1.set_yticks([0, 0.25, 0.5])
    ax1.tick_params(axis="y", labelsize=7)
    return fig
