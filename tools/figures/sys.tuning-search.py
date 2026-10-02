"""Figures for sys.tuning-search (Tuning Playbook: search and reading studies)."""
import numpy as np

from figures.style import figure, register


def _radical_inverse(i, base):
    x, f = 0.0, 1.0 / base
    while i > 0:
        i, r = divmod(i, base)
        x += r * f
        f /= base
    return x


def halton(n, bases=(2, 3), shift=None):
    """Randomly shifted Halton points in [0,1)^d (Cranley-Patterson rotation)."""
    pts = np.array([[_radical_inverse(i, b) for b in bases] for i in range(1, n + 1)])
    if shift is not None:
        pts = (pts + shift) % 1.0
    return pts


@register("sys.tuning-search", "grid-vs-quasirandom")
def grid_vs_quasirandom(p):
    """Nine trials: a 3x3 grid tests 3 values of the important axis; quasi-random tests 9."""
    fig, axes = figure(2.15, ncols=2)
    g = lambda x: 0.55 + 0.35 * np.exp(-((x - 0.62) ** 2) / 0.012)  # important-axis response
    xs = np.linspace(0, 1, 300)
    grid = np.array([[a, b] for a in (1 / 6, 1 / 2, 5 / 6) for b in (1 / 6, 1 / 2, 5 / 6)])
    qr = halton(9, shift=np.array([0.11, 0.37]))
    for ax, pts, title in [(axes[0], grid, "grid (3 × 3)"), (axes[1], qr, "quasi-random (Halton)")]:
        ax.plot(xs, 1.02 + 0.28 * (g(xs) - 0.55) / 0.35, color=p.c(1), lw=1.3, clip_on=False)
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=4.5, color=p.accent, zorder=4)
        for x in pts[:, 0]:
            ax.plot([x, x], [-0.09, -0.02], color=p.label, lw=1.4, clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        ax.set_title(title, fontsize=8.5, pad=24)
        n = len(np.unique(np.round(pts[:, 0], 6)))
        ax.text(0.5, -0.2, f"{n} distinct values", ha="center", va="top", fontsize=7.5,
                color=p.label, transform=ax.transData)
    axes[0].set_ylabel("unimportant", fontsize=7.5)
    axes[0].text(0.5, -0.36, "important hyperparameter", ha="center", va="top", fontsize=7.5,
                 color=p.muted)
    axes[1].text(0.5, -0.36, "important hyperparameter", ha="center", va="top", fontsize=7.5,
                 color=p.muted)
    return fig


def _study(rng, lo, hi, n, x_star=-1.7, cliff=None):
    x = rng.uniform(lo, hi, n)
    other = rng.uniform(-1, 1, n)
    err = 0.232 + 0.03 * (x - x_star) ** 2 + 0.012 * other ** 2 + rng.normal(0, 0.002, n)
    bad = np.zeros(n, bool) if cliff is None else x > cliff
    return x, err, bad


def _axis_plots(p, quiz=False):
    rng = np.random.default_rng(7)
    fig, axes = figure(2.2, ncols=2, sharey=True)
    titles = ["Study A", "Study B"] if quiz else ["bad: best at the edge", "good: best inside"]
    for ax, (lo, hi, cliff), title in zip(axes, [(-4, -2, None), (-4, 0, -0.9)], titles):
        x, err, bad = _study(rng, lo, hi, 40, cliff=cliff)
        ok = ~bad
        ax.plot(x[ok], err[ok], "o", ms=3, color=p.accent, alpha=0.85)
        if bad.any():
            ax.plot(x[bad], np.full(bad.sum(), 0.362), "x", ms=4.5, color=p.bad, mew=1.2)
            ax.text(-0.45, 0.352, "diverged", color=p.bad, fontsize=7, ha="center", va="top")
        if not quiz:
            i = np.flatnonzero(ok)[err[ok].argmin()]
            ax.plot([x[i]], [err[i]], "*", ms=10, color=p.label, zorder=5)
        ax.set_xlim(lo - 0.15, hi + 0.15)
        ax.set_title(title, fontsize=8.2)
        ticks = list(range(lo, hi + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks], fontsize=7)
    if not quiz:
        axes[0].axvspan(-2.25, -2.0, color=p.label, alpha=0.18, lw=0)
    axes[0].set_ylabel("best validation error")
    axes[0].set_ylim(0.22, 0.372)
    fig.supxlabel("learning rate (log scale)", fontsize=8.5, color=p.muted)
    return fig


@register("sys.tuning-search", "axis-plot-boundaries")
def axis_plot_boundaries(p):
    """Basic hyperparameter axis plots (illustrative): best point at the edge (bad) vs inside (good)."""
    return _axis_plots(p)


@register("sys.tuning-search", "axis-plot-quiz")
def axis_plot_quiz(p):
    """Same studies with neutral titles and no highlighting, for use in a question stem."""
    return _axis_plots(p, quiz=True)


@register("sys.tuning-search", "trial-budget-bootstrap")
def trial_budget_bootstrap(p):
    """Bootstrap best-of-k from a 100-trial study: spread shrinks slowly with k."""
    rng = np.random.default_rng(11)
    x = rng.uniform(-4, 0, 100)
    y = rng.uniform(-1, 1, 100)
    err = 0.300 + 0.012 * (x + 1.7) ** 2 + 0.01 * y ** 2 + rng.normal(0, 0.001, 100)
    err = np.where(x > -0.6, np.nan, err)          # divergent trials
    ok = err[~np.isnan(err)]
    ks = [2, 4, 6, 10, 20, 30, 50]
    data = [np.min(rng.choice(ok, (4000, k), replace=True), axis=1) * 100 for k in ks]
    fig, ax = figure(2.3)
    bp = ax.boxplot(data, positions=range(len(ks)), widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=p.label, lw=1.4),
                    boxprops=dict(facecolor=p.surface, edgecolor=p.accent, lw=1),
                    whiskerprops=dict(color=p.muted, lw=0.9), capprops=dict(color=p.muted, lw=0.9))
    best = ok.min() * 100
    ax.axhspan(best - 0.1, best + 0.1, color=p.good, alpha=0.18, lw=0)
    ax.text(-0.3, best - 0.14, "assumed seed noise (±0.1)", color=p.good, fontsize=7.2, ha="left",
            va="top")
    ax.set_ylim(best - 0.45, None)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("trials in the study (bootstrapped)")
    ax.set_ylabel("best validation error (%)")
    return fig


@register("sys.tuning-search", "isolation-plot")
def isolation_plot(p):
    """Isolation plot: best trial per weight-decay bucket, after optimising away the LR."""
    rng = np.random.default_rng(5)
    n = 160
    wd = rng.uniform(-6, -2, n)        # log10 weight decay (scientific)
    lr = rng.uniform(-3, 0, n)         # log10 LR (nuisance)
    lr_star = -1.2 - 0.25 * (wd + 4)   # best LR shifts with weight decay
    err = 0.316 + 0.006 * (wd + 4.2) ** 2 + 0.03 * (lr - lr_star) ** 2 + rng.normal(0, 0.0012, n)
    fig, ax = figure(2.3)
    ax.plot(wd, err * 100, "o", ms=2.4, color=p.muted, alpha=0.55, label="all trials")
    edges = np.linspace(-6, -2, 9)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (wd >= a) & (wd < b)
        j = np.flatnonzero(m)[err[m].argmin()]
        cx.append(wd[j])
        cy.append(err[j] * 100)
    ax.plot(cx, cy, "-o", ms=4, color=p.accent, lw=1.6, label="best per bucket")
    ax.axhline(32.4, color=p.c(1), lw=1, ls="--")
    ax.text(-2.0, 32.0, "no weight decay\n(LR tuned)", color=p.c(1), fontsize=7.2, ha="right", va="top")
    ax.set_ylim(31.2, 35.5)
    ax.set_xticks([-6, -5, -4, -3, -2])
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in [-6, -5, -4, -3, -2]])
    ax.set_xlabel("weight decay (log scale)")
    ax.set_ylabel("validation error (%)")
    ax.legend(loc="lower center", ncol=2, handlelength=1.2, bbox_to_anchor=(0.5, 0.98))
    return fig
