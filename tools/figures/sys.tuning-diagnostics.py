"""Figures for sys.tuning-diagnostics."""
import numpy as np

from figures.style import figure, register


def _curve_gallery(p, order="ABCD"):
    """Four training-curve patterns: A overfitting, B noisy late, C still improving, D saturated early."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 200)
    fig, axes = figure(2.7, nrows=2, ncols=2, sharex=True)
    train = {
        "A": 0.15 + 1.6 * np.exp(-6 * t),
        "B": 0.40 + 1.4 * np.exp(-8 * t),
        "C": 0.55 + 1.3 * np.exp(-2.2 * t),
        "D": 0.30 + 1.5 * np.exp(-25 * t),
    }
    val = {
        "A": 0.55 + 1.3 * np.exp(-7 * t) + 0.45 * np.clip(t - 0.35, 0, None) ** 1.3,
        "B": 0.62 + 1.2 * np.exp(-8 * t) + rng.normal(0, 0.05, t.size) * (t > 0.3),
        "C": 0.72 + 1.2 * np.exp(-2.2 * t),
        "D": 0.50 + 1.35 * np.exp(-25 * t),
    }
    for ax, k, lab in zip(axes.flat, order, "ABCD"):
        ax.plot(t, train[k], color=p.muted, lw=1.2, ls="--")
        ax.plot(t, val[k], color=p.accent, lw=1.5)
        ax.text(0.97, 0.93, lab, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=p.label,
                fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0, 2.0)
    axes[0, 0].text(0.05, 0.12, "val", color=p.accent, fontsize=7.5, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.05, 0.02, "train (dashed)", color=p.muted, fontsize=7, transform=axes[0, 0].transAxes)
    axes[1, 0].set_xlabel("step", fontsize=8)
    axes[1, 1].set_xlabel("step", fontsize=8)
    axes[0, 0].set_ylabel("loss", fontsize=8)
    axes[1, 0].set_ylabel("loss", fontsize=8)
    return fig


@register("sys.tuning-diagnostics", "curve-gallery")
def curve_gallery(p):
    return _curve_gallery(p)


@register("sys.tuning-diagnostics", "curve-gallery-quiz")
def curve_gallery_quiz(p):
    """Same four patterns, panels shuffled, for a question stem."""
    return _curve_gallery(p, order="CDAB")


@register("sys.tuning-diagnostics", "stability-threshold")
def stability_threshold(p):
    """Gradient descent on f = lambda x^2 / 2: x_t = (1 - eta*lambda)^t x_0."""
    t = np.arange(0, 21)
    fig, ax = figure(2.9)
    for r, col, lab in [(0.5, p.c(2), r"$\eta\lambda=0.5$: monotone"),
                        (1.5, p.c(0), r"$\eta\lambda=1.5$: oscillates, converges"),
                        (2.1, p.bad, r"$\eta\lambda=2.1$: oscillates, diverges")]:
        x = (1 - r) ** t
        ax.plot(t, x, "-o", ms=2.6, lw=1.2, color=col, label=lab)
    ax.axhline(0, color=p.faint, lw=0.8)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlim(0, 20)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$x_t / x_0$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=1.4, fontsize=7.2)
    return fig


@register("sys.tuning-diagnostics", "grad-norm-clipping")
def grad_norm_clipping(p):
    """Unclipped gradient norms over training, and the 90th-percentile clipping threshold."""
    rng = np.random.default_rng(4)
    n = 3000
    steps = np.arange(n)
    base = 1.0 + 1.5 * np.exp(-steps / 300)
    g = base * np.exp(rng.normal(0, 0.25, n))
    spikes = rng.choice(n, 25, replace=False)
    g[spikes] *= rng.uniform(3, 12, spikes.size)
    thr = np.percentile(g, 90)
    frac = (g > thr).mean()
    fig, (a1, a2) = figure(2.9, nrows=2, height_ratios=[1, 1.1])
    a1.semilogy(steps, g, color=p.accent, lw=0.5)
    a1.axhline(thr, color=p.label, lw=1.1, ls="--")
    a1.set_ylabel(r"$\|g\|$ (log)", fontsize=8)
    a1.set_xlabel("step", fontsize=8)
    bins = np.logspace(np.log10(g.min()), np.log10(g.max()), 50)
    a2.hist(g, bins=bins, color=p.surface, edgecolor=p.accent, lw=0.6)
    a2.hist(g[g > thr], bins=bins, color=p.label, alpha=0.6, lw=0)
    a2.axvline(thr, color=p.label, lw=1.1, ls="--")
    a2.set_xscale("log")
    a2.set_xlabel(r"unclipped gradient norm $\|g\|$", fontsize=8)
    a2.set_ylabel("count", fontsize=8)
    a2.text(thr * 1.15, a2.get_ylim()[1] * 0.75, f"threshold = 90th percentile\n{frac:.0%} of steps clipped", color=p.label, fontsize=7.2)
    return fig
