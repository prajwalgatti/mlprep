"""Figures for sys.tuning-pipeline."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


@register("sys.tuning-pipeline", "adam-epsilon")
def adam_epsilon(p):
    """Steady-state Adam step |m|/(sqrt(v)+eps) for a constant gradient of size r, against r."""
    r = np.logspace(-12, 0, 400)
    fig, ax = figure(2.5)
    spots = {-8: (1.3e-12, 3e-2), -4: (1.5e-9, 3e-3), -2: (2.5e-3, 2e-5)}
    for i, eps in enumerate([1e-8, 1e-4, 1e-2]):
        exp = int(round(np.log10(eps)))
        ax.loglog(r, r / (r + eps), color=p.c(i), lw=1.8)
        ax.text(*spots[exp], rf"$\epsilon=10^{{{exp}}}$", color=p.c(i), fontsize=7.5)
    ax.axhline(1, color=p.muted, lw=0.7, ls=":")
    ax.text(3e-12, 1.35, r"$|g|\gg\epsilon$: step $\approx\alpha$ (scale-free)", fontsize=7.2, color=p.fg)
    ax.text(3e-7, 4e-7, r"$|g|\ll\epsilon$: step $\approx(\alpha/\epsilon)\,|g|$" + "\n(momentum-SGD-like)",
            fontsize=7.2, color=p.fg)
    ax.set_xlabel(r"gradient RMS $|g|$")
    ax.set_ylabel(r"step size / $\alpha$")
    ax.set_ylim(1e-7, 4)
    ax.set_xlim(1e-12, 1)
    return fig


@register("sys.tuning-pipeline", "checkpoint-selection")
def checkpoint_selection(p):
    """Validation error evaluated every 1k steps: the best checkpoint is often not the last."""
    rng = np.random.default_rng(8)
    steps = np.arange(1, 51) * 1000
    t = steps / steps[-1]
    val = 0.24 + 0.25 * np.exp(-7 * t) + 0.012 * np.clip(t - 0.55, 0, None) + rng.normal(0, 0.0035, t.size)
    fig, ax = figure(2.3)
    ax.plot(steps, val * 100, "-o", ms=2.6, lw=1.1, color=p.accent)
    top = np.argsort(val)[:3]
    ax.plot(steps[top], val[top] * 100, "o", ms=6, mfc="none", mec=p.label, mew=1.2, label="3 best kept")
    b = np.argmin(val)
    ax.annotate("best", (steps[b], val[b] * 100), xytext=(steps[b], val[b] * 100 - 0.9), ha="center",
                fontsize=7.5, color=p.label, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.annotate("last", (steps[-1], val[-1] * 100), xytext=(steps[-1] - 3000, val[-1] * 100 + 1.1), ha="center",
                fontsize=7.5, color=p.muted, arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.8))
    ax.set_ylim(23.0, 28.5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation error (%)")
    ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])
    ax.legend(loc="upper right", handlelength=1.0)
    return fig


@register("sys.tuning-pipeline", "ghost-bn")
def ghost_bn(p):
    """Two devices, per-device batch 256 split into ghost batches of 64; EMA stats averaged at checkpoint."""
    fig, ax = figure(2.5)
    blank(ax, (0, 12), (0, 8.4))
    for d, y0 in enumerate([4.6, 0.6]):
        box(ax, (0.2, y0), 7.4, 3.0, "", p, fill=p.surface)
        ax.text(0.5, y0 + 2.45, f"device {d}: 256 examples", fontsize=7, color=p.muted, va="center")
        for k in range(4):
            box(ax, (0.5 + 1.75 * k, y0 + 0.35), 1.5, 1.5, "64\n" + r"$\mu,\sigma^2$", p, color=p.accent, fontsize=7, lw=0.9)
        box(ax, (8.6, y0 + 0.35), 3.1, 1.5, f"EMA {d}", p, fontsize=7.5)
        arrow(ax, (7.55, y0 + 1.1), (8.55, y0 + 1.1), p)
        ax.text(8.05, y0 + 1.35, "update", fontsize=6.5, color=p.muted, ha="center")
    box(ax, (7.9, 7.0), 3.9, 1.35, "at checkpoint:\nmean(EMA 0, EMA 1)", p, color=p.label, fontsize=7)
    arrow(ax, (10.15, 6.5), (10.15, 6.95), p, color=p.label)
    arrow(ax, (11.4, 2.5), (11.4, 6.95), p, color=p.label)
    return fig
