"""Figures for sys.tuning-batch-size, drawn from McCandlish et al. (2018) eqs. 2.6-2.12."""
import numpy as np

from figures.style import figure, register


@register("sys.tuning-batch-size", "steps-vs-batch")
def steps_vs_batch(p):
    """S/S_min = 1 + B_noise/B and E/E_min = 1 + B/B_noise against B/B_noise (log-log)."""
    b = np.logspace(-2, 2, 400)
    S = 1 + 1 / b
    E = 1 + b
    fig, ax = figure(2.5)
    ax.loglog(b, S, color=p.c(0), lw=2, label=r"steps $S/S_{\min}$")
    ax.loglog(b, E, color=p.c(1), lw=2, label=r"examples $E/E_{\min}$")
    ax.loglog(b[b < 1.5], 1 / b[b < 1.5], color=p.c(0), lw=1, ls=":")
    ax.plot([1], [2], "o", color=p.label, ms=5, zorder=5)
    ax.annotate(r"$B=B_{\rm crit}$: both $\times 2$", (1, 2), xytext=(1.6, 7), fontsize=7.5, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.axvspan(0.01, 0.15, color=p.faint, alpha=0.6, lw=0)
    ax.axvspan(5, 100, color=p.faint, alpha=0.6, lw=0)
    ax.text(0.045, 1.15, "perfect\nscaling", ha="center", fontsize=7.2, color=p.muted)
    ax.text(1.0, 30, "diminishing\nreturns", ha="center", fontsize=7.2, color=p.muted)
    ax.text(22, 1.15, "maximal data\nparallelism", ha="center", fontsize=7.2, color=p.muted)
    ax.text(0.11, 3.2, r"$\propto 1/B$", color=p.c(0), fontsize=7.5)
    ax.set_xlabel(r"batch size $B/B_{\rm noise}$")
    ax.set_ylabel("relative cost")
    ax.set_ylim(0.9, 150)
    ax.set_xlim(0.01, 100)
    ax.legend(loc="upper center", ncol=2, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, 1.03))
    return fig


@register("sys.tuning-batch-size", "lr-vs-batch")
def lr_vs_batch(p):
    """Optimal step eps_opt(B) = eps_max/(1+B_noise/B) vs the linear scaling rule."""
    b = np.logspace(-2, 2, 400)
    opt = 1 / (1 + 1 / b)
    fig, ax = figure(2.45)
    ax.fill_between(b, 2 * opt, 400, color=p.bad, alpha=0.13, lw=0)
    ax.loglog(b, 2 * opt, color=p.bad, lw=1, ls="--")
    ax.loglog(b, opt, color=p.accent, lw=2.2, label=r"optimal $\epsilon_{\rm opt}(B)$")
    ax.loglog(b, b, color=p.c(1), lw=1.4, ls="-.", label=r"linear rule (anchored at small $B$)")
    ax.axhline(1, color=p.muted, lw=0.8, ls=":")
    ax.text(0.012, 1.12, r"$\epsilon_{\max}$", color=p.muted, fontsize=8)
    ax.text(28, 3.4, "loss rises\n" + r"(step $> 2\epsilon_{\rm opt}$)", color=p.bad, fontsize=7.2, ha="center")
    ax.plot([1], [1], "o", color=p.label, ms=4.5, zorder=5)
    ax.annotate(r"crosses $2\epsilon_{\rm opt}$ at $B=2B_0+B_{\rm noise}$", (1, 1),
                xytext=(0.0115, 11), fontsize=7.2, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.008, 30)
    ax.set_xlabel(r"batch size $B/B_{\rm noise}$")
    ax.set_ylabel(r"step size / $\epsilon_{\max}$")
    ax.legend(loc="lower right", handlelength=1.6)
    return fig


@register("sys.tuning-batch-size", "time-compute-tradeoff")
def time_compute_tradeoff(p):
    """The hyperbola (S/S_min - 1)(E/E_min - 1) = 1, traced by the batch size."""
    b = np.logspace(-1.6, 1.6, 300)
    S, E = 1 + 1 / b, 1 + b
    fig, ax = figure(2.45)
    ax.plot(E, S, color=p.accent, lw=2)
    for bb, lab, off in [(1 / 8, r"$B_{\rm noise}/8$", (6, 0)), (1 / 2, r"$B_{\rm noise}/2$", (6, 2)),
                         (1, r"$B_{\rm noise}$", (6, 4)), (2, r"$2B_{\rm noise}$", (4, 6)),
                         (8, r"$8B_{\rm noise}$", (-6, 6))]:
        ax.plot([1 + bb], [1 + 1 / bb], "o", color=p.label, ms=4.5, zorder=5)
        ax.annotate(lab, (1 + bb, 1 + 1 / bb), xytext=off, textcoords="offset points", fontsize=7.5,
                    color=p.fg, ha="left" if off[0] > 0 else "right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.95, 50)
    ax.set_ylim(0.95, 50)
    ax.axhline(1, color=p.muted, lw=0.7, ls=":")
    ax.axvline(1, color=p.muted, lw=0.7, ls=":")
    ax.set_xlabel(r"compute: examples $E/E_{\min}$")
    ax.set_ylabel(r"time: steps $S/S_{\min}$")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.text(30, 1.25, "fast but\nwasteful", fontsize=7.2, color=p.muted, ha="center")
    ax.text(1.25, 30, "frugal\nbut slow", fontsize=7.2, color=p.muted, ha="left")
    return fig
