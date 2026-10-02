"""Figures for fund.kl-estimation. Curves are the exact formulas from the lesson."""
import numpy as np

from figures.style import figure, register


def _kl_bern(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


@register("fund.kl-estimation", "fisher-approx")
def fisher_approx(p):
    """KL between Bernoulli(0.5) and Bernoulli(0.5+delta), both directions, vs the Fisher quadratic."""
    d = np.linspace(-0.45, 0.45, 401)
    fwd = _kl_bern(0.5, 0.5 + d)
    rev = _kl_bern(0.5 + d, 0.5)
    quad = 0.5 * 4 * d ** 2
    fig, ax = figure(2.4)
    ax.plot(d, fwd, color=p.c(0), lw=2)
    ax.plot(d, rev, color=p.c(1), lw=2)
    ax.plot(d, quad, color=p.fg, lw=1.3, ls="--")
    ax.text(-0.44, 0.6, r"KL$(p_{0.5}\|p_{0.5+\delta})$", color=p.c(0), fontsize=7.5)
    ax.text(0.10, 0.62, r"KL$(p_{0.5+\delta}\|p_{0.5})$", color=p.c(1), fontsize=7.5)
    ax.text(0, 0.33, r"dashed: $\frac{1}{2} F\delta^2$, $F=4$", color=p.fg, fontsize=7.5, ha="center")
    ax.set_xlabel(r"step $\delta$ in the Bernoulli parameter")
    ax.set_ylabel("nats")
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.46, 0.46)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    return fig


@register("fund.kl-estimation", "estimator-curves")
def estimator_curves(p):
    """Per-sample values of k1, k2, k3 as functions of the ratio r = p(x)/q(x)."""
    r = np.logspace(np.log10(0.1), np.log10(10), 400)
    k1 = -np.log(r)
    k2 = 0.5 * np.log(r) ** 2
    k3 = (r - 1) - np.log(r)
    fig, ax = figure(2.5)
    ax.axhline(0, color=p.muted, lw=0.7)
    ax.axvline(1, color=p.muted, lw=0.6, ls=":")
    ax.plot(r, k1, color=p.c(0), lw=1.9)
    ax.plot(r, k2, color=p.c(2), lw=1.9)
    ax.plot(r, k3, color=p.c(1), lw=1.9)
    ax.fill_between(r, k1, 0, where=k1 < 0, color=p.bad, alpha=0.15, lw=0)
    for y, c in [(-np.log(2), p.c(0)), (1 - np.log(2), p.c(1)), (0.5 * np.log(2) ** 2, p.c(2))]:
        ax.plot([2], [y], "o", color=c, ms=4, zorder=5)
    ax.text(0.13, -0.8, r"$k_1=-\log r$", color=p.c(0), fontsize=7.5)
    ax.text(3.2, 1.75, r"$k_3=(r-1)-\log r$", color=p.c(1), fontsize=7.5, ha="center")
    ax.text(0.42, 1.0, r"$k_2=\frac{1}{2}(\log r)^2$", color=p.c(2), fontsize=7.5)
    ax.text(2.2, -2.15, r"$k_1<0$ when $p>q$", color=p.bad, fontsize=7.5, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(-2.4, 2.6)
    ax.set_xticks([0.1, 0.5, 1, 2, 10])
    ax.set_xticklabels(["0.1", "0.5", "1", "2", "10"])
    ax.set_xlabel(r"ratio $r=p(x)/q(x)$ at a sample $x\sim q$")
    ax.set_ylabel("estimate from one sample")
    return fig
