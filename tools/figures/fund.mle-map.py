"""Figures for fund.mle-map. Beta densities, priors and change of variables computed exactly."""
import math

import numpy as np

from figures.style import figure, register


def _beta(t, a, b):
    return np.exp((a - 1) * np.log(t) + (b - 1) * np.log1p(-t)
                  + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


@register("fund.mle-map", "beta-posterior")
def beta_posterior(p):
    """3 heads in 3 flips, Beta(2,2) prior: posterior Beta(5,2); MLE 1, MAP 4/5, mean 5/7."""
    t = np.linspace(0.001, 0.999, 500)
    fig, ax = figure(2.45)
    ax.plot(t, _beta(t, 2, 2), color=p.muted, lw=1.4, ls="--", label="prior Beta(2,2)")
    ax.plot(t, _beta(t, 4, 1), color=p.c(2), lw=1.4, ls=":", label=r"likelihood $\theta^3$ (scaled)")
    ax.plot(t, _beta(t, 5, 2), color=p.accent, lw=2.2, label="posterior Beta(5,2)")
    for x, lab, col, dy in [(5 / 7, "mean 5/7", p.c(4), -0.95), (0.8, "MAP 4/5", p.label, 0.25),
                            (1.0, "MLE 1", p.bad, 0.25)]:
        y = _beta(np.array([min(x, 0.999)]), 5, 2)[0]
        ax.plot([x], [y], "o", color=col, ms=5, zorder=5)
        if dy < 0:
            ax.annotate(lab, (x, y), xytext=(0.5, 2.3), color=col, fontsize=7.5, ha="right",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
        else:
            ax.text(x - 0.02, y + dy, lab, color=col, fontsize=7.5, ha="right")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel(r"$\theta=P(\mathrm{heads})$")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", handlelength=1.6)
    return fig


@register("fund.mle-map", "beta-posterior-q")
def beta_posterior_q(p):
    """Posterior Beta(6,3) (4 heads in 5, Beta(2,2) prior) with four unlabeled candidate markers.

    A = 5/7 (mode), B = 1/2 (prior mean), C = 4/5 (MLE), D = 2/3 (posterior mean)."""
    t = np.linspace(0.001, 0.999, 500)
    fig, ax = figure(2.3)
    ax.plot(t, _beta(t, 6, 3), color=p.accent, lw=2.2)
    for x, lab in [(5 / 7, "A"), (0.5, "B"), (0.8, "C"), (2 / 3, "D")]:
        ax.axvline(x, color=p.muted, lw=0.8, ls=":")
        ax.text(x, 3.05, lab, ha="center", fontsize=8.5, color=p.label, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.3)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p(\theta\mid D)$")
    return fig


@register("fund.mle-map", "l1-l2-priors")
def l1_l2_priors(p):
    """Gaussian (tau=1) and Laplace (b=1/sqrt 2, same variance) densities and negative log densities."""
    w = np.linspace(-3, 3, 601)
    b = 1 / np.sqrt(2)
    gauss = np.exp(-w ** 2 / 2) / np.sqrt(2 * np.pi)
    lap = np.exp(-np.abs(w) / b) / (2 * b)
    fig, (a0, a1) = figure(2.25, ncols=2)
    a0.plot(w, gauss, color=p.c(0), lw=2, label="Gaussian")
    a0.plot(w, lap, color=p.c(1), lw=2, label="Laplace")
    a0.set_title("prior density", fontsize=8.5)
    a0.set_yticks([])
    a0.set_xlabel(r"$w$")
    a0.legend(loc="upper left", fontsize=7, handlelength=1.2, bbox_to_anchor=(-0.04, 1.02))
    a1.plot(w, w ** 2 / 2, color=p.c(0), lw=2)
    a1.plot(w, np.abs(w) / b, color=p.c(1), lw=2)
    a1.set_title("penalty $-\\log p(w)$ + const", fontsize=8.5)
    a1.text(1.25, 0.35, r"$\frac{w^2}{2\tau^2}$: L2", color=p.c(0), fontsize=7.8)
    a1.text(-1.95, 3.6, r"$\frac{|w|}{b}$: L1", color=p.c(1), fontsize=7.8)
    a1.annotate("kink at 0", (0, 0), xytext=(0.5, 2.6), fontsize=7.2, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    a1.set_ylim(0, 4.5)
    a1.set_yticks([])
    a1.set_xlabel(r"$w$")
    return fig


@register("fund.mle-map", "map-reparam")
def map_reparam(p):
    """Beta(5,2) over p and the same distribution over phi = logit(p) (Jacobian p(1-p))."""
    t = np.linspace(0.001, 0.999, 600)
    phi = np.linspace(-3, 5, 600)
    s = 1 / (1 + np.exp(-phi))
    dens_phi = _beta(s, 5, 2) * s * (1 - s)
    fig, (a0, a1) = figure(2.35, ncols=2)
    a0.plot(t, _beta(t, 5, 2), color=p.accent, lw=2)
    a1.plot(phi, dens_phi, color=p.accent, lw=2)
    pairs = [(0.8, p.label, "0.80"), (5 / 7, p.c(2), "0.71")]
    for x, col, lab in pairs:
        y0 = _beta(np.array([x]), 5, 2)[0]
        a0.plot([x, x], [0, y0], color=col, lw=1.2, ls="--")
        lx = math.log(x / (1 - x))
        y1 = _beta(np.array([x]), 5, 2)[0] * x * (1 - x)
        a1.plot([lx, lx], [0, y1], color=col, lw=1.2, ls="--")
    a0.text(0.81, 2.65, "mode", color=p.label, fontsize=7.4, ha="left")
    a0.text(0.03, 2.2, r"$p=5/7$ maps to" + "\n" + "the mode on\nthe right", color=p.c(2),
            fontsize=7.2, ha="left")
    a1.text(math.log(2.5) - 0.15, 0.55, "mode", color=p.c(2), fontsize=7.4, ha="right")
    a1.text(math.log(4) + 0.15, 0.5, r"logit$(0.8)$", color=p.label, fontsize=7.2, ha="left")
    a0.set_title(r"density over $p$", fontsize=8.5)
    a1.set_title(r"over $\phi=\mathrm{logit}\,p$", fontsize=8.5)
    a0.set_xlabel(r"$p$")
    a1.set_xlabel(r"$\phi$")
    a0.set_yticks([])
    a1.set_yticks([])
    a0.set_ylim(0, 3.0)
    a1.set_ylim(0, 0.62)
    return fig
