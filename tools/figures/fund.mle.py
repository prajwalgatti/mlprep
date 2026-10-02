"""Figures for fund.mle. Synthetic data; every curve is computed from the stated formula."""
import numpy as np

from figures.style import figure, register


def _npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)


@register("fund.mle", "likelihood-curve")
def likelihood_curve(p):
    """Bernoulli log-likelihood minus its maximum, for 3/10 and 30/100 heads."""
    t = np.linspace(0.005, 0.995, 600)
    fig, ax = figure(2.45)
    for (h, n), col, lab in [((3, 10), p.c(0), "3 heads / 10"), ((30, 100), p.c(1), "30 heads / 100")]:
        ll = h * np.log(t) + (n - h) * np.log(1 - t)
        llmax = h * np.log(h / n) + (n - h) * np.log(1 - h / n)
        ax.plot(t, ll - llmax, color=col, lw=2, label=lab)
    ax.axvline(0.3, color=p.muted, lw=0.8, ls=":")
    ax.axhline(-1.92, color=p.muted, lw=0.8, ls="--")
    ax.text(0.99, -1.75, r"$-1.92$", ha="right", fontsize=7.2, color=p.muted)
    ax.text(0.31, -7.6, r"$\hat p=0.3$", fontsize=7.5, color=p.muted)
    ax.set_xlim(0, 1)
    ax.set_ylim(-8, 0.6)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\ell(p)-\ell(\hat p)$")
    ax.legend(loc="upper right", handlelength=1.4)
    return fig


@register("fund.mle", "gmm-singularity")
def gmm_singularity(p):
    """Log-likelihood vs sigma of a component whose mean is pinned to data point x_1.

    Mixture: 0.5 N(mean, sd) + 0.5 N(x_1, sigma). Single Gaussian: N(x_1, sigma)."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20)
    lsig = np.linspace(-30, 1, 600)
    sig = 10.0 ** lsig
    m, s = x.mean(), x.std()

    def mixll(sg):
        a = np.log(0.5) + np.log(_npdf(x, m, s))
        z = -0.5 * ((x - x[0]) / sg) ** 2 - np.log(np.sqrt(2 * np.pi) * sg) + np.log(0.5)
        return np.sum(np.logaddexp(a, z))

    mix = np.array([mixll(sg) for sg in sig])
    single = np.array([np.sum(-0.5 * ((x - x[0]) / sg) ** 2 - np.log(np.sqrt(2 * np.pi) * sg)) for sg in sig])
    fig, ax = figure(2.45)
    ax.plot(lsig, mix, color=p.bad, lw=2, label="2-component mixture")
    ax.plot(lsig, single, color=p.c(0), lw=2, label="single Gaussian")
    ax.set_ylim(-60, 50)
    ax.set_xlim(-30, 1)
    ax.set_xticks([-30, -20, -10, 0])
    ax.set_xticklabels([r"$10^{-30}$", r"$10^{-20}$", r"$10^{-10}$", r"$1$"])
    ax.set_xlabel(r"$\sigma$ of the component centred on $x_1$")
    ax.set_ylabel("log-likelihood, 20 points")
    ax.text(-29, 38, r"grows like $-\log\sigma$: unbounded", fontsize=7.5, color=p.bad)
    ax.text(-3.2, -55, r"$\to-\infty$", fontsize=7.5, color=p.c(0), ha="right")
    ax.axhline(0, color=p.faint, lw=0.6)
    ax.legend(loc="lower left", handlelength=1.4, bbox_to_anchor=(0.0, 0.08))
    return fig


@register("fund.mle", "mle-kl")
def mle_kl(p):
    """Bimodal synthetic data; MLE Gaussian (moment matching = forward KL) vs the reverse-KL Gaussian."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2.5, 0.6, 300), rng.normal(2.5, 0.6, 300)])
    g = np.linspace(-6, 6, 500)
    fig, ax = figure(2.45)
    ax.hist(x, bins=40, density=True, color=p.faint, edgecolor=p.muted, lw=0.4)
    ax.plot(g, _npdf(g, x.mean(), x.std()), color=p.accent, lw=2.2)
    ax.plot(g, _npdf(g, 2.5, 0.6), color=p.c(1), lw=1.4, ls="--")
    ax.text(0, 0.19, "MLE fit\n(forward KL)", color=p.accent, fontsize=7.5, ha="center")
    ax.text(4.1, 0.55, "reverse-KL fit", color=p.c(1), fontsize=7.5, ha="left")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.72)
    ax.set_yticks([])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    return fig
