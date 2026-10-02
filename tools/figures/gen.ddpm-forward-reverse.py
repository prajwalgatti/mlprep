"""Figures for gen.ddpm-forward-reverse (Ho et al. 2020 notation: beta_t, alpha_t = 1 - beta_t, abar_t)."""
import numpy as np

from figures.style import arrow, blank, box, figure, register

T = 1000
BETA = np.linspace(1e-4, 0.02, T)          # BETA[t-1] = beta_t (Ho's linear schedule)
ALPHA = 1 - BETA
ABAR = np.cumprod(ALPHA)                   # ABAR[t-1] = abar_t


def abar(t):
    return 1.0 if t == 0 else ABAR[t - 1]


def swiss_roll(n, rng):
    """2-D Swiss roll, standardised to zero mean and unit variance per coordinate."""
    u = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = np.stack([u * np.cos(u), u * np.sin(u)], 1) + 0.35 * rng.standard_normal((n, 2))
    x = (x - x.mean(0)) / x.std(0)
    return x


@register("gen.ddpm-forward-reverse", "markov-chain")
def markov_chain(p):
    """Graphical model: fixed forward q (top arrows), learned reverse p_theta (bottom), one-shot q(x_t|x_0) arc."""
    from matplotlib.patches import FancyArrowPatch
    fig, ax = figure(1.85)
    blank(ax, (0, 10.4), (-0.75, 3.5))
    xs = [0.2, 2.2, 4.4, 6.6, 8.8]
    labels = [r"$x_0$", r"$x_1$", r"$x_{t-1}$", r"$x_t$", r"$x_T$"]
    w, h, y = 1.35, 0.8, 1.3
    for x, lab in zip(xs, labels):
        box(ax, (x, y), w, h, lab, p, fontsize=9)
    ax.text(xs[0] + w / 2, y - 0.15, "data", ha="center", va="top", fontsize=7.5, color=p.muted)
    ax.text(xs[-1] + w / 2, y - 0.15, r"$\approx N(0,I)$", ha="center", va="top", fontsize=7.5, color=p.muted)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        ls = ":" if (i, j) in [(1, 2), (3, 4)] else "-"
        a, b = xs[i] + w, xs[j]
        arrow(ax, (a, y + 0.6), (b, y + 0.6), p, color=p.c(1), lw=1.1, linestyle=ls)
        arrow(ax, (b, y + 0.2), (a, y + 0.2), p, color=p.c(0), lw=1.1, linestyle=ls)
    ax.add_patch(FancyArrowPatch((xs[0] + w / 2, y + h + 0.05), (xs[3] + w / 2, y + h + 0.05),
                                 connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=9,
                                 color=p.label, lw=1.0, linestyle="--"))
    ax.text((xs[0] + xs[3] + w) / 2, 3.3, r"one shot: $q(x_t|x_0)$", ha="center", va="center",
            fontsize=7.5, color=p.label)
    ax.text(5.2, -0.05, r"$\rightarrow$ $q(x_t|x_{t-1})$: fixed noising step", ha="center", fontsize=7.5, color=p.c(1))
    ax.text(5.2, -0.6, r"$\leftarrow$ $p_\theta(x_{t-1}|x_t)$: learned denoising step", ha="center", fontsize=7.5,
            color=p.c(0))
    return fig


@register("gen.ddpm-forward-reverse", "forward-2d")
def forward_2d(p):
    """A 2-D Swiss roll pushed through the closed-form forward marginal at several t (linear schedule)."""
    rng = np.random.default_rng(0)
    x0 = swiss_roll(1500, rng)
    eps = rng.standard_normal(x0.shape)
    ts = [0, 25, 50, 100, 250, 1000]
    fig, axes = figure(2.55, nrows=2, ncols=3)
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, t in zip(axes.flat, ts):
        a = abar(t)
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        for r in (1, 2):
            ax.plot(r * np.cos(th), r * np.sin(th), color=p.faint, lw=0.8, zorder=0)
        ax.scatter(xt[:, 0], xt[:, 1], s=0.6, color=p.c(0), alpha=0.6, lw=0, rasterized=True)
        snr = np.inf if t == 0 else a / (1 - a)
        sub = r"SNR $\infty$" if t == 0 else (f"SNR {snr:.0f}" if snr >= 10 else
                                                (f"SNR {snr:.2g}" if snr > 1e-3 else r"SNR $4\times10^{-5}$"))
        ax.set_title(f"t = {t}", fontsize=8.5, pad=8)
        ax.text(0.5, 1.0, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=p.muted)
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-3.3, 3.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    return fig


@register("gen.ddpm-forward-reverse", "schedule-curves")
def schedule_curves(p):
    """Signal and noise coefficients of q(x_t|x_0) for the linear schedule; they cross where SNR = 1."""
    t = np.arange(0, T + 1)
    a = np.concatenate([[1.0], ABAR])
    fig, ax = figure(2.2)
    ax.plot(t, np.sqrt(a), color=p.c(0))
    ax.plot(t, np.sqrt(1 - a), color=p.c(1))
    tc = int(np.argmax(a / (1 - a) < 1))     # first t with SNR < 1
    ax.axvline(tc, color=p.muted, lw=0.8, ls="--")
    ax.plot([tc], [np.sqrt(0.5)], "o", color=p.label, ms=4, zorder=5)
    ax.annotate(f"SNR = 1\nt ≈ {tc - 1}", (tc, np.sqrt(0.5)), xytext=(tc + 90, 0.42), fontsize=7.5,
                color=p.label, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.text(10, 0.665, r"signal $\sqrt{\bar\alpha_t}$", color=p.c(0), fontsize=8)
    ax.text(560, 0.88, r"noise $\sqrt{1-\bar\alpha_t}$", color=p.c(1), fontsize=8)
    ax.text(985, 0.08, r"$\bar\alpha_{1000}\approx4\times10^{-5}$", color=p.muted, fontsize=7, ha="right")
    ax.set_xlabel("diffusion step t")
    ax.set_ylabel("coefficient")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 250, 500, 750, 1000])
    return fig


@register("gen.ddpm-forward-reverse", "posterior-blend")
def posterior_blend(p):
    """Coefficients of x_0 and x_t in the posterior mean mu_tilde_t, linear schedule."""
    t = np.arange(2, T + 1)
    ab_t = ABAR[t - 1]
    ab_tm1 = ABAR[t - 2]
    c0 = np.sqrt(ab_tm1) * BETA[t - 1] / (1 - ab_t)
    ct = np.sqrt(ALPHA[t - 1]) * (1 - ab_tm1) / (1 - ab_t)
    fig, ax = figure(2.2)
    ax.plot(t, ct, color=p.c(0))
    ax.plot(t, c0, color=p.c(1))
    ax.set_xscale("log")
    ax.set_xlim(2, T)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([2, 10, 100, 1000])
    ax.set_xticklabels(["2", "10", "100", "1000"])
    ax.set_xlabel("diffusion step t (log scale)")
    ax.set_ylabel(r"weight in $\tilde\mu_t$")
    ax.text(25, 0.86, r"coefficient on $x_t$", color=p.c(0), fontsize=8)
    ax.text(25, 0.2, r"coefficient on $x_0$", color=p.c(1), fontsize=8)
    ax.plot([2], [c0[0]], "o", color=p.label, ms=3.5)
    ax.annotate(f"t = 2: {c0[0]:.2f}", (2, c0[0]), xytext=(3.4, 0.47), fontsize=7, color=p.label)
    return fig
