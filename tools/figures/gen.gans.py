"""Figures for gen.gans. All data are synthetic, computed from the formulas in the lesson."""
import numpy as np

from figures.style import figure, register


def _npdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


@register("gen.gans", "optimal-discriminator-1d")
def optimal_discriminator_1d(p):
    """Bimodal data vs a unimodal generator, with D* = p_data/(p_data+p_g) underneath."""
    x = np.linspace(-4.5, 4.5, 1200)
    pd = 0.5 * _npdf(x, -1.5, 0.5) + 0.5 * _npdf(x, 1.5, 0.5)
    pg = _npdf(x, 0.3, 1.0)
    d = pd / (pd + pg)
    fig, (a1, a2) = figure(3.1, nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.05]))
    a1.fill_between(x, pd, color=p.c(0), alpha=0.18, lw=0)
    a1.plot(x, pd, color=p.c(0))
    a1.plot(x, pg, color=p.c(1))
    a1.text(-1.5, 0.43, r"$p_{data}$", color=p.c(0), ha="center", fontsize=8.5)
    a1.text(0.3, 0.43, r"$p_g$", color=p.c(1), ha="center", fontsize=8.5)
    a1.set_ylim(0, 0.5)
    a1.set_yticks([])
    a1.set_ylabel("density")
    a2.axhline(0.5, color=p.muted, lw=0.8, ls="--")
    a2.plot(x, d, color=p.fg, lw=2)
    cross = np.where(np.diff(np.sign(pd - pg)) != 0)[0]
    for i in cross:
        if abs(x[i]) < 3.0:
            a1.axvline(x[i], color=p.faint, lw=0.8)
            a2.axvline(x[i], color=p.faint, lw=0.8)
            a2.plot([x[i]], [0.5], "o", color=p.label, ms=4, zorder=5)
    a2.text(0.1, 0.62, r"$D^*=\frac{1}{2}$" "\nat crossings", color=p.label, fontsize=7.5, ha="center")
    a2.text(4.45, 1.03, r"$D^*\to 0$ in both tails:" "\n" r"wider $p_g$ dominates", color=p.muted,
            fontsize=7.2, ha="right", va="top")
    a2.set_ylim(-0.03, 1.05)
    a2.set_yticks([0, 0.5, 1])
    a2.set_ylabel(r"$D^*(x)$")
    a2.set_xlabel("x")
    a2.set_xticks([])
    return fig


@register("gen.gans", "which-dstar-quiz")
def which_dstar_quiz(p):
    """Four candidate discriminator curves for one pair of densities. Only one is D*."""
    x = np.linspace(-4.5, 4.0, 900)
    pd = _npdf(x, -0.8, 0.6)
    pg = _npdf(x, 0.8, 1.0)
    true = pd / (pd + pg)
    cands = {
        "A": 1 - true,                              # p_g/(p_data+p_g)
        "B": true,                                  # correct
        "C": _sigmoid(-2.2 * x),                    # monotone logistic, wrong in the far-left tail
        "D": pd / pd.max(),                         # data density rescaled
    }
    fig, axes = figure(3.0, nrows=2, ncols=2, sharex=True, sharey=True)
    scale = 0.9 / max(pd.max(), pg.max())
    for ax, (k, c) in zip(axes.flat, cands.items()):
        ax.fill_between(x, pd * scale, color=p.c(0), alpha=0.16, lw=0)
        ax.plot(x, pd * scale, color=p.c(0), lw=1)
        ax.plot(x, pg * scale, color=p.c(1), lw=1)
        ax.axhline(0.5, color=p.muted, lw=0.6, ls="--")
        ax.plot(x, c, color=p.fg, lw=1.9)
        ax.set_title(k, loc="left", fontsize=9.5, fontweight="bold", color=p.label, pad=2)
        ax.set_ylim(-0.03, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([])
    axes[0, 0].text(-0.8, 0.93, r"$p_{data}$", color=p.c(0), fontsize=7.5, ha="center")
    axes[0, 0].text(2.5, 0.3, r"$p_g$", color=p.c(1), fontsize=7.5, ha="center")
    return fig


@register("gen.gans", "saturating-vs-nonsaturating")
def saturating_vs_nonsaturating(p):
    """Generator losses vs D(G(z)), and the gradient each sends back through the logit."""
    d = np.linspace(0.002, 0.998, 500)
    fig, (a1, a2) = figure(2.15, ncols=2)
    a1.plot(d, np.log(1 - d), color=p.c(0))
    a1.plot(d, -np.log(d), color=p.c(1))
    a1.text(0.06, -1.8, r"minimax" "\n" r"$\log(1-D)$", color=p.c(0), fontsize=7.5)
    a1.text(0.12, 3.6, r"non-sat." "\n" r"$-\log D$", color=p.c(1), fontsize=7.5)
    a1.axhline(0, color=p.faint, lw=0.6)
    a1.set_ylim(-4, 5.5)
    a1.set_xlabel(r"$D(G(z))$")
    a1.set_ylabel("generator loss")
    a1.set_xticks([0, 0.5, 1])
    a1.set_yticks([-4, 0, 4])
    a2.plot(d, d, color=p.c(0))
    a2.plot(d, 1 - d, color=p.c(1))
    a2.axvspan(0, 0.12, color=p.label, alpha=0.12, lw=0)
    a2.text(0.14, 0.9, "early training:\nD rejects fakes", color=p.label, fontsize=7)
    a2.text(0.66, 0.5, "minimax", color=p.c(0), fontsize=7.5, va="top")
    a2.text(0.03, 0.58, "non-sat.", color=p.c(1), fontsize=7.5, va="top")
    a2.set_xlabel(r"$D(G(z))$")
    a2.set_ylabel(r"$|\partial\,\mathrm{loss}/\partial a|$  (logit $a$)")
    a2.set_xticks([0, 0.5, 1])
    a2.set_yticks([0, 0.5, 1])
    a2.set_ylim(0, 1.08)
    return fig


def _dirac_run(theta, psi, h, steps, mode="sim", gamma=0.0):
    """Gradient play on the zero-sum Dirac-GAN, V = log sig(0) + log(1 - sig(psi*theta)), minus R1 for D."""
    tr = [(theta, psi)]
    for _ in range(steps):
        if mode == "sim":
            s = _sigmoid(psi * theta)
            theta, psi = theta + h * psi * s, psi - h * (theta * s + gamma * psi)
        else:  # alternating: discriminator first, then generator sees the new psi
            s = _sigmoid(psi * theta)
            psi = psi - h * (theta * s + gamma * psi)
            s = _sigmoid(psi * theta)
            theta = theta + h * psi * s
        tr.append((theta, psi))
    return np.array(tr)


@register("gen.gans", "dirac-gan-dynamics")
def dirac_gan_dynamics(p):
    """Left: unregularized simultaneous GD spirals out, alternating GD cycles. Right: R1 converges."""
    fig, (a1, a2) = figure(2.35, ncols=2, sharex=True, sharey=True)
    sim = _dirac_run(1.0, 1.0, 0.25, 200, "sim")
    alt = _dirac_run(1.0, 1.0, 0.25, 260, "alt")
    a1.plot(sim[:, 0], sim[:, 1], color=p.c(1), lw=1.1)
    a1.plot(alt[:, 0], alt[:, 1], color=p.c(0), lw=1.4)
    a1.text(-3.3, -3.3, "simultaneous:\nspirals out", color=p.c(1), fontsize=7, va="bottom")
    a1.text(-3.3, 3.3, "alternating:\ncycles", color=p.c(0), fontsize=7, va="top")
    a1.set_title("no penalty", fontsize=8.5)
    g03 = _dirac_run(1.0, 1.0, 0.1, 400, "sim", gamma=0.3)
    g1 = _dirac_run(1.0, 1.0, 0.1, 400, "sim", gamma=1.0)
    a2.plot(g03[:, 0], g03[:, 1], color=p.c(2), lw=1.3)
    a2.plot(g1[:, 0], g1[:, 1], color=p.c(3), lw=1.3)
    a2.text(-3.3, -3.3, r"$\gamma=0.3$: damped spiral", color=p.c(2), fontsize=7)
    a2.text(1.15, 1.45, r"$\gamma=1$:" "\nno rotation", color=p.c(3), fontsize=7)
    a2.set_title(r"R1 penalty on $D$", fontsize=8.5)
    for ax in (a1, a2):
        ax.plot([0], [0], "*", color=p.label, ms=8, zorder=5)
        ax.plot([1], [1], "o", color=p.fg, ms=3.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-3.4, 3.4)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r"$\theta$ (generator)")
    a1.set_ylabel(r"$\psi$ (disc. slope)")
    return fig


@register("gen.gans", "mode-collapse")
def mode_collapse(p):
    """Eight-Gaussian ring (data) and a collapsed generator that covers two of the modes."""
    rng = np.random.default_rng(3)
    ang = np.arange(8) * 2 * np.pi / 8
    centers = np.stack([2 * np.cos(ang), 2 * np.sin(ang)], 1)
    data = centers[rng.integers(0, 8, 800)] + 0.12 * rng.standard_normal((800, 2))
    gen = centers[rng.choice([1, 2], 300, p=[0.65, 0.35])] + 0.09 * rng.standard_normal((300, 2))
    fig, ax = figure(2.6)
    ax.scatter(data[:, 0], data[:, 1], s=3, color=p.c(0), alpha=0.45, lw=0)
    ax.scatter(gen[:, 0], gen[:, 1], s=4, color=p.c(1), alpha=0.9, lw=0)
    ax.text(-3.55, 2.55, "data: 8 modes", color=p.c(0), fontsize=7.5)
    ax.text(-3.55, 2.15, "generator: 2 modes", color=p.c(1), fontsize=7.5)
    ax.text(2.9, -2.35, "precision high\n(samples look real)\n\nrecall low\n(6 modes missing)",
            color=p.muted, fontsize=7.2, ha="left", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 5.6)
    ax.set_ylim(-2.9, 2.9)
    ax.axis("off")
    return fig
