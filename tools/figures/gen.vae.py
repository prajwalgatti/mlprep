"""Figures for gen.vae. The architecture is a diagram; the other figures are computed with numpy."""
import numpy as np
from matplotlib.patches import Circle

from figures.style import arrow, blank, box, figure, register


@register("gen.vae", "architecture")
def architecture(p):
    """Encoder -> (mu, log sigma^2) -> z = mu + sigma*eps -> decoder; loss = recon + KL; gradients avoid eps."""
    fig, ax = figure(2.55)
    blank(ax, (0, 12.4), (0, 6.4))
    box(ax, (0.1, 2.3), 1.1, 1.3, "$x$", p, fontsize=9, color=p.fg)
    box(ax, (1.7, 2.1), 1.9, 1.7, "encoder\n$\\phi$", p, fontsize=7.6, color=p.c(0))
    box(ax, (4.1, 3.25), 1.5, 0.95, "$\\mu$", p, fontsize=8, color=p.c(0))
    box(ax, (4.1, 1.7), 1.5, 0.95, "$\\log\\sigma^2$", p, fontsize=7.4, color=p.c(0))
    ax.add_patch(Circle((6.85, 2.95), 0.55, facecolor=p.surface, edgecolor=p.accent, lw=1.4))
    ax.text(6.85, 2.95, "$z$", ha="center", va="center", fontsize=9, color=p.accent)
    ax.add_patch(Circle((6.85, 0.75), 0.5, facecolor=p.surface, edgecolor=p.muted, lw=1.0, ls="--"))
    ax.text(6.85, 0.75, "$\\epsilon$", ha="center", va="center", fontsize=8.5, color=p.muted)
    ax.text(7.5, 0.75, "$\\sim\\mathcal{N}(0,I)$", ha="left", va="center", fontsize=7.2, color=p.muted)
    box(ax, (8.1, 2.1), 1.9, 1.7, "decoder\n$\\theta$", p, fontsize=7.6, color=p.c(2))
    box(ax, (10.5, 2.3), 1.8, 1.3, "$\\mu_\\theta(z)$", p, fontsize=7.6, color=p.fg)
    arrow(ax, (1.2, 2.95), (1.7, 2.95), p, color=p.accent, lw=1.3)
    arrow(ax, (3.6, 3.2), (4.1, 3.7), p, color=p.accent, lw=1.3)
    arrow(ax, (3.6, 2.7), (4.1, 2.2), p, color=p.accent, lw=1.3)
    arrow(ax, (5.6, 3.7), (6.33, 3.1), p, color=p.accent, lw=1.3)
    arrow(ax, (5.6, 2.2), (6.33, 2.8), p, color=p.accent, lw=1.3)
    arrow(ax, (6.85, 1.25), (6.85, 2.4), p, color=p.muted, lw=1.0)
    arrow(ax, (7.4, 2.95), (8.1, 2.95), p, color=p.accent, lw=1.3)
    arrow(ax, (10.0, 2.95), (10.5, 2.95), p, color=p.accent, lw=1.3)
    ax.text(6.85, 4.0, "$z=\\mu+\\sigma\\odot\\epsilon$", ha="center", va="bottom", fontsize=7.4, color=p.accent)
    box(ax, (3.55, 5.0), 2.6, 1.05, "KL$(q_\\phi\\,\\|\\,\\mathcal{N}(0,I))$", p, fontsize=6.9, color=p.label)
    arrow(ax, (4.85, 4.2), (4.85, 5.0), p, color=p.label, lw=0.9)
    box(ax, (9.5, 5.0), 2.8, 1.05, "$-\\log p_\\theta(x\\mid z)$", p, fontsize=6.9, color=p.label)
    arrow(ax, (11.4, 3.6), (11.4, 5.0), p, color=p.label, lw=0.9)
    ax.text(3.4, 5.52, "loss =", ha="right", va="center", fontsize=7.4, color=p.label)
    ax.text(7.82, 5.52, "+", ha="center", va="center", fontsize=9, color=p.label)
    ax.text(0.15, 0.95, "accent = differentiable path\n(backprop reaches $\\phi$ via $\\mu,\\sigma$)",
            ha="left", va="center", fontsize=6.6, color=p.accent)
    return fig


@register("gen.vae", "estimator-variance")
def estimator_variance(p):
    """Per-sample estimates of d/dmu E[z^2] under q=N(1,1): reparameterization vs score function."""
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(200_000)
    z = 1.0 + eps
    rep = 2 * z
    sf = z ** 2 * (z - 1.0)
    fig, ax = figure(2.45)
    bins = np.linspace(-12, 16, 113)
    ax.hist(sf, bins=bins, density=True, color=p.c(1), alpha=0.55, lw=0, label="score function\n(var 30)")
    ax.hist(rep, bins=bins, density=True, histtype="step", color=p.accent, lw=1.6,
            label="reparam.\n(var 4)")
    ax.axvline(2, color=p.fg, lw=0.9, ls="--")
    ax.text(2.4, 0.37, "true gradient = 2", color=p.fg, fontsize=7.4, ha="left")
    ax.set_xlim(-12, 16)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([])
    ax.set_xlabel(r"one-sample estimate of $\partial_\mu\,\mathbb{E}_{q}[z^2]$")
    ax.set_ylabel("density")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.6), handlelength=1.3, fontsize=7, labelspacing=0.9)
    ax.set_xlim(-14, 16)
    return fig


def _agg_setup():
    rng = np.random.default_rng(3)
    centers = [(1.35 * np.cos(a), 1.35 * np.sin(a)) for a in np.deg2rad([90, 210, 330])]
    mus = []
    for c in centers:
        mus.append(np.array(c) + 0.28 * rng.standard_normal((60, 2)))
    return np.concatenate(mus), 0.17


@register("gen.vae", "aggregate-posterior-holes")
def aggregate_posterior_holes(p):
    """Illustrative 2-D latent: aggregate posterior (mixture of per-datapoint q's) vs the N(0, I) prior."""
    mus, s = _agg_setup()
    g = np.linspace(-3, 3, 241)
    X, Y = np.meshgrid(g, g)
    dens = np.zeros_like(X)
    for m in mus:
        dens += np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (2 * s * s))
    dens /= len(mus) * 2 * np.pi * s * s
    fig, ax = figure(2.85)
    lv = np.array([0.02, 0.08, 0.2, 0.4]) * dens.max()
    ax.contourf(X, Y, dens, levels=list(lv) + [dens.max() * 1.01], colors=[p.c(0)], alpha=0.0)
    for i, l in enumerate(lv):
        ax.contourf(X, Y, dens, levels=[l, dens.max() * 1.01], colors=[p.c(0)], alpha=0.18)
    for r, lab in [(1.0, "prior 1 sd"), (2.0, "2 sd")]:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(t), r * np.sin(t), color=p.muted, lw=0.9, ls="--")
    ax.text(-2.95, -2.5, "dashed: prior $\\mathcal{N}(0,I)$, 1 and 2 sd", color=p.muted, fontsize=7, ha="left", va="bottom")
    ax.text(-2.95, 2.75, "shaded: aggregate posterior $q_\\phi(z)$", color=p.c(0), fontsize=7.2, ha="left", va="top")
    for (x, y, lab) in [(0.0, 1.35, "A"), (0.0, -0.05, "B"), (2.45, 1.9, "C")]:
        ax.plot([x], [y], "o", color=p.label, ms=4.5, zorder=5)
        ax.text(x + 0.12, y + 0.1, lab, color=p.label, fontsize=8.5, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.6, 2.9)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    return fig


@register("gen.vae", "blur-averaging")
def blur_averaging(p):
    """Two inputs x=-1, x=+1 with posteriors N(+-m, s^2); the optimal Gaussian decoder outputs E[x|z]=tanh(mz/s^2)."""
    m = 1.0
    z = np.linspace(-3, 3, 400)
    fig, (a1, a2) = figure(3.1, nrows=2, sharex=True)
    for s, ls, lab in [(0.35, "-", "little overlap ($s=0.35$)"), (1.0, "--", "heavy overlap ($s=1$)")]:
        n = lambda mu: np.exp(-(z - mu) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
        a1.plot(z, n(-m), color=p.c(0), ls=ls, lw=1.4)
        a1.plot(z, n(m), color=p.c(1), ls=ls, lw=1.4)
        a2.plot(z, np.tanh(m * z / (s * s)), color=p.accent, ls=ls, lw=1.8, label=lab)
    a1.text(-1.0, 1.22, "$q(z\\mid x{=}-1)$", color=p.c(0), fontsize=7.4, ha="center")
    a1.text(1.0, 1.22, "$q(z\\mid x{=}+1)$", color=p.c(1), fontsize=7.4, ha="center")
    a1.set_ylim(0, 1.45)
    a1.set_yticks([])
    a1.set_ylabel("encoder", fontsize=7.8)
    a2.axhline(0, color=p.faint, lw=0.8)
    a2.axhspan(-0.35, 0.35, xmin=0.4, xmax=0.6, color=p.label, alpha=0.18, lw=0)
    a2.text(0.75, -0.2, "outputs a value\nno input had", color=p.label, fontsize=7, ha="left", va="top")
    a2.set_ylim(-1.25, 1.25)
    a2.set_yticks([-1, 0, 1])
    a2.set_ylabel(r"decoder $\mathbb{E}[x\mid z]$", fontsize=7.8)
    a2.set_xlabel("latent $z$")
    a2.legend(loc="upper left", fontsize=6.8, handlelength=1.8)
    return fig
