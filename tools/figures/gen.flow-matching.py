"""Figures for gen.flow-matching. Lipman's convention: t = 0 noise (x0 ~ N(0, I)), t = 1 data.

Linear (sigma_min -> 0) path x_t = t x1 + (1 - t) x0 with data a Gaussian mixture. The marginal field
u_t(x) = E[x1 - x0 | x_t = x] is computed exactly: within a component, (x1, x0, x_t) are jointly Gaussian,
and the component posteriors weight the per-component conditional means.
"""
from functools import lru_cache

import numpy as np
from matplotlib.patches import Circle
from scipy.special import logsumexp

from figures.style import figure, register


def _field(mu, tau, pi):
    mu = np.asarray(mu, float)
    pi = np.asarray(pi, float)
    dim = mu.shape[1]

    def u(x, t):
        v = t ** 2 * tau ** 2 + (1 - t) ** 2          # Var(x_t | component), per coordinate
        d = x[:, None, :] - t * mu[None]
        lw = -0.5 * (d ** 2).sum(-1) / v - dim / 2 * np.log(v) + np.log(pi)
        r = np.exp(lw - logsumexp(lw, 1, keepdims=True))
        ex1 = mu[None] + (t * tau ** 2 / v) * d
        ex0 = ((1 - t) / v) * d
        return (r[:, :, None] * (ex1 - ex0)).sum(1)

    return u


def _integrate(u, x0, steps=200):
    ts = np.linspace(0, 1, steps + 1)
    xs = [x0]
    x = x0
    for a, b in zip(ts[:-1], ts[1:]):
        h = b - a
        k1 = u(x, a)
        k2 = u(x + h * k1, b)
        x = x + h * (k1 + k2) / 2
        xs.append(x)
    return ts, np.array(xs)


MU2 = np.array([[3.0, 1.8], [3.0, -1.8], [3.6, 0.0]])
TAU2 = 0.15


@lru_cache(maxsize=None)
def _paths2d():
    rng = np.random.default_rng(3)
    n = 22
    x0 = rng.standard_normal((4 * n, 2))
    x0 = x0[np.linalg.norm(x0, axis=1) < 2.0][:n]          # keep the picture inside the frame
    comp = rng.choice(3, size=n)
    x1 = MU2[comp] + TAU2 * rng.standard_normal((n, 2))
    _, xs = _integrate(_field(MU2, TAU2, [1 / 3] * 3), x0)
    return x0, x1, xs


@register("gen.flow-matching", "conditional-vs-marginal")
def conditional_vs_marginal(p):
    x0, x1, xs = _paths2d()
    fig, axes = figure(1.95, ncols=2, sharey=True)
    titles = ["conditional paths (training)", "marginal flow (sampling)"]
    for ax, title in zip(axes, titles):
        for m in MU2:
            ax.add_patch(Circle(m, 2 * TAU2, color=p.c(1), alpha=0.25, lw=0))
        ax.add_patch(Circle((0, 0), 1.0, fill=False, color=p.muted, lw=0.8, ls="--"))
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.set_xlim(-2.3, 4.1)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    a, b = axes
    for i in range(len(x0)):
        a.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], color=p.c(0), lw=0.8, alpha=0.85)
        b.plot(xs[:, i, 0], xs[:, i, 1], color=p.c(2), lw=0.9, alpha=0.9)
    for ax in axes:
        ax.scatter(x0[:, 0], x0[:, 1], s=5, color=p.fg, zorder=3, lw=0)
    a.scatter(x1[:, 0], x1[:, 1], s=5, color=p.c(1), zorder=3, lw=0)
    b.scatter(xs[-1, :, 0], xs[-1, :, 1], s=5, color=p.c(1), zorder=3, lw=0)
    a.text(0, -1.25, r"noise $x_0$", color=p.muted, fontsize=7, ha="center", va="top")
    a.text(3.3, -2.35, r"data $x_1$", color=p.muted, fontsize=7, ha="center", va="top")
    return fig


@lru_cache(maxsize=None)
def _path1d():
    mu = np.array([[-1.6], [1.4]])
    pi = np.array([0.4, 0.6])
    tau = 0.25
    ts = np.linspace(0, 1, 241)
    xg = np.linspace(-3.2, 3.2, 321)
    dens = np.zeros((len(xg), len(ts)))
    for j, t in enumerate(ts):
        v = t ** 2 * tau ** 2 + (1 - t) ** 2
        for m, w in zip(mu[:, 0], pi):
            dens[:, j] += w * np.exp(-0.5 * (xg - t * m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    starts = np.linspace(-2.2, 2.2, 13)[:, None]
    _, xs = _integrate(_field(mu, tau, pi), starts, steps=240)
    return ts, xg, dens, xs


@register("gen.flow-matching", "probability-path")
def probability_path(p):
    from matplotlib.colors import LinearSegmentedColormap
    ts, xg, dens, xs = _path1d()
    fig, ax = figure(2.2)
    cmap = LinearSegmentedColormap.from_list("pp", [p.surface, p.accent])
    ax.pcolormesh(ts, xg, dens ** 0.7, cmap=cmap, shading="auto", rasterized=True)
    tt = np.linspace(0, 1, xs.shape[0])
    for i in range(xs.shape[1]):
        ax.plot(tt, xs[:, i, 0], color=p.label, lw=0.9)
    ax.set_xlabel(r"$t$   (0 = noise $\mathcal{N}(0,1)$,  1 = data)")
    ax.set_ylabel("$x$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(-3.2, 3.2)
    return fig


@register("gen.flow-matching", "posterior-average")
def posterior_average(p):
    """Marginal velocity at t = 0.5 for two data points ±2 vs the two conditional fields."""
    t = 0.5
    x = np.linspace(-2.5, 2.5, 500)
    u = _field([[2.0], [-2.0]], 1e-6, [0.5, 0.5])(x[:, None], t)[:, 0]
    fig, ax = figure(2.2)
    for x1, lab in ((2.0, r"$u_t(x\,|\,x_1{=}{+}2)$"), (-2.0, r"$u_t(x\,|\,x_1{=}{-}2)$")):
        ax.plot(x, (x1 - x) / (1 - t), color=p.c(1) if x1 > 0 else p.c(2), lw=1.1, ls="--", label=lab)
    ax.plot(x, u, color=p.c(0), lw=2.0, label=r"marginal $u_t(x)$")
    ax.axhline(0, color=p.faint, lw=0.8, zorder=0)
    ax.axvline(0, color=p.faint, lw=0.8, zorder=0)
    ax.set_xlabel(r"$x$  at  $t=0.5$")
    ax.set_ylabel("velocity")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-9.5, 9.5)
    ax.legend(loc="upper right", fontsize=7, handlelength=1.6)
    return fig
