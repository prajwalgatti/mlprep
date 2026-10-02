"""Figures for gen.guidance. All toys use exact (analytic) scores of Gaussian mixtures, no learned model.

Noise convention for the toys: variance-exploding, x_sigma = x_0 + sigma * eps, sampled with the
probability-flow ODE dx/dsigma = -sigma * score (Heun steps on a log-spaced sigma grid).
Guidance scale s: guided score = s_u + s * (s_c - s_u), so s = 1 is plain conditional sampling.
"""
from functools import lru_cache

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

from figures.style import figure, register


# ---------------------------------------------------------------- 1-D: tilted densities
@register("gen.guidance", "sharpening")
def sharpening(p):
    """p(x) p(+|x)^s for a two-class 1-D mixture, s = 0, 1, 3, 10."""
    x = np.linspace(-4.5, 5, 1200)
    px = 0.5 * norm.pdf(x, -1, 1) + 0.5 * norm.pdf(x, 1, 1)
    py = expit(2 * x)                       # exact p(+|x) for this mixture
    fig, ax = figure(2.2)
    styles = [(0, p.muted, "--", r"$s=0$: $p(x)$"), (1, p.c(0), "-", r"$s=1$: $p(x\,|\,+)$"),
              (3, p.c(1), "-", r"$s=3$"), (10, p.c(2), "-", r"$s=10$")]
    for s, col, ls, lab in styles:
        d = px * py ** s
        d /= np.trapz(d, x)
        ax.plot(x, d, color=col, ls=ls, lw=1.6 if s else 1.2, label=lab)
    ax.axvline(0, color=p.faint, lw=0.8, zorder=0)
    ax.text(-0.15, 0.62, "class −", color=p.muted, fontsize=7.5, ha="right")
    ax.text(0.15, 0.62, "class +", color=p.muted, fontsize=7.5, ha="left")
    ax.set_xlim(-4.5, 5)
    ax.set_ylim(0, 0.68)
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\propto p(x)\,p(+|x)^s$")
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.6, borderaxespad=0.2)
    return fig


# ---------------------------------------------------------------- 2-D: trade-off
TAU2 = 0.45
MU2 = np.array([[1.5, 0.9], [-0.5, -0.6], [-1.3, -1.0], [0.3, 1.7]])
CL2 = np.array([0, 0, 1, 1])                 # class A = components 0 (major) and 1 (minor)
PI2 = np.array([0.35, 0.15, 0.30, 0.20])     # joint weights; class A has mass 0.5, minor share 0.3


def _logcomp(x, sig):
    v = TAU2 ** 2 + sig ** 2
    d = x[:, None, :] - MU2[None]
    return -0.5 * (d ** 2).sum(-1) / v - np.log(2 * np.pi * v), d, v


def _score2(x, sig, cls=None):
    lc, d, v = _logcomp(x, sig)
    lw = lc + np.log(PI2)[None]
    if cls is not None:
        lw = np.where(CL2[None] == cls, lw, -np.inf)
    r = np.exp(lw - logsumexp(lw, 1, keepdims=True))
    return -(r[:, :, None] * d).sum(1) / v


def _sample2(s, n=2500, seed=0, smax=40.0, smin=0.005, steps=300):
    rng = np.random.default_rng(seed)
    sigs = np.geomspace(smax, smin, steps)
    comp = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    x = MU2[comp] + np.sqrt(TAU2 ** 2 + smax ** 2) * rng.standard_normal((n, 2))

    def f(x_, sg):
        su = _score2(x_, sg)
        return -sg * (su + s * (_score2(x_, sg, 0) - su))

    for a, b in zip(sigs[:-1], sigs[1:]):
        h = b - a
        k1 = f(x, a)
        k2 = f(x + h * k1, b)
        x = x + h * (k1 + k2) / 2
    return x


def _resp(x):
    lc, _, _ = _logcomp(x, 0.0)
    lw = lc + np.log(PI2)[None]
    return np.exp(lw - logsumexp(lw, 1, keepdims=True))


@lru_cache(maxsize=None)
def _tradeoff_data():
    ss = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8])
    conf, cover, samples = [], [], {}
    for s in ss:
        x = _sample2(float(s))
        r = _resp(x)
        conf.append(r[:, CL2 == 0].sum(1).mean())
        cover.append((r[:, 1] > r[:, 0]).mean() / 0.3)
        if s in (1, 3, 8):
            samples[float(s)] = x[:500]
    # exact tilted density p(x) p(A|x)^s on a grid, for comparison
    g = np.linspace(-5, 5, 401)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    r = _resp(P)
    lc, _, _ = _logcomp(P, 0.0)
    lp = logsumexp(lc + np.log(PI2)[None], 1)
    pA = r[:, CL2 == 0].sum(1)
    tilt_cover = []
    sd = np.linspace(1, 8, 30)
    for s in sd:
        ld = lp + s * np.log(pA + 1e-300)
        w = np.exp(ld - ld.max())
        w /= w.sum()
        tilt_cover.append((w * (r[:, 1] > r[:, 0])).sum() / 0.3)
    return ss, np.array(conf), np.array(cover), samples, sd, np.array(tilt_cover)


@register("gen.guidance", "tradeoff-toy")
def tradeoff_toy(p):
    ss, conf, cover, samples, sd, tilt = _tradeoff_data()
    fig = figure(3.5)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    g = np.linspace(-3, 3.2, 200)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    lc, _, _ = _logcomp(P, 0.0)
    for j, s in enumerate((1.0, 3.0, 8.0)):
        ax = fig.add_subplot(gs[0, j])
        for k, col in ((0, p.c(0)), (1, p.c(1))):
            dens = np.exp(lc[:, CL2 == k] + np.log(PI2[CL2 == k])).sum(1).reshape(X.shape)
            ax.contour(X, Y, dens, levels=[0.05, 0.25], colors=[col], linewidths=0.6, alpha=0.7)
        xs = samples[s]
        ax.scatter(xs[:, 0], xs[:, 1], s=1.2, color=p.fg, alpha=0.6, lw=0)
        ax.set_xlim(-3, 3.2)
        ax.set_ylim(-2.6, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"$s={s:g}$", fontsize=8.5, pad=2)
        if j == 0:
            ax.text(1.5, 2.55, "A", color=p.c(0), fontsize=8, ha="center")
            ax.text(-0.6, 0.15, "A", color=p.c(0), fontsize=7, ha="center")
            ax.text(-2.2, -1.9, "B", color=p.c(1), fontsize=8, ha="center")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ss, conf, "o-", color=p.c(0), ms=3, label=r"class confidence $E[p(A|x)]$")
    ax.plot(ss, cover, "o-", color=p.c(2), ms=3, label="minor-mode coverage (guided ODE)")
    ax.plot(sd, tilt, "--", color=p.c(2), lw=1.1, label=r"coverage of exact $p(x)p(A|x)^s$")
    ax.axhline(1, color=p.faint, lw=0.8, zorder=0)
    ax.set_xlabel("guidance scale $s$  ($s=1$: no guidance)")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(0.8, 8.2)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=7, handlelength=1.6, borderaxespad=0.1)
    return fig


# ---------------------------------------------------------------- 1-D: over-saturation
MU1 = np.array([0.35, 0.8, -0.8, -0.35])
CL1 = np.array([0, 0, 1, 1])
TAU1 = 0.08


def _den1(x, sig, cls=None):
    v = TAU1 ** 2 + sig ** 2
    lw = -0.5 * (x[:, None] - MU1) ** 2 / v
    if cls is not None:
        lw = np.where(CL1 == cls, lw, -np.inf)
    r = np.exp(lw - logsumexp(lw, 1, keepdims=True))
    return (r * (MU1 + TAU1 ** 2 / v * (x[:, None] - MU1))).sum(1)   # exact E[x0 | x_sigma]


@lru_cache(maxsize=None)
def _sat_data(s, n=4000, smax=40.0, smin=0.002, steps=400):
    rng = np.random.default_rng(0)
    sigs = np.geomspace(smax, smin, steps)
    x = MU1[rng.choice([0, 1], size=n)] + np.sqrt(TAU1 ** 2 + smax ** 2) * rng.standard_normal(n)

    def gden(x_, sg):
        du = _den1(x_, sg)
        return du + s * (_den1(x_, sg, 0) - du)

    rec_s, med, lo, hi = [], [], [], []
    for a, b in zip(sigs[:-1], sigs[1:]):
        h = b - a
        k1 = (x - gden(x, a)) / a
        xe = x + h * k1
        k2 = (xe - gden(xe, b)) / b
        x = x + h * (k1 + k2) / 2
        x0 = gden(x, b)
        rec_s.append(b)
        q = np.percentile(x0, [10, 50, 90])
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    return np.array(rec_s), np.array(med), np.array(lo), np.array(hi), x


@register("gen.guidance", "oversaturation")
def oversaturation(p):
    fig, (a1, a2) = figure(2.5, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    cols = {1.0: p.muted, 3.0: p.c(1), 7.5: p.c(5) if p.theme != "terminal" else p.bad}
    for s, col in cols.items():
        sg, med, lo, hi, xf = _sat_data(s)
        a1.fill_between(sg, lo, hi, color=col, alpha=0.18, lw=0)
        a1.plot(sg, med, color=col, lw=1.5, label=f"$s={s:g}$")
        a2.hist(xf, bins=np.linspace(-0.2, 1.3, 61), density=True, histtype="step", color=col, lw=1.2)
    a1.axhspan(-1, 1, color=p.faint, alpha=0.6, lw=0, zorder=0)
    a1.set_xscale("log")
    a1.set_xlim(40, 0.003)
    a1.set_ylim(-0.3, 5.5)
    a1.set_xlabel(r"noise level $\sigma$ (sampling →)")
    a1.set_ylabel(r"guided $\hat x_0$")
    a1.text(0.004, -0.22, "data range [−1, 1]", color=p.muted, fontsize=7, ha="right", va="bottom")
    a1.legend(loc="upper right", fontsize=7, handlelength=1.2)
    a2.axvline(1, color=p.fg, lw=0.8, ls=":")
    a2.text(1.03, a2.get_ylim()[1] * 0.92, "edge", color=p.muted, fontsize=7)
    a2.set_yticks([])
    a2.set_xlabel("final sample")
    a2.set_xticks([0, 0.35, 0.8, 1.2])
    a2.set_xticklabels(["0", ".35", ".8", "1.2"])
    return fig
