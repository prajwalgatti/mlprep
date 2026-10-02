"""Figures for fund.kl-divergence. Fits and divergences are computed numerically on a grid."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from figures.style import figure, register

X = np.linspace(-8, 8, 8001)
DX = X[1] - X[0]


def _mix(w, m=2.0, s=0.6):
    """w * N(-m, s^2) + (1 - w) * N(m, s^2)."""
    return w * norm.pdf(X, -m, s) + (1 - w) * norm.pdf(X, m, s)


def _rev_kl(params, p):
    mu, log_s = params
    q = norm.pdf(X, mu, np.exp(log_s))
    m = q > 1e-300
    return np.sum(q[m] * (np.log(q[m]) - np.log(p[m] + 1e-300))) * DX


def _moment_match(p):
    mean = np.sum(X * p) * DX
    var = np.sum((X - mean) ** 2 * p) * DX
    return mean, np.sqrt(var)


def _rev_fit(p, init):
    r = minimize(_rev_kl, [init, 0.0], args=(p,), method="Nelder-Mead")
    return r.x[0], np.exp(r.x[1])


@register("fund.kl-divergence", "forward-vs-reverse")
def forward_vs_reverse(p):
    """Best single Gaussian under each KL direction for an equal-weight bimodal p."""
    P = _mix(0.5)
    mf, sf = _moment_match(P)
    mr, sr = _rev_fit(P, 2.0)
    fig, ax = figure(2.35)
    ax.fill_between(X, P, color=p.muted, alpha=0.25, lw=0)
    ax.plot(X, P, color=p.muted, lw=1.2)
    ax.plot(X, norm.pdf(X, mf, sf), color=p.c(0), lw=2)
    ax.plot(X, norm.pdf(X, mr, sr), color=p.c(1), lw=2)
    ax.text(-2.0, 0.36, "target p", color=p.muted, fontsize=8, ha="center")
    ax.text(-5.8, 0.55, r"argmin KL$(p\|q)$" + "\nmass-covering", color=p.c(0), fontsize=7.5, ha="left")
    ax.text(3.0, 0.62, r"argmin KL$(q\|p)$" + "\nmode-seeking", color=p.c(1), fontsize=7.5, ha="left")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.78)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    return fig


@register("fund.kl-divergence", "reverse-quiz")
def reverse_quiz(p):
    """Three candidate Gaussians for a 0.7/0.3 bimodal p (question stem; no answer marked)."""
    P = _mix(0.7)
    cands = [_moment_match(P), _rev_fit(P, -2.0), _rev_fit(P, 2.0)]
    fig, ax = figure(2.35)
    ax.fill_between(X, P, color=p.muted, alpha=0.25, lw=0)
    ax.plot(X, P, color=p.muted, lw=1.2)
    ax.text(-4.8, 0.5, "target p", color=p.muted, fontsize=8)
    pos = [(-0.8, 0.25), (-2.0, 0.72), (2.0, 0.72)]
    for k, ((m, s), name, (tx, ty)) in enumerate(zip(cands, "ABC", pos)):
        ax.plot(X, norm.pdf(X, m, s), color=p.c(k), lw=1.9)
        ax.text(tx, ty, name, color=p.c(k), fontsize=10, ha="center", fontweight="bold")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    return fig


@register("fund.kl-divergence", "kl-asymmetry")
def kl_asymmetry(p):
    """Densities N(0,1), N(0,2^2) and the two KL integrands."""
    xw = np.linspace(-16, 16, 32001)   # wide grid so the heavy q-tail integral converges
    aw, bw = norm.pdf(xw, 0, 1), norm.pdf(xw, 0, 2)
    kl_ab = np.sum(aw * np.log(aw / bw)) * (xw[1] - xw[0])
    kl_ba = np.sum(bw * np.log(bw / aw)) * (xw[1] - xw[0])
    a = norm.pdf(X, 0, 1)
    b = norm.pdf(X, 0, 2)
    f_ab = a * np.log(a / b)
    f_ba = b * np.log(b / a)
    fig, (ax1, ax2) = figure(3.2, nrows=2, sharex=True)
    ax1.plot(X, a, color=p.c(0), lw=1.8)
    ax1.plot(X, b, color=p.c(1), lw=1.8)
    ax1.text(0.9, 0.36, r"$p=\mathcal{N}(0,1)$", color=p.c(0), fontsize=8)
    ax1.text(2.6, 0.13, r"$q=\mathcal{N}(0,2^2)$", color=p.c(1), fontsize=8)
    ax1.set_yticks([])
    ax1.set_ylabel("density")
    ax2.axhline(0, color=p.muted, lw=0.6)
    ax2.plot(X, f_ab, color=p.c(0), lw=1.8)
    ax2.plot(X, f_ba, color=p.c(1), lw=1.8)
    ax2.text(-5.8, 0.33, rf"$p\log(p/q)$: area {kl_ab:.3f}", color=p.c(0), fontsize=7.5)
    ax2.text(-5.8, 0.41, rf"$q\log(q/p)$: area {kl_ba:.3f}", color=p.c(1), fontsize=7.5)
    ax2.set_ylabel("integrand")
    ax2.set_xlabel("x")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-0.15, 0.48)
    ax2.set_yticks([0, 0.2, 0.4])
    return fig


def _jsd(a, b):
    m = 0.5 * (a + b)
    t1 = np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(m > 0, m, 1)), 0)
    t2 = np.where(b > 0, b * np.log(np.where(b > 0, b, 1) / np.where(m > 0, m, 1)), 0)
    return 0.5 * (t1.sum() + t2.sum()) * DX


@register("fund.kl-divergence", "js-saturation")
def js_saturation(p):
    """JSD and W1 between P0 and its copy shifted by theta: point masses and narrow Gaussians."""
    th = np.linspace(-3, 3, 241)
    s = 0.25
    a = norm.pdf(X, 0, s)
    jsd_g = np.array([_jsd(a, norm.pdf(X, t, s)) for t in th])
    fig, (ax1, ax2) = figure(2.2, ncols=2)
    ax1.axhline(np.log(2), color=p.muted, lw=0.7, ls=":")
    ax1.plot(th, np.where(th == 0, np.nan, np.log(2)), color=p.c(1), lw=2)
    ax1.plot([0], [0], "o", color=p.c(1), ms=4)
    ax1.plot([0], [np.log(2)], "o", mfc="none", mec=p.c(1), ms=4)
    ax1.plot(th, jsd_g, color=p.c(0), lw=1.6)
    ax1.text(-2.9, 0.47, "point\nmasses", color=p.c(1), fontsize=7.5)
    ax1.text(-2.9, 0.12, "Gaussians,\n" + r"$\sigma=0.25$", color=p.c(0), fontsize=7.5)
    ax1.set_title("JSD (nats)", fontsize=8.5)
    ax1.set_yticks([0, np.log(2)])
    ax1.set_yticklabels(["0", "log 2"])
    ax1.set_ylim(-0.03, 0.8)
    ax1.set_xlabel(r"shift $\theta$")
    ax2.plot(th, np.abs(th), color=p.c(2), lw=2)
    ax2.text(0, 2.5, "both cases:\n" + r"$W_1=|\theta|$", color=p.c(2), fontsize=7.5, ha="center")
    ax2.set_title(r"$W_1$ distance", fontsize=8.5)
    ax2.set_xlabel(r"shift $\theta$")
    ax2.set_yticks([0, 1, 2, 3])
    return fig
