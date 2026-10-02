"""Figures for fund.sgd-momentum. Runs of the real update rules on small synthetic quadratics."""
import numpy as np

from figures.style import figure, register


def _contours(ax, lams, p, lim_x, lim_y, n=6):
    gx, gy = np.meshgrid(np.linspace(-lim_x, lim_x, 300), np.linspace(-lim_y, lim_y, 300))
    f = 0.5 * (lams[0] * gx ** 2 + lams[1] * gy ** 2)
    levels = 0.5 * lams[0] * (lim_x * np.linspace(0.15, 1.0, n)) ** 2
    ax.contour(gx, gy, f, levels=levels, colors=p.faint, linewidths=0.8)


@register("fund.sgd-momentum", "sgd-noise-ball")
def sgd_noise_ball(p):
    """SGD with additive gradient noise on f = (x^2 + 4y^2)/2: constant vs decaying step size."""
    lams = np.array([1.0, 4.0])
    rng = np.random.default_rng(3)
    T = 600
    sigma = 1.0
    fig, axes = figure(2.1, ncols=2, sharey=True)
    for ax, mode in zip(axes, ["const", "decay"]):
        x = np.array([-2.6, 1.0])
        xs = [x.copy()]
        for t in range(T):
            eta = 0.12 if mode == "const" else 0.12 / (1 + t / 40)
            g = lams * x + sigma * rng.normal(size=2)
            x = x - eta * g
            xs.append(x.copy())
        xs = np.array(xs)
        _contours(ax, lams, p, 3.0, 1.5)
        ax.plot(xs[:60, 0], xs[:60, 1], "-", lw=0.8, color=p.muted, alpha=0.9)
        ax.plot(xs[300:, 0], xs[300:, 1], ".", ms=2.2, color=p.accent, alpha=0.8)
        ax.plot([0], [0], "*", color=p.label, ms=7, zorder=5)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("constant $\\eta$" if mode == "const" else r"$\eta_t=\eta_0/(1+t/40)$", fontsize=8.5)
    axes[0].text(-2.9, -1.45, "grey: first 60 steps\ndots: steps 300-600", fontsize=6.8, color=p.muted)
    return fig


@register("fund.sgd-momentum", "momentum-ravine")
def momentum_ravine(p):
    """GD, heavy-ball and Nesterov on f = (x^2 + 50 y^2)/2 with the same eta, 40 steps."""
    lams = np.array([1.0, 50.0])
    eta, beta, T = 0.024, 0.8, 40
    x0 = np.array([-4.0, 0.6])

    def run(kind):
        x, v = x0.copy(), np.zeros(2)
        xs = [x.copy()]
        for _ in range(T):
            if kind == "gd":
                x = x - eta * lams * x
            elif kind == "hb":
                v = beta * v - eta * lams * x
                x = x + v
            else:
                v = beta * v - eta * lams * (x + beta * v)
                x = x + v
            xs.append(x.copy())
        return np.array(xs)

    fig, axes = figure(2.9, nrows=3, sharex=True)
    for ax, kind, col, name in zip(axes, ["gd", "hb", "nag"], [p.c(1), p.c(0), p.c(2)],
                                   ["gradient descent", r"heavy ball, $\beta=0.8$", r"Nesterov, $\beta=0.8$"]):
        xs = run(kind)
        _contours(ax, lams, p, 4.4, 0.75, n=5)
        ax.plot(xs[:, 0], xs[:, 1], "-o", ms=1.8, lw=0.9, color=col)
        ax.plot([0], [0], "*", color=p.label, ms=6, zorder=5)
        ax.set_xlim(-4.4, 1.6)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(1.55, 0.45, name, ha="right", fontsize=7.5, color=col)
        d = np.linalg.norm(xs[-1])
        ax.text(1.55, -0.62, f"distance after {T} steps: {d:.2f}", ha="right", fontsize=6.8, color=p.muted)
    return fig


@register("fund.sgd-momentum", "momentum-contraction")
def momentum_contraction(p):
    """Per-step contraction vs curvature: GD at its best step vs heavy ball at its best (eta, beta); mu=1, L=100."""
    mu, L = 1.0, 100.0
    lam = np.logspace(-0.5, 2.4, 600)
    eta_gd = 2 / (L + mu)
    gd = np.abs(1 - eta_gd * lam)
    sk = np.sqrt(L / mu)
    beta = ((sk - 1) / (sk + 1)) ** 2
    eta_hb = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    hb = []
    for l in lam:
        M = np.array([[1 + beta - eta_hb * l, -beta], [1, 0]])
        hb.append(np.max(np.abs(np.linalg.eigvals(M))))
    hb = np.array(hb)
    fig, ax = figure(2.4)
    ax.axvspan(mu, L, color=p.faint, alpha=0.6, lw=0)
    ax.plot(lam, gd, color=p.c(1), label=r"GD, $\eta=2/(L+\mu)$")
    ax.plot(lam, hb, color=p.c(0), label=r"heavy ball, optimal $(\eta,\beta)$")
    ax.axhline(1, color=p.muted, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.3)
    ax.set_xlim(lam[0], lam[-1])
    ax.set_xlabel(r"curvature $\lambda$ of an eigendirection")
    ax.set_ylabel("contraction per step")
    ax.text(1.15, 0.05, r"spectrum $[\mu,L]=[1,100]$", fontsize=7.2, color=p.muted)
    ax.text(1.2, 0.70, r"$\sqrt{\beta}=9/11\approx0.82$", fontsize=7.5, color=p.c(0))
    ax.text(3.0, 1.03, r"worst $=99/101\approx0.98$", fontsize=7.5, color=p.c(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=7.2, handlelength=1.4)
    return fig


def _sched(kind, t, T=10000, w=600, peak=1.0):
    s = np.zeros_like(t, dtype=float)
    warm = t < w
    tau = np.clip((t - w) / (T - w), 0, 1)
    if kind == "cosine":
        s = 0.5 * peak * (1 + np.cos(np.pi * tau))
    elif kind == "linear":
        s = peak * (1 - tau)
    elif kind == "step":
        s = peak * np.where(tau < 0.5, 1.0, np.where(tau < 0.8, 0.1, 0.01))
        return np.where(warm, peak * t / w, s)
    elif kind == "wsd":
        d = 0.8
        s = np.where(tau < d, peak, peak * (1 - (tau - d) / (1 - d)))
    return np.where(warm, peak * t / w, s)


@register("fund.sgd-momentum", "schedules-quiz")
def schedules_quiz(p):
    """[fig-Q] Four schedules, each after linear warmup: A linear decay, B WSD, C cosine, D step."""
    t = np.linspace(0, 10000, 1000)
    fig, axes = figure(2.4, nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, (lab, kind) in zip(axes.flat, [("A", "linear"), ("B", "wsd"), ("C", "cosine"), ("D", "step")]):
        ax.plot(t, _sched(kind, t), color=p.accent, lw=1.5)
        ax.text(0.95, 0.9, lab, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=p.label,
                fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 1])
        ax.set_xticks([0, 5000, 10000])
        ax.set_xticklabels(["0", "5k", "10k"])
    for ax in axes[1]:
        ax.set_xlabel("step", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\eta_t/\eta_{max}$", fontsize=8)
    return fig
