"""Figures for fund.gradient-descent. All curves are exact GD runs on small quadratics."""
import numpy as np

from figures.style import figure, register


def _gd_path(lams, x0, eta, steps):
    lams = np.asarray(lams, float)
    xs = [np.asarray(x0, float)]
    for _ in range(steps):
        xs.append(xs[-1] - eta * lams * xs[-1])
    return np.array(xs)


def _contours(ax, lams, p, lim_x, lim_y):
    gx, gy = np.meshgrid(np.linspace(-lim_x, lim_x, 300), np.linspace(-lim_y, lim_y, 300))
    f = 0.5 * (lams[0] * gx ** 2 + lams[1] * gy ** 2)
    levels = 0.5 * lams[0] * (lim_x * np.linspace(0.12, 1.0, 7)) ** 2
    ax.contour(gx, gy, f, levels=levels, colors=p.faint, linewidths=0.8)


@register("fund.gradient-descent", "zigzag")
def zigzag(p):
    """GD with the best fixed step on a well-conditioned (kappa=2) and an ill-conditioned (kappa=20) quadratic."""
    fig, axes = figure(2.0, ncols=2)
    for ax, lams, title in zip(axes, [(1, 2), (1, 20)], [r"$\kappa=2$", r"$\kappa=20$"]):
        eta = 2 / (lams[0] + lams[1])
        path = _gd_path(lams, (-4.0, 0.9), eta, 25)
        _contours(ax, lams, p, 4.4, 1.25)
        ax.plot(path[:, 0], path[:, 1], "-o", ms=2.2, lw=1.1, color=p.accent)
        ax.plot([0], [0], "*", color=p.label, ms=7, zorder=5)
        ax.set_xlim(-4.4, 4.4)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title + "  (25 steps)", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    return fig


@register("fund.gradient-descent", "eigen-contraction")
def eigen_contraction(p):
    """|1 - eta*lambda| against eta for lambda_min = 1 and lambda_max = 10."""
    lmin, lmax = 1.0, 10.0
    eta = np.linspace(0, 0.25, 500)
    fig, ax = figure(2.4)
    ax.plot(eta, np.abs(1 - eta * lmin), color=p.c(2), label=r"$|1-\eta\lambda_{min}|$, $\lambda_{min}=1$")
    ax.plot(eta, np.abs(1 - eta * lmax), color=p.c(1), label=r"$|1-\eta\lambda_{max}|$, $\lambda_{max}=10$")
    worst = np.maximum(np.abs(1 - eta * lmin), np.abs(1 - eta * lmax))
    ax.plot(eta, worst, color=p.fg, lw=2.4, alpha=0.35, label="worst case (sets the rate)")
    es = 2 / (lmin + lmax)
    r = (lmax - lmin) / (lmax + lmin)
    ax.plot([es], [r], "o", color=p.label, ms=5, zorder=5)
    ax.annotate(r"$\eta^*=2/11$, rate $9/11$", (es, r), xytext=(0.035, 1.22), color=p.label, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.axhline(1, color=p.muted, lw=0.8, ls="--")
    ax.axvline(2 / lmax, color=p.bad, lw=0.8, ls=":")
    ax.text(2 / lmax + 0.004, 0.08, r"$2/\lambda_{max}$", color=p.bad, fontsize=7.5)
    ax.set_xlabel(r"step size $\eta$")
    ax.set_ylabel("contraction per step")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7, handlelength=1.4)
    return fig


@register("fund.gradient-descent", "lr-regimes")
def lr_regimes(p):
    """[fig-Q] log loss vs step for four step sizes on f = (x^2 + 10 y^2)/2, start (1, 1e-3)."""
    lams = np.array([1.0, 10.0])
    x0 = np.array([1.0, 1e-3])
    # panel label -> eta ; 2/L = 0.2
    panels = [("A", 0.195), ("B", 0.02), ("C", 0.205), ("D", 0.1)]
    fig, axes = figure(2.6, nrows=2, ncols=2, sharex=True, sharey=True)
    k = np.arange(151)
    for ax, (lab, eta) in zip(axes.flat, panels):
        path = _gd_path(lams, x0, eta, 150)
        f = 0.5 * (path ** 2 * lams).sum(axis=1)
        ax.plot(k, np.log10(f), color=p.accent, lw=1.4)
        ax.text(0.95, 0.9, lab, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=p.label,
                fontweight="bold")
        ax.set_ylim(-14, 7)
        ax.set_yticks([-12, -6, 0, 6])
    for ax in axes[1]:
        ax.set_xlabel("step", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10} f$", fontsize=8)
    return fig
