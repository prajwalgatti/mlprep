"""Figures for fund.logistic-regression. All data are synthetic; every model is actually fitted."""
import numpy as np
from scipy.optimize import minimize

from figures.style import figure, register


def _sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def _newton_lr(X, y, lam=1e-6, iters=50):
    """Logistic regression by Newton/IRLS on the mean NLL plus (lam/2)||w||^2 (X includes a ones column)."""
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        mu = _sig(X @ w)
        g = X.T @ (mu - y) / n + lam * w
        H = (X * (mu * (1 - mu))[:, None]).T @ X / n + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


@register("fund.logistic-regression", "sigmoid-boundary")
def sigmoid_boundary(p):
    """Two overlapping Gaussian classes; contours of the fitted p(y=1|x) at 0.1, 0.5 and 0.9."""
    rng = np.random.default_rng(4)
    n = 60
    X0 = rng.multivariate_normal([-1.0, -0.5], [[1.0, 0.4], [0.4, 1.0]], n)
    X1 = rng.multivariate_normal([1.2, 0.8], [[1.0, 0.4], [0.4, 1.0]], n)
    Xr = np.r_[X0, X1]
    y = np.r_[np.zeros(n), np.ones(n)]
    w = _newton_lr(np.c_[np.ones(2 * n), Xr], y)
    g1, g2 = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-3.5, 3.5, 200))
    P = _sig(w[0] + w[1] * g1 + w[2] * g2)
    fig, ax = figure(2.6)
    ax.plot(X0[:, 0], X0[:, 1], "o", ms=3.2, color=p.c(0), alpha=0.8, label="y = 0")
    ax.plot(X1[:, 0], X1[:, 1], "^", ms=3.4, color=p.c(1), alpha=0.85, label="y = 1")
    ax.contour(g1, g2, P, levels=[0.1, 0.5, 0.9], colors=[p.muted, p.fg, p.muted],
               linewidths=[0.9, 1.6, 0.9], linestyles=["--", "-", "--"])
    # label each contour where it leaves the top edge: w0 + w1 x1 + w2 x2 = logit(level)
    top = 3.5
    for lev in (0.1, 0.5, 0.9):
        xt = (np.log(lev / (1 - lev)) - w[0] - w[2] * top) / w[1]
        ax.text(xt, top + 0.12, f"{lev}", color=p.fg if lev == 0.5 else p.muted, fontsize=7.5,
                ha="center", va="bottom")
    ax.text(-4, top + 0.12, "p(y=1|x):", color=p.muted, fontsize=7.5, ha="left", va="bottom")
    # weight vector, drawn from the p = 0.5 line near the bottom edge, normal to it
    yb = -2.7
    c = np.array([(-w[0] - w[2] * yb) / w[1], yb])
    d = np.array([w[1], w[2]]) / np.hypot(w[1], w[2])
    ax.annotate("", xy=c + 1.2 * d, xytext=c, arrowprops=dict(arrowstyle="-|>", color=p.label, lw=1.3))
    ax.text(*(c + 1.3 * d + np.array([0.05, 0.05])), r"$w$", color=p.label, fontsize=9.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    return fig


def _separable_data():
    rng = np.random.default_rng(2)
    n = 40
    X0 = rng.normal([-1.3, -0.4], 0.55, (n, 2))
    X1 = rng.normal([1.3, 0.6], 0.55, (n, 2))
    X = np.r_[X0, X1]
    s = np.r_[-np.ones(n), np.ones(n)]          # +-1 labels; no bias, so the separator passes through 0
    keep = s * (X @ np.array([1.0, 0.45])) > 0.35    # enforce a clear margin
    return X[keep], s[keep]


def _max_margin_direction(X, s):
    cons = {"type": "ineq", "fun": lambda w: s * (X @ w) - 1.0}
    res = minimize(lambda w: 0.5 * w @ w, np.array([1.0, 0.0]), constraints=[cons], method="SLSQP")
    return res.x / np.linalg.norm(res.x)


def _gd_path(X, s, lam, T, eta=1.0):
    w = np.zeros(2)
    rec_t = np.unique(np.round(np.logspace(0, np.log10(T), 120)).astype(int))
    norms, angles, losses = [], [], []
    j = 0
    for t in range(1, T + 1):
        m = s * (X @ w)
        g = -(X * (s * _sig(-m))[:, None]).mean(0) + lam * w
        w -= eta * g
        if t == rec_t[j]:
            norms.append(np.linalg.norm(w))
            losses.append(np.mean(np.logaddexp(0, -s * (X @ w))))
            angles.append(w / np.linalg.norm(w))
            j += 1
            if j == len(rec_t):
                break
    return rec_t, np.array(norms), np.array(angles), np.array(losses)


@register("fund.logistic-regression", "separable-divergence")
def separable_divergence(p):
    """GD on separable data: ||w|| grows without bound (unregularized) and its direction creeps toward max margin."""
    X, s = _separable_data()
    u = _max_margin_direction(X, s)
    T = 100_000
    t, n0, a0, _ = _gd_path(X, s, 0.0, T)
    t2, n2, _, _ = _gd_path(X, s, 0.01, T)
    ang = np.degrees(np.arccos(np.clip(a0 @ u, -1, 1)))
    fig, axes = figure(2.2, ncols=2)
    ax = axes[0]
    ax.plot(t, n0, color=p.c(0), label="no penalty")
    ax.plot(t2, n2, color=p.c(1), label=r"L2, $\lambda=0.01$")
    ax.set_xscale("log")
    ax.set_xlabel("GD step")
    ax.set_ylabel(r"$\|w_t\|$")
    ax.set_xticks([1, 100, 10_000])
    ax.set_xticklabels(["1", "$10^2$", "$10^4$"])
    ax.legend(loc="upper left", handlelength=1.3)
    ax = axes[1]
    ax.plot(t, ang, color=p.c(0))
    ax.set_xscale("log")
    ax.set_xticks([1, 100, 10_000])
    ax.set_xticklabels(["1", "$10^2$", "$10^4$"])
    ax.set_xlabel("GD step")
    ax.set_ylabel("angle to max-margin w (°)")
    ax.set_ylim(0, None)
    return fig


@register("fund.logistic-regression", "softmax-temperature")
def softmax_temperature(p):
    """[fig-Q] Softmax over logits (z1, 1, 0) as z1 varies, at temperature 1 and 3. Curves are unlabeled letters."""
    z1 = np.linspace(-4, 5, 300)
    fig, axes = figure(2.2, ncols=2, sharey=True)
    for ax, T in zip(axes, [1, 3]):
        Z = np.stack([z1, np.ones_like(z1), np.zeros_like(z1)], 1) / T
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        for k, name in enumerate("ABC"):
            ax.plot(z1, P[:, k], color=p.c(k), lw=1.6)
        ia = np.searchsorted(z1, 2.4)
        ax.text(2.4, P[ia, 0] + 0.07, "A", color=p.c(0), fontsize=8.5, va="bottom", ha="right")
        ax.text(-3.8, P[0, 1] + 0.03, "B", color=p.c(1), fontsize=8.5, va="bottom")
        ax.text(-3.8, P[0, 2] - 0.03, "C", color=p.c(2), fontsize=8.5, va="top")
        ax.set_title(f"T = {T}", fontsize=8.5)
        ax.set_xlabel(r"logit $z_1$")
        ax.set_ylim(0, 1)
        ax.set_xticks([-4, -2, 0, 2, 4])
    axes[0].set_ylabel("probability")
    return fig
