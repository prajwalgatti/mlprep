"""Figures for fund.linear-regression (OLS and its geometry).

All data are synthetic. Fits are real OLS fits computed with numpy.
"""
import numpy as np
from matplotlib.patches import Polygon

from figures.style import arrow, blank, figure, register


def _proj(P):
    """Oblique projection of 3-D points (rows) to the page."""
    P = np.atleast_2d(P)
    M = np.array([[1.0, 0.0], [0.55, 0.32], [0.0, 1.0]])   # x→right, y→up-right (depth), z→up
    return P @ M


@register("fund.linear-regression", "projection")
def projection(p):
    """y in R^3 projected orthogonally onto the plane spanned by the columns x1, x2."""
    x1 = np.array([2.2, 0.0, 0.0])
    x2 = np.array([0.6, 2.0, 0.0])
    y = np.array([1.9, 1.3, 1.7])
    X = np.c_[x1, x2]
    w = np.linalg.solve(X.T @ X, X.T @ y)
    yh = X @ w
    fig, ax = figure(2.5)
    blank(ax, (-0.5, 4.5), (-0.35, 2.1))
    corners = np.array([[-0.3, -0.3, 0], [3.0, -0.3, 0], [3.0, 2.6, 0], [-0.3, 2.6, 0]])
    ax.add_patch(Polygon(_proj(corners), closed=True, facecolor=p.surface, edgecolor=p.faint, lw=1))
    o = _proj([0, 0, 0])[0]
    for v, name, off in [(x1, r"$x_1$", (0.05, -0.17)), (x2, r"$x_2$", (0.08, 0.02))]:
        e = _proj(v)[0]
        arrow(ax, o, e, p, color=p.c(2), lw=1.4)
        ax.text(e[0] + off[0], e[1] + off[1], name, color=p.c(2), fontsize=9)
    ye, yhe = _proj(y)[0], _proj(yh)[0]
    arrow(ax, o, ye, p, color=p.fg, lw=1.6)
    arrow(ax, o, yhe, p, color=p.accent, lw=1.6)
    ax.plot([yhe[0], ye[0]], [yhe[1], ye[1]], color=p.label, lw=1.5, ls="--")
    # right-angle marker at y_hat, built in 3-D from the residual and the in-plane direction
    r = (y - yh) / np.linalg.norm(y - yh)
    u = -yh / np.linalg.norm(yh)
    s = 0.17
    sq = _proj(np.array([yh + s * u, yh + s * u + s * r, yh + s * r]))
    ax.plot(sq[:, 0], sq[:, 1], color=p.label, lw=0.9)
    ax.text(ye[0] + 0.06, ye[1] + 0.02, r"$y$", color=p.fg, fontsize=9.5)
    ax.text(yhe[0] + 0.06, yhe[1] - 0.2, r"$\hat y = Hy$", color=p.accent, fontsize=9)
    mid = (ye + yhe) / 2
    ax.text(mid[0] + 0.08, mid[1], r"$r = y-\hat y$", color=p.label, fontsize=8.5)
    c = _proj([2.9, 2.45, 0])[0]
    ax.text(c[0], c[1] - 0.12, "col(X)", color=p.muted, fontsize=8, ha="right", va="top")
    return fig


def _lev_data():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 4, 14)
    y = 1.0 + 0.8 * x + rng.normal(0, 0.45, x.size)
    return x, y


def _fit(x, y):
    X = np.c_[np.ones_like(x), x]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _leverage(x):
    X = np.c_[np.ones_like(x), x]
    return np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)


# Explainer figure: one far, off-trend point and one central outlier.
_FAR = (9.0, 1.0 + 0.8 * 9.0 - 4.5)
_C = (2.0, 1.0 + 0.8 * 2.0 + 3.0)
# Quiz figure: B is off-trend at moderate leverage; A has the highest leverage but sits exactly on the
# line fitted to all the other points, so deleting it changes nothing (its leave-one-out residual is 0).
_B = (6.5, 1.0 + 0.8 * 6.5 - 3.5)


def _quiz_A():
    x0, y0 = _lev_data()
    w = _fit(np.r_[x0, _B[0], _C[0]], np.r_[y0, _B[1], _C[1]])
    return (10.0, w[0] + w[1] * 10.0)


@register("fund.linear-regression", "leverage")
def leverage(p):
    """Leverage as marker size; the fit with and without the high-leverage, off-trend point B."""
    x0, y0 = _lev_data()
    x = np.r_[x0, _FAR[0], _C[0]]
    y = np.r_[y0, _FAR[1], _C[1]]
    h = _leverage(x)
    w_all = _fit(x, y)
    w_noB = _fit(np.r_[x0, _C[0]], np.r_[y0, _C[1]])
    fig, ax = figure(2.5)
    ax.scatter(x, y, s=12 + 260 * h, color=p.accent, alpha=0.75, edgecolors="none", zorder=3)
    xs = np.linspace(-0.3, 9.8, 50)
    ax.plot(xs, w_all[0] + w_all[1] * xs, color=p.fg, lw=1.6, label="OLS, all points")
    ax.plot(xs, w_noB[0] + w_noB[1] * xs, color=p.c(2), lw=1.4, ls="--", label="OLS without the far point")
    iB, iC = len(x) - 2, len(x) - 1
    ax.annotate(f"far point, h = {h[iB]:.2f}", (x[iB], y[iB]), xytext=(x[iB] - 4.6, y[iB] - 1.6),
                color=p.label, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.annotate(f"outlier, h = {h[iC]:.2f}", (x[iC], y[iC]), xytext=(x[iC] + 0.8, y[iC] + 0.5),
                color=p.label, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.legend(loc="upper left", handlelength=1.6)
    return fig


@register("fund.linear-regression", "leverage-quiz")
def leverage_quiz(p):
    """[fig-Q] Same cloud plus points A, B, C, all equal size, with the all-points fit only."""
    x0, y0 = _lev_data()
    A = _quiz_A()
    x = np.r_[x0, A[0], _B[0], _C[0]]
    y = np.r_[y0, A[1], _B[1], _C[1]]
    w = _fit(x, y)
    fig, ax = figure(2.4)
    ax.plot(x0, y0, "o", ms=4, color=p.accent, alpha=0.8)
    for (px, py), name, off in [(A, "A", (0.25, 0.05)), (_B, "B", (0.25, -0.1)), (_C, "C", (0.25, 0.05))]:
        ax.plot([px], [py], "o", ms=5.5, color=p.label, zorder=4)
        ax.text(px + off[0], py + off[1], name, color=p.label, fontsize=9, va="center")
    xs = np.linspace(-0.3, 9.8, 50)
    ax.plot(xs, w[0] + w[1] * xs, color=p.fg, lw=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, 10.4)
    ax.set_ylim(-0.3, 7.3)
    return fig


@register("fund.linear-regression", "collinearity")
def collinearity(p):
    """RSS contours in (w1, w2) for uncorrelated vs strongly correlated standardized features."""
    rng = np.random.default_rng(0)
    n = 60
    fig, axes = figure(2.2, ncols=2, sharey=True)
    for ax, rho in zip(axes, [0.0, 0.95]):
        z = rng.normal(size=(n, 2))
        z = z - z.mean(0)
        z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T, bias=True))).T   # exactly white
        X = z @ np.linalg.cholesky(np.array([[1, rho], [rho, 1]])).T          # exactly corr = rho
        y = X @ np.array([1.0, 1.0]) + rng.normal(0, 1.0, n)
        what = np.linalg.solve(X.T @ X, X.T @ y)
        g = np.linspace(-2.0, 4.0, 241)
        W1, W2 = np.meshgrid(g, g)
        Wf = np.stack([W1.ravel(), W2.ravel()], 1)
        R = ((y[None, :] - Wf @ X.T) ** 2).sum(1).reshape(W1.shape)
        rmin = R.min()
        levels = rmin * np.array([1.05, 1.2, 1.5, 2.0, 3.0, 4.5])
        ax.contour(W1, W2, R, levels=levels, colors=[p.c(0)], linewidths=0.9)
        ax.plot([what[0]], [what[1]], "o", color=p.label, ms=4.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        ax.set_xticks([-1, 1, 3])
        ax.set_yticks([-1, 1, 3])
        ax.set_xlabel(r"$w_1$")
        r_emp = np.corrcoef(X.T)[0, 1]
        ax.set_title(rf"corr$(x_1,x_2)$ = {r_emp:.2f}", fontsize=8.5)
    axes[0].set_ylabel(r"$w_2$")
    return fig
