"""Figures for fund.svm-margin-dual. Synthetic 2-D data; SVMs are fitted with scikit-learn's SVC (linear kernel)."""
import numpy as np
from sklearn.svm import SVC

from figures.style import figure, register


def _sep_data():
    rng = np.random.default_rng(5)
    n = 18
    Xp = rng.normal([1.4, 1.2], 0.6, (n, 2))
    Xn = rng.normal([-1.2, -1.0], 0.6, (n, 2))
    X = np.r_[Xp, Xn]
    y = np.r_[np.ones(n), -np.ones(n)]
    keep = y * (X @ np.array([1.0, 1.0])) > 0.9          # make sure the classes are separable
    return X[keep], y[keep]


def _lines(ax, clf, p, xlim, lw=1.6, band=True, color=None):
    w, b = clf.coef_[0], clf.intercept_[0]
    xs = np.linspace(*xlim, 50)
    for off, ls, c in [(0, "-", color or p.fg), (1, "--", p.muted), (-1, "--", p.muted)]:
        if off and not band:
            continue
        ax.plot(xs, (off - b - w[0] * xs) / w[1], ls=ls, color=c, lw=lw if off == 0 else 0.9)


def _style(ax, p, lim=3.2, ylim=None):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(*(ylim or (-lim, lim)))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(p.faint)


@register("fund.svm-margin-dual", "margin")
def margin(p):
    """Separable data, the max-margin boundary, the margin band, circled support vectors and the width 2/||w||."""
    X, y = _sep_data()
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    fig, ax = figure(2.6)
    ax.plot(*X[y > 0].T, "o", ms=4, color=p.c(2))
    ax.plot(*X[y < 0].T, "s", ms=3.6, color=p.c(0))
    sv = clf.support_vectors_
    ax.plot(sv[:, 0], sv[:, 1], "o", ms=9, mfc="none", mec=p.label, mew=1.3)
    _lines(ax, clf, p, (-3.2, 3.2))
    # width arrow between the two margin lines, along w, through the point nearest the origin on the boundary
    u = w / np.linalg.norm(w)
    c0 = -b * w / (w @ w) + np.array([-1.3, 1.3]) * 0.9 / np.sqrt(2)
    a, e = c0 - u / np.linalg.norm(w), c0 + u / np.linalg.norm(w)
    ax.annotate("", xy=e, xytext=a, arrowprops=dict(arrowstyle="<|-|>", color=p.label, lw=1.1,
                                                     mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(*(c0 + np.array([-0.15, 0.22])), r"$2/\|w\|$", color=p.label, fontsize=8.5, ha="right")
    ax.text(-3.1, -2.8, r"solid: $w^\top x+b=0$", color=p.fg, fontsize=7.5)
    ax.text(-3.1, -3.17, r"dashed: $w^\top x+b=\pm 1$", color=p.muted, fontsize=7.5)
    ax.text(-3.1, -3.55, "circled: support vectors", color=p.label, fontsize=7.5)
    _style(ax, p, ylim=(-3.75, 3.2))
    return fig


def _outlier_data():
    X, y = _sep_data()
    X = np.r_[X, [[-0.6, 0.1]]]          # a +1 point sitting deep among the -1 side's margin
    y = np.r_[y, 1.0]
    return X, y


@register("fund.svm-margin-dual", "soft-margin-c")
def soft_margin_c(p):
    """[fig-Q] The separable data plus one outlier, fitted at three values of C (panels deliberately unlabeled by C)."""
    X, y = _outlier_data()
    fig, axes = figure(1.4, ncols=3)
    # panel order is shuffled on purpose: A = 1, B = 100, C = 0.01
    for ax, C, name in zip(axes, [1.0, 100.0, 0.01], "ABC"):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        ax.plot(*X[y > 0].T, "o", ms=2.6, color=p.c(2))
        ax.plot(*X[y < 0].T, "s", ms=2.4, color=p.c(0))
        ax.plot([X[-1, 0]], [X[-1, 1]], "o", ms=3.6, color=p.c(2), mec=p.fg, mew=0.6)
        _lines(ax, clf, p, (-3.2, 3.2), lw=1.3)
        _style(ax, p)
        ax.set_title(name, fontsize=9)
    return fig


@register("fund.svm-margin-dual", "alpha-types")
def alpha_types(p):
    """Overlapping classes at C = 1: points coloured by alpha = 0, 0 < alpha < C and alpha = C."""
    rng = np.random.default_rng(8)
    n = 30
    X = np.r_[rng.normal([0.9, 0.8], 0.85, (n, 2)), rng.normal([-0.9, -0.8], 0.85, (n, 2))]
    y = np.r_[np.ones(n), -np.ones(n)]
    C = 1.0
    clf = SVC(kernel="linear", C=C).fit(X, y)
    alpha = np.zeros(len(y))
    alpha[clf.support_] = np.abs(clf.dual_coef_[0])
    tol = 1e-6 * C
    zero = alpha < tol
    box = alpha > C - 1e-3
    free = ~zero & ~box
    fig, ax = figure(2.7)
    mk = np.where(y > 0, "o", "s")
    groups = [(zero, p.muted, r"$\alpha=0$: outside margin"), (free, p.c(2), r"$0<\alpha<C$: on margin"),
              (box, p.label, r"$\alpha=C$: inside / wrong side")]
    for mask, col, lab in groups:
        first = True
        for m in ("o", "s"):
            sel = mask & (mk == m)
            ax.plot(*X[sel].T, m, ms=4 if m == "o" else 3.6, color=col, alpha=0.95,
                    label=lab if first else None, ls="none")
            first = False
    _lines(ax, clf, p, (-3.2, 3.2))
    _style(ax, p)
    ax.text(3.0, 2.95, "circles: y = +1\nsquares: y = −1", color=p.muted, fontsize=7, ha="right", va="top")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1, fontsize=7)
    return fig


@register("fund.svm-margin-dual", "hinge-vs-logistic")
def hinge_vs_logistic(p):
    """0-1, hinge and logistic losses as functions of the margin m = y f(x). Logistic is in bits so it passes (0, 1)."""
    m = np.linspace(-2.5, 3, 400)
    fig, ax = figure(2.3)
    ax.plot(m, (m <= 0).astype(float), color=p.muted, lw=1.2, drawstyle="steps-mid", label="0–1")
    ax.plot(m, np.maximum(0, 1 - m), color=p.c(0), label=r"hinge $[1-m]_+$")
    ax.plot(m, np.logaddexp(0, -m) / np.log(2), color=p.c(1), label=r"logistic $\log_2(1+e^{-m})$")
    ax.axvline(1, color=p.faint, lw=0.9, zorder=0)
    ax.text(1.06, 1.55, "m = 1", color=p.muted, fontsize=7.5)
    ax.set_xlabel(r"margin $m = y\,f(x)$")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 3.4)
    ax.set_xlim(-2.5, 3)
    ax.legend(loc="upper right", handlelength=1.4)
    return fig
