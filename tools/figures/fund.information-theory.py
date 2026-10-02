"""Figures for fund.information-theory. All values are computed from the formulas in the lesson."""
import numpy as np
from matplotlib.patches import Circle

from figures.style import blank, figure, register


def _h2(t):
    t = np.clip(t, 1e-12, 1 - 1e-12)
    return -(t * np.log2(t) + (1 - t) * np.log2(1 - t))


@register("fund.information-theory", "binary-entropy")
def binary_entropy(p):
    """H(theta) in bits for a Bernoulli(theta) variable."""
    t = np.linspace(0, 1, 401)
    fig, ax = figure(2.3)
    ax.plot(t, _h2(t), color=p.accent, lw=2)
    ax.plot([0.5], [1.0], "o", color=p.label, ms=5, zorder=5)
    ax.annotate(r"max 1 bit at $\theta = 0.5$", (0.5, 1.0), xytext=(0.5, 1.13), ha="center",
                fontsize=8, color=p.label)
    h = _h2(0.1)
    ax.plot([0.1, 0.9], [h, h], "o", color=p.c(2), ms=4.5, zorder=5)
    ax.plot([0.1, 0.9], [h, h], color=p.c(2), lw=0.8, ls=":")
    ax.annotate(f"H(0.1) = H(0.9) = {h:.3f} bits", (0.1, h), xytext=(0.17, 0.25),
                fontsize=7.5, color=p.c(2),
                arrowprops=dict(arrowstyle="-", color=p.c(2), lw=0.8))
    ax.set_xlabel(r"$\theta = P(X=1)$")
    ax.set_ylabel(r"$H(X)$ in bits")
    ax.set_ylim(0, 1.25)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.5, 1])
    return fig


def _venn(p, labels, values=None):
    fig, ax = figure(2.25)
    blank(ax, (-3.3, 3.3), (-1.75, 1.75))
    r, d = 1.45, 0.75
    for cx, col in [(-d, p.c(0)), (d, p.c(1))]:
        ax.add_patch(Circle((cx, 0), r, facecolor=col, alpha=0.18, edgecolor=col, lw=1.6))
    ax.text(-d - 0.55, 1.5, "H(X)" if values else "X", color=p.c(0), fontsize=9, ha="right")
    ax.text(d + 0.55, 1.5, "H(Y)" if values else "Y", color=p.c(1), fontsize=9, ha="left")
    xs = [-1.45, 0, 1.45]
    for x, lab in zip(xs, labels):
        ax.text(x, 0.08 if values else 0, lab, ha="center", va="center", fontsize=8.5 if values else 11,
                color=p.fg)
    if values:
        for x, v in zip(xs, values):
            ax.text(x, -0.32, v, ha="center", va="center", fontsize=7.5, color=p.muted)
        ax.text(0, -1.68, "union of both circles = H(X,Y)", ha="center", fontsize=7.5, color=p.muted)
    return fig


@register("fund.information-theory", "entropy-venn")
def entropy_venn(p):
    """Information diagram with the bit values of the 2x2 worked example."""
    i = 1 - _h2(0.2)
    c = _h2(0.2)
    return _venn(p, ["H(X|Y)", "I(X;Y)", "H(Y|X)"],
                 [f"{c:.3f}", f"{i:.3f}", f"{c:.3f}"])


@register("fund.information-theory", "venn-quiz")
def venn_quiz(p):
    """Unlabelled information diagram used as a question stem."""
    return _venn(p, ["A", "B", "C"])


@register("fund.information-theory", "ce-decomposition")
def ce_decomposition(p):
    """H(p,q) = H(p) + KL(p||q) for a fixed p and four model distributions q (bits)."""
    P = np.array([0.6, 0.3, 0.1])
    qs = [("q = p", P),
          ("(.5,.3,.2)", np.array([0.5, 0.3, 0.2])),
          ("uniform", np.ones(3) / 3),
          ("(.1,.3,.6)", np.array([0.1, 0.3, 0.6]))]
    H = -(P * np.log2(P)).sum()
    fig, ax = figure(2.45)
    for k, (name, q) in enumerate(qs):
        kl = (P * np.log2(P / q)).sum()
        ax.bar(k, H, color=p.c(0), width=0.6, alpha=0.85)
        ax.bar(k, kl, bottom=H, color=p.c(1), width=0.6, alpha=0.9)
        ax.text(k, H + kl + 0.05, f"{H + kl:.2f}", ha="center", fontsize=7.5, color=p.fg)
    ax.text(0, H / 2, f"H(p)\n{H:.2f}", fontsize=7.5, color=p.surface, va="center", ha="center")
    ax.text(3.0, H + 0.5, r"$\mathrm{KL}(p\|q)$", fontsize=8, color=p.surface, ha="center")
    ax.set_xticks(range(4))
    ax.set_xticklabels([n for n, _ in qs], fontsize=7.5)
    ax.set_ylabel("cross-entropy H(p,q), bits")
    ax.set_xlabel("model q for p = (0.6, 0.3, 0.1)")
    ax.set_ylim(0, 3.1)
    ax.set_yticks([0, 1, 2, 3])
    return fig
