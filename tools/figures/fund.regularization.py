"""Figures for fund.regularization: constraint geometry, shrinkage operators, eigen-direction shrinkage."""
import numpy as np

from figures.style import figure, register

# A quadratic loss 0.5 (w - w*)^T H (w - w*) with correlated features.
W_STAR = np.array([2.0, 0.9])
H = np.array([[1.0, -0.3], [-0.3, 0.8]])


def _loss(w):
    d = w - W_STAR
    return 0.5 * np.einsum("...i,ij,...j->...", d, H, d)


def _best_on_boundary(points):
    """Lowest-loss point on a sampled constraint boundary (the constrained optimum when w* is outside)."""
    return points[np.argmin(_loss(points))]


@register("fund.regularization", "l1-l2-geometry")
def l1_l2_geometry(p):
    """Loss contours first touch the L2 disk off-axis and the L1 diamond at a corner."""
    t = np.linspace(0, 2 * np.pi, 4000)
    r = 1.0
    disk = np.c_[r * np.cos(t), r * np.sin(t)]
    diamond = r * np.c_[np.cos(t), np.sin(t)]
    diamond /= np.abs(diamond).sum(axis=1, keepdims=True)    # rescale each ray onto |w1| + |w2| = r
    g = np.linspace(-1.6, 3.6, 300)
    G = np.stack(np.meshgrid(g, g), axis=-1)
    Z = _loss(G)
    fig, axes = figure(1.95, ncols=2)
    for ax, shape, title in zip(axes, [disk, diamond], [r"L2: $\|w\|_2\leq t$", r"L1: $\|w\|_1\leq t$"]):
        w_hat = _best_on_boundary(shape)
        level = _loss(w_hat)
        ax.contour(g, g, Z, levels=[level * f for f in (0.25, 0.55, 1.0)], colors=[p.muted, p.muted, p.c(1)],
                   linewidths=[0.7, 0.7, 1.4])
        ax.fill(shape[:, 0], shape[:, 1], color=p.accent, alpha=0.18, lw=0)
        ax.plot(shape[:, 0], shape[:, 1], color=p.accent, lw=1.4)
        ax.axhline(0, color=p.faint, lw=0.8, zorder=0)
        ax.axvline(0, color=p.faint, lw=0.8, zorder=0)
        ax.plot(*W_STAR, "o", color=p.fg, ms=4)
        ax.annotate(r"$w^*$", W_STAR, xytext=(4, 3), textcoords="offset points", fontsize=8, color=p.fg)
        ax.plot(*w_hat, "o", color=p.label, ms=5, zorder=5)
        ax.annotate(r"$\hat w$", w_hat, xytext=(-13, 5), textcoords="offset points", fontsize=8.5,
                    color=p.label)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(-1.4, 3.3)
        ax.set_ylim(-1.5, 2.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$w_1$", labelpad=1)
    axes[0].set_ylabel(r"$w_2$", labelpad=1)
    return fig


def _shrinkage(p, labels):
    """Orthonormal-design estimators as functions of the unpenalized estimate w* (lambda = 1).

    `labels` maps each operator to its legend text, in legend order.
    """
    w = np.linspace(-3, 3, 601)
    curves = {
        "L1": np.sign(w) * np.maximum(np.abs(w) - 1, 0),
        "L2": w / 2,
        "L0": np.where(np.abs(w) >= np.sqrt(2), w, 0.0),
    }
    styles = {"L1": (p.c(0), "-"), "L2": (p.c(1), "--"), "L0": (p.c(2), ":")}
    fig, ax = figure(2.35)
    ax.plot(w, w, color=p.faint, lw=1.2)
    ax.text(2.9, 2.45, r"$\hat w=w^*$", color=p.muted, fontsize=7.5, ha="right")
    ax.axhline(0, color=p.faint, lw=0.7)
    ax.axvline(0, color=p.faint, lw=0.7)
    for key, name in labels.items():
        color, ls = styles[key]
        y = curves[key]
        pieces = [w <= -np.sqrt(2), np.abs(w) < np.sqrt(2), w >= np.sqrt(2)] if key == "L0" else [w == w]
        for i, sel in enumerate(pieces):   # draw the L0 jump as a gap, not a vertical line
            ax.plot(w[sel], y[sel], color=color, ls=ls, lw=1.9, label=name if i == 0 else None)
    ax.legend(loc="upper left", handlelength=2.2)
    ax.set_xlabel(r"unpenalized estimate $w^*$")
    ax.set_ylabel(r"penalized estimate $\hat w$")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-2, 0, 2])
    return fig


@register("fund.regularization", "shrinkage-curves")
def shrinkage_curves(p):
    return _shrinkage(p, {"L1": "L1: soft threshold", "L2": "L2: scale by 1/2", "L0": "L0: hard threshold"})


@register("fund.regularization", "shrinkage-abc")
def shrinkage_abc(p):
    """[fig-Q] The same three operators with neutral labels."""
    return _shrinkage(p, {"L0": "A", "L2": "B", "L1": "C"})


@register("fund.regularization", "eigen-shrink")
def eigen_shrink(p):
    """L2 keeps h_i/(h_i + lambda) of w* along each Hessian eigenvector (lambda = 1)."""
    h = np.array([100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01])
    lam = 1.0
    keep = h / (h + lam)
    fig, ax = figure(2.3)
    x = np.arange(h.size)
    colors = [p.c(0) if k >= 0.5 else p.c(1) for k in keep]
    ax.bar(x, keep, color=colors, width=0.7)
    for xi, k in zip(x, keep):
        ax.text(xi, k + 0.03, f"{k:.2f}", ha="center", fontsize=6.8, color=p.fg)
    ax.axhline(0.5, color=p.muted, lw=0.8, ls=":")
    ax.text(8.45, 0.53, r"$h_i=\lambda$", color=p.muted, fontsize=7.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in h])
    ax.set_xlabel(r"Hessian eigenvalue $h_i$ (curvature), $\lambda=1$")
    ax.set_ylabel(r"fraction kept $\frac{h_i}{h_i+\lambda}$")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    return fig
