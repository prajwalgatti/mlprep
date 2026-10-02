"""Figures for gen.latent-variables-elbo. All curves are computed exactly from the toy models in the lesson."""
import numpy as np
from matplotlib.patches import Ellipse

from figures.style import figure, register

L2P = np.log(2 * np.pi)


def _elbo(m, s2, x=1.0):
    """ELBO of q=N(m,s2) for z~N(0,1), x|z~N(z,1)."""
    return -0.5 * L2P + 0.5 + 0.5 * np.log(s2) - 0.5 * (m * m + (x - m) ** 2) - s2


@register("gen.latent-variables-elbo", "jensen-chord")
def jensen_chord(p):
    """Two-point X in {1, 4}: E[log X] (chord midpoint) sits below log E[X] (curve)."""
    xs = np.linspace(0.45, 5.2, 300)
    a, b = 1.0, 4.0
    m = (a + b) / 2
    fig, ax = figure(2.45)
    ax.plot(xs, np.log(xs), color=p.accent, lw=2)
    tx = np.linspace(1.6, 5.2, 50)
    ax.plot(tx, np.log(m) + (tx - m) / m, color=p.muted, lw=0.9, ls="--")
    ax.plot([a, b], [np.log(a), np.log(b)], color=p.c(1), lw=1.4)
    ax.plot([a, b], [np.log(a), np.log(b)], "o", color=p.c(1), ms=4.5)
    ax.plot([m], [np.log(m)], "o", color=p.accent, ms=5, zorder=5)
    ax.plot([m], [(np.log(a) + np.log(b)) / 2], "o", color=p.c(1), ms=5, zorder=5)
    ax.plot([m, m], [(np.log(a) + np.log(b)) / 2, np.log(m)], color=p.label, lw=1.2, ls=":", zorder=4)
    ax.text(0.6, 2.12, r"$\log\mathbb{E}[X]=0.916$  (on the curve)", color=p.accent, fontsize=7.6, va="top")
    ax.text(0.6, 1.86, r"$\mathbb{E}[\log X]=0.693$  (chord midpoint)", color=p.c(1), fontsize=7.6, va="top")
    ax.text(0.6, 1.60, "Jensen gap = 0.223", color=p.label, fontsize=7.6, va="top")
    ax.text(5.15, 1.88, "tangent at\n" + r"$\mathbb{E}[X]$", color=p.muted, fontsize=7, ha="right", va="bottom")
    ax.text(4.95, np.log(4.95) - 0.12, r"$\log x$", color=p.accent, fontsize=8, ha="right", va="top")
    ax.set_xticks([a, m, b])
    ax.set_xticklabels(["1", "2.5", "4"])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xlim(0.45, 5.2)
    ax.set_ylim(-0.6, 2.25)
    ax.set_xlabel(r"$x$   ($X=1$ or $4$, each w.p. $\frac{1}{2}$)")
    return fig


@register("gen.latent-variables-elbo", "elbo-gap")
def elbo_gap(p):
    """log p(x) = ELBO + KL(q || posterior) for four Gaussian q's in the toy model (x = 1)."""
    lp = -0.5 * np.log(4 * np.pi) - 0.25
    qs = [(0.0, 1.0, "prior\n$N(0,1)$"), (0.5, 1.0, "$N(0.5,1)$"), (1.0, 0.5, "$N(1,0.5)$"),
          (0.5, 0.5, "posterior\n$N(0.5,0.5)$")]
    base = -2.2
    fig, ax = figure(2.55)
    for i, (m, s2, lab) in enumerate(qs):
        e = _elbo(m, s2)
        ax.bar(i, e - base, bottom=base, width=0.58, color=p.c(0), alpha=0.85, lw=0)
        gap = lp - e
        if gap > 1e-6:
            ax.bar(i, gap, bottom=e, width=0.58, color=p.c(1), alpha=0.9, lw=0)
            ax.text(i, e + gap / 2, f"{gap:.3f}", ha="center", va="center", fontsize=7.3, color=p.fg)
        else:
            ax.text(i, lp + 0.03, "gap 0", ha="center", va="bottom", fontsize=7.3, color=p.label)
        ax.text(i, e - 0.05, f"{e:.3f}", ha="center", va="top", fontsize=7.3, color=p.fg)
    ax.axhline(lp, color=p.fg, lw=1, ls="--")
    ax.text(1.5, lp + 0.03, r"$\log p(x)=-1.516$", ha="center", va="bottom", fontsize=7.6, color=p.fg)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=p.c(0), alpha=0.85, label="ELBO"),
                       Patch(color=p.c(1), alpha=0.9, label="gap = KL(q || posterior)")],
              loc="upper left", ncol=2, handlelength=1.0, columnspacing=1.2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([q[2] for q in qs], fontsize=7)
    ax.set_ylim(base, -1.12)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("nats")
    return fig


@register("gen.latent-variables-elbo", "em-bound")
def em_bound(p):
    """EM on z~N(theta,1), x|z~N(z,1), x=2: each bound touches log p at theta_old; its max is the next theta."""
    x = 2.0
    th = np.linspace(-3.2, 3.6, 400)
    ell = -0.5 * np.log(4 * np.pi) - (x - th) ** 2 / 4
    fig, ax = figure(2.6)
    ax.plot(th, ell, color=p.fg, lw=2.1)
    ax.text(3.55, -0.7, r"$\ell(\theta)=\log p_\theta(x)$", color=p.fg, fontsize=7.8, ha="right", va="top")
    for k, (told, col) in enumerate([(-2.0, p.c(0)), (0.0, p.c(2))]):
        bnd = -0.5 * np.log(4 * np.pi) - (x - th) ** 2 / 4 - (th - told) ** 2 / 4
        ax.plot(th, bnd, color=col, lw=1.4)
        tnew = (told + x) / 2
        ax.plot([told], [-0.5 * np.log(4 * np.pi) - (x - told) ** 2 / 4], "o", color=col, ms=5, zorder=5)
        bmax = -0.5 * np.log(4 * np.pi) - (x - tnew) ** 2 / 4 - (tnew - told) ** 2 / 4
        ax.plot([tnew], [bmax], "s", color=col, ms=4.5, zorder=5)
        top = -0.5 * np.log(4 * np.pi) - (x - tnew) ** 2 / 4
        if top - bmax > 0.3:   # skip arrows too short to render cleanly
            ax.annotate("", (tnew, top - 0.05), (tnew, bmax + 0.05),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.text(-2.05, -6.3, r"bound with $q$ from $\theta=-2$", color=p.c(0), fontsize=7.2, ha="left")
    ax.text(0.05, -6.85, r"bound with $q$ from $\theta=0$", color=p.c(2), fontsize=7.2, ha="left")
    ax.plot([-2.95], [-1.45], "o", color=p.muted, ms=4.5)
    ax.plot([-2.95], [-1.95], "s", color=p.muted, ms=4)
    ax.text(-2.75, -1.45, "E-step: bound touches $\\ell$", color=p.muted, fontsize=7, ha="left", va="center")
    ax.text(-2.75, -1.95, "M-step: maximise bound", color=p.muted, fontsize=7, ha="left", va="center")
    ax.set_xlim(-3.2, 3.6)
    ax.set_ylim(-7.2, -0.6)
    ax.set_xticks([-2, 0, 1, 2])
    ax.set_xlabel(r"model parameter $\theta$")
    ax.set_ylabel("nats")
    return fig


@register("gen.latent-variables-elbo", "meanfield")
def meanfield(p):
    """Correlated Gaussian posterior (rho=0.9) vs factorised fits: reverse KL (too narrow) and forward KL."""
    rho = 0.9
    fig, ax = figure(2.75)
    ang = 45
    lam1, lam2 = 1 + rho, 1 - rho
    for k in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(lam1), 2 * k * np.sqrt(lam2), angle=ang,
                             fill=False, ec=p.fg, lw=1.6 if k == 1 else 0.9))
    s_rev = np.sqrt(1 - rho ** 2)
    for k in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * k * s_rev, 2 * k * s_rev, fill=False, ec=p.c(1),
                             lw=1.6 if k == 1 else 0.9))
        ax.add_patch(Ellipse((0, 0), 2 * k, 2 * k, fill=False, ec=p.c(2), ls="--",
                             lw=1.3 if k == 1 else 0.8))
    ax.text(1.75, 2.45, r"true posterior, $\rho=0.9$", color=p.fg, fontsize=7.4, ha="left")
    ax.annotate("", (1.12, 1.22), (1.7, 2.4), arrowprops=dict(arrowstyle="-", color=p.fg, lw=0.6))
    ax.text(-3.3, 2.75, "KL$(q\\|p)$ fit\nsd 0.44", color=p.c(1), fontsize=7.4, ha="left", va="top")
    ax.annotate("", (-0.31, 0.31), (-2.3, 2.1), arrowprops=dict(arrowstyle="-", color=p.c(1), lw=0.6))
    ax.text(2.1, -1.55, "KL$(p\\|q)$ fit\nsd 1.0", color=p.c(2), fontsize=7.4, ha="left", va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-3.4, 3.9)
    ax.set_ylim(-2.9, 2.9)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    return fig
