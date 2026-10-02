"""Figures for fund.adam. Exact recursions of the Adam update; the race uses hand-picked step sizes."""
import numpy as np

from figures.style import figure, register

B1, B2 = 0.9, 0.999


@register("fund.adam", "bias-correction")
def bias_correction(p):
    """Top: raw vs bias-corrected m_t and sqrt(v_t) for a constant gradient g=1.
    Bottom: the uncorrected step m_t/sqrt(v_t) relative to the corrected one."""
    fig, (ax1, ax2) = figure(3.6, nrows=2)
    t = np.arange(1, 51)
    m = 1 - B1 ** t
    v = 1 - B2 ** t
    ax1.plot(t, m, color=p.c(0), label=r"raw $m_t$")
    ax1.plot(t, np.sqrt(v), color=p.c(1), label=r"raw $\sqrt{v_t}$")
    ax1.axhline(1, color=p.fg, lw=1.2, ls="--", label=r"corrected $\hat m_t$, $\sqrt{\hat v_t}$")
    ax1.set_ylim(0, 1.15)
    ax1.set_xlim(1, 50)
    ax1.set_xlabel("step $t$", fontsize=8)
    ax1.set_ylabel("estimate of $|g|$", fontsize=8)
    ax1.legend(loc="center right", fontsize=7, handlelength=1.3)
    ax1.set_title(r"constant gradient $g=1$, $\beta_1=0.9$, $\beta_2=0.999$", fontsize=8)

    T = np.arange(1, 10001)
    r = (1 - B1 ** T) / np.sqrt(1 - B2 ** T)
    ax2.plot(T, r, color=p.accent)
    ax2.axhline(1, color=p.muted, lw=0.8, ls="--")
    i = r.argmax()
    ax2.plot([T[i]], [r[i]], "o", color=p.label, ms=4)
    ax2.annotate(f"peak {r[i]:.1f}× at t={T[i]}", (T[i], r[i]), xytext=(60, 5.6), fontsize=7.5,
                 color=p.label, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax2.text(1.3, 2.2, "3.16× at t=1", fontsize=7.2, color=p.muted)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 10000)
    ax2.set_ylim(0, 7.5)
    ax2.set_xlabel("step $t$ (log scale)", fontsize=8)
    ax2.set_ylabel("uncorrected / corrected", fontsize=8)
    return fig


def _run(kind, lr, steps, x0, lams):
    x = np.array(x0, float)
    m = np.zeros(2)
    v = np.zeros(2)
    xs = [x.copy()]
    for t in range(1, steps + 1):
        g = lams * x
        if kind == "sgd":
            x = x - lr * g
        elif kind == "mom":
            m = 0.9 * m + g
            x = x - lr * m
        elif kind == "rms":
            v = 0.9 * v + 0.1 * g ** 2
            x = x - lr * g / (np.sqrt(v) + 1e-8)
        else:
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g ** 2
            mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
            x = x - lr * mh / (np.sqrt(vh) + 1e-8)
        xs.append(x.copy())
    return np.array(xs)


@register("fund.adam", "optimizer-race")
def optimizer_race(p):
    """Four optimizers, 50 deterministic steps on f = (x^2 + 100 y^2)/2 from (-3, 0.25)."""
    lams = np.array([1.0, 100.0])
    x0 = (-3.0, 0.25)
    runs = [("sgd", 0.019, r"GD, $\eta=0.019$", p.c(1)),
            ("mom", 0.004, r"momentum 0.9, $\eta=0.004$", p.c(4)),
            ("rms", 0.06, r"RMSProp, $\eta=0.06$", p.c(2)),
            ("adam", 0.12, r"Adam, $\eta=0.12$", p.c(0))]
    fig, axes = figure(3.4, nrows=4, sharex=True)
    gx, gy = np.meshgrid(np.linspace(-3.4, 1.0, 300), np.linspace(-0.4, 0.4, 200))
    f = 0.5 * (gx ** 2 + 100 * gy ** 2)
    for ax, (kind, lr, name, col) in zip(axes, runs):
        xs = _run(kind, lr, 50, x0, lams)
        ax.contour(gx, gy, f, levels=[0.1, 0.5, 1.5, 3, 4.5], colors=p.faint, linewidths=0.7)
        ax.plot(xs[:, 0], xs[:, 1], "-o", ms=1.6, lw=0.9, color=col)
        ax.plot([0], [0], "*", color=p.label, ms=6, zorder=5)
        ax.set_xlim(-3.4, 1.0)
        ax.set_ylim(-0.4, 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.98, 0.78, name, transform=ax.transAxes, ha="right", fontsize=7.3, color=col)
        d = np.linalg.norm(xs[-1])
        ax.text(0.98, 0.06, f"distance after 50: {d:.2g}", transform=ax.transAxes, ha="right",
                fontsize=6.8, color=p.muted)
    return fig


def _l2_vs_wd(p, labels):
    s = np.logspace(-4, -1, 200)
    eta, lam_l2, wd = 1e-3, 1e-3, 0.1
    l2 = eta * lam_l2 / (s + 1e-8)
    adamw = np.full_like(s, eta * wd)
    fig, ax = figure(2.4)
    ax.plot(s, l2, color=p.c(0), label=labels[0])
    ax.plot(s, adamw, color=p.c(1), label=labels[1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"weight's gradient RMS $\sqrt{\hat v}$")
    ax.set_ylabel("shrink per step")
    ax.set_ylim(1e-6, 2e-2)
    return fig, ax


@register("fund.adam", "l2-vs-wd")
def l2_vs_wd(p):
    """Per-step shrinkage from the regularizer: Adam+L2 (coef 1e-3) vs AdamW (lr 1e-3, wd 0.1)."""
    fig, ax = _l2_vs_wd(p, ["Adam + L2: $\\eta\\lambda'/\\sqrt{\\hat v}$", "AdamW: $\\eta\\lambda$ (flat)"])
    ax.legend(loc="upper right", fontsize=7.3, handlelength=1.4)
    return fig


@register("fund.adam", "l2-vs-wd-quiz")
def l2_vs_wd_quiz(p):
    """[fig-Q] Same two curves, unlabelled except A/B."""
    fig, ax = _l2_vs_wd(p, ["A", "B"])
    ax.text(2e-4, 1.0e-3, "A", color=p.c(0), fontsize=9, fontweight="bold")
    ax.text(2e-4, 1.4e-4, "B", color=p.c(1), fontsize=9, fontweight="bold")
    return fig
