"""Figures for llm.positional-encodings: sinusoidal PE computed from the Vaswani formula."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figures.style import blank, figure, register

TOPIC = "llm.positional-encodings"


def _pe(n_pos, d, base=10000.0):
    """Interleaved sinusoidal PE: column 2i = sin(p w_i), column 2i+1 = cos(p w_i)."""
    pos = np.arange(n_pos)[:, None]
    w = base ** (-2 * np.arange(d // 2) / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(pos * w)
    pe[:, 1::2] = np.cos(pos * w)
    return pe, w


def _cmap(p):
    return LinearSegmentedColormap.from_list("div", [p.c(0), p.surface, p.c(1)])


@register(TOPIC, "heatmap")
def heatmap(p):
    """PE matrix for d=64, positions 0..99: fast columns on the left, slow on the right."""
    pe, _ = _pe(100, 64)
    fig, ax = figure(2.7)
    im = ax.imshow(pe, aspect="auto", cmap=_cmap(p), vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("dimension (pair i = column // 2)")
    ax.set_ylabel("position p")
    ax.set_xticks([0, 16, 32, 48, 63])
    ax.set_yticks([0, 25, 50, 75, 99])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    ax.text(3, -3, "fast", color=p.label, fontsize=7.5, ha="left", va="bottom")
    ax.text(60, -3, "slow", color=p.label, fontsize=7.5, ha="right", va="bottom")
    return fig


@register(TOPIC, "clock")
def clock(p):
    """One (sin, cos) pair as a point on the unit circle, for a fast and a slow frequency."""
    fig, axes = figure(2.0, ncols=2)
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, w, title in zip(axes, [1.0, 0.1], [r"fast pair, $\omega=1$", r"slow pair, $\omega=0.1$"]):
        blank(ax, (-1.45, 1.45), (-1.35, 1.35))
        ax.plot(np.cos(t), np.sin(t), color=p.faint, lw=1)
        for pos in range(7):
            # point = (cos, sin) drawn as x = PE_{2i+1} = cos, y = PE_{2i} = sin
            x, y = np.cos(pos * w), np.sin(pos * w)
            ax.plot([0, x], [0, y], color=p.muted if pos else p.accent, lw=0.6)
            ax.plot([x], [y], "o", color=p.c(0) if pos else p.label, ms=4)
            r = 1.18
            ax.text(r * x, r * y, str(pos), fontsize=7, ha="center", va="center", color=p.fg)
        ax.set_title(title, fontsize=8.5)
    axes[0].text(0, -1.33, r"$x=\cos(p\omega),\ y=\sin(p\omega)$", fontsize=7, ha="center", color=p.muted)
    axes[1].text(0, -1.33, r"each step: $+\omega$ radians", fontsize=7, ha="center", color=p.muted)
    return fig


@register(TOPIC, "dot-offset")
def dot_offset(p):
    """PE_p · PE_{p+δ} / (d/2) = mean_i cos(δ ω_i), d = 128: depends only on δ, symmetric."""
    d = 128
    w = 10000.0 ** (-2 * np.arange(d // 2) / d)
    delta = np.arange(-200, 201)
    s = np.cos(delta[:, None] * w).mean(axis=1)
    fig, ax = figure(2.2)
    ax.axhline(0, color=p.faint, lw=0.8)
    ax.plot(delta, s, color=p.accent, lw=1.4)
    ax.set_xlabel(r"offset $\delta$")
    ax.set_ylabel(r"$PE_p\cdot PE_{p+\delta}\;/\;(d/2)$")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-0.1, 1.05)
    ax.annotate(r"1 at $\delta=0$", (0, 1), xytext=(45, 0.93), fontsize=7.5, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.text(-195, 0.75, r"same value at $\pm\delta$", fontsize=7.5, color=p.muted)
    ax.text(195, 0.75, "d = 128", fontsize=7.5, color=p.muted, ha="right")
    return fig


@register(TOPIC, "toeplitz")
def toeplitz(p):
    """Position–position term p_i^T A p_j for two choices of A (d = 16, 48 positions)."""
    d, n = 16, 48
    pe, _ = _pe(n, d)
    rng = np.random.default_rng(3)
    a_rot = np.zeros((d, d))
    for i in range(d // 2):
        a, b = rng.normal(size=2)
        a_rot[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[a, b], [-b, a]]
    a_gen = rng.normal(size=(d, d))
    fig, axes = figure(2.1, ncols=2)
    for ax, A, title in zip(axes, [a_rot, a_gen],
                            ["A: blocks [[a, b], [−b, a]]", "A: generic matrix"]):
        m = pe @ A @ pe.T
        v = np.abs(m).max()
        ax.imshow(m, cmap=_cmap(p), vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("key position j", fontsize=7.5)
    axes[0].set_ylabel("query position i", fontsize=7.5)
    return fig


@register(TOPIC, "period-q")
def period_q(p):
    """[fig-Q] One PE coordinate sin(p ω_i) for d = 128 and i = 16 (wavelength 2π·10 ≈ 62.8)."""
    d, i = 128, 16
    w = 10000.0 ** (-2 * i / d)
    pos = np.arange(0, 201)
    fig, ax = figure(1.9)
    ax.axhline(0, color=p.faint, lw=0.8)
    ax.plot(pos, np.sin(pos * w), color=p.accent, lw=1.5)
    ax.plot(pos[::4], np.sin(pos[::4] * w), "o", color=p.accent, ms=1.8)
    ax.set_xlabel("position p")
    ax.set_ylabel(r"$PE_{p,\,2i}$")
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_xlim(0, 200)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, axis="x")
    return fig
