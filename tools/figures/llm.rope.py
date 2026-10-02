"""Figures for llm.rope, computed from the RoPE formulas (theta_i = b^(-2i/d))."""
import numpy as np

from figures.style import blank, box, figure, register

TOPIC = "llm.rope"


def _theta(d, base):
    return base ** (-2 * np.arange(d // 2) / d)


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@register(TOPIC, "rotation-2d")
def rotation_2d(p):
    """q rotated by mθ and k by nθ: the angle between them depends only on n − m."""
    th = 0.3
    q0 = np.array([1.0, 0.25])
    k0 = np.array([0.55, 0.85])
    fig, axes = figure(1.9, ncols=2)
    for ax, (m, n) in zip(axes, [(2, 5), (7, 10)]):
        blank(ax, (-1.35, 1.35), (-1.3, 1.3))
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color=p.faint, lw=0.8)
        q = _rot(m * th) @ q0
        k = _rot(n * th) @ k0
        for v, c, lab in [(q, p.c(0), f"$R_{{{m}}}q$"), (k, p.c(1), f"$R_{{{n}}}k$")]:
            ax.annotate("", xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, mutation_scale=9))
            ax.text(1.17 * v[0], 1.17 * v[1], lab, color=c, fontsize=8.5, ha="center", va="center")
        a0 = np.arctan2(q[1], q[0])
        diff = np.arctan2(k[1], k[0]) - a0
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        a1 = a0 + diff
        arc = np.linspace(a0, a1, 40)
        ax.plot(0.32 * np.cos(arc), 0.32 * np.sin(arc), color=p.label, lw=1.2)
        mid = (a0 + a1) / 2
        ax.text(0.5 * np.cos(mid), 0.5 * np.sin(mid), f"{np.degrees(a1 - a0):.0f}°",
                color=p.label, fontsize=7.5, ha="center", va="center")
        ax.set_title(f"m = {m}, n = {n}", fontsize=8.5)
    fig.supxlabel(r"$\theta=0.3$ rad/position; both pairs are 3 positions apart", fontsize=7, color=p.muted)
    return fig


@register(TOPIC, "wavelengths")
def wavelengths(p):
    """Wavelength of each of the 64 pairs (d_head = 128) for two bases vs training contexts."""
    d = 128
    i = np.arange(d // 2)
    fig, ax = figure(2.5)
    for base, c, lab, y in [(1e4, p.c(0), r"$b=10^4$", 1.2e3), (5e5, p.c(1), r"$b=5\times10^5$", 2.0e6)]:
        lam = 2 * np.pi / _theta(d, base)
        ax.plot(i, lam, color=c, lw=1.8)
        ax.text(63, y, lab, color=c, fontsize=8, ha="right", va="bottom")
    for L, ls in [(4096, ":"), (8192, "--")]:
        ax.axhline(L, color=p.muted, lw=0.9, ls=ls)
    ax.text(1, 4096 * 0.62, "4k", color=p.muted, fontsize=7.5)
    ax.text(1, 8192 * 1.18, "8k context", color=p.muted, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlim(0, 63)
    ax.set_ylim(3, 6e6)
    ax.set_xlabel("pair index i (fast → slow)")
    ax.set_ylabel(r"wavelength $2\pi/\theta_i$ (tokens)")
    ax.text(2, 1.2e5, "above the line: less than one\nfull turn over the context",
            fontsize=7, color=p.label, va="bottom")
    return fig


def _expected_score(d, base, delta):
    th = _theta(d, base)
    return np.cos(np.outer(delta, th)).mean(axis=1)


@register(TOPIC, "decay")
def decay(p):
    """Top: E[score] ∝ mean_i cos(δθ_i) for two bases. Bottom: RoFormer's Abel bound (b = 10^4)."""
    d = 128
    delta = np.unique(np.round(np.logspace(0, 5, 1500)).astype(int))
    fig, (ax, bx) = figure(3.9, nrows=2)
    ax.axhline(0, color=p.faint, lw=0.8)
    for base, c, lab, xy in [(1e4, p.c(0), r"$b=10^4$", (1.3, 0.3)),
                             (5e5, p.c(1), r"$b=5\times10^5$", (1.3, 0.08))]:
        ax.plot(delta, _expected_score(d, base, delta), color=c, lw=1.2)
        ax.text(*xy, lab, color=c, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(-0.25, 1.05)
    ax.set_ylabel(r"mean$_i\cos(\delta\theta_i)$")
    ax.set_title(r"expected score when $\mathbb{E}[kq^\top]=cI$  (d = 128)", fontsize=8.5)
    dl = np.arange(0, 251)
    th = _theta(d, 1e4)
    bound = np.array([np.abs(np.cumsum(np.exp(1j * x * th))).mean() for x in dl])
    bx.plot(dl, bound, color=p.accent, lw=1.4)
    bx.set_xlim(0, 250)
    bx.set_ylim(0, 34)
    bx.set_xlabel(r"relative distance $\delta=|m-n|$")
    bx.set_ylabel(r"$\frac{1}{d/2}\sum_j |S_j|$")
    bx.set_title(r"RoFormer's bound factor ($b=10^4$, d = 128)", fontsize=8.5)
    return fig


@register(TOPIC, "layouts")
def layouts(p):
    """Which coordinates are rotated together: interleaved (RoFormer, GPT-J) vs half-split (GPT-NeoX, Llama)."""
    d = 8
    fig, ax = figure(2.3)
    blank(ax, (-1.2, 8.4), (-0.9, 3.9))
    rows = [("interleaved", 2.6, [(0, 1), (2, 3), (4, 5), (6, 7)]),
            ("half-split", 0.3, [(0, 4), (1, 5), (2, 6), (3, 7)])]
    for name, y, pairs in rows:
        ax.text(-1.15, y + 0.85, name, fontsize=8, color=p.fg, ha="left")
        pair_of = {}
        for f, (a, b) in enumerate(pairs):
            pair_of[a] = pair_of[b] = f
        for j in range(d):
            box(ax, (j + 0.06, y), 0.88, 0.55, f"{j}", p, color=p.c(pair_of[j] + 2 if pair_of[j] else 0),
                fontsize=7.5, mono=True)
        for f, (a, b) in enumerate(pairs):
            xa, xb = a + 0.5, b + 0.5
            h = 0.25 + 0.12 * (b - a)
            xs = np.linspace(xa, xb, 30)
            ys = y + 0.55 + h * np.sin(np.pi * (xs - xa) / (xb - xa))
            ax.plot(xs, ys, color=p.c(f + 2 if f else 0), lw=1.1)
            ax.text((xa + xb) / 2, y + 0.6 + h, rf"$\theta_{f}$", fontsize=7.5,
                    color=p.c(f + 2 if f else 0), ha="center", va="bottom")
    ax.text(3.6, -0.75, "same frequency set, different coordinate pairs",
            fontsize=7, color=p.muted, ha="center")
    return fig


@register(TOPIC, "decay-q")
def decay_q(p):
    """[fig-Q] Expected-score curves for two bases (A = 10^6, B = 500), d = 128, unlabeled by base."""
    d = 128
    delta = np.unique(np.round(np.logspace(0, 5, 1500)).astype(int))
    fig, ax = figure(2.2)
    ax.axhline(0, color=p.faint, lw=0.8)
    for base, c, lab, xy in [(1e6, p.c(0), "A", (200, 0.8)), (500, p.c(1), "B", (5, 0.3))]:
        ax.plot(delta, _expected_score(d, base, delta), color=c, lw=1.2)
        ax.text(*xy, lab, color=c, fontsize=9.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlabel(r"offset $\delta$")
    ax.set_ylabel(r"mean$_i\cos(\delta\theta_i)$")
    return fig
