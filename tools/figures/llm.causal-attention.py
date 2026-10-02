"""Figures for llm.causal-attention.

Masks are drawn from their exact definitions; the FLOP curves use the per-token formulas in the lesson
(linear layers 24 d^2, attention 4 d per query-key pair, forward FLOPs, one layer, illustrative d).
"""
import numpy as np
from matplotlib.colors import ListedColormap

from figures.style import figure, register


def _bidir(n):
    return np.ones((n, n), bool)


def _causal(n):
    return np.tril(np.ones((n, n), bool))


def _prefix(n, k):
    m = _causal(n)
    m[:k, :k] = True
    return m


def _docs(lengths):
    n = sum(lengths)
    m = np.zeros((n, n), bool)
    s = 0
    for L in lengths:
        m[s:s + L, s:s + L] = np.tril(np.ones((L, L), bool))
        s += L
    return m


def _window(n, w):
    i, j = np.indices((n, n))
    return (j <= i) & (j > i - w)


def _draw_mask(ax, m, p, title=None):
    cmap = ListedColormap([p.surface, p.accent])
    ax.imshow(m.astype(int), cmap=cmap, vmin=0, vmax=1)
    n = m.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=p.faint, lw=0.6)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(p.muted)
        s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8)


@register("llm.causal-attention", "masks")
def masks(p):
    """Four 8x8 attention masks (rows = queries, columns = keys; filled = may attend)."""
    n = 8
    panels = [
        ("bidirectional (encoder)", _bidir(n)),
        ("causal (decoder)", _causal(n)),
        ("prefix-LM, prefix = 3", _prefix(n, 3)),
        ("causal + document mask", _docs([3, 3, 2])),
    ]
    fig, axes = figure(3.55, nrows=2, ncols=2)
    for ax, (t, m) in zip(axes.flat, panels):
        _draw_mask(ax, m, p, t)
    axes[1, 0].set_xlabel("key position", fontsize=7.5)
    axes[1, 0].set_ylabel("query position", fontsize=7.5)
    return fig


@register("llm.causal-attention", "masks-q")
def masks_q(p):
    """[fig-Q] Four unlabelled 8x8 masks: sliding window 3, document mask (3,5), prefix-LM 3, prefix-LM 5."""
    n = 8
    panels = [("A", _window(n, 3)), ("B", _docs([3, 5])), ("C", _prefix(n, 3)), ("D", _prefix(n, 5))]
    fig, axes = figure(1.35, ncols=4)
    for ax, (t, m) in zip(axes, panels):
        _draw_mask(ax, m, p, t)
    return fig


@register("llm.causal-attention", "timeline")
def timeline(p):
    """Score entries computed during generation: one prefill block, then one new query row per decode step."""
    n_prompt, n_dec = 5, 4
    n = n_prompt + n_dec
    grid = np.full((n, n), np.nan)
    for i in range(n_prompt):
        grid[i, : i + 1] = 0
    for s in range(n_dec):
        i = n_prompt + s
        grid[i, : i + 1] = 1 + s
    cols = [p.accent, p.c(1), p.c(2), p.c(3), p.c(4)]
    fig, ax = figure(2.75)
    from matplotlib.colors import ListedColormap as LC
    ax.imshow(np.ma.masked_invalid(grid), cmap=LC(cols), vmin=-0.5, vmax=4.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=p.faint, lw=0.6)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(p.muted)
        s.set_linewidth(0.6)
    ax.set_xlabel("key position (= KV-cache slot)", fontsize=7.5)
    ax.set_ylabel("query position", fontsize=7.5)
    ax.text(n - 0.3, 2.0, "prefill: prompt rows\nin one parallel pass",
            fontsize=7.2, color=p.accent, ha="right", va="center")
    for s in range(n_dec):
        i = n_prompt + s
        ax.text(n - 0.3, i, f"decode {s + 1}: 1 query, {i + 1} keys",
                fontsize=6.8, color=cols[1 + s], ha="left", va="center")
    ax.set_xlim(-0.5, n + 5.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


@register("llm.causal-attention", "cache-flops")
def cache_flops(p):
    """Cumulative forward FLOPs to generate T tokens from scratch, one layer, d = 1024 (illustrative)."""
    d = 1024
    T = np.unique(np.logspace(0, np.log10(32768), 200).astype(int))
    t = np.arange(1, T.max() + 1, dtype=float)
    lin_no = 24 * d**2 * t
    att_no = 4 * d * t * (t + 1) / 2          # full causal re-run over t tokens
    lin_c = np.full_like(t, 24 * d**2)
    att_c = 4 * d * t                          # one query vs t cached keys
    cum = lambda x: np.cumsum(x)[T - 1]
    fig, ax = figure(2.45)
    ax.plot(T, cum(lin_no + att_no), color=p.bad, label="no cache: re-run prefix")
    ax.plot(T, cum(lin_c + att_c), color=p.good, label="KV cache")
    ax.plot(T, cum(att_no), color=p.bad, lw=1.0, ls="--")
    ax.plot([], [], color=p.muted, lw=1.0, ls="--", label="dashed: attention part only")
    ax.plot(T, cum(att_c), color=p.good, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens generated $T$")
    ax.set_ylabel("cumulative FLOPs")
    ax.text(3000, 4e15, r"$\propto T^3$", color=p.bad, fontsize=8)
    ax.text(9000, 2e10, r"$\propto T^2$", color=p.good, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
    return fig
