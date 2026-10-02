"""Figures for llm.self-attention.

All data is synthetic and computed from the real formulas: random Gaussian queries/keys for the
saturation curves and heatmaps, and the exact FLOP-count formulas for the attention share.
"""
import numpy as np

from figures.style import arrow, blank, box, figure, register


def _softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


@register("llm.self-attention", "dataflow")
def dataflow(p):
    """One multi-head self-attention layer with the tensor shape on every edge (batch dim omitted)."""
    fig, ax = figure(3.9)
    blank(ax, (0, 10), (0, 15.2))
    bh = 1.05
    # input
    box(ax, (3.6, 13.6), 2.8, bh, r"$X$", p, color=p.accent, fontsize=9)
    ax.text(6.6, 14.1, r"$n\times d$", fontsize=7.5, color=p.muted, va="center")
    # projections
    xs = [0.5, 3.6, 6.7]
    names = ["Q", "K", "V"]
    for x0, nm in zip(xs, names):
        arrow(ax, (5.0, 13.6), (x0 + 1.4, 12.25), p)
        box(ax, (x0, 11.2), 2.8, bh, rf"$XW_{nm}$", p, fontsize=8.5)
        ax.text(x0 + 1.4, 10.55, r"$h\times n\times d_k$", fontsize=7.2, color=p.muted, ha="center")
    ax.text(0.2, 14.1, r"$W_{Q,K,V}$: $d\times d$", fontsize=7.2, color=p.muted, va="center")
    # scores
    box(ax, (0.5, 8.3), 5.1, bh, r"$S=QK^\top/\sqrt{d_k}$", p, fontsize=8.5)
    arrow(ax, (1.9, 10.3), (1.9, 9.4), p)
    arrow(ax, (5.0, 10.3), (5.0, 9.4), p)
    ax.text(5.8, 8.82, r"$h\times n\times n$", fontsize=7.2, color=p.muted, va="center")
    # softmax
    box(ax, (0.5, 6.1), 5.1, bh, "row-wise softmax", p, color=p.label, fontsize=8)
    arrow(ax, (3.05, 8.3), (3.05, 7.2), p)
    ax.text(5.8, 6.62, r"$P$: $h\times n\times n$", fontsize=7.2, color=p.muted, va="center")
    # weighted sum; V comes down the right-hand side
    box(ax, (2.0, 3.9), 5.0, bh, r"$PV$ per head", p, fontsize=8.5)
    arrow(ax, (3.05, 6.1), (3.05, 5.0), p)
    ax.plot([8.1, 8.1], [10.3, 8.0], color=p.muted, lw=1.0)
    ax.plot([8.1, 8.6, 8.6], [8.0, 7.6, 4.42], color=p.muted, lw=1.0)
    arrow(ax, (8.6, 4.42), (7.0, 4.42), p)
    ax.text(8.75, 6.0, r"$V$", fontsize=8, color=p.muted, va="center")
    ax.text(1.85, 4.42, r"$h\times n\times d_k$", fontsize=7.2, color=p.muted, va="center", ha="right")
    # concat + W_O
    box(ax, (2.0, 1.7), 5.0, bh, r"concat, $\times W_O$", p, fontsize=8.5)
    arrow(ax, (4.5, 3.9), (4.5, 2.8), p)
    ax.text(7.2, 2.22, r"$n\times hd_k\!=\!n\times d$", fontsize=7.2, color=p.muted, va="center")
    arrow(ax, (4.5, 1.7), (4.5, 0.75), p, color=p.accent)
    ax.text(4.5, 0.25, r"output $n\times d$", fontsize=7.8, color=p.accent, ha="center", va="center")
    return fig


def _saturation_stats(dks, n_keys=32, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for scaled in (False, True):
        mx, jac = [], []
        for dk in dks:
            q = rng.normal(size=(trials, 1, dk))
            k = rng.normal(size=(trials, n_keys, dk))
            s = (q * k).sum(-1)
            if scaled:
                s = s / np.sqrt(dk)
            pr = _softmax(s)
            mx.append(pr.max(-1).mean())
            # Frobenius norm of the softmax Jacobian diag(p) - p p^T
            j = np.einsum("ti,ij->tij", pr, np.eye(n_keys)) - pr[:, :, None] * pr[:, None, :]
            jac.append(np.sqrt((j ** 2).sum(axis=(1, 2))).mean())
        out[scaled] = (np.array(mx), np.array(jac))
    return out


@register("llm.self-attention", "saturation")
def saturation(p):
    """Max attention weight and softmax-Jacobian norm vs d_k, with and without 1/sqrt(d_k), at init."""
    dks = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    st = _saturation_stats(dks)
    fig, axes = figure(2.25, ncols=2)
    for ax, idx, ttl in zip(axes, [0, 1], ["mean max weight", r"$\|\partial p/\partial z\|_F$"]):
        ax.plot(dks, st[False][idx], color=p.bad, marker="o", ms=2.5, label="unscaled")
        ax.plot(dks, st[True][idx], color=p.good, marker="o", ms=2.5, label=r"$\div\sqrt{d_k}$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 32, 256])
        ax.set_xticklabels(["4", "32", "256"])
        ax.set_xlabel(r"$d_k$")
        ax.set_title(ttl, fontsize=8.5)
    axes[0].set_ylim(0, 1.02)
    axes[0].axhline(1 / 32, color=p.muted, lw=0.8, ls="--")
    axes[0].text(1100, 0.045, "uniform 1/32", fontsize=6.8, color=p.muted, ha="right", va="bottom")
    axes[1].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, handlelength=1.4, fontsize=7.5)
    return fig


@register("llm.self-attention", "heatmaps-q")
def heatmaps_q(p):
    """[fig-Q] Three 6x6 attention maps from the same random q, k with d_k = 256, under three logit scalings.

    A: divided by d_k (too much). B: unscaled. C: divided by sqrt(d_k). Panels are unlabelled on purpose.
    """
    rng = np.random.default_rng(3)
    dk, n = 256, 6
    q = rng.normal(size=(n, dk))
    k = rng.normal(size=(n, dk))
    raw = q @ k.T
    maps = {"A": _softmax(raw / dk), "B": _softmax(raw), "C": _softmax(raw / np.sqrt(dk))}
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("att", [p.surface, p.accent])
    fig, axes = figure(1.65, ncols=3)
    for ax, (name, m) in zip(axes, maps.items()):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(p.faint)
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("query", fontsize=7.5)
    axes[1].set_xlabel("key", fontsize=7.5)
    return fig


@register("llm.self-attention", "flop-share")
def flop_share(p):
    """Fraction of forward FLOPs in the two attention matmuls (QK^T and PV) vs n/d, under three conventions."""
    r = np.logspace(-1, np.log10(60), 300)   # n/d
    # per token, per layer, forward, in units of d^2: attention 4 n d -> 4 r ; linear 24 or 32
    curves = [
        ("non-gated 4d MLP, full: $n=6d$", 4 * r, 24, p.c(0), 6),
        ("gated MLP $F{=}4d$, full: $n=8d$", 4 * r, 32, p.c(1), 8),
        ("non-gated, causal: $n=12d$", 2 * r, 24, p.c(2), 12),
    ]
    fig, ax = figure(2.35)
    for lab, att, lin, col, cross in curves:
        ax.plot(r, att / (att + lin), color=col, label=lab)
        ax.plot([cross], [0.5], "o", color=col, ms=4)
    ax.axhline(0.5, color=p.muted, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 6, 12, 50])
    ax.set_xticklabels(["0.1", "1", "6", "12", "50"])
    ax.set_xlabel(r"sequence length / model width  ($n/d$)")
    ax.set_ylabel("attention share of FLOPs")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.2)
    return fig
