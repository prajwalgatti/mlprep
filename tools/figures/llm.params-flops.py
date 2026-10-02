"""Figures for llm.params-flops (per-layer parameter breakdown, parameter shares, attention FLOP share)."""
import numpy as np

from figures.style import figure, register

# Llama 3 configs (Table 3 of Dubey et al. 2024); head dim 128; released vocab 128,256.
CFG = {
    "8B": dict(L=32, d=4096, h=32, hkv=8, dff=14336),
    "70B": dict(L=80, d=8192, h=64, hkv=8, dff=28672),
    "405B": dict(L=126, d=16384, h=128, hkv=8, dff=53248),
}
V, DH = 128256, 128


def _layer(c):
    attn = 2 * c["d"] * DH * (c["h"] + c["hkv"])
    ffn = 3 * c["d"] * c["dff"]
    return attn, ffn


@register("llm.params-flops", "layer-breakdown")
def layer_breakdown(p):
    """Parameters of one Llama-3-70B block, matrix by matrix (norm gains omitted: 16k)."""
    d, h, hkv, dff = 8192, 64, 8, 28672
    names = ["$W_Q$", "$W_K$", "$W_V$", "$W_O$", "gate", "up", "down"]
    vals = np.array([d * h * DH, d * hkv * DH, d * hkv * DH, h * DH * d, d * dff, d * dff, d * dff]) / 1e6
    cols = [p.c(0)] * 4 + [p.c(1)] * 3
    fig, ax = figure(2.35)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    tot = vals.sum()
    for yi, v in zip(y, vals):
        ax.text(v + 4, yi, f"{v:.1f}M ({100 * v / tot:.0f}%)", va="center", fontsize=7, color=p.fg)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 330)
    ax.set_xlabel("parameters (millions)")
    ax.text(150, 5.6, f"attention {vals[:4].sum():.0f}M", color=p.c(0), fontsize=7.4)
    ax.text(150, 4.9, f"SwiGLU MLP {vals[4:].sum():.0f}M", color=p.c(1), fontsize=7.4)
    ax.tick_params(axis="y", length=0)
    return fig


@register("llm.params-flops", "param-share")
def param_share(p):
    """Share of parameters in embedding+head, attention and MLP for four real configs."""
    rows = []
    # GPT-2 small: tied 50,257 x 768 embedding + 1024 learned positions; MHA, 4d MLP, biases.
    d, L = 768, 12
    emb = 50257 * d + 1024 * d
    attn = L * (4 * d * d + 4 * d)
    ffn = L * (8 * d * d + 5 * d)
    rows.append(("GPT-2 small\n124M", emb, attn, ffn))
    for name, c in CFG.items():
        a, f = _layer(c)
        rows.append((f"Llama 3\n{name}", 2 * V * c["d"], c["L"] * a, c["L"] * f))
    fig, ax = figure(2.4)
    labels = ["embed + head", "attention", "MLP"]
    cols = [p.c(2), p.c(0), p.c(1)]
    y = np.arange(len(rows))[::-1]
    for yi, (name, *parts) in zip(y, rows):
        tot = sum(parts)
        left = 0
        for j, v in enumerate(parts):
            w = 100 * v / tot
            ax.barh(yi, w, left=left, color=cols[j], height=0.6, edgecolor=p.surface, lw=0.6)
            if w > 7:
                ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=6.8, color=p.surface)
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of parameters")
    for j, (lab, col) in enumerate(zip(labels, cols)):
        ax.text(2 + 34 * j, len(rows) - 0.35, lab, color=col, fontsize=7.2)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.tick_params(axis="y", length=0)
    return fig


@register("llm.params-flops", "attention-share")
def attention_share(p):
    """Attention-score share of forward FLOPs per token vs context, Llama-3 8B and 70B, two conventions."""
    n = np.logspace(np.log10(1024), np.log10(1 << 20), 300)
    fig, ax = figure(2.45)
    for i, name in enumerate(["8B", "70B"]):
        c = CFG[name]
        a, f = _layer(c)
        lin = 2 * c["L"] * (a + f) + 2 * c["d"] * V
        hd = c["h"] * DH
        for conv, mult, ls in [("causal avg", 2, "-"), ("non-causal", 4, "--")]:
            att = mult * c["L"] * n * hd
            share = att / (att + lin)
            ax.semilogx(n, share, color=p.c(i), ls=ls, lw=1.7 if ls == "-" else 1.2)
            k = np.argmin(np.abs(share - 0.5))
            ax.plot([n[k]], [0.5], "o", color=p.c(i), ms=3.2)
        ax.text(1.15e3, 0.9 - 0.09 * i, f"Llama 3 {name} (d = {c['d']})", color=p.c(i), fontsize=7.2)
    ax.axhline(0.5, color=p.muted, lw=0.7, ls=":")
    ax.text(1.15e3, 0.69, "solid: causal average\ndashed: no causal halving", color=p.muted, fontsize=6.8,
            va="top")
    ax.set_xticks([1024, 8192, 65536, 524288])
    ax.set_xticklabels(["1k", "8k", "64k", "512k"])
    ax.set_xlabel("context length $n$ (tokens)")
    ax.set_ylabel("attention share of fwd FLOPs")
    ax.set_ylim(0, 1)
    return fig
