"""Figures for sb.transformer-memory (Scaling Book ch. 4, MoE, rematerialization, KV caches)."""
import numpy as np

from figures.style import figure, register


@register("sb.transformer-memory", "remat-tradeoff")
def remat_tradeoff(p):
    """Saved activations per layer vs training FLOPs for three checkpointing policies (book's rough counts)."""
    pols = [("save everything", 20, 6.0), ("big matmuls only", 7, 6.0), ("block remat", 1, 8.0)]
    fig, (a1, a2) = figure(2.1, ncols=2)
    y = np.arange(3)[::-1]
    for ax, idx, xlab in [(a1, 1, "saved tensors per layer\n(≈ BTD-sized, rough)"), (a2, 2, "FLOPs per param per token")]:
        vals = [q[idx] for q in pols]
        ax.barh(y, vals, color=[p.c(0), p.c(2), p.c(1)], height=0.55, alpha=0.9)
        for yi, v in zip(y, vals):
            lab = f"{v:g}" + (" (+ attn recompute)" if idx == 2 and yi == 1 else "")
            ax.text(v + (0.4 if idx == 1 else 0.15), yi, lab, va="center", fontsize=7.5 if idx == 1 else 6.8, color=p.fg)
        ax.set_xlabel(xlab, fontsize=7.5)
        ax.set_yticks([])
    for yi, (lab, _, _) in zip(y, pols):
        a1.text(0.3, yi + 0.37, lab, fontsize=6.8, color=p.muted)
    a1.set_xlim(0, 25)
    a2.set_xlim(0, 13)
    return fig


@register("sb.transformer-memory", "kv-vs-context")
def kv_vs_context(p):
    """int8 KV cache per sequence for LLaMA-3 70B shapes (L=80, H=128) with K=8 KV heads vs full multi-head (K=64)."""
    S = np.logspace(np.log10(1024), np.log10(131072), 200)
    fig, ax = figure(2.4)
    for K, col, lab in [(64, p.c(1), "MHA, K = 64"), (8, p.c(0), "GQA, K = 8")]:
        kv = 2 * 80 * K * 128 * S / 2 ** 30
        ax.loglog(S, kv, color=col)
        ax.text(S[-1] * 0.5, kv[-1] * 1.35, lab, color=col, fontsize=7.2, ha="center")
    ax.axhline(16, color=p.muted, lw=0.8, ls=":")
    ax.text(1100, 18.5, "one v5e's HBM (16 GiB)", fontsize=7, color=p.muted)
    ax.set_xlabel("context length S (tokens)")
    ax.set_ylabel("KV cache per sequence (GiB)")
    ax.set_xticks([1024, 8192, 32768, 131072])
    ax.set_xticklabels(["1k", "8k", "32k", "128k"])
    ax.set_ylim(0.1, 400)
    return fig


@register("sb.transformer-memory", "attention-intensity")
def attention_intensity(p):
    """Arithmetic intensity of dot-product attention: decode (T=1) saturates at G; prefill (T=S) grows like S."""
    S = np.logspace(1, 5, 300)
    fig, ax = figure(2.4)
    for G, col in [(1, p.c(1)), (8, p.c(0))]:
        ax.loglog(S, S * S * G / (S * G + S), color=col, ls="-")
        ax.loglog(S, S * G / (G + S), color=col, ls="--")
        ax.text(*((250, 2500) if G == 8 else (2.2e4, 900)), f"prefill, G = {G}", color=col, fontsize=7, ha="center")
        ax.text(2e4, G * 1.25, f"decode, G = {G}", color=col, fontsize=7, ha="center")
    ax.axhline(240, color=p.muted, lw=0.7, ls=":")
    ax.text(12, 290, "v5e ridge ≈ 240", fontsize=7, color=p.muted)
    ax.set_xlabel("KV length S (prefill: T = S)")
    ax.set_ylabel("FLOPs per byte")
    ax.set_ylim(0.5, 1e5)
    return fig
