"""Figures for sb.transformer-accounting (Scaling Book ch. 4, params and FLOPs per layer)."""
import numpy as np

from figures.style import figure, register


@register("sb.transformer-accounting", "flops-breakdown")
def flops_breakdown(p):
    """Training FLOPs per token per layer for LLaMA-3 70B shapes, split by MLP, QKVO projections and attention scores."""
    D, F, N, K, H = 8192, 28672, 64, 8, 128
    mlp = 18 * D * F
    proj = 12 * D * (N + K) * H
    Ts = [4096, 32768, 131072]
    fig, ax = figure(2.3)
    for i, T in enumerate(Ts):
        att = 12 * T * N * H          # full (non-causal) dot-product attention
        tot = mlp + proj + att
        left = 0
        for val, col, lab in [(mlp, p.c(0), "MLP"), (proj, p.c(2), "QKVO"), (att, p.c(1), "scores")]:
            ax.barh(i, val / 1e9, left=left, height=0.55, color=col, alpha=0.85, edgecolor=p.surface)
            if val / tot > 0.2:
                ax.text(left + val / 2e9, i, f"{lab}\n{100 * val / tot:.0f}%", ha="center", va="center",
                        fontsize=6.5, color=p.surface)
            left += val / 1e9
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=p.c(0), label="MLP"), Patch(color=p.c(2), label="QKVO proj."),
                       Patch(color=p.c(1), label="attn scores")], loc="upper right", fontsize=6.8, ncol=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"T = {t // 1024}k" for t in Ts])
    ax.invert_yaxis()
    ax.set_xlabel("training GFLOPs per token per layer")
    return fig


@register("sb.transformer-accounting", "attention-fraction")
def attention_fraction(p):
    """Share of matmul FLOPs spent on dot-product attention vs sequence length (F = 4D, N = K, D = NH): T/(8D) ratio."""
    T = np.logspace(3, 6, 300)
    fig, ax = figure(2.4)
    for i, D in enumerate([2048, 4608, 8192, 16384]):
        r = T / (8 * D)
        ax.semilogx(T, r / (1 + r), color=p.c(i), lw=1.6)
        ax.plot([8 * D], [0.5], "o", color=p.c(i), ms=3.5)
        ax.text(8 * D * 1.15, 0.47 - 0.0 * i, "", fontsize=6)
        lab = f"D = {D}" + ("  (Gemma-27B)" if D == 4608 else "")
        ax.text(1.3e3, 0.93 - i * 0.08, lab, color=p.c(i), fontsize=7)
    ax.axhline(0.5, color=p.muted, lw=0.7, ls=":")
    ax.text(1.3e5 * 4, 0.53, "T = 8D", fontsize=7, color=p.muted, ha="right")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("attention share of FLOPs")
    ax.set_ylim(0, 1)
    return fig
