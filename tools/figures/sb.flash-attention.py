"""Figures for sb.flash-attention (Scaling Book ch. 4, Appendix A)."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import arrow, blank, figure, register


@register("sb.flash-attention", "tiling")
def tiling(p):
    """Blocked attention: one query block stays resident while key/value blocks stream past; causal blocks above the diagonal are skipped."""
    n = 6
    fig, ax = figure(2.8)
    blank(ax, (-1.6, 9.6), (-1.6, 7.0))
    c = 0.95
    qrow = 3
    for i in range(n):
        for j in range(n):
            skip = j > i
            active = i == qrow and not skip
            face = p.faint if skip else (p.c(0) if active else p.surface)
            ax.add_patch(Rectangle((j * c, (n - 1 - i) * c), c, c, facecolor=face, alpha=0.9 if active else 1,
                                   edgecolor=p.muted, lw=0.6, hatch="////" if skip else None))
            if i == qrow and not skip:
                ax.text(j * c + c / 2, (n - 1 - i) * c + c / 2, f"{j + 1}", ha="center", va="center", fontsize=7,
                        color=p.surface)
    ax.text(n * c / 2, n * c + 0.25, "keys / values (S) →", ha="center", fontsize=7.5, color=p.fg)
    ax.text(-0.25, n * c / 2, "queries (T) →", ha="right", va="center", fontsize=7.5, color=p.fg, rotation=90)
    ax.text(6.2, 5.6, "hatched = skipped\n(causal mask)", ha="left", va="center", fontsize=7, color=p.muted)
    y = (n - 1 - qrow) * c + c / 2
    arrow(ax, (0.1, -0.35), (4 * c - 0.1, -0.35), p, color=p.label, lw=1.2)
    ax.text(2 * c, -0.6, "query block 4 visits KV blocks 1..4", ha="center", va="top", fontsize=6.8, color=p.label)
    ax.text(6.2, 4.5, "kept on-chip for\nthe query block:\n\n• Q block\n• running max M\n• running sum L\n• unnormalized\n  output A",
            fontsize=7, color=p.fg, va="top")
    return fig


@register("sb.flash-attention", "memory-vs-T")
def memory_vs_T(p):
    """Memory for one layer's attention (64 heads, H=128, bf16, one sequence): materialized score matrix vs Q-sized activations."""
    T = np.logspace(np.log10(1024), np.log10(131072), 200)
    N, H = 64, 128
    naive = 2 * T * T * N / 2 ** 30
    flash = 2 * T * N * H / 2 ** 30
    fig, ax = figure(2.4)
    ax.loglog(T, naive, color=p.bad)
    ax.loglog(T, flash, color=p.good)
    ax.text(1100, 9, "full score matrix\n$2\\,T^2 N$ bytes", color=p.bad, fontsize=7.2)
    ax.text(15000, flash[120] * 0.12, "Q/O-sized buffers\n$2\\,TNH$ bytes", color=p.good, fontsize=7.2)
    ax.axhline(96e9 / 2 ** 30, color=p.muted, lw=0.7, ls=":")
    ax.text(1100, 115, "v5p HBM (96 GB ≈ 89 GiB)", fontsize=6.8, color=p.muted)
    ax.set_xticks([1024, 8192, 32768, 131072])
    ax.set_xticklabels(["1k", "8k", "32k", "128k"])
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("GiB per layer")
    return fig
