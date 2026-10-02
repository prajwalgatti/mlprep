"""Figures for sys.flops-mfu."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register

# 7B-class config (LLaMA-7B-like): layers, hidden, gated-MLP width, vocab
L, H, FF, V = 32, 4096, 11008, 32000


def _mat(ax, x, y, w, h, label, p, color, hl=None):
    """Matrix rectangle; hl = 'w' or 'h' marks the contracted edge with a thick accent line."""
    ax.add_patch(Rectangle((x, y), w, h, fc=p.surface, ec=color, lw=1.0))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, color=p.fg)
    if hl == "w":     # contracted along width (columns)
        ax.plot([x, x + w], [y + h, y + h], color=p.label, lw=2.6, solid_capstyle="butt")
    if hl == "h":     # contracted along height (rows)
        ax.plot([x, x], [y, y + h], color=p.label, lw=2.6, solid_capstyle="butt")


@register("sys.flops-mfu", "matmul-backward")
def matmul_backward(p):
    """Forward Y = XW and the two backward matmuls; each contracts one dimension, each costs 2bkn."""
    fig, ax = figure(3.3)
    blank(ax, (0, 12), (0, 11.2))
    b, k, m = 2.4, 1.6, 2.0          # drawn sizes for tokens b, in-features k, out-features m
    rows = [
        (8.0, "forward", [("X", b, k, "w"), ("W", k, m, "h"), ("Y", b, m, None)],
         ["b×k", "k×m", "b×m"], "contract k  →  2bkm"),
        (4.4, "input grad", [("dY", b, m, "w"), (r"$W^{T}$", m, k, "h"), ("dX", b, k, None)],
         ["b×m", "m×k", "b×k"], "contract m  →  2bkm"),
        (0.6, "weight grad", [(r"$X^{T}$", k, b, "w"), ("dY", b, m, "h"), ("dW", k, m, None)],
         ["k×b", "b×m", "k×m"], "contract b  →  2bkm"),
    ]
    for y0, name, mats, dims, note in rows:
        ax.text(0.0, y0 + 2.75, name, fontsize=7.5, color=p.accent, va="center")
        x = 0.0
        for i, ((lab, h, w, hl), d) in enumerate(zip(mats, dims)):
            yy = y0 + (2.4 - h) / 2
            _mat(ax, x, yy, w, h, lab, p, p.accent if i == 2 else p.muted, hl=hl)
            ax.text(x + w / 2, yy - 0.3, d, ha="center", va="center", fontsize=6.3, color=p.muted)
            x += w
            if i == 0:
                ax.text(x + 0.3, y0 + 1.2, "·", fontsize=12, color=p.fg, ha="center", va="center")
                x += 0.6
            elif i == 1:
                ax.text(x + 0.35, y0 + 1.2, "=", fontsize=9, color=p.fg, ha="center", va="center")
                x += 0.7
        ax.text(12.0, y0 + 1.2, note, fontsize=7, color=p.fg, ha="right", va="center")
    ax.text(12.0, 10.85, "thick edge = contracted dimension", fontsize=6.5, color=p.label, ha="right")
    return fig


@register("sys.flops-mfu", "flops-breakdown")
def flops_breakdown(p):
    """Per-token training FLOPs of a 7B-class decoder by component, against sequence length."""
    seqs = [2048, 4096, 8192, 32768, 131072]
    mlp = 6 * 3 * H * FF * L
    proj = 6 * 4 * H * H * L
    head = 6 * H * V
    parts = [("MLP", mlp, p.c(0)), ("attn. projections", proj, p.c(2)), ("LM head", head, p.c(3))]
    fig, ax = figure(2.7)
    x = np.arange(len(seqs))
    for i, s in enumerate(seqs):
        bottom = 0.0
        attn = 12 * L * s * H
        for name, val, col in parts + [("attention scores", attn, p.c(1))]:
            ax.bar(i, val / 1e9, bottom=bottom, color=col, width=0.62,
                   label=name if i == 0 else None)
            bottom += val / 1e9
        share = attn / 1e9 / bottom
        ax.text(i, bottom + 4, f"{share:.0%}", ha="center", fontsize=7, color=p.c(1))
    ax.set_xticks(x)
    ax.set_xticklabels(["2k", "4k", "8k", "32k", "128k"])
    ax.set_xlabel("sequence length s")
    ax.set_ylabel("training GFLOPs per token")
    ax.set_ylim(0, 290)
    ax.legend(loc="upper left", handlelength=1.0, fontsize=7)
    ax.text(4.35, 262, "% = attention-score share", fontsize=6.5, color=p.c(1), ha="right")
    return fig


@register("sys.flops-mfu", "step-time")
def step_time(p):
    """Illustrative step-time decomposition: where the time goes between 100% and the achieved MFU."""
    ideal = 1.0                       # model FLOPs at peak
    eff = 0.85                        # matmul kernel efficiency
    segs = [
        ("model FLOPs\nat peak", ideal, p.good),
        ("kernels at\n85% of peak", ideal / eff - ideal, p.c(0)),
        ("full\nrecompute", (ideal / 3) / eff, p.c(2)),
        ("memory-\nbound ops", 0.30, p.c(1)),
        ("exposed\ncomm", 0.15, p.c(3)),
    ]
    busy = sum(v for _, v, _ in segs)
    p_, m = 4, 32
    bubble = busy * (p_ - 1) / m      # bubble fraction (p-1)/(m+p-1) of total time
    segs.append(("bubble", bubble, p.faint))
    total = busy + bubble
    mfu = ideal / total
    hfu = (ideal * 4 / 3) / total
    fig, ax = figure(2.0)
    left = 0.0
    for i, (name, v, col) in enumerate(segs):
        waste_hatch = "////" if i == 1 else None
        ax.barh(0, v, left=left, color=p.surface if i == 1 else col, height=0.5,
                edgecolor=col if i == 1 else p.surface, lw=0.8, hatch=waste_hatch)
        yt = 0.42 if i % 2 == 0 else -0.42
        ax.text(left + v / 2, yt, name, ha="center", va="center", fontsize=6.3, color=p.fg)
        left += v
    ax.annotate("", xy=(ideal, -0.75), xytext=(0, -0.75),
                arrowprops=dict(arrowstyle="<->", color=p.good, lw=0.9))
    ax.text(0.02, -0.98, f"MFU = {ideal:.2f} / {total:.2f} ≈ {mfu:.0%}   ·   HFU ≈ {hfu:.0%}",
            fontsize=7.2, color=p.fg, va="center")
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-1.15, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("step time (units of ideal compute time)")
    ax.spines["left"].set_visible(False)
    return fig
