"""Figures for sys.roofline. H100 SXM: 989e12 dense bf16 FLOP/s, 1.979e15 fp8, 3.35e12 B/s HBM."""
import numpy as np

from figures.style import figure, register

C_BF16, C_FP8, W = 989e12, 1.979e15, 3.35e12
C_CUDA = 66.9e12                      # CUDA-core fp32 with FMA (132*4*32 lanes * 1.98 GHz * 2)
D, F = 4096, 11008                    # 7B-class MLP up-projection


def mm_intensity(B, D=D, F=F):
    """bf16 [B,D]x[D,F]: 2BDF FLOPs over 2(BD+DF+BF) bytes."""
    return B * D * F / (B * D + D * F + B * F)


def _roof(ax, p, fp8=True, cuda=True, labels=True, bf16_label=True):
    I = np.logspace(-1.3, 3.9, 400)
    ax.loglog(I, np.minimum(C_BF16, I * W) / 1e12, color=p.fg, lw=1.8)
    if fp8:
        ax.loglog(I, np.minimum(C_FP8, I * W) / 1e12, color=p.muted, lw=1.1, ls="--")
    if cuda:
        ax.axhline(C_CUDA / 1e12, color=p.c(2), lw=0.9, ls=":")
    if labels:
        r = C_BF16 / W
        ax.axvline(r, color=p.faint, lw=0.8)
        ax.text(r * 1.08, 0.13, f"ridge ≈ {r:.0f}", fontsize=7, color=p.muted, rotation=90, va="bottom")
        if bf16_label:
            ax.text(7.5e3, C_BF16 / 1e12 * 0.42, "bf16 roof", fontsize=7, color=p.fg, ha="right")
        if fp8:
            ax.text(3e3, C_FP8 / 1e12 * 1.22, "fp8 roof", fontsize=7, color=p.muted, ha="center")
        if cuda:
            ax.text(1500, C_CUDA / 1e12 * 0.55, "CUDA cores only\n(no Tensor Cores)", fontsize=6.4, color=p.c(2), ha="center", va="top")
    ax.set_xlim(0.05, 8e3)
    ax.set_ylim(0.1, 4e3)
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("attainable TFLOP/s")


@register("sys.roofline", "roofline-h100")
def roofline_h100(p):
    """H100 roofline with bf16 elementwise ops and a 7B-class MLP matmul at several token counts."""
    fig, ax = figure(2.9)
    _roof(ax, p, bf16_label=False)
    pts = [("residual add", 1 / 6, (0.07, 2.2)), ("LayerNorm", 2.0, (0.9, 20))]
    for B, xy in [(16, (0.25, 120)), (128, (2.0, 700)), (4096, (250, 200))]:
        pts.append((f"matmul, {B} tokens", mm_intensity(B), xy))
    for name, I, xy in pts:
        y = min(C_BF16, I * W) / 1e12
        ax.plot([I], [y], "o", color=p.accent, ms=4.5, zorder=5)
        ax.annotate(name, (I, y), xytext=xy, fontsize=6.8, color=p.accent,
                    arrowprops=dict(arrowstyle="-", color=p.accent, lw=0.6))
    return fig


@register("sys.roofline", "roofline-points")
def roofline_points(p):
    """Unlabelled points A, B, C on the H100 bf16 roofline (for a quiz)."""
    fig, ax = figure(2.5)
    _roof(ax, p, fp8=False, cuda=False)
    for name, I in [("A", 2.0), ("B", mm_intensity(64)), ("C", mm_intensity(4096))]:
        y = min(C_BF16, I * W) / 1e12
        ax.plot([I], [y], "o", color=p.label, ms=5, zorder=5)
        dy, va = (2.0, "bottom") if name == "C" else (0.45, "top")
        ax.text(I, y * dy, name, fontsize=8.5, color=p.label, ha="center", va=va, weight="bold")
    return fig


@register("sys.roofline", "batch-sweep")
def batch_sweep(p):
    """Time and throughput of the bf16 [B,4096]x[4096,11008] matmul on H100 vs token count B."""
    B = np.logspace(0, 4.5, 300)
    flops = 2 * B * D * F
    t_math = flops / C_BF16
    t_mem = 2 * (B * D + D * F + B * F) / W
    t = np.maximum(t_math, t_mem)
    b_star = 1 / (W / C_BF16 - 1 / D - 1 / F)   # solve I(B) = C/W
    fig, (a1, a2) = figure(3.2, nrows=2, sharex=True)
    a1.loglog(B, t * 1e6, color=p.accent, lw=1.8)
    a1.loglog(B, t_mem * 1e6, color=p.muted, lw=0.9, ls="--")
    a1.loglog(B, t_math * 1e6, color=p.muted, lw=0.9, ls=":")
    a1.text(1.3, 36, r"$T_{mem}$: weights dominate, ≈ 27 µs", fontsize=6.8, color=p.fg)
    a1.text(2.2, 2.2, r"$T_{math}\propto B$", fontsize=6.8, color=p.muted)
    a1.set_ylabel("time per matmul (µs)")
    a1.set_ylim(0.5, 4e3)
    tok = B / t
    a2.loglog(B, tok, color=p.c(1), lw=1.8)
    a2.set_ylabel("tokens / s")
    a2.set_xlabel("tokens in the batch, B")
    a2.text(1.3, 2e7, "throughput $\\propto$ B\n(doubling B is free)", fontsize=6.8, color=p.fg, va="top")
    a2.text(1500, 4e6, "flat: compute-bound", fontsize=6.8, color=p.fg)
    a2.set_ylim(1e4, 6e7)
    for a in (a1, a2):
        a.axvline(b_star, color=p.label, lw=0.9, ls="-")
    a1.text(b_star * 1.12, 1500, f"B ≈ {b_star:.0f}", fontsize=7, color=p.label)
    return fig
