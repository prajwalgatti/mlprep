"""Figures for sys.gpu-basics. H100 numbers from the Scaling Book Part 12 spec tables."""
import math

import numpy as np
from matplotlib.patches import Rectangle

from figures.style import arrow, blank, box, figure, register


@register("sys.gpu-basics", "gpu-anatomy")
def gpu_anatomy(p):
    """HBM -> L2 -> 132 SMs; one SM expanded into 4 subpartitions plus SMEM. Host CPU over PCIe."""
    fig, ax = figure(3.75)
    blank(ax, (0, 12), (0, 12.4))
    # HBM and L2
    box(ax, (0.3, 0.2), 11.4, 1.1, "HBM (off-chip)   80 GB  ·  3.35 TB/s", p, color=p.c(1), fontsize=7.5)
    arrow(ax, (6.0, 1.35), (6.0, 1.95), p, style="<|-|>")
    box(ax, (0.3, 2.0), 11.4, 0.9, "L2 cache  50 MB, shared by all SMs · ≈5.5 TB/s", p, fontsize=6.8)
    # SM row
    for k in range(6):
        x = 0.3 + k * 1.15
        hl = k == 2
        box(ax, (x, 3.55), 0.95, 0.85, "SM", p, color=p.accent if hl else None, fontsize=6.8, lw=1.3 if hl else 1.0)
        arrow(ax, (x + 0.47, 3.5), (x + 0.47, 2.95), p, lw=0.6, style="<|-|>")
    ax.text(7.25, 3.97, "… ×132", fontsize=7.5, color=p.muted, va="center")
    # host
    box(ax, (9.3, 3.55), 2.4, 0.85, "host CPU", p, color=p.c(2), fontsize=7)
    ax.annotate("", xy=(10.5, 2.95), xytext=(10.5, 3.5),
                arrowprops=dict(arrowstyle="<|-|>", color=p.c(2), lw=0.9, mutation_scale=8))
    ax.text(10.65, 3.22, "PCIe", fontsize=6.5, color=p.c(2), va="center")
    # expanded SM
    x0, y0, w, h = 0.3, 5.0, 11.4, 7.1
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=p.accent, lw=1.2, ls="-"))
    ax.plot([2.6, x0], [4.42, y0], color=p.accent, lw=0.7, ls=":")
    ax.plot([3.25, x0 + w], [4.42, y0], color=p.accent, lw=0.7, ls=":")
    ax.text(x0 + 0.2, y0 + h - 0.42, "one SM: 4 subpartitions", fontsize=7.5, color=p.accent, va="center")
    cw = 2.6
    for i in range(4):
        cx = x0 + 0.35 + i * (cw + 0.12)
        box(ax, (cx, 9.35), cw, 1.65, "Tensor Core\n(matmul unit)", p, color=p.accent, fontsize=6.3)
        box(ax, (cx, 7.95), cw, 1.2, "warp sched.\n32 CUDA cores", p, fontsize=6.1)
        box(ax, (cx, 6.75), cw, 1.0, "registers\n64 KiB", p, fontsize=6.3)
    box(ax, (x0 + 0.35, 5.3), w - 0.7, 1.1, "SMEM / L1   256 kB per SM (programmer-managed)", p, color=p.c(2), fontsize=6.8)
    return fig


@register("sys.gpu-basics", "wave-quantisation")
def wave_quantisation(p):
    """Modelled GEMM time in waves vs N for M = 4096, 128x256 output tiles, one tile per SM, 132 SMs."""
    M, tm, tn, sms = 4096, 128, 256, 132
    N = np.arange(1024, 8192 + 1, 1)
    tiles = math.ceil(M / tm) * np.ceil(N / tn)
    waves = np.ceil(tiles / sms)
    ideal = (M / tm) * (N / tn) / sms
    fig, ax = figure(2.45)
    ax.plot(N, waves, color=p.accent, lw=1.6, drawstyle="steps-post", label="waves launched (time)")
    ax.plot(N, ideal, color=p.muted, lw=1.0, ls="--", label="useful work, in waves")
    ax.annotate("N = 4096: 512 tiles\n→ 4 waves, 97% busy", (4096, 4), xytext=(1150, 6.3), fontsize=7,
                color=p.fg, arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.7))
    ax.annotate("N = 4097: 544 tiles\n→ 5 waves, 82% busy", (4097, 5), xytext=(5000, 2.2), fontsize=7,
                color=p.label, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.set_xlabel("N (output columns), M = 4096")
    ax.set_ylabel("kernel time (waves)")
    ax.set_xlim(1024, 8192)
    ax.set_ylim(0, 8.6)
    ax.set_xticks([1024, 2048, 4096, 6144, 8192])
    ax.legend(loc="upper left", handlelength=1.6, bbox_to_anchor=(0.0, 1.02))
    return fig


def _blk(ax, x, w, y, h, text, fc, p, ec=None, tc=None, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec or fc, lw=0.6, hatch=hatch))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4, color=tc or p.fg)


@register("sys.gpu-basics", "streams-async")
def streams_async(p):
    """CPU enqueues launches ahead of the GPU; a .item() sync drains the queue and leaves GPU gaps."""
    fig, axes = figure(2.9, nrows=2, gridspec_kw={"height_ratios": [3, 2.3]})
    rows = {"CPU": 2.0, "GPU compute": 1.0, "GPU comm": 0.0}
    lw, kw = 0.35, 1.0     # launch cost on CPU, kernel time on GPU
    # (a) no sync
    ax = axes[0]
    for i in range(7):
        _blk(ax, i * 0.5, lw, rows["CPU"] + 0.15, 0.6, "", p.c(2), p)
    for i in range(7):
        _blk(ax, 0.4 + i * kw, kw - 0.04, rows["GPU compute"] + 0.15, 0.6, f"k{i + 1}", p.surface, p, ec=p.accent)
    _blk(ax, 3.45, 3.0, rows["GPU comm"] + 0.15, 0.6, "all-reduce (overlaps)", p.surface, p, ec=p.c(1))
    ax.annotate("", xy=(0.4, rows["GPU compute"] + 0.78), xytext=(0.17, rows["CPU"] + 0.12),
                arrowprops=dict(arrowstyle="-|>", color=p.muted, lw=0.6, mutation_scale=6))
    ax.text(3.6, rows["CPU"] + 0.45, "CPU runs ahead, queue stays full", fontsize=6.8, color=p.muted, va="center")
    ax.set_title("(a) no sync: launches are queued", fontsize=8, loc="left")
    # (b) .item() after k3 each step
    ax = axes[1]
    t = 0.0
    gpu_t = 0.4
    for step in range(2):
        for j in range(3):
            _blk(ax, t, lw, rows["CPU"] + 0.15, 0.6, "", p.c(2), p)
            start = max(gpu_t, t + 0.4)
            if start > gpu_t + 1e-9:
                _blk(ax, gpu_t, start - gpu_t, rows["GPU compute"] + 0.15, 0.6, "", p.bad, p)
            _blk(ax, start, kw - 0.04, rows["GPU compute"] + 0.15, 0.6, f"k{3 * step + j + 1}", p.surface, p, ec=p.accent)
            gpu_t = start + kw
            t += 0.5
        # .item(): CPU waits for GPU to finish
        _blk(ax, t, gpu_t - t, rows["CPU"] + 0.15, 0.6, ".item() waits", p.surface, p, ec=p.bad, tc=p.bad)
        t = gpu_t + 0.05
    ax.text(3.62, rows["GPU compute"] - 0.12, "GPU idle", fontsize=6.6, color=p.bad, va="center", ha="center")
    ax.set_title("(b) .item() every step: the queue drains", fontsize=8, loc="left")
    for a, lo in zip(axes, (-0.1, 0.7)):
        a.set_xlim(-0.05, 7.6)
        a.set_ylim(lo, 2.9)
        ticks = [("CPU", rows["CPU"]), ("GPU stream 0", rows["GPU compute"]), ("GPU stream 1", rows["GPU comm"])]
        ticks = [(n, y + 0.45) for n, y in ticks if y + 0.45 > lo]
        a.set_yticks([y for _, y in ticks])
        a.set_yticklabels([n for n, _ in ticks], fontsize=7)
        a.set_xticks([])
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
    axes[1].set_xlabel("time →")
    return fig
