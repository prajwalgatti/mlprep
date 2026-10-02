"""Figures for sys.ddp: overlap timeline, bucket-size trade-off (alpha-beta model), DP compute/comm roofline."""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import blank, figure, register

C = 989e12          # H100 dense bf16 FLOP/s (peak)
N = 1.3e9           # parameters in the worked example


def _canvas(h):
    fig, ax = figure(h)
    fig.set_layout_engine(None)
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    return fig, ax


def _block(ax, x, y, w, h, label, p, fill, text_color=None, fs=7.0, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=p.surface, lw=0.8, hatch=hatch))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=text_color or p.surface)


@register("sys.ddp", "ddp-overlap")
def ddp_overlap(p):
    """Naive DDP (one all-reduce after backward) vs bucketed overlap. 4 layers, one bucket per layer.
    Forward 1 unit/layer, backward 2 units/layer, all-reduce of one bucket 1.25 units (illustrative)."""
    L, f, b, c = 4, 1.0, 2.0, 1.25
    fig, ax = _canvas(2.55)
    blank(ax, (-3.2, 18.4), (-3.4, 7.2))
    rows = {"naive": (5.0, 3.9), "overlap": (1.4, 0.3)}   # (compute y, comm y)
    h = 0.85
    for name, (yc, ym) in rows.items():
        ax.text(-0.25, yc + h / 2, "compute", ha="right", va="center", fontsize=6.8, color=p.muted)
        ax.text(-0.25, ym + h / 2, "comm", ha="right", va="center", fontsize=6.8, color=p.muted)
        t = 0.0
        for l in range(L):
            _block(ax, t, yc, f, h, f"F{l + 1}", p, p.muted)
            t += f
        ready = []
        for l in reversed(range(L)):
            _block(ax, t, yc, b, h, f"B{l + 1}", p, p.c(l))
            t += b
            ready.append((l, t))
        t_bwd = t
        if name == "naive":
            _block(ax, t_bwd, ym, c * L, h, "all-reduce", p, p.bad, fs=6.8)
            end = t_bwd + c * L
            ax.text(8.6, yc + h + 0.35, "naive: whole backward, then one all-reduce", ha="center",
                    fontsize=7.2, color=p.fg)
        else:
            tc = 0.0
            for l, tr in ready:
                start = max(tr, tc)
                _block(ax, start, ym, c, h, f"AR{l + 1}", p, p.c(l))
                ax.annotate("", (start + 0.05, ym + h), (tr, yc), arrowprops=dict(arrowstyle="-|>", color=p.fg,
                                                                                  lw=0.6, mutation_scale=6))
                tc = start + c
            end = tc
            ax.add_patch(Rectangle((t_bwd, ym - 0.12), end - t_bwd, h + 0.24, facecolor="none", edgecolor=p.label,
                                   lw=1.1, ls="--"))
            ax.text(t_bwd + (end - t_bwd) / 2, ym - 0.5, "exposed:\nlast bucket", ha="center", va="top",
                    fontsize=6.6, color=p.label)
            ax.text(8.6, yc + h + 0.35, "DDP: all-reduce each bucket once it is ready", ha="center",
                    fontsize=7.2, color=p.fg)
        ax.plot([end, end], [ym - 0.15, yc + h + 0.15], color=p.label, lw=1.0)
        ax.text(end + 0.15, yc + h / 2, "step\nends", fontsize=6.5, color=p.label, va="center")
    ax.text(7.6, -2.75, "backward runs layer 4 → 1, so bucket 4 is ready first\ncolour = layer = bucket",
            ha="center", va="center", fontsize=6.6, color=p.muted, linespacing=1.3)
    return fig


def _exposed(bucket, alpha, S=2.6e9, Tb=4 * N * 8192 / C, W=400e9, P=64):
    """Exposed comm time for one backward: gradients appear uniformly over Tb, buckets all-reduce in order
    on one comm stream, each costing alpha + 2(P-1)/P * bytes / W."""
    k = int(np.ceil(S / bucket - 1e-6))   # tolerance: bucket sizes that divide S exactly
    t = 0.0
    for i in range(1, k + 1):
        ready = min(i * bucket, S) / S * Tb
        sz = min(bucket, S - (i - 1) * bucket)
        t = max(t, ready) + alpha + 2 * (P - 1) / P * sz / W
    return t - Tb


@register("sys.ddp", "bucket-tradeoff")
def bucket_tradeoff(p):
    """Exposed communication vs bucket size for 1.3B bf16 grads on 64 GPUs (8192 tokens/GPU), alpha-beta model."""
    ks = np.unique(np.round(np.logspace(0, np.log10(2.6e9 / 2 ** 18), 220)).astype(int))
    mib = 2.6e9 / ks / 2 ** 20                         # bucket sizes that divide the buffer evenly
    fig, ax = figure(2.5)
    for i, (a, lab) in enumerate([(20e-6, r"$\alpha$ = 20 µs"), (50e-6, r"$\alpha$ = 50 µs")]):
        y = np.array([_exposed(m * 2 ** 20, a) for m in mib]) * 1e3
        ax.loglog(mib, np.maximum(y, 1e-2), color=p.c(i), lw=1.8)
        ax.text(6 if i else 0.27, 300 if i else 1.2, lab, color=p.c(i), fontsize=7.5)
    ax.axvline(25, color=p.muted, lw=0.8, ls=":")
    ax.text(27, 120, "default\n25 MiB", fontsize=7, color=p.muted)
    ax.text(0.3, 0.02, "too many small\nall-reduces:\nlatency-bound", fontsize=7, color=p.fg)
    ax.text(150, 0.02, "few big buckets:\nlast one waits for\nthe whole backward", fontsize=7, color=p.fg)
    full = 2 * 63 / 64 * 2.6e9 / 400e9 * 1e3
    ax.axhline(full, color=p.label, lw=0.8, ls="--")
    ax.text(0.3, full * 1.3, "one bucket: all 12.8 ms exposed", fontsize=7, color=p.label)
    ax.set_xlabel("bucket size (MiB, log)")
    ax.set_ylabel("exposed comm per step (ms, log)")
    ax.set_ylim(1e-2, 1e3)
    return fig


@register("sys.ddp", "dp-roofline")
def dp_roofline(p):
    """Backward compute time vs all-reduce time per step for a 1.3B model, as per-GPU tokens vary (large-P limit)."""
    x = np.logspace(np.log10(256), np.log10(32768), 200)
    S = 2 * N                      # bf16 gradient bytes
    tb = 4 * N * x / C * 1e3       # ms
    fig, ax = figure(2.6)
    ax.loglog(x, tb, color=p.fg, lw=2.0)
    ax.text(1000, 2.2, "backward:\n$4N\\cdot$tokens$/C$", color=p.fg, fontsize=7.3, ha="left")
    lines = [(2 * S / 450e9, "bf16, one node", p.c(0)),
             (2 * S / 400e9, "bf16, across nodes", p.c(2)),
             (2 * 2 * S / 400e9, "fp32, across nodes", p.c(1))]
    for i, (tc, lab, col) in enumerate(lines):
        ax.axhline(tc * 1e3, color=col, lw=1.4, ls="--")
        xc = tc * C / (4 * N)
        ax.plot([xc], [tc * 1e3], "o", color=col, ms=4.5, zorder=5)
        below = i == 0
        ax.text(280, tc * 1e3 * (0.9 if below else 1.08), lab, color=col, fontsize=6.9,
                va="top" if below else "bottom")
        if i == 2:   # keep the fp32 label clear of the backward line: up and to the left
            ax.text(xc / 1.15, tc * 1e3 * 1.15, f"{xc:,.0f}", color=col, fontsize=6.9, va="bottom", ha="right")
        elif i == 1:  # below-right of its dot, under the line
            ax.text(xc * 1.12, tc * 1e3 * 0.93, f"{xc:,.0f}", color=col, fontsize=6.9, va="top")
        else:         # further below-right of its dot
            ax.text(xc * 1.1, tc * 1e3 * 0.62, f"{xc:,.0f}", color=col, fontsize=6.9, va="top")
    ax.text(330, 0.62, "comm-bound\n(comm exposed)", fontsize=7, color=p.bad)
    ax.text(30000, 0.62, "compute-bound\n(comm hidden)", fontsize=7, color=p.good, ha="right")
    ax.set_xlabel("tokens per GPU per step (log)")
    ax.set_ylabel("time per step (ms, log)")
    ax.set_xlim(256, 32768)
    ax.set_ylim(0.4, 200)
    return fig
