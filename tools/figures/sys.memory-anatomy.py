"""Figures for sys.memory-anatomy: memory over two training steps, per-layer activation terms, memory vs sequence length.

All curves are computed from the lesson's formulas for stated configs (no measured data):
- model states, mixed-precision Adam (layout: bf16 params + bf16 grads + fp32 master + fp32 m, v)
- activations from Korthikanti et al. 2022: sbh(34 + 5as/h) bytes per layer, or 34sbh with FlashAttention.
"""
import numpy as np

from figures.style import figure, register

GB = 1e9


def n_params(h, L, V):
    return h * V + L * (12 * h * h + 13 * h) + 2 * h


@register("sys.memory-anatomy", "memory-timeline")
def memory_timeline(p):
    """Two steps of a 1.97B GPT-style model (h=2560, L=24, V=32k), s=4096, micro-batch 8, FlashAttention."""
    h, L, V, s, b = 2560, 24, 32000, 4096, 8
    N = n_params(h, L, V)
    w16, master, opt, g16 = 2 * N / GB, 4 * N / GB, 8 * N / GB, 2 * N / GB
    act_layer = 34 * s * b * h / GB
    # time axis per step: forward L units, backward 2L units, optimizer 3 units
    t_all, comp = [], {k: [] for k in ["params", "opt", "grads", "acts"]}
    t0 = 0.0
    for step in range(2):
        ts = np.linspace(0, 3 * L + 3, 400)
        for t in ts:
            if t <= L:                          # forward: activations build up
                acts, grads = act_layer * t, 0.0
            elif t <= 3 * L:                    # backward: activations freed, grads appear
                frac = (t - L) / (2 * L)
                acts, grads = act_layer * L * (1 - frac), g16 * frac
            else:                               # optimizer step (grads freed at zero_grad afterwards)
                acts, grads = 0.0, g16
            has_opt = step > 0 or t > 3 * L + 1
            comp["params"].append(w16 + master)
            comp["opt"].append(opt if has_opt else 0.0)
            comp["grads"].append(grads)
            comp["acts"].append(acts)
            t_all.append(t0 + t)
        t0 += 3 * L + 3 + 2
    t_all = np.array(t_all)
    fig, ax = figure(2.6)
    ys = [np.array(comp[k]) for k in ["params", "opt", "grads", "acts"]]
    labels = ["weights (bf16 + fp32)", "Adam m, v (fp32)", "gradients (bf16)", "activations"]
    cols = [p.c(0), p.c(1), p.c(2), p.c(3)]
    ax.stackplot(t_all, *ys, colors=cols, alpha=0.85, lw=0)
    ax.axhline(85.5, color=p.bad, lw=1.0, ls="--")
    ax.text(62, 88, "H100 total ≈ 85.5 GB (79.6 GiB)", color=p.bad, fontsize=6.6, ha="center")
    peak1 = (w16 + master) + act_layer * L
    peak2 = peak1 + opt
    ax.annotate(f"step 1 peak {peak1:.0f} GB", (L, peak1), xytext=(L + 4, peak1 + 14), fontsize=6.8,
                color=p.fg, ha="center", arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.7))
    s2 = 3 * L + 5 + L
    ax.annotate(f"step 2 peak {peak2:.0f} GB: OOM", (s2, peak2), xytext=(s2 + 14, peak2 + 6), fontsize=6.8,
                color=p.bad, ha="center", arrowprops=dict(arrowstyle="-", color=p.muted, lw=0.7))
    for x, lab in [(L / 2, "fwd"), (2 * L, "bwd"), (3 * L + 1.5, "opt")]:
        ax.text(x, -7, lab, fontsize=6.6, color=p.muted, ha="center", va="top")
        ax.text(x + 3 * L + 5, -7, lab, fontsize=6.6, color=p.muted, ha="center", va="top")
    from matplotlib.patches import Rectangle
    for i, (lab, c) in enumerate(zip(labels, cols)):
        x0, y0 = (2 + 78 * (i % 2), 141 - 10 * (i // 2))
        ax.add_patch(Rectangle((x0, y0), 3, 5, color=c, clip_on=False))
        ax.text(x0 + 4.5, y0 + 2.5, lab, color=p.fg, fontsize=6.5, va="center")
    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 150)
    ax.set_xlim(0, t_all[-1])
    ax.set_ylabel("GB")
    ax.set_xlabel("time over two training steps →", labelpad=14)
    return fig


@register("sys.memory-anatomy", "activation-breakdown")
def activation_breakdown(p):
    """Share of each Korthikanti term in per-layer activation bytes; h=4096, a=32, b=1."""
    h, a, b = 4096, 32, 1
    seqs = [1024, 4096, 16384]
    names = ["attention, linear\n11sbh", "attention scores\n$5as^2b$", "MLP\n19sbh", "LayerNorms\n4sbh"]
    cols = [p.c(0), p.c(3), p.c(2), p.c(1)]
    fig, ax = figure(2.5)
    for j, s in enumerate(seqs):
        terms = np.array([11 * s * b * h, 5 * a * s * s * b, 19 * s * b * h, 4 * s * b * h], dtype=float)
        frac = terms / terms.sum()
        left = 0.0
        for k in range(4):
            ax.barh(j, frac[k], left=left, color=cols[k], height=0.62)
            if frac[k] > 0.07:
                ax.text(left + frac[k] / 2, j, f"{frac[k]:.0%}", ha="center", va="center", fontsize=7,
                        color=p.surface)
            left += frac[k]
        ax.text(1.02, j, f"{terms.sum() / GB:.2g} GB", va="center", fontsize=7, color=p.fg)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"s = {s // 1024}k" for s in seqs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of per-layer bytes (total at right)")
    names = [n.replace("\n", " ") for n in names]
    for k, (nm, c) in enumerate(zip(names, cols)):
        ax.text(0.0 + 0.6 * (k % 2), -1.15 + 0.38 * (k // 2), nm, color=c, fontsize=6.6, va="center")
    ax.set_ylim(2.5, -1.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig


@register("sys.memory-anatomy", "memory-vs-seq")
def memory_vs_seq(p):
    """1.27B model (h=2048, L=24, a=16), b=1: model states vs activations as sequence length grows."""
    h, L, a, V, b = 2048, 24, 16, 32000, 1
    N = n_params(h, L, V)
    s = np.logspace(np.log2(512), np.log2(65536), 200, base=2)
    states = np.full_like(s, 16 * N / GB)
    act_fa = 34 * s * b * h * L / GB
    act_naive = s * b * h * L * (34 + 5 * a * s / h) / GB
    fig, ax = figure(2.5)
    ax.plot(s, states, color=p.c(0), lw=1.8)
    ax.plot(s, states + act_fa, color=p.c(2), lw=1.8)
    ax.plot(s, states + act_naive, color=p.c(3), lw=1.8)
    ax.axhline(85.5, color=p.bad, lw=1.0, ls="--")
    ax.text(600, 92, "H100 ≈ 85.5 GB", color=p.bad, fontsize=7)
    ax.text(600, 13, f"model states only ({16 * N / GB:.0f} GB, flat)", color=p.c(0), fontsize=6.8)
    ax.text(2.0e4, 32, "+ activations,\nFlashAttention\n(linear in s)", color=p.c(2), fontsize=6.8)
    ax.text(1150, 150, "+ activations,\nscores stored\n(quadratic in s)", color=p.c(3), fontsize=6.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([512, 2048, 8192, 32768])
    ax.set_xticklabels(["512", "2k", "8k", "32k"])
    ax.set_yticks([10, 30, 100, 300, 1000])
    ax.set_yticklabels(["10", "30", "100", "300", "1000"])
    ax.set_ylim(8, 2000)
    ax.set_xlim(512, 65536)
    ax.set_xlabel("sequence length $s$ (micro-batch 1)")
    ax.set_ylabel("GB (log scale)")
    return fig
