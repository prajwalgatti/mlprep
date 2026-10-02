"""Figures for sys.mixed-precision: loss-scaling histogram, dtype flow of a step, dynamic loss scale."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


@register("sys.mixed-precision", "grad-histogram")
def grad_histogram(p):
    """Synthetic log2-magnitude histogram of activation gradients, before and after scaling by S = 2^12."""
    rng = np.random.default_rng(3)
    e = rng.normal(-29, 5.0, 200_000)          # log2 |g|, synthetic
    shift = 12
    bins = np.arange(-50, 20, 1.0)
    fig, ax = figure(2.55)
    ax.axvspan(-24, -14, color=p.c(0), alpha=0.10, lw=0)
    ax.axvspan(-14, 16, color=p.c(0), alpha=0.22, lw=0)
    ax.text(1, 0.112, "fp16 normal\nrange", fontsize=6.8, color=p.c(0), ha="center", va="top")
    ax.text(-19, 0.112, "sub-\nnormal", fontsize=6.8, color=p.c(0), ha="center", va="top")
    h0, _ = np.histogram(e, bins=bins)
    h1, _ = np.histogram(e + shift, bins=bins)
    h0 = h0 / len(e)
    h1 = h1 / len(e)
    ax.stairs(h0, bins, color=p.muted, lw=1.2, fill=False)
    ax.stairs(h1, bins, color=p.label, lw=1.6, fill=False)
    lost0 = np.mean(e < -25)
    lost1 = np.mean(e + shift < -25)
    ax.text(-40, 0.098, f"unscaled:\n{lost0:.0%} round to 0", fontsize=6.8, color=p.muted, ha="center")
    ax.text(-8.5, 0.072, f"× $2^{{{shift}}}$:\n{lost1:.0%} round to 0", fontsize=6.8, color=p.label, ha="left")
    arrow(ax, (-31, 0.086), (-19.5, 0.086), p, color=p.label)
    ax.axvline(-25, color=p.bad, lw=0.9, ls="--")
    ax.text(-49, 0.062, "dashed: $2^{-25}$,\nbelow it fp16\nrounds to 0", fontsize=6.4, color=p.bad, ha="left")
    ax.set_xlim(-50, 19)
    ax.set_ylim(0, 0.115)
    ax.set_yticks([])
    ax.set_xticks([-48, -40, -32, -24, -16, -8, 0, 8, 16])
    ax.set_xlabel(r"$\log_2|g|$ (synthetic gradients)")
    ax.set_ylabel("fraction of values")
    return fig


@register("sys.mixed-precision", "dtype-flow")
def dtype_flow(p):
    """One mixed-precision training step: which tensor lives in which dtype, and where S enters and leaves."""
    fig, ax = figure(3.55)
    blank(ax, (0, 10), (0, 9.9))
    w, hgt = 4.3, 1.05
    L, R = 0.2, 5.5
    box(ax, (L, 8.5), w, hgt, "fp32 master weights", p, color=p.c(0), fontsize=7.5)
    box(ax, (R, 8.5), w, hgt, "cast → bf16/fp16 copy", p, fontsize=7.5)
    box(ax, (R, 6.6), w, hgt, "forward: half matmuls,\nfp32 softmax/norm/loss", p, fontsize=7)
    box(ax, (R, 4.7), w, hgt, "loss × S  (fp32)", p, color=p.label, fontsize=7.5)
    box(ax, (R, 2.8), w, hgt, "backward: half grads\n(scaled by S)", p, fontsize=7)
    box(ax, (L, 2.8), w, hgt, "unscale ÷S in fp32;\ninf/NaN? → skip, S/2", p, color=p.label, fontsize=7)
    box(ax, (L, 4.7), w, hgt, "clip grad norm\n(on unscaled grads)", p, fontsize=7)
    box(ax, (L, 6.6), w, hgt, "Adam step: m, v fp32\nupdate in fp32", p, color=p.c(0), fontsize=7)
    arrow(ax, (L + w, 9.02), (R, 9.02), p)
    arrow(ax, (R + w / 2, 8.5), (R + w / 2, 7.65), p)
    arrow(ax, (R + w / 2, 6.6), (R + w / 2, 5.75), p)
    arrow(ax, (R + w / 2, 4.7), (R + w / 2, 3.85), p)
    arrow(ax, (R, 3.32), (L + w, 3.32), p)
    arrow(ax, (L + w / 2, 3.85), (L + w / 2, 4.7), p)
    arrow(ax, (L + w / 2, 5.75), (L + w / 2, 6.6), p)
    arrow(ax, (L + w / 2, 7.65), (L + w / 2, 8.5), p, color=p.c(0))
    ax.text(R + w + 0.1, 6.2, "activations\nsaved in half", fontsize=6.3, color=p.muted, ha="right", va="center")
    ax.text(5.0, 1.75, "fp32: weights, optimizer state, reductions, unscale\n"
                       "half: matmul inputs, saved activations, activation grads",
            fontsize=6.6, color=p.muted, ha="center", va="center")
    ax.text(5.0, 0.6, "with bf16, S = 1 and the scale/skip boxes drop out",
            fontsize=6.6, color=p.label, ha="center", va="center")
    return fig


@register("sys.mixed-precision", "dynamic-scale")
def dynamic_scale(p):
    """Synthetic GradScaler trace: init 2^16, x0.5 and skip on overflow, x2 after 2000 clean steps."""
    steps = 10_000
    # synthetic overflow threshold: log2 of the largest safe scale drifts with training
    t = np.arange(steps)
    safe = 13.55 + 0.7 * np.sin(2 * np.pi * (t - 1500) / 7000.0)
    spikes = {5300, 8600}   # isolated bad batches that overflow at any scale
    log_s, clean = 16, 0
    trace, skips = [], []
    for i in range(steps):
        trace.append(log_s)
        overflow = log_s > safe[i] or i in spikes
        if overflow:
            skips.append((i, log_s))
            log_s -= 1
            clean = 0
        else:
            clean += 1
            if clean == 2000:
                log_s += 1
                clean = 0
    fig, ax = figure(2.4)
    ax.step(t, trace, where="post", color=p.accent, lw=1.6)
    xs, ys = zip(*skips)
    ax.plot(xs, ys, "x", color=p.bad, ms=5, mew=1.3)
    ax.text(250, 15.7, "start $2^{16}$:\nfirst steps overflow\nand halve", fontsize=6.6, color=p.fg, va="top")
    ax.text(9800, 16.3, "× = overflow: step skipped, S halved\nflat stretches of 2000 clean steps → S doubles",
            fontsize=6.6, color=p.muted, ha="right", va="top")
    ax.set_xlabel("step (synthetic run)")
    ax.set_ylabel(r"$\log_2 S$")
    ax.set_ylim(10.5, 16.6)
    ax.set_yticks([11, 12, 13, 14, 15, 16])
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k"])
    return fig
