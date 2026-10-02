"""Figures for sys.pipeline-parallel (GPipe, the bubble, 1F1B).

Schedules are produced by a small dependency-driven simulator (not hand-placed blocks):
F(i, k) waits for F(i-1, k); B(i, k) waits for B(i+1, k) (or F(i, k) on the last stage);
each stage runs its ops in a fixed order (all-forward-all-backward, or 1F1B).
"""
import numpy as np
from matplotlib.patches import Rectangle

from figures.style import figure, register


def _orders(p, m, kind):
    out = []
    for i in range(p):
        if kind == "gpipe":
            o = [("F", k) for k in range(m)] + [("B", k) for k in range(m)]
        else:  # 1F1B (PipeDream-Flush): p-i-1 warm-up forwards, then alternate, then cool-down
            w = min(p - i - 1, m)
            o = [("F", k) for k in range(w)]
            f, bq = w, 0
            while f < m:
                o += [("F", f), ("B", bq)]
                f += 1
                bq += 1
            o += [("B", k) for k in range(bq, m)]
        out.append(o)
    return out


def simulate(p, m, kind, tf=1.0, tb=2.0):
    """Return per-stage lists of (op, microbatch, start, end) and the makespan."""
    ords = _orders(p, m, kind)
    done, ptr, free = {}, [0] * p, [0.0] * p
    sched = [[] for _ in range(p)]
    progress = True
    while progress:
        progress = False
        for i in range(p):
            while ptr[i] < len(ords[i]):
                op, k = ords[i][ptr[i]]
                if op == "F":
                    dep = done.get(("F", i - 1, k)) if i > 0 else 0.0
                else:
                    dep = done.get(("B", i + 1, k)) if i < p - 1 else done.get(("F", i, k))
                if dep is None:
                    break
                st = max(free[i], dep)
                d = tf if op == "F" else tb
                done[(op, i, k)] = st + d
                free[i] = st + d
                sched[i].append((op, k, st, st + d))
                ptr[i] += 1
                progress = True
    return sched, max(free)


def peak_inflight(sched):
    res = []
    for s in sched:
        cur = mx = 0
        for op, _, _, _ in s:
            cur += 1 if op == "F" else -1
            mx = max(mx, cur)
        res.append(mx)
    return res


def _gantt(ax, p_, sched, T, labels=True, fs=5.6):
    p = len(sched)
    for i, s in enumerate(sched):
        y = p - 1 - i
        ax.add_patch(Rectangle((0, y + 0.08), T, 0.84, facecolor=p_.faint, edgecolor="none", zorder=0))
        for op, k, a, b in s:
            col = p_.c(0) if op == "F" else p_.c(1)
            ax.add_patch(Rectangle((a, y + 0.08), b - a, 0.84, facecolor=col, edgecolor=p_.surface,
                                   lw=0.6, zorder=1))
            if labels:
                ax.text((a + b) / 2, y + 0.5, str(k + 1), ha="center", va="center", fontsize=fs,
                        color=p_.surface, zorder=2)
    ax.set_xlim(0, T)
    ax.set_ylim(0, p)
    ax.set_yticks([p - 1 - i + 0.5 for i in range(p)])
    ax.set_yticklabels([f"stage {i + 1}" for i in range(p)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


@register("sys.pipeline-parallel", "gpipe-schedule")
def gpipe_schedule(p):
    """GPipe (all-forward-all-backward), p=4, m=8, t_b = 2 t_f. Grey = bubble."""
    P, M = 4, 8
    sched, T = simulate(P, M, "gpipe")
    fig, ax = figure(1.9)
    _gantt(ax, p, sched, T)
    ax.set_xlabel(f"time (units of $t_f$; total = $(m+p-1)(t_f+t_b)$ = {T:.0f})", fontsize=7.5)
    ax.text(0.3, 4.15, "forward", color=p.c(0), fontsize=7)
    ax.text(5.2, 4.15, "backward (2× longer)", color=p.c(1), fontsize=7)
    ax.text(T - 0.3, 4.15, "grey = idle", color=p.muted, fontsize=7, ha="right")
    ax.set_ylim(0, 4.6)
    return fig


@register("sys.pipeline-parallel", "1f1b-schedule")
def one_f_one_b_schedule(p):
    """1F1B, p=4, m=8, t_b = 2 t_f, with the peak number of micro-batches whose activations each stage holds."""
    P, M = 4, 8
    sched, T = simulate(P, M, "1f1b")
    peaks = peak_inflight(sched)
    fig, ax = figure(1.9)
    _gantt(ax, p, sched, T)
    for i, pk in enumerate(peaks):
        ax.text(T + 0.5, P - 1 - i + 0.5, f"peak {pk}", fontsize=6.8, color=p.label, va="center",
                clip_on=False)
    ax.set_xlim(0, T + 4.2)
    ax.set_xlabel(f"time (units of $t_f$; same total = {T:.0f})", fontsize=7.5)
    ax.text(0.3, 4.15, "warm-up", color=p.muted, fontsize=7)
    ax.text(T - 0.3, 4.15, "cool-down", color=p.muted, fontsize=7, ha="right")
    ax.set_ylim(0, 4.6)
    return fig


@register("sys.pipeline-parallel", "schedule-quiz")
def schedule_quiz(p):
    """Unannotated 1F1B schedule, p=4, m=6, for a figure question (no in-flight counts shown)."""
    P, M = 4, 6
    sched, T = simulate(P, M, "1f1b")
    fig, ax = figure(1.7)
    _gantt(ax, p, sched, T)
    ax.text(0.3, 4.15, "F", color=p.c(0), fontsize=7.5)
    ax.text(1.4, 4.15, "B", color=p.c(1), fontsize=7.5)
    ax.text(2.6, 4.15, "numbers = micro-batch", color=p.muted, fontsize=7)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("time", fontsize=7.5)
    return fig


@register("sys.pipeline-parallel", "bubble-vs-m")
def bubble_vs_m(p):
    """Bubble vs number of micro-batches m for p = 4, 8, 16: fraction of total time (solid) and Narayanan's ratio to ideal time (dashed)."""
    m = np.arange(1, 129)
    fig, ax = figure(2.5)
    for j, P in enumerate([4, 8, 16]):
        frac = (P - 1) / (m + P - 1)
        ratio = (P - 1) / m
        ax.plot(m, 100 * frac, color=p.c(j), lw=1.8)
        ax.plot(m, 100 * ratio, color=p.c(j), lw=1.1, ls="--")
        ax.text(132, {4: 0.0, 8: 6.5, 16: 13.0}[P], f"$p$={P}", color=p.c(j), fontsize=7.5, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 128])
    ax.set_xticklabels(["1", "4", "16", "64", "128"])
    ax.set_ylim(-7, 100)
    ax.set_xlim(1, 185)
    ax.set_xlabel("micro-batches per flush $m$")
    ax.set_ylabel("bubble (%)")
    ax.plot([32], [100 * 7 / 39], "o", color=p.label, ms=4, zorder=5)
    ax.annotate("$p$=8, $m$=32: 17.9% of time idle\n(ratio to ideal 21.9%)", (32, 100 * 7 / 39),
                xytext=(1.15, 1.5), fontsize=6.5, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    ax.text(21, 99, "solid: $(p-1)/(m+p-1)$\ndashed: $(p-1)/m$", fontsize=7, color=p.muted, va="top")
    ax.grid(True, axis="y")
    return fig
