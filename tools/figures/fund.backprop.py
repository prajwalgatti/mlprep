"""Figures for fund.backprop."""
import numpy as np

from figures.style import arrow, blank, box, figure, register

# Worked example: L = (tanh(w x + b) - y)^2 + lam w^2 with w=0.25, x=2, b=0, y=1, lam=0.1
_W, _X, _B, _Y, _LAM = 0.25, 2.0, 0.0, 1.0, 0.1


def _worked_values():
    a1 = _W * _X
    a = a1 + _B
    h = np.tanh(a)
    r = h - _Y
    sq = r ** 2
    pen = _LAM * _W ** 2
    L = sq + pen
    rb = 2 * r
    ab = rb * (1 - h ** 2)
    return dict(a1=a1, a=a, h=h, r=r, sq=sq, pen=pen, L=L,
                rb=rb, hb=rb, ab=ab, a1b=ab, bb=ab, wb1=ab * _X, wb2=2 * _LAM * _W)


@register("fund.backprop", "graph-forward-backward")
def graph_forward_backward(p):
    """Computational graph of the worked example: forward values above each node, adjoints below."""
    v = _worked_values()
    fig, ax = figure(2.75)
    blank(ax, (-0.2, 14.2), (-0.6, 7.6))
    bw, bh = 1.55, 0.95
    yc = 4.2
    nodes = [  # (x-left, label, forward value, adjoint text)
        (1.7, r"$\times$", f"{v['a1']:.2f}", f"{v['a1b']:.3f}"),
        (3.75, "+", f"{v['a']:.2f}", f"{v['ab']:.3f}"),
        (5.8, "tanh", f"{v['h']:.3f}", f"{v['hb']:.3f}"),
        (7.85, r"$-y$", f"{v['r']:.3f}", f"{v['rb']:.3f}"),
        (9.9, r"$(\cdot)^2$", f"{v['sq']:.3f}", "1"),
        (12.2, "+", f"{v['L']:.3f}", "1"),
    ]
    for i, (x0, lab, fv, adj) in enumerate(nodes):
        box(ax, (x0, yc - bh / 2), bw, bh, lab, p, fontsize=8.5)
        ax.text(x0 + bw / 2 + 0.12, yc + 0.85, fv.replace("-", "\u2212"), ha="left", fontsize=7.3, color=p.fg)
        ax.text(x0 + bw / 2, yc - 1.15, adj.replace("-", "\u2212"), ha="center", fontsize=7.3, color=p.label)
        if i < len(nodes) - 1:
            nx = nodes[i + 1][0]
            arrow(ax, (x0 + bw, yc), (nx, yc), p)
    # inputs
    ax.text(0.15, yc, "$w$", fontsize=10, color=p.accent, ha="center", va="center")
    ax.text(0.15, yc + 0.85, "0.25", ha="center", fontsize=7.3, color=p.fg)
    arrow(ax, (0.45, yc), (1.7, yc), p, color=p.accent)
    ax.text(2.47, 6.9, "$x{=}2$", fontsize=8, ha="center", color=p.muted)
    arrow(ax, (2.47, 6.6), (2.47, yc + bh / 2), p)
    ax.text(4.52, 6.9, "$b{=}0$", fontsize=8, ha="center", color=p.muted)
    arrow(ax, (4.52, 6.6), (4.52, yc + bh / 2), p)
    ax.text(8.62, 6.9, "$y{=}1$", fontsize=8, ha="center", color=p.muted)
    arrow(ax, (8.62, 6.6), (8.62, yc + bh / 2), p)
    # penalty branch (fan-out of w)
    px = 9.9
    box(ax, (px, 0.35), bw, bh, r"$\lambda w^2$", p, fontsize=8)
    ax.text(px + bw + 0.12, 1.2, f"{v['pen']:.4f}", fontsize=7.3, color=p.fg, va="center")
    ax.text(px + bw + 0.12, 0.45, "1", fontsize=7.3, color=p.label, va="center")
    ax.plot([0.15, 0.15], [yc - 0.4, 0.82], color=p.accent, lw=1.0)
    arrow(ax, (0.15, 0.82), (px, 0.82), p, color=p.accent)
    arrow(ax, (px + bw / 2 + 0.3, 0.35 + bh), (12.6, yc - bh / 2), p)
    # w adjoint = sum of two paths
    ax.text(0.55, 2.1, r"$\bar w = -1.692 + 0.050 = -1.642$", fontsize=7.6, color=p.label)
    ax.text(0.55, 1.35, "(two paths: adjoints add)", fontsize=7.0, color=p.label)
    ax.text(px + bw / 2, -0.1, r"$\lambda{=}0.1$", fontsize=7.2, color=p.muted, ha="center", va="center")
    # legend
    ax.text(5.2, 7.25, "forward value", fontsize=7.3, color=p.fg, ha="left")
    ax.text(9.6, 7.25, r"adjoint $\partial L/\partial(\cdot)$", fontsize=7.3, color=p.label, ha="left")
    return fig


@register("fund.backprop", "forward-vs-reverse")
def forward_vs_reverse(p):
    """Jacobian chain for f: R^6 -> R^1 through two 4-wide layers; VJP sweep vs JVP sweep."""
    fig, ax = figure(2.9)
    blank(ax, (0, 16), (0, 11.2))
    u = 0.42  # size of one matrix entry

    def mat(x0, ytop, rows, cols, col, lab=None, fill=None):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x0, ytop - rows * u), cols * u, rows * u, facecolor=fill or p.surface,
                               edgecolor=col, lw=1.0))
        if lab:
            ax.text(x0 + cols * u / 2, ytop + 0.25, lab, ha="center", va="bottom", fontsize=7.2, color=p.muted)
        return x0 + cols * u

    # Row 1: reverse mode
    ax.text(0.1, 10.7, "reverse mode: one sweep of vector-Jacobian products", fontsize=7.8, color=p.fg)
    y1 = 9.4
    x = 0.4
    x = mat(x, y1, 1, 1, p.label, r"$\bar L$", fill=p.label) + 0.35
    x = mat(x, y1, 1, 4, p.muted, r"$J_3$ (1×4)") + 0.35
    x = mat(x, y1, 4, 4, p.muted, r"$J_2$ (4×4)") + 0.35
    mat(x, y1, 4, 6, p.muted, r"$J_1$ (4×6)")
    ax.annotate("", xy=(13.6, 7.15), xytext=(0.6, 7.15),
                arrowprops=dict(arrowstyle="-|>", color=p.label, lw=1.1))
    ax.text(0.4, 6.5, "evaluate left to right: every partial product is a row vector\n"
            r"1×1 → 1×4 → 1×4 → 1×6 = the whole gradient $\nabla_x L$", fontsize=7.0, color=p.label, va="top")
    # Row 2: forward mode
    ax.text(0.1, 4.55, "forward mode: one Jacobian-vector product per input", fontsize=7.8, color=p.fg)
    y2 = 3.5
    x = 0.4
    x = mat(x, y2, 1, 4, p.muted, r"$J_3$") + 0.35
    x = mat(x, y2, 4, 4, p.muted, r"$J_2$") + 0.35
    x = mat(x, y2, 4, 6, p.muted, r"$J_1$") + 0.35
    mat(x, y2, 6, 1, p.accent, r"$e_i$", fill=p.accent)
    ax.annotate("", xy=(0.6, 0.55), xytext=(12.4, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=p.accent, lw=1.1))
    ax.text(12.9, 2.4, "right to left:\n6×1 → 4×1\n→ 4×1 → 1×1\n= one entry\n" r"$\partial L/\partial x_i$",
            fontsize=7.0, color=p.accent, va="top")
    ax.text(0.4, 0.0, "repeat for i = 1…6: six sweeps", fontsize=7.0, color=p.accent, va="bottom")
    return fig


@register("fund.backprop", "checkpointing")
def checkpointing(p):
    """Stored activations over time for a 16-layer chain: store-all vs sqrt(L) checkpoints."""
    L, k = 16, 4
    seg = L // k
    # store everything: forward 1 unit/layer, backward 2 units/layer
    t0, m0 = [0], [0]
    for i in range(L):
        t0.append(t0[-1] + 1)
        m0.append(i + 1)
    for i in range(L):
        t0.append(t0[-1] + 2)
        m0.append(L - i - 1)
    # checkpointing: keep segment boundaries in forward; recompute each segment during backward
    t1, m1 = [0], [0]
    for i in range(L):
        t1.append(t1[-1] + 1)
        stored = (i + 1) // seg
        transient = 0 if (i + 1) % seg == 0 else 1
        m1.append(stored + transient)
    ck = k
    for s in reversed(range(k)):
        ck_before = s  # checkpoints left (boundaries before this segment)
        for j in range(seg):  # recompute the segment from its starting checkpoint
            t1.append(t1[-1] + 1)
            m1.append(ck_before + 1 + j + 1)
        for j in range(seg):  # backward through the segment
            t1.append(t1[-1] + 2)
            m1.append(ck_before + 1 + seg - j - 1)
        ck = ck_before
    fig, ax = figure(2.45)
    ax.step(t0, m0, where="post", color=p.c(1), lw=1.6)
    ax.step(t1, m1, where="post", color=p.accent, lw=1.6)
    ax.axhline(max(m0), color=p.c(1), lw=0.6, ls=":")
    ax.axhline(max(m1), color=p.accent, lw=0.6, ls=":")
    ax.text(36, 15.0, "store all: peak 16,\ntime 48", color=p.c(1), fontsize=7.3, va="top")
    ax.text(52, 9.0, f"4 checkpoints: peak {max(m1)},\ntime {t1[-1]} (+1 forward)", color=p.accent,
            fontsize=7.3, va="bottom", ha="center")
    ax.axvline(16, color=p.faint, lw=0.8)
    ax.text(8, 17.2, "forward", ha="center", fontsize=7.3, color=p.muted)
    ax.text(40, 17.2, "backward", ha="center", fontsize=7.3, color=p.muted)
    ax.set_xlim(0, 66)
    ax.set_ylim(0, 18.5)
    ax.set_xlabel("time (units of one layer's forward)")
    ax.set_ylabel("activations stored")
    ax.set_yticks([0, 4, 8, 12, 16])
    return fig


def _fd_errors():
    """Errors of forward and centred differences for the worked-example loss, float64."""
    def f(w):
        return (np.tanh(_X * w) - _Y) ** 2 + _LAM * w ** 2
    w0 = _W
    t = np.tanh(_X * w0)
    exact = 2 * (t - _Y) * (1 - t ** 2) * _X + 2 * _LAM * w0
    hs = np.logspace(-14, -1, 300)
    fwd = np.abs((f(w0 + hs) - f(w0)) / hs - exact)
    ctr = np.abs((f(w0 + hs) - f(w0 - hs)) / (2 * hs) - exact)
    return hs, np.maximum(fwd, 1e-17), np.maximum(ctr, 1e-17)


@register("fund.backprop", "fd-error")
def fd_error(p):
    """Absolute error of finite-difference derivatives against the exact derivative, float64."""
    hs, fwd, ctr = _fd_errors()
    fig, ax = figure(2.5)
    ax.loglog(hs, fwd, color=p.c(1), lw=1.1)
    ax.loglog(hs, ctr, color=p.accent, lw=1.1)
    ax.text(1e-6, 1e-3, "forward\n" r"error $\propto h$", color=p.c(1), fontsize=7.3)
    ax.text(5e-3, 1e-7, "centred\n" r"error $\propto h^2$", color=p.accent, fontsize=7.3)
    ax.text(2e-14, 1e-11, "round-off\n" r"$\propto \epsilon/h$", color=p.muted, fontsize=7.3)
    for x0, col in [(1.5e-8, p.c(1)), (6e-6, p.accent)]:
        ax.axvline(x0, color=col, lw=0.6, ls=":")
    ax.text(1.6e-8, 3e-14, r"$\sqrt{\epsilon}$", color=p.c(1), fontsize=7.5)
    ax.text(6.5e-6, 3e-14, r"$\epsilon^{1/3}$", color=p.accent, fontsize=7.5)
    ax.set_xlabel("step size $h$ (float64)")
    ax.set_ylabel("absolute error")
    ax.set_ylim(1e-14, 1)
    ax.set_xlim(1e-14, 1e-1)
    return fig
