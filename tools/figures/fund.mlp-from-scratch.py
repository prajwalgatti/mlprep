"""Figures for fund.mlp-from-scratch."""
import numpy as np

from figures.style import arrow, blank, box, figure, register


@register("fund.mlp-from-scratch", "xor-solution")
def xor_solution(p):
    """XOR in input space vs in the hidden ReLU space of DLB's 2-unit solution."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    W = np.array([[1, 1], [1, 1]], float)
    c = np.array([0, -1], float)
    H = np.maximum(X @ W + c, 0)
    fig, axes = figure(2.1, ncols=2)
    for ax, P, title in [(axes[0], X, "input space $x$"), (axes[1], H, "hidden space $h$")]:
        for cls, col, mk in [(0, p.c(0), "o"), (1, p.c(1), "s")]:
            ax.scatter(P[y == cls, 0], P[y == cls, 1], s=46, color=col, marker=mk, zorder=3,
                       edgecolor="none")
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
    ax0, ax1 = axes
    ax0.set_xlim(-0.4, 1.4)
    ax0.set_ylim(-0.4, 1.4)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xlabel("$x_1$")
    ax0.set_ylabel("$x_2$")
    ax0.text(0.5, -0.32, "no line separates", ha="center", fontsize=7, color=p.muted)
    ax1.set_xlim(-0.4, 2.4)
    ax1.set_ylim(-0.65, 1.5)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1])
    ax1.set_xlabel("$h_1$")
    ax1.set_ylabel("$h_2$")
    hh = np.linspace(-0.4, 2.4, 10)
    ax1.plot(hh, (hh - 0.5) / 2, color=p.label, lw=1.1, ls="--")
    ax1.text(1.0, -0.47, "(0,1), (1,0) → (1,0)", ha="center", va="center", fontsize=6.4,
             color=p.muted)
    ax1.text(-0.3, 1.25, r"$h_1-2h_2=\frac{1}{2}$", ha="left", fontsize=7.2, color=p.label)
    for ax in axes:
        ax.tick_params(labelsize=7)
    return fig


@register("fund.mlp-from-scratch", "shapes-diagram")
def shapes_diagram(p):
    """Forward (left, down) and backward (right, up) for a 784-256-10 MLP, batch 64, with shapes."""
    fig, ax = figure(4.1)
    blank(ax, (0, 10.4), (0.2, 12.4))
    fx, bx, w, h = 0.2, 5.6, 4.4, 1.0
    rows = {"X": 11.0, "Z1": 8.6, "A1": 6.2, "Z2": 3.8, "L": 1.4}
    fwd = [("X", r"$X$  (64×784)"), ("Z1", r"$Z_1=XW_1^\top+b_1$  (64×256)"),
           ("A1", r"$A_1=\mathrm{ReLU}(Z_1)$  (64×256)"), ("Z2", r"$Z_2=A_1W_2^\top+b_2$  (64×10)"),
           ("L", r"$L=\mathrm{CE}(Z_2,y)$, mean  (scalar)")]
    for key, txt in fwd:
        box(ax, (fx, rows[key] - h / 2), w, h, txt, p, fontsize=6.9)
    ops = [("X", "Z1", r"$W_1$: 256×784"), ("Z1", "A1", "ReLU"), ("A1", "Z2", r"$W_2$: 10×256"),
           ("Z2", "L", "softmax + CE")]
    for a, b_, lab in ops:
        arrow(ax, (fx + w / 2, rows[a] - h / 2), (fx + w / 2, rows[b_] + h / 2), p)
        ax.text(fx + w / 2 + 0.12, (rows[a] + rows[b_]) / 2, lab, fontsize=6.6, color=p.muted, va="center")
    ax.text(fx + w / 2, 12.15, "forward (cache Z, A)", ha="center", fontsize=7.5, color=p.fg)
    ax.text(bx + w / 2, 12.15, "backward", ha="center", fontsize=7.5, color=p.label)
    bwd = [("Z2", r"$\delta_2=(P-Y)/64$  (64×10)"), ("A1", r"$\bar A_1=\delta_2W_2$  (64×256)"),
           ("Z1", r"$\delta_1=\bar A_1\odot[Z_1>0]$  (64×256)")]
    for key, txt in bwd:
        box(ax, (bx, rows[key] - h / 2), w, h, txt, p, color=p.label, fontsize=6.9)
    arrow(ax, (bx + w / 2, rows["Z2"] + h / 2), (bx + w / 2, rows["A1"] - h / 2), p, color=p.label)
    arrow(ax, (bx + w / 2, rows["A1"] + h / 2), (bx + w / 2, rows["Z1"] - h / 2), p, color=p.label)
    arrow(ax, (fx + w, rows["L"]), (bx + w / 2, rows["Z2"] - h / 2), p, color=p.label,
          connectionstyle="arc3,rad=0.25")
    # parameter gradients
    ax.text(bx + w / 2, rows["Z2"] + 0.95, r"$\bar W_2=\delta_2^\top A_1$ (10×256),  $\bar b_2=\sum_{\rm rows}\delta_2$",
            ha="center", fontsize=6.5, color=p.label)
    ax.text(bx + w / 2, rows["Z1"] + 0.95, r"$\bar W_1=\delta_1^\top X$ (256×784),  $\bar b_1=\sum_{\rm rows}\delta_1$",
            ha="center", fontsize=6.5, color=p.label)
    # cache reads
    for src, dst_y in [("A1", rows["Z2"] + 0.95), ("Z1", rows["Z1"]), ("X", rows["Z1"] + 0.95)]:
        arrow(ax, (fx + w, rows[src]), (bx - 0.05, dst_y), p, color=p.muted, lw=0.7, linestyle=(0, (2, 2)))
    ax.text(5.0, 0.55, "dashed: cached forward values read by backward", fontsize=6.5, color=p.muted, ha="center")
    return fig


def _net_221():
    x = np.array([2.0, 1.0])
    W1 = np.array([[0.5, 0.0], [-0.25, 1.0]])
    w2 = np.array([1.0, -2.0])
    z1 = W1 @ x
    h = np.maximum(z1, 0)
    z2 = w2 @ h
    pr = 1 / (1 + np.exp(-z2))
    return x, W1, w2, z1, h, z2, pr


@register("fund.mlp-from-scratch", "worked-2-2-1")
def worked_221(p):
    """2-2-1 network (ReLU hidden, sigmoid output, zero biases) with forward values only."""
    x, W1, w2, z1, h, z2, pr = _net_221()
    fig, ax = figure(2.5)
    blank(ax, (0, 12), (0, 7.4))
    xin = [(1.0, 5.4), (1.0, 1.8)]
    hid = [(6.0, 5.4), (6.0, 1.8)]
    out = (10.6, 3.6)
    r = 0.62
    from matplotlib.patches import Circle
    for (cx, cy), val, name in zip(xin, x, ["x_1", "x_2"]):
        ax.add_patch(Circle((cx, cy), r, facecolor=p.surface, edgecolor=p.muted, lw=1))
        ax.text(cx, cy, f"{val:g}", ha="center", va="center", fontsize=8.5, color=p.fg)
        ax.text(cx, cy + r + 0.3, f"${name}$", ha="center", fontsize=8, color=p.muted)
    for k, ((cx, cy), zz, hh) in enumerate(zip(hid, z1, h)):
        ax.add_patch(Circle((cx, cy), r, facecolor=p.surface, edgecolor=p.accent, lw=1.1))
        ax.text(cx, cy, f"{hh:g}", ha="center", va="center", fontsize=8.5, color=p.fg)
        ax.text(cx, cy + r + 0.3, f"$h_{k + 1}$  ($z={zz:g}$)", ha="center", fontsize=7.4, color=p.muted)
    ax.add_patch(Circle(out, r, facecolor=p.surface, edgecolor=p.label, lw=1.1))
    ax.text(*out, f"{pr:g}", ha="center", va="center", fontsize=8.5, color=p.fg)
    ax.text(out[0], out[1] + r + 0.3, f"$p=\\sigma(z_2)$, $z_2={z2:g}$", ha="center", fontsize=7.4,
            color=p.muted)
    # edges with weights
    for i, (sx, sy) in enumerate(xin):
        for j, (tx, ty) in enumerate(hid):
            ax.plot([sx + r, tx - r], [sy, ty], color=p.muted, lw=0.9, zorder=0)
            f = 0.22 if i == j else 0.3
            lx = sx + r + f * (tx - sx - 2 * r)
            ly = sy + f * (ty - sy)
            ax.text(lx, ly, f"{W1[j, i]:g}".replace("-", "\u2212"), fontsize=7.6, color=p.accent, ha="center",
                    va="center", bbox=dict(boxstyle="round,pad=0.15", facecolor=p.surface, edgecolor="none"))
    for j, (sx, sy) in enumerate(hid):
        ax.plot([sx + r, out[0] - r], [sy, out[1]], color=p.muted, lw=0.9, zorder=0)
        lx, ly = sx + r + 0.45 * (out[0] - sx - 2 * r), sy + 0.45 * (out[1] - sy)
        ax.text(lx, ly + 0.3, f"{w2[j]:g}".replace("-", "−"), fontsize=7.6, color=p.accent, ha="center")
    ax.text(6.0, 0.2, "ReLU hidden, sigmoid output, all biases 0", ha="center", fontsize=7, color=p.muted)
    return fig
