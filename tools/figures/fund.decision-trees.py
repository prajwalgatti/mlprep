"""Figures for fund.decision-trees. Trees are actually fitted with sklearn (CART) on synthetic data."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from figures.style import arrow, blank, box, figure, register


def _gini(p):
    return 2 * p * (1 - p)


def _mis(p):
    return np.minimum(p, 1 - p)


def _ent_scaled(p):
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q)) / 2


@register("fund.decision-trees", "impurity-curves")
def impurity_curves(p):
    """Top: the three two-class impurities. Bottom: gain = curve at parent minus chord, Gini vs misclassification."""
    x = np.linspace(0, 1, 400)
    fig, (ax, bx) = figure(4.3, nrows=2)
    ax.plot(x, _mis(x), color=p.c(1), label="misclassification")
    ax.plot(x, _gini(x), color=p.c(0), label="Gini $2p(1-p)$")
    ax.plot(x, _ent_scaled(x), color=p.c(2), label="entropy / 2 (bits)")
    ax.set_xlabel("proportion $p$ of class 1 in the node")
    ax.set_ylabel("impurity")
    ax.set_ylim(0, 0.62)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12), handlelength=1.3, columnspacing=1)

    # One split: parent p=0.25 into two equal-size children with p=0.1 and p=0.4.
    pl, pr, pbar = 0.1, 0.4, 0.25
    for f, col, name, dy in [(_gini, p.c(0), "Gini", 0.0), (_mis, p.c(1), "misclass.", 0.0)]:
        bx.plot(x, f(x), color=col, lw=1.4)
        bx.plot([pl, pr], [f(pl), f(pr)], color=col, lw=1, ls="--")
        bx.plot([pl, pr], [f(pl), f(pr)], "o", color=col, ms=4)
        chord = (f(pl) + f(pr)) / 2
        bx.plot([pbar], [chord], "s", color=col, ms=4)
        bx.plot([pbar], [f(pbar)], "o", color=col, ms=5, mfc="none")
    g_gain = _gini(pbar) - (_gini(pl) + _gini(pr)) / 2
    bx.annotate("", (pbar, _gini(pbar)), (pbar, _gini(pbar) - g_gain),
                arrowprops=dict(arrowstyle="<->", color=p.fg, lw=1, shrinkA=0, shrinkB=0))
    bx.annotate(f"Gini gain {g_gain:.3f}", (pbar - 0.005, _gini(pbar) - g_gain / 2), xytext=(0.03, 0.43),
                color=p.c(0), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=p.c(0), lw=0.7))
    bx.text(0.34, 0.05, "misclass. gain = 0\n(chord on the line)", color=p.c(1), fontsize=7.5, ha="left")
    bx.text(pl, -0.07, "$p_L$", color=p.muted, fontsize=8, ha="center")
    bx.text(pr, -0.07, "$p_R$", color=p.muted, fontsize=8, ha="center")
    bx.text(pbar, -0.07, r"$\bar p$", color=p.muted, fontsize=8, ha="center")
    bx.set_xlim(0, 0.62)
    bx.set_ylim(0, 0.52)
    bx.set_xticks([])
    bx.set_ylabel("impurity")
    bx.set_title("split $\\bar p=0.25 \\to$ (0.1, 0.4), equal sizes", fontsize=8.5)
    return fig


def _toy():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (80, 2))
    y = (((X[:, 0] > 0.55) & (X[:, 1] < 0.65)) | ((X[:, 0] < 0.3) & (X[:, 1] > 0.55))).astype(int)
    flip = rng.uniform(size=80) < 0.06
    y[flip] = 1 - y[flip]
    return X, y


def _layout(tree):
    """x-positions: leaves in in-order sequence; internal nodes centred over children."""
    t = tree.tree_
    pos, order = {}, []

    def rec(n, d):
        if t.children_left[n] == -1:
            order.append(n)
            pos[n] = (len(order) - 1, d)
            return
        rec(t.children_left[n], d + 1)
        rec(t.children_right[n], d + 1)
        pos[n] = ((pos[t.children_left[n]][0] + pos[t.children_right[n]][0]) / 2, d)

    rec(0, 0)
    return pos, order


@register("fund.decision-trees", "tree-and-partition")
def tree_and_partition(p):
    """A 5-leaf CART tree fitted to 80 noisy points, drawn beside the partition it induces."""
    X, y = _toy()
    clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X, y)
    t = clf.tree_
    pos, order = _layout(clf)
    names = {n: f"R{i + 1}" for i, n in enumerate(order)}
    fig, (ax, bx) = figure(5.0, nrows=2, height_ratios=[1, 1.25])
    nleaf = len(order)
    depth = max(d for _, d in pos.values())
    blank(ax, (-0.6, nleaf - 0.4), (-depth - 0.55, 0.45))
    ax.set_aspect("auto")
    for n, (xc, d) in pos.items():
        yc = -d
        if t.children_left[n] != -1:
            for c in (t.children_left[n], t.children_right[n]):
                cx, cd = pos[c]
                arrow(ax, (xc, yc - 0.2), (cx, -cd + 0.2), p, style="-")
            lab = f"x{t.feature[n] + 1} ≤ {t.threshold[n]:.2f}"
            box(ax, (xc - 0.62, yc - 0.2), 1.24, 0.4, lab, p, fontsize=7.5)
        else:
            cls = int(np.argmax(t.value[n][0]))
            col = p.c(0) if cls == 0 else p.c(1)
            box(ax, (xc - 0.42, yc - 0.2), 0.84, 0.4, f"{names[n]}: {'●' if cls == 0 else '▲'}", p,
                color=col, fontsize=7.5)
    ax.text(-0.55, 0.32, "left = condition true", fontsize=7, color=p.muted)

    # partition panel
    gx = np.linspace(0, 1, 300)
    G = np.array(np.meshgrid(gx, gx)).reshape(2, -1).T
    leaf = clf.apply(G).reshape(300, 300)
    pred = clf.predict(G).reshape(300, 300)
    from matplotlib.colors import ListedColormap
    bx.imshow(pred, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap([p.c(0), p.c(1)]),
              alpha=0.16, aspect="auto", interpolation="nearest")
    bx.contour(gx, gx, leaf, levels=np.unique(leaf)[:-1] + 0.5, colors=p.fg, linewidths=0.9)
    for cls, mk, col in [(0, "o", p.c(0)), (1, "^", p.c(1))]:
        m = y == cls
        bx.plot(X[m, 0], X[m, 1], mk, color=col, ms=3.2, alpha=0.9)
    for n in order:
        m = leaf == n
        yy, xx = np.nonzero(m)
        bx.text(gx[int(xx.mean())], gx[int(yy.mean())], names[n], fontsize=8.5, color=p.label,
                ha="center", va="center", fontweight="bold")
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.set_xlabel("$x_1$")
    bx.set_ylabel("$x_2$")
    bx.set_xticks([0, 0.5, 1])
    bx.set_yticks([0, 0.5, 1])
    return fig


@register("fund.decision-trees", "which-leaf")
def which_leaf(p):
    """[fig-Q] A hand-built tree; the question gives a point and asks which leaf it lands in."""
    fig, ax = figure(2.5)
    blank(ax, (0, 10), (-0.45, 4.85))
    ax.set_aspect("auto")
    nodes = {"root": (5.0, 4.4, "x1 ≤ 3"), "L": (2.4, 2.8, "x2 ≤ 2"), "R": (7.6, 2.8, "x2 ≤ 5"),
             "RL": (6.2, 1.3, "x1 ≤ 6")}
    leaves = {"A": (1.3, 1.3), "B": (3.5, 1.3), "C": (5.1, 0.0), "D": (7.3, 0.0), "E": (9.0, 1.3)}
    edges = [("root", "L"), ("root", "R"), ("L", "A"), ("L", "B"), ("R", "RL"), ("R", "E"),
             ("RL", "C"), ("RL", "D")]
    allpos = {k: v[:2] for k, v in nodes.items()} | leaves
    for a, b in edges:
        (x0, y0), (x1, y1) = allpos[a], allpos[b]
        arrow(ax, (x0, y0 - 0.3), (x1, y1 + 0.3), p, style="-")
        side = "yes" if x1 < x0 else "no"
        ax.text((x0 + x1) / 2 + (-0.45 if side == "yes" else 0.45), (y0 + y1) / 2 + 0.1, side,
                fontsize=7, color=p.muted, ha="center", va="center")
    for k, (xc, yc, lab) in nodes.items():
        box(ax, (xc - 0.9, yc - 0.3), 1.8, 0.6, lab, p, fontsize=8)
    for k, (xc, yc) in leaves.items():
        box(ax, (xc - 0.45, yc - 0.3), 0.9, 0.6, k, p, color=p.accent, fontsize=8.5)
    return fig


@register("fund.decision-trees", "axis-aligned")
def axis_aligned(p):
    """CART on a diagonal boundary x2 > x1: 4 leaves vs 32 leaves; both approximate it by a staircase."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = (X[:, 1] > X[:, 0]).astype(int)
    gx = np.linspace(0, 1, 300)
    G = np.array(np.meshgrid(gx, gx)).reshape(2, -1).T
    fig, axes = figure(2.1, ncols=2, sharey=True)
    from matplotlib.colors import ListedColormap
    for ax, leaves in zip(axes, [4, 32]):
        clf = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(X, y)
        pred = clf.predict(G).reshape(300, 300)
        ax.imshow(pred, origin="lower", extent=(0, 1, 0, 1), cmap=ListedColormap([p.c(0), p.c(1)]),
                  alpha=0.22, aspect="auto", interpolation="nearest")
        ax.contour(gx, gx, pred, levels=[0.5], colors=p.fg, linewidths=1.1)
        ax.plot([0, 1], [0, 1], color=p.label, lw=1.2, ls="--")
        Xt = np.random.default_rng(1).uniform(0, 1, (20000, 2))
        te = np.mean(clf.predict(Xt) != (Xt[:, 1] > Xt[:, 0]))
        ax.set_title(f"{leaves} leaves\ntest error {te:.1%}", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].text(0.62, 0.5, "true\nboundary", color=p.label, fontsize=7.2, ha="left", va="top")
    return fig


@register("fund.decision-trees", "pruning-cv")
def pruning_cv(p):
    """10-fold CV error along the cost-complexity path of a CART tree on noisy synthetic data, with the 1-SE choice."""
    rng = np.random.default_rng(2)
    n = 500
    X = rng.uniform(0, 1, (n, 5))
    y = (((X[:, 0] > 0.5) & (X[:, 1] > 0.3)) | (X[:, 2] > 0.8)).astype(int)
    flip = rng.uniform(size=n) < 0.15
    y[flip] = 1 - y[flip]
    full = DecisionTreeClassifier(random_state=0).fit(X, y)
    path = full.cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas
    # geometric midpoints of the alpha sequence, as in CART's CV over alpha
    mids = np.concatenate([[0.0], np.sqrt(np.maximum(alphas[:-1], 1e-12) * alphas[1:])])
    sizes = np.array([DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X, y).get_n_leaves() for a in mids])
    kf = KFold(10, shuffle=True, random_state=0)
    errs = np.zeros((len(mids), 10))
    for j, (tr, va) in enumerate(kf.split(X)):
        for i, a in enumerate(mids):
            c = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X[tr], y[tr])
            errs[i, j] = np.mean(c.predict(X[va]) != y[va])
    m = errs.mean(1)
    se = errs.std(1, ddof=1) / np.sqrt(10)
    keep = sizes >= 2
    sizes, m, se = sizes[keep], m[keep], se[keep]
    i_min = int(np.argmin(m))
    thresh = m[i_min] + se[i_min]
    ok = np.nonzero(m <= thresh)[0]
    i_1se = ok[np.argmin(sizes[ok])]
    fig, ax = figure(2.4)
    ax.errorbar(sizes, m, yerr=se, fmt="o-", color=p.c(0), ms=3, lw=1.2, elinewidth=0.7, capsize=0)
    ax.axhline(thresh, color=p.muted, lw=0.8, ls="--")
    ax.text(sizes.max(), thresh + 0.004, "min + 1 SE", color=p.muted, fontsize=7.5, ha="right", va="bottom")
    ax.plot([sizes[i_min]], [m[i_min]], "o", color=p.label, ms=7, mfc="none", mew=1.4)
    ax.annotate(f"min CV error\n({sizes[i_min]} leaves)", (sizes[i_min], m[i_min]),
                xytext=(sizes[i_min] * 2.0, m[i_min] - 0.012), fontsize=7.5, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.plot([sizes[i_1se]], [m[i_1se]], "*", color=p.good, ms=11, zorder=5)
    ax.annotate(f"1-SE choice\n({sizes[i_1se]} leaves)", (sizes[i_1se], m[i_1se]),
                xytext=(2.1, m[i_1se] + 0.075), fontsize=7.5, color=p.good, ha="left",
                arrowprops=dict(arrowstyle="-", color=p.good, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("leaves |T| of the pruned tree (log scale)")
    ax.set_ylabel("10-fold CV error")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    return fig
