"""Figures for fund.initialization.

Every curve is simulated: random MLPs of width 256 are pushed forward on Gaussian inputs, and a tiny
tanh network is trained by plain gradient descent for the symmetry figure.
"""
from functools import lru_cache

import numpy as np

from figures.style import figure, register

WIDTH, DEPTH, BATCH = 256, 20, 2000


def _relu(z):
    return np.maximum(z, 0)


def _forward(act, var, uniform=False, seed=0, keep="pre"):
    """Push N(0, I) inputs through DEPTH square layers with weight variance `var`.

    Returns per-layer std of the pre-activations (keep="pre") or of the activations (keep="post"),
    plus the activation samples of layers 1, 5, 10, 20 for histograms.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((BATCH, WIDTH))
    stds, samples = [], {}
    for layer in range(1, DEPTH + 1):
        if uniform:
            a = np.sqrt(3 * var)              # Var U(-a, a) = a^2 / 3
            W = rng.uniform(-a, a, (WIDTH, WIDTH))
        else:
            W = rng.normal(0, np.sqrt(var), (WIDTH, WIDTH))
        y = h @ W
        h = act(y)
        stds.append(y.std() if keep == "pre" else h.std())
        if layer in (1, 5, 10, 20):
            samples[layer] = h[:200].ravel()
    return np.array(stds), samples


@lru_cache(maxsize=None)
def _depth_curves():
    n = WIDTH
    tanh = {
        r"$U(\pm 1/\sqrt{n})$ (old)": _forward(np.tanh, 1 / (3 * n), uniform=True, keep="post")[0],
        "Xavier": _forward(np.tanh, 1 / n, uniform=True, keep="post")[0],
        r"$\mathcal{N}(0, 4/n)$": _forward(np.tanh, 4 / n, keep="post")[0],
    }
    relu = {
        "std 0.01": _forward(_relu, 1e-4)[0],
        "Xavier": _forward(_relu, 1 / n)[0],
        "He": _forward(_relu, 2 / n)[0],
        "std 0.1": _forward(_relu, 1e-2)[0],
    }
    return tanh, relu


@register("fund.initialization", "activation-std-depth")
def activation_std_depth(p):
    """Std through 20 tanh layers (activations) and 20 ReLU layers (pre-activations), width 256."""
    tanh, relu = _depth_curves()
    layers = np.arange(1, DEPTH + 1)
    fig, (a1, a2) = figure(2.5, ncols=2)
    colors = [p.c(1), p.c(0), p.c(3)]
    for (name, s), col in zip(tanh.items(), colors):
        a1.plot(layers, np.maximum(s, 1e-9), color=col, lw=1.6)
    a1.set_yscale("log")
    a1.set_ylim(1e-6, 3)
    a1.set_title("tanh layers", fontsize=8.5)
    a1.text(20, 0.9, "N(0,4/n): saturated", color=p.c(3), fontsize=7, ha="right")
    a1.text(20, 0.06, "Xavier", color=p.c(0), fontsize=7.5, ha="right")
    a1.text(7.5, 2e-5, r"$U(\pm1/\sqrt{n})$", color=p.c(1), fontsize=7.5)
    colors = [p.c(1), p.c(0), p.c(2), p.c(3)]
    for (name, s), col in zip(relu.items(), colors):
        a2.plot(layers, np.maximum(s, 1e-12), color=col, lw=1.6)
    a2.set_yscale("log")
    a2.set_ylim(1e-6, 1e2)
    a2.set_title("ReLU layers", fontsize=8.5)
    a2.text(20, 0.35, "He", color=p.c(2), fontsize=7.5, ha="right", va="top")
    a2.text(20, 22, "std 0.1", color=p.c(3), fontsize=7.5, ha="right", va="bottom")
    a2.text(11, 1.5e-3, "Xavier", color=p.c(0), fontsize=7.5, ha="center", va="top")
    a2.text(6.2, 1e-5, "std 0.01", color=p.c(1), fontsize=7.5)
    for ax in (a1, a2):
        ax.set_xlabel("layer")
        ax.set_xticks([1, 10, 20])
    a1.set_ylabel("std (log scale)")
    return fig


@lru_cache(maxsize=None)
def _quiz_curves():
    n = WIDTH
    # Order chosen so the letters don't follow the variance order.
    specs = [("A", 2 / n), ("B", 1 / (3 * n)), ("C", 3 / n), ("D", 1 / n)]
    return [(name, _forward(_relu, v, seed=3)[0]) for name, v in specs]


@register("fund.initialization", "which-init")
def which_init(p):
    """[fig-Q] Four inits of a 20-layer ReLU MLP (width 256, zero biases). Letters only."""
    layers = np.arange(1, DEPTH + 1)
    fig, ax = figure(2.3)
    for i, (name, s) in enumerate(_quiz_curves()):
        s = np.maximum(s, 1e-12)
        ax.plot(layers, s, color=p.c(i), lw=1.7)
        ax.text(20.4, s[-1], name, color=p.c(i), fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e4)
    ax.set_xlim(1, 21.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("layer")
    ax.set_ylabel("std of pre-activations")
    return fig


@lru_cache(maxsize=None)
def _hists():
    n = WIDTH
    return [
        (r"$U(\pm1/\sqrt{n})$", _forward(np.tanh, 1 / (3 * n), uniform=True, keep="post")[1]),
        ("Xavier", _forward(np.tanh, 1 / n, uniform=True, keep="post")[1]),
        (r"$\mathcal{N}(0,4/n)$", _forward(np.tanh, 4 / n, keep="post")[1]),
    ]


@register("fund.initialization", "tanh-histograms")
def tanh_histograms(p):
    """Histograms of tanh activations at layers 1, 5, 10, 20 for three inits (after Glorot Fig. 6)."""
    fig, axes = figure(2.6, ncols=3, sharey=True)
    bins = np.linspace(-1, 1, 41)
    offsets = {1: 3, 5: 2, 10: 1, 20: 0}
    for ax, (title, samples) in zip(axes, _hists()):
        for layer, off in offsets.items():
            counts, _ = np.histogram(samples[layer], bins=bins)
            dens = counts / counts.max() * 0.85
            centers = 0.5 * (bins[1:] + bins[:-1])
            ax.fill_between(centers, off, off + dens, step="mid", color=p.c(0), alpha=0.35, lw=0)
            ax.step(centers, off + dens, where="mid", color=p.c(0), lw=0.9)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([-1, 0, 1])
        ax.set_xlim(-1.05, 1.05)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    axes[0].set_yticks([0.4, 1.4, 2.4, 3.4])
    axes[0].set_yticklabels(["L20", "L10", "L5", "L1"])
    axes[1].set_xlabel("activation value")
    return fig


@lru_cache(maxsize=None)
def _symmetry_runs():
    """Train a 2-4-1 tanh net on XOR-like targets with GD; record each hidden unit's first incoming weight."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float) * 2 - 1
    y = np.array([0, 1, 1, 0], float) * 2 - 1
    out = {}
    for name in ("constant", "random"):
        rng = np.random.default_rng(1)
        if name == "constant":
            W1 = np.full((2, 4), 0.5)
            v = np.full(4, 0.5)
        else:
            W1 = rng.normal(0, 0.7, (2, 4))
            v = rng.normal(0, 0.5, 4)
        b1 = np.zeros(4)
        traj = []
        for _ in range(400):
            a = X @ W1 + b1
            h = np.tanh(a)
            pred = h @ v
            err = (pred - y) / len(y)              # d(MSE/2)/d pred
            dv = h.T @ err
            dh = np.outer(err, v) * (1 - h ** 2)
            dW1 = X.T @ dh
            db1 = dh.sum(0)
            W1 -= 0.5 * dW1
            b1 -= 0.5 * db1
            v -= 0.5 * dv
            traj.append(W1[0].copy())
        out[name] = np.array(traj)
    return out


@register("fund.initialization", "symmetry")
def symmetry(p):
    """First incoming weight of each of 4 hidden units during GD, constant vs random init."""
    runs = _symmetry_runs()
    fig, axes = figure(2.2, ncols=2, sharey=True)
    for ax, name in zip(axes, ("constant", "random")):
        tr = runs[name]
        for j in range(4):
            ax.plot(tr[:, j], color=p.c(j), lw=1.5, ls="-" if name == "random" else ["-", "--", ":", "-."][j])
        ax.set_title("all weights 0.5" if name == "constant" else "random init", fontsize=8.5)
        ax.set_xlabel("GD step")
        ax.set_xticks([0, 200, 400])
    axes[0].set_ylabel(r"weight $w_{1j}$, units $j=1..4$")
    axes[0].text(200, axes[0].get_ylim()[0] + 0.1, "4 curves, exactly on top", color=p.muted,
                 fontsize=7, ha="center", va="bottom")
    return fig
