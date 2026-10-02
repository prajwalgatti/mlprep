"""Figures for fund.bias-variance. Reference example for figure authors.

`u-curve` is the minimal pattern to copy: one function per figure, palette colours only, direct labels.
The other three are computed from the real formulas (simulated fits, the exact kNN decomposition).
"""
import numpy as np
from numpy.polynomial import Polynomial

from figures.style import figure, register


@register("fund.bias-variance", "u-curve")
def u_curve(p):
    """Bias², variance and total expected error against model capacity (schematic)."""
    c = np.linspace(0.05, 1, 200)
    bias2 = 0.9 * (1 - c) ** 2.2 + 0.02
    var = 0.05 + 0.85 * c ** 3
    noise = np.full_like(c, 0.12)
    total = bias2 + var + noise
    fig, ax = figure(2.4)
    ax.plot(c, bias2, color=p.c(0), label=r"bias$^2$")
    ax.plot(c, var, color=p.c(1), label="variance")
    ax.plot(c, noise, color=p.muted, lw=1, ls="--", label=r"noise $\sigma^2$")
    ax.plot(c, total, color=p.fg, lw=2.2, label="expected test error")
    i = total.argmin()
    ax.plot([c[i]], [total[i]], "o", color=p.label, ms=5, zorder=5)
    ax.annotate("sweet spot", (c[i], total[i]), xytext=(c[i] - 0.05, total[i] + 0.35),
                color=p.label, fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.set_xlabel("model capacity →")
    ax.set_ylabel("error")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.text(0.04, 1.17, "underfit", color=p.muted, fontsize=7.5)
    ax.text(0.97, 1.17, "overfit", color=p.muted, fontsize=7.5, ha="right")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95), handlelength=1.4, columnspacing=1)
    return fig


def _truth(x):
    return np.sin(2 * np.pi * x)


@register("fund.bias-variance", "fits-grid")
def fits_grid(p):
    """Polynomial fits at degrees 1, 3 and 15 on a fixed design of 20 inputs; only the noise is resampled.

    Least squares is linear in y, so the mean fit E_D[f_hat] is exactly the fit to the noiseless f(x).
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 20)
    xs = np.linspace(0, 1, 200)
    fig, axes = figure(2.0, ncols=3, sharey=True)
    for ax, deg in zip(axes, [1, 3, 15]):
        for _ in range(20):
            y = _truth(x) + rng.normal(0, 0.3, x.size)
            ax.plot(xs, Polynomial.fit(x, y, deg)(xs), color=p.c(0), lw=0.6, alpha=0.5)
        ax.plot(xs, _truth(xs), color=p.fg, lw=1.2, ls="--", label="truth $f$")
        mean_fit = Polynomial.fit(x, _truth(x), deg)(xs)
        ax.plot(xs, mean_fit, color=p.label, lw=1.8, label=r"mean fit $\bar f$")
        ax.set_title(f"degree {deg}", fontsize=8.5)
        ax.set_ylim(-1.9, 1.9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].plot([], [], color=p.c(0), lw=0.8, label="single fits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, handlelength=1.4, columnspacing=1.2)
    return fig


@register("fund.bias-variance", "knn-tradeoff")
def knn_tradeoff(p):
    """Exact kNN decomposition on a fixed design: bias² from the neighbours' f, variance σ²/k."""
    x_train = np.linspace(0, 1, 50)
    sigma2 = 0.09
    x_test = np.linspace(0.02, 0.98, 300)
    ks = np.arange(1, 41)
    order = np.argsort(np.abs(x_test[:, None] - x_train[None, :]), axis=1)
    bias2 = [np.mean((_truth(x_test) - _truth(x_train[order[:, :k]]).mean(axis=1)) ** 2) for k in ks]
    bias2 = np.array(bias2)
    var = sigma2 / ks
    total = sigma2 + bias2 + var
    fig, ax = figure(2.3)
    ax.plot(ks, bias2, color=p.c(0), label=r"bias$^2$")
    ax.plot(ks, var, color=p.c(1), label=r"variance $\sigma^2/k$")
    ax.axhline(sigma2, color=p.muted, lw=1, ls="--", label=r"noise $\sigma^2$")
    ax.plot(ks, total, color=p.fg, lw=2.2, label="expected error")
    i = total.argmin()
    ax.plot([ks[i]], [total[i]], "o", color=p.label, ms=5, zorder=5)
    ax.annotate(f"best k = {ks[i]}", (ks[i], total[i]), xytext=(ks[i] + 6, total[i] + 0.07),
                color=p.label, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.set_xlabel(r"neighbours $k$  (more flexible ←)")
    ax.set_ylabel("error at a test point (avg)")
    ax.set_ylim(0, 0.42)
    ax.set_xlim(0, 40)
    ax.legend(loc="upper center", ncol=2, handlelength=1.4, columnspacing=1, bbox_to_anchor=(0.5, 1.02))
    return fig


@register("fund.bias-variance", "learning-curves")
def learning_curves(p):
    """[fig-Q] Train and validation MSE vs N for a degree-1 and a degree-12 polynomial on a sine."""
    rng = np.random.default_rng(1)
    ns = np.array([20, 25, 32, 40, 50, 65, 80, 100, 130, 160, 200])
    x_val = rng.uniform(0, 1, 2000)
    y_val = _truth(x_val) + rng.normal(0, 0.3, 2000)
    fig, axes = figure(2.0, ncols=2, sharey=True)
    for ax, deg, name in zip(axes, [12, 1], ["A", "B"]):
        tr, va = [], []
        for n in ns:
            t, v = [], []
            for _ in range(150):
                x = rng.uniform(0, 1, n)
                y = _truth(x) + rng.normal(0, 0.3, n)
                X = np.vander(2 * x - 1, deg + 1)
                w = np.linalg.solve(X.T @ X + 1e-6 * np.eye(deg + 1), X.T @ y)
                t.append(np.mean((X @ w - y) ** 2))
                v.append(np.mean((np.vander(2 * x_val - 1, deg + 1) @ w - y_val) ** 2))
            tr.append(np.mean(t))
            va.append(np.median(v))
        ax.plot(ns, tr, color=p.c(2), label="train")
        ax.plot(ns, va, color=p.c(1), label="validation")
        ax.axhline(0.09, color=p.muted, lw=0.9, ls="--")
        ax.set_title(name, fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200])
        ax.set_xticklabels(["20", "50", "100", "200"])
        ax.set_xlabel("training set size N")
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("MSE")
    axes[0].text(150, 0.105, r"$\sigma^2$", color=p.muted, fontsize=7.5)
    axes[0].legend(loc="upper right", handlelength=1.3)
    return fig
