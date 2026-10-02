"""Figures for fund.training-loop. All curves come from small real simulations (numpy), not hand-drawn shapes."""
from functools import lru_cache

import numpy as np

from figures.style import arrow, blank, box, figure, register


@register("fund.training-loop", "loop-diagram")
def loop_diagram(p):
    """One iteration of a PyTorch training loop, with what each line does to p.grad and the parameters."""
    steps = [
        ("x, y = next(loader)", "fresh shuffled batch"),
        ("logits = model(x)", "records graph + activations"),
        ("loss = loss_fn(logits, y)", "raw logits in, scalar out"),
        ("opt.zero_grad()", "p.grad ← None"),
        ("loss.backward()", r"p.grad ← $\nabla L$, graph freed"),
        ("clip_grad_norm_(ps, c)", r"rescale if $\|g\|>c$, return $\|g\|$"),
        ("opt.step()", r"update p from p.grad + state"),
        ("sched.step()", r"set next LR $\eta$"),
        ("log(loss, gnorm, lr)", "unclipped norm, current LR"),
    ]
    n = len(steps)
    fig, ax = figure(3.9)
    blank(ax, (0, 11.6), (-0.3, n * 1.25 + 0.4))
    bw, bh = 5.0, 0.82
    for i, (code, eff) in enumerate(steps):
        y = (n - 1 - i) * 1.25 + 0.3
        col = p.label if i in (3, 4, 5, 6) else None
        box(ax, (0.9, y), bw, bh, code, p, color=col, fontsize=6.3, mono=True)
        ax.text(6.15, y + bh / 2, eff, fontsize=6.3, color=p.label if col else p.muted, va="center")
        if i < n - 1:
            arrow(ax, (0.9 + bw / 2, y), (0.9 + bw / 2, y - 1.25 + bh), p)
    top = (n - 1) * 1.25 + 0.3 + bh / 2
    bot = 0.3 + bh / 2
    ax.plot([0.9, 0.35, 0.35], [bot, bot, top], color=p.accent, lw=1.0)
    arrow(ax, (0.35, top), (0.9, top), p, color=p.accent)
    ax.text(0.12, (top + bot) / 2, "next step", rotation=90, fontsize=7, color=p.accent, va="center", ha="center")
    ax.text(6.15, n * 1.25 + 0.1, "effect", fontsize=7.2, color=p.fg)
    ax.text(0.9 + bw / 2, n * 1.25 + 0.1, "code", fontsize=7.2, color=p.fg, ha="center")
    return fig


# ---------- symptom curves: real runs on a quadratic and on least squares ----------
_LAM = np.array([10.0, 1.0, 0.1])


def _quad_run(eta, steps, mode="gd", th0=(1e-3, 1.0, 1.0)):
    th = np.array(th0, float)
    acc = np.zeros(3)
    out = []
    for _ in range(steps):
        g = _LAM * th
        if mode == "nozero":       # .grad never cleared: the step uses the running sum of all gradients
            acc += g
            th = th - eta * acc
        else:
            th = th - eta * g
        out.append(0.5 * np.sum(_LAM * th ** 2))
    return np.array(out)


@lru_cache(maxsize=None)
def _overfit_run():
    rng = np.random.default_rng(3)
    n, d = 40, 60
    s = 1 / np.arange(1, d + 1)
    Xtr = rng.normal(size=(n, d)) * s
    Xva = rng.normal(size=(2000, d)) * s
    wt = np.zeros(d)
    wt[:3] = [2, -1.5, 1]
    ytr = Xtr @ wt + rng.normal(0, 0.5, n)
    yva = Xva @ wt + rng.normal(0, 0.5, 2000)
    eta = 0.5 / np.linalg.eigvalsh(Xtr.T @ Xtr / n).max()
    w = np.zeros(d)
    ts, tr, va = [], [], []
    for t in range(3001):
        if t % 15 == 0:
            ts.append(t)
            tr.append(np.mean((Xtr @ w - ytr) ** 2))
            va.append(np.mean((Xva @ w - yva) ** 2))
        w -= eta * Xtr.T @ (Xtr @ w - ytr) / n
    return np.array(ts), np.array(tr), np.array(va)


@register("fund.training-loop", "symptom-curves")
def symptom_curves(p):
    """Four training-loss pathologies from real runs (synthetic problems)."""
    fig, axes = figure(2.9, nrows=2, ncols=2)
    a = _quad_run(0.21, 95)
    b = _quad_run(0.002, 95)
    d = _quad_run(0.02, 300, "nozero", th0=(1.0, 1.0, 1.0))
    ts, tr, va = _overfit_run()
    ax = axes[0, 0]
    ax.semilogy(a, color=p.bad, lw=1.4)
    ax.set_title("A: LR too high", fontsize=7.8)
    ax = axes[0, 1]
    ax.plot(b, color=p.accent, lw=1.4)
    ax.set_ylim(0, 0.6)
    ax.set_title("B: LR too low", fontsize=7.8)
    ax = axes[1, 0]
    ax.plot(ts, tr, color=p.muted, lw=1.2, ls="--")
    ax.plot(ts, va, color=p.accent, lw=1.4)
    ax.set_ylim(0, 1.0)
    ax.text(0.97, 0.9, "val", transform=ax.transAxes, ha="right", fontsize=7, color=p.accent)
    ax.text(0.97, 0.12, "train (dashed)", transform=ax.transAxes, ha="right", fontsize=7, color=p.muted)
    ax.set_title("C: overfitting", fontsize=7.8)
    ax = axes[1, 1]
    ax.plot(d, color=p.c(1), lw=1.1)
    ax.set_title("D: zero_grad missing", fontsize=7.8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
    axes[1, 0].set_xlabel("step", fontsize=7.5)
    axes[1, 1].set_xlabel("step", fontsize=7.5)
    axes[0, 0].set_ylabel("loss (log)", fontsize=7.5)
    axes[1, 0].set_ylabel("loss", fontsize=7.5)
    return fig


# ---------- overfit one batch: correct CE vs softmax applied before CE ----------
@lru_cache(maxsize=None)
def _one_batch_runs(steps=600, lr=0.05):
    out = {}
    for mode in ("correct", "double"):
        rng = np.random.default_rng(0)
        B, d, hdim, K = 8, 20, 64, 10
        X = rng.normal(size=(B, d))
        y = rng.integers(0, K, B)
        W1 = rng.normal(0, np.sqrt(2 / d), (hdim, d))
        b1 = np.zeros(hdim)
        W2 = rng.normal(0, 0.01, (K, hdim))
        b2 = np.zeros(K)
        losses = []
        for _ in range(steps):
            Z1 = X @ W1.T + b1
            A1 = np.maximum(Z1, 0)
            z = A1 @ W2.T + b2
            s = np.exp(z - z.max(1, keepdims=True))
            s /= s.sum(1, keepdims=True)
            Y = np.eye(K)[y]
            if mode == "correct":
                losses.append(-np.log(s[np.arange(B), y]).mean())
                dz = (s - Y) / B
            else:   # cross_entropy(softmax(z)): the probabilities are treated as logits
                q = np.exp(s - s.max(1, keepdims=True))
                q /= q.sum(1, keepdims=True)
                losses.append(-np.log(q[np.arange(B), y]).mean())
                g = (q - Y) / B
                dz = s * (g - (s * g).sum(1, keepdims=True))
            dW2, db2 = dz.T @ A1, dz.sum(0)
            d1 = (dz @ W2) * (Z1 > 0)
            dW1, db1 = d1.T @ X, d1.sum(0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        out[mode] = np.array(losses)
    return out


@register("fund.training-loop", "overfit-one-batch")
def overfit_one_batch(p):
    """Loss on one batch of 8 examples, 10 classes: correct CE vs CE applied to softmax outputs."""
    runs = _one_batch_runs()
    fig, ax = figure(2.35)
    ax.plot(runs["correct"], color=p.good, lw=1.6)
    ax.plot(runs["double"], color=p.bad, lw=1.6)
    floor = np.log(1 + 9 / np.e)
    ax.axhline(np.log(10), color=p.muted, lw=0.7, ls=":")
    ax.axhline(floor, color=p.bad, lw=0.7, ls=":")
    ax.text(590, np.log(10) + 0.06, r"$\ln 10=2.30$ (chance, at init)", fontsize=7, color=p.muted, ha="right",
            va="bottom")
    ax.text(590, floor + 0.06, r"floor $\ln(1+9/e)=1.46$", fontsize=7, color=p.bad, ha="right", va="bottom")
    ax.text(120, 0.25, "correct: reaches ≈0", fontsize=7.3, color=p.good)
    ax.text(120, 1.75, "softmax before cross_entropy", fontsize=7.3, color=p.bad)
    ax.set_ylim(0, 2.6)
    ax.set_xlim(0, 600)
    ax.set_xlabel("step (same 8 examples every step)")
    ax.set_ylabel("training loss")
    return fig


# ---------- mystery curve (question stem): class-sorted data, never shuffled ----------
@lru_cache(maxsize=None)
def _sorted_run():
    rng = np.random.default_rng(1)
    K, per, d, B = 3, 200, 2, 20
    means = np.array([[2, 0], [-1, 1.7], [-1, -1.7]]) * 0.9
    X = np.concatenate([rng.normal(means[k], 1.0, (per, d)) for k in range(K)])
    y = np.repeat(np.arange(K), per)          # sorted by class; the loader never shuffles
    W = np.zeros((K, d))
    b = np.zeros(K)
    losses = []
    for _ in range(4):
        for i in range(0, len(y), B):
            xb, yb = X[i:i + B], y[i:i + B]
            z = xb @ W.T + b
            s = np.exp(z - z.max(1, keepdims=True))
            s /= s.sum(1, keepdims=True)
            losses.append(-np.log(s[np.arange(len(yb)), yb]).mean())
            g = s.copy()
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            W -= 1.0 * g.T @ xb
            b -= 1.0 * g.sum(0)
    return np.array(losses), len(y) // B


@register("fund.training-loop", "mystery-curve")
def mystery_curve(p):
    """Per-step training loss of a 3-class model; dotted lines mark epoch boundaries."""
    losses, spe = _sorted_run()
    fig, ax = figure(2.2)
    ax.plot(losses, color=p.accent, lw=1.2)
    for e in range(1, 4):
        ax.axvline(e * spe, color=p.muted, lw=0.7, ls=":")
    for e in range(4):
        ax.text(e * spe + spe / 2, ax.get_ylim()[1] * 0.97 if False else losses.max() * 1.02,
                f"epoch {e + 1}", ha="center", fontsize=7, color=p.muted)
    ax.set_ylim(0, losses.max() * 1.15)
    ax.set_xlim(0, len(losses))
    ax.set_xlabel("step")
    ax.set_ylabel("minibatch loss")
    return fig
