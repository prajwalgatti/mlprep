"""Figures for fund.bagging-random-forests. All models are actually fitted with sklearn on synthetic data."""
import warnings

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

from figures.style import figure, register

warnings.filterwarnings("ignore", category=UserWarning)


@register("fund.bagging-random-forests", "bagged-boundary")
def bagged_boundary(p):
    """One fully grown tree vs the average of 100 bootstrap trees on noisy two-moons data."""
    X, y = make_moons(n_samples=300, noise=0.32, random_state=2)
    Xt, yt = make_moons(n_samples=20000, noise=0.32, random_state=7)
    gx = np.linspace(-1.6, 2.6, 260)
    gy = np.linspace(-1.2, 1.7, 200)
    G = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0).fit(X, y)
    fig, axes = figure(2.3, ncols=2, sharey=True)
    for ax, model, name in zip(axes, [tree, bag], ["1 deep tree", "bag of 100 trees"]):
        prob = model.predict_proba(G)[:, 1].reshape(len(gy), len(gx))
        ax.contourf(gx, gy, prob, levels=[0, 0.5, 1], colors=[p.c(0), p.c(1)], alpha=0.2)
        ax.contour(gx, gy, prob, levels=[0.5], colors=p.fg, linewidths=0.9)
        for cls, mk, col in [(0, "o", p.c(0)), (1, "^", p.c(1))]:
            m = y == cls
            ax.plot(X[m, 0], X[m, 1], mk, color=col, ms=2.2, alpha=0.85)
        err = np.mean(model.predict(Xt) != yt)
        ax.set_title(f"{name}\ntest error {err:.1%}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])
    return fig


@register("fund.bagging-random-forests", "oob-vs-test")
def oob_vs_test(p):
    """OOB error and test error of a random forest vs number of trees (500 training points, 5000 test; mean of 3 seeds)."""
    X, y = make_classification(n_samples=5500, n_features=20, n_informative=6, n_redundant=4,
                               flip_y=0.08, random_state=1)
    Xtr, ytr, Xte, yte = X[:500], y[:500], X[500:], y[500:]
    Bs = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    oob, te = [], []
    for seed in range(3):
        rf = RandomForestClassifier(n_estimators=5, warm_start=True, oob_score=True, random_state=seed, n_jobs=-1)
        o, t = [], []
        for B in Bs:
            rf.set_params(n_estimators=B)
            rf.fit(Xtr, ytr)
            o.append(1 - rf.oob_score_)
            t.append(1 - rf.score(Xte, yte))
        oob.append(o)
        te.append(t)
    oob, te = np.mean(oob, 0), np.mean(te, 0)
    fig, ax = figure(2.3)
    ax.plot(Bs, oob, "o-", color=p.c(1), ms=3, label="OOB error")
    ax.plot(Bs, te, "o-", color=p.c(0), ms=3, label="test error")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500"])
    ax.set_xlabel("number of trees $B$ (log scale)")
    ax.set_ylabel("error")
    ax.set_ylim(0.1, 0.27)
    ax.text(11, 0.255, "OOB: each point gets\nonly ~37% of the votes", color=p.c(1), fontsize=7.5, va="top")
    ax.text(480, 0.103, "no upturn as $B$ grows", color=p.muted, fontsize=7.5, ha="right", va="bottom")
    ax.legend(loc="upper right", handlelength=1.3)
    return fig


@register("fund.bagging-random-forests", "m-tradeoff")
def m_tradeoff(p):
    """[fig-Q] Test error vs m (features tried per split): 50 features, 5 informative, 600 training points, 300 trees."""
    X, y = make_classification(n_samples=5500, n_features=50, n_informative=5, n_redundant=0,
                               flip_y=0.05, random_state=3)
    Xtr, ytr, Xte, yte = X[:600], y[:600], X[600:], y[600:]
    ms = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    errs = []
    for m in ms:
        errs.append(np.mean([1 - RandomForestClassifier(300, max_features=m, random_state=s, n_jobs=-1)
                             .fit(Xtr, ytr).score(Xte, yte) for s in range(3)]))
    fig, ax = figure(2.3)
    ax.plot(ms, errs, "o-", color=p.c(0), ms=3.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("features tried per split $m$ (log scale, $p=50$)")
    ax.set_ylabel("test error")
    ax.set_ylim(0.17, 0.28)
    return fig


@register("fund.bagging-random-forests", "importance-bias")
def importance_bias(p):
    """MDI vs permutation importance (held-out) with a random row-ID column and a random binary column added."""
    rng = np.random.default_rng(0)
    n = 1000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x3 = rng.integers(0, 2, n)
    logit = 1.5 * x1 + 1.0 * x2 + 1.0 * x3 - 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    row_id = rng.permutation(n).astype(float)
    junk_bin = rng.integers(0, 2, n)
    X = np.c_[x1, x2, x3, row_id, junk_bin]
    names = ["$x_1$ real", "$x_2$ real", "$x_3$ real, 0/1", "row ID, noise", "0/1 noise"]
    tr = np.arange(n) < 600
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    mdi = rf.feature_importances_
    perm = permutation_importance(rf, X[~tr], y[~tr], n_repeats=20, random_state=0).importances_mean
    fig, axes = figure(2.3, ncols=2, sharey=True)
    ypos = np.arange(5)[::-1]
    cols = [p.c(0), p.c(0), p.c(0), p.bad, p.bad]
    for ax, vals, title in zip(axes, [mdi, perm], ["MDI (train)", "permutation (test)"]):
        ax.barh(ypos, vals, color=cols, height=0.6)
        ax.axvline(0, color=p.muted, lw=0.8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names, fontsize=7.5)
    axes[0].set_xlabel("impurity share")
    axes[1].set_xlabel("accuracy drop")
    return fig
