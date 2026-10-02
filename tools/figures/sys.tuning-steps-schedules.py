"""Figures for sys.tuning-steps-schedules."""
import numpy as np

from figures.style import figure, register

H = np.logspace(-2, 0, 30)   # Hessian eigenvalues of the noisy quadratic


def expected_loss(etas, sigma2=1.0, B=1.0, x0=10.0):
    """Exact E[L_t] for SGD on L = 1/2 sum h_i x_i^2 with gradient noise N(0, sigma2/B) per coordinate."""
    m = np.full_like(H, x0)        # E[x]
    v = np.zeros_like(H)           # Var[x]
    out = []
    for eta in etas:
        out.append(0.5 * np.sum(H * (m ** 2 + v)))
        m = (1 - eta * H) * m
        v = (1 - eta * H) ** 2 * v + eta ** 2 * sigma2 / B
    return np.array(out)


def _noise_floor(p, label=True):
    """High constant LR: fast then stuck at a high floor. Low LR: low floor but slow. Decay gets both."""
    T = 3000
    t = np.arange(T)
    hi, lo = 0.5, 0.02
    cos = 0.5 * hi * (1 + np.cos(np.pi * t / T))
    runs = [(np.full(T, hi), p.c(1), "constant, high"), (np.full(T, lo), p.c(2), "constant, low"),
            (cos, p.accent, "cosine decay (high → 0)")]
    fig, ax = figure(2.45)
    for etas, col, lab in runs:
        ax.semilogy(t, expected_loss(etas), color=col, lw=1.8, label=lab)
    for eta, col in [(hi, p.c(1)), (lo, p.c(2))]:
        floor = np.sum(eta / (2 * (2 - eta * H)))
        ax.axhline(floor, color=col, lw=0.8, ls=":")
    if label:
        ax.text(T * 0.99, np.sum(hi / (2 * (2 - hi * H))) * 1.15, r"floor $\propto\eta\sigma^2/B$",
                color=p.c(1), fontsize=7.2, ha="right", va="bottom")
    ax.set_xlabel("step")
    ax.set_ylabel("expected loss (log)")
    ax.set_xlim(0, T)
    ax.legend(loc="upper right", handlelength=1.4, bbox_to_anchor=(1.0, 0.93))
    return fig


@register("sys.tuning-steps-schedules", "noise-floor")
def noise_floor(p):
    return _noise_floor(p)


@register("sys.tuning-steps-schedules", "noise-floor-quiz")
def noise_floor_quiz(p):
    """Same figure without the floor label, for a question stem."""
    return _noise_floor(p, label=False)


def _warm(t, w):
    return np.minimum(1.0, (t + 1) / w)


@register("sys.tuning-steps-schedules", "schedule-families")
def schedule_families(p):
    """Common LR schedules over a 10k-step run, each with 500 steps of linear warmup."""
    T, w = 10000, 500
    t = np.arange(T)
    frac = np.clip((t - w) / (T - w), 0, 1)
    sched = {
        "constant": np.ones(T),
        "step (×0.1 at 50%, 75%)": np.where(t < 0.5 * T, 1, np.where(t < 0.75 * T, 0.1, 0.01)),
        "linear decay": 1 - frac,
        "cosine": 0.5 * (1 + np.cos(np.pi * frac)),
        "inverse sqrt": np.minimum(1.0, np.sqrt(w / np.maximum(t, 1))),
    }
    fig, ax = figure(2.5)
    labels_at = {"constant": (7600, 1.04), "step (×0.1 at 50%, 75%)": (5150, 0.15),
                 "linear decay": (6300, 0.47), "cosine": (5600, 0.66), "inverse sqrt": (7300, 0.30)}
    for i, (name, s) in zip([0, 1, 2, 3, None], sched.items()):
        col = p.fg if i is None else p.c(i)
        ax.plot(t, s * _warm(t, w), color=col, lw=1.6, ls="--" if i is None else "-")
        x, y = labels_at[name]
        ax.text(x, y, name, color=col, fontsize=7.2)
    ax.axvspan(0, w, color=p.faint, alpha=0.7, lw=0)
    ax.set_xlabel("step")
    ax.set_ylabel("LR / peak LR")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, T)
    ax.set_xticks([0, 2500, 5000, 7500, 10000])
    ax.set_xticklabels(["0", "2.5k", "5k", "7.5k", "10k"])
    return fig


@register("sys.tuning-steps-schedules", "round2-extension")
def round2_extension(p):
    """Extending a 10k-step Round 1 schedule to a 30k-step Round 2 run."""
    T1, T2, w = 10000, 30000, 1000
    fig, axes = figure(3.0, nrows=3, sharex=True)
    t1, t2 = np.arange(T1), np.arange(T2)

    def rsqrt(t):
        return np.minimum(1.0, np.sqrt(w / np.maximum(t, 1))) * np.minimum(1.0, (t + 1) / w)

    def lin(t, T, decay=5000):
        start = T - decay
        return np.clip(1 - (t - start) / decay, 0, 1) * np.minimum(1.0, (t + 1) / w)

    def cos(t, T):
        frac = np.clip((t - w) / (T - w), 0, 1)
        return 0.5 * (1 + np.cos(np.pi * frac)) * np.minimum(1.0, (t + 1) / w)

    rows = [("inverse sqrt: extended tail sits\nbelow Round 1's final LR", rsqrt(t1), rsqrt(t2)),
            ("linear: keep the 5k decay,\nextend the constant phase", lin(t1, T1), lin(t2, T2)),
            ("cosine: same peak LR,\nstretch to the new horizon", cos(t1, T1), cos(t2, T2))]
    for ax, (title, r1, r2) in zip(axes, rows):
        ax.plot(t2, r2, color=p.accent, lw=1.6)
        ax.plot(t1, r1, color=p.c(1), lw=1.3, ls="--")
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 1])
        ax.text(29500, 0.98, title, fontsize=7, color=p.fg, ha="right", va="top")
    floor = np.sqrt(w / T1)
    axes[0].axhline(floor, color=p.label, lw=0.8, ls=":")
    axes[0].fill_between(t2, 0, rsqrt(t2), where=t2 >= T1, color=p.label, alpha=0.18, lw=0)
    axes[2].set_xticks([0, 10000, 20000, 30000])
    axes[2].set_xticklabels(["0", "10k", "20k", "30k"])
    axes[2].set_xlabel("step   (dashed: Round 1, 10k; solid: Round 2, 30k)", fontsize=7.8)
    axes[1].set_ylabel("LR / peak")
    return fig
