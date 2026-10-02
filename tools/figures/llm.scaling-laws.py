"""Figures for llm.scaling-laws.

All curves are computed from fitted scaling laws, not measured runs:
- Chinchilla Approach 3 as printed (Hoffmann et al. 2022 App. D.2, eq. 10): E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
- the same fit with the unrounded TeX-source values (Besiroglu et al. 2024 eq. 4): alpha=0.3392, beta=0.2849, E=1.6934
- the Epoch AI refit (Besiroglu et al. 2024 eq. 3): E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658
"""
import numpy as np

from figures.style import figure, register

FITS = {
    "printed": (1.69, 406.4, 410.7, 0.34, 0.28),
    "unrounded": (1.6934, 406.4, 410.7, 0.3392, 0.2849),
    "refit": (1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}
C_CHIN = 5.76e23


def _loss(fit, n, d):
    e, a, b, al, be = fit
    return e + a / n ** al + b / d ** be


def _opt(fit, c):
    """Closed-form compute-optimal (N, D) under C = 6ND."""
    e, a, b, al, be = fit
    g = (al * a / (be * b)) ** (1 / (al + be))
    n = g * (c / 6) ** (be / (al + be))
    return n, c / (6 * n)


def _fmt_c(c):
    k = int(round(np.log10(c)))
    return rf"$10^{{{k}}}$"


@register("llm.scaling-laws", "isoflop")
def isoflop(p):
    """Iso-FLOP curves L(N, C/6N) from the refit law at five budgets, minima joined."""
    fit = FITS["refit"]
    budgets = [1e19, 1e20, 1e21, 1e22, 1e23]
    fig, ax = figure(2.7)
    mins = []
    for i, c in enumerate(budgets):
        n_opt, _ = _opt(fit, c)
        n = np.logspace(np.log10(n_opt) - 1.1, np.log10(n_opt) + 1.1, 300)
        d = c / (6 * n)
        loss = _loss(fit, n, d)
        ax.plot(n, loss, color=p.c(i), lw=1.6)
        mins.append((n_opt, _loss(fit, n_opt, c / (6 * n_opt))))
        ax.text(n[-1] * 1.25, loss[-1], _fmt_c(c), color=p.c(i), fontsize=7.5, ha="left", va="center")
    mn = np.array(mins)
    ax.plot(mn[:, 0], mn[:, 1], color=p.fg, lw=1.0, ls="--", zorder=4)
    ax.plot(mn[:, 0], mn[:, 1], "o", color=p.fg, ms=3.5, zorder=5)
    ax.annotate("minima =\ncompute-optimal $N$", (mn[1, 0], mn[1, 1]), xytext=(1.5e6, 2.28),
                color=p.label, fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.8))
    ax.text(1.2e6, 1.93, "label = budget $C$ (FLOPs)", color=p.muted, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("parameters $N$   (tokens $D = C/6N$)")
    ax.set_ylabel("predicted loss (nats/token)")
    ax.set_xlim(1e6, 3e12)
    ax.set_ylim(1.9, 3.6)
    ax.set_xticks([1e6, 1e8, 1e10, 1e12])
    return fig


@register("llm.scaling-laws", "tokens-per-param")
def tokens_per_param(p):
    """Optimal D/N vs compute for the three constant sets; Chinchilla's actual choice marked."""
    c = np.logspace(19, 26, 200)
    fig, ax = figure(2.6)
    for j, (lab, col) in enumerate([("printed constants (0.34, 0.28)", p.c(1)),
                                    ("unrounded TeX constants", p.c(3)), ("Epoch AI refit", p.c(0))]):
        ax.text(1.3e19, 200 / 1.32 ** j, lab, color=col, fontsize=7.5, va="center")
    styles = [("printed", "printed constants (0.34, 0.28)", p.c(1)),
              ("unrounded", "unrounded TeX constants", p.c(3)),
              ("refit", "Epoch AI refit", p.c(0))]
    for key, lab, col in styles:
        n, d = _opt(FITS[key], c)
        ax.plot(c, d / n, color=col, lw=1.8)
        n0, d0 = _opt(FITS[key], C_CHIN)
        ax.plot([C_CHIN], [d0 / n0], "o", color=col, ms=4, zorder=5)
        ax.text(C_CHIN * 1.6, d0 / n0 * (0.88 if key == "refit" else 1.0), f"{d0 / n0:.0f}", color=col, fontsize=7.5, va="center")
    ax.axhline(20, color=p.muted, lw=0.9, ls=":")
    ax.text(4e25, 21.5, "20", color=p.muted, fontsize=7.5, ha="center", va="bottom")
    ax.plot([C_CHIN], [1.4e12 / 70e9], "*", color=p.fg, ms=9, zorder=6)
    ax.annotate("Chinchilla 70B / 1.4T", (C_CHIN, 20), xytext=(2.5e21, 11),
                color=p.fg, fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=p.fg, lw=0.7))
    ax.axvline(C_CHIN, color=p.faint, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e19, 1e26)
    ax.set_ylim(8, 250)
    ax.set_yticks([10, 20, 50, 100, 200])
    ax.set_yticklabels(["10", "20", "50", "100", "200"])
    ax.set_xticks([1e19, 1e21, 1e23, 1e25])
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("optimal tokens per parameter")
    return fig


@register("llm.scaling-laws", "kaplan-vs-chinchilla")
def kaplan_vs_chinchilla(p):
    """N_opt vs C: Kaplan's 0.73 exponent vs Chinchilla's 0.50, both anchored at Chinchilla App. D.4 (1e21 FLOPs)."""
    c = np.logspace(19, 25, 100)
    kap = 4.68e9 * (c / 1e21) ** 0.73
    chi = 2.86e9 * (c / 1e21) ** 0.50
    fig, ax = figure(2.6)
    ax.plot(c, kap, color=p.c(1), lw=1.8)
    ax.plot(c, chi, color=p.c(0), lw=1.8)
    ax.text(1.3e19, 4.5e9, r"Kaplan: $N\propto C^{0.73}$", color=p.c(1), fontsize=7.5)
    ax.text(5e21, 2.2e9, r"Chinchilla: $N\propto C^{0.50}$", color=p.c(0), fontsize=7.5)
    pts = [(3.15e23, 175e9, "GPT-3"), (C_CHIN, 280e9, "Gopher"), (C_CHIN, 70e9, "Chinchilla")]
    for cc, nn, lab in pts:
        ax.plot([cc], [nn], "o", color=p.fg, ms=4, zorder=5)
    ax.text(3.15e23 / 1.3, 175e9 / 1.9, "GPT-3", color=p.fg, fontsize=7.5, ha="right", va="center")
    ax.text(C_CHIN * 1.5, 280e9 / 1.25, "Gopher", color=p.fg, fontsize=7.5, ha="left", va="center")
    ax.text(C_CHIN * 1.5, 70e9, "Chinchilla", color=p.fg, fontsize=7.5, ha="left", va="center")
    ax.axvline(1e21, color=p.faint, lw=0.8, zorder=0)
    ax.text(1e21 * 1.2, 4e11, "anchors at\n$10^{21}$", color=p.muted, fontsize=7, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e19, 1e25)
    ax.set_ylim(1e8, 3e12)
    ax.set_xticks([1e19, 1e21, 1e23, 1e25])
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("compute-optimal params $N$")
    return fig


@register("llm.scaling-laws", "ratio-quiz")
def ratio_quiz(p):
    """[fig-Q] Optimal D/N vs C for three made-up fits with different (alpha, beta). No answer annotated."""
    fits = {"A": (1.8, 400.0, 192.0, 0.38, 0.30),
            "B": (1.8, 400.0, 5936.0, 0.30, 0.38),
            "C": (1.8, 400.0, 1108.0, 0.34, 0.34)}
    c = np.logspace(19, 26, 200)
    fig, ax = figure(2.3)
    for i, (k, f) in enumerate(fits.items()):
        n, d = _opt(f, c)
        r = d / n
        ax.plot(c, r, color=p.c(i), lw=1.8)
        ax.text(1.6e26, r[-1], k, color=p.c(i), fontsize=9, va="center", ha="left", weight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e19, 1e26)
    ax.set_xticks([1e19, 1e21, 1e23, 1e25])
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("optimal tokens per parameter")
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_title("synthetic fits", fontsize=8, color=p.muted)
    return fig
