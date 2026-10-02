"""Figures for gen.ddpm-objective (Ho et al. 2020 notation, linear schedule, T = 1000)."""
import numpy as np

from figures.style import figure, register

T = 1000
BETA = np.linspace(1e-4, 0.02, T)          # BETA[t-1] = beta_t
ABAR = np.cumprod(1 - BETA)                # ABAR[t-1] = abar_t
SNR = ABAR / (1 - ABAR)                    # SNR[t-1] = SNR(t)
TS = np.arange(2, T + 1)                   # the L_{t-1} terms, t = 2..T


def elbo_eps_weight():
    """Per-step ELBO weight on ||eps - eps_hat||^2 with sigma_t^2 = beta_tilde_t: (SNR(t-1)/SNR(t) - 1)/2."""
    return 0.5 * (SNR[TS - 2] / SNR[TS - 1] - 1)


def elbo_eps_weight_beta():
    """Same weight with sigma_t^2 = beta_t: beta_t / (2 alpha_t (1 - abar_t))."""
    b = BETA[TS - 1]
    return b / (2 * (1 - b) * (1 - ABAR[TS - 1]))


@register("gen.ddpm-objective", "elbo-vs-simple-weights")
def elbo_vs_simple(p):
    """True-ELBO weight per step vs the flat weight of L_simple (both on the eps-MSE)."""
    w = elbo_eps_weight()
    wb = elbo_eps_weight_beta()
    flat = np.full_like(w, w.mean())       # L_simple, scaled to the same average weight
    fig, ax = figure(2.35)
    ax.plot(TS, w, color=p.c(0))
    ax.plot(TS, wb, color=p.c(0), lw=1.0, ls="--")
    ax.plot(TS, flat, color=p.c(1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, T)
    ax.set_ylim(2e-3, 1.5)
    ax.set_xticks([2, 10, 100, 1000])
    ax.set_xticklabels(["2", "10", "100", "1000"])
    ax.set_xlabel("diffusion step t (log scale)")
    ax.set_ylabel(r"weight on $\|\epsilon-\epsilon_\theta\|^2$")
    ax.text(2.6, 0.75, r"ELBO, $\sigma_t^2=\tilde\beta_t$", color=p.c(0), fontsize=8)
    ax.text(2.3, 0.02, r"dashed: $\sigma_t^2=\beta_t$", color=p.c(0), fontsize=7.5)
    ax.text(140, 0.0135, r"$L_{\rm simple}$ (flat)", color=p.c(1), fontsize=8)
    i = 500 - 2
    ax.plot([500], [w[i]], "o", color=p.label, ms=3.5, zorder=5)
    ax.annotate(f"t = 500: {w[i]:.4f}", (500, w[i]), xytext=(70, 0.0032), fontsize=7, color=p.label,
                arrowprops=dict(arrowstyle="-", color=p.label, lw=0.7))
    return fig


@register("gen.ddpm-objective", "elbo-terms")
def elbo_terms(p):
    """Per-step KL terms L_{t-1} of the optimal model for 1-D Gaussian data x0 ~ N(0, 0.5^2)."""
    s2 = 0.25
    snr_prev = SNR[TS - 2]
    snr_t = SNR[TS - 1]
    # Optimal mean is E[mu_tilde | x_t]; with sigma_t^2 = beta_tilde_t the KL is
    # (SNR(t-1) - SNR(t))/2 * Var(x0 | x_t), and Var(x0 | x_t) = 1 / (1/s2 + SNR(t)) for Gaussian data.
    L = 0.5 * (snr_prev - snr_t) / (1 / s2 + snr_t)
    frac10 = L[TS <= 10].sum() / L.sum()
    frac100 = L[TS <= 100].sum() / L.sum()
    fig, ax = figure(2.3)
    ax.plot(TS, L, color=p.c(0), lw=1.6)
    ax.fill_between(TS, 1e-7, L, color=p.c(0), alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.8, T)
    ax.set_ylim(1e-6, 1.5)
    ax.set_xticks([2, 10, 100, 1000])
    ax.set_xticklabels(["2", "10", "100", "1000"])
    ax.set_xlabel("diffusion step t (log scale)")
    ax.set_ylabel(r"$L_{t-1}$ (nats)")
    ax.axvline(10.5, color=p.muted, lw=0.8, ls="--")
    ax.axvline(100.5, color=p.muted, lw=0.8, ls="--")
    ax.text(3.0, 2e-5, f"t ≤ 10:\n{frac10:.0%} of\n$\\Sigma\\,L_{{t-1}}$", fontsize=7.5, color=p.label, ha="center")
    ax.text(32, 2e-5, f"t ≤ 100:\n{frac100:.0%}", fontsize=7.5, color=p.label, ha="center")
    ax.text(900, 0.4, f"$\\Sigma\\,L_{{t-1}}$ = {L.sum():.2f} nats", fontsize=7.5, color=p.muted, ha="right")
    return fig
