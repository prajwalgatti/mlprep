"""Figures for gen.diffusion-parameterizations.

Notation: x_t = alpha_t x_0 + sigma_t eps with alpha_t = sqrt(abar_t), sigma_t = sqrt(1 - abar_t), lambda = log SNR = log(a^2/b^2).
"""
import numpy as np

from figures.style import figure, register

T = 1000
T_AX = np.arange(1, T + 1)


def linear_abar():
    return np.cumprod(1 - np.linspace(1e-4, 0.02, T))


def cosine_abar(s=0.008):
    """Nichol & Dhariwal cosine schedule, betas clipped at 0.999, abar rebuilt as a cumulative product."""
    t = np.arange(0, T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    beta = np.minimum(1 - ab[1:] / ab[:-1], 0.999)
    return np.cumprod(1 - beta)


def log_snr(ab):
    return np.log(ab / (1 - ab))


def _crossing(ab):
    return int(np.argmax(ab / (1 - ab) < 1))     # index of first t (minus 1) with SNR < 1, i.e. last t with SNR >= 1


@register("gen.diffusion-parameterizations", "schedules")
def schedules(p):
    """abar_t and log-SNR for the linear and cosine schedules (T = 1000)."""
    lin, cos = linear_abar(), cosine_abar()
    fig, (a1, a2) = figure(3.4, nrows=2, sharex=True)
    for ab, c, name in [(lin, p.c(1), "linear"), (cos, p.c(0), "cosine")]:
        a1.plot(T_AX, ab, color=c)
        a2.plot(T_AX, log_snr(ab), color=c)
        tc = _crossing(ab)
        a2.plot([tc], [0], "o", color=c, ms=3.5, zorder=5)
    a1.set_ylabel(r"$\bar\alpha_t$")
    a1.set_ylim(0, 1.05)
    a1.text(120, 0.35, "linear", color=p.c(1), fontsize=8)
    a1.text(640, 0.45, "cosine", color=p.c(0), fontsize=8)
    a2.axhline(0, color=p.muted, lw=0.7, ls="--")
    a2.set_ylabel(r"log-SNR $\lambda_t$")
    a2.set_ylim(-14, 11)
    a2.set_yticks([-10, -5, 0, 5, 10])
    a2.set_xlabel("diffusion step t")
    a2.set_xlim(0, T)
    a2.set_xticks([0, 250, 500, 750, 1000])
    tl, tcs = _crossing(lin), _crossing(cos)
    a2.annotate(f"SNR = 1 at t ≈ {tl}", (tl, 0), xytext=(20, -7.5), fontsize=7, color=p.c(1),
                arrowprops=dict(arrowstyle="-", color=p.c(1), lw=0.7))
    a2.annotate(f"t ≈ {tcs}", (tcs, 0), xytext=(560, 4.5), fontsize=7, color=p.c(0),
                arrowprops=dict(arrowstyle="-", color=p.c(0), lw=0.7))
    return fig


@register("gen.diffusion-parameterizations", "schedules-quiz")
def schedules_quiz(p):
    """Unlabelled abar_t curves for an MCQ: A = linear, B = cosine (do not annotate)."""
    lin, cos = linear_abar(), cosine_abar()
    fig, ax = figure(2.1)
    ax.plot(T_AX, lin, color=p.c(1))
    ax.plot(T_AX, cos, color=p.c(0))
    ax.text(330, 0.5, "A", color=p.c(1), fontsize=9, fontweight="bold")
    ax.text(600, 0.5, "B", color=p.c(0), fontsize=9, fontweight="bold")
    ax.set_ylabel(r"$\bar\alpha_t$")
    ax.set_xlabel("diffusion step t")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 250, 500, 750, 1000])
    return fig


@register("gen.diffusion-parameterizations", "weightings")
def weightings(p):
    """Implied weight on the eps-MSE per unit log-SNR (Kingma & Gao 2023 convention), relative to lambda = 0."""
    lam = np.linspace(-10, 10, 801)
    # epsilon-MSE with t ~ U: weight = p(lambda) = -dt/dlambda.
    lin_ab = linear_abar()
    lam_lin = log_snr(lin_ab)
    dens_lin = 1.0 / (T * np.abs(np.gradient(lam_lin)))           # density of lambda under uniform t
    order = np.argsort(lam_lin)
    w_eps_lin = np.interp(lam, lam_lin[order], dens_lin[order], left=np.nan, right=np.nan)
    w_eps_lin /= np.interp(0, lam_lin[order], dens_lin[order])
    w_eps_cos = 1 / np.cosh(lam / 2)                                # cosine: p(lambda) = sech(lambda/2) / (2 pi)
    w_v_cos = np.exp(-lam / 2)                                      # v-MSE, cosine (= FM-OT): sech * (1 + e^-lambda)
    fig, ax = figure(2.45)
    ax.axhline(1, color=p.fg, lw=1.6)
    ax.plot(lam, w_eps_cos, color=p.c(0))
    ax.plot(lam, w_eps_lin, color=p.c(1), ls="--", lw=1.3)
    ax.plot(lam, w_v_cos, color=p.c(2))
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 300)
    ax.set_xlim(-10, 10)
    ax.set_xlabel(r"log-SNR $\lambda$   (← noisier · cleaner →)")
    ax.set_ylabel(r"weight on $\epsilon$-MSE per unit $\lambda$")
    ax.text(9.6, 1.35, "ELBO (flat)", color=p.fg, fontsize=7.5, ha="right")
    ax.text(-6.3, 0.018, r"$\epsilon$-loss, cosine", color=p.c(0), fontsize=7.5)
    ax.text(-6.3, 0.0055, r"$\epsilon$-loss, linear (dashed)", color=p.c(1), fontsize=7.5)
    ax.text(-9.6, 60, "v-loss, cosine = FM-OT", color=p.c(2), fontsize=7.5)
    return fig


@register("gen.diffusion-parameterizations", "error-amplification")
def error_amplification(p):
    """How a unit error in the network output maps to x0-error (top) and eps-error (bottom)."""
    lam = np.linspace(-10, 10, 401)
    snr = np.exp(lam)
    a = np.sqrt(snr / (1 + snr))
    b = np.sqrt(1 / (1 + snr))
    fig, (t1, t2) = figure(3.3, nrows=2, sharex=True)
    for ax, curves, ylab in [
        (t1, [(b / a, 0, r"$\epsilon$-pred: $\sigma_t/\alpha_t$"), (np.ones_like(lam), 1, r"$x_0$-pred: 1"), (b, 2, r"v-pred: $\sigma_t$")],
         r"$x_0$ error"),
        (t2, [(np.ones_like(lam), 0, r"$\epsilon$-pred: 1"), (a / b, 1, r"$x_0$-pred: $\alpha_t/\sigma_t$"), (a, 2, r"v-pred: $\alpha_t$")],
         r"$\epsilon$ error"),
    ]:
        for y, i, _ in curves:
            ax.plot(lam, y, color=p.c(i), lw=1.6 if i != 2 else 2.0)
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 300)
        ax.set_ylabel(ylab)
    t1.text(-9.6, 40, r"$\epsilon$-pred: $\sigma_t/\alpha_t$", color=p.c(0), fontsize=7.5)
    t1.text(4.2, 1.5, r"$x_0$-pred: 1", color=p.c(1), fontsize=7.5)
    t1.text(-9.6, 0.25, r"v-pred: $\sigma_t$", color=p.c(2), fontsize=7.5)
    t2.text(-9.6, 1.5, r"$\epsilon$-pred: 1", color=p.c(0), fontsize=7.5)
    t2.text(-1.0, 40, r"$x_0$-pred: $\alpha_t/\sigma_t$", color=p.c(1), fontsize=7.5)
    t2.text(-3.5, 0.012, r"v-pred: $\alpha_t$", color=p.c(2), fontsize=7.5)
    t1.set_title("error per unit error in the network output", fontsize=8, pad=3)
    t2.set_xlabel(r"log-SNR $\lambda$   (← noisier · cleaner →)")
    t2.set_xlim(-10, 10)
    return fig
