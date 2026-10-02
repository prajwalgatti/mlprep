# Review: content plan Part B (Generative Modeling, `gen.*`)

Reviewer pass, 2026-10-02. Scope: `docs/content-plan.md` Part B (B.1–B.5) and the end-of-plan count line. Part A was not
edited. Standard: `docs/writing-brief.md` (teach from scratch, derive rather than assert, cover the whole topic, 6–10 cards a lesson).
Backup of the pre-review file: `$SCRATCH/content-plan.before-genreview.md`.

Tags: **blocking** = a writer following the plan would produce wrong or untaught content; **major** = a real coverage,
scoping or sourcing gap; **minor** = precision, wording or citation detail.

## Summary

- 19 → 22 lessons. `gen.ddpm-objective` and `gen.ddim-solvers` were each about 12 cards of derivation, so they were split.
  `gen.diffusion-parameterizations` (W1) and `gen.diffusion-solvers` are new, and `gen.ddim-solvers` was renamed `gen.ddim`.
  `gen.diffusion-transformers` (DiT/MM-DiT backbones) was added. `gen.evaluation` moved to the end. All lessons were renumbered.
- W1: 8 of 22 (36%). `gen.diffusion-parameterizations` was added to W1, and `gen.distillation` moved from W3 to W2.
- Fixed wrong citations: Liu §2.3, Lipman Thm 2 section, Kingma 2019 §1.9/§2.7, all Luo "§" numbers (Luo has no numbered
  sections), Song-SDE §3.3/§4, SD3 resolution shift. The VQ-VAE loss sign is also fixed.
- B.4 notation table rebuilt. It had a guidance row in the wrong column. It was also missing the velocity sign flip
  (Lipman/Liu vs SD3), the opposite noise-level indexing of NCSN and Song-SDE, the three λ conventions (DPM-Solver uses half
  the log-SNR) and EDM's swapped x/y.
- Every formula and (calc) number in Part B was recomputed in Python. All of them were right. New calcs were added and checked.
- Five uncached papers are needed (DiT, R1, dynamic thresholding, CFG-rescale/zero-terminal-SNR, guidance interval). They are
  listed in B.3 for the coordinator to fetch. I did not download anything.

## Findings and changes

### Coverage vs real interviews (image/video generation labs)

| # | tag | finding | action |
|---|---|---|---|
| C1 | **blocking** | The DDPM posterior $q(x_{t-1}\mid x_t,x_0)$ was planned as "show the key lines". It is the single most common diffusion whiteboard question. | `gen.ddpm-forward-reverse` item 6 now requires the full line-by-line derivation: Markov property, precisions adding ($1/\tilde\beta_t=\alpha_t/\beta_t+1/(1-\bar\alpha_{t-1})$), and the precision-weighted mean. It also gets a new derivation-step question. I verified the formulas numerically. |
| C2 | major | The "weighting" in the ELBO → ε-MSE reduction was only given as Ho's raw coefficient. The SNR form, which links to VDM, Kingma & Gao and flow matching, was missing. | Added to `gen.ddpm-objective` item 4: with $\sigma_t^2=\tilde\beta_t$ the weight is $\frac12(\mathrm{SNR}(t-1)-\mathrm{SNR}(t))$ on the $x_0$-error and $\frac12(\mathrm{SNR}(t-1)/\mathrm{SNR}(t)-1)$ on the ε-error (both checked numerically). It also gets a new figure and a calc question. |
| C3 | major | The equivalence of the ε/x0/v/score parameterizations, the SNR view and the schedules were crammed into items 5–6 of an already overfull lesson. | New lesson `gen.diffusion-parameterizations` (W1). It covers the conversions, score ↔ ε via a 2-line Tweedie, loss equivalences as SNR weightings ($\|\Delta\epsilon\|^2=\mathrm{SNR}\|\Delta x_0\|^2$, $\|\Delta v\|^2=(1+\mathrm{SNR})\|\Delta x_0\|^2$), numerical conditioning (why v), linear vs cosine schedules with calcs, VDM endpoint invariance, terminal SNR and resolution shift. |
| C4 | major | The PF-ODE was "sketched via Fokker–Planck", and the reader has no SDE background. | `gen.score-sde`: a new inline SDE refresher (Brownian increments, Euler–Maruyama, Fokker–Planck stated). The PF-ODE is now **derived** in three lines by rewriting the Laplacian as a drift. A closed-form VE PF-ODE calc for Gaussian data was added (checked numerically). |
| C5 | major | Flow-matching Thm 1 (conditional → marginal field) was "sketch". | Now derived in three lines via linearity of the continuity equation, with the Bayes-weighted form of $u_t(x)$. A derivation-step question on the CFM cross term was added. |
| C6 | major | CFG over-saturation was "named; not cached", with no mechanism. | `gen.guidance` item 6 now gives the mechanism ($\hat x_0$ leaves the data range, amplified by $\sigma_t/\alpha_t$) and one line each on static/dynamic thresholding, CFG rescale and the guidance interval. There is a new figure `gen.guidance/oversaturation` and a new predict-question. The papers are flagged for caching (B.3). |
| C7 | major | Diffusion transformers (which the user explicitly asked for) were only named in `gen.latent-diffusion`, and the vision area that "owns" DiT has no plan yet. | Added `gen.diffusion-transformers`: U-Net, injecting t/c (AdaGN/adaLN), DiT patchify and adaLN-Zero, patch-size compute calc, MM-DiT joint attention, scaling, and a video pointer. The DiT paper is not cached, so DiT-specific claims stay "named" until it is fetched. The coverage table tells the vision area to link here. |
| C8 | major | EDM preconditioning was listed as a formula without its derivation, inside an overfull lesson. | In `gen.diffusion-solvers` item 6 the c's are derived from first principles (App. B.6): the unit-variance input/target requirements and minimal $c_{out}$. It links $c_{skip}$ to the Gaussian-data optimal denoiser in `gen.ebm-score-matching`. Calc at σ = σ_data = 0.5 (0.5, 0.354, 1.414, weight 8) is checked. |
| C9 | major | R1 / zero-centred gradient penalty was missing, though it is standard for modern GANs (StyleGAN). | Added to `gen.wgan` item 5, contrasted with WGAN-GP, with an R1-vs-GP question. The Dirac-GAN (PML2 §26.3.5, cached) was added to `gen.gans` item 6. Mescheder 2018 is flagged for caching. |
| C10 | minor | FID's assumptions were only partly stated. | `gen.evaluation` item 5 now lists the Gaussian (two-moment) assumption with the "FID 0 ≠ same distribution" counterexample, the ImageNet feature bias, KID as the assumption-free alternative, and FVD (named). There is a new which-is-false question. |
| C11 | minor | DDIM needed its marginal-preserving construction derived. Inversion's approximation was not explained. | `gen.ddim` items 3, 6 and 7: $q_\sigma$ written explicitly with the induction, the Euler form $d\bar x=\epsilon_\theta d\bar\sigma$ derived from eq. 12 (checked), and why inversion is approximate. |
| C12 | minor | The VQ-VAE "who gets which gradient" probe was missing. | Added to `gen.vae-variants` item 6. |
| C13 | minor | The FM loss weighting was "derive for the linear path" with no target result. | `gen.rectified-flow` item 4 now states $\|\Delta v\|^2=\|\Delta\epsilon\|^2/\alpha_t^2=\|\Delta x_{data}\|^2/\sigma_t^2$, i.e. a factor $(1+\mathrm{SNR}^{-1/2})^2$ on the ε-loss (factor 4 at SNR = 1; checked numerically). |
| C14 | minor | Padding candidates: `gen.overview` item 9 (historical arc), `gen.gans` item 9 (architecture name list), the β-VAE disentanglement metric. These are cheap paragraphs, not padding that crowds out derivations. | Flagged, not cut. Writers should keep them to one short paragraph each. |
| C15 | minor | Discrete/masked diffusion (MaskGIT, D3PM, LLaDA) wasn't mentioned anywhere. | Named in one paragraph in `gen.overview` item 6. Deferred, consistent with the LLM plan. |
| C16 | minor | `gen.overview` item 6 derived the EBM MLE gradient, duplicating `gen.ebm-score-matching` item 2. | The derivation was removed from the overview and one sentence plus a pointer kept. Its question was replaced with a predict-question. |

Coverage I checked and found adequate: the forward marginal (with the variance-merge identity, checked), the ELBO
decomposition into $L_T/L_{t-1}/L_0$ (Ho eq. 5, now with the Bayes-on-$x_0$ trick spelled out), the ε-reduction and Ho's
weight, $L_{\text{simple}}$, Tweedie/DSM, the reverse SDE, CFG derivation, rectified flow/reflow, latent diffusion,
consistency models, posterior collapse, the Gaussian KL (now derived via moments, plus the general two-Gaussian KL), the GAN
optimal D → JSD, mode collapse, KR duality (statement + intuition, which is the right depth), change-of-variables flows,
autoregressive likelihood, and IS/FID/precision–recall.

### Lesson scoping

| # | tag | finding | action |
|---|---|---|---|
| S1 | **blocking** | `gen.ddpm-objective` held 9 items, of which 5 needed derivations. That is about 12 cards, and the brief says never compress a derivation. | Split into `gen.ddpm-objective` (ELBO, Gaussian KL, ε-param, SNR-form weight, $L_{\text{simple}}$, variances, likelihood; about 9 cards) and `gen.diffusion-parameterizations` (about 8–9 cards). |
| S2 | **blocking** | `gen.ddim-solvers` held the DDIM derivation, ODE-solver theory, DPM-Solver, the EDM design space with preconditioning, SDE vs ODE, and guidance. That is about 13 cards. | Split into `gen.ddim` (about 8) and `gen.diffusion-solvers` (about 9). |
| S3 | major | `gen.latent-variables-elbo` required `fund.gmm-em` (W2), which already derives the same ELBO. So a W1 lesson was gated on a W2 lesson that duplicates it, and the user's own example ("a topic I don't know, like ELBO") demands a self-contained lesson. | Prereqs are now [fund.kl-divergence, fund.mle], with gmm-em as a cross-link. A card budget was added, and refreshers (Jensen, KL ≥ 0, importance sampling) are made explicit. |
| S4 | major | `gen.vae` (W1) required `fund.gradient-estimators` (W3), and `gen.vae-variants` required `fund.gumbel-softmax` (W3). | Both became pointers. Reparameterization is taught inline in `gen.vae`, and `fund.autoencoders` was added as a prereq (the AE-vs-VAE item needs it). |
| S5 | minor | `gen.flow-matching` (W1) depends on `gen.flows` (W2). | Kept the prereq, but the lesson must recap CNFs and the continuity equation inline (noted in B.1 and in the lesson). |
| S6 | minor | Dense lessons had no card plan. | Added **Card budget** lines to latent-variables-elbo, ebm-score-matching, ddpm-forward-reverse, ddpm-objective, diffusion-parameterizations, score-sde, ddim, diffusion-solvers, guidance, flow-matching and diffusion-transformers. |
| S7 | minor | `gen.evaluation` sat at order 20 but refers to mode collapse, guidance curves and diffusion memorization. | Moved to order 220. Its prereqs are now [gen.overview, gen.gans]. |
| S8 | minor | Wave choice: `gen.distillation` was W3, but few-step generation (consistency, ADD/DMD) is a standard probe at image/video labs. | Moved to W2. `gen.latent-diffusion` and `gen.score-sde` are also strong W1 candidates for a video/image lab; I left them at W2 to keep W1 ≈ a third (**flagged**). |
| S9 | minor | `gen.guidance` and `gen.score-sde` need score = −ε/σ. | Their prereqs now point at `gen.diffusion-parameterizations`. |

### Sources (spot-checks against the cache)

More than 12 cited locations were grepped. Confirmed:
- PML2 TOC: ch.20–26 page ranges, §10.1 (pdf p.473–479), §2.7.2–2.7.3, §6.8, §25.x and §26.x pages.
- Bishop §9.4 (p.470) and §10.1 (p.482–490). CS229 ch.11 and ch.14 pages.
- Ho 2020: eqs. 4–8, 11, 12, 14, §3.1–3.4, Table 2, and the "two extreme" variances.
- DDIM: §3.1–3.2, eq. 12, §4.1–4.3, App. C.1, and the App. C.2 notation note.
- EDM: §2–5, Table 1, App. B.1–B.6, ρ = 7, NFE = 35, $P_{mean}$ = −1.2, $P_{std}$ = 1.2, σ_data = 0.5, $c_{noise}$.
- Lipman: §3.1/§3.2/§4/§4.1, Thms 1–3, the OT path. Liu: §2.1–2.3, Thms 3.3/3.5/3.7, the "not too many reflows" remark.
- Song-SDE: §2.1 (ascending σ), §3.1–3.4, §4.1–4.3, §5, the App. B VP schedule (0.1 → 20).
- Salimans & Ho: §4, eq. 9 (ε-loss = SNR-weighted x-loss), v and App. D.
- Nichol: §3.1–3.3, s = 0.008, $L_{\text{hybrid}}$.
- Ho & Salimans: eq. 6, the implicit-classifier gradient, the w ∈ {0,…,4} sweep with best FID at w = 0.1–0.3.
- Dhariwal: §3 (AdaGN), §4.1–4.3, Alg. 1–2, the $p(y\mid x)^s/Z$ reading.
- Rombach: §3.1–3.3, §4.1, App. G (rescaling by $1/\hat\sigma$).
- Arjovsky & Bottou Thm 2.5 (KL − 2JSD). Gulrajani λ = 10. Goodfellow Prop. 1/Thm 1 (−log 4).
- Kingma 2019 TOC. Kingma 2014 App. B. VQ-VAE §3.2 (β = 0.25, robust 0.1–2.0). NCSN λ(σ) = σ².
- VDM §3–5 (endpoint invariance, the SNR-difference loss). Kingma & Gao Table 1 and App. D.3.
- SD3: §2–5, $\lambda_t$ = log-SNR, CFG scales 1.0–5.0 (App. B.3), §5.3.2 shift.
- Consistency models §3–5. DPM-Solver λ = ½ log-SNR (§2.2, §3.1–3.2). Tong minibatch OT. Bowman KL annealing. IWAE Thm 1.
- DLB §20.10.x and §20.14.

Wrong references fixed:

| # | tag | wrong | fixed |
|---|---|---|---|
| R1 | major | `gen.distillation`: "Liu … §2.3 reflow & distillation" | §2.3 is the *nonlinear extension*. Reflow and distillation are in §2.2 (p.7–8). |
| R2 | major | Every "Luo 2022 §2.x" (§2.1–2.2, §2.4) | Luo's sections are unnumbered. B.3 now lists title → page, and all citations use titles and pages. |
| R3 | minor | Lipman: Thm 2 placed in §3.1 | Thm 2 is in §3.2. The Gaussian $u_t$ formula is Thm 3 (§4), and the special instances are §4.1. |
| R4 | minor | Kingma & Welling 2019 "§1.7–1.9"; "§2.7 marginal-likelihood estimation" | There is no §1.9, so it is §1.7–1.8. Marginal likelihood is §2.6, and full-covariance posteriors §2.5.1. |
| R5 | minor | Song-SDE "§3.3–3.4 VE/VP…", "§4 predictor–corrector" | §3.3 is time-conditional DSM and §3.4 is VE/VP/sub-VP. Predictor–corrector is §4.2. |
| R6 | minor | SD3 "§2–3 … resolution-dependent timestep shifting" | The shift is §5.3.2. Logit-normal is §3.1. |
| R7 | minor | Rombach "Appendix: the latent scaling factor", "SD v1 uses 0.18215" | App. G gives the *method* (divide by the std from the first batch). 0.18215 is a code constant and is not in the paper; the plan now says so. |
| R8 | minor | DDIM "App. C the notation note"; EDM preconditioning without the derivation's location | App. C.2; EDM App. B.6. |

Unsourced items: DiT, R1, dynamic thresholding, CFG rescale and zero-terminal SNR, and the guidance interval are not cached.
They are added to B.3 "Not cached but needed" (arXiv ids to confirm on fetch), and the lessons treat them as named only
until they are cached. **Flagged for the coordinator; I did not download anything.**

### Correctness (Python-verified)

All recomputed in `$SCRATCH/genchk.py` and `genchk2.py`:
- bits/dim 2.818; log 100 = 4.605; log 512 = 6.238; WaveNet receptive field 1024.
- Linear-Gaussian ELBO: log p = −1.5155, ELBO at the prior = −1.9189, gap 0.4034 = the KL; the ELBO at the posterior equals log p.
- Gaussian KLs 0.5 and 0.318. Flow $p_x(1)$ = 0.1995. Score −0.5.
- Tweedie (both the unit and the general Gaussian cases).
- DDPM: $x_t$ = 1.866; $0.98^{100}$ = 0.1326, SNR 0.153; the variance-merge identity; posterior mean via ε = via $x_0$; posterior variance = $\tilde\beta_t$.
- The SNR-form weights; $L_T$ ≈ 2.0e-5 nats.
- Linear vs cosine ᾱ and SNR tables (SNR = 1 at t ≈ 259 vs 496).
- DDIM step 1.134 / 1.207. v = −0.2. CFG 1.1. EDM: NFE 35 and the c's.
- VE PF-ODE closed form vs numeric (0.5883 vs 0.5883). FM weighting factors.
- Fenchel conjugate $f^*(1)$ = 1. Progressive distillation 8 rounds. LDM 48×.

| # | tag | finding | action |
|---|---|---|---|
| K1 | major | B.4: the "guidance weight" row put Ho & Salimans's CFG formula in the VDM column, and attributed the s-convention to SD3 itself. | Rebuilt the table. Guidance has its own row: Ho & Salimans eq. 6 (w = 0 unguided) vs the SD-style scale (s = 1 unguided, s = 1 + w), with SD3's reported scales cited to App. B.3. |
| K2 | major | B.4 was missing the clashes that actually bite. | Added rows: (i) the velocity sign, where Lipman/Liu's $u=x_1-x_0$ (data − noise) is opposite to SD3's $v=\epsilon-x_0$; (ii) noise-level indexing, where NCSN has σ_1 largest and Song-SDE σ_1 smallest; (iii) log-SNR symbols, where VDM's γ = −log SNR, Salimans/Ho's λ = log SNR and DPM-Solver's λ = ½ log SNR; (iv) EDM's y = data and x = noisy; (v) the continuous VP β(t) (0.1 → 20 = T × Ho's β); (vi) Salimans & Ho's v as $dz/d\phi$. |
| K3 | minor | VQ-VAE loss copied the paper's "log p(x\|z_q) + …", which is a sign error (it's a loss). | Now $-\log p$, with a note about the paper's typo. |
| K4 | minor | CFG implicit classifier: "CFG ≈ classifier guidance with scale (1+w)" didn't say on which base model. | Rewritten: weight w on the conditional model, which equals scale 1 + w on the unconditional model. Ho & Salimans's caveat that $\epsilon_\theta$ is not any classifier's gradient was added. |
| K5 | minor | "x0-prediction is ill-posed at low noise" (my own first wording) was imprecise. | Rephrased: at low noise x0-prediction makes errors land directly on the output; EDM's skip path addresses this. |
| K6 | minor | Rectified flow: "DDIM ≈ an Euler step of the FM ODE" was vague. Reflow straightening had no statement. | Gao et al.'s exact claim was added (identical for the linear schedule, plus DDIM's invariance to linear schedule rescaling), together with Liu Thm 3.7 (the best straightness among K flows is O(1/K)) and the "don't over-reflow" remark. |

All other formulas in Part B check out against the papers, including: Ho's ε-weight; $\tilde\mu_t$ and $\tilde\beta_t$;
the DDIM eq. 12 in Ho notation; the DDIM ODE form; EDM's $dx/d\sigma=(x-D)/\sigma$; the VP/VE SDE coefficients; the
reverse SDE and PF-ODE; Lipman's Gaussian and OT-path $u_t$; Ho & Salimans eq. 6; Dhariwal's guided ε; IS/FID formulas;
WGAN-GP; f-GAN; the MAF/IAF directions; Glow's log-det; Hyvärinen's objective; the DSM target; and Tweedie.

### Pedagogy

| # | tag | finding | action |
|---|---|---|---|
| P1 | major | Inline refreshers a newcomer needs were not called out: the Gaussian sum/product/completing-the-square identities (DDPM), SDE basics (score-sde), divergence/continuity (flow matching), the Fenchel conjugate (f-GAN), importance sampling (ELBO). | Each was added explicitly where it is used, self-contained, with a pointer to the fund.* lesson. Jensen and change of variables were already present. |
| P2 | minor | `gen.ddpm-objective` item 1 didn't explain *why* the Bayes-on-$x_0$ rewrite matters. | Added: it turns Monte-Carlo terms into closed-form Gaussian KLs (Ho's Rao–Blackwellization point). |
| P3 | minor | Order check: each syllabus now opens with the problem. The new lessons open with "the cost" (DDIM), "what the backbone must do" (DiT), and "four targets, one piece of information" (parameterizations). | — |

### Questions and figures

| # | tag | finding | action |
|---|---|---|---|
| Q1 | minor | Some question ideas were recall-only (e.g. "Time convention in Lipman: which end is noise?"). | That one was replaced with a CFM-proof derivation step and a marginal-velocity predict question. Derivation-step and predict questions were added to the ELBO, DDPM I/II, parameterizations, score-sde, DDIM, solvers, guidance, flow matching, rectified flow, wgan, latent diffusion and evaluation lessons. |
| Q2 | minor | The CFG trade-off figure was "schematic". | Now `gen.guidance/tradeoff-toy`, computed from exact scores on a 2-D two-class mixture. If the FID/IS curve is shown, it must be labelled as redrawn from Ho & Salimans. Added `gen.guidance/oversaturation`. |
| Q3 | minor | The solver quality-vs-NFE figure was schematic. | Now computed on a toy with a known exact PF-ODE solution. New figures were added: `diffusion-parameterizations/{schedules,weightings,error-amplification}`, `diffusion-solvers/precond-coefficients`, `ddpm-objective/elbo-vs-simple-weights`, and three `diffusion-transformers` figures. |
| Q4 | — | The key-idea figures the brief asks for are all present: forward diffusion of a 2-D toy, score fields, Langevin chains, the CFG trade-off, straight conditional vs curved marginal FM paths, and reflow straightening. | — |

## Flagged, not changed

1. **Cache five papers** (B.3 list) before writing `gen.diffusion-transformers`, the R1 part of `gen.wgan`, `gen.guidance` item 6 and `gen.diffusion-parameterizations` item 8. Downloading needs the user's permission.
2. **Part A, `fund.gmm-em`** header says it covers "the ELBO foundation for gen.latent-variables-elbo". The gen lesson no longer depends on it, so the Part A reviewer may want to reword this to a cross-link. I didn't touch Part A.
3. **Vision area**: when it is planned, it should link to `gen.diffusion-transformers` for DiT rather than duplicate it, and own ViT, video diffusion (3-D VAEs, spacetime attention) and FVD.
4. **W1 candidates**: `gen.latent-diffusion` and `gen.score-sde` would be reasonable W1 picks for a lab focused on image/video generation. Left at W2 to keep W1 ≈ a third.
5. **Padding candidates** (C14): keep to one paragraph each; not removed.
6. **The end-of-plan line** "63 fundamentals lessons" is Part A's count; I changed only the gen count (19 → 22).
