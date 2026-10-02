# Content review: gen.gans (Generative Adversarial Networks)

Reviewed: `PrepApp/Content/genmodels/gen.gans.yaml`, `tools/figures/gen.gans.py`, all 5 previews (workbench, plus terminal or
editorial), the plan in `docs/content-plan.md` §60, the gen.wgan plan §70 (to check overlap), and `docs/reviews/plan-generative-review.md` (C9, C14).
Claims checked against the cache: Goodfellow 2014 §3 (k = 1); Goodfellow 2016 tutorial §3.2.5, §4.2 (eq. 15), §5.1.1; Arjovsky & Bottou
Thm 2.5 proof and Thm 2.6 (Cauchy); Mescheder 2018 §2.1, §2.3 (TTUR remark, Lemmas 2.4–2.5), §3.3 (Lemma 3.3, γ_critical = 2|f′(0)|),
§4.1 (eq. 9), §5 and App. (1024² CelebA-HQ and all-ImageNet with R1, no progressive growing); Salimans 2016 §3.1–3.4; Kynkäänniemi 2019
Fig. 5 (ψ = 0 gives zero recall); PML2 §26.4 (concat, multi-layer, projection, auxiliary classifier), §26.6.2 (DCGAN), §26.6.5, §26.7.1
(pix2pix U-Net + PatchGAN, StyleGAN coarse/fine styles).

All numbers recomputed independently (`scratchpad/crit_gans.py`): worked-example D* = (½, ⅔, 0), C = −1.1705, JSD = 0.1079 (also computed
directly from the two KLs to m); q4 C = −0.9548, JSD = 0.2158, distractor −0.549; q1 0.25; q5 99 and −log 0.01 = 4.61; q8 2.02, radius 2.326 after
100 steps; alternating-step eigenvalue moduli 1 for h < 2 and not for h = 2.1; Dirac-GAN eigenvalues ±i/2 and −γ/2 ± √(γ²/4 − ¼) (q9: −0.25 ± 0.433i);
exact Dirac-GAN simulation (simultaneous play grows, alternating stays bounded, R1 with γ = 0.3/0.5/1 converges); one-sided smoothing D* by brute-force maximization.
**The Arjovsky–Bottou result was checked numerically:** on a softmax family, ∇θ E_pθ[−log D*_θ0] = ∇θ[KL(pθ‖p_data) − 2 JSD] at θ0 to 1e-6, and
the envelope step ∇ E_pθ[log(1−D*)] = ∇ 2JSD also holds. The writer's `calc.py` agrees.
`validate.py`: no errors for this file. The only warning is the missing prereq `gen.overview` (planned).

---

## gen.gans

Indexing: `explainer[i]` is 0-based, as in the YAML. "Card n" in prose is 1-based, so card 5 = `explainer[4]`.

**Verdict: revise** (no blocking errors; the fixes are a few teaching gaps, two recall-duplicate MCQs, one ambiguous figure question, and trimming)

### The writer's flagged points
- **Truncation, described generically:** acceptable. Kynkäänniemi Fig. 5 supports both "precision up, recall down" and "zero recall at full
  truncation (ψ = 0)" for StyleGAN. "Draw latents closer to their mean" is a fair generic description. Keep it.
- **"StyleGAN uses R1", left out:** correct call. Neither Mescheder nor PML2 §26.6.5/§26.6.6 says it, so it can't be verified from the cache.
- **Architectures named only:** matches the plan (item 9, and review C14 asks to keep it to one short paragraph). PML2 supports the DCGAN, pix2pix
  and StyleGAN descriptions.
- **Worked example, Dirac-GAN/R1 eigenvalues, KL − 2·JSD:** all correct (see above). In the lesson's convention (D = σ(ψx), real data = label 1),
  f′(0) = ½, which matches Mescheder's Lemma 3.3 with γ_critical = 1.

### Length (11 cards)
The length is mostly earned, and I would **not** split the lesson or move a card to gen.wgan. Cards 1–9 each carry a derivation or mechanism the
plan requires, and none is over about 300 words. The extra reading time comes from redundancy, not from content that belongs elsewhere:
- **Trim card 10 ("What interviewers probe", 292 words).** Its first two bullets and the "non-saturating", "what it optimizes" and "Fix?" bullets
  restate formulas that card 11 already lists. Keep the follow-ups that add something ("Does training actually minimize the JSD?", "Can I watch
  the loss?", "mode collapse cause and fixes", "why diffusion won") and the common wrong answers. Cut the rest to one line each, or drop them.
  This saves about 120 words.
- **Card 9 (300 words):** cut the spectral-norm definition to "(gen.wgan)". gen.wgan item 5 owns it, and the full definition is repeated there. This saves about 30 words.
- **Keep the R1 eigenvalue analysis here.** The plan puts R1 in gen.wgan item 5, but the Dirac-GAN lives here and the R1 fix is its natural end.
  Moving it would split one argument across two lessons. **Tell the gen.wgan writer:** point back to gen.gans for the Dirac-GAN and the R1
  eigenvalues, and only add the R1 vs WGAN-GP contrast. Likewise, gen.wgan item 1 should build on card 5's disjoint-supports result, not
  re-derive it.

### Findings
1. **major: explainer[7] (mode collapse) lists the Salimans remedies without saying why most of them work, and frames them all as
   mode-collapse fixes.** Salimans §3 presents them as heuristics "to encourage convergence". Only minibatch discrimination targets collapse.
   Add one clause each:
   - feature matching stops G from over-fitting to the current D, because G matches statistics instead of maximizing D's output;
   - historical averaging is inspired by fictitious play, and damps the orbits that plain gradient play falls into on toy games. Link this
     back to card 6's rotation;
   - one-sided label smoothing stops D from producing extreme logits, i.e. from becoming overconfident (Goodfellow tutorial §4.2). At present
     the card explains why not to smooth the fakes, but never why to smooth at all.

   Retitle the list "Remedies for collapse and instability".
2. **major: q5 and q8 repeat numbers that are worked in the explainer, so they test recall.** Card 4 states "At D(G(z)) = 0.01 … a factor
   of 99", and card 6 states "From (1,1) with h = 0.1, x²+y² goes from 2 to 2.02". Change the numbers in the MCQs:
   - q5: use D = 0.02. The answer becomes 49, and the distractors are 1, 3.91 (= −log 0.02) and 0.02.
   - q8: start from (1, 2) with h = 0.2. The answer is 5.20. Distractors: 5.00 (the flow invariant), 4.80 (the sign error 1 − h²) and
     6.00 (multiplying by 1 + h).
3. **major: q2 (which D*) is ambiguous as drawn.** B and C differ only in the far-left tail, where both plotted densities are visually zero.
   B is right only because both densities are Gaussian and p_g is wider, so its tail dominates. Nothing in the stem says so: with a
   compact-support p_g, C would be defensible. Fix the prompt to "Two Gaussian densities ($p_{data}$ filled, narrower; $p_g$ wider)…". Also,
   in the explanation, "once near x ≈ 0" refers to an axis with no ticks; say "between the two peaks".
4. **major: explainer[4] skips two steps of the Arjovsky–Bottou derivation and leaves a sign for the reader to fix.**
   (a) Show the one-line identity behind the first bullet: $\log\frac{p_{\theta_0}}{p_{data}}=\log\frac{p_\theta}{p_{data}}-\log\frac{p_\theta}{p_{\theta_0}}$.
   (b) The second bullet discusses $\mathbb E[\log(1-D^*)]$, but the display has $-\mathbb E[\log(1-D^*)]$. State that this term therefore contributes
   $-\nabla\,2\mathrm{JSD}$.
   (c) Name the envelope theorem and give its line: $\frac{d}{d\theta}V(D^*_\theta,\theta)=\partial_\theta V+\partial_DV\cdot\frac{dD^*}{d\theta}$, and
   $\partial_DV=0$ at the optimum. q7 uses "the envelope theorem" in a distractor and in its explanation, but the explainer never introduces the term.
   Also say once that $p_g=p_\theta$, because the card switches between the two names.
5. **minor: f5 uses $f'(0)$, which is never defined.** It is Mescheder's notation, not the lesson's. Drop "($=\pm f'(0)i$ with $f'(0)=\tfrac12$ for the
   standard loss)", or define it as "the slope of the loss at logit 0, ½ here".
6. **minor: f6 has two scope slips.** "Push D toward zero input-gradient on the data" is true for R1 only. Write "on real data (R1) or on fakes
   (R2)". "Where unregularized training, WGAN and WGAN-GP … do not" describes the Dirac-GAN analysis, not Thm 4.1. Write "which unregularized
   training, WGAN and WGAN-GP with fixed critic steps fail on the Dirac-GAN".
7. **minor: explainer[6], scope of Thm 4.1.** Add its main assumption: the realizable case, where some generator matches $p_{data}$ exactly
   and D = 0 near the data (Assumption I).
8. **minor: q10 distractor "The JSD is flat near θ=0, so the generator's gradient vanishes" is half true.** For two Diracs, JSD = log 2 for every
   θ ≠ 0, so the JSD *is* flat. Keep the distractor, but have the explanation concede the first clause: the JSD is flat, but D is not optimal, so
   G still gets non-zero (rotating) gradients.
9. **minor: q11 distractor "would turn the zero-sum game into a non-zero-sum one"** is technically true of *any* label smoothing, one-sided
   included, so it is not wrong; it just isn't the reason. Add a sentence to the explanation: one-sided smoothing already breaks zero-sum,
   so that can't be the reason to keep β = 0.
10. **minor: q6 distractor "Points straight toward the data manifold with unit norm" is implausible.** Replace it with "Matches the non-saturating
    gradient, since the two losses share a fixed point". That misconception appears in the lesson's own "common wrong answers".
11. **minor: q13 asks "What is Goodfellow's explanation…", which tests attribution recall.** Reword the stem as "Which mechanism explains collapse even
    though the minimax solution is $p_{data}$?" and keep the choices.
12. **minor: explainer[7], Goodfellow's argument against the divergence as the cause, is half-told.** He gives two reasons: GANs that approximate
    forward KL collapse too, *and* GANs often collapse to fewer modes than reverse KL would choose, since reverse KL prefers as many modes as
    capacity allows. Add the second reason and cite tutorial §3.2.5 alongside §5.1.1.
13. **minor: explainer[4], disjoint supports.** "Two such sets generically do not overlap" is slightly too strong. A&B Thm 2.2 covers manifolds
    that intersect, provided they don't perfectly align: the intersection has measure zero. Write "generically meet in at most a measure-zero set".
14. **minor: explainer[8] uses "projection discriminator" and "perceptual loss" without defining them.** Add a clause for each: the projection
    discriminator adds an inner product between an embedding of y and D's features to D's logit; a perceptual loss is a distance in a
    pretrained network's feature space.
15. **minor: f12 "Diffusion improves reliably with compute"** isn't in the cited Dhariwal & Nichol §1, which says GANs are "difficult to scale".
    Reword to match the source, or cite something that supports it.
16. **minor, figure `saturating-vs-nonsaturating`:** in the right panel, the orange line runs through the "non-sat." label. Move the label to
    about (0.05, 0.72), or put it below the line.

### Coverage
All 10 plan subtopics are covered, and all plan question ideas appear (q3, q5, q6, q8, q12, f9). Two optional gaps, neither required by the plan:
the "maximum-likelihood game" generator cost (tutorial §3.2.4, named in the plan's sources) and unrolled GANs (tutorial §5.1.1). Add each as one
line at most, or skip them, given the length.

### Figures
All 5 are correct, legible, and generated from the real formulas. The Dirac-GAN panel is simulated on the exact objective, which I re-ran.
The planned `bilinear-dynamics` figure is reasonably replaced by the left Dirac-GAN panel, which shows the same spiral-out vs cycle behaviour.
Captions say what to notice, and the synthetic figures say so.

### Strengths
Real derivations throughout: D*, the JSD, the logit-level saturation argument, Arjovsky–Bottou's KL − 2·JSD, and the bilinear and Dirac-GAN
eigenvalue analysis with R1. Scope and convention notes are good (the PML2 ½ factor, natural logs, function-space vs parameter-space). The MCQ set
has a strong mix of question types, and no length giveaways (the correct choice is strictly longest only in q7, and only by one character).
