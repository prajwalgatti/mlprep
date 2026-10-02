# Content review: gen.latent-variables-elbo, gen.vae

Reviewer: critic agent. Bar: docs/writing-brief.md and docs/review-rubric.md. Plan: content-plan.md B.5 §30 and §40, plus plan-generative-review.md (S3, S4, P1). B.4 (diffusion notation) doesn't apply here, apart from the two-Gaussian KL that gen.vae says diffusion reuses. That formula is consistent with fund.kl-divergence.

## What I verified

**Numbers.** I recomputed everything independently (scratchpad/critic_elbo/check.py, check2.py), using quadrature, closed forms, sympy moments and Monte Carlo. All of them are correct.
- **Naive MC (ELBO card 1).**
  - Posterior N(2.475, 0.0995²).
  - 0.77% of prior mass lies within ±2 sd.
  - Relative variance 153.8, so 15,380 samples.
  - With 10 dimensions: 7.9e23 ≈ 10²⁴.
- **Worked example.**
  - log p = −1.5155.
  - ELBOs −1.919 / −1.669 / −1.766 / −1.516, with gaps 0.403 / 0.153 / 0.250 / 0. Each gap equals the two-Gaussian KL.
  - At the posterior: reconstruction −1.294, KL to prior 0.222.
- **EM toy.**
  - θ goes −2 → 0 → 1 → 1.5.
  - ℓ goes −5.266 → −2.266, and the bound goes to −3.266.
  - The gap is exactly KL(N((θ_old+2)/2, ½) ‖ N((θ+2)/2, ½)) = (θ−θ_old)²/4. Checked at three steps: 1, 0.25, 0.0625.
- **ELBO MCQs.**
  - q2: 0.250, distractor 0.097 = KL to the prior. q2's arithmetic line checks out.
  - q5: 2. q6: 0.64 (1.5625 = Λ_jj).
  - q11: (4⁵−1)/0.01 = 1.02e5, distractors 2.43e4 and 1.5e3.
  - q12: N(1, 0.8).
  - ρ = 0.9 gives v = 0.19 (sd 0.436).
- **Estimators.**
  - Reparameterized variance 4. Score-function variance 30, or 18 with baseline 2.
  - The variance-optimal baseline is b = 4, giving 14, as f11 says.
  - All exact via moments, and confirmed by 4M-sample MC.
- **β.**
  - β = 2σ_x² follows by multiplying SSE/(2σ_x²) + KL by 2σ_x².
  - Learned σ_x² = S/D (derivative checked).
  - The reductions give 784, and 784/32 = 24.5.
- **VAE numbers.**
  - KL 0.5 and 0.318. q1 = 0.943, distractors 0.472 / 0.250 / 0.722 (each reproduces its stated error). q5 = 0.72.
  - tanh(mz/s²) derived from the Bayes ratio exp(2mz/s²).
  - Bernoulli-on-[0,1] "density" integrates to 0.5.
  - IWAE gaps (q = prior, x = 1): 0.404 / 0.0191 / 0.0022 at K = 1 / 10 / 100.
  - Interpolation midpoint norms: 8 vs 5.66.

**Source spot-checks (all supported unless noted).**
- Kingma & Welling 2019 §2.2.1 "Two for One"; §2.5.1 eqs. 2.53–2.55 (L0 + diag σ, log-det = Σ log σ_i).
- Bishop eqs. 10.12–10.15 and the "too compact" remark (§10.1.2).
- Cremer 2018:
  - §1 defines the gaps, and contribution (b) says the decoder accommodates the approximation.
  - §5.2: on MNIST each gap is about half; on Fashion-MNIST and CIFAR the amortization gap is larger. So "often the larger" is fair.
- PML2:
  - §10.1.6 covers the amortization gap and semi-amortized VI (via [Kim+18c]).
  - §21.3.1 eq. 21.23 and the decoder-averaging argument.
  - eqs. 21.21–21.22.
  - §20.3.5 covers interpolation, says nonlinear interpolation is sometimes better, and cites White 2016.
- Kingma & Welling 2014: M = 100, L = 1 (§3, §5); App. C.2 encoder outputs **log σ²**.
- Goodfellow §20.10.3: "causes … not yet known", the ML mass-covering point, and the point that small features get ignored.
- Rombach: KL weight ~10⁻⁶.
- Burda Thm 1: monotone in k, and converges if p/q is bounded.

**Validator.** No errors in either file. One warning: the prereq `fund.autoencoders` doesn't exist yet. That is a plan-level gap, not the writer's.

---

## gen.latent-variables-elbo

**Verdict: revise** (light). It is correct throughout and is the best from-scratch treatment in the area so far. The fixes are mostly small teach-from-scratch insertions and cutting cross-card repetition.

### Findings

#### Major

1. **major (B), whole lesson: "variational" is never defined.** The title promises variational inference, and the user explicitly has never met VI. The term first appears unexplained in explainer[4] ("That is variational inference in miniature"), then in "variational EM" and "variational free energy".
   - Fix: add two sentences to the end of explainer[2] or the start of explainer[3]. "**Variational inference (VI)** turns posterior inference into optimisation: choose a family $\mathcal Q$ of tractable distributions and pick the $q\in\mathcal Q$ that maximises the ELBO. 'Variational' refers to optimising over a function (the distribution $q$), as in the calculus of variations."

2. **major (B), explainer[4]: the ELBO algebra jumps from the ingredients to the final line.** A newcomer can't check $-\tfrac12\log2\pi+\tfrac12+\tfrac12\log s^2-\tfrac12(m^2+(x-m)^2)-s^2$ without redoing it. The cancellation of $-\log 2\pi$ against $+\tfrac12\log 2\pi$, and the two $-s^2/2$ terms adding up to $-s^2$, are invisible. The Gaussian entropy formula is also asserted.
   - Fix: insert three lines before the combined result, one per term:
     - $\mathbb E_q\log p(z)=-\tfrac12\log2\pi-\tfrac12(m^2+s^2)$
     - $\mathbb E_q\log p(x\mid z)=-\tfrac12\log2\pi-\tfrac12((x-m)^2+s^2)$
     - $H(q)=\tfrac12\log2\pi s^2+\tfrac12$, because $\mathbb E_q(z-m)^2=s^2$
   - Then add one prose line: "adding, one $\log 2\pi$ cancels and the two $s^2/2$ terms combine."

3. **major (C), explainer[9] vs explainer[3], [5], [7]: the "Common mistakes" list mostly restates earlier cards.**
   - "Mixing up the two KLs" is already explainer[5] "Don't mix up the two KLs" (and f2's back).
   - "Sign confusion" is already explainer[5] "Names and signs".
   - "'Maximising the ELBO minimises KL(p‖q)'" is already explainer[3] "Which KL" and explainer[7].
   - Fix: delete those three bullets, and keep "higher ELBO ≠ higher likelihood" and "Support". Also delete explainer[3]'s bullet "**Tight iff q = p_θ(z|x)**, matching the Jensen equality condition", which restates explainer[2] "When is it tight?" verbatim in substance. That saves about 80 words and makes room for findings 1–2.

4. **major (B, length), card count 11.** This is mostly earned. Cards 0–8 each carry a distinct derivation or result, and the user named this exact topic as the one to teach from scratch, so don't compress. The real excess is (a) the duplicated pitfalls (finding 3) and (b) explainer[8], which teaches the full Cremer decomposition. gen.vae explainer[7] then re-teaches the same decomposition ("The gap, split"), the same fixes (semi-amortized) and an equivalent MCQ (ELBO q7 vs VAE q11).
   - Recommendation: keep 11 cards, but in explainer[8] cut the displayed Cremer decomposition down to one sentence: "One encoder may not output the best $q$ in the family for every $x$; the resulting shortfall is the **amortization gap** (split out and measured in gen.vae)."
   - Move the semi-amortized and linear-encoder remarks into gen.vae explainer[7]. Move f6 into gen.vae, and keep q7 here (it tests reasoning, not the definition).
   - Net effect: about 120 fewer words and no lost content.
   - Merging the pitfalls card into Key results is the alternative route to 10 cards, but it would blur the quick-revision card. Not recommended.

#### Minor

5. **minor (B), explainer[4]: forward reference to "derived in gen.vae"** (the writer's flag). The two-Gaussian KL is already derived in the prereq fund.kl-divergence ("Gaussian KL in closed form"). Change it to "(derived in fund.kl-divergence)". That is a backward reference to a prereq, and it also covers q2's "You may use…".

6. **minor (B), explainer[0]: the posterior "near z=2.48 with sd 0.1" is used before the Gaussian-posterior calculation (explainer[4]).** Add a parenthetical: "(precisions add: $1+1/0.1^2$; derived in the worked example)".

7. **minor (B), explainer[0]: "needs $154/0.1^2$ samples" skips the step "the mean of N draws has relative variance 154/N; set $\sqrt{154/N}=0.1$".** Add that clause. The same reasoning is reused in q11's explanation, so the card should state it first.

8. **minor (B), explainer[1]: the joint $p_\theta(x,z)$ appears before it's defined.** It is used in "For us, $p_\theta(x)=\mathbb E_q[p_\theta(x,z)/q(z)]$" but only defined in explainer[2] step 1. Define $p_\theta(x,z)=p(z)p_\theta(x\mid z)$ in explainer[0], next to the evidence integral. Also define "proposal" in the importance-sampling paragraph: "the sampling distribution $q$ (the *proposal*)".

9. **minor (B), explainer[6] toy run: "the gap is $(\theta-\theta^{old})^2/4$" is asserted, and so is "the M-step gives $\theta^{new}=\mathbb E_q[z]$".**
   - Add: "maximising $\mathbb E_q[-\tfrac12(z-\theta)^2]$ gives $\theta=\mathbb E_q z$".
   - Add: "the gap is KL between two posteriors with variance ½ whose means differ by $(\theta-\theta^{old})/2$, i.e. $((\theta-\theta^{old})/2)^2/(2\cdot\tfrac12)$".

10. **minor (B), explainer[7]:**
    - $\mathbb E_q[(z-\mu)^\top\Lambda(z-\mu)]=\dots+\sum_j\Lambda_{jj}v_j$ needs its two-word reason: "the cross term has mean zero, and $\mathbb E[\epsilon^\top\Lambda\epsilon]=\mathrm{tr}(\Lambda\,\mathrm{diag}\,v)$".
    - "$1/\Lambda_{jj}$ is the variance of $z_j$ given all the other coordinates" is a standard fact. Add "(Gaussian conditioning; Bishop eq. 2.75)" or similar.

11. **minor (E), q2: figure leak by symmetry.** The stem's figure annotates gap 0.250 for $\mathcal N(1,0.5)$. The asked $q=\mathcal N(0,0.5)$ is its mirror image about the posterior mean, so the answer is printed in the stem.
    - Replacement: $q=\mathcal N(0,0.25)$. Correct answer: **0.347**.
    - Distractors: 0.318 (KL to the prior), 0.653 (reversed KL, KL(post‖q)), 0.403 (the prior's gap).
    - ELBO check: −1.862.

12. **minor (E):**
    - q5's explanation doesn't explain distractor "1", which is the value after one iteration. Add a clause.
    - q6's explanation doesn't explain 0.40 (= 1−ρ). Add a clause.
    - q9's correct answer is the longest choice and the only hedged one ("Nothing definite…"). Shorten it to "Undetermined: B's is higher if its gap exceeds A's by over 2 nats", or hedge one distractor too.
    - Length check: the correct choice is strictly longest in 4/13 (q4, q7, q9, q10). That's acceptable, but q9 is the clear case.

13. **minor (E), flashcards:**
    - f5, f12 and q6 all test the mean-field 1/Λ_jj result with the same 0.19-vs-1 numbers.
    - f8 asks the reader to recall toy numbers.
    - Replace f8 with a reasoning item the deck lacks, e.g. "Why is $\log\frac1K\sum_k w_k$ biased low for $\log p(x)$, and why does it equal the ELBO in expectation at K = 1?"

14. **minor (F), jensen-chord:**
    - The double-headed gap arrow at x ≈ 2.4 is only 0.22 units tall and renders as an illegible asterisk-like blob in every theme. Either drop it (the text already states the gap) or draw a bracket offset to the left with a "0.223" label.
    - em-bound has the same tiny-arrow artefact above the θ = 1 square. Suppress arrows shorter than about 0.3 units.

15. **minor (F/D), missing figure: the plan's `graphical-model`** (z → x generative arrow, with a dashed q(z|x) inference arrow) was dropped. For a reader who has never met latent-variable models it would anchor explainer[0]. It's optional, but cheap with `box`/`arrow`.

16. **minor (C), explainer[2]: "This derivation hides *how big* the gap is. The next card computes it exactly."** Keep it as the motivation for derivation 2, but fold it into the previous paragraph as one sentence.

### Strengths
- Both derivations are line by line, with each step justified in prose. The worked example has every number checkable and recovers the posterior by optimisation. The EM figure is exact and teaches the monotonicity proof visually.
- The questions are mostly compute, predict and which-is-false, with misconception-based distractors. The q7 (amortization measurement) and q13 (variational EM monotonicity) questions are excellent.

---

## gen.vae

**Verdict: revise.** The numbers are correct, coverage is complete, and the toy-based blur derivation is a real asset. There is one wrong claim (BCE "not a bound"), one unstated assumption in the reductions paragraph, three over-length cards, and a pitfalls card that is almost entirely restatement.

### Findings

#### Blocking

1. **blocking (A), explainer[3] and explainer[8]: "The resulting number is then not a bound on any $\log p(x)$" / "the reported 'ELBO' is not a bound".** This is false as stated. $\tilde p(x\mid z)=\prod p_i^{x_i}(1-p_i)^{1-x_i}$ integrates to at most 1 on $[0,1]^D$, so it is at most the normalised continuous-Bernoulli density with the same parameters. The BCE "ELBO" is therefore a (looser) lower bound on that model's $\log p(x)$ (Loaiza-Ganem & Cunningham 2019).
   - Fix: "…is not a normalised likelihood. The missing normaliser depends on the decoder output, so it biases the decoder (toward 0.5-ish outputs) and the reported number is a loose bound that isn't comparable with properly normalised models."
   - Change the pitfalls bullet to match.

#### Major

2. **major (A/B), explainer[3] "Reductions change β silently": the 784× claim silently assumes $\sigma_x^2=\tfrac12$.** For general $\sigma_x^2$ the ELBO loss is proportional to $\text{SSE}+2\sigma_x^2\,\mathrm{KL}$, while mean-MSE + KL is proportional to $\text{SSE}+784\,\mathrm{KL}$. So the KL is over-weighted by $784/(2\sigma_x^2)$, which is 784 only when $\sigma_x^2=\tfrac12$.
   - Fix: state "take $\sigma_x^2=\tfrac12$, so the ELBO's reconstruction term is exactly the SSE" in the card, as q4 does. Otherwise the reader gets a different answer by combining this with the β = 2σ_x² paragraph directly above it.

3. **major (C), explainer[8] "Common mistakes": all 7 bullets restate earlier cards.**
   - KL needs MC: explainer[2] and q2.
   - Reductions: explainer[3] and q4.
   - BCE: explainer[3].
   - ELBO comparison: explainer[7] (and the ELBO lesson).
   - σ_x² is β: explainer[3] and q3.
   - Pathwise = reparam: explainer[1].
   - AE-plus-noise: explainer[0] "The name".
   - The interview chain below then restates them a third time.
   - Fix: cut "Common mistakes" to the two plan-listed source-disagreement items (pathwise vs reparam naming; "AE plus noise" vs amortized VI) and fold them into the chain. That saves about 120 words and gets the card under 300.

4. **major (G), over-length cards: explainer[5] 324 words, explainer[6] 322, explainer[7] 303, explainer[8] 302** (prose only, excluding display math).
   - explainer[5]: move the "AE vs VAE" paragraph to explainer[0], next to "The name", where it belongs conceptually.
   - explainer[6]: trim "Remedies" to one line each. VQGAN and latent diffusion already appear in f12.
   - explainer[7]: absorb the semi-amortized material moved over from the ELBO lesson (ELBO finding 4), and cut the duplicated definition sentence. That makes it net shorter.

5. **major (B), explainer[7] "Where the conflict lives": the conclusion doesn't follow from the displayed equation.** "Because x has far more dimensions than z, reconstruction usually wins over faithful posterior inference" is Murphy's reading of eq. 21.22 (prior-matching vs reconstruction KL). It is not a reading of the displayed eq. 21.21, which has no reconstruction term. "Up to a constant" is also unexplained.
   - Fix: say "the constant is the data entropy $H(p_D)$". Then either show eq. 21.22 as well, or rephrase: "with a limited q the two compete; in practice (Murphy §21.2.4) the optimiser favours fitting $p_\theta(x)$ over posterior accuracy, because…".

#### Minor

6. **minor (A), explainer[0]: "Kingma & Welling output log σ, which is equivalent."** True for the 2019 intro (eq. 2.53). The 2014 paper outputs log σ² (App. C.2). Write "Kingma & Welling (2019) output log σ".

7. **minor (B), explainer[6]: the two-image toy skips the Bayes ratio, and reuses σ for the logistic function.** σ already denotes standard deviations everywhere in this lesson.
   - Add: "the likelihood ratio is $\mathcal N(z;m,s^2)/\mathcal N(z;-m,s^2)=e^{2mz/s^2}$".
   - Write $P(x{=}{+}1\mid z)=1/(1+e^{-2mz/s^2})$ without σ.

8. **minor (B), explainer[1]: "differentiable f and g": g is undefined in the card** (it first appears in f11). Write "$z=g_\phi(\epsilon)$" in the reparameterization display, or say "the map $\epsilon\mapsto z$".

9. **minor (A/source), explainer[5] interpolation (writer's flag).**
   - Cite PML2 §20.3.5. It covers latent interpolation, notes that nonlinear interpolation is sometimes better, and cites White 2016 for slerp.
   - The √d norm argument is the writer's own computation and is correct; label it as such.
   - "Kingma & Welling 2019 §2.7" (Marginal likelihood and ELBO as KL divergences) doesn't support anything in this card. Move that citation to explainer[7]'s conflict paragraph, which it does support.
   - The plan also lists latent *arithmetic* (PML2 §20.3.5, Fig. 20.10). Add one sentence.

10. **minor (A/framing), blur (writer's flag).** explainer[6] frames it correctly ("treat these as complementary"). But the interview-chain answer "*Why blurry?* … the optimal decoder outputs $\mathbb E[x\mid z]$" and f12 ("This is an optimum, not undertraining") present the averaging argument as the whole cause. Goodfellow says the causes are unknown.
    - Fix: add "one leading explanation; Goodfellow et al. list ML's mass-covering and pixelwise likelihoods too" to the chain answer.
    - In f12, scope "optimum" to "for the trained encoder".

11. **minor (B), explainer[3] learned σ_x²:** add the caveat that a learned shared variance can shrink without bound as the SSE goes to 0 (−log p → −∞). In practice it is floored or clamped.

12. **minor (E), MCQs:**
    - q3's correct choice is 101 characters against 70–82 for the others. Trim it to "Reconstruction dominates (smaller β), so prior samples worsen".
    - q11's correct choice uses "can only", an absolute word that flags the false statement. Rephrase it as "Enlarging the encoder network shrinks the approximation gap rather than the amortization gap".
    - q7's choices carry their own justification ("near the prior's mode but far from every training code"), which turns a figure-reading question into a text-matching one. Use bare labels ("A", "B", "C", "None of them") and keep the reasoning in the explanation.

13. **minor (F), estimator-variance:** the caption says "Both are centred on 2", but the score-function histogram's mode is at 0, and its variance lives in a tail that is clipped at 15. Write "Both have mean 2; the score-function estimates have a sharp spike near 0 and a heavy right tail (clipped here) that carries the variance of 30."

14. **minor (D):**
    - The plan mentions categorical (256-way) per-pixel decoders. Add half a sentence in explainer[3].
    - The plan's `latent-space` figure (decoded grid) was dropped. That's acceptable, because aggregate-posterior-holes carries the key idea.

15. **minor (C), cross-lesson:** the general two-Gaussian KL is derived in fund.kl-divergence, explainer[2] here, f3 here and f10's last line. Per plan item 4 it only needs "stating" here. Keep the 1-D standard-normal derivation, and turn the general case into a statement plus a pointer to fund.kl-divergence.

### Strengths
- The reparameterization card has an exactly computed variance comparison (4 vs 30 vs 18) with a matching figure. The training-step code is correct line by line. The aggregate-posterior decomposition is derived, not asserted. The tanh toy turns "blur" into something derivable.
- The questions are well built around real misconceptions: q4 (reduction β), q5 (one-sample ∂σ), q9 (doubled stroke), q13 (β = 0).
