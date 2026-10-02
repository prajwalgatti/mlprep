# Review: fund.information-theory and fund.kl-divergence

Date: 2026-10-02. I read both YAMLs in full, along with plan Part A §90/§100, the plan-review notes, both figure modules and all 8
previews (workbench for each figure, plus editorial or terminal for ce-decomposition, js-saturation and reverse-quiz).
I recomputed every number in Python (scripts are in `scratchpad/critic_ikl/`): all explainer numbers, every MCQ answer and
distractor, the Gaussian-mixture fits, JSD(N(0,1),N(4,1)), and the k1/k2/k3 table (2M-sample simulation).
I spot-checked about 20 citations against the cache. `validate.py` reports 0 errors. Its only warning for these files is the
expected missing prereq `fund.probability-basics`.

## Writer's flagged points (checked first)

| Point | Result |
|---|---|
| "√JSD is a metric" (Endres & Schindelin 2003) isn't in the cache | **The claim is correct.** It is a standard result (Endres & Schindelin, IEEE Trans. Inf. Theory 49(7), 2003). The lesson's matching claim that JSD *itself* fails the triangle inequality is also correct. Check: JSD(δ₀, Bern(½)) = 0.216, twice that is 0.432, which is less than JSD(δ₀, δ₁) = log 2 = 0.693. Keep both claims. Optionally add this 3-point example in one line, so the "JSD isn't a metric" half is shown rather than asserted. |
| nn.KLDivLoss behaviour | **Verified on the local PyTorch 2.1.1.** With B=4 and C=5: `batchmean` = 0.5777, which equals the hand-computed mean of KL(target‖model); default `mean` = 0.1155, which is that value divided by C; `log_target=True` agrees. So input = log-probs, target = probs, and the KL direction is target‖model, all as stated. One caveat: PyTorch emits a warning that "'mean' will be changed to behave the same as 'batchmean' in the next major release". Add "(PyTorch 2.x)" to explainer[10] and f5, and change their `source:` to `PyTorch docs (torch.nn.KLDivLoss), checked in PyTorch 2.1`. |
| VAE-blurriness link worded as a heuristic | The wording is fine, but the lesson omits **cached counter-evidence** for its GAN half. See K-M1 below. |
| k3 is used by GRPO | **Verified.** Shao et al. 2024 §4.1.1, eq. 4 is $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$, attributed to Schulman 2020, i.e. k3 with $r=\pi_{ref}/\pi_\theta$. One caveat is worth a clause: GRPO samples come from $\pi_{\theta_{old}}$ (Alg. 1, line 7), so the estimator is exactly unbiased only while $\pi_\theta=\pi_{\theta_{old}}$. |
| q7 rev 2 (μ=1, σ=0.5 → 0.818 nats) | **Verified**: ½(1+0.25−1+ln 4) = 0.8181. All three distractors map to real errors: 0.5 (mean term only, which is also the explainer's σ=1 example), 0.125 (drops −log σ²), and 0.597 (σ plugged in for σ²: ½(1+0.5−1+ln 2)). `rev: 2` is appropriate, because the id is kept and the answer changed. |

---

## fund.information-theory

**Verdict: revise** (two major, both quick fixes. The teaching itself is strong.)

### Findings

1. **major, `explainer[4]` figure caption (ce-decomposition).** "The blue part never moves" is only right in workbench. In
   editorial, H(p) is orange-red and the *KL* segment is blue, so the caption points at the wrong segment. In terminal, H(p)
   is green. Fix: "The bottom segment, $H(p)$, never moves. Only the top, KL, segment depends on the model…". Don't name colours.

2. **major, Questions replay the explainer's worked numbers.** These MCQs reuse the exact numbers worked in the explainer,
   and in one case shown in a figure, so they test recall rather than reasoning:
   - q3: the 0.4/0.1 table, whose answer 0.278 is printed in the entropy-venn figure.
   - q5: the 10% flip rate, 0.325 nats.
   - q7: $Y=X^2$ on {−1,0,1}, 0.918 bits.
   - q14: XOR.
   - q10: metres→cm, the same ln 100.

   Change the numbers on at least q3, q5 and q7 (verified values):
   - q3: diagonal 0.45, off-diagonal 0.05 → $I=1-H_2(0.1)=$ **0.531 bits**. Distractors: 0.469 ($H(X\mid Y)$), 1.469 ($H(X,Y)$), 0.800 (ρ).
   - q5: flip rate 0.2 → **0.500 nats**. Distractors: 0 (memorize), 0.2 (error rate), 0.693.
   - q7: $X$ uniform on {−2,…,2}, $Y=X^2$ → ρ=0 and $I=H(\tfrac15,\tfrac25,\tfrac25)=$ **1.522 bits**. Distractors: 0, 2.322 ($H(X)=\log_2 5$), and ρ=1 with 1.522.

   q14 can stay as the XOR concept check, but its explanation should say why "0 and 0.5" is wrong. q10's numbers could
   change to km→m (+ln 1000 = 6.91).

3. **minor, `explainer[7]`.** "$I(X;Y\mid Z)\ge0$" is asserted, and the DPI proof rests on it. Add: "It is the average over
   $z$ of $I(X;Y\mid Z{=}z)$, each an ordinary MI, so it is ≥ 0."

4. **minor, `explainer[2]`.** "$H_\Delta\approx h(X)-\log\Delta$" skips its one line. Add
   $H_\Delta=-\sum p(x_i)\Delta\log(p(x_i)\Delta)\approx-\int p\log p-\log\Delta$, using $\sum p(x_i)\Delta\approx1$.

5. **minor, `explainer[6]`.** "Gaussian MI … follows from $h(X)+h(Y)-h(X,Y)$". Give the line (PML1 §6.3.5):
   $h(X,Y)=\tfrac12\log[(2\pi e)^2\sigma_x^2\sigma_y^2(1-\rho^2)]$, so everything cancels except $-\tfrac12\log(1-\rho^2)$.

6. **minor, `explainer[7]` and f10.** "InfoNCE … can never exceed $\log K$" needs a one-line reason: the estimate is
   $\log K$ minus a K-way cross-entropy, which is ≥ 0.

7. **minor, `explainer[7]` and f11.** The sufficient-statistic definition via $I(\theta;s(D))=I(\theta;D)$ treats θ as random
   (the Bayesian view, PML1 §6.3.9). Say so in a clause.

8. **minor, redundancy.** "Perplexities are comparable only … same tokenizer" (explainer[5]) is repeated almost verbatim in
   the Pitfalls list. One of the two can be cut to a few words.

9. **minor, q8.** The correct choice is the longest (56 chars vs 44–53). Shorten it to "No, only the average must be at most
   $H(X)$". Across the topic, only 2 of 14 correct answers are the longest, which is fine.

10. **minor, `explainer[0]`.** Say in a clause that the constant $c$ is the choice of log base (unit). This ties the
    derivation to the units paragraph that follows it.

Coverage: every subtopic-map item (1–10), every question idea, all three planned figures, and both pitfalls are present.
Derivations are given for log-uniqueness, the max-entropy results (uniform and Gaussian), subadditivity, the CE decomposition,
the MI forms, the MI chain rule and the DPI. Nothing is missing.

Spot-checked citations, all confirmed in the cache: PML1 §6.1.5 (perplexity), §6.3.5 (Gaussian MI derivation), §6.3.8
(DPI proof via the two-order chain rule, matching the lesson), §6.3.9 (sufficient statistics), MacKay Thm 5.1 (source coding,
H ≤ L < H+1), Bishop eqs. 1.102 and 1.110 (binned limit; negative for σ² < 1/2πe), PML2 §5.3.6.4 (InfoNCE), d2l §22.11.

Figures: binary-entropy, entropy-venn and venn-quiz are correct and legible. venn-quiz doesn't leak the answer. For
ce-decomposition the numbers are right (1.30/1.35/1.58/2.59 bits); the only problem is the caption (finding 1).

### Strengths
The explainer derives things rather than stating them: the functional-equation route to −log p, Jensen proofs with their
equality cases, and the counterexample of conditioning increasing entropy at one value of y. The ML hooks (CE floor under label
noise, initial loss ln V, bits per byte) are exactly what interviewers ask about.

---

## fund.kl-divergence

**Verdict: revise.** There is no wrong maths, but there are two major scope and evidence gaps, one weak MCQ, and the coordinator's split.

### Findings

**K-M1, major, `explainer[5]` ("GANs: implicitly reverse-ish"), q18 and f12. A cached source contradicts the causal story.**
Goodfellow 2016 GAN tutorial §3.2.5 (`papers/goodfellow2016_gan_tutorial.txt`, around line 975 and lines 1215–1280) says that
f-GANs trained to minimize an approximation of *forward* $\mathrm{KL}(p_{data}\|p_{model})$ still produce sharp samples and
still collapse to a few modes. It also says GANs often use fewer modes than reverse KL would prefer. So mode collapse is
"driven by a factor other than the choice of divergence", namely the training procedure (§5.1.1). Arjovsky & Bottou Thm 2.5 is
what the divergence *predicts*. It is not the established cause.
- Fix in explainer[5]: after the Arjovsky sentence, add "But f-GANs trained on *forward* KL still give sharp samples and
  still drop modes, so Goodfellow (2016, §3.2.5) attributes mode collapse mainly to the training dynamics, not to the
  divergence." Then cite Goodfellow 2016 §3.2.5 in `source:`.
- q18 can stay, because it asks what the term *predicts*. Add one sentence to its explanation with the same caveat.
- The VAE-blur sentence is acceptably hedged.

**K-M2, major, `explainer[5]`, Key results, f2, f6. "Distillation uses forward KL" is too general for an LLM-lab reader.**
Classic soft-label (Hinton) distillation is forward KL. But two cached sources argue for the reverse direction for generative LMs:
MiniLLM (`llm_gu2023_minillm.txt`) distils with reverse KL, and GKD (`llm_agarwal2023_gkd.txt` §2) uses on-policy samples with
reverse KL or a generalized JSD, precisely because reverse KL lets a small student concentrate on the teacher's modes. This is a
natural interviewer follow-up to "which direction where".
- Fix in explainer[5]: add one sentence after the distillation line: "For LLM students, MiniLLM and GKD instead minimize
  reverse KL on the student's own samples, so limited capacity goes to the teacher's main modes (llm.distillation)."
- In Key results and f2, write "classic distillation".

**K-M3, major, q10, distractor "A, since both directions share the optimum for Gaussian q".** It is the same answer as choice A,
it is 59 characters against 1 for the others (an obvious odd-one-out), and the lesson's own text refutes it outright.
Replace it with a real misconception: **"B and C tie, since each sits on one mode"**. The costs are $-\ln0.7=0.36$ vs
$-\ln0.3=1.20$, and the explanation already gives both numbers.

**K-m4, minor, `explainer[9]`, the k2 sentence.** "Its expectation is an f-divergence with f''(1)=1" contradicts the lesson's own
definition (f convex): $f(u)=\tfrac12(\log u)^2$ has $f''(u)=(1-\log u)/u^2<0$ for $u>e$. (Schulman's blog says the same
thing loosely.) Reword: "Its expectation is $\mathbb{E}_q f(r)$ with $f(u)=\tfrac12(\log u)^2$. This f has $f(1)=f'(1)=0$ and
$f''(1)=1$, like KL, so the two agree to second order, although f is not convex, so it isn't a true f-divergence."

**K-m5, minor, `explainer[9]` GRPO sentence.** Add "(samples come from $\pi_{\theta_{old}}$, so it is exactly unbiased only on the
first inner step)". The source is Shao et al. Alg. 1.

**K-m6, minor, `explainer[3]` multivariate Gaussian.** The trace trick is used without saying so. Add:
"$\mathbb{E}[z^\top Az]=\mathrm{tr}(A\,\mathrm{Cov}z)+(\mathbb{E}z)^\top A\,\mathbb{E}z$. Applied to $p$'s own quadratic
form it gives $\mathrm{tr}(I)=d$, which is where $-d$ comes from." The "−d" explanation currently only says it makes KL=0 at p=q.

**K-m7, minor, `explainer[6]`.** $F=\mathbb{E}[ss^\top]=-\mathbb{E}[\nabla^2\log p]$ is asserted. Either add the one line
(differentiate $\mathbb{E}[s]=0$ once more) or add `fund.fisher-information` (order 72) as a prereq. This moves to the follow-on
lesson anyway (see below).

**K-m8, minor, redundancy.** There are two triangle-inequality counterexamples: Bernoulli 0.9/0.5/0.1 in explainer[2], and
N(0,1)→N(2,1) via N(1,1) in explainer[3]. Keep the Gaussian one, which is cleaner and shows the "squared distance" point. In
explainer[2], replace the Bernoulli computation with a forward reference, or keep it and drop the Gaussian sentence.

**K-m9, minor, card length.** explainer[4] is 301 prose words, at the limit, and explainer[5] is 274. If K-M1 and K-M2 add
about 60 words to [5], move the RLHF paragraph's last sentence or the GAN paragraph's Thm 2.5 restatement into q18's
explanation, or move the factorized-Gaussian caveat in [4] to f6, where it is already repeated.

**K-m10, minor, q14** reuses the explainer's Bernoulli(0.5→0.6) check verbatim. When it moves (see the split), change it to
Bern(0.2)→Bern(0.25): $F=6.25$, $\delta=0.05$ → **0.0078**, with an exact value of 0.0070. Distractors: 0.0156 (no ½),
0.0002 (divides by F), 0.156 (δ not squared).

**K-m11, minor, f10** bundles f-divergence generators with the k3 estimator, which are two unrelated facts. The split separates them.

Everything else is correct. Verified numbers:
- 0.511/0.368; 1.758 vs 0.879; 0.318/0.807 (and the figure areas); 0.5 (VAE example).
- Mixture fits: N(0, 4.36); reverse-KL optima 0.692 per mode; the moment-matched fit scores 1.273 (equal weights) and 1.279
  (0.7/0.3, matching q10's "about 1.28"); −ln 0.7/−ln 0.3 = 0.357/1.204.
- Fisher 0.020 vs exact 0.0204; JSD 0.102; JSD(N(0,1),N(4,1)) = 0.633.
- q9 0.223/0.193 (0.322/0.278 bits); q17 N(0,5).
- k1/k2/k3 table: 20×; 1.42× with 0.2% bias; 1.42×; at KL=0.5, k2 bias 25% and k3 1.68×. All match Schulman.
- $r\log r-(r-1)$ is unbiased for KL(p‖q) (simulated 0.5007 vs 0.5) and non-negative.
- The f-generators for KL, reverse KL, JS, TV and χ² are all checked. The JS generator reduces exactly to the mixture form.

Every MCQ has exactly one defensible answer (q11's three "true" statements really are true). The correct choice is never the
longest (0 of 14).

Spot-checked citations, all confirmed in the cache: PML2 §5.1.2.3 (reparameterization invariance), §5.1.3.3 (weight of
evidence), §5.1.4.1–5.1.4.3 (M/I projection), §5.1.9, §2.7.1, §26.2.2 (defines JSD and D*; it does not derive the MI form,
which the lesson derives itself, so that's fine); Arjovsky & Bottou Thm 2.3, 2.4, and 2.5 (exact statement, "fixed for a
value θ₀"); Shao et al. §4.1.1 eq. 4; Schulman blog (k1–k3, the tables, and the $r\log r-(r-1)$ form).

Figures: all four are correct, legible and computed from the real formulas. reverse-quiz gives B and C near-equal heights, so
it doesn't leak the answer. js-saturation correctly shows the open and closed dots at θ=0. The captions say what to notice.

### Strengths
The Gibbs proof with both equality conditions, the derivation of forward KL as moment matching, the reverse-KL ≈ −log w_k
argument with real numbers, and the derivation of JSD as an MI all stay. The "which direction is forced by what you can sample
and evaluate" framing is the best part of the lesson.

---

## Recommended split of fund.kl-divergence

**What moves:** explainer[6] "Locally KL is a quadratic form, and the f-divergence family" and explainer[9] "Estimating KL from
samples". The main lesson keeps 10 cards, 12 MCQs and 10 flashcards (3 open), all within budget.

**New lesson:** for example `fund.kl-estimation`, "Fisher Metric, f-Divergences & Estimating KL", order 102, level intermediate,
W2. Prereqs: `[fund.kl-divergence, fund.fisher-information, fund.monte-carlo]`. Monte-carlo (order 15) owns control variates,
which the k3 derivation relies on.

**What the follow-on needs as its own intro or recap** (one short opening card, about 120 words):
- Recap $\mathrm{KL}(p\|q)=\mathbb{E}_p\log\frac pq$, Gibbs (≥ 0, the $\log t\le t-1$ form, which k3 reuses), and which
  direction RLHF uses: $\mathrm{KL}(\pi_\theta\|\pi_{ref})$ on policy samples.
- Motivation: there is no closed form for KL between two LMs over sequences, and steps in parameter space need a local notion
  of distance. These two questions are what the lesson's two halves answer.
- Notation $r=p(x)/q(x)$ with $x\sim q$, and the fact $\mathbb{E}_q r=1$.
- A one-line score and Fisher recap (score has mean zero; $F=\mathbb{E}[ss^\top]=-\mathbb{E}\nabla^2\log p$), pointing to
  fund.fisher-information. This also fixes K-m7.

Then the two moved cards, in their current order (Fisher/f-div before estimators, because the k2 explanation needs $f''(1)$
locality). Then a short pitfalls/probes card and a key-results card. That gives about 5 cards. Natural additions, so it isn't
thin:
- The general control-variate form $-\log r+\lambda(r-1)$, and why λ=1 is chosen (Schulman).
- Per-token sums vs sequence-level KL for LMs.
- The GRPO $\pi_{\theta_{old}}$ caveat (K-m5).
- The JSD local constant $f''(1)=\tfrac14$, as a worked check.

**Items that move with the cards.** They get new ids in the new file; review history is keyed by topic/item, so it resets
whichever way this is done.
- MCQs: **q14** (Fisher approximation; change the numbers per K-m10) and **q15** (k3 unbiased and non-negative).
- Flashcards: **f9** (local Fisher quadratic) and **f10**. Split f10 into two cards: f-divergence generators, and the k3
  estimator plus $r\log r-(r-1)$.
- Reading: Schulman (2020) moves. Copy "PML2 §2.7" over as well, and trim the main lesson's PML2 note to §5.1.
- The new lesson needs about 8 more MCQs to reach 10, and 1–2 open cards. Ideas:
  - Compute k1 and k3 for a single sample with r=2: −0.693 and 0.307.
  - Which-is-false on k2 being unbiased.
  - Compute TV and χ² for a small pair.
  - Predict how the Fisher approximation's error grows with δ.
  - Explain why k1 averaged over 4 samples can be negative.
  - Open: "Derive k3 and explain why it is always ≥ 0."
  - Open: "Why does every f-divergence look like the Fisher metric locally?"

**Edits needed in the main lesson so nothing dangles:**
- `summary`: drop "the Fisher metric, f-divergences, sample estimators".
- explainer[1]: change "The same bound reappears in the KL estimator later in this lesson" to "…in fund.kl-estimation".
- explainer[7] (JSD): it cites Nowozin Table 1 for the generator, which is fine. Add a pointer: "JSD is also an f-divergence
  (fund.kl-estimation)".
- explainer[10]: move the pitfall "Negative estimates" and the probes "How do you estimate the RLHF KL penalty?" and "What
  does KL look like for small parameter changes?". Replace them with one-line pointers if wanted.
- explainer[11] Key results: move the $\tfrac12\delta^\top F\delta$ display, the $D_f$ bullet and the k3 bullet.
- explainer[5]: optionally add "(estimating it: fund.kl-estimation)" after the RLHF paragraph.
- Plan `content-plan.md` A.1 and §100: move subtopic items 6 (Fisher half), 9 and 10 into the new entry. Also add the new
  lesson as a prereq of `fund.second-order`, which uses the Fisher/KL metric, and as a cross-reference from llm.rlhf-ppo and
  llm.rlvr-grpo for k3.
