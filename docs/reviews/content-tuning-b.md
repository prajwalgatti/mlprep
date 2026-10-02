# Review: Tuning Playbook lessons, batch B

Files: `sys.tuning-steps-schedules`, `sys.tuning-diagnostics`, `sys.tuning-pipeline`, their figure modules, and the
previews (workbench for all 9 figures, plus editorial or terminal for 4). I checked claims against
`tp_tuning_playbook.md` and the cached papers: Zhang 2019 NQM, Smith 2018, Wu 2018, Cohen 2021, Gilmer 2021, Choi 2019
App. A and §4, Wortsman 2023 §3.4, Müller 2019, Shallue 2019 §4.6, Goyal 2017, Zhang 2020 (clipping) and Chinchilla
App. B. I recomputed every number in Python. `validate.py` gives 0 errors. Its only warning for these files is
`fund.batchnorm` missing as a prereq of pipeline, which is outside the writer's control.

## The derivations you asked about

| Item | Maths right? | Assumptions and caveats stated? |
|---|---|---|
| SGD noise floor $\approx\eta\sigma^2/4B$ | **Yes.** I re-derived $v^*=\eta\sigma^2/(Bh(2-\eta h))$ and $\tfrac12hv^*=\eta\sigma^2/(2B(2-\eta h))$. The 0.025 example is correct for $\eta h\ll1$ (it is 0.05 at $h=10$). | **No.** The card doesn't state that the noise is additive, state-independent, constant-variance and independent of $x_t$. It doesn't define $\sigma^2$ (per-example gradient variance) or say that the floor sums over eigendirections ($\approx\eta\,\mathrm{tr}C/4B$). The source is also wrong: Smith §2 has the SDE noise scale, not this recursion. See S1 and S2. |
| $\eta<2/\lambda_{max}$; edge of stability | **Yes** for GD on a quadratic, including the $1<\eta\lambda<2$ oscillating case. | **No.** Nothing reconciles "past 2 it diverges" with Cohen's finding that GD sits *at* $2/\eta$ and keeps decreasing the loss. There's no momentum threshold $(2+2\beta)/\eta$, and no warning that the bound isn't the Adam criterion (Gilmer: preconditioned sharpness, empirical $40/\eta$). Yet the worked example uses Adam. See D1. |
| Adam $\epsilon\to\infty$ → momentum with LR $\alpha(1-\beta_1)/\epsilon$ | **Yes**, at late $t$, relative to the heavy-ball buffer $u$. | **Partly.** "After bias correction" waves away the factor $1/(1-\beta_1^t)$, which Choi absorb into an LR schedule $\alpha_t\propto\epsilon(1-\beta_1^t)$. The figure caption says "LR $\alpha/\epsilon$" (the convention relative to $\hat m$), while the text says $\alpha(1-\beta_1)/\epsilon$. The two conventions are never reconciled. See P2. |
| Label-smoothing gap 4.51 | **Yes.** $\log(0.91/0.01)=\log91=4.511$, with $K=10$, $\varepsilon=0.1$ stated every time it appears. | Yes. The gap is derived via Gibbs/KL. |
| Eval SE 0.42% | **Yes.** $\sqrt{0.77\cdot0.23/10^4}=0.421\%$. The distractors check out: 1.33% (n=1k), 0.04% (n=1M). | **Partly.** The playbook doesn't contain this formula, but the card cites only the playbook. The i.i.d. binomial assumption isn't stated, and neither is the unpaired difference SE (0.59%). See P8. |
| Clipping heuristics | Algebra right: step ≤ $\eta\lambda$; constant clipping = normalized GD; effective LR $\eta\lambda/\|g\|$. | **No.** The derivation holds only for plain SGD, yet it is applied to an Adam workload. "Just an odd way of lowering the LR" contradicts the Zhang 2020 paragraph in the same card. The figure caption contradicts the figure. See D2 and D4. |

**Attribution of the writer's two flags.** Label smoothing: the definition is correctly given in Müller §1.1, and Müller §2
supports the constant logit gap. Calibration and distillation are correctly Müller's. The technique is due to
Szegedy et al. 2016; credit them in one clause. Adam ε: the inclusion proof is correctly Choi App. A, the
$(\epsilon,\alpha/\epsilon)$ search is correctly Choi §4, and the gradient-RMS collapse, the 1e-15 result and the 1e-6
instability are correctly Wortsman §3.4. Choi credit the ε–LR coupling observation to Savarese et al. 2019 (optional).

**Synthetic figures.** No figure claims to be real data. The `grad-norm-clipping` and `checkpoint-selection` captions
read like measurements, though, so add "synthetic". The diagnostics reading note says "the figures this lesson
redraws"; these are synthetic illustrations, not redraws.

**Pink and red in `schedule-families`.** In workbench, cosine (#F07AAE) vs inverse sqrt (#F0616D) gives
ΔE76 ≈ 30. That is distinguishable for normal vision at this size, but it is the same hue family, and the two curves
cross near 6.3k. Under protan/deutan vision they will merge. Editorial is **worse**: linear (#8FC48A) vs inverse sqrt
(#7DBFB2) gives ΔE ≈ 24, and you can see the problem in the preview. **Fix:** draw inverse sqrt in `p.fg` with
`ls="--"`. Its minimum ΔE against the other four is 42 (workbench), 46 (terminal) and 36 (editorial), and the dash
adds a non-colour cue that suits the only family not tied to $T$. Switching to `p.c(4)` doesn't help (ΔE 25 in
workbench and editorial).

---

## sys.tuning-steps-schedules

**Verdict: revise**

### Findings
1. **major (S1), explainer[1], q1–q3, f2, f8: the noise-floor derivation is cited to the wrong source.** Smith 2018 §2
   gives the SDE noise scale $g=\epsilon(N/B-1)$, not the recursion or the $2B(2-\eta h)$ floor. The exact derivation is
   Zhang et al. 2019 §3.1, eqs. (3)–(4), which is cached (`tp_zhang2019`). Wu et al. 2018 §3 (after Schaul 2013) is the
   other match. Change the `source:` lines to "Zhang et al. 2019 §3.1". Keep Smith only for the $\eta/B$ equivalence.
2. **major (S2), explainer[1]: unstated assumptions.** Add 3–4 lines:
   - The noise is additive, with constant variance $\sigma^2/B$ (where $\sigma^2$ is the per-example gradient variance
     along this direction), drawn fresh each step and independent of $x_t$. Zhang call the matching codiagonal-$H,C$
     assumption "nontrivial".
   - In $d$ dimensions the floor sums over directions, $\approx\eta\,\mathrm{tr}(C)/(4B)$. The figure in fact uses 30
     directions, which is why its floor is ≈ 3.75, not 0.25.
   - The "≈" needs $\eta h\ll1$. As $\eta h\to2$ the floor blows up, which links to the stability lesson.
   - Real gradient noise scales with the loss and is correlated with $H$, so treat this as a model.

   Then fix the q1 explanation's "To first order, curvature cancels out of the stationary loss". That holds only
   because the noise is specified directly as gradient variance. In Wu's noisy-minimum model the gradient variance is
   $h^2\sigma^2$ and $h$ doesn't cancel. Say "in this model, curvature cancels".
3. **major, explainer[4] (compute-bound) last paragraph: Chinchilla is used for the opposite of what the playbook uses
   it for.** The playbook cites Hoffmann et al. as *suggesting that decay schedules transfer*, and disagrees ("we don't
   believe this is true in general"), with the rsqrt example. The lesson cites Chinchilla's cycle-length result and
   then says "The playbook doesn't believe schedules transfer". A reader will think the playbook and Chinchilla agree.
   Rewrite it along these lines: "Chinchilla's recipe, a cosine matched to the run, worked across scales, which the
   playbook reads as a suggestion that schedules transfer. It disagrees in general: an rsqrt schedule tuned short and
   run long spends most steps at too small an LR." Also add the playbook's caveat that most schedules are "good
   enough" at extreme budgets, but tuning still pays.
4. **major, q3: figure leak.** The `noise-floor` figure annotates the high-LR plateau with "floor $\propto\eta\sigma^2/B$",
   which is the answer to "why does it flatten?". Render a variant without the text label (keep the dotted lines) and
   use it in the stem.
5. **minor, q9 and explainer[5]: "the smallest LR Round 1 ever used" is false.** Warmup ramps from ~0, as the figure
   shows. Write "below Round 1's final LR (the smallest post-warmup LR it validated)". The explainer's "validated" is
   closer, but say "post-warmup" there too.
6. **minor, explainer[1]: "A larger $\eta$ shrinks it faster"** is true only for $\eta h\le1$. For $1<\eta h<2$,
   $|1-\eta h|$ grows with $\eta$. Add "(for $\eta h\le1$)". The same applies to f8's "faster for larger $\eta$".
7. **minor, explainer[0]: the gradient-variance claim is asserted before it is explained.** Add a forward pointer: a
   larger $\sigma^2$ raises the floor $\eta\sigma^2/4B$, so the run must decay to a smaller LR, which takes more steps.
8. **minor, coverage:** the plan's "step-budget tension" from §"Choosing the initial configuration" is half-covered.
   Add the other side: an oversized initial budget is hard to cut later, once the schedule has been tuned to it.
   Cite that section on card 3 too (for "start constant").
9. **minor, questions:** q4, q8 and q12 are all "what does the playbook say/advise", one more recall item than the
   brief allows. Convert q12 into a predict question, e.g. "you copy the paper's schedule but use 2× the batch;
   what's the likely outcome and why?".

### Strengths
The noise-floor derivation is clean, and it is used well to explain decay, the $\eta/B$ equivalence and short-horizon
bias. The Round 2 rsqrt arithmetic (0.32 → 0.18, two-thirds of steps) and the figure make the playbook's speculative
advice concrete.

---

## sys.tuning-diagnostics

**Verdict: revise**

### Findings
1. **major (D1), explainer[1], f2, f8, interviewer card: $\eta<2/\lambda_{max}$ is presented as a law, not a local
   model.** Add a short "limits of the quadratic picture" paragraph:
   - Cohen et al. find that once sharpness reaches $2/\eta$, GD does **not** diverge. The loss keeps falling
     non-monotonically while sharpness is held near $2/\eta$ (`tp_cohen2021` l.10, l.477). So the threshold marks where
     the local quadratic model predicts blow-up, not where training necessarily fails. The current "past 2 … the loss
     shoots up" followed by "then hovers there" leaves the contradiction unexplained.
   - With heavy-ball momentum $\beta$, the quadratic threshold is $(2+2\beta)/\eta$ (Cohen App.).
   - For Adam the relevant quantity is the preconditioned sharpness $\lambda_{max}(D^{-1/2}HD^{-1/2})$. Gilmer §7 and
     App. say the Adam analogue is unclear, and an empirical $40/\eta$ fit their Transformers.

   This matters because the worked diagnosis and f8 apply the $2/\eta$ story to an Adam Transformer.
2. **major (D2), explainer[4], f4, f10, q9, worked example step 4: the clipping-to-LR equivalence is SGD-only, and it
   is overstated.**
   - "Normalized GD with step $\eta\lambda$" holds for plain SGD (roughly for momentum). Global-norm clipping before
     Adam rescales the inputs to $m$ and $v$, and Adam is approximately invariant to a slowly varying gradient scale.
     So constant clipping under Adam is *not* a lower LR. The worked example is Adam, and its ">50% → lower the LR" is
     applied without comment. State the scope, and for Adam say what the rule rests on: the threshold, rather than the
     LR, has become the knob controlling the steps.
   - "Extremely aggressive clipping is just an odd way of lowering the LR" contradicts the same card's Zhang 2020
     paragraph. Zhang prove that normalized/clipped GD can converge *faster* than any fixed step, precisely because
     its effective LR adapts. Present >50% as the playbook's practical convention ("we would usually consider…
     somehow"), with the actual reasons: magnitude information is lost, $\eta$ and $\lambda$ collapse into one product,
     and the run behaves like a different, untuned optimizer. Don't present it as an equivalence.
3. **major (D3), explainer[3], f3, q7, key results: the "10× the unstable LR" warmup target lacks caveats.** It reads as
   a prescription for the training LR ("Peak = 10×"). Say that:
   - It is the authors' heuristic: "at least one order of magnitude", with 100× as a repeat.
   - The 10× peak is a *probe* for finding the shortest stabilising warmup. The final peak LR must be re-tuned after
     warmup is added and may land well below 10×.
   - In the curvature picture, reaching 10× needs $\lambda_1$ to fall about 10× during warmup. That can fail, and
     warmup doesn't address mid-training instability.
   - For Adam the stability threshold isn't $2/\eta$ (see D1).

   Also add the playbook's 🤖 open questions here and in the clipping card (clipping during warmup; adaptive
   thresholds), as Part A of the plan requires.
4. **major (D4), `grad-norm-clipping` caption: it contradicts its own figure.** The caption says "clips rare spikes and
   leaves typical steps alone". I recomputed the figure: 300 steps are clipped, only 25 of them spikes, and 78% of the
   clipped steps fall in the first 500 steps, where 47% of steps are clipped because the norm is elevated early. A
   90th-percentile threshold clips 10% *by construction*. With drifting norms, a whole-run percentile mostly clips
   early, typical steps. Rewrite it as: "Synthetic norms. A whole-run 90th-percentile threshold clips 10% of steps:
   the spikes, but mostly the elevated early phase. Check how clipping is distributed over time, not just the
   fraction." This is a genuinely useful caveat to teach.
5. **minor, worked example:**
   - Step 3 targets a peak of $2\times10^{-2}$, but the study range is $10^{-4}$–$10^{-2}$. So step 5's "sit *inside*
     the search space this time" needs "after widening the range to include ~$3\times10^{-2}$".
   - Step 4's "check the clipped fraction: about 10% is fine" is vacuous at a 90th-percentile threshold. Say instead
     "recompute the percentile on the new (warmed-up, higher-LR) run, because the norm distribution shifts".
6. **minor, explainer[3]: the "Adam's statistics" warmup explanation has no source** (it is Liu et al. 2020, RAdam).
   Cite it, or mark it as a hypothesis. Bias correction already removes the zero-init bias, and what remains is
   variance.
7. **minor, q2 explanation:** it doesn't address the tempting "stop at its best checkpoint and compare". Retrospective
   selection already does that, and the comparison would still partly measure regularization. Add one sentence.
8. **minor, q1:** the explainer caption maps letters to answers ("A: problematic overfitting"), and the quiz reuses the
   same panel order, so spaced repetition will train "A". Use a panel-shuffled variant in the stem.
9. **minor, questions:** q7, q10 and q13 are recall of playbook rules, one more than the brief allows. Turn q10 into a
   compute question: "threshold at the median clips what fraction; is that 'extremely aggressive'?".
10. **minor, explainer[2], "Infeasible fraction":** cite §"Identifying bad search space boundaries" in the source line.

### Strengths
The curve-reading card and the instability-detection protocol are faithful and well motivated. The Pre-LN/Post-LN
and ReZero reasoning explains *why* the fixes work, not just that they do. The stability-threshold figure is exactly
right.

---

## sys.tuning-pipeline

**Verdict: revise** (one blocking error)

### Findings
1. **blocking, explainer[0]: "SGD ⊆ momentum ⊆ Nesterov-type methods (set $\gamma=0$)" is wrong.** Choi §3 and App. A
   give SGD ⊆ Momentum ⊆ RMSProp, SGD ⊆ Momentum ⊆ Adam, and SGD ⊆ Nesterov ⊆ NAdam. Setting $\gamma=0$ shows that
   SGD is included in *each*. Momentum and Nesterov don't include each other. Replace the bullet with those three
   chains.
2. **major (P2), explainer[1] figure caption vs text and key results: momentum LR conventions clash.** The caption says
   "momentum SGD with LR $\alpha/\epsilon$", and the text and key results say $\alpha(1-\beta_1)/\epsilon$. Both are
   right in different conventions: $\alpha/\epsilon$ is relative to the EMA $\hat m$, and $\alpha(1-\beta_1)/\epsilon$
   is relative to the heavy-ball buffer $u$. Fix the caption to "…momentum with LR $\alpha/\epsilon$ on the EMA
   $\hat m$, i.e. $\alpha(1-\beta_1)/\epsilon$ on the heavy-ball buffer". Add one sentence to the text.
3. **major, explainer[1], q2: "after bias correction" hides a step.** With the playbook form
   $\alpha\,b_t\,m/(\sqrt v+\epsilon)$ and large ε, the step is $\alpha b_t(1-\beta_1)u/\epsilon$, where
   $b_t=\sqrt{1-\beta_2^t}/(1-\beta_1^t)$. So Adam is momentum with a **time-varying** LR
   $\alpha(1-\beta_1)b_t/\epsilon$, which tends to $\alpha(1-\beta_1)/\epsilon$ as $t\to\infty$. Choi make it exact by
   choosing $\alpha_t\propto\epsilon(1-\beta_1^t)$ (App. A, with $\beta_2=0$ for convenience). Also say that the
   inclusion requires α to grow with ε. At fixed α the step tends to 0. And note that the derivation switches from the
   playbook form ($\epsilon$ added to uncorrected $\sqrt v$) to the $\hat m/(\sqrt{\hat v}+\epsilon)$ form. Pick one.
4. **major, q5: a "true" distractor is arguably false.** "Adam approximates momentum arbitrarily well as
   $\epsilon\to\infty$" is false at fixed α, since the step vanishes. Write "…as $\epsilon\to\infty$ with
   $\alpha/\epsilon$ scaled appropriately".
5. **major, q7: giveaway.** The stem says "only device 0's running averages are saved", and the correct choice repeats
   it. Rewrite the stem to give the symptom ("eval after reload is worse than the in-training eval; BN statistics are
   per-device and unsynced"). Or ask a reasoning question: "why is averaging the per-device EMAs at checkpoint time
   exact?", answered by EMA linearity.
6. **major, explainer[2]: "Shallue et al. found label smoothing helped only at large batches" overgeneralizes.** That
   held for ResNet-50/ImageNet. On MNIST and Fashion-MNIST it helped at all batch sizes (Shallue §4.6). Say "for
   ResNet-50 on ImageNet". Adjust f10's "matters more at large batch" the same way.
7. **minor, explainer[1]: Wortsman's 1e-6 result looks backwards without comment.** Raising ε *shrinks* steps, yet it
   caused an instability. Add "counter-intuitively; the paper doesn't give a mechanism" so the reader isn't left
   puzzled.
8. **minor (P8), explainer[5], q12:**
   - The SE formula isn't in the playbook. Add a source, e.g. `llm_miller2024_error_bars` (cached) or "binomial SE".
   - State the i.i.d.-examples assumption, and that this is sampling noise only: seed and training variance come on
     top.
   - Give the unpaired difference SE: $\sqrt2\times0.42\approx0.59\%$.
   - In q12's explanation, "0.18% would need n≈50,000" should be ≈55,000 (it is 0.188% at 50k).
9. **minor, explainer[3], q9: the duplicate-examples claim assumes every host reads the same unsharded stream.** With
   per-host file shards (which the same card recommends), a shared shuffle seed doesn't duplicate examples; it only
   correlates augmentation. Add "each host reads the full dataset" to the q9 stem and the explainer.
10. **minor, explainer[4] item 5, q10: "silently multiplies the LR by k" is for SGD.** Under Adam the scale mostly
    cancels, and it shows up only through ε, clipping thresholds and L2-style decay. One clause fixes it.
11. **minor, attribution:** credit Szegedy et al. 2016 for label smoothing. Change the f4 and q6 sources to "Müller et
    al. 2019 §1.1, §2".
12. **minor, figures:**
    - `ghost-bn` draws arrows from the whole device box to the EMA. It doesn't show per-ghost-batch statistics, or that
      the averaging happens only at checkpoint time. Add a "μ,σ² per 64" arrow from each ghost box, and label the top
      box "at checkpoint: average EMA 0, EMA 1".
    - Mark `checkpoint-selection` as synthetic in its caption.
13. **minor, format:** the "Setting up periodic evaluation" card is 312 words. Trim the three-settings list to one line.

### Strengths
The label-smoothing derivation, with $K$ and $\varepsilon$ explicit, is interview-ready. The multi-host checklist gives a
*reason* for every rule (the seed rule especially). The ε-collapse arithmetic ($r/(r+\epsilon)=\tfrac12$) is well tied to
Wortsman.
