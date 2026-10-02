# Content review: Tuning Playbook, Part A (process, search, batch size)

Reviewed: `PrepApp/Content/applied/sys.tuning-{process,search,batch-size}.yaml`, `tools/figures/sys.tuning-{process,search,batch-size}.py`,
all 9 previews (workbench, plus editorial or terminal), and the plan in `docs/content-plan-tuning.md` Part A.
I checked these claims against the sources: the playbook (lines 113–425, 440–1149, 1645–1798, 1987–2026), McCandlish §2.2–2.4,
Bergstra & Bengio §4, Shallue §1.2, §4.6–4.7, §5, Goyal §2.1–2.2, Zhang 2019 abstract, Kaplan §5.1, Choi §4 and Hoffer abstract.
Every number was recomputed in Python. These all check out: 0.90 linear-scale fraction, $1-0.95^{20}=0.642$, $1-0.95^{60}=0.954$, $N\ge58.4$,
$1-0.9^{10}=0.651$, $1-0.9^{20}=0.878$, $\sigma\sqrt{2/n}=0.0632$ (4.74 SE and 1.58 SE; 2.37 SE in f10), distractors 0.0447 / 0.020 / 0.141,
$40^{1/3}=3.42$, $81^{1/4}=3$, S/E at $B_{noise}/4$, $B_{noise}$, $4B_{noise}$, Goyal 3.2 and $\sqrt{}$-rule 0.566, the overshoot factor $(1+B/B_{noise})/2$,
and the figure readings (η* = 1e-3 vs 5e-3; 0.26 vs 0.30; errors 0.320 vs 0.3045 at the default LR; cliff at $10^{-0.9}$).
`validate.py`: 0 errors. The only warning is the missing prereq `fund.sgd-momentum` (planned, not yet written).

---

## sys.tuning-process

**Verdict: revise**

### Findings
1. **blocking: q5 has two arguably-false statements.** The distractor "Momentum exists only when the optimizer is Nesterov or heavy-ball
   momentum" is false to a careful expert, because Adam and NAdam have a momentum coefficient ($\beta_1$). That gives the question two
   defensible answers. Fix: "Nesterov's momentum coefficient $\gamma$ exists only when optimizer = Nesterov".
2. **major: explainer[3] and q3 restate the playbook's central heuristic ("optimizer hyperparameters interact with almost every other
   change") without saying why.** This is exactly the reasoning the plan asks for. Add two sentences on the mechanism. The usable LR
   range is set by curvature (GD on a quadratic is stable only for $\eta<2/\lambda_{max}$) and by gradient noise ($\epsilon_{opt}$ grows with
   $B/B_{noise}$; see sys.tuning-batch-size). Depth, width, normalization, batch size, augmentation and regularization all move $\lambda_{max}$
   or the noise scale, so $\eta^*$ moves with them. The same sentence explains the figure's "deeper wants a smaller LR", which q2's explanation
   asserts without a reason.
3. **major: too many recall MCQs.** q8 (pick the initial config), q9 (the Adam setup early on) and q11 (which benefit is not listed) are pure
   recall, and q3 is close. The brief allows at most 2. Suggested conversions:
   - q9 becomes a compute/predict question: "Study has 15 trials per arm; per the budget rule, which Adam knobs do you search?" (LR and $\beta_1$).
   - q11 becomes a scenario: "After 5 rounds you see that no change to the LR range has moved the best error by more than seed noise.
     Which exploration benefit is this?" (saturation).
4. **minor: q11 giveaway.** The correct choice "It *guarantees* … global optimum" is the only absolute statement in a "which is not" stem.
   Replace it with something plausible but unlisted, e.g. "It shortens each training run by reducing the step budget".
5. **minor: q12 distractor** "It has no role, because activations do not interact with normalization" contradicts the card ("every
   hyperparameter gets one of three roles"), so nobody will pick it. Replace it with "Nuisance, because its best value may differ with and without BN".
6. **minor: explainer[3] regularizer rule.** The playbook says inclusion is "scientific *or fixed*" (line ~640). The card and f4 say only
   "scientific". Add "or fixed".
7. **minor: forward references.** "Bayesian optimisation" (explainer[2]) and "isolation plot" (incremental-loop figure, f8) are used before
   they are defined (next lesson). Add a 4-word gloss or "(next lesson)".
8. **minor: q6 and sys.tuning-search q1 test the same fact** (a grid gives $N^{1/d}$ distinct values). Keep one. In process, ask instead what
   dropping one depth or fixing $\beta_1$ buys (60 trials/study, or $40^{1/2}\approx6$ values per axis).
9. **minor: figure `fixed-vs-tuned`.** In workbench, `p.label` is the same orange as the 4-layer curve, so "tuned: 8 layers wins" reads as a
   4-layer label. Colour that annotation `p.c(0)` (or `p.fg`). The caption and explainer[4] ("In the figure…") should say the curves are
   synthetic/illustrative.
10. **minor: f9 back** mostly re-lists explainer[2]'s six benefits. Lead with the argument, "a greedy result is invalidated by the next
    change; insight transfers", and cut the list to three items.

### Strengths
The "optimising away" card ($\min_\eta f(s,\eta)$ vs $f(s,\eta_0)$, which disagree exactly when $\eta^*(s)$ varies) is clean, and the figure
encodes it well. Playbook fidelity is high throughout, and the conditional-LR range argument ($\eta g/(1-\gamma)$ vs ≈$\alpha$) is a good
added derivation.

---

## sys.tuning-search

**Verdict: revise**

### Findings
1. **major: the q5 figure gives away the answer.** The stem reuses `axis-plot-boundaries`, whose panel titles say "bad: best at the edge" and
   "good: best inside", and the left panel shades the edge band. Register a quiz variant (e.g. `axis-plot-quiz`) with neutral titles
   ("Study A", "Study B"), no shading and no star, and use that in q5.
2. **major: the synthetic bootstrap figure reads as the playbook's real ResNet-50 result.** `trial-budget-bootstrap` is simulated
   (`rng(11)`, toy quadratic), but its y-range (~23%) and ±0.1 band copy the playbook's ResNet-50/ImageNet numbers, and the next paragraph
   cites that example. Caption fix: "Synthetic study (100 simulated trials) resampled…". Do the same for `isolation-plot` (the playbook's
   Fig. 2 is also ResNet-50 weight decay) and `axis-plot-boundaries`. Also tone down the caption: at k=20 the simulated IQR (0.18) is
   *smaller* than the ±0.1 band; only the 5–95% range (0.48) exceeds it. Say "lucky vs unlucky studies (whiskers) still differ by more than
   seed noise", which is the playbook's actual claim.
3. **major: "low-discrepancy" is never defined** (explainer[0], [2], q12), and q12's explanation defines it wrongly ("each new point fills
   the largest gap, which is what low discrepancy means"). Add one line: the discrepancy of $N$ points is the worst-case gap, over boxes, between
   the fraction of points inside the box and its volume. i.i.d. points have discrepancy $\sim N^{-1/2}$; Halton/Sobol have
   $\sim(\log N)^d/N$, so they leave smaller holes. Also say *why* the bases are distinct primes: coprime bases keep the coordinates from
   being correlated.
4. **major: the grid-vs-random coverage math stops at "distinct values".** Connect it to the hit probability the next card derives.
   Suppose the important axis has a good interval of width $w$ (as a fraction of its range). A grid with $s=N^{1/d}$ values per axis hits it
   with probability ≈ $\min(1, s\,w)$ for a randomly placed interval. Random points hit it with probability $1-(1-w)^N$. Example ($w=0.05$,
   $N=81$, $d=4$): grid ≈ 0.15, random 0.984. This turns the argument into a derivation and makes a good compute MCQ.
5. **major: q4's distractors aren't advantages at all.** "Each trial being a full, independent training run", "log scale" and "categorical
   support" are not advantages of quasi-random search over BO, so the stem nearly answers itself. Use plausible capabilities that
   survive a switch to BO: "Marking divergent trials as infeasible" (Vizier BO supports it), "Searching the LR on a log scale", "Including
   a categorical hyperparameter such as the activation".
6. **minor: explainer[3] item 5** says "Infeasible trials are harmless", but explainer[4] says a large infeasible fraction "wastes budget".
   Reword it as "infeasible trials don't distort later sampling (some BO tools mishandle them)".
7. **minor: bootstrap caveats missing** (explainer[5]). Resampling with replacement from $M$ trials can never beat the study's best, so it is only
   informative for $k\ll M$; that is why the k=30/50 boxes in the figure sit on the floor. The figure also drops divergent trials before resampling,
   although they cost budget in a real study. One sentence covers both.
8. **minor: recall load.** q8, q9 and q12 are mostly recall. Turn q9 into a scenario with numbers, e.g. "8 parallel workers vs 200; which
   algorithm and why".
9. **minor: wording.** In "mirror its digits about the decimal point", say "radix point". Cut "Here is why it matters." (explainer[0]).
10. **minor: source** for the "Choi et al. … for exactly this reason" claim. Choi tune on log scales (§4, App.) but don't state that reason.
    Drop "for exactly this reason".

### Strengths
$1-(1-q)^N$ and the $\sigma\sqrt{2/n}$ yardstick are derived properly, with correct numbers and good distractors (q2, q10). The six
non-adaptivity reasons are explained, not just listed, and the Bergstra details (Sobol +few points at 100–300 trials, LHS ≈ random,
rotated targets) are accurate.

---

## sys.tuning-batch-size

**Verdict: revise**

### Findings
1. **blocking: two different "critical batch sizes" are conflated.** The lesson defines $B_{crit}=E_{min}/S_{min}$ (McCandlish), the point
   where steps *and* examples are both 2× their minima, mid-way through diminishing returns. It then attaches the playbook's claims to it.
   The playbook, Shallue and Zhang use "critical batch size" for the *end of perfect scaling*. As a result, explainer[6] step 2 ("usually below
   $B_{crit}$ and so cuts training time almost for free") is false under the lesson's own definition: at $B=B_{crit}/2$ you already spend 1.5×
   $E_{min}$, and 2× at $B_{crit}$. Fix: add a short convention note in explainer[1]: "the playbook's critical batch = where perfect scaling ends,
   roughly $B\lesssim0.1$–$0.2\,B_{noise}$; McCandlish's $B_{crit}\approx B_{noise}$ = the 2×/2× balance point". Then reword step 2 as "usually in or
   near perfect scaling, so time falls almost for free".
2. **major: the linear-rule failure point depends on an unstated anchor.** In explainer[4], q6 and `lr-vs-batch`, the linear rule
   $\epsilon_{max}B/B_{noise}$ is the small-$B$ tangent of $\epsilon_{opt}$. In practice you anchor at a tuned $B_0$, so the rule is
   $\epsilon_{opt}(B_0)B/B_0=\epsilon_{max}B/(B_0+B_{noise})$, which crosses $2\epsilon_{opt}$ at $B=2B_0+B_{noise}$, not at $B_{noise}$. State the
   tangent assumption and give the general crossing point. In q6, say "eventually passes $2\epsilon_{opt}$". Label the figure line "linear rule
   (anchored at small $B$)".
3. **major: the linear-scaling argument is missing its strongest reason.** Goyal's "k small steps ≈ one big step" only matches the *mean*.
   Add the second half, which is what distinguishes linear from square-root scaling. Over $k$ small steps the noise covariance adds to
   $k\eta^2\Sigma/n$. One big step at $k\eta$ on $kn$ examples has covariance $(k\eta)^2\Sigma/(kn)=k\eta^2\Sigma/n$, the same. So the linear rule
   preserves both drift and diffusion per example seen (Smith's $\eta N/B$ is this statement). The $\sqrt{k}$ rule keeps per-step noise but cuts
   drift per example by $\sqrt k$. For failure (b), replace the bare observation "held only up to ~8k" with the mechanism. The approximation
   needs $w_{t+j}\approx w_t$, i.e. a step small relative to curvature. Once $k\eta$ approaches the curvature limit ($\epsilon_{max}$; divergence near
   $2/\lambda_{max}$), the big step overshoots, while the $k$ small sequential steps "see" the curvature and stay stable. This is the same
   saturation as $\epsilon_{opt}\to\epsilon_{max}$, and it ties the two halves of the card together.
4. **minor: explainer[2] skips the key line.** Show $\epsilon_{opt}=|G|^2/(G^\top HG+\mathrm{tr}(H\Sigma)/B)$ before rewriting it, and
   $\Delta L_{opt}=\tfrac12\epsilon_{opt}|G|^2$ for "substituting back". f8 has these lines but the explainer doesn't.
5. **minor: explainer[3] assumption.** "Suppose every step makes progress $\Delta L_{max}/(1+B_{noise}/B)$, with $B_{noise}$ averaged" also needs
   $\Delta L_{max}$ to be summed over the run. State "treat $B_{noise}$ as constant along the trajectory; McCandlish App. D does the weighted average".
6. **minor: explainer[6] "Past $B_{crit}$, examples per target grow".** Under the model, $E/E_{min}=1+B/B_{noise}$ grows for *every* $B$: slowly
   below $B_{noise}$ (at most 2×), linearly above it. Reword.
7. **minor: explainer[4] calls $\epsilon_{max}$ "the curvature limit".** It is the exact-gradient line-search optimum $|G|^2/G^\top HG$; the loss
   rises past $2\epsilon_{max}$. Say "the curvature-limited step".
8. **minor: underived quantities.** The effective LR $\eta/(1-\gamma)$ (explainer[4]) needs its one-line reason: steady-state velocity
   $\sum\gamma^i g=g/(1-\gamma)$. An interviewer may also ask how to *measure* $B_{simple}$. Add a line: $\mathbb{E}|G_B|^2=|G|^2+\mathrm{tr}\Sigma/B$,
   so two batch sizes give both terms (McCandlish App. A.1).
9. **minor: `steps-vs-batch`.** The regime shading (0.2 and 5 × $B_{noise}$) is arbitrary, and in the model $S\to S_{min}$ only asymptotically.
   Add "regime boundaries illustrative" to the caption. The figure sits beside Shallue's empirical results, so "model curve, not
   measurements" is worth saying explicitly.
10. **minor: questions.** q12 is paper trivia (16 vs 256). Turn it into "predict the effect of switching SGD→Nesterov on the steps-vs-batch
    curve". q11's "Large batches see fewer distinct images per epoch" is implausible. In q13, the explanation should address the LR-decay
    distractor: decay *raises* $B_{noise}$ (McCandlish App. C temperature) rather than shrinking variance.
11. **minor: f8 back** has a ~60-character inline equation for $\mathbb{E}L$. Make it a display `aligned` block for phone width.
12. **minor: throat-clearing.** Cut "This lesson explains why, and where that rule stops working." (explainer[0]).

### Strengths
The McCandlish derivation ($\mathbb{E}[v^\top Hv]$ identity → $\epsilon_{opt}$, $\Delta L_{opt}$, $B_{noise}$, $B_{simple}$ with its
$\mathbb{E}|G_{est}-G|^2$ reading) and the hyperbola $(S/S_{min}-1)(E/E_{min}-1)=1$ are correct and well paced. All three figures are computed
from the real formulas. The Shallue claims (label smoothing, momentum extending perfect scaling, effective LR following no rule,
Transformer raising momentum) all match the source.
