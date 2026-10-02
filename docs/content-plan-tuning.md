# Content plan: Deep Learning Tuning Playbook

This plan covers how the **Google Research Deep Learning Tuning Playbook** (Godbole, Dahl, Gilmer, Shallue & Nado, 2023,
v1.0, https://github.com/google-research/tuning_playbook) enters the app. The playbook is licensed **CC-BY 4.0**, so we may adapt it but
must attribute it. Write in your own words and cite it in `source:` lines as `Tuning Playbook §"<section heading>"`.

It has two parts.
- **Part A** lists the dedicated lessons (`area: applied`, ids `sys.tuning-*`). They teach the playbook's core material, which no other
  topic owns. The Applied ML curator has agreed that these ids belong to this plan. They are written and live in `PrepApp/Content/applied/`.
- **Part B** is an **insertions table**. Each row maps a remaining playbook point to the planned topic that should cover it, with a one-line note
  on what to add. **Writers of `fund.*`, `llm.*` and `sys.*` topics: check Part B for rows that name your topic.**

## Sources

- Source cache: `$SRC = sources/`.
  Read `$SRC/INDEX-tp.md` first.
- `tp_tuning_playbook.md` is the playbook README as raw markdown. `grep -n '^##'` lists its sections, and INDEX-tp.md gives their line numbers.
- Supporting papers, cached as `tp_*.txt`: Shallue et al. 2019 (batch size), Bergstra & Bengio 2012 (random search), Choi et al. 2019
  (optimizer comparisons), Gilmer et al. 2021 (curvature and instability), Cohen et al. 2021 (edge of stability), Zhang et al. 2019 (NQM),
  Smith et al. 2018, Loshchilov & Hutter 2017 (SGDR), Wu et al. 2018 (short-horizon bias), Hoffer et al. 2017 (ghost BN), Bachlechner et al. 2020
  (ReZero), Kaplan et al. 2020, Nado et al. 2021.
- Already cached by other curators: McCandlish et al. 2018 (`llm_mccandlish2018_critical_batch.txt`), Goyal et al. 2017
  (`papers/goyal2017_large_minibatch.txt`), Xiong et al. 2020 (`papers/xiong2020_preln.txt`), Müller et al. 2019 (label smoothing),
  Wortsman et al. 2023 (`sys_wortsman2023_small_scale_instabilities.txt`).

**The playbook gives advice, not derivations.** Wherever a lesson states one of its heuristics, it also gives the *reason* the heuristic works, and
it takes that reason from a supporting source. Examples: the McCandlish noise-scale derivation explains why the best learning rate grows with batch
size; Bergstra & Bengio's low-effective-dimensionality argument explains why quasi-random search beats grid search; GD stability on a quadratic
($\eta<2/\lambda_{max}$) explains instability and warmup; the stationary variance of SGD on a noisy quadratic explains why schedules decay.
Where the playbook flags something as open (🤖), the lesson says so.

---

## Part A: Dedicated lessons (`area: applied`)

| order | id | title | level | prereqs |
|---|---|---|---|---|
| 310 | `sys.tuning-process` | Tuning Playbook: The Incremental Strategy | intermediate | fund.adam |
| 320 | `sys.tuning-search` | Tuning Playbook: Search and Reading Studies | intermediate | sys.tuning-process |
| 330 | `sys.tuning-batch-size` | Tuning Playbook: Choosing the Batch Size | intermediate | sys.tuning-process, fund.sgd-momentum |
| 340 | `sys.tuning-steps-schedules` | Tuning Playbook: Step Budgets and LR Schedules | intermediate | sys.tuning-batch-size |
| 350 | `sys.tuning-diagnostics` | Tuning Playbook: Diagnosing Training Failures | intermediate | sys.tuning-steps-schedules |
| 360 | `sys.tuning-pipeline` | Tuning Playbook: Optimizer Settings and Pipeline Hygiene | intermediate | sys.tuning-search, fund.adam, fund.batchnorm |

The task suggested three lessons (process, batch size and steps, diagnostics). Six are needed to keep each one to 6–10 cards without compressing
derivations. Process splits into *designing* experiments and *searching and reading* them. Batch size splits from step budgets and schedules,
because each has its own derivation (critical batch size; the SGD noise floor). The pipeline and FAQ items (Adam tuning, BN details, multi-host
pipelines, evaluation, checkpoints) form their own lesson.

Orders 310–360 sit after the general systems topics (`content-plan-applied.md` uses 10–300; `sys.frameworks` is 300). The Applied curator may renumber them, but should keep them contiguous and in this order.

### 310 · `sys.tuning-process`: The Incremental Strategy
- **Playbook sections**: "Guide for starting a new project" (architecture, optimizer, initial configuration), "The incremental tuning strategy",
  "Exploration vs exploitation", "Choosing the goal…", "Identifying scientific, nuisance, and fixed hyperparameters", "Creating a set of studies",
  "Striking a balance between informative and affordable experiments", and the "metaparameter" FAQ.
- **Syllabus (cards)**
  1. The problem. Tuning is a confounded experiment with two failure modes. You can adopt a change that only won by luck, which adds complexity,
     or you can compare settings unfairly and reach a wrong scientific conclusion. A note on the terms hyperparameter and metaparameter.
  2. The starting point. Reuse an architecture that works (choosing one means choosing a *family* of models). Use the most popular optimizer for the
     problem type, with its extra knobs fixed at first. Choose an initial configuration that is simple, fast and cheap and gets a "reasonable"
     result (constant LR, small model). The tension in setting the step budget.
  3. The incremental loop. It has four steps (goal → design → learn → launch?). A "launch" needs strong evidence. Exploration (insight) gets most of
     the time, and the card lists the six things that insight buys.
  4. Scientific, nuisance and fixed hyperparameters. Definitions, the depth example, and why a hyperparameter's role depends on the goal (the
     activation-function example). Rules of thumb: optimizer hyperparameters are nuisance; the choice of optimizer is scientific or fixed; whether to
     include a regularizer is scientific while its strength is nuisance; architecture is scientific or fixed because it changes cost.
  5. Worked example, "optimizing away". Write the comparison as $\min_\eta f(s,\eta)$, not $f(s,\eta_0)$. A fixed LR can reverse the conclusion
     because $\eta^*(s)$ depends on $s$. Converting nuisance to fixed adds caveats, and the cost grows with the interaction (weight decay vs model size).
  6. Studies, trials, search spaces and conditional hyperparameters. One study per scientific value, or one mixed study that must sample the
     scientific value uniformly. "learning_rate" under Adam and under SGD are different hyperparameters.
  7. The budget across three desiderata (how many scientific values, how wide the nuisance ranges, how dense the sampling). Worked arithmetic on
     trials per study and samples per axis.
  8. What interviewers probe.
  9. Key results.
- **Figures**: `fixed-vs-tuned`: validation error vs LR for a shallow and a deep model; at the default LR the shallow model looks better, but the tuned
  deep model wins. `incremental-loop`: a process diagram of one round of experiments.
- **Question ideas**: classify each hyperparameter for a stated goal; predict the result of comparing at a fixed LR; why optimizer hyperparameters are
  rarely scientific; dropout inclusion vs dropout rate; why fixing weight decay while comparing model sizes is misleading; (calc) trials per study
  and distinct values per axis; conditional hyperparameters; what qualifies a launch.

### 320 · `sys.tuning-search`: Search and Reading Studies
- **Playbook sections**: "Creating a set of studies" (algorithm choice), "Extracting insight from experimental results", "Identifying bad search space
  boundaries", "Not sampling enough points…", "Detecting whether a change is useful with isolation plots", "Automate generically useful plots",
  "Determining whether to adopt…", "After exploration concludes", and the FAQs on quasi-random search, its implementations, and how many trials.
- **Supporting**: Bergstra & Bengio 2012 §1, §4 (low effective dimensionality, Sobol/Halton, $1-(1-v/V)^T$); Choi et al. 2019 §4 (log scales,
  validating boundaries).
- **Syllabus**
  1. The search problem: black-box, noisy, expensive and parallel, and the objective may change after the fact. Log-scale parameterisation and why
     to use it (the effects are multiplicative).
  2. Why grid search wastes trials. Derive it from $f(x,y)\approx g(x)$: an $s^d$ grid tests only $s$ distinct values per axis, while $N$ random
     points test $N$.
  3. Random vs quasi-random search. Derive $P(\text{hit top-}q)=1-(1-q)^N$, with numbers. How a Halton sequence works (radical inverse in prime
     bases). Why low-discrepancy points leave smaller holes ("a jittered, shuffled grid").
  4. Quasi-random search during exploration and Bayesian optimisation afterwards, with the six reasons spelled out. When BO is worth it, and how to
     handle divergent trials. Test set checks, and folding validation into training only for one-off workloads.
  5. Axis plots and search-space boundaries; infeasible trials (an edge of divergence points to instability; a large infeasible fraction points to
     a reparameterisation or a bug). Automate the plots.
  6. Have we sampled enough? Bootstrap best-of-$k$ from a finished study, and compare the spread to the retrain variance.
  7. Isolation plots (approximated by bucketing), the three sources of variance, and a statistical yardstick for adopting a change
     (std error of a difference).
  8. What interviewers probe. 9. Key results.
- **Figures**: `grid-vs-quasirandom` (9 trials each, with the projection onto the important axis); `axis-plot-boundaries` (best point at the edge vs
  inside, with infeasible ×'s); `trial-budget-bootstrap` (box plots of best-of-$k$ for $k$ = 2…50 from 100 synthetic trials); `isolation-plot`
  (bucketed best per weight-decay bin vs all trials).
- **Question ideas**: (calc) distinct values per axis for a 3⁴ grid vs 81 random points; (calc) $1-0.95^{60}$; which advantage of quasi-random search
  disappears in the exploitation phase; read a figure and say whether the boundary is bad; what an infeasible cliff next to the best LR means;
  study variance vs trial variance; (calc) std error of a difference with 5 seeds.

### 330 · `sys.tuning-batch-size`: Choosing the Batch Size
- **Playbook sections**: "Choosing the batch size" and all its subsections, plus the FAQ "Why shouldn't the batch size be tuned to directly improve
  validation set performance?".
- **Supporting**: Shallue et al. 2019 §2.2 (effective LR), §4.1, §4.4, §4.6–4.8, §5; McCandlish et al. 2018 §2.2–2.4; Goyal et al. 2017 §2.1–2.2;
  Smith et al. 2018 §2; Zhang et al. 2019 (preconditioning extends the critical batch size); Kaplan et al. 2020 §5.1.
- **Syllabus**
  1. Batch size is a speed knob: time = time-per-step × steps. Find feasible batch sizes with a power-of-2 sweep, and measure throughput.
     Bottlenecks; gradient accumulation gives no throughput.
  2. Steps-to-result vs batch size has three regimes (perfect scaling, diminishing returns, maximal data parallelism). Larger batches never
     increase the steps needed, provided everything is re-tuned.
  3. Derive the knee: the second-order expansion with a noisy gradient gives $\epsilon_{opt}(B)$, $\Delta L_{opt}(B)$, $B_{noise}$ and $B_{simple}$.
  4. Derive the steps/examples tradeoff $(S/S_{min}-1)(E/E_{min}-1)=1$; $B_{crit}=E_{min}/S_{min}$; at $B_{crit}$ both are doubled. Numbers.
  5. Why the best LR grows with batch size: linear growth in the noise-dominated regime, from $\epsilon_{opt}(B)$; Goyal's "k small steps ≈ one big
     step" argument and where it breaks (early training → warmup; past the critical batch). Linear vs square-root rules (keeping $\eta/B$ fixed vs
     keeping update variance fixed). Shallue: neither rule holds universally; tune momentum jointly; effective LR $\eta/(1-\gamma)$.
  6. Batch size is not a validation hyperparameter. Re-tuned LR, regularization and steps erase the gap. Small-batch noise regularizes, so larger
     batches may need more regularization (label smoothing, Shallue §4.6). Epoch budgets vs step budgets bias the comparison.
  7. Resource consumption (three cases), upfront costs, and the practical rule: use the largest batch that fits and still reduces time. Momentum and
     Adam extend perfect scaling. $B_{crit}$ grows as loss falls. A changed batch means re-tuning.
  8. What interviewers probe. 9. Key results.
- **Figures**: `steps-vs-batch` (log-log $S/S_{min}$ and $E/E_{min}$ against $B/B_{noise}$, with the regimes labelled); `lr-vs-batch` ($\epsilon_{opt}$
  against the linear rule); `time-compute-tradeoff` (the $S$–$E$ hyperbola with points marked at several batch sizes).
- **Question ideas**: (calc) $S/S_{min}$ and $E/E_{min}$ at $B=B_{noise}/4$ and $4B_{noise}$; (calc) Goyal LR at 8192; why epoch-budget studies favour
  small batches; what happens to time per step when the accelerator saturates; which is false about gradient accumulation; the source of the
  "generalization gap" claim; why the linear rule fails early in training.

### 340 · `sys.tuning-steps-schedules`: Step Budgets and LR Schedules
- **Playbook sections**: "Determining the number of steps for each training run" (all), "Round 1", "Round 2", the schedule FAQs (best family, default,
  complicated schedules), and "Choosing the initial configuration" (the step-budget tension).
- **Supporting**: SGD on a noisy quadratic (stationary variance $\eta\sigma^2/(B h(2-\eta h))$; see also `fund.sgd-momentum`); Smith et al. 2018
  (noise scale $\eta N/B$); Loshchilov & Hutter 2017 §3 (cosine); Wu et al. 2018 (short-horizon bias); Hoffmann et al. 2022 (cosine length should
  match the run).
- **Syllabus**
  1. Compute-bound vs not compute-bound, and what each implies. Anything that adds gradient variance (smaller batch, augmentation, dropout) needs
     more steps.
  2. Derive why the LR should decay: the noise floor $\approx\eta\sigma^2/(4B)$ against the contraction $(1-\eta h)^t$.
  3. Schedule families with formulas (constant, step, linear, cosine, inverse square root, each with a warmup prefix). Playbook default: linear or
     cosine. Some non-constant schedule matters; which family is best is an open question.
  4. `max_train_steps` when not compute-bound: never tune it inside a study; use retrospective checkpoint selection and read where the best step
     falls; the LR-sweep algorithm for the initial value; the self-deception trap.
  5. Compute-bound: tune in rounds, with the transfer list (warmup/init > architecture > optimizer/regularization/augmentation > schedule).
     Short-horizon bias.
  6. Round 2 schedule extension, worked: inverse-sqrt decay extended 10×; linear (keep the decay length, extend the constant phase); cosine (stretch
     to the new horizon).
  7. Complicated piecewise schedules in papers: copy the algorithm, not the schedule.
  8. What interviewers probe. 9. Key results.
- **Figures**: `noise-floor` (simulated SGD on a noisy quadratic: high constant, low constant, decayed); `schedule-families`; `round2-extension`.
- **Question ideas**: (calc) the noise floor halves when $\eta$ halves or $B$ doubles; where the best checkpoint falls implies whether to train
  longer; which hyperparameters transfer from short runs; what goes wrong when you stretch an rsqrt schedule; how to extend a cosine schedule;
  why not to copy a paper's piecewise schedule.

### 350 · `sys.tuning-diagnostics`: Diagnosing Training Failures
- **Playbook sections**: "Examining the training curves", "How can optimization failures be debugged and mitigated?" (identifying instability,
  potential fixes, warmup: when and how, gradient clipping), and "Setting up periodic evaluations" (periodicity as a bug signal).
- **Supporting**: Gilmer et al. 2021 (λ₁ vs 2/η, warmup lowers curvature); Cohen et al. 2021 (progressive sharpening); Goyal et al. 2017 §2.2;
  Xiong et al. 2020 (Post-LN gradients at init); Bachlechner et al. 2020 (ReZero); Wortsman et al. 2023 (qk-layernorm, z-loss, Adam ε).
- **Syllabus**
  1. Read curves, not single numbers. Problematic overfitting, and the selection bias it creates (a small LR acting as a regularizer). Late
     step-to-step variance: causes and remedies. Still improving vs saturated early. Training loss rising means a bug; periodic validation metrics
     mean leakage or a shuffling bug.
  2. Derive instability: GD on a quadratic is stable only if $\eta<2/\lambda_{max}$. Sharpness, progressive sharpening and the "edge of stability".
     Instability only matters when it forces a small LR. Early vs mid-training instability.
  3. Detecting it: LR sweep, then curves just above $lr^*$; an axis plot with an infeasible cliff; spikes after a steady decline; log the gradient norm;
     run about 500 steps at $2\times lr^*$ while evaluating every step.
  4. Warmup: why it works (a rapidly changing network breaks the linear-scaling assumption; self-stabilization lowers curvature) and the protocol
     (base = 10× the unstable LR; sweep warmup over orders of magnitude, ≤10% of steps; post-warmup = 2× warmup; prepend it and add the steps).
  5. Gradient clipping: $g'=\lambda g/\|g\|$ preserves direction; threshold near the 90th percentile of norms; if more than 50% of steps are clipped,
     you are effectively running normalized GD with step $\eta\lambda$, so lower the LR instead.
  6. Architecture and optimizer fixes: residual connections and normalization; Pre-LN $x+f(\mathrm{Norm}(x))$ vs Post-LN $\mathrm{Norm}(x+f(x))$ and why
     it matters; zero-initialized residual branches; switching to Adam; lowering the LR as a last resort. Transformer extras (pointer to
     `llm.training-stability`).
  7. A worked diagnosis from an LR sweep through to the fix.
  8. What interviewers probe. 9. Key results.
- **Figures**: `curve-gallery` (2×2: problematic overfitting, noisy late, still improving, saturated early) **[fig-Q]**; `stability-threshold`
  (loss on a quadratic for $\eta\lambda$ = 0.5, 1.5, 1.95, 2.05); `grad-norm-clipping` (histogram of gradient norms with the 90th-percentile threshold).
- **Question ideas**: figure question "which panel shows problematic overfitting?"; (calc) the maximum stable LR for $\lambda_{max}=50$; what the
  playbook's warmup protocol prescribes; (calc) total steps after prepending warmup; aggressive clipping is equivalent to what; Pre-LN vs Post-LN; why
  early instability can be invisible; periodic validation accuracy.

### 360 · `sys.tuning-pipeline`: Optimizer Settings and Pipeline Hygiene
- **Playbook sections**: "Choosing the optimizer", the FAQs "How should Adam's hyperparameters be tuned?" and "What are the update rules…",
  "Batch normalization implementation details", "Considerations for multi-host pipelines", "Evaluating model performance" (all),
  "Saving checkpoints and retrospectively selecting the best checkpoint", "Setting up experiment tracking", "Optimizing the input pipeline",
  and regularizers (label smoothing, dropout, weight decay) as nuisance hyperparameters.
- **Supporting**: Choi et al. 2019 (inclusion hierarchy, ε coupling, App. A proof that Adam → momentum as ε → ∞); Wortsman et al. 2023 §3.4
  (gradient RMS ≈ ε shrinks updates); Hoffer et al. 2017 (ghost BN); Goyal et al. 2017 §2.3, §3 (BN n=32, loss normalisation); Müller et al. 2019
  and Shallue et al. 2019 §4.6 (label smoothing).
- **Syllabus**
  1. The optimizer inclusion hierarchy and why rankings depend on the tuning protocol. Update rules and the effective LR.
  2. Tuning Adam: which hyperparameters to tune at which budget, and why ε is not an independent knob. Derive the large-ε limit (momentum SGD with
     LR $\alpha/\epsilon$), so search $(\epsilon,\alpha/\epsilon)$ jointly. Small ε when gradient RMS is tiny.
  3. Regularizers as nuisance hyperparameters: label smoothing (derive the optimal logit gap $\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$),
     dropout and weight decay; cheap fixes for problematic overfitting.
  4. BatchNorm implementation details: per-device statistics, ghost BN (~64 examples), decoupling the BN batch from the gradient batch, subsampling
     when the per-device batch exceeds the virtual batch, synchronizing the EMA before checkpoints.
  5. Multi-host pitfalls: log and checkpoint on one host; sync BN statistics; the RNG seed rule (same for init, different for data) and why; shard
     data files; normalize the loss by the global batch.
  6. Periodic evaluation: step intervals, eval batch ≥ train batch, zero-weighting padded examples, sample size (std error of accuracy),
     imbalanced classes.
  7. Checkpoints and retrospective selection, experiment tracking, and input-pipeline bottlenecks.
  8. What interviewers probe. 9. Key results.
- **Figures**: `adam-epsilon` (per-coordinate step size vs gradient RMS for ε = 1e-8, 1e-4, 1e-2); `checkpoint-selection` (a noisy validation curve
  with the best checkpoint vs the last); `ghost-bn` (devices → per-device batch → ghost batches → statistics, with EMA sync at checkpoint time).
- **Question ideas**: Adam tuning order at a 15-trial budget; (calc) step shrink when ε = gradient RMS; which is false about the inclusion hierarchy;
  (calc) label-smoothing logit gap for K=10, ε=0.1; the RNG seed rule; (calc) std error of 77% accuracy on 10k examples; the bias from padded examples;
  why to evaluate at step intervals.

---

## Part B: Insertions into other planned topics

Ids follow `content-plan.md` (fundamentals), `content-plan-applied.md` (systems) and `content-plan-llms.md` (LLMs); all were checked against
those files. Following the rule agreed with the LLM plan, a `sys.tuning-*` lesson owns the generic derivation, and the `llm.*` lesson keeps only
the transformer-specific numbers and links back. "→ link" means one or two sentences plus a cross-reference to the dedicated lesson; don't re-teach the material.
Cite as `Tuning Playbook §"…"`.

### Fundamentals

| Playbook point | Topic | What to add |
|---|---|---|
| GD stability $\eta<2/\lambda_{max}$ as the cause of training instability | fund.gradient-descent | One paragraph tying the "η > 2/L" picture to deep nets: sharpness rises during training (progressive sharpening, Cohen et al. 2021), and divergence happens when $\lambda_1>2/\eta$ (Gilmer et al. 2021). → link sys.tuning-diagnostics. |
| Momentum update in the playbook/TF convention ($v\leftarrow\gamma v+g$, $\theta\leftarrow\theta-\eta v$); the playbook prefers Nesterov | fund.sgd-momentum | Already planned (conventions pitfall). Add Shallue §2.2's *effective LR* $\eta/(1-\gamma)$ as the quantity that matters when comparing across batch sizes. |
| Constant-LR noise floor and why schedules decay | fund.sgd-momentum | The noise-ball item is planned. Add the explicit 1-D result: stationary excess loss $\approx\eta\sigma^2/(4B)$, so halving $\eta$ ≈ doubling $B$. → link sys.tuning-steps-schedules (which derives it and simulates it). |
| Critical batch size, perfect scaling, linear scaling rule, warmup | fund.sgd-momentum | Planned (items 3, 6). Keep it short and → link sys.tuning-batch-size for the McCandlish derivation and Shallue's finding that measured optimal LRs follow no simple rule. Don't duplicate the `batch-size-steps` figure in depth. |
| Schedule families; "linear or cosine as default"; the best family is open | fund.sgd-momentum | One line in the schedules item citing the playbook's position. → link sys.tuning-steps-schedules for extending schedules between tuning rounds. |
| Polyak / iterate averaging cuts late step-to-step variance | fund.sgd-momentum | One line in item 2: the playbook's remedy list for noisy end-of-training metrics (larger batch, LR decay, Polyak averaging). |
| NAdam; Adam has 4 tunable hyperparameters "and they can all matter"; the inclusion hierarchy | fund.adam | Add the NAdam update (Nesterov look-ahead in the numerator). Cite Choi et al. 2019: Adam ⊇ momentum as ε → ∞, and optimizer rankings flip with the tuning protocol. |
| How to tune Adam by budget (<10 trials: LR; 10–25: +β₁; 25+: +ε; ≫25: +β₂) | fund.adam | A flashcard, with β's tuned on log(1−β). → link sys.tuning-pipeline. |
| ε is coupled to the LR | fund.adam | Large ε turns Adam into momentum SGD with LR $\alpha/\epsilon$ (one-line derivation), so search $(\epsilon,\alpha/\epsilon)$ jointly. Gradient RMS ≈ ε silently shrinks updates (Wortsman §3.4). |
| Preconditioned optimizers extend perfect batch scaling | fund.adaptive-lr or fund.adam | One line: Adam and K-FAC reach larger critical batch sizes than momentum (Zhang et al. 2019). |
| BN statistics are per device unless synced; ghost BN (~64 examples) | fund.batchnorm | Item 8 (batch dependence): BN batch ≠ gradient batch. Ghost BN (Hoffer et al. 2017). Goyal keeps the per-worker n = 32 fixed so the loss function doesn't change as workers scale. |
| BN EMA statistics must be synced before checkpointing | fund.batchnorm | Pitfall: some implementations save only device 0's running statistics. The EMA is linear, so averaging at checkpoint time is exact. |
| "Nowadays BN can often be replaced with LayerNorm" | fund.normalization | Say why: LN has no batch dependence, so there are no multi-device, small-batch or train/test discrepancies. |
| Normalization inside the residual: $x+f(\mathrm{Norm}(x))$, not $\mathrm{Norm}(x+f(x))$ | fund.normalization; llm.transformer-block | Pre-LN vs Post-LN with Xiong et al. 2020's gradient-at-init argument; Post-LN needs warmup. The playbook lists this among its instability fixes. |
| Zero-initialize residual branches (ReZero; last-BN γ = 0 per block) | fund.initialization; fund.cnn-architectures | The network starts as identity (or shallow), so early curvature is low (Bachlechner et al. 2020; Goyal et al. 2017 §5). |
| Label smoothing as a cheap regularizer that matters more at large batch | fund.regularization | Shallue §4.6: it helped (up to ~1 point on ImageNet) only at large batch sizes. Optimal logit gap $\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$ (4.51 for K=10, ε=0.1). |
| Retrospective checkpoint selection vs prospective early stopping | fund.dropout-early-stopping | In item 1: with a fixed step budget, keep the N best checkpoints and choose afterwards. No patience parameter is needed, and the playbook says prospective early stopping is usually unnecessary. |
| Dropout and augmentation increase gradient variance, so more steps are needed | fund.dropout-early-stopping | One line in the practicalities item. |
| Problematic overfitting = validation loss *rising*; best-trial selection favours "hobbled" configs | fund.bias-variance | Add to the overfitting discussion: a small LR can act as an accidental regularizer, so best-trial selection can favour bad optimizer settings. |
| Grid vs random search, log scales, BO | fund.cross-validation | Item 9 overlaps sys.tuning-search. Keep it to one card (Bergstra & Bengio's argument) and → link sys.tuning-search for quasi-random search, the explore-vs-exploit choice of algorithm, and reading studies. |
| Trial vs study vs data variance; retrain the best trial N times; SE of a difference $\sigma\sqrt{2/n}$ | fund.hypothesis-testing | A worked example: 5 seeds per arm, σ = 0.1% → SE 0.063%. The playbook's warning that seed variance alone can make identical configs differ "significantly". |
| Use the test set only after exploration; fold validation into training only for one-off workloads | fund.model-selection | Add to the selection-bias discussion. |
| Periodic validation metrics signal train/val overlap or a shuffling bug | fund.model-selection | Add to the leakage list; evaluate at fixed step intervals so the period is visible. |
| Imbalanced evaluation: log the *count* correct for rare classes | fund.classification-metrics | Example: "+0.05 sensitivity" on a class with 20 positives is one more example. |
| "Hyperparameter" vs "metaparameter" | fund.mle-map | One sentence where prior hyperparameters are defined: deep learning's "hyperparameters" are not parameters of a prior. |
| Tuning order and monitoring in the training loop | fund.training-loop | **Fix item 10**: the playbook does *not* tune the batch size for validation. Pick it for throughput, then tune LR (nuisance) and regularization, start with a constant LR, and add a schedule later. Item 6 should name the playbook's checks: loss rising means a bug, validation periodicity means leakage, log the unclipped gradient norm. Item 3: evaluate at step intervals, not time intervals. → link sys.tuning-diagnostics. |
| Zero-weight padded examples in partial eval batches | fund.training-loop | One line in the evaluation item (otherwise the padding biases the metric). |

### Applied ML & Systems (`content-plan-applied.md`)

| Playbook point | Topic | What to add |
|---|---|---|
| Gradient accumulation gives no throughput benefit; the playbook advises avoiding it in applied work | sys.gradient-accumulation | Accumulation simulates a bigger batch for memory reasons only; time per example does not fall. → link sys.tuning-batch-size for choosing the batch size. |
| Normalize the per-worker loss by the global batch (Goyal §3, remark 3) | sys.ddp; sys.gradient-accumulation | The all-reduce sums, so divide by $kn$, not $n$, or the LR silently scales with worker count. |
| Multi-host hygiene: log and checkpoint from one host; sync BN statistics before eval/checkpoint; RNG seeds the same for init but different for data; shard data files across hosts | sys.ddp (item 10); sys.fsdp (item 9) | The playbook's checklist, with the reason for each rule (diverging replicas; duplicated examples shrinking the effective batch). → link sys.tuning-pipeline. |
| Input-bound pipelines: profile; prefetch; drop unused features early; preprocess offline; avoid host–device sync barriers (e.g. metric syncs) | sys.profiling | The playbook's list of causes and fixes. |
| Throughput sweep: doubling the batch should double examples/s until the accelerator saturates | sys.profiling; sys.roofline | How to read a throughput-vs-batch sweep; flat throughput means saturation or a bottleneck. |
| Clipping threshold ≈ 90th percentile of unclipped norms; >50% of steps clipped means "just lower the LR" | sys.gradient-clipping | Already planned as a link (item 4). Keep the rule in sys.tuning-diagnostics; sys.gradient-clipping derives why clipping preserves direction and how it interacts with AMP (unscale first). |
| Instability fixes: warmup, clipping, normalization placement, optimizer switch, lower LR last | sys.vanishing-exploding; sys.stability-tricks | → link sys.tuning-diagnostics for the playbook's diagnosis protocol (LR sweep, curves above $lr^*$, 500-step high-LR check with per-step eval). |
| Adam ε at small gradient RMS | sys.stability-tricks | The Wortsman §3.4 observation (gradient RMS ≈ ε shrinks updates; lowering ε to 1e-15 helped) next to the playbook's advice to tune ε only with ≥25 trials. |

### LLMs (`content-plan-llms.md`)

| Playbook point | Topic | What to add |
|---|---|---|
| Critical batch size grows as the loss falls; batch-size ramps | llm.pretraining-optimization | Kaplan §5.1 $B_{crit}(L)$ alongside McCandlish's noise scale; why LLM recipes ramp the batch. → link sys.tuning-batch-size for the derivation. |
| Extending schedules between tuning rounds; schedules don't transfer across run lengths | llm.pretraining-optimization; llm.scaling-laws-practice | The playbook's Round 2 advice (keep the peak LR and stretch the cosine; extend the constant phase of linear decay) next to WSD; Chinchilla App. B (a cosine cycle longer than the run hurts). |
| Warmup protocol (transformers may need 40k+ steps), clipping rule, β₂ = 0.95 | llm.training-stability | Use the playbook's warmup protocol and clipping rule as the general recipe, then add the LLM-specific fixes (qk-layernorm, z-loss, small ε). → link sys.tuning-diagnostics. |
| Which hyperparameters transfer from short or small runs | llm.mup | Contrast the playbook's transfer list (warmup/init very likely; schedule unlikely) with μP's width transfer. |

### Scaling Book special edition

| Playbook point | Topic | What to add |
|---|---|---|
| Data-parallel batch size and its limit | sb.data-parallelism | One sentence: the batch size that keeps chips busy may exceed the critical batch size, beyond which extra data parallelism wastes compute. → link sys.tuning-batch-size. |

**Covered in full by the dedicated lessons** (don't duplicate them elsewhere; link instead): scientific/nuisance/fixed hyperparameters,
studies and budgets, quasi-random search vs grid and BO, axis and isolation plots, trial-budget bootstrapping, adopting changes,
the batch-size regimes and noise-scale derivation, `max_train_steps` selection and tuning rounds, the training-curve checklist,
the warmup protocol, the clipping-threshold rule, ghost-BN implementation details, the multi-host checklist, and evaluation setup.
