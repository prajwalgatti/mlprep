# Review: content plan Part A (Fundamentals, `fund.*`)

Date: 2026-10-02. Scope: `docs/content-plan.md` Part A only (Part B is reviewed separately). Standard: `docs/writing-brief.md`.
All edits below were made directly to Part A. Part B was not touched: the new Part A was spliced onto the live Part B at write time.

Tags: **blocking** = wrong content or answer key that a writer would copy; **major** = missing interview-critical material, budget
overflow, broken ordering or a wrong citation; **minor** = convention, wording or polish.

## Summary

- Part A goes from **63 to 66 lessons**. Three new lessons came from splitting overloaded ones: `fund.monte-carlo`, `fund.fisher-information`
  and `fund.hypothesis-testing`. All six protected ids are unchanged.
- **Ordering fixed**: four lessons had a prereq with a *higher* order number. Convexity and gradient descent now form an
  "Optimization foundations" section (orders 104, 106). Double descent moved to 225 and calibration to 245.
- **Waves**: fund.information-theory and fund.gradient-descent were promoted to W1, and fund.gradient-boosting and
  fund.cross-validation were demoted to W2. W1 is 23 of 66 (35%).
- **Coverage added**: the exponential family (with convexity in the natural parameter and moment matching), the U(0,θ) MLE,
  importance sampling, control variates, GDA/LDA/QDA, the RBF feature map, the Mercer PSD proof, SVD-vs-eigendecomposition
  stability for PCA, practical convexity tests, the information-theory identities, power and sample size, and class-imbalance remedies.
- **Correctness**: all 113 "(calc)" and new numbers were recomputed in Python and all pass (script:
  `scratchpad/review/check_calcs.py`). The fixes were to wrong answer keys and garbled claims, not to arithmetic: the MDS
  "which is false" question, Ba et al.'s invariance table, the label-smoothing loss, and the ESL 0-1 bias claim.
- **Sources**: about 230 cited sections and pages were spot-checked against the cache TOCs and text. Four citations were wrong and are fixed.
- **Tuning Playbook**: the coordinator's fix to fund.training-loop item 10 is applied, plus one-line pointers for all 28 fundamentals
  rows of `content-plan-tuning.md` Part B.

## 1. Structural changes (lesson scoping, ordering, waves)

| # | Tag | Change | Why |
|---|---|---|---|
| S1 | major | **New `fund.monte-carlo`** (order 15, W2): the MC estimator, inverse-CDF sampling, rejection sampling, importance sampling (including ESS and self-normalized IS), control variates and antithetic sampling, and ML uses. Sources: PML2 ch.11, Bishop §11.1, MacKay §29.1–29.3. | fund.probability-basics had about 12 cards of material. Importance sampling and control variates were missing, yet REINFORCE baselines (fund.gradient-estimators) and importance weighting (fund.distribution-shift) depend on them. Change of variables and Jensen stay in probability-basics because Part B's gen.flows and gen.latent-variables-elbo point there. |
| S2 | major | **New `fund.fisher-information`** (order 72, W2, prereq fund.mle): the score, Fisher information, observed vs expected information, the CRLB (with a Cauchy–Schwarz proof), MLE asymptotics, the delta method, the KL metric, Laplace and EWC. | fund.estimation items 7–10 plus fund.mle's asymptotics overflowed both lessons, and the asymptotics need the MLE first. fund.second-order and fund.kl-divergence also rely on Fisher information. |
| S3 | major | **fund.confidence-intervals split**: CIs and the bootstrap stay (now order 74). **New `fund.hypothesis-testing`** (order 76) takes tests, duality, power and sample size (new), paired tests (McNemar, paired bootstrap), seed variance, multiple comparisons, LRT and the Bayesian critique. | The old lesson had 10 dense items, about 13 cards. Power and sample size were missing and are commonly asked. |
| S4 | major | **Order violations fixed**: svm-margin-dual (270) → convexity (390); gradient-boosting (330) → gradient-descent (400); double-descent (140) → ridge-lasso (220); calibration (200) → logistic-regression (230). Convexity and GD moved to a new "Optimization foundations" section at 104 and 106. Double descent moved to 225 (after ridge-lasso, since its worked example is min-norm least squares). Calibration moved to 245 (after loss functions, since proper scoring rules are losses). The rest of the optimization block is renamed "Optimization algorithms". | Prereqs must come first in the reading sequence. Placing convexity first also helps logistic regression (Hessian PSD ⇒ convex), the lasso (subgradients) and ridge (strict convexity). |
| S5 | minor | **fund.cnn → fund.cnn-architectures**: the 1×1 and depthwise-separable items, their calc question and their figure moved. The lesson is retitled "CNN Architectures, Residual Connections & Efficient Convolutions". | fund.cnn had 11 items. The bottleneck and MobileNet context lives in the architectures lesson. |
| S6 | major | **Waves**: fund.information-theory → W1 (prereq of the W1 lessons kl-divergence and logistic-regression; CE and perplexity are core for LLM roles). fund.gradient-descent → W1 (prereq of W1 sgd-momentum; η < 2/L and conditioning are standard probes). fund.gradient-boosting → W2 (its prereq boosting is W2, and XGBoost internals are probed less at frontier labs). fund.cross-validation → W2 (already familiar to the reader). | W1 should be the most critical third and should mostly depend on W1. Remaining W1→W2 prereqs (mle → estimation, svm-margin-dual → convexity/loss-functions, bias-variance → estimation/learning-setups, regularization → none) cause expected `validate.py` warnings between waves. This is noted under the table. |
| S7 | minor | Tables updated: A.1 (66 rows, new section rows, the W1 count, a note on ordering) and A.2 (MLE vs MAP, Expectation and Confidence intervals rows point to the new lessons). | Consistency. |

## 2. Coverage additions (axis 1)

| # | Tag | Lesson | Added |
|---|---|---|---|
| C1 | major | fund.mle | **Exponential families**: derive ∇A = E[T] and ∇²A = Cov[T] ⪰ 0, so the NLL is convex *in the natural parameter*; MLE = moment matching; the canonical link gives "prediction − target" gradients. Sources: PML1 §3.4, Bishop §2.4, CS229 ch.3. The old "concavity for exponential families" was unqualified. |
| C2 | major | fund.mle | **U(0,θ) MLE** = max xᵢ (a boundary maximum, biased low with E = nθ/(n+1)). A classic whiteboard question, plus a calc. The old item 6 (bias) and item 9 (DL) were merged in to hold the budget. |
| C3 | major | fund.monte-carlo | Importance sampling, control variates (optimal c* and the (1−ρ²) factor), rejection sampling, inverse CDF (see S1). |
| C4 | major | fund.logistic-regression | **GDA: LDA vs QDA vs logistic regression** (linear vs quadratic boundary, efficiency vs robustness; PML1 §9.2, ESL §4.3/§4.4.5, CS229 ch.4), plus a compare question. |
| C5 | major | fund.svm-kernels | The **RBF infinite feature map** via a 1-D Taylor expansion; the **PSD proof** $\sum c_ic_jk=\|\sum c_i\phi(x_i)\|^2$; the Mercer / Moore–Aronszajn converse. Gaussian processes named (same Gram matrix; posterior mean = kernel ridge). |
| C6 | major | fund.pca | **SVD vs eigendecomposition**: derive XᵀX = VΣ²Vᵀ and λᵢ = σᵢ²/(N−1); forming XᵀX squares κ; cost comparison. Plus a question. |
| C7 | major | fund.convexity | **Practical convexity tests**: restriction to a line, quadratic forms (A ⪰ 0, 2×2 trace/det), the least-squares and logistic Hessians, sublevel sets vs quasi-convexity. Plus a calc (x² + 3xy + y² has eigenvalues 5 and −1, so it isn't convex). |
| C8 | major | fund.information-theory | **Identities**: subadditivity, I = H(X)+H(Y)−H(X,Y), conditional MI and its chain rule, and "conditioning reduces entropy only on average". Plus an MI calc (0.278 bits). |
| C9 | major | fund.hypothesis-testing | **Power and sample-size formula** (calc ≈ 1,570 per arm for a 5-point difference), McNemar calc (χ² = 4.36, p ≈ 0.037), p-value misreadings. |
| C10 | minor | fund.loss-functions | **Class-imbalance remedies** compared (resampling vs weights vs threshold), plus the prior-shift correction of probabilities after rebalancing (calc 0.8 → 0.039). |
| C11 | minor | fund.learning-setups | Parametric vs nonparametric (one line). |
| C12 | minor | fund.backprop | Gradient checking is now owned here, with the Taylor derivation of the O(h²) vs O(h) error (moved from fund.gradient-descent). |
| C13 | major | several | **Inline refreshers named** where a newcomer may be shaky (axis 5): Lagrange multipliers and log-det/trace derivatives (mle), matrix-calculus identities (linear-regression), completing the square and the Schur complement (gaussian), the spectral theorem and Lagrange multipliers (pca), multivariate Taylor (newton). |

## 3. De-duplication (axis 2, padding)

| # | Tag | Change |
|---|---|---|
| D1 | major | L1 soft-thresholding and the L1/L2 geometry were derived in both fund.regularization and fund.ridge-lasso, with two near-identical figures. fund.regularization now owns them. fund.ridge-lasso recaps them and spends the space on the lasso path, coordinate descent and an **SVD-shrinkage vs PCR** figure (`fund.ridge-lasso/svd-shrinkage`). |
| D2 | major | Label smoothing was taught in both fund.regularization and fund.loss-functions. regularization owns the definition and loss; loss-functions keeps only its effects (calibration, distillation, clustering). |
| D3 | minor | fund.kl-divergence item 8 re-derived $D^*$ and $C(G)=-\log4+2\mathrm{JSD}$, which gen.gans derives in full. It now states the result and keeps only the saturation example. |
| D4 | minor | The decorative `fund.cnn-architectures/timeline` figure was replaced by the depthwise-separable cost diagram. |

## 4. Correctness fixes (axis 4)

| # | Tag | Where | Problem → fix |
|---|---|---|---|
| F1 | **blocking** | fund.nonlinear-dim-reduction | "Which is false: classical MDS on Euclidean distances is equivalent to PCA" has a *true* statement as its answer. Replaced with a false t-SNE claim; the MDS fact became a true distractor. |
| F2 | **blocking** | fund.normalization item 3 | The invariance list contradicted Ba et al. Table 1 ("Neither is invariant to individual weight-vector rescaling in LN's case" was garbled, and BN's weight-vector and dataset-rescaling invariances were missing). Rewritten from the cached table: BN is invariant to weight-matrix and weight-vector rescaling and to dataset rescaling and re-centring; LN is invariant to weight-matrix rescaling and re-centring and to single-case rescaling. |
| F3 | major | fund.regularization item 9, fund.loss-functions item 8 | "Label smoothing = CE + ε·CE(u,p)" is wrong. The smoothed loss is $(1-\varepsilon)\mathrm{CE}(y,p)+\varepsilon\mathrm{CE}(u,p)$. Also added the optimal logit gap $\ln\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}=4.51$ (K=10, ε=0.1; verified). |
| F4 | major | fund.bias-variance item 8 | "ESL §7.3.1 shows bias can *help* when the estimate is on the right side of ½" misstates ESL. ESL says such errors *don't hurt*, and Ex. 7.2 says that with the mean estimate on the *wrong* side, extra *variance* can help. Rewritten. |
| F5 | major | fund.svm-kernels | "Degree-2 polynomial kernel in d=100 has C(102,2) = 5151 features" holds only for $(x^\top z+c)^2$ with c > 0. The homogeneous kernel has 5050. Both are now stated. |
| F6 | minor | fund.confidence-intervals | Test-set size: 1.96²·0.09/0.01² = 3,457.4, so the required n is **3,458** (round up), not "≈ 3,457". |
| F7 | minor | fund.gradient-descent | "GD diverges for η ≥ 0.02" became η > 0.02 (at η = 2/L the stiff coordinate oscillates without decaying). |
| F8 | minor | fund.svm-margin-dual | "C → ∞ on non-separable data: no feasible solution" is wrong for the soft margin: every finite C is feasible. Rephrased. |
| F9 | minor | fund.second-order | "Which is false: natural gradient is invariant to reparameterizations" was ambiguous (the flow is invariant). Now "finite steps are *exactly* invariant", which is the false version. |
| F10 | minor | fund.rnn | The subtopic used σ_max while the question used "spectral radius". Made consistent (largest singular value; γ = 1 for tanh). Noted that Pascanu et al. phrase it with eigenvalues while their proof bounds norms. |
| F11 | minor | fund.ridge-lasso | The "which is false" stem was itself a true statement ("ridge never sets coefficients exactly to zero"). Replaced with a false claim (monotone lasso paths) plus true distractors. |
| F12 | minor | fund.calibration | "Temperature scaling doesn't change AUC" holds for binary models (σ(z/T) is monotone). One-vs-rest AUC of a multiclass softmax can shift slightly. Qualified. |
| F13 | minor | fund.normalization item 4 | RMSNorm "no bias": the paper keeps a bias in the surrounding layer; LLaMA-style models drop it. Qualified. |
| F14 | minor | fund.convexity pitfalls | The claim that some texts call "strongly convex" what Boyd calls "strictly convex" was unsupported. Replaced with the standard caveat (x⁴ is strictly but not strongly convex; eˣ is strictly convex with no minimizer). |
| F15 | minor | fund.dropout-early-stopping item 6 | Made the p convention explicit: p/(1−p) with p = drop probability after rescaling, vs Srivastava §9.1's (1−p)/p with p = retain probability (checked in the cache). |
| F16 | minor | fund.regularization item 11 | "Equal for SGD up to λ′ = λ/α" didn't say which λ is which. Now explicit, matching Loshchilov & Hutter Prop. 1 (λ′ is the L2 coefficient, λ the per-step decay). |
| F17 | minor | fund.distribution-shift | Label-shift calc: "target 0.9/0.1" didn't say which class gets 0.9. Now $p_T(y{=}1)=0.9$. |

All other formulas I checked were correct. They include: MAP λ = σ²/τ² and 2σ²/b; Bayesian LR; Gaussian conditionals; the CRLB;
the NFL constants (1/4, 1/8, 1/7); the PAC bound; the hold-out bound; LOOCV; the AdaBoost bound; XGBoost w* and gain; Glorot, He
and the PyTorch default init; the BN backward pass; Adam's bias correction (3.16×); the AdamW update; Cover–Hart; JL; t-SNE/Concrete
log-convexity (τ ≤ 1/(n−1), confirmed in Maddison); ProtoNets w = 2c, b = −‖c‖²; the PAD; and ESS.
I also checked the Tuning Playbook numbers I inserted: ησ²/(4B) (confirmed by simulation), the 4.51 logit gap, and σ√(2/n) = 0.063%.

## 5. Source spot-checks (axis 3)

I checked about 230 citations against the cache (TOC files for PML1, PML2, Bishop, ESL, CS229; heading greps for DLB, Boyd, UML,
MacKay; and the paper files). Sample: PML1 (60 sections: §2.3.2, §4.7.5, §6.2.6, §8.4.6, §9.3.4, §11.3.2, §13.5.6, §17.3.5, §18.2.2,
§19.6, §21.3.4, …), PML2 (23: §3.3.3–3.3.5, §5.1.9, §6.3.4–6.3.8, §14.2.1, §19.2.3, §19.5.4, §26.7.6, …), Bishop (36), ESL (60),
CS229 (14 page anchors), DLB (37 headings), Boyd (12), UML (15), MacKay (11), and paper section and equation numbers (Kingma, Loshchilov,
Glorot, He, Lin, Chen & Guestrin, Ben-David, Ganin, Snell, Maddison, Srivastava, Ba, Zhang & Sennrich). Every cited cache file exists.

Wrong citations, all fixed:

| # | Tag | Citation | Fix |
|---|---|---|---|
| R1 | major | Boyd "§9.5.2 Newton decrement" (fund.newton) | The Newton decrement is in §9.5.1 (pdf p.500–501). §9.5.2 is "Newton's method" (damped, with backtracking). |
| R2 | minor | Glorot & Bengio "eq. 12 the normalized init" (fund.initialization) | It is **eq. 16** in the cached paper. |
| R3 | minor | "cite ESL eq. 7.64" for the LOOCV shortcut (fund.cross-validation) | The text gives LOOCV as **eq. 7.51** and GCV as eq. 7.52. Eq. 7.64 is Exercise 7.3's per-residual identity. Both are now cited. |
| R4 | minor | Ben-David et al. 2010 credited with the "proxy A-distance" (fund.domain-adaptation) | The PAD is defined in Ganin et al. 2016 §3.2 (after Ben-David et al. 2006). Moved there. |

New lessons cite only verified sections: PML2 ch.11 (pdf p.517–531), Bishop §11.1 (pdf p.546–554), MacKay §29.1–29.3
(pdf p.369–377), PML1 §5.5.1–5.5.4 (pdf p.228–232), PML1 §3.4, Bishop §2.4, PML1 §9.2, ESL §4.3 and Bishop §4.4.
UML page numbers are about 1 off in places (e.g. §5.1 starts at pdf p.61, cited as p.60–65), which is harmless because writers cite by section.

## 6. Tuning Playbook integration (coordinator request)

- **fund.training-loop item 10 is fixed**: don't tune the batch size against validation. Choose it for throughput and memory, then
  retune the LR and other optimizer hyperparameters, because the optimal LR depends on the batch size. Start with a constant LR and add a
  schedule later. I added a question that tests this. Items 3 and 6 got the playbook's checks (step-interval eval, zero-weighted padding,
  rising loss = bug, periodic validation = leakage, log the unclipped gradient norm).
- One-line "Tuning-plan insertion" pointers were added to gradient-descent, sgd-momentum (×4), adaptive-lr, adam (×3), batchnorm (×2),
  normalization (×2), initialization, cnn-architectures, regularization, dropout-early-stopping (×2), bias-variance, cross-validation,
  model-selection (×2), classification-metrics, mle-map and training-loop. Each cites the cached `tp_*` file where one exists.
- The seed-variance row that `content-plan-tuning.md` assigns to **fund.confidence-intervals** now lives in **fund.hypothesis-testing**
  item 5, because of the split. **Flag:** that row in `content-plan-tuning.md` should be retargeted. I didn't edit that file.

## 7. Flagged but not changed

| # | Tag | Finding | Suggestion |
|---|---|---|---|
| X1 | major | **Budget watch**: some lessons are still at or above 10 items: fund.mle-map (11), fund.regularization (12, but several are now one-line pointers), fund.svm-kernels (11), fund.kl-divergence (10), fund.cross-validation (10), fund.training-loop (10), fund.classification-metrics (10). | Writers should report overflow, as the brief says. Natural splits if needed: mle-map → "conjugate priors & posterior predictive" vs "MAP as regularization"; svm-kernels → move SVR, multiclass and Platt to a short third SVM lesson. |
| X2 | major | **Part B cross-references** (not mine to edit): gen.ebm-score-matching's pitfall cites "fund.estimation" for the statistics score. It should now cite **fund.fisher-information**. gen.flows and gen.latent-variables-elbo correctly cite fund.probability-basics (change of variables and Jensen were kept there for this reason). | Tell the Part B reviewer. |
| X3 | minor | **Gaussian processes** are only named (svm-kernels, ensembles). GP regression comes up in Bayesian-ML-flavoured interviews. | Optional W3 lesson `fund.gaussian-processes` after ridge-lasso and svm-kernels, if the user wants it. It's not in the original list. |
| X4 | minor | Fisher's LDA as supervised dimensionality reduction is only named (PCA limitations, PML1 §9.2.6). | Add one card to fund.logistic-regression or fund.pca if interviews in the user's target area ask for it. |
| X5 | minor | The ensemble variance formula appears in four lessons (variance-covariance, bias-variance, ensembles, bagging). | fund.variance-covariance owns the derivation; the others should recap in one line. This is fine as long as writers don't re-derive it. |
| X6 | minor | Some figures are mostly decorative (`fund.cross-validation/kfold-diagram`, `fund.learning-setups/setups-map`, `fund.training-loop/loop-diagram`). | Kept, since they're cheap and help orientation. Writers could make them [fig-Q] stems ("which fold is validation in round 3?"). |
| X7 | minor | W1 still contains classical-ML lessons (svm-margin-dual, decision-trees, bagging-random-forests) that frontier-lab research interviews weight less than LLM-adjacent topics. | Kept, because they're on the user's list and are classic whiteboard derivations. Revisit if the user's target roles are pure LLM research. |
| X8 | minor | **Concurrency**: another agent was editing Part B of the same file while I worked. I wrote Part A by splicing it onto the live Part B, but a later write from a stale copy could revert Part A. | After all reviewers finish, confirm that Part A says "66 lessons" and contains `fund.monte-carlo`. |
