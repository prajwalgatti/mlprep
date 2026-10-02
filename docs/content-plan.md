# Content plan: General ML fundamentals + Generative Modeling (v2)

Plan for rebuilding the `fundamentals` area (from the "General ML" and "technical prep" items in
[S. Sapora, *ML interviews*](https://silviasapora.github.io/blog/ml-interviews.html)) and for building the new
`genmodels` area. The reader is a 3rd-year PhD student preparing for Research Scientist interviews. They have never
met a given topic before, and **explainers must teach it from the problem upward and derive results**
(see [writing-brief.md](writing-brief.md)).

Big areas are split into **sequences of focused lessons** (one topic file each, about 6–10 explainer cards, 5–10 minutes),
linked by `prereqs`. Depth comes from the sequence. Each topic below has a **subtopic map**: the full syllabus that
the lesson must cover, with what the reader must be able to do for each item. Writers should work through all of it. If
an item doesn't fit, the topic should be split further. Don't drop the item.

## How writers should use this plan

- **Source cache** `$SRC` = `sources/`.
  Read `$SRC/INDEX.md` first. Grep the cached text before writing any formula or claim.
- **Page numbers** are **`pdf p.N`** and match the cache page markers (`=== [esl pdf p.80] ===`). Printed page = pdf page
  minus the offset: ESL 19, Bishop 20, PML1 30, PML2 34, Boyd 14, MacKay 12, UML 0, CS229 1. Goodfellow DL ("DLB")
  references are by section. The files are per chapter (`dlb_chNN_*.txt`), so use `grep -n '^8.5' dlb_ch08_optimization.txt`.
- **Equations in PDF-derived text are mangled.** Find the passage, then rebuild the maths yourself. The `d2l/` and `web/`
  files keep their LaTeX. Other agents have added `llm_*`, `sys_*`, `tp_*` and `sb_*` files to the same folder. A few are
  cited here (e.g. `tp_bergstra2012_random_search.txt`, `sys_chen2016_sublinear_memory.txt`, `llm_hu2021_lora.txt`).
- **`source:` strings**: use the citation keys below, e.g. `ESL §7.10.2`, `Bishop PRML §9.1`, `Goodfellow DL §8.5.3`,
  `Murphy PML1 §6.2.6`, `Murphy PML2 §25.2`, `Boyd CVX §9.5`, `SSBD §5.1` (= UML), `MacKay ITILA §2.6`, `CS229 notes ch.6`,
  `d2l §5.3`, `Ho et al. 2020 §3.2`.
- **Figures** are named `topic-id/slug` for `tools/figures/<topic-id>.py`. **[fig-Q]** marks a figure meant as an MCQ stem.
- **(calc)** marks worked numbers checked while writing this plan. Recompute them in Python anyway, as the brief requires.
- **Existing ids**: `fund.bias-variance`, `fund.mle-map`, `fund.adam`, `fund.normalization`, `fund.classification-metrics`
  and also `fund.kl-divergence` are kept for the split piece closest to their old content. Each topic's header says which.
  Keep item ids where the question is essentially unchanged (writing brief, "IDs and revisions").
- **Wave**: **W1** = most interview-critical, write first (about a third). **W2** = core, write second. **W3** = advanced or niche.

---

# Part A: Fundamentals (`area: fundamentals`, prefix `fund.`)

## A.1 Topic sequence (66 lessons)

| order | id | title | level | wave | prereqs | list items covered | status |
|---|---|---|---|---|---|---|---|
| **Probability & statistics** |||||||
| 10 | fund.probability-basics | Random Variables, PMFs/PDFs, Expectation & Change of Variables | core | W2 | – | PDF/PMF, expectation | new |
| 15 | fund.monte-carlo | Sampling & Monte Carlo Estimation | core | W2 | probability-basics | expectation (estimating it) | new (review) |
| 20 | fund.variance-covariance | Variance, Covariance & Correlation | core | W2 | probability-basics | variance & covariance | new |
| 30 | fund.bayes-theorem | Bayes' Theorem, Base Rates & Naive Bayes | core | **W1** | probability-basics | Bayes theorem | new |
| 40 | fund.gaussian | The Multivariate Gaussian | core | W2 | variance-covariance | (support for many) | new |
| 50 | fund.estimation | Estimators: Bias, Variance, Consistency & the CLT | core | W2 | variance-covariance | expectation (of estimators) | new |
| 70 | fund.mle | Maximum Likelihood Estimation | core | **W1** | estimation | MLE vs MAP (MLE half) | new |
| 72 | fund.fisher-information | Fisher Information, Cramér–Rao & MLE Asymptotics | intermediate | W2 | mle | MLE vs MAP (MLE properties), confidence intervals (Wald) | new (review) |
| 74 | fund.confidence-intervals | Confidence Intervals & the Bootstrap | core | W2 | estimation, fisher-information | confidence intervals | new |
| 76 | fund.hypothesis-testing | Hypothesis Tests, p-values & Comparing Models | core | W2 | confidence-intervals | confidence intervals (duality, model comparison) | new (review) |
| 80 | fund.mle-map | MAP Estimation, Priors & Bayesian Prediction | core | **W1** | mle, bayes-theorem | MLE vs MAP | **keeps id** |
| 90 | fund.information-theory | Entropy, Cross-Entropy & Mutual Information | core | **W1** | probability-basics | entropy | new |
| 100 | fund.kl-divergence | KL & Jensen–Shannon Divergences | core | **W1** | information-theory | KL divergence, JS divergence | **keeps id** (rewritten) |
| **Optimization foundations** |||||||
| 104 | fund.convexity | Convex Sets & Convex Functions | core | W2 | – | convex functions | new |
| 106 | fund.gradient-descent | Gradient Descent: Convergence & Conditioning | core | **W1** | convexity | gradient descent | new |
| **Learning theory & methodology** |||||||
| 110 | fund.learning-setups | Learning Setups, Risk, ERM & No Free Lunch | core | W2 | probability-basics | supervised vs unsupervised, NFL | new |
| 120 | fund.pac-vc | PAC Learning & VC Dimension | advanced | W3 | learning-setups | (NFL context, generalization) | new |
| 130 | fund.bias-variance | Bias–Variance Decomposition & Overfitting | core | **W1** | estimation, learning-setups | bias–variance, overfitting | **keeps id** |
| 150 | fund.regularization | Parameter Norm Penalties & Other Regularizers | core | **W1** | bias-variance, mle-map | regularisation methods | new |
| 160 | fund.dropout-early-stopping | Dropout & Early Stopping | core | W2 | regularization | early stopping, regularisation | new |
| 170 | fund.cross-validation | Cross-Validation & Hyperparameter Search | core | W2 | bias-variance | cross-validation | new |
| 180 | fund.model-selection | Leakage, Nested CV & Selection Bias | intermediate | W2 | cross-validation | cross-validation (pitfalls) | new |
| 190 | fund.classification-metrics | Precision, Recall, F1 & ROC/PR Curves | core | **W1** | bayes-theorem | precision/recall/F1/AUC-ROC | **keeps id** |
| **Classical models** |||||||
| 210 | fund.linear-regression | Linear Regression: OLS & its Geometry | core | **W1** | gaussian, mle | linear regression | new |
| 220 | fund.ridge-lasso | Ridge, Lasso & Bayesian Linear Regression | core | W2 | linear-regression, mle-map | linear regression, regularisation | new |
| 225 | fund.double-descent | Double Descent & Generalization in Deep Nets | intermediate | W2 | bias-variance, ridge-lasso | overfitting (modern view) | new |
| 230 | fund.logistic-regression | Logistic & Softmax Regression | core | **W1** | mle, information-theory | loss functions (CE) | new |
| 240 | fund.loss-functions | Loss Functions & Surrogate Losses | core | W2 | logistic-regression | loss functions | new |
| 245 | fund.calibration | Calibration & Proper Scoring Rules | intermediate | W2 | classification-metrics, logistic-regression | (AUC/metrics context) | new |
| 250 | fund.knn | k-Nearest Neighbours | core | W2 | bias-variance | kNN | new |
| 260 | fund.curse-of-dimensionality | The Curse of Dimensionality | core | W2 | knn | curse of dimensionality | new |
| 270 | fund.svm-margin-dual | SVMs I: Max Margin, Duality & Hinge Loss | intermediate | **W1** | loss-functions, convexity | SVMs | new |
| 280 | fund.svm-kernels | SVMs II: Kernels, Multiclass & Probabilities | intermediate | W2 | svm-margin-dual | SVMs | new |
| 290 | fund.decision-trees | Decision Trees | core | **W1** | bias-variance | decision trees | new |
| 300 | fund.ensembles | Why Ensembles Work: Voting, Averaging & Stacking | core | W2 | variance-covariance, bias-variance | ensembles | new |
| 310 | fund.bagging-random-forests | Bagging & Random Forests | core | **W1** | decision-trees, ensembles | bagging, ensembles | new |
| 320 | fund.boosting | Boosting & AdaBoost | intermediate | W2 | decision-trees, ensembles, loss-functions | boosting | new |
| 330 | fund.gradient-boosting | Gradient Boosting & XGBoost | intermediate | W2 | boosting, gradient-descent | boosting | new |
| 340 | fund.clustering | k-Means & Clustering | intermediate | W2 | variance-covariance | clustering (k-means), unsupervised | new |
| 350 | fund.gmm-em | Gaussian Mixtures & the EM Algorithm | intermediate | W2 | clustering, mle, kl-divergence | clustering, unsupervised | new |
| 360 | fund.pca | Principal Component Analysis | core | **W1** | variance-covariance, gaussian | dimensionality reduction | new |
| 370 | fund.whitening | Standardization & Whitening | intermediate | W2 | pca | data whitening | new |
| 380 | fund.nonlinear-dim-reduction | Kernel PCA, t-SNE, UMAP & Random Projections | intermediate | W3 | pca, curse-of-dimensionality | dimensionality reduction | new |
| **Optimization algorithms** |||||||
| 410 | fund.sgd-momentum | SGD, Momentum & Learning-Rate Schedules | core | **W1** | gradient-descent | SGD, training loops (theory) | new |
| 420 | fund.adaptive-lr | AdaGrad, RMSProp & Preconditioning | core | W2 | sgd-momentum | Adagrad | new |
| 430 | fund.adam | Adam & AdamW | core | **W1** | adaptive-lr, regularization | Adam/AdamW | **keeps id** |
| 440 | fund.newton | Newton's Method | intermediate | W2 | gradient-descent | Newton's method, second-order | new |
| 450 | fund.second-order | Quasi-Newton, Gauss–Newton & Natural Gradient | advanced | W3 | newton, kl-divergence, fisher-information | second-order methods | new |
| **Deep learning** |||||||
| 460 | fund.backprop | Backpropagation & Automatic Differentiation | core | **W1** | – | backprop implementation | new |
| 470 | fund.mlp-from-scratch | Implementing an MLP: Forward & Backward | core | **W1** | backprop, logistic-regression | MLP forward/backward | new |
| 480 | fund.training-loop | The Training Loop: Implementation & Debugging | core | **W1** | mlp-from-scratch, sgd-momentum | training loops with SGD | new |
| 490 | fund.activations | Activation Functions | core | W2 | backprop | activation functions | new |
| 500 | fund.initialization | Weight Initialization | core | **W1** | activations, variance-covariance | weight initialisation | new |
| 510 | fund.batchnorm | Batch Normalization | intermediate | **W1** | initialization | BatchNorm | new |
| 520 | fund.normalization | LayerNorm, RMSNorm & Where to Normalize | intermediate | **W1** | batchnorm | LayerNorm/RMSNorm | **keeps id** |
| 530 | fund.cnn | Convolutions: Mechanics & Inductive Bias | core | W2 | mlp-from-scratch | CNNs | new |
| 540 | fund.cnn-architectures | CNN Architectures, Residual Connections & Efficient Convolutions | core | W2 | cnn, batchnorm | CNNs | new |
| 550 | fund.rnn | RNNs, BPTT & Vanishing Gradients | intermediate | W2 | backprop, activations | RNNs | new |
| 560 | fund.lstm-gru | LSTMs & GRUs | intermediate | W2 | rnn | LSTMs | new |
| 570 | fund.autoencoders | Autoencoders | intermediate | W2 | pca, mlp-from-scratch | autoencoders | new |
| 580 | fund.gradient-estimators | Gradients Through Sampling: REINFORCE vs Reparameterization | advanced | W3 | backprop, monte-carlo | Gumbel-Softmax (prerequisite) | new |
| 590 | fund.gumbel-softmax | Gumbel-Max, Gumbel-Softmax & Straight-Through | advanced | W3 | gradient-estimators | Gumbel-Softmax | new |
| **Transfer & shift** |||||||
| 600 | fund.transfer-learning | Transfer Learning & Fine-Tuning | intermediate | W2 | cnn-architectures, regularization | transfer learning | new |
| 610 | fund.few-zero-shot | Few-Shot & Zero-Shot Learning | intermediate | W2 | transfer-learning | few-/zero-shot | new |
| 620 | fund.distribution-shift | Distribution Shift: Covariate, Label & Concept | advanced | W3 | learning-setups, bayes-theorem, monte-carlo | domain adaptation | new |
| 630 | fund.domain-adaptation | Unsupervised Domain Adaptation: Theory & Methods | advanced | W3 | distribution-shift, kl-divergence | domain adaptation | new |

W1 count: 23 of 66 (about 35%). The prereqs column abbreviates `fund.` ids. Write the full id in the YAML.
Every prereq now has a lower order number than its dependant (review 2026-10-02 fixed four violations: svm-margin-dual → convexity,
gradient-boosting → gradient-descent, double-descent → ridge-lasso, calibration → logistic-regression). Some W1 lessons still list W2
prereqs (e.g. mle → estimation, svm-margin-dual → convexity). `validate.py` warns "prereq does not exist (yet)" until the W2 lesson is
written; that warning is expected between waves.

## A.2 Coverage check (every list item → lesson)

| list item | lesson(s) |
|---|---|
| Overfitting / underfitting | fund.bias-variance; modern view in fund.double-descent; remedies in fund.regularization, fund.dropout-early-stopping |
| Backprop implementation | fund.backprop, fund.mlp-from-scratch |
| Training loops with SGD | fund.training-loop (code & debugging); fund.sgd-momentum (theory) |
| Implementing MLPs forward/backward | fund.mlp-from-scratch |
| Curse of dimensionality | fund.curse-of-dimensionality |
| CNNs | fund.cnn, fund.cnn-architectures |
| RNNs / LSTMs | fund.rnn, fund.lstm-gru |
| Autoencoders | fund.autoencoders (VAE → gen.latent-variables-elbo, gen.vae) |
| Gumbel-Softmax | fund.gradient-estimators → fund.gumbel-softmax |
| MLE vs MAP | fund.mle, fund.mle-map (MLE asymptotics: fund.fisher-information) |
| Newton's method | fund.newton |
| Linear regression | fund.linear-regression, fund.ridge-lasso |
| Activation functions | fund.activations |
| Loss functions | fund.logistic-regression, fund.loss-functions |
| No Free Lunch theorem | fund.learning-setups (formal version in fund.pac-vc) |
| BatchNorm / LayerNorm / RMSNorm | fund.batchnorm, fund.normalization |
| Variance & covariance | fund.variance-covariance |
| Adam / AdamW / Adagrad | fund.adaptive-lr, fund.adam |
| Bias–variance tradeoff | fund.bias-variance |
| Regularisation methods | fund.regularization, fund.dropout-early-stopping, fund.ridge-lasso |
| Supervised vs unsupervised | fund.learning-setups |
| Clustering (k-means) | fund.clustering, fund.gmm-em |
| kNN | fund.knn |
| SVMs | fund.svm-margin-dual, fund.svm-kernels |
| Boosting | fund.boosting, fund.gradient-boosting |
| Bagging | fund.bagging-random-forests |
| Decision trees | fund.decision-trees |
| Ensembles | fund.ensembles (+ bagging/boosting lessons) |
| Bayes theorem | fund.bayes-theorem |
| Precision / Recall / F1 / AUC-ROC | fund.classification-metrics (+ fund.calibration) |
| KL divergence | fund.kl-divergence |
| Jensen–Shannon divergence | fund.kl-divergence (GAN link: gen.gans) |
| Weight initialisation | fund.initialization |
| Gradient descent / SGD | fund.gradient-descent, fund.sgd-momentum |
| Cross-validation | fund.cross-validation, fund.model-selection |
| Data whitening | fund.whitening |
| Convex functions | fund.convexity |
| Early stopping | fund.dropout-early-stopping |
| Domain adaptation | fund.distribution-shift, fund.domain-adaptation |
| Dimensionality reduction | fund.pca, fund.nonlinear-dim-reduction |
| Transfer learning | fund.transfer-learning |
| Few-shot / zero-shot | fund.few-zero-shot |
| Second-order methods | fund.newton, fund.second-order |
| Expectation | fund.probability-basics (Monte Carlo estimation of it: fund.monte-carlo; estimators: fund.estimation) |
| Entropy | fund.information-theory |
| PDF / PMF | fund.probability-basics |
| Confidence intervals | fund.confidence-intervals (Wald intervals: fund.fisher-information; tests and model comparison: fund.hypothesis-testing) |

Sapora's General-ML list also includes **S4** (state-space models). It is not in the user's list, so it is left out here.
It belongs with Mamba and linear RNNs in the LLM/sequence area, and fund.lstm-gru should only link to it.

## A.3 Sources deliberately excluded
- **Wikipedia** and Medium/Towards-Data-Science posts. These are unvetted and not citable as primary sources. Replace the current Wikipedia reading links.
- **Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024)**. bishopbook.com only offers an in-browser viewer, so there is no text to cache. Optional reading link only.
- **Paywalled originals**: Wolpert 1996 (NFL), Cover & Hart 1967 (kNN), Amari 1998 (natural gradient), Shimodaira 2000 (covariate shift),
  Freund & Schapire 1997 (JCSS), Lin 1991 (JS) and Hinton & Salakhutdinov 2006. Their results are covered by cached books (UML ch.5, ESL §13.3,
  Martens 2014, PML2 §19.5.2, UML ch.10, PML2 §2.7, DLB ch.14).
- **Not free**: Cover & Thomas, Wasserman's *All of Statistics*, Nocedal & Wright. Use MacKay, PML1/PML2, Boyd and Bottou et al. 2018 instead.
- **ISLR/ISLP**: free, but ESL covers the same at greater depth. Optional reading only. **Chip Huyen's *Designing ML Systems***: applied systems, out of scope here.
- **Blogs** (Olah, Distill, Karpathy; Weng/Song/Dieleman in Part B) are cached as **secondary** sources for intuition and reading links.
  Never cite one as the sole support for a technical claim.

---

## A.4 Lesson plans

Each lesson has the following fields:
- **Header**: level, wave, prereqs.
- **Sources**: citation key, location, URL and cache file.
- **Subtopic map**: the full syllabus. *Must understand* = what the reader should be able to do afterwards.
- **Question ideas**
- **Figures**
- **Pitfalls & source disagreements**

### Probability & statistics

---

#### 10 · `fund.probability-basics`: Random Variables, PMFs/PDFs, Expectation & Change of Variables
- **Level** core · **W2** · **prereqs** [] · **covers** PDF/PMF, expectation.
- **Sources**
  - Goodfellow DL §3.2–3.3 (random variables, PMF, PDF), §3.4–3.6 (marginal, conditional, chain rule), §3.8 (expectation), §3.12 (technical details incl. change of variables). `dlb_ch03_prob.txt` · https://www.deeplearningbook.org/contents/prob.html
  - Murphy PML1 §2.2.1–2.2.5 (discrete/continuous RVs, CDF, moments; pdf p.65–73), §2.8.1–2.8.3 (transformations, bijections; pdf p.96–99). `pml1.txt` · https://probml.github.io/pml-book/book1.html
  - Bishop PRML §1.2.1 densities incl. the Jacobian change of variables (pdf p.37), §1.2.2 expectations (pdf p.39). `bishop.txt`
  - d2l "Random Variables" appendix (LaTeX preserved). `d2l/d2l_random_variables.txt`
  - MacKay ITILA §2.7 Jensen's inequality (pdf p.47). `mackay.txt`. Boyd CVX §3.1.8 Jensen's inequality (pdf p.91). `boyd.txt`
- **Budget**: about 10 cards. Monte Carlo moved to fund.monte-carlo (review 2026-10-02) so the change-of-variables and Jensen derivations aren't compressed. Part B's gen.flows and gen.latent-variables-elbo point here for both, so they stay in this lesson.
- **Subtopic map**
  1. *Random variable, outcome space, event*. Must understand: an RV is a function of the outcome, and discrete vs continuous determines whether we sum or integrate.
  2. *PMF vs PDF vs CDF*. Must understand: a density is a derivative of the CDF, its value can exceed 1 (U(0, 0.5) has density 2), $P(X=x)=0$ for continuous X, and $P(a<X\le b)=F(b)-F(a)$.
  3. *Joint, marginal, conditional; sum and product rules; chain rule of probability*. Must be able to marginalize a 2×2 joint table and factorize $p(x_1,\dots,x_n)$.
  4. *Independence vs conditional independence*: definitions and a 3-variable example where one holds without the other.
  5. *Change of variables*: derive $p_Y(y)=p_X(g^{-1}(y))\,|dg^{-1}/dy|$ from the CDF for monotone g. Multivariate Jacobian determinant. Worked examples: $Y=aX+b$ and log-normal. Why densities pick up a Jacobian while probabilities don't (this is used later in MAP non-invariance and flows).
  6. *Expectation*: definition (sum/integral), LOTUS ($E[g(X)]$ without finding $p_Y$), linearity without independence, and $E[XY]=E[X]E[Y]$ under independence or uncorrelatedness.
  7. *Conditional expectation* as a random variable, and the tower rule $E[E[Y|X]]=E[Y]$ (derive).
  8. *Jensen's inequality*: statement, a short convexity proof, and examples $E[\log X]\le\log E[X]$ and $E[X^2]\ge E[X]^2$. This is the backbone of the ELBO and KL ≥ 0 later.
  9. *Catalogue of distributions*: Bernoulli, binomial, categorical, Poisson, uniform, exponential, Gaussian, Laplace and Beta, with their means and variances. Derive two as LOTUS exercises (uniform variance $(b-a)^2/12$, exponential mean $1/\lambda$); the rest go in the Key-results card as a reference list (memorization, not derivation).
  10. *Pointer*: sample averages estimate expectations; the Monte Carlo estimator, sampling methods and importance sampling are the next lesson (fund.monte-carlo).
- **Question ideas**
  - (calc) Density of $Y=2X+1$ at y=1 for $X\sim N(0,1)$: $\phi(0)/2\approx0.199$.
  - Which is false: "a PDF value can never exceed 1". True distractors: linearity needs no independence; P(X=x)=0 for a continuous X; a CDF is non-decreasing.
  - (calc) Joint table → marginal and conditional probability.
  - (calc) $E[e^X]$ for $X\sim N(0,1)$ via LOTUS = $e^{1/2}\approx1.65$. Compare with $e^{E X}=1$ (Jensen).
  - Derivation step: in the change-of-variables proof, which step needs g to be monotone?
- **Figures**
  - `fund.probability-basics/pdf-vs-pmf`: binomial PMF bars next to a narrow Gaussian whose peak is above 1. Notice that density ≠ probability.
  - `fund.probability-basics/change-of-variables`: X uniform mapped through $y=x^2$ on [0,1], with histograms of X and Y. Notice the density piling up where g is flat.
  - `fund.probability-basics/jensen`: a convex f with two points, the chord, and the gap $E f(X)-f(EX)$.
- **Pitfalls & source disagreements**
  - DLB writes $P$ for PMFs and $p$ for PDFs. Bishop and PML1 use $p$ for both.
  - "Probability density" and "likelihood" are often conflated. Likelihood is a function of the parameters (setup for fund.mle).

---

#### 15 · `fund.monte-carlo`: Sampling & Monte Carlo Estimation
- **Level** core · **W2** · **prereqs** [fund.probability-basics] · **covers** expectation (estimating it by sampling). Groundwork for REINFORCE baselines, importance weighting under shift, IWAE and off-policy RL. *New in review 2026-10-02* (split from fund.probability-basics, plus importance sampling and control variates, which were missing).
- **Sources**
  - Murphy PML2 ch.11: §11.2 Monte Carlo integration and its accuracy (pdf p.517–520), §11.3.1 inverse-cdf sampling (pdf p.520), §11.4 rejection sampling incl. §11.4.4 high dimensions (pdf p.521–524), §11.5 importance sampling: direct, self-normalized, choosing the proposal (pdf p.524–526), §11.6 controlling variance: Rao–Blackwellization, control variates, antithetic sampling (pdf p.528–531). `pml2.txt`
  - Bishop PRML §11.1.1 standard distributions via the inverse CDF, §11.1.2 rejection sampling, §11.1.4 importance sampling (pdf p.546–554). `bishop.txt`
  - MacKay ITILA §29.1–29.3: the problems to be solved, importance sampling and its failure in high dimensions, rejection sampling (pdf p.369–377). `mackay.txt`
  - Murphy PML1 §2.8.7 Monte Carlo approximation (pdf p.103). `pml1.txt`
- **Subtopic map**
  1. *The problem*: most quantities we want are expectations $E_p[f(X)]$ (risks, gradients, marginal likelihoods, posterior means), and the integral is rarely tractable. Quadrature on a grid needs $r^d$ points (pointer to fund.curse-of-dimensionality).
  2. *The Monte Carlo estimator* $\hat\mu_n=\frac1n\sum_if(x_i)$: unbiased, with variance $\mathrm{Var}(f)/n$ (derive), so the error falls as $1/\sqrt n$ **whatever the dimension**. Error bars from the CLT and the sample variance. A minibatch gradient is exactly this estimator.
  3. *Sampling simple distributions*: inverse-CDF sampling (prove $F^{-1}(U)\sim F$; worked exponential $-\ln(1-U)/\lambda$). Ancestral sampling for factorized joints. Box–Muller named.
  4. *Rejection sampling*: draw $x\sim q$ and accept with probability $\tilde p(x)/(Mq(x))$. Derive that accepted samples follow p and that the acceptance rate is $Z_p/M$ (1/M for normalized p). Why M grows exponentially with dimension (PML2 §11.4.4).
  5. *Importance sampling*:
     - $E_p[f]=E_q[f\,p/q]$ (derive). It is unbiased when q > 0 wherever $fp\ne0$.
     - The variance can be infinite when q has lighter tails than p. The zero-variance proposal is $q^*\propto|f|p$ (statement, with the one-line reason).
     - Self-normalized IS for unnormalized p: biased but consistent.
     - Weight degeneracy and the effective sample size $(\sum w)^2/\sum w^2$. Collapse in high dimensions (MacKay §29.2).
  6. *Variance reduction*:
     - Control variates: $f-c\,(g-E g)$ with optimal $c^*=\mathrm{Cov}(f,g)/\mathrm{Var}(g)$ (derive), leaving variance $(1-\rho^2)\mathrm{Var}f$. This is exactly the REINFORCE baseline (pointer to fund.gradient-estimators).
     - Antithetic sampling, Rao–Blackwellization and common random numbers, one line each.
  7. *ML uses*: SGD's minibatch gradients, REINFORCE, importance-weighted ERM under covariate shift (pointer to fund.distribution-shift), IWAE bounds, off-policy RL ratios, MC dropout. MCMC (Metropolis–Hastings) named as the tool when p can't be sampled directly (one paragraph; pointer to the gen area).
- **Question ideas**
  - Predict: the Monte Carlo standard error when n goes from 100 to 10,000 (it falls by 10×, in any dimension).
  - (calc) Naive MC for $P(X>3)$ with $X\sim N(0,1)$ (p ≈ 1.35e−3) and n = 10⁴: relative SE ≈ $\sqrt{(1-p)/(np)}\approx0.27$. Why does an importance proposal centred at 3 help?
  - (calc) A control variate with correlation ρ = 0.9 cuts the variance to 0.19 of the original.
  - (calc) Importance weights (2, 1, 1, 0): ESS = 16/6 ≈ 2.67 of 4.
  - Which is false: "importance sampling is unbiased for any proposal q with the same mean as p".
  - Derivation step: in the inverse-CDF proof, where is the monotonicity of F used?
- **Figures**
  - `fund.monte-carlo/mc-error`: a running MC estimate with a ±2 SE band vs n on a log axis. Notice the $1/\sqrt n$ funnel.
  - `fund.monte-carlo/rejection`: target $\tilde p$ under the envelope $Mq$, with accepted and rejected points. **[fig-Q]**: "what fraction is accepted?"
  - `fund.monte-carlo/importance-weights`: p, a good q and a too-narrow q, with the weight function p/q exploding in the tails of the narrow one.
- **Pitfalls & source disagreements**
  - The same ratio p/q is called a likelihood ratio, density ratio, importance weight or propensity weight, depending on the field.
  - Self-normalized IS is biased at finite n. Plain IS needs normalized p and q.

---

#### 20 · `fund.variance-covariance`: Variance, Covariance & Correlation
- **Level** core · **W2** · **prereqs** [fund.probability-basics] · **covers** variance & covariance.
- **Sources**
  - Goodfellow DL §3.8 (variance, covariance, covariance matrix). `dlb_ch03_prob.txt`
  - Murphy PML1 §2.2.5 moments (pdf p.70), §2.2.6 limitations of summary statistics (pdf p.73), §2.8.4 moments of linear transformations (pdf p.99), §3.1.1–3.1.5 covariance, correlation, "uncorrelated does not imply independent", correlation vs causation, Simpson's paradox (pdf p.107–110). `pml1.txt`
  - Bishop PRML §1.2.2 expectations & covariances (pdf p.39). `bishop.txt`
  - d2l "Random Variables" appendix (variance, covariance, correlation sections). `d2l/d2l_random_variables.txt`
- **Subtopic map**
  1. *Variance*: definition, the shortcut $E[X^2]-E[X]^2$ (derive), $\mathrm{Var}(aX+b)=a^2\mathrm{Var}X$, standard deviation and units.
  2. *Covariance and correlation*: definitions, bilinearity, $\mathrm{Cov}(X,X)=\mathrm{Var}X$, the Cauchy–Schwarz bound $|\rho|\le1$ (prove), and that ρ is invariant to positive affine rescaling.
  3. *Variance of sums*: $\mathrm{Var}(\sum_iX_i)=\sum\mathrm{Var}X_i+2\sum_{i<j}\mathrm{Cov}$. The iid mean has variance $\sigma^2/n$. With equal pairwise correlation ρ it is $\rho\sigma^2+(1-\rho)\sigma^2/n$ (derive; this is the ensemble formula used in fund.ensembles).
  4. *Covariance matrix*: $\Sigma=E[(x-\mu)(x-\mu)^\top]$, symmetric. **PSD proof**: $v^\top\Sigma v=\mathrm{Var}(v^\top x)\ge0$. When it is singular (a linear dependency among the coordinates).
  5. *Linear transforms*: $E[Ax+b]=A\mu+b$, $\mathrm{Cov}(Ax+b)=A\Sigma A^\top$ (derive). Use it to get the variance of a projection $u^\top x$ (sets up PCA).
  6. *Uncorrelated ≠ independent*: counterexample $X\sim N(0,1)$, $Y=X^2$. Joint Gaussianity is the exception. Marginally Gaussian is not enough (random-sign counterexample).
  7. *Law of total variance*: $\mathrm{Var}Y=E[\mathrm{Var}(Y|X)]+\mathrm{Var}(E[Y|X])$ (derive), applied to a two-component mixture. This sets up bias–variance and aleatoric vs epistemic uncertainty.
  8. *Sample estimates*: sample covariance with 1/N vs 1/(N−1) (full treatment in fund.estimation). The Pearson correlation of a sample. Correlation vs causation, and Simpson's paradox (PML1 §3.1.5).
  9. *Numerical stability*: the one-pass formula $E[X^2]-E[X]^2$ suffers catastrophic cancellation, so use Welford/two-pass (`sys_wiki_variance_algorithms.txt` in the cache).
- **Question ideas**
  - (calc) Var(X − 2Y) with Var X = 1, Var Y = 2, Cov = 0.5 → 1 + 8 − 2 = 7.
  - (calc) Mean of 10 predictors with σ²=1 and ρ=0.5 → 0.55. Its limit as n → ∞ is 0.5.
  - Which is false: "Cov(X,Y)=0 implies X and Y are independent".
  - (calc) Mixture variance via total variance: ½N(−1, 1) + ½N(1, 1) → 1 + 1 = 2.
  - Flashcard: prove Σ is PSD in one line.
  - Predict: the correlation after rescaling X by −3 (the sign flips, the magnitude is unchanged).
- **Figures**
  - `fund.variance-covariance/uncorrelated-dependent`: scatter of $(X,X^2)$ with a flat regression line. Notice ρ = 0 despite deterministic dependence.
  - `fund.variance-covariance/correlation-gallery`: six scatters with ρ ∈ {−0.9, …, 0.9} and one nonlinear ρ≈0 case. **[fig-Q]**: "which scatter has ρ ≈ 0.7?"
  - `fund.variance-covariance/simpson`: two groups with positive within-group slopes and a negative pooled slope.
- **Pitfalls & source disagreements**
  - The default normalization differs by library: numpy `var` uses ddof=0, while pandas and `torch.var` use correction=1.
  - "Correlation" usually means Pearson (linear only). Mutual information is the general measure (fund.information-theory).

---

#### 30 · `fund.bayes-theorem`: Bayes' Theorem, Base Rates & Naive Bayes
- **Level** core · **W1** · **prereqs** [fund.probability-basics] · **covers** Bayes theorem.
- **Sources**
  - Murphy PML1 §2.3 Bayes' rule incl. §2.3.1 COVID-test and §2.3.2 Monty Hall (pdf p.74–79); §9.3 naive Bayes incl. §9.3.4 connection to logistic regression (pdf p.362–366); §9.4 generative vs discriminative (pdf p.366). `pml1.txt`
  - MacKay ITILA §2.3 forward vs inverse probability (pdf p.39–44), ch.3 More about inference (pdf p.60). `mackay.txt`
  - Goodfellow DL §3.11 Bayes' rule. `dlb_ch03_prob.txt`
  - CS229 notes ch.4 generative learning: GDA, naive Bayes, Laplace smoothing, event models (pdf p.36–49). `cs229.txt`
  - d2l "Naive Bayes" appendix. `d2l/d2l_naive_bayes.txt`. ESL §6.6.3 (pdf p.229).
- **Subtopic map**
  1. *Derivation* from the product rule. Name each term: prior, likelihood, evidence, posterior. The evidence as a marginalization over hypotheses.
  2. *Base-rate fallacy*, worked twice:
     - With numbers (calc): prevalence 1%, sensitivity 99%, FPR 5% gives P(disease|+) ≈ 0.167.
     - With a natural-frequency picture (1,000 people).
     - Then a second positive test: sequential updating, where yesterday's posterior becomes today's prior (calc: ≈ 0.80).
  3. *Odds form*: posterior odds = likelihood ratio × prior odds. Log-odds add, which links to logistic regression's logit.
  4. *Monty Hall and similar puzzles*: setting up the right likelihood ("the host's choice depends on where the car is").
  5. *Conditional independence and explaining away*: two independent causes become dependent given a common effect (a v-structure), with a numeric example.
  6. *Bayes for classification*: $p(y|x)\propto p(x|y)p(y)$. The Bayes-optimal decision. Generative vs discriminative modelling.
  7. *Naive Bayes*:
     - The conditional-independence assumption $p(x|y)=\prod_jp(x_j|y)$.
     - Bernoulli and multinomial event models for text.
     - The MLE is counting. Laplace/additive smoothing fixes zero counts (this is MAP with a Dirichlet prior, previewed for fund.mle-map).
     - Use log-probabilities to avoid underflow.
  8. *Why naive Bayes works despite a wrong assumption*: classification needs only the argmax, so miscalibrated probabilities can still rank correctly. Naive Bayes (Gaussian, shared variances) gives a linear decision boundary, the same form as logistic regression (PML1 §9.3.4).
  9. *Generative vs discriminative trade-offs* (PML1 §9.4; Ng & Jordan intuition via CS229): sample efficiency, missing features, and calibration.
- **Question ideas**
  - (calc) Base-rate MCQ. Distractors are sensitivity (0.99), 1−FPR (0.95), and the prior.
  - (calc) Second independent positive test: posterior ≈ 0.80.
  - Monty Hall variant: the host opens a door at random and it happens to show a goat. What is P(win by switching) now? (1/2)
  - Explaining away: given the alarm rang and an earthquake occurred, does P(burglary) go up or down? (down)
  - (calc) Naive Bayes with a zero count in a test word → posterior 0 for that class. The fix is add-1 smoothing.
  - Which is false: "naive Bayes probabilities are well calibrated because the model is generative".
- **Figures**
  - `fund.bayes-theorem/base-rate-grid`: 1,000 icons coloured as TP/FP/FN/TN. **[fig-Q]**: "estimate P(sick | +)".
  - `fund.bayes-theorem/sequential-update`: the posterior P(disease) after 0, 1, 2 and 3 positive tests.
  - `fund.bayes-theorem/explaining-away`: a 3-node v-structure diagram with conditional probability bars before and after observing the effect.
- **Pitfalls & source disagreements**
  - "Bayesian" in "Bayes' theorem" does not imply Bayesian statistics. The theorem is just probability calculus.
  - Naive Bayes is a generative classifier. Calling it "Bayesian" is a misnomer unless priors are put on its parameters (PML1 §9.3.3).

---

#### 40 · `fund.gaussian`: The Multivariate Gaussian
- **Level** core · **W2** · **prereqs** [fund.variance-covariance] · **covers** support for PCA, linear regression, GMMs, VAEs and diffusion.
- **Sources**
  - Bishop PRML §2.3 Gaussian incl. §2.3.1 conditional, §2.3.2 marginal, §2.3.3 Bayes' theorem for Gaussians (pdf p.98–113). `bishop.txt`
  - Murphy PML1 §2.6 univariate Gaussian (pdf p.87–91), §3.2 MVN incl. §3.2.2 Mahalanobis and §3.2.3 marginals & conditionals (pdf p.110–116), §3.3 linear Gaussian systems (pdf p.116). `pml1.txt`
  - CS229 notes ch.4 "The multivariate normal distribution" (pdf p.37). `cs229.txt`
  - Goodfellow DL §3.9.3. d2l "Distributions" appendix. `d2l/d2l_distributions.txt`
- **Subtopic map**
  1. *Density*: the univariate form, then the MVN, $(2\pi)^{-d/2}|\Sigma|^{-1/2}\exp(-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu))$. What each factor does. Why Σ must be PD for the density to exist.
  2. *Geometry*: eigendecomposition $\Sigma=U\Lambda U^\top$. Contours are ellipsoids with axes $\sqrt{\lambda_i}u_i$. Mahalanobis distance as Euclidean distance after whitening.
  3. *Affine transformations*: if $x\sim N(\mu,\Sigma)$ then $Ax+b\sim N(A\mu+b,A\Sigma A^\top)$. Reparameterized sampling $x=\mu+L\varepsilon$ with $LL^\top=\Sigma$ (Cholesky). This is the VAE trick in miniature.
  4. *Marginals*: read off the sub-block. *Conditionals*: $\mu_{a|b}=\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)$ and $\Sigma_{a|b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$. Derive the 2-D case by completing the square. The conditional variance doesn't depend on the observed value. *Inline refreshers*: completing the square in a quadratic form, and the block-matrix inverse via the Schur complement.
  5. *Sums of independent Gaussians* (means add, covariances add). Product of two Gaussian densities (precision-weighted mean). Both are reused in DDPM ("merging Gaussians").
  6. *Linear-Gaussian Bayes*: prior $N(\mu_0,\Sigma_0)$ with likelihood $y\sim N(Ax,\Sigma_y)$ gives a Gaussian posterior. State the formulas. They set up Bayesian linear regression and Kalman filtering.
  7. *MLE of μ and Σ* (statement and pointer to fund.mle). Diagonal vs full vs isotropic covariance parameter counts: d, d(d+1)/2, 1.
  8. *Why Gaussians are everywhere*: CLT, maximum entropy for a fixed mean and covariance, and closure under affine maps and conditioning.
  9. *High-dimensional behaviour*: samples concentrate near radius $\sqrt d$ (preview for fund.curse-of-dimensionality).
- **Question ideas**
  - (calc) Conditional of a bivariate Gaussian with ρ=0.8, unit variances, $x_b=1$: mean 0.8, variance 0.36.
  - (calc) Parameter count of a full-covariance Gaussian in d=100: 100 + 5050 = 5150.
  - Which is false: "if X and Y are each Gaussian, then X + Y is Gaussian".
  - Predict: how a 2-D Gaussian's contour changes when the correlation goes from 0 to 0.9.
  - Derivation step: completing the square gives the conditional precision. Which matrix block is it?
- **Figures**
  - `fund.gaussian/contours-eigen`: a 2-D Gaussian's contours with eigenvector axes scaled by √λ.
  - `fund.gaussian/conditional-slice`: the joint contours, a vertical slice at $x_b$, and the conditional density beside it. Notice the shifted mean and smaller variance.
  - `fund.gaussian/correlation-contours`: contours for ρ ∈ {0, 0.5, 0.9}. **[fig-Q]**
- **Pitfalls & source disagreements**
  - Precision-matrix parameterization (Bishop uses Λ = Σ⁻¹) vs covariance. The same symbol Λ means eigenvalues elsewhere.
  - "Uncorrelated Gaussians are independent" requires *joint* Gaussianity.

---

#### 50 · `fund.estimation`: Estimators: Bias, Variance, Consistency & the CLT
- **Level** core · **W2** · **prereqs** [fund.variance-covariance] · **covers** expectation (of estimators), estimator theory.
- **Sources**
  - Goodfellow DL §5.4.1–5.4.5 (point estimation, bias, variance & standard error, MSE trade-off, consistency). `dlb_ch05_ml.txt`
  - Murphy PML1 §4.7.1–4.7.2 sampling distributions and the Gaussian approximation of the MLE (pdf p.184–186), §2.8.6 CLT (pdf p.102), §5.3.1–5.3.3 risk, consistent and admissible estimators (pdf p.219–223). `pml1.txt`
  - d2l "Statistics" appendix §evaluating estimators (bias, variance, MSE). `d2l/d2l_statistics.txt`
- **Subtopic map**
  1. *Estimator as a random variable*, and its sampling distribution. Frequentist setup: θ fixed, data random.
  2. *Bias, variance, MSE*. Derive $\mathrm{MSE}=\mathrm{bias}^2+\mathrm{Var}$ (the cross term vanishes). Why a biased estimator can have lower MSE (shrinkage preview).
  3. *Worked estimators*:
     - Sample mean: unbiased, variance σ²/n.
     - $\hat\mu=X_1$: unbiased but useless.
     - Variance with 1/N: biased. Derive $E=\frac{N-1}N\sigma^2$, which gives Bessel's correction.
     - $s$ is biased for σ even with N−1 (by Jensen).
  4. *Consistency* (convergence in probability) vs unbiasedness. Examples of each without the other. Law of large numbers (statement, Chebyshev-based proof sketch for finite variance).
  5. *Standard error*: SE of the mean is σ/√n, estimated by s/√n. The √n law (4× data → half the error).
  6. *CLT*: statement, conditions (iid, finite variance), the Cauchy counterexample, and how fast it kicks in for skewed data. Multivariate CLT in one line.
  7. *Pointer*: the score, Fisher information, the Cramér–Rao bound, asymptotic normality of the MLE and the delta method moved to fund.fisher-information (review 2026-10-02: together they overflowed the 10-card budget, and they need the MLE first).
- **Question ideas**
  - Classify each estimator as biased/unbiased and consistent/inconsistent: $X_1$; $\bar X$; $\frac1N\sum(x_i-\bar x)^2$; $\bar X+1/N$.
  - (calc) SE of a mean with s = 2 and n = 400 → 0.1. With n = 1,600 → 0.05.
  - Derivation step: where does the N−1 come from in Bessel's correction? (fitting $\bar x$ uses up one degree of freedom)
  - Predict: SE when n quadruples (halves).
  - Which is false: "an unbiased estimator always has lower MSE than a biased one".
- **Figures**
  - `fund.estimation/bias-variance-darts`: dartboard 2×2 grid (low/high bias × low/high variance) built from simulated estimator draws.
  - `fund.estimation/clt-convergence`: sample-mean histograms of an exponential for n = 1, 5, 30 with the Normal overlay.
  - `fund.estimation/shrinkage-mse`: MSE of $c\bar X$ vs c. Notice the minimum at c < 1 (bias buys lower variance). **[fig-Q]**
- **Pitfalls & source disagreements**
  - DLB's standard error uses √(unbiased variance), which is itself biased. DLB notes this.
  - The bias–variance trade-off for *estimators* (here) is distinct from the one for *predictions* (fund.bias-variance).

---

#### 70 · `fund.mle`: Maximum Likelihood Estimation
- **Level** core · **W1** · **prereqs** [fund.estimation] · **covers** MLE vs MAP (the MLE half).
- **Sources**
  - Murphy PML1 §4.2 MLE: §4.2.1 definition, §4.2.2 justification via KL, worked MLEs §4.2.3 Bernoulli, §4.2.4 categorical, §4.2.5 univariate Gaussian, §4.2.6 MVN, §4.2.7 linear regression (pdf p.137–145). §4.3 ERM & surrogate losses (pdf p.145–147). `pml1.txt`
  - Goodfellow DL §5.5, §5.5.1 conditional log-likelihood & MSE, §5.5.2 properties of ML. `dlb_ch05_ml.txt`
  - Bishop PRML §1.2.4–1.2.5 Gaussian MLE & curve fitting (pdf p.44–50), §2.3.4 (pdf p.113), §9.2.1 GMM singularities (pdf p.452). `bishop.txt`
  - CS229 notes ch.1 "Probabilistic interpretation" (pdf p.17), ch.2 logistic regression MLE (pdf p.22), ch.3 "The exponential family" and GLMs (pdf p.31). `cs229.txt`
  - Murphy PML1 §3.4 the exponential family incl. §3.4.3 log partition = cumulant generating function (pdf p.123–125). Bishop PRML §2.4 (pdf p.133).
  - d2l "Maximum Likelihood" appendix. `d2l/d2l_maximum_likelihood.txt`. MacKay §22.1–22.4 (pdf p.312–318).
- **Subtopic map**
  1. *The likelihood function* $L(\theta)=p(D|\theta)$ is a function of θ, not a density over θ. With iid data it factorizes. Why we take logs: monotone, turns products into sums, numerically stable.
  2. *Worked MLEs* (derive each):
     - Bernoulli: $\hat p=h/n$.
     - Categorical: Lagrange multiplier on the simplex gives counts/n.
     - Gaussian: $\hat\mu=\bar x$, $\hat\sigma^2=\frac1N\sum(x_i-\bar x)^2$.
     - MVN covariance: trace trick and matrix derivative.
     - Poisson: λ̂ = mean.
     - Uniform $U(0,\theta)$: the likelihood is $\theta^{-n}$ for θ ≥ max xᵢ and 0 otherwise, so $\hat\theta=\max_ix_i$. Setting a derivative to zero fails (the maximum sits on a boundary), and $E\hat\theta=\frac n{n+1}\theta$ (biased low).
     - *Inline refreshers*: Lagrange multipliers (categorical), and $\nabla_A\log|A|=A^{-\top}$, $\nabla_A\mathrm{tr}(AB)=B^\top$ (MVN).
  3. *Conditional MLE for prediction*:
     - Gaussian noise gives MSE, and the noise MLE is $\hat\sigma^2=\mathrm{RSS}/N$.
     - Laplace noise gives MAE.
     - Bernoulli/categorical gives cross-entropy.
     - This is the NLL ↔ loss dictionary. "Training with CE/MSE" and next-token prediction are conditional MLE.
  4. *MLE = minimizing forward KL* to the empirical distribution: $\frac1N\sum-\log p_\theta(x_i)=H(\hat p)+\mathrm{KL}(\hat p\|p_\theta)$ (derive). Consequence: MLE is mass-covering.
  5. *Properties*:
     - Consistency: under identifiability and regularity.
     - Asymptotic normality and efficiency: statement only; derived in fund.fisher-information (next lesson).
     - Invariance: the MLE of g(θ) is g(θ̂). Explain why, since the likelihood is a function, not a density.
     - Bias: the Gaussian-variance and uniform examples show the MLE can be biased yet consistent.
  6. *Failure modes*:
     - Zero counts give zero probability, so the test NLL = ∞.
     - Separable logistic regression gives ‖w‖ → ∞.
     - GMM singularities: a component collapses on one point and the likelihood → ∞.
     - Overfitting with many parameters relative to data.
     - Non-identifiability (label switching, scaling symmetries in NNs).
  7. *Computing the MLE, and exponential families*: closed form vs iterative (GD/Newton); local optima in latent-variable models (pointer to fund.gmm-em). Then one card on exponential families, $p(x|\eta)=h(x)\exp(\eta^\top T(x)-A(\eta))$:
     - Derive $\nabla A(\eta)=E[T(x)]$ and $\nabla^2A(\eta)=\mathrm{Cov}[T(x)]\succeq0$.
     - So the NLL is convex **in the natural parameter η** (not necessarily in other parameterizations), and the MLE is moment matching: $E_{\hat\eta}[T]=\frac1N\sum_iT(x_i)$.
     - Bernoulli, categorical, Gaussian and Poisson as instances. The canonical link is why GLM gradients take the form (prediction − target) (pointer to fund.logistic-regression).
- **Question ideas**
  - (calc) Exponential distribution MLE of the rate from data {1, 2, 3} → 1/2.
  - (calc) MLE of $p^2$ given 3 heads in 10 → 0.09 (invariance).
  - (calc) $U(0,\theta)$ data {0.2, 0.9, 0.5}: θ̂ = 0.9, and $E\hat\theta=\frac34\theta$ for n = 3. Why can't you find it by setting the derivative to zero?
  - Derivation: for an exponential family, show $\nabla A(\eta)=E[T(x)]$, and conclude that the MLE matches moments.
  - Predict: an EM-fitted GMM component's variance → 0 on one point. What does the likelihood do?
  - Which is false: "the MLE is always unbiased".
  - Derivation step: in the categorical MLE, what enforces $\sum p_k=1$? (the Lagrange multiplier)
  - MLE ↔ loss matching: Laplace noise → MAE; Poisson → Poisson NLL.
- **Figures**
  - `fund.mle/likelihood-curve`: Bernoulli log-likelihood vs p for 3/10 and 30/100 heads. Notice that more data gives a sharper peak.
  - `fund.mle/gmm-singularity`: likelihood vs σ of a component pinned to one point. Notice the blow-up.
  - `fund.mle/mle-kl`: empirical histogram vs fitted Gaussian with the KL shaded.
- **Pitfalls & source disagreements**
  - Likelihood vs probability terminology. "Maximizing likelihood" vs "minimizing NLL": averaging over N changes no argmax but changes gradient scales.
  - Bishop uses β for noise precision. PML1 uses σ².

---

#### 72 · `fund.fisher-information`: Fisher Information, Cramér–Rao & MLE Asymptotics
- **Level** intermediate · **W2** · **prereqs** [fund.mle] · **covers** MLE vs MAP (properties of the MLE), confidence intervals (the Wald construction). Groundwork for natural gradient, K-FAC, the Laplace approximation and EWC. *New in review 2026-10-02* (split from fund.estimation items 7–10 and fund.mle's asymptotics).
- **Sources**
  - Murphy PML2 §3.3.3 asymptotic normality of the MLE, §3.3.4 the Fisher information matrix (pdf p.109–114). `pml2.txt`
  - Murphy PML1 §4.7.2 Gaussian approximation of the sampling distribution of the MLE (pdf p.185–186). `pml1.txt`
  - Goodfellow DL §5.5.2 properties of maximum likelihood (consistency, statistical efficiency, Cramér–Rao). `dlb_ch05_ml.txt`
  - Bishop PRML §4.4 the Laplace approximation (pdf p.233). `bishop.txt`
  - Martens 2020 (`papers/martens2014_natural_gradient.txt`) for Fisher vs empirical Fisher (named here; taught in fund.second-order).
- **Subtopic map**
  1. *The question*: how precisely can data pin down θ? Answer: by the curvature of the log-likelihood around its peak, averaged over datasets.
  2. *Score function* $s(\theta)=\nabla_\theta\log p(x|\theta)$. Derive $E[s]=0$ by swapping ∇ and ∫ (which needs the support not to depend on θ). Flag the clash with the diffusion "score" $\nabla_x\log p$.
  3. *Fisher information* $I(\theta)=\mathrm{Var}(s)=E[ss^\top]$, and the identity $I=-E[\nabla^2\log p]$ (derive). Additivity over iid samples ($nI$). Worked: Bernoulli $1/(p(1-p))$, Gaussian mean $1/\sigma^2$, Poisson $1/\lambda$. For exponential families $I(\eta)=\nabla^2A(\eta)=\mathrm{Cov}[T]$ (pointer to fund.mle).
  4. *Observed vs expected information*: $-\nabla^2\log L(\hat\theta)$ vs $nI(\hat\theta)$, and when they coincide (canonical exponential families).
  5. *Cramér–Rao lower bound*: $\mathrm{Var}(\hat\theta)\ge1/(nI(\theta))$ for unbiased estimators (derive the 1-D case from Cauchy–Schwarz on $\mathrm{Cov}(\hat\theta,s)=1$). Efficiency. Biased (shrinkage) estimators can beat it in MSE.
  6. *Asymptotic normality and efficiency of the MLE*: $\sqrt n(\hat\theta-\theta)\to N(0,I^{-1})$. Sketch: Taylor-expand the score around θ, apply the CLT to the average score and the LLN to the Hessian. Conditions (identifiability, θ in the interior, regularity) and a failure case: the $U(0,\theta)$ MLE has variance of order $1/n^2$, because the support depends on θ.
  7. *The delta method*: $\mathrm{Var}\,g(\hat\theta)\approx g'(\theta)^2\mathrm{Var}(\hat\theta)$ (derive from a first-order Taylor expansion). Example: the SE of the log-odds. This sets up Wald intervals (fund.confidence-intervals).
  8. *Geometry and ML links*: Fisher as the local metric of KL, $\mathrm{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^\top I\delta$ (pointers to fund.kl-divergence and fund.second-order). The Laplace approximation $p(\theta|D)\approx N(\hat\theta,H^{-1})$. Empirical Fisher vs Fisher, and EWC's Fisher-diagonal penalty (named).
- **Question ideas**
  - (calc) Cramér–Rao bound for Bernoulli p=0.3 with n=100: Var ≥ 0.0021.
  - (calc) Delta method: SE of $\hat p^2$ at p=0.5, n=100 → 2·0.5·0.05 = 0.05.
  - (calc) SE of the log-odds at $\hat p=0.2$, n=100: $1/\sqrt{np(1-p)}=0.25$.
  - Derivation step: which assumption lets us conclude $E[s]=0$? (differentiating under the integral sign, with a support that doesn't depend on θ)
  - Which is false: "the Cramér–Rao bound limits the MSE of every estimator, biased or not".
  - Predict: does the CRLB apply to the $U(0,\theta)$ MLE? (no: the support depends on θ, and its variance shrinks like $1/n^2$)
- **Figures**
  - `fund.fisher-information/curvature`: Bernoulli log-likelihoods for n = 10 and n = 100 with their quadratic (Fisher) approximations at the MLE. Notice that the curvature grows with n.
  - `fund.fisher-information/mle-sampling-dist`: histograms of simulated MLEs vs the $N(\theta,1/(nI))$ density for n = 5 and 50.
- **Pitfalls & source disagreements**
  - Per-sample $I(\theta)$ vs dataset $nI(\theta)$: texts differ on which they call "the" Fisher information.
  - "Fisher" in DL papers often means the empirical Fisher (fund.second-order).

---

#### 74 · `fund.confidence-intervals`: Confidence Intervals & the Bootstrap
- **Level** core · **W2** · **prereqs** [fund.estimation, fund.fisher-information] · **covers** confidence intervals. Split in review 2026-10-02: hypothesis tests, model comparison and multiple testing moved to fund.hypothesis-testing (the combined lesson was about 13 cards).
- **Sources**
  - Murphy PML1 §4.7.3 bootstrap, §4.7.4 CIs, §4.7.5 "CIs are not credible" (pdf p.186–189), §4.6.6 credible intervals (pdf p.176). `pml1.txt`
  - d2l "Statistics" appendix: constructing confidence intervals. `d2l/d2l_statistics.txt`
  - ESL §7.11 bootstrap methods (pdf p.268), §8.2 bootstrap & ML inference (pdf p.280–286). `esl.txt`
  - Murphy PML2 §3.3.2 bootstrap (pdf p.107). `pml2.txt`
  - MacKay ITILA §37.3 confidence intervals (pdf p.476–477). `mackay.txt`
- **Subtopic map**
  1. *What a CI is*: a random interval with coverage $P(\theta\in[L,U])=1-\alpha$ over repeated samples. The correct and incorrect interpretations, and the contrast with Bayesian credible intervals.
  2. *z-interval for a mean* (known σ), with a derivation from the CLT pivot. *t-interval* when σ is estimated, and why the t has heavier tails at small n.
  3. *Intervals for proportions and test accuracy*:
     - Wald $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$ (calc: 90/100 → [0.84, 0.96]).
     - Its failure near 0 or 1 and at small n.
     - Wilson and Clopper–Pearson (concept).
     - Rule of three (0 errors in n → 95% upper bound ≈ 3/n).
     - Test-set size planning (calc: ±1% at p=0.9 needs $1.96^2\cdot0.09/0.01^2=3{,}457.4$, so 3,458).
  4. *Wald intervals from Fisher information* (pointer to fund.fisher-information): $\hat\theta\pm z/\sqrt{nI(\hat\theta)}$. The delta method for transformed parameters, and why it's better to build the interval on the log-odds scale and map it back.
  5. *Bootstrap*:
     - Resample with replacement and recompute the statistic.
     - The percentile interval (basic/BCa named only).
     - P(a point is absent from a resample) $=(1-1/n)^n\to e^{-1}$, so a resample contains ≈ 63.2% of the unique points.
     - When the bootstrap fails: the max of a uniform, heavy tails, dependent data (use the block bootstrap).
  6. *Reading error bars in ML papers*: ±1 SE vs ±2 SE vs a 95% CI vs ± one std across seeds, and what each claims. Credible intervals as the Bayesian counterpart (one paragraph; MacKay's critique continues in fund.hypothesis-testing).
- **Question ideas**
  - Which statement about a 95% CI is correct? Distractors are Bayesian-sounding misreadings.
  - (calc) Accuracy 0.9 on n=100: Wald CI. Then n=400: the half-width halves.
  - (calc) Zero failures in 300 trials: 95% upper bound ≈ 1%.
  - (calc) Test-set size for ±1% at p = 0.9: 3,458.
  - Predict: the Wald interval when a model gets 20/20 right (degenerate [1, 1]), and what to use instead.
  - Bootstrap: fraction of unique points ≈ 0.632, and the OOB fraction ≈ 0.368.
- **Figures**
  - `fund.confidence-intervals/ci-coverage`: 50 simulated 95% CIs with the misses highlighted.
  - `fund.confidence-intervals/wald-vs-wilson`: coverage vs p for n=20. **[fig-Q]**: "which curve is Wald?"
  - `fund.confidence-intervals/bootstrap-dist`: bootstrap histogram of a median with percentile bounds.
- **Pitfalls & source disagreements**
  - MacKay ch.37 attacks CIs. PML1 and d2l present them neutrally. Teach both views.
  - z vs t at small n. "95% CI" with ±2·SE vs ±1.96·SE (cosmetic).

---

#### 76 · `fund.hypothesis-testing`: Hypothesis Tests, p-values & Comparing Models
- **Level** core · **W2** · **prereqs** [fund.confidence-intervals] · **covers** confidence intervals (test–interval duality), significance and model comparison. *New in review 2026-10-02* (split from fund.confidence-intervals; power and sample size added).
- **Sources**
  - Murphy PML1 §5.5 frequentist hypothesis testing: §5.5.1 likelihood-ratio test, §5.5.2 type I/II errors & Neyman–Pearson, §5.5.3 NHST and p-values, §5.5.4 "p-values considered harmful" (pdf p.228–232); §5.2.1 Bayesian hypothesis testing (pdf p.209). `pml1.txt`
  - d2l "Statistics" appendix: hypothesis testing. `d2l/d2l_statistics.txt`
  - ESL §18.7 multiple testing & FDR incl. §18.7.1 Benjamini–Hochberg (pdf p.702–711). `esl.txt`
  - MacKay ITILA ch.37 §37.1–37.2 (p-values, irrelevant information; pdf p.469–476). `mackay.txt`
  - Murphy PML2 §3.3.5 counterintuitive properties of frequentist statistics (pdf p.114). `pml2.txt`
  - Tuning Playbook (`tp_tuning_playbook.md`) on trial, study and seed variance.
- **Subtopic map**
  1. *The logic of a test*: null and alternative, test statistic, the sampling distribution under H0, the p-value (and three things it is not), type I/II errors, significance level and power. A worked one-sample z-test.
  2. *Duality with CIs*: reject H0: θ = θ0 at level α iff θ0 lies outside the 1−α interval (derive for the z case).
  3. *Power and sample size*: power as a function of effect size and n. Per-arm $n\approx2(z_{1-\alpha/2}+z_{1-\beta})^2\sigma^2/\delta^2$ (derive). Why underpowered comparisons are common on ML benchmarks. An A/B test is the same calculation.
  4. *Comparing two models on the same test set*: paired designs. McNemar's test on the discordant pairs (derive the statistic from a binomial on b vs c), and the paired bootstrap or permutation test of per-example differences. Why overlapping marginal CIs is not a test, and why the paired SE is smaller.
  5. *Seeds and variance in DL results* (Tuning-plan insertion): trial-to-trial variance from init and data order. The SE of a difference of two means is $\sigma\sqrt{2/n}$; worked example with 5 seeds per arm and σ = 0.1% → SE ≈ 0.063%. Seed variance alone can make identical configs differ "significantly", so retrain the best trial several times before claiming a gain.
  6. *Multiple comparisons*: family-wise error and Bonferroni, FDR and Benjamini–Hochberg (ESL §18.7.1), and why the best of k configurations on a test set is optimistic (pointer to fund.model-selection).
  7. *Likelihood-ratio tests* and the Neyman–Pearson lemma (statement). Wilks' χ² approximation named.
  8. *Critiques and the Bayesian alternative*: p-hacking, optional stopping, MacKay's irrelevant-information example. Bayes factors and credible intervals (one card).
- **Question ideas**
  - Which statement about p = 0.03 is correct? Distractors: "P(H0 is true) = 0.03", "a 3% chance the result is due to chance", "a 97% chance the effect is real".
  - Spot the flaw: "A 85.1±1.2, B 84.5±1.3, the intervals overlap, so there is no difference".
  - (calc) McNemar with b = 30 and c = 15 discordant pairs: χ² with continuity correction = 14²/45 ≈ 4.36, p ≈ 0.037.
  - (calc) Bonferroni threshold for 20 tests at α=0.05 → 0.0025.
  - (calc) Examples per arm to detect a 5-point accuracy difference near 50% at α = 0.05 with power 0.8: ≈ 1,570.
  - (calc) 5 seeds per arm with σ = 0.1%: SE of the difference ≈ 0.063%.
  - Predict: run 20 identical configs and report the best one. What does the "improvement" look like? (pure selection bias)
- **Figures**
  - `fund.hypothesis-testing/p-value`: the null sampling distribution with the observed statistic and the shaded tail. **[fig-Q]**: "which area is the p-value?"
  - `fund.hypothesis-testing/power-curve`: power vs effect size for n = 100, 400 and 1,600.
  - `fund.hypothesis-testing/paired-vs-unpaired`: per-example score differences showing a significant paired difference despite overlapping marginal CIs.
- **Pitfalls & source disagreements**
  - MacKay ch.37 and PML1 §5.5.4 are critical of p-values; d2l presents them neutrally. Teach both views.
  - One- vs two-sided tests. McNemar with or without the continuity correction.

---

#### 80 · `fund.mle-map`: MAP Estimation, Priors & Bayesian Prediction *(keeps id)*
- **Level** core · **W1** · **prereqs** [fund.mle, fund.bayes-theorem] · **covers** MLE vs MAP.
- **Sources**
  - Murphy PML1 §4.5 regularization: §4.5.1 MAP for Bernoulli, §4.5.2 MAP for MVN, §4.5.3 weight decay (pdf p.150–154). §4.6.1–4.6.2 conjugate priors & beta-binomial (pdf p.159–167). §4.6.3 Dirichlet-multinomial (pdf p.167). §4.6.4 Gaussian-Gaussian (pdf p.171). §4.6.7 Bayesian ML (pdf p.177). `pml1.txt`
  - Goodfellow DL §5.6, §5.6.1 MAP. `dlb_ch05_ml.txt`
  - Bishop PRML §2.1.1 beta (pdf p.91), §2.2.1 Dirichlet (pdf p.96), §2.3.6 Bayesian inference for the Gaussian (pdf p.117), §3.3.1–3.3.2 posterior & predictive for linear regression (pdf p.172–179). `bishop.txt`
  - CS229 notes ch.9 "Bayesian statistics and regularization" (pdf p.145). `cs229.txt`
- **Subtopic map**
  1. *From MLE to MAP*: $\arg\max p(\theta|D)=\arg\max[\log p(D|\theta)+\log p(\theta)]$, with the evidence dropped. Why: it doesn't depend on θ.
  2. *Priors as regularizers*:
     - Gaussian prior → L2/ridge, with λ = σ²/τ² (derive by multiplying through by 2σ²).
     - Laplace prior → L1/lasso, with λ = 2σ²/b.
     - A tighter prior means a larger λ.
     - The sparsity of L1-MAP comes from the kink at 0. The posterior *mean* under a Laplace prior is not sparse.
  3. *Conjugacy*: definition. Beta–Bernoulli posterior $\mathrm{Beta}(\alpha+h,\beta+n-h)$, derived by multiplying kernels. Pseudo-count interpretation. Where prior hyperparameters (α, β) are introduced, add one sentence: deep learning's "hyperparameters" (learning rate, weight decay) are not parameters of a prior; the Tuning Playbook calls them metaparameters (Tuning-plan insertion).
  4. *Three point estimates from one posterior*:
     - Mode (MAP): $\frac{h+\alpha-1}{n+\alpha+\beta-2}$.
     - Mean: $\frac{h+\alpha}{n+\alpha+\beta}$.
     - MLE: h/n.
     - Worked numbers (calc: 3/3 heads with Beta(2,2) gives MAP 4/5, mean 5/7, MLE 1).
  5. *Dirichlet–categorical*: add-α smoothing = MAP or posterior mean. Laplace smoothing in naive Bayes.
  6. *Gaussian mean with known variance*: the posterior mean is a precision-weighted average, $\mu_N=\frac{\sigma^2\mu_0+N\tau^2\bar x}{\sigma^2+N\tau^2}$, and the posterior variance shrinks as 1/N (derive).
  7. *Posterior predictive vs plug-in*: $p(x_{new}|D)=\int p(x_{new}|\theta)p(\theta|D)d\theta$. For Beta–Bernoulli the predictive is the posterior mean (rule of succession).
     Bayesian LR predictive variance = noise + parameter uncertainty (pointer to fund.ridge-lasso).
  8. *MAP is not reparameterization-invariant*: the Jacobian moves the mode (worked: the Beta posterior over p vs over logit p). MLE is invariant. The MAP mode can be atypical in high dimensions.
  9. *Large-N behaviour*: the likelihood dominates and MAP → MLE. Bernstein–von Mises (posterior ≈ Gaussian around the MLE) as a statement.
  10. *Weight decay and dataset size*: with a *mean* loss, λ = σ²/(Nτ²). This is why weight-decay values don't transfer across dataset sizes.
  11. *Beyond MAP*: full Bayes via VI, MCMC, Laplace approximation, and deep ensembles as approximations (one card, pointing to the gen/LLM areas).
- **Question ideas** (keep existing q1–q8 where unchanged, and add):
  - (calc) Posterior mean of μ: prior N(0, 1), σ²=1, n=4, $\bar x=2$ → 1.6.
  - (calc) Add-1 smoothing for a 3-class categorical with counts (5, 0, 1) → (6/9, 1/9, 2/9).
  - Predict: MAP vs MLE as N → ∞ with a fixed prior.
  - Which is false: "MAP and the posterior mean coincide for any symmetric posterior". (They coincide for symmetric *unimodal* posteriors, so the claim is false.)
  - Weight decay: dataset size doubles with a mean loss. To keep the same prior, λ should… (halve)
- **Figures**
  - `fund.mle-map/beta-posterior`: prior, likelihood and posterior for 3/3 heads with MLE, MAP and mean marked. **[fig-Q]**: "which marker is the MAP?"
  - `fund.mle-map/l1-l2-priors`: Gaussian vs Laplace densities and their negative logs.
  - `fund.mle-map/map-reparam`: the same posterior over θ and over logit θ, with the modes mapped across.
  - `fund.mle-map/posterior-concentration`: posterior for n = 1, 10, 100.
- **Pitfalls & source disagreements**
  - Bishop writes λ = α/β (α prior precision, β noise precision). PML1 writes σ²/τ². This is the same quantity.
  - Some texts call any regularized MLE "MAP" even when the penalty is not a normalized log-prior (e.g. non-convex penalties).

---

#### 90 · `fund.information-theory`: Entropy, Cross-Entropy & Mutual Information
- **Level** core · **W1** · **prereqs** [fund.probability-basics] · **covers** entropy (+ cross-entropy, MI). Promoted to W1 in review: cross-entropy/perplexity are core for LLM roles, and the W1 lessons fund.kl-divergence and fund.logistic-regression build on it.
- **Sources**
  - Murphy PML1 §6.1.1–6.1.6 entropy, cross-entropy, joint & conditional entropy, perplexity, differential entropy (pdf p.237–243), §6.3 MI incl. §6.3.8 data processing (pdf p.247–255). `pml1.txt`
  - MacKay ITILA §2.4–2.5 entropy & decomposability (pdf p.44–46), ch.4 §4.1 information content (pdf p.79), ch.8 dependent random variables (pdf p.150). `mackay.txt`
  - Bishop PRML §1.6 (pdf p.68–75), §1.6.1 MI part (pdf p.75). `bishop.txt`
  - Goodfellow DL §3.13. d2l "Information Theory" appendix. `d2l/d2l_information_theory.txt`
- **Subtopic map**
  1. *Information content* −log p(x). Why log: additivity for independent events. Units: bits (log₂) vs nats (ln).
  2. *Entropy*: definition, interpretation as expected surprise and as the optimal average code length (source coding, statement only). The binary entropy curve. Entropy is concave.
  3. *Maximum entropy*: the uniform over K outcomes maximizes it at log K (prove via Gibbs/Jensen). The Gaussian maximizes differential entropy for a fixed variance.
  4. *Differential entropy*: definition, it can be negative, and it changes under rescaling ($h(aX)=h(X)+\log|a|$). The Gaussian value $\frac12\log(2\pi e\sigma^2)$.
  5. *Joint and conditional entropy*, the chain rule $H(X,Y)=H(X)+H(Y|X)$, subadditivity $H(X,Y)\le H(X)+H(Y)$ (equality iff independent), and "conditioning reduces entropy" on average (but $H(X|Y=y)$ can exceed $H(X)$ for a particular y; give an example).
  6. *Cross-entropy* $H(p,q)=-\sum p\log q$. Decomposition $H(p,q)=H(p)+\mathrm{KL}(p\|q)$ (derive; full KL treatment in the next lesson).
     - The CE of a one-hot label is $-\log q_y$.
     - Why minimizing CE over q equals minimizing KL.
     - The loss floor is H(p) (label noise means the CE can't reach 0).
  7. *Perplexity* = exp(CE). The uniform distribution over V has perplexity V. Interpretation as an effective branching factor (LLM link).
  8. *Mutual information*: $I(X;Y)=H(X)-H(X|Y)=H(X)+H(Y)-H(X,Y)=\mathrm{KL}(p(x,y)\|p(x)p(y))$ (derive the equalities). Symmetric, ≥ 0, and = 0 iff independent. Conditional MI and the chain rule $I(X;Y,Z)=I(X;Y)+I(X;Z|Y)$. Contrast with correlation (worked: $Y=X^2$ has I > 0 and ρ = 0).
  9. *Data processing inequality*: X→Y→Z implies $I(X;Z)\le I(X;Y)$. Implication: a network can't create information about the input.
  10. *ML uses*: decision-tree information gain (fund.decision-trees), the InfoNCE bound in contrastive learning (named), and the information bottleneck (named).
- **Question ideas**
  - (calc) Entropy of (½, ¼, ¼) is 1.5 bits. Of the uniform over 8 it is 3 bits.
  - (calc) A model's CE is 2.0 nats → perplexity e² ≈ 7.39.
  - (calc) Joint table $p=\begin{pmatrix}0.4&0.1\\0.1&0.4\end{pmatrix}$: $I(X;Y)=1-H_2(0.2)\approx0.278$ bits.
  - Which is false: "differential entropy is always non-negative".
  - Predict: CE floor on a dataset with 10% label noise in binary labels (≥ H(0.1) ≈ 0.325 nats).
  - Mutual information vs correlation for $Y=X^2$.
  - Information diagram: identify H(X|Y) in the Venn diagram.
- **Figures**
  - `fund.information-theory/binary-entropy`: H(p) curve with its maximum at 0.5.
  - `fund.information-theory/entropy-venn`: information diagram showing H(X), H(Y), H(X|Y), H(Y|X) and I(X;Y). **[fig-Q]**
  - `fund.information-theory/ce-decomposition`: stacked bar for several q's showing H(p) plus KL(p‖q) = H(p,q).
- **Pitfalls & source disagreements**
  - $H(p,q)$ (cross-entropy) vs $H(X,Y)$ (joint entropy) notation.
  - Bits vs nats: PyTorch losses are in nats, while papers often report bits per dimension or bits per character.

---

#### 100 · `fund.kl-divergence`: KL & Jensen–Shannon Divergences *(keeps id; rewritten)*
- **Level** core · **W1** · **prereqs** [fund.information-theory] · **covers** KL divergence, JS divergence.
- **Sources**
  - Murphy PML1 §6.2 KL: §6.2.1–6.2.2 definition & interpretation, §6.2.3 Gaussians, §6.2.4 non-negativity, §6.2.5 KL & MLE, §6.2.6 forward vs reverse (pdf p.243–247). `pml1.txt`
  - Murphy PML2 §5.1 KL deep dive (§5.1.3 thinking about KL, §5.1.4 minimizing KL, §5.1.5 properties, §5.1.9 Fisher approximation; pdf p.253–267), §2.7.1 f-divergences, §2.7.2–2.7.4 IPMs, MMD, TV (pdf p.89–95). `pml2.txt`
  - MacKay ITILA §2.6 Gibbs' inequality, §2.7 Jensen (pdf p.46–47). `mackay.txt`
  - Bishop PRML §1.6.1 relative entropy (pdf p.75), §10.1.2 KL(q‖p) vs KL(p‖q) behaviour (pdf p.486). `bishop.txt`
  - Goodfellow et al. 2014 Prop. 1/Thm 1 (`papers/goodfellow2014_gan.txt`). Arjovsky & Bottou 2017 (`papers/arjovsky2017_principled_gan.txt`). Nowozin et al. 2016 f-GAN (`papers/nowozin2016_fgan.txt`).
- **Subtopic map**
  1. *Definition and interpretation*: the extra nats paid for coding p with a code optimized for q. Discrete and continuous forms.
  2. *Non-negativity*: proof via Jensen (−log is convex), with equality iff p = q a.e. (Gibbs' inequality).
  3. *Properties*:
     - Asymmetric.
     - Not a metric (no triangle inequality).
     - Infinite if q = 0 where p > 0.
     - Invariant to invertible reparameterizations of x (contrast with differential entropy).
     - Additive for independent factors.
  4. *Gaussian KL in closed form*: the univariate case, derived. The multivariate formula, with each term interpreted: trace, Mahalanobis and log-det. The special case $\mathrm{KL}(N(\mu,\sigma^2)\|N(0,1))=\frac12(\mu^2+\sigma^2-1-\log\sigma^2)$, used in the VAE.
  5. *Forward vs reverse KL*:
     - Fitting a unimodal q to a bimodal p: KL(p‖q) is mass-covering (zero-avoiding), KL(q‖p) is mode-seeking (zero-forcing).
     - Explain the asymmetry by where each one's log ratio blows up.
     - MLE uses forward KL. VI uses reverse KL. Links to blurry VAEs vs mode-dropping GANs.
  6. *KL and MLE* (recap from fund.mle). *KL and Fisher information*: locally, $\mathrm{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^\top F\delta$ (sets up natural gradient).
  7. *Jensen–Shannon*:
     - Definition via the mixture M = (P+Q)/2.
     - Symmetric, bounded by log 2 nats (1 bit), and √JSD is a metric.
     - JSD equals the MI between a sample and the indicator of which distribution produced it (derive).
  8. *JSD in GANs* (state the result; gen.gans owns the derivation of $D^*$ and $C(G)=-\log4+2\,\mathrm{JSD}$, so don't duplicate it):
     - With disjoint or low-dimensional supports, JSD = log 2 is constant, so the gradient vanishes (Arjovsky). Work the two-point-mass example.
     - Contrast with Wasserstein (preview of gen.wgan).
  9. *f-divergence family*: $D_f(P\|Q)=E_Q[f(p/q)]$ with f convex and f(1)=0. KL, reverse KL, JS, TV and χ² as special cases. Variational (Fenchel) lower bound (named; detail in gen.wgan).
  10. *Estimating KL from samples*: the Monte Carlo log-ratio estimator and its variance. Schulman's k1/k2/k3 estimators are named in the cache as `llm_schulman2020_kl_approx.txt` (used in RLHF).
- **Question ideas**
  - (calc) KL between (0.5, 0.5) and (0.9, 0.1) in both directions: 0.511 vs 0.368 nats. This shows the asymmetry.
  - (calc) KL(N(1,1)‖N(0,1)) = 0.5.
  - Predict: fitting a Gaussian to a bimodal mixture by reverse KL (locks onto one mode).
  - (calc) JSD between disjoint distributions = log 2.
  - Which is false: "KL obeys the triangle inequality". True distractors: invariant to invertible maps of x; JSD symmetric; KL ≥ 0.
  - Derivation step: in the KL ≥ 0 proof, which inequality is applied and to which function?
- **Figures**
  - `fund.kl-divergence/forward-vs-reverse`: bimodal p with the best Gaussian under each direction. **[fig-Q]**: "which fit minimized KL(q‖p)?"
  - `fund.kl-divergence/kl-asymmetry`: the integrands $p\log(p/q)$ and $q\log(q/p)$ plotted for two Gaussians with different variances.
  - `fund.kl-divergence/js-saturation`: JSD and W1 vs separation θ for two point masses. Notice JSD flat at log 2.
- **Pitfalls & source disagreements**
  - "Forward/reverse" naming is inconsistent (PML1 calls KL(p‖q) "forwards/inclusive"; some VI texts flip it). Always write the formula.
  - JSD bound is log 2 vs 1 depending on the log base. Goodfellow's −log 4 is in nats.
  - Item ids: this file replaces the old KL topic's content. Keep ids only for items whose question is unchanged.

---
### Optimization foundations

---

#### 104 · `fund.convexity`: Convex Sets & Convex Functions
- **Level** core · **W2** · **prereqs** [] · **covers** convex functions.
- **Sources**
  - Boyd & Vandenberghe:
    - §2.1–2.3 convex sets and operations that preserve convexity (pdf p.35–57).
    - §3.1 convex functions: §3.1.1 definition, §3.1.3 first-order condition (pdf p.83), §3.1.4 second-order condition (pdf p.85), §3.1.5 examples, §3.1.8 Jensen's inequality.
    - §3.2 operations preserving convexity incl. composition rules (pdf p.93–104).
    - §9.1.2 strong convexity & implications (pdf p.473).
    - `boyd.txt` · https://web.stanford.edu/~boyd/cvxbook/
  - UML (SSBD) §12.1 convexity, Lipschitzness, smoothness (p.157–163), §12.2–12.3 convex learning problems & surrogates (p.163–168). `uml.txt`
  - Murphy PML1 §8.1.3 convex vs non-convex, §8.1.4 smooth vs non-smooth, incl. subgradients (pdf p.307–312). `pml1.txt`
  - d2l convexity (`d2l/d2l_convexity.txt`).
- **Subtopic map**
  1. *Why convexity matters*: every local minimum is global (prove in two lines), the set of minimizers is convex, and duality and convergence guarantees follow. Most DL losses are non-convex in the parameters but convex in the predictions.
  2. *Convex sets*: definition. Examples (halfspaces, balls, PSD cone). Intersections preserve convexity.
  3. *Convex functions*, three equivalent characterizations, each with its geometric picture:
     - Chord above the graph (Jensen).
     - First-order: $f(y)\ge f(x)+\nabla f(x)^\top(y-x)$, the tangent is a global under-estimator.
     - Second-order: $\nabla^2f\succeq0$.
     - Show first-order ⇒ "∇f = 0 is optimal".
     - Practical tests: restriction to a line ($g(t)=f(x+tv)$ convex for all x, v); a quadratic $x^\top Ax+b^\top x$ is convex iff $A\succeq0$ (2×2 check via trace and determinant); the Hessians of least squares ($2X^\top X$) and of the logistic loss ($X^\top SX$) as PSD examples. Sublevel sets of a convex function are convex, but the converse fails (quasi-convexity).
  4. *Strict and strong convexity*: μ-strongly convex means $f(y)\ge f(x)+\nabla f^\top(y-x)+\frac\mu2\|y-x\|^2$, equivalently $\nabla^2f\succeq\mu I$. Gives a unique minimizer and quadratic growth.
  5. *Smoothness*: L-Lipschitz gradient, equivalently $\nabla^2f\preceq LI$. The quadratic upper bound. The condition number κ = L/μ (sets up GD rates).
  6. *Operations that preserve convexity*:
     - Non-negative sums.
     - Composition with affine maps.
     - Pointwise max/sup (e.g. hinge, max of linear functions).
     - Composition rules: f convex nondecreasing ∘ g convex is convex, with a counterexample when f isn't monotone.
     - Partial minimization.
     - Perspective (named).
  7. *Catalogue of ML-relevant convex functions*, each with a one-line justification:
     - Norms.
     - Squared loss.
     - Log-sum-exp (Hessian argument).
     - Logistic loss.
     - Negative log-likelihood of exponential families in natural parameters.
     - −log det on PD matrices.
     - Hinge.
     - Max eigenvalue.
  8. *Non-smooth convex functions and subgradients*: definition, the subdifferential of |x| at 0 is [−1, 1], subgradient optimality 0 ∈ ∂f. Used for lasso and hinge.
  9. *Recognizing non-convexity*: neural nets (weight-space symmetries give multiple equivalent minima, so the loss can't be strictly convex), matrix factorization, and the k-means objective in (μ, r).
  10. *Convex surrogates* for the 0-1 loss (pointer to fund.loss-functions).
- **Question ideas**
  - Which is not convex on ℝ: $x^3$, |x|, $e^x$, $\log(1+e^x)$? ($x^3$)
  - Is the max of convex functions convex? The min? (yes; no)
  - Is $f(g(x))$ convex if both f and g are convex? (not in general. It is guaranteed when f is also nondecreasing. Counterexample: $f(u)=e^{-u}$ and $g(x)=x^2$ give $e^{-x^2}$, which is not convex)
  - (calc) Hessian of log-sum-exp at z=(0,0): $\frac14\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$, which is PSD but singular. What does the singular direction mean? (shift invariance)
  - Why can't a neural network loss be strictly convex in its weights? (permutation symmetry of hidden units)
  - Subgradient of max(0, 1−x) at x=1? ([−1, 0])
  - (calc) Is $f(x,y)=x^2+3xy+y^2$ convex? Its Hessian $\begin{pmatrix}2&3\\3&2\end{pmatrix}$ has eigenvalues 5 and −1, so no: it's a saddle.
- **Figures**
  - `fund.convexity/definitions`: one convex function showing the chord above the graph and the tangent below it.
  - `fund.convexity/strong-vs-smooth`: f sandwiched between the quadratic lower bound (μ) and upper bound (L) at a point.
  - `fund.convexity/convex-or-not`: six function plots. **[fig-Q]**: "which are convex?"
  - `fund.convexity/subgradients`: |x| at 0 with a fan of subgradient lines.
- **Pitfalls & source disagreements**
  - "Concave up" vs "convex" terminology. Strict and strong convexity are often conflated: $x^4$ is strictly but not strongly convex (its second derivative vanishes at 0), and $e^x$ is strictly convex with no minimizer.
  - Quasi-convexity (unimodality) is weaker than convexity.

---

#### 106 · `fund.gradient-descent`: Gradient Descent: Convergence & Conditioning
- **Level** core · **W1** · **prereqs** [fund.convexity] · **covers** gradient descent. Promoted to W1 in review: it is the prereq of the W1 lesson fund.sgd-momentum, and conditioning and η < 2/L are standard interview probes.
- **Sources**
  - Boyd & Vandenberghe §9.2 descent methods & line search (pdf p.477), §9.3 GD incl. §9.3.1 convergence analysis (pdf p.480–489), §9.4 steepest descent incl. §9.4.4 choice of norm/preconditioning (pdf p.489–498). `boyd.txt`
  - Goodfellow DL §4.3 gradient-based optimization, §4.3.1 Jacobian/Hessian, curvature and the second-order Taylor step analysis, condition number. `dlb_ch04_numerical.txt`
  - Murphy PML1 §8.2 first-order methods: descent direction, step size & line search, §8.2.3 convergence rates incl. the quadratic analysis (pdf p.312–317). `pml1.txt`
  - UML (SSBD) §14.1 GD & the convex-Lipschitz analysis (p.185–189). Bottou, Curtis & Nocedal 2018 §4.1 (`papers/bottou2018_large_scale_opt.txt`). d2l gd (`d2l/d2l_gd.txt`).
- **Subtopic map**
  1. *Update and motivation*: first-order Taylor expansion, so −∇f is the steepest-descent direction in L2 (derive). Step size η.
  2. *What the Hessian tells you* (DLB §4.3.1): second-order Taylor expansion of $f(x-\eta g)$ gives $f-\eta g^\top g+\frac{\eta^2}2g^\top Hg$. The optimal η along g is $g^\top g/g^\top Hg$. Curvature ruins steps.
  3. *Descent lemma for L-smooth f*: $f(x-\eta\nabla f)\le f(x)-\eta(1-\frac{L\eta}2)\|\nabla f\|^2$ (derive from the quadratic upper bound). It guarantees decrease for η < 2/L. Best fixed η = 1/L.
  4. *Rates*:
     - Convex + L-smooth: O(1/k) for $f(x_k)-f^*$ (proof sketch).
     - μ-strongly convex + L-smooth: linear rate $(1-\mu/L)^k$.
     - Non-convex + smooth: $\min_k\|\nabla f\|^2=O(1/k)$, convergence to a stationary point.
  5. *Exact quadratic analysis*: for $f=\frac12x^\top Ax$, the error in each eigendirection evolves as $(1-\eta\lambda_i)^k$. Stability needs η < 2/λ_max. The best fixed η = 2/(λ_max + λ_min) gives the rate (κ−1)/(κ+1). Zig-zag in ill-conditioned valleys.
  6. *Line search*: exact vs backtracking (Armijo). Why DL uses fixed schedules instead (noise, cost).
  7. *Preconditioning and steepest descent in other norms*: $x\leftarrow x-\eta P^{-1}\nabla f$. Choosing P ≈ H makes κ → 1 (Newton as the limit, pointer to fund.newton). Feature scaling/whitening as preconditioning (pointer to fund.whitening). Adam as a diagonal preconditioner (pointer to fund.adam).
  8. *Non-convex landscapes*: saddle points vs local minima in high dimensions (DLB §8.2.3). GD can be slow near saddles. Plateaus. Cliffs and exploding gradients (DLB §8.2.4).
  9. *Why deep nets go unstable* (Tuning-plan insertion): the sharpness $\lambda_{max}$ of the loss Hessian rises during training (progressive sharpening; Cohen et al. 2021, `tp_cohen2021_edge_of_stability.txt`) and divergence occurs when $\lambda_1>2/\eta$ (Gilmer et al. 2021, `tp_gilmer2021_curvature_instability.txt`). One paragraph tying item 5 to real training; → link sys.tuning-diagnostics. (Gradient checking moved to fund.backprop.)
- **Question ideas**
  - (calc) $f=\frac12(x^2+100y^2)$: GD diverges for η > 0.02 (at η = 0.02 exactly, the y-coordinate flips sign forever without decaying). The optimal fixed η = 2/101 gives rate 99/101 ≈ 0.98, so ~230 steps per 1e-2 error reduction.
  - Predict: η slightly above 2/L (oscillating divergence along the top eigenvector).
  - **[fig-Q]** Loss curves for four learning rates: which has η > 2/L?
  - Which is false: "GD on a convex smooth function converges linearly".
  - Derivation step: in the descent lemma, which property bounds the second-order term?
  - (calc) Optimal step along g for H = diag(1, 10), g = (1, 1): $g^\top g/g^\top Hg$ = 2/11.
- **Figures**
  - `fund.gradient-descent/zigzag`: GD paths on well- vs ill-conditioned quadratic contours.
  - `fund.gradient-descent/lr-regimes`: loss vs iteration for η too small, good, near 2/L and above 2/L. **[fig-Q]**
  - `fund.gradient-descent/eigen-contraction`: |1 − ηλ| vs η for λ_min and λ_max, with the optimal η at the crossing.
- **Pitfalls & source disagreements**
  - Rates are stated for different quantities: function gap, distance to the optimum, gradient norm. Read carefully.
  - DLB uses ε for the learning rate, others use η or α.

---
### Learning theory & methodology

---

#### 110 · `fund.learning-setups`: Learning Setups, Risk, ERM & No Free Lunch
- **Level** core · **W2** · **prereqs** [fund.probability-basics] · **covers** supervised vs unsupervised, NFL theorem.
- **Sources**
  - UML (SSBD) ch.2 ERM, overfitting, inductive bias (p.33–41); §5.1 NFL with proof, §5.1.1 NFL & prior knowledge, §5.2 error decomposition (p.60–65). `uml.txt` · https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
  - Goodfellow DL §5.1 (task, performance, experience; §5.1.3 supervised vs unsupervised and how each reduces to the other), §5.2.1 NFL. `dlb_ch05_ml.txt`
  - Murphy PML1 §1.2–1.4 (supervised, unsupervised, self-supervised, RL; §1.2.4 NFL; pdf p.31–48), §5.1.1 Bayesian decision theory basics (pdf p.197), §5.4.1–5.4.2 empirical & structural risk (pdf p.223–226). `pml1.txt`
  - ESL §2.4 statistical decision theory (pdf p.37), §14.1 & §14.2.4 unsupervised as supervised (pdf p.504, p.514). `esl.txt`
- **Subtopic map**
  1. *The setups*:
     - Supervised: learn f: x→y, or p(y|x).
     - Unsupervised: model p(x), or find structure such as clusters, low-dimensional manifolds or density.
     - Self-supervised: build labels from the data itself (masking, next-token, contrastive views).
     - Semi-supervised and weakly supervised.
     - RL: reward feedback, no labels.
     - Give a concrete example of each, and explain why the boundaries blur.
     - Parametric vs nonparametric models: a fixed number of parameters (logistic regression) vs capacity that grows with N (kNN, trees, kernel methods). One line, as interviewers often ask it alongside this question.
  2. *Interconversion* (DLB §5.1.3): the chain rule turns density estimation into a sequence of supervised problems, and Bayes turns a joint model into p(y|x).
  3. *Generative vs discriminative models* (pointer to fund.bayes-theorem).
  4. *Loss, risk, empirical risk*: $R(f)=E_{(x,y)\sim D}\ell(f(x),y)$ vs $\hat R(f)=\frac1n\sum\ell$. Why we can't minimize R directly.
  5. *Bayes-optimal predictors* (derive each from pointwise minimization of the conditional risk):
     - Squared loss → E[y|x].
     - Absolute loss → median.
     - 0-1 loss → argmax p(y|x).
     - Bayes error as an irreducible floor.
  6. *ERM and overfitting*: the memorizing-lookup-table counterexample (UML §2.2.1). Why we need to restrict the hypothesis class H (inductive bias).
  7. *Error decomposition*: approximation error + estimation error (UML §5.2), plus optimization error (Bottou–Bousquet). The bias–complexity trade-off.
  8. *No Free Lunch*:
     - The UML statement: for any learner and m ≤ |X|/2 samples, some distribution with a perfect labelling function makes the expected error ≥ 1/4, so P(error ≥ 1/8) ≥ 1/7.
     - Proof idea: average over all labellings of the unseen points.
     - Wolpert's "averaged over all target functions" version.
     - What NFL does and doesn't imply.
  9. *Inductive biases in practice*: linearity, smoothness/locality, translation equivariance (CNNs), permutation invariance (sets/GNNs), and priors in Bayesian models.
  10. *iid and stationarity assumptions*, and what breaks without them (pointer to fund.distribution-shift).
- **Question ideas**
  - What does NFL imply? Correct: "no learner is best on all distributions without assumptions". Distractors: "CV is useless", "all algorithms tie on ImageNet", "deep nets can't beat random".
  - Bayes-optimal prediction under absolute loss (median). Under the asymmetric pinball loss τ=0.9 (the 0.9-quantile).
  - Classify: contrastive pretraining, k-means, next-token prediction, label propagation, RLHF.
  - (calc) Bayes error of two equal-prior 1-D Gaussians N(±1, 1) = Φ(−1) ≈ 0.159.
  - Which term in the error decomposition shrinks with more data for a fixed H? (estimation error)
- **Figures**
  - `fund.learning-setups/setups-map`: 2×2 map (labels used? / models p(x) or p(y|x)?) with example methods placed.
  - `fund.learning-setups/nfl-intuition`: two hypotheses that agree on the training points and disagree everywhere else.
  - `fund.learning-setups/error-decomposition`: nested sets (all functions ⊃ H ∋ ERM solution) with approximation and estimation error arrows.
  - `fund.learning-setups/bayes-error`: two overlapping class-conditional densities with the Bayes-error region shaded. **[fig-Q]**
- **Pitfalls & source disagreements**
  - The NFL versions differ: Wolpert (uniform average over targets), UML (adversarial distribution per learner), DLB/PML1 (informal).
  - The supervised/unsupervised boundary is blurry, and DLB says so. Self-supervised learning is sometimes classed as unsupervised.

---

#### 120 · `fund.pac-vc`: PAC Learning & VC Dimension
- **Level** advanced · **W3** · **prereqs** [fund.learning-setups] · **covers** generalization theory behind NFL/overfitting.
- **Sources**
  - UML (SSBD) ch.3 PAC & agnostic PAC (p.43–51), ch.4 uniform convergence & finite classes (p.54–58), ch.6 VC dimension incl. §6.3 examples and the fundamental theorem (p.67–82). `uml.txt`
  - CS229 notes ch.8 "Sample complexity bounds": finite H and infinite H (pdf p.129–137). `cs229.txt`
  - ESL §7.9 VC dimension (pdf p.256), §7.4–7.7 optimism, AIC/BIC (pdf p.247–254). `esl.txt`
  - Goodfellow DL §5.2 (capacity, VC dimension mention). Zhang et al. 2017 (`papers/zhang2017_rethinking_generalization.txt`) for the modern caveat.
- **Subtopic map**
  1. *PAC definition*: (ε, δ), realizable case, sample complexity $m_H(\varepsilon,\delta)$.
  2. *Finite H, realizable*: derive $m\ge\frac{\ln|H|+\ln(1/\delta)}\varepsilon$ via "a bad hypothesis survives all m samples" plus a union bound.
  3. *Agnostic PAC and uniform convergence*: Hoeffding's inequality (statement), giving $m=O(\frac{\ln|H|+\ln(1/\delta)}{\varepsilon^2})$. Note the 1/ε vs 1/ε² gap.
  4. *Infinite classes*: why |H| fails, shattering, VC dimension.
     - Examples: thresholds (1), intervals (2), linear separators in ℝ^d (d+1), axis-aligned rectangles (4).
     - Show that 3 points can be shattered by lines and 4 can't (the XOR configuration).
  5. *Fundamental theorem*: finite VC ⇔ PAC learnable, with $m=\Theta(\frac{d+\ln(1/\delta)}{\varepsilon^2})$ in the agnostic case (statement). Generalization bound form: train error + $\sqrt{d/m}$-ish.
  6. *Bias–complexity in theory*: structural risk minimization (named), AIC/BIC as complexity penalties (pointer to fund.cross-validation).
  7. *Why classical bounds are vacuous for deep nets*: Zhang et al.'s random-label experiments. Pointers to norm/margin bounds, PAC-Bayes and compression (named only).
- **Question ideas**
  - (calc) $|H|=2^{20}$, ε=0.01, δ=0.05 → m ≥ ≈1,686 (realizable).
  - VC dimension of linear classifiers with bias in ℝ² (3). Why 4 fails (XOR).
  - Predict: how the agnostic sample complexity scales when ε halves (×4).
  - Which is false: "a hypothesis class with infinitely many functions cannot be PAC learnable".
  - Interpretation: what does Zhang et al. imply for VC-based explanations of deep-net generalization?
- **Figures**
  - `fund.pac-vc/shattering`: 3 points in all 8 labellings with separating lines, plus the 4-point XOR that can't be separated. **[fig-Q]**
  - `fund.pac-vc/bound-vs-m`: bound value vs m for several VC dimensions.
- **Pitfalls & source disagreements**
  - Realizable vs agnostic rates get conflated. Sources differ in constants and in log factors.

---

#### 130 · `fund.bias-variance`: Bias–Variance Decomposition & Overfitting *(keeps id)*
- **Level** core · **W1** · **prereqs** [fund.estimation, fund.learning-setups] · **covers** bias–variance tradeoff, overfitting/underfitting.
- **Sources**
  - ESL §7.2–7.3 incl. the kNN and linear-model decompositions and §7.3.1 (0-1 loss interaction) (pdf p.238–247), §2.9 (pdf p.56). `esl.txt`
  - Bishop PRML §3.2 (pdf p.167–172) with the 100-dataset ridge experiment (Fig. 3.5–3.6), §1.1 polynomial curve fitting (pdf p.24–32). `bishop.txt`
  - Goodfellow DL §5.2 capacity, over- and underfitting, §5.4.4. `dlb_ch05_ml.txt`
  - CS229 notes ch.8 bias–variance trade-off & mathematical decomposition (pdf p.118–124). `cs229.txt`
  - d2l "Generalization" (`d2l/d2l_generalization.txt`).
- **Subtopic map**
  1. *Setup*: $y=f(x)+\varepsilon$ with $E\varepsilon=0$ and $\mathrm{Var}\,\varepsilon=\sigma^2$. Training set D random, so $\hat f_D$ is random. What the expectation is over (D and ε, at a fixed x).
  2. *Full derivation*: $E[(y-\hat f(x))^2]=\sigma^2+(f(x)-E_D\hat f(x))^2+\mathrm{Var}_D\hat f(x)$. Add and subtract $E_D\hat f$, and show each cross term vanishes, with the reason for each (independence of ε and D; definition of the mean).
  3. *Interpretation*: noise is irreducible; bias is the systematic error of the average model; variance is sensitivity to the particular training sample. Averaged over x: integrated bias² and variance.
  4. *Worked example 1, kNN regression*: $\sigma^2+[f(x_0)-\frac1k\sum f(x_{(l)})]^2+\sigma^2/k$ (ESL §7.3). k controls the trade-off.
  5. *Worked example 2, polynomial degree / ridge λ*: simulate many datasets, plot the fitted curves, and estimate bias² and variance (Bishop's experiment).
  6. *Worked example 3, linear regression*: the variance term averages to ≈ σ²p/N over training inputs. More features mean more variance, more data less.
  7. *Overfitting and underfitting* as high variance and high bias. Capacity (DLB §5.2). Learning-curve diagnosis: train/val error vs N and vs capacity, and what each remedy does (more data, regularization, bigger model, better features). Tuning-plan insertion: problematic overfitting means validation loss *rising*; a too-small learning rate can act as an accidental regularizer, so best-trial selection can favour "hobbled" optimizer settings.
  8. *0-1 loss*: no additive decomposition. ESL §7.3.1: estimation errors that leave the estimate on the right side of ½ don't hurt, so squared bias can rise while the error rate stays flat (kNN example). ESL Ex. 7.2: bias and variance interact multiplicatively, and when the mean estimate is on the wrong side of ½, more variance can even help. Name the competing decompositions.
  9. *Averaging and variance*: the ensemble formula $\rho\sigma^2+(1-\rho)\sigma^2/B$ (pointer to fund.ensembles). Bagging reduces variance and boosting mainly reduces bias.
  10. *Where the classic U-curve breaks*: preview of double descent (next lesson).
- **Question ideas**
  - Predict: increasing k in kNN regression (bias ↑, variance ↓). Adding features to OLS with fixed N (variance ↑).
  - **[fig-Q]** Learning curves: which panel is high variance?
  - (calc) kNN: σ²=1, k=4 → variance term 0.25.
  - Derivation step: which cross term vanishes because ε is independent of D?
  - Which is false: "regularization reduces both bias and variance".
  - Open: "Explain bias–variance to an interviewer using kNN, then say where it breaks."
- **Figures**
  - `fund.bias-variance/u-curve`: bias², variance, noise and total vs polynomial degree from simulation.
  - `fund.bias-variance/fits-grid`: 20 resampled fits at low, medium and high degree (spread = variance; offset = bias).
  - `fund.bias-variance/learning-curves`: train and val error vs N for high-bias vs high-variance models. **[fig-Q]**
  - `fund.bias-variance/knn-tradeoff`: bias² and variance vs k for kNN regression on a sine.
- **Pitfalls & source disagreements**
  - DLB §5.4 decomposes the *parameter-estimator* MSE, while ESL, Bishop and CS229 decompose *prediction* error. Keep the two separate.
  - There are multiple 0-1 decompositions (Domingos, Kohavi–Wolpert, James). None is standard.

---

#### 150 · `fund.regularization`: Parameter Norm Penalties & Other Regularizers
- **Level** core · **W1** · **prereqs** [fund.bias-variance, fund.mle-map] · **covers** regularisation methods.
- **Sources**
  - Goodfellow DL §7.1.1 L2 (eigenbasis analysis, linear-regression case), §7.1.2 L1 (soft-thresholding), §7.2 norm penalties as constraints, §7.3 under-constrained problems, §7.4 augmentation, §7.5 noise robustness & §7.5.1 label smoothing, §7.7 multitask, §7.9 parameter sharing, §7.10 sparse representations. `dlb_ch07_regularization.txt` · https://www.deeplearningbook.org/contents/regularization.html
  - ESL §3.4.1–3.4.3 (pdf p.80–92), Table 3.4 under orthonormal design. `esl.txt`
  - Murphy PML1 §4.5.3–4.5.4 weight decay & picking λ (pdf p.153–155), §13.5.2–13.5.3 weight decay & sparse DNNs (pdf p.485). `pml1.txt`
  - Bishop PRML §3.1.4 regularized LS & the $\|w\|_q$ family (pdf p.164), §5.5.5 training with transformed data ≈ Tikhonov (pdf p.285). `bishop.txt`
  - d2l weight decay (`d2l/d2l_weight_decay.txt`). Loshchilov & Hutter 2019 (`papers/loshchilov2017_adamw.txt`) for L2 ≠ WD.
- **Subtopic map**
  1. *Why regularize*: control variance. Any preference for simpler solutions beyond what the data dictate. Explicit (penalties) vs implicit (optimizer, architecture).
  2. *L2/weight decay mechanics*: gradient step = shrink-then-update, $w\leftarrow(1-\alpha\lambda)w-\alpha\nabla L$.
  3. *L2 analysis in the Hessian eigenbasis* (DLB §7.1.1): quadratic approximation around $w^*$ gives $\tilde w_i=\frac{\lambda_i}{\lambda_i+\alpha}w^*_i$. Directions the loss barely cares about shrink most. Same as the ridge SVD picture.
  4. *L1* (this lesson owns the derivation; fund.ridge-lasso reuses it): subgradient at 0. With a diagonal Hessian, the soft-threshold solution $w_i=\mathrm{sign}(w^*_i)\max(|w^*_i|-\alpha/H_{ii},0)$ (derive). This gives sparsity and feature selection.
  5. *Constrained view and geometry* (owned here; fund.ridge-lasso points back): the Lagrangian equivalence, $\min L$ s.t. $\|w\|\le t$. The L1 diamond's corners vs the L2 disk. The $\|w\|_q$ family. Elastic net (grouping of correlated features).
  6. *Probabilistic view*: L2 = Gaussian prior, L1 = Laplace prior (pointer to fund.mle-map).
  7. *Under-constrained problems*: the singular $X^\top X$ becomes invertible with $+\lambda I$. Logistic regression on separable data needs a penalty to have a finite solution.
  8. *Data augmentation* as encoding known invariances. *Noise injection*: input noise ≈ a Tikhonov penalty on gradients (Bishop §5.5.5). Weight noise.
  9. *Label smoothing* (owned here): target $(1-\varepsilon)y+\varepsilon/K$. The loss is $(1-\varepsilon)\,\mathrm{CE}(y,p)+\varepsilon\,\mathrm{CE}(u,p)$ with u uniform (derive). The optimal logits have a finite gap $\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$ (4.51 for K=10, ε=0.1) instead of growing without bound. Tuning-plan insertion: Shallue et al. §4.6 found it helped (up to ~1 point on ImageNet) mainly at large batch sizes. Calibration and distillation effects are in fund.loss-functions.
  10. *Parameter sharing/tying* (CNNs, RNNs). Multitask learning as a regularizer. Max-norm constraints. Sparse activations.
  11. *Weight decay vs L2 under adaptive optimizers*: for SGD with learning rate α, decoupled decay $w\leftarrow(1-\lambda)w-\alpha\nabla L$ equals an L2 penalty $\frac{\lambda'}2\|w\|^2$ with $\lambda'=\lambda/\alpha$ (Loshchilov & Hutter Prop. 1); under Adam they differ (derived in fund.adam, one line here). Weight decay on norm-layer parameters and biases is usually disabled (why: scale invariance; see fund.batchnorm).
  12. *Choosing λ*: validation or CV, log-scale search. Why λ for a mean loss depends on N.
- **Question ideas**
  - (calc) $w^*=[3,0.5]$, H = I, α=1: L1 → [2, 0], L2 → [1.5, 0.25].
  - (calc) Eigenvalues of H are 10 and 0.1, α=1: L2 shrink factors 0.909 and 0.091.
  - Under Adam, which weights does an L2 penalty regularize least? (large-gradient-history coordinates)
  - Which is false: "L1 regularization yields sparse posterior means under a Laplace prior".
  - Predict: label smoothing with ε=0.1 for K=10. The target probability of the true class is 0.91, and the optimal logit gap is ln 91 ≈ 4.51.
  - Open: "Why does L1 give sparsity? Give three views (geometry, subgradient, prior)."
- **Figures**
  - `fund.regularization/l1-l2-geometry`: loss contours meeting the L1 diamond at a corner and the L2 disk off-axis.
  - `fund.regularization/shrinkage-curves`: output vs $w^*$ for L1 (soft), L2 (proportional) and best-subset (hard). **[fig-Q]**: "which is L1?"
  - `fund.regularization/eigen-shrink`: bars of $\lambda_i/(\lambda_i+\alpha)$ for a Hessian spectrum.
- **Pitfalls & source disagreements**
  - "Weight decay" and "L2" are used interchangeably in papers. That's wrong for Adam.
  - DLB writes α for the penalty strength, while ESL and PML1 use λ.

---

#### 160 · `fund.dropout-early-stopping`: Dropout & Early Stopping
- **Level** core · **W2** · **prereqs** [fund.regularization] · **covers** early stopping, regularisation methods.
- **Sources**
  - Goodfellow DL §7.8 early stopping (algorithm, L2 equivalence derivation), §7.11 bagging, §7.12 dropout (ensemble view, weight-scaling rule). `dlb_ch07_regularization.txt`
  - Srivastava et al. 2014 (`papers/srivastava2014_dropout.txt`): §§ model, training, the linear-regression marginalization (§9), and dropout rates in practice.
  - Murphy PML1 §4.5.6 early stopping (pdf p.156), §13.5.1 & §13.5.4 (pdf p.485–487). `pml1.txt`
  - Bishop PRML §5.5.2 early stopping with the effective-parameters picture (pdf p.279). `bishop.txt`
  - d2l dropout (`d2l/d2l_dropout.txt`).
- **Subtopic map**
  1. *Early stopping as an algorithm*: monitor validation loss, patience, keep the best checkpoint. The two retraining strategies (retrain on all data for the same number of steps, or continue until training loss matches). Cost: almost free. Tuning-plan insertion: with a fixed step budget, keep the best few checkpoints and choose retrospectively; the playbook calls prospective early stopping usually unnecessary (no patience parameter).
  2. *Why it regularizes*: the effective capacity grows with training time. For a quadratic loss and GD from 0, early stopping ≈ L2 with α ≈ 1/(τε) (derive DLB §7.8 via the eigenbasis: $(1-\varepsilon\lambda_i)^\tau$ vs $\alpha/(\lambda_i+\alpha)$).
  3. *Caveats*: the equivalence is approximate. Validation noise. Double-descent-in-epochs interactions.
  4. *Dropout mechanics*: Bernoulli masks per unit per example. **Inverted dropout** scales by 1/(1−p) at train time so test time is unchanged. Where to apply it (hidden units, sometimes inputs with lower p, attention/residual dropout in transformers).
  5. *Ensemble view*: training a weighted ensemble of $2^n$ subnetworks with shared weights. The weight-scaling inference rule. Exact for a single softmax layer, approximate otherwise.
  6. *Dropout = adaptive L2 in linear regression*: marginalizing masks gives MSE plus a penalty $\frac p{1-p}\sum_j\tilde w_j^2\|X_{:,j}\|^2$ with p the *drop* probability and $\tilde w=(1-p)w$ (derive). Srivastava §9.1 writes p for the *retain* probability, giving $\frac{1-p}p$; state the convention.
  7. *Co-adaptation argument* and noise injection on hidden units. Interaction with BN (variance shift). Reduced use in large LLM pretraining.
  8. *MC dropout*: dropout kept at test time to estimate predictive uncertainty (named; pointer to ensembles).
  9. *Practicalities*: `model.train()` vs `model.eval()`, common p values (0.1–0.5), and why not to double-scale. Dropout and augmentation increase gradient variance, so they need more training steps (Tuning-plan insertion).
- **Question ideas**
  - (calc) Inverted dropout with $p_{drop}=0.5$: a kept activation of 2.0 → 4.0 at train time, 2.0 at test time.
  - Spot the bug: dropout left on at evaluation, or test-time scaling applied *as well as* inverted dropout.
  - (calc) Early stopping after τ=100 steps at lr ε=0.01 ≈ L2 with α ≈ 1.
  - Which is false: "dropout's weight-scaling rule is exact for deep ReLU networks".
  - Predict: validation loss curves with and without dropout (a smaller gap with dropout).
  - Open: "Explain why early stopping is a regularizer."
- **Figures**
  - `fund.dropout-early-stopping/train-val-curves`: train and val loss vs epoch with the best checkpoint marked. **[fig-Q]**: "where should training stop?"
  - `fund.dropout-early-stopping/es-vs-l2-path`: the GD trajectory on a 2-D quadratic overlaid with the ridge path as α varies.
  - `fund.dropout-early-stopping/dropout-subnets`: a small MLP with three sampled masks.
- **Pitfalls & source disagreements**
  - p means *drop* probability in PyTorch `nn.Dropout(p)` but *retain* probability in Srivastava et al. and TF1 `keep_prob`. DLB uses an inclusion probability.

---

#### 170 · `fund.cross-validation`: Cross-Validation & Hyperparameter Search
- **Level** core · **W2** · **prereqs** [fund.bias-variance] · **covers** cross-validation. Demoted to W2 in review: the reader already knows K-fold CV, and W1 makes room for fund.gradient-descent and fund.information-theory.
- **Sources**
  - ESL §7.10 CV: §7.10.1 K-fold incl. the LOOCV/GCV formula (pdf p.260–264), §7.10.3 does CV work (pdf p.266); §7.11 bootstrap & .632 (pdf p.268–273); §7.4–7.7 optimism, $C_p$/AIC, effective parameters, BIC (pdf p.247–254). `esl.txt`
  - UML (SSBD) ch.11: §11.2.1 hold-out bound (p.146), §11.2.2 validation for model selection, §11.2.4 k-fold (p.149), §11.2.5 train-val-test (p.150). `uml.txt`
  - Goodfellow DL §5.3, §5.3.1; §11.4 hyperparameters (manual, grid §11.4.3, random §11.4.4, model-based §11.4.5). `dlb_ch05_ml.txt`, `dlb_ch11_guidelines.txt`
  - Bengio & Grandvalet 2004 (`papers/bengio2004_cv_variance.txt`). Bergstra & Bengio 2012 random search (`tp_bergstra2012_random_search.txt`, in the cache root).
  - Murphy PML1 §4.5.5 (pdf p.155), §5.2.4–5.2.5 CV vs marginal likelihood & information criteria (pdf p.214–217). `pml1.txt`
- **Subtopic map**
  1. *Why hold out data*: training error is optimistically biased (ESL "optimism"). Roles of train / validation / test, and why the test set is touched once.
  2. *Hold-out bound*: $|L_V-L_D|\le\sqrt{\ln(2/\delta)/(2m_v)}$. Selecting among |H| candidates gives $\ln(2|H|/\delta)$ (UML §11.2). This shows how validation-set size limits how many configurations you can compare.
  3. *K-fold CV procedure*: the estimate is the mean over folds. Choice of K: each fold trains on (K−1)/K of the data, so small K is pessimistically biased, and the variance depends on fold correlation and learner stability. K = 5 or 10 heuristics.
  4. *LOOCV*: nearly unbiased. For linear smoothers there is a closed form, $\frac1N\sum\big(\frac{y_i-\hat f(x_i)}{1-S_{ii}}\big)^2$ (ESL eq. 7.51; the per-residual identity is ESL Ex. 7.3, eq. 7.64; derive via Sherman–Morrison). GCV replaces $S_{ii}$ by tr(S)/N (ESL eq. 7.52).
  5. *Uncertainty of CV estimates*: fold-wise SE is too small because the folds share training data. No unbiased variance estimator exists (Bengio & Grandvalet). The 1-SE rule for choosing simpler models.
  6. *Stratified, repeated and grouped K-fold*. Time-series forward-chaining (rolling-origin) CV.
  7. *Bootstrap error estimates*: the leave-one-out bootstrap, and why .632 and .632+ correct its pessimism.
  8. *Information criteria vs CV*: AIC (−2 log L + 2d) and BIC (−2 log L + d log N). Consistency (BIC) vs predictive efficiency (AIC). Effective df = tr(S).
  9. *Hyperparameter search*: grid vs random. Random wins when few hyperparameters matter (Bergstra & Bengio). Log-scale sampling. Successive halving and Bayesian optimization (named). The budget per trial. Keep this to one card (Bergstra & Bengio's argument) and → link sys.tuning-search for quasi-random search and reading studies (Tuning-plan insertion).
  10. *What CV estimates*: expected test error over training sets, more than the conditional error of *your* model (ESL §7.12, pdf p.273).
- **Question ideas**
  - (calc) Hoeffding validation size for |H|=100, ε=0.02, δ=0.05: ≈ 10,400.
  - (calc) Leverage $S_{ii}=0.5$, residual 1 → LOO residual 2.
  - Predict: CV estimate bias as K goes 2 → N (less pessimistic).
  - Which CV for time series? (forward chaining) For patients with multiple scans? (group K-fold)
  - Why is random search better than grid search with 2 important out of 10 hyperparameters?
  - Which is false: "the standard error across K folds is an unbiased estimate of the CV estimator's SE".
- **Figures**
  - `fund.cross-validation/kfold-diagram`: K=5 folds with train/val blocks.
  - `fund.cross-validation/grid-vs-random`: 9 grid points vs 9 random points projected onto the one important axis (Bergstra's picture). **[fig-Q]**
  - `fund.cross-validation/one-se-rule`: CV error vs complexity with error bars and the 1-SE choice.
  - `fund.cross-validation/timeseries-cv`: expanding-window splits along time.
- **Pitfalls & source disagreements**
  - ESL claims LOOCV has high variance. That's learner-dependent (Bengio & Grandvalet).
  - Validation vs development set naming. "Test" used to mean validation in many papers.

---

#### 180 · `fund.model-selection`: Leakage, Nested CV & Selection Bias
- **Level** intermediate · **W2** · **prereqs** [fund.cross-validation] · **covers** cross-validation pitfalls.
- **Sources**
  - ESL §7.10.2 the wrong and right way to do CV (pdf p.264–266). `esl.txt`
  - Cawley & Talbot 2010 (`papers/cawley2010_model_selection_overfit.txt`): over-fitting in model selection, selection bias, nested CV.
  - UML (SSBD) §11.3 what to do if learning fails (p.151). `uml.txt`
  - Goodfellow DL §11.3 when to gather more data, §11.5 debugging strategies. `dlb_ch11_guidelines.txt`
  - Kaggle-style leakage examples aren't cached. Use ESL's simulation and Cawley & Talbot as the citable support.
- **Subtopic map**
  1. *The wrong vs right way*: any supervised preprocessing (feature selection, target encoding, oversampling) must happen inside each training fold. ESL's simulation (N=50, 5,000 noise features) gives near-zero CV error when done wrong. Re-run the numbers.
  2. *Leakage taxonomy* with an example of each:
     - Preprocessing fit on all data (scaler, PCA, imputer).
     - Duplicates and near-duplicates across splits.
     - Temporal leakage (future info, random splits of time series).
     - Group leakage (same user/patient in train and test).
     - Target proxies.
     - Benchmark contamination in pretraining data.
     - Symptom to recognize: periodic structure in validation metrics points to train/val overlap or a shuffling bug; evaluate at fixed step intervals so the period is visible (Tuning-plan insertion).
  3. *Selection bias from tuning*: the max of k noisy validation scores is optimistic even when every model is identical (simulate). It grows with the number of configurations and with the validation noise.
  4. *Nested CV*: an inner loop for selection and an outer loop for assessment. It estimates the performance of the *whole procedure*. Cost.
  5. *Test-set reuse* over a research cycle (adaptive overfitting to benchmarks). Holdout hygiene: touch the test set only after exploration, and fold validation data into training only for one-off workloads (Tuning-plan insertion).
  6. *Distribution mismatch between splits*: why splits should mirror deployment (time, geography). Pointer to fund.distribution-shift.
  7. *Diagnosing failure* (UML §11.3, DLB §11.3): is the problem approximation, estimation or optimization error? What to try for each.
- **Question ideas**
  - Spot the flaw: standardize (or select the top-100 correlated features) on the full data, then run CV.
  - (calc) Expected max of 100 iid N(0, 1) noise scores ≈ 2.5σ: the optimism from tuning.
  - Which split for click logs with a strong weekly trend? (time-based)
  - After CV-tuning 500 configurations, is the best CV score biased? Which way? Fix?
  - Which is false: "nested CV gives the performance of the single model you finally ship".
- **Figures**
  - `fund.model-selection/wrong-right-cv`: CV estimate with selection outside vs inside the folds vs the true error. **[fig-Q]**
  - `fund.model-selection/selection-bias`: expected best-of-k validation score vs k for identical models.
  - `fund.model-selection/nested-cv`: outer and inner loop diagram.
- **Pitfalls & source disagreements**
  - Some papers call nested CV "double CV". "Validation" and "test" are swapped in some literature.

---

#### 190 · `fund.classification-metrics`: Precision, Recall, F1 & ROC/PR Curves *(keeps id)*
- **Level** core · **W1** · **prereqs** [fund.bayes-theorem] · **covers** precision/recall/F1/AUC-ROC.
- **Sources**
  - Murphy PML1 §5.1.2 classification decisions & costs, §5.1.3 ROC curves, §5.1.4 PR curves (pdf p.199–206). `pml1.txt`
  - Fawcett 2006 (`papers/fawcett2006_roc_intro.txt`): ROC space, AUC = Wilcoxon/Mann–Whitney, prior invariance, convex hull, averaging.
  - Davis & Goadrich 2006 (`papers/davis2006_pr_roc.txt`): ROC dominance ⇔ PR dominance, non-linear PR interpolation, achievable PR curve.
  - Flach & Kull 2015 (`papers/flach2015_prg.txt`): why AUC-PR and F-score averaging are problematic, and PR-Gain.
  - Goodfellow DL §11.1 performance metrics (precision/recall, coverage). `dlb_ch11_guidelines.txt`
- **Subtopic map**
  1. *Confusion matrix*: TP/FP/FN/TN. Precision, recall (TPR, sensitivity), specificity, FPR and accuracy, each with a one-line meaning. Which ones depend on prevalence.
  2. *Prevalence and precision*: derive precision = $\frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}$ via Bayes. Worked rare-positive example (calc: π=1%, TPR 0.9, FPR 0.05 gives precision ≈ 15%). Why accuracy misleads under imbalance. Report counts for rare classes: "+0.05 sensitivity" on a class with 20 positives is one more example (Tuning-plan insertion).
  3. *F1 and Fβ*: F1 is the harmonic mean (why harmonic: it punishes imbalance between P and R), $F_1=\frac{2TP}{2TP+FP+FN}$, and it ignores TN. Fβ weights recall β times as much. F1 isn't symmetric under swapping the positive class.
  4. *Thresholds*: scores → decisions. The cost-sensitive Bayes threshold $t^*=\frac{c_{FP}}{c_{FP}+c_{FN}}$ on calibrated probabilities (derive). The F1-optimal threshold isn't 0.5.
  5. *ROC curve*: TPR vs FPR as the threshold sweeps. Diagonal = random. Concave hull. Prior invariance (both axes condition on the true class).
  6. *AUC = P(score of random positive > score of random negative)*, with ties counted ½. Derive from the Mann–Whitney statistic and compute from ranks on a small example. Invariant to monotone transforms of the scores. AUC < 0.5 means flip the scores.
  7. *PR curve and average precision*: the baseline is the prevalence. Non-monotone. Linear interpolation in PR space is wrong (Davis & Goadrich). AP (step-wise) vs trapezoidal AUC-PR.
  8. *ROC vs PR under imbalance*: ROC can look good while precision is poor, because FPR stays small even with many FPs. Dominance equivalence theorem.
  9. *Multi-class*: macro vs micro vs weighted averaging. Micro-F1 = accuracy in single-label multiclass. One-vs-rest ROC.
  10. *Ranking and retrieval metrics* in one card: precision@k, recall@k, MAP, NDCG (named, with formulas for P@k and DCG).
- **Question ideas**
  - (calc) TP=40, FP=10, FN=20 → P = 0.8, R ≈ 0.667, F1 ≈ 0.727.
  - (calc) Rare positives: precision ≈ 0.15.
  - (calc) AUC from scores: positives {0.9, 0.6}, negatives {0.7, 0.2} → 3/4.
  - Which metric is unchanged if you duplicate every negative 10×? (ROC-AUC/TPR/FPR, not precision, PR-AUC or accuracy)
  - Micro-F1 in single-label multiclass equals? (accuracy)
  - Predict: apply a monotone transform (sigmoid → log-odds) to the scores. AUC? (unchanged)
- **Figures**
  - `fund.classification-metrics/roc-pr-pair`: the same classifier's ROC and PR curves at 50% and 1% prevalence. Notice that ROC is unchanged while PR collapses. **[fig-Q]**
  - `fund.classification-metrics/score-histograms`: positive and negative score distributions with a sweeping threshold, linked to a point on the ROC.
  - `fund.classification-metrics/pr-interpolation`: correct (non-linear) vs naive linear interpolation between two PR points.
  - `fund.classification-metrics/f1-contours`: iso-F1 contours in (P, R) space.
- **Pitfalls & source disagreements**
  - AP vs AUC-PR differ across libraries (sklearn `average_precision_score` is step-wise, not trapezoidal).
  - Averaging per-fold F1 ≠ pooled F1 (Flach & Kull discuss related issues).
  - "ROC is insensitive to class imbalance" holds for the curve, not for how useful the operating points are.

---

### Classical models

---

#### 210 · `fund.linear-regression`: Linear Regression: OLS & its Geometry
- **Level** core · **W1** · **prereqs** [fund.gaussian, fund.mle] · **covers** linear regression.
- **Sources**
  - ESL §3.2 least squares (pdf p.63–76): the hat matrix and geometry, sampling distribution of β̂, §3.2.2 Gauss–Markov (pdf p.70), §3.2.3 multiple regression via successive orthogonalization/QR (pdf p.71). `esl.txt`
  - Bishop PRML §3.1.1–3.1.3 (ML & LS, geometry, sequential/LMS; pdf p.160–164). `bishop.txt`
  - Murphy PML1 §11.2 (terminology, LS estimation, QR/SVD & other approaches, goodness of fit/R²; pdf p.401–411). `pml1.txt`
  - CS229 notes ch.1 (LMS, normal equations, matrix derivatives, probabilistic interpretation, locally weighted regression; pdf p.10–21). `cs229.txt`
  - Goodfellow DL §5.1.4 (linear regression example), §4.5 (linear least squares). d2l linear-regression (`d2l/d2l_linear_regression.txt`).
- **Subtopic map**
  1. *Model and loss*: $y=Xw+\varepsilon$ with a bias column. Why squared loss (Gaussian MLE, pointer to fund.mle; convex; closed form).
  2. *Normal equations*: derive $\nabla_w\|y-Xw\|^2=-2X^\top(y-Xw)=0$ with matrix calculus (inline refresher: $\nabla_w a^\top w=a$, $\nabla_w w^\top Aw=(A+A^\top)w$). Uniqueness iff $X^\top X$ is invertible (full column rank).
  3. *Geometry*: $\hat y=Hy$ with $H=X(X^\top X)^{-1}X^\top$ (symmetric, idempotent, projects onto col(X)). Residual ⟂ col(X). tr H = p (degrees of freedom). Leverage $h_{ii}$ and influential points.
  4. *Statistical properties* under $\varepsilon\sim N(0,\sigma^2I)$, each derived:
     - $E\hat w=w$.
     - $\mathrm{Cov}\hat w=\sigma^2(X^\top X)^{-1}$.
     - The unbiased $\hat\sigma^2=\mathrm{RSS}/(N-p)$ (why N−p).
     - The t-statistics of coefficients (named).
  5. *Gauss–Markov*: OLS is BLUE under zero-mean, homoscedastic, uncorrelated errors. Gaussianity is not needed. "Linear unbiased" is a restriction, and biased estimators (ridge) can have lower MSE.
  6. *Interpreting coefficients*:
     - Partial effects "holding the others fixed".
     - Successive orthogonalization: $\hat w_j$ is the regression of y on $x_j$ after residualizing the other features (ESL §3.2.3).
     - Collinearity inflates variance (condition number).
     - A duplicated feature makes the problem singular.
  7. *p > N*: infinitely many solutions. The pseudo-inverse gives the min-norm solution, and GD from 0 converges to it (pointer to fund.double-descent).
  8. *Computation*: never form an explicit inverse. Use QR (stable) or Cholesky of $X^\top X$ (faster, squares the condition number) or SVD (rank-deficient cases). Cost O(Np² + p³). GD/SGD (LMS) for large N.
  9. *Extensions in one card*: basis functions/polynomial features (still linear in w), weighted LS for heteroscedastic noise, robust losses (L1/Huber), locally weighted regression.
  10. *Goodness of fit*: R² = 1 − RSS/TSS. It increases monotonically with features; use adjusted R². Test R² can be negative.
- **Question ideas**
  - (calc) OLS slope and intercept through (0,1), (1,3), (2,5): slope 2, intercept 1.
  - Which is false: "tr(H) = N" (it's p), "H is idempotent", "residuals are orthogonal to the columns of X".
  - Duplicate a feature column: OLS? (singular, infinitely many solutions)
  - Rescale feature j by 10: OLS coefficient j → /10 and predictions are unchanged.
  - Which assumption does Gauss–Markov *not* need? (Gaussian errors)
  - (calc) N=100, p=5, RSS=190 → σ̂² = 2.0.
- **Figures**
  - `fund.linear-regression/projection`: y projected onto the column-space plane with the residual ⟂ (3-D sketch).
  - `fund.linear-regression/leverage`: a high-leverage point pulling the fit, with leverage shown as marker size. **[fig-Q]**
  - `fund.linear-regression/collinearity`: contour of RSS in $(w_1,w_2)$ for correlated features: a long thin valley.
- **Pitfalls & source disagreements**
  - β means coefficients in ESL but noise precision in Bishop. PML1 uses w.
  - Intercept handling (a column of ones vs centring) matters once penalties are added (next lesson).

---

#### 220 · `fund.ridge-lasso`: Ridge, Lasso & Bayesian Linear Regression
- **Level** core · **W2** · **prereqs** [fund.linear-regression, fund.mle-map] · **covers** linear regression, regularisation.
- **Sources**
  - ESL §3.4.1 ridge incl. the SVD form, df(λ) and the PCA connection (pdf p.80–87), §3.4.2 lasso (pdf p.87), §3.4.3 subset vs ridge vs lasso, Table 3.4 & the geometry figure (pdf p.88–92), §3.8.6 coordinate descent (pdf p.111). `esl.txt`
  - Murphy PML1 §11.3 ridge incl. §11.3.2 ridge & PCA and §11.3.3 choosing λ (pdf p.411–415), §11.4 lasso (pdf p.415–429), §11.7 Bayesian linear regression (pdf p.435–441). `pml1.txt`
  - Bishop PRML §3.1.4 (pdf p.164), §3.3 Bayesian LR: posterior, predictive, equivalent kernel (pdf p.172–181), §3.5 evidence approximation (pdf p.185). `bishop.txt`
  - Goodfellow DL §7.1.1–7.1.2. `dlb_ch07_regularization.txt`
- **Subtopic map**
  1. *Ridge objective and closed form* $(X^\top X+\lambda I)^{-1}X^\top y$. Always invertible for λ > 0. MAP interpretation. Don't penalize the intercept, and standardize features first (why: the penalty isn't scale invariant).
  2. *SVD view*: $\hat y=\sum_ju_j\frac{d_j^2}{d_j^2+\lambda}u_j^\top y$. Shrinks most along low-variance principal directions. Effective degrees of freedom $\mathrm{df}(\lambda)=\sum d_j^2/(d_j^2+\lambda)$.
  3. *Bias and variance of ridge* (derive): $E\hat w_\lambda=(X^\top X+\lambda I)^{-1}X^\top Xw$ and $\mathrm{Cov}=\sigma^2AX^\top XA$. Some λ > 0 always beats OLS in MSE (statement, with intuition from derivatives at λ=0).
  4. *Lasso*: no closed form. Under an orthonormal design each coordinate decouples into the 1-D problem already solved in fund.regularization, giving soft-thresholding (recap in two lines; don't re-derive). Comparison with ridge's proportional shrinkage and best-subset's hard threshold (ESL Table 3.4).
  5. *Lasso path and solvers*: piecewise-linear path (LARS named). Coordinate descent: derive the per-coordinate soft-threshold update from partial residuals. Lasso picks one of a group of correlated features arbitrarily, while the elastic net groups them. (The diamond-vs-disk geometry lives in fund.regularization; one-line pointer.)
  6. *Choosing λ*: CV, GCV, or the 1-SE rule (pointer to fund.cross-validation).
  7. *Bayesian linear regression*:
     - Prior $N(0,\alpha^{-1}I)$ gives posterior $N(m_N,S_N)$ with $S_N^{-1}=\alpha I+\beta\Phi^\top\Phi$ and $m_N=\beta S_N\Phi^\top y$ (derive by completing the square).
     - The posterior mean = ridge with λ = α/β.
  8. *Predictive distribution*: variance $\frac1\beta+\phi(x)^\top S_N\phi(x)$, i.e. aleatoric + epistemic. Uncertainty grows away from the data. Sequential updating.
  9. *Evidence / empirical Bayes for α, β* (named, one paragraph: maximize the marginal likelihood instead of CV).
- **Question ideas**
  - (calc) Singular values 10 and 0.1 with λ=1: shrink factors 0.990 and 0.0099.
  - (calc) Orthonormal design, OLS coefficient 3, λ=1: ridge (with the $\|y-Xw\|^2+\lambda\|w\|^2$ convention) → 1.5; lasso with $\frac12\|\cdot\|^2+\lambda\|w\|_1$ → 2. State the conventions.
  - Duplicate a feature: ridge splits the weight equally (why: symmetry plus strict convexity). Lasso picks an arbitrary split.
  - Predict: Bayesian LR predictive variance far from the training inputs (grows; it's dominated by the epistemic term).
  - Which is false: "as λ grows, every lasso coefficient shrinks monotonically toward zero" (false: when correlated features drop out, others can grow, so paths can be non-monotone). True distractors: ridge has a unique solution for any λ > 0 even when p > N; lasso can set coefficients exactly to zero; ridge shrinks low-variance principal directions most.
- **Figures**
  - `fund.ridge-lasso/coef-paths`: ridge and lasso coefficient paths vs log λ. **[fig-Q]**: "which panel is lasso?"
  - `fund.ridge-lasso/svd-shrinkage`: per-principal-direction factors $d_j^2/(d_j^2+\lambda)$ next to PCR's 0/1 truncation. Notice that ridge is a soft version of PCR. (Replaces a geometry figure that duplicated `fund.regularization/l1-l2-geometry`.)
  - `fund.ridge-lasso/bayes-predictive`: predictive mean ± 2σ widening away from the data, with posterior samples of functions.
- **Pitfalls & source disagreements**
  - Objective scaling conventions (½ vs no ½, sum vs mean) change λ's numeric value across ESL, PML1 and sklearn (sklearn's Lasso uses $\frac1{2N}\|\cdot\|^2+\alpha\|w\|_1$).
  - Bishop's α/β are precisions, not the regularization strength itself.

---

#### 225 · `fund.double-descent`: Double Descent & Generalization in Deep Nets
- **Level** intermediate · **W2** · **prereqs** [fund.bias-variance, fund.ridge-lasso] · **covers** overfitting (modern view).
- **Sources**
  - Belkin, Hsu, Ma & Mandal 2019 (`papers/belkin2019_double_descent.txt`): the interpolation threshold, random-feature and tree experiments.
  - Nakkiran et al. 2019 (`papers/nakkiran2019_deep_double_descent.txt`): model-wise, epoch-wise and sample-wise double descent; effective model complexity.
  - CS229 notes ch.8 "The double descent phenomenon" (pdf p.124–128), ch.9 "Implicit regularization effect" (pdf p.140). `cs229.txt`
  - Zhang et al. 2017 (`papers/zhang2017_rethinking_generalization.txt`); Neal et al. 2018 (`papers/neal2018_modern_bias_variance.txt`).
  - Murphy PML1 §13.5.6 implicit regularization of SGD, §13.5.7 over-parameterized models (pdf p.487–489). d2l "Generalization in deep learning" (`d2l/d2l_generalization_deep.txt`).
- **Subtopic map**
  1. *The puzzle*: networks with far more parameters than examples generalize. Zhang et al. show they can also fit random labels and random pixels. What this rules out (uniform capacity explanations).
  2. *Model-wise double descent*:
     - Test error vs parameter count shows the classical U, a peak at the interpolation threshold (p ≈ N), then a second descent.
     - Min-norm least squares with random features as the cleanest worked example. Why the peak occurs: near p = N the system is barely solvable, the smallest singular values → 0, and the solution norm explodes.
  3. *Min-norm interpolation and implicit bias*: GD from 0 on least squares converges to the min-norm solution (derive via the row space). Bigger p means more interpolants to pick from, and the min-norm one is smoother.
  4. *Epoch-wise and sample-wise double descent* (Nakkiran): more data can *hurt* near the threshold. Effective model complexity.
  5. *Role of regularization and label noise*: the peak shrinks with optimal ridge and grows with label noise (Nakkiran's figures).
  6. *Bias–variance revisited*: Neal et al. decompose variance into its sources and find variance decreasing with width.
  7. *Implicit regularization*: SGD noise and small batches, large learning rate, early stopping, architecture.
  8. *Benign overfitting* (named), and practical implications: scale models, and don't choose size by the classical U-curve alone.
- **Question ideas**
  - Predict: test error of min-norm linear regression as p crosses N (a spike).
  - Which is false: "increasing data size always lowers test error".
  - Why does the norm of the min-norm solution blow up at p = N? (smallest singular value ≈ 0)
  - What happens to the peak with optimal ridge? (largely disappears)
  - Interpretation of fitting random labels (memorization capacity, so uniform bounds are vacuous).
- **Figures**
  - `fund.double-descent/model-wise`: test and train error vs #random features for min-norm LS. Notice the peak at p = N. **[fig-Q]**
  - `fund.double-descent/norm-spike`: the solution norm ‖ŵ‖ vs p peaking at the threshold.
  - `fund.double-descent/ridge-flattens`: the same curve for several ridge λ.
- **Pitfalls & source disagreements**
  - "Double descent" is not universal: it depends on noise, regularization and the complexity measure. The optimal-ridge result (Nakkiran et al. 2020) isn't cached, so present it as a claim.
  - People conflate this with "grokking" (a different phenomenon).

---

#### 230 · `fund.logistic-regression`: Logistic & Softmax Regression
- **Level** core · **W1** · **prereqs** [fund.mle, fund.information-theory] · **covers** loss functions (CE), classification models.
- **Sources**
  - Murphy PML1 §10.2 binary LR: linear/nonlinear classifiers, §10.2.3 MLE with gradient and Hessian, §10.2.4 SGD, §10.2.5 perceptron, §10.2.6 IRLS, §10.2.7 MAP, §10.2.8 standardization (pdf p.369–380). §10.3 multinomial LR (pdf p.380–389). §2.4.2 sigmoid, §2.5.2 softmax, §2.5.4 log-sum-exp (pdf p.80–86). `pml1.txt`
  - Bishop PRML §4.3.2 logistic regression, §4.3.3 IRLS, §4.3.4 multiclass (pdf p.225–230), §4.2 probabilistic generative models giving the sigmoid (pdf p.216). `bishop.txt`
  - CS229 notes ch.2 (logistic regression, perceptron, multiclass, Newton; pdf p.22–31). `cs229.txt`
  - Murphy PML1 §9.2 Gaussian discriminant analysis: §9.2.1–9.2.2 quadratic and linear boundaries, §9.2.3 the LDA–logistic connection (pdf p.353–356). CS229 notes ch.4 GDA (pdf p.37–41). ESL §4.3 LDA (pdf p.125).
  - ESL §4.4 incl. §4.4.1 fitting and §4.4.5 LR vs LDA (pdf p.138–148). Goodfellow DL §6.2.2.2–6.2.2.3 sigmoid & softmax units. d2l softmax regression (`d2l/d2l_softmax_regression.txt`).
- **Subtopic map**
  1. *Why not linear regression for classes*: unbounded outputs and sensitivity to far-away correct points. Model the log-odds as linear: $\log\frac p{1-p}=w^\top x$, so $p=\sigma(w^\top x)$.
  2. *Sigmoid facts*: σ(−z) = 1 − σ(z), σ′ = σ(1−σ), and the logit inverse. Where the sigmoid comes from: class-conditional Gaussians with shared covariance give a linear log-odds (Bishop §4.2). Name the generative counterpart: Gaussian discriminant analysis. Shared covariance (LDA) gives a linear boundary of the same form as logistic regression; class-specific covariances (QDA) give a quadratic one. LDA is more data-efficient when the Gaussian assumption holds, and logistic regression is more robust when it doesn't (ESL §4.4.5).
  3. *NLL = binary cross-entropy*. Derive the gradient $X^\top(\sigma(Xw)-y)$: the "prediction minus label" form and why it's so clean (canonical link).
  4. *Hessian* $X^\top SX$ with $S=\mathrm{diag}(p_i(1-p_i))$. PSD, so the loss is convex. No closed form. Newton/IRLS (pointer to fund.newton).
  5. *Separable data*: no finite MLE (‖w‖ → ∞ along the max-margin direction). The fix is L2/MAP. GD's implicit bias toward max margin (Soudry et al. 2018, not cached; state as a claim).
  6. *Softmax regression*: $p_k=e^{z_k}/\sum e^{z_j}$. Over-parameterization (shift invariance). Derive the gradient $\partial L/\partial z=p-\mathrm{onehot}(y)$ via the softmax Jacobian $\mathrm{diag}(p)-pp^\top$.
  7. *Numerics*: log-sum-exp, $\log\sum e^{z}=m+\log\sum e^{z-m}$. Fused log-softmax + NLL. Why computing log(softmax) naively overflows or underflows.
  8. *Interpretation*: coefficients as log-odds ratios. Decision boundary $w^\top x=t$ is linear. Thresholds and costs (pointer to fund.classification-metrics).
  9. *Variants*: perceptron (sign instead of σ; converges iff separable), probit, multi-label (independent sigmoids vs softmax), standardization effects on optimization and on regularization.
- **Question ideas**
  - (calc) Logits (2, 1, 0), label 0: softmax ≈ (0.665, 0.245, 0.090), gradient ≈ (−0.335, 0.245, 0.090).
  - (calc) w·x = 0 → p = 0.5. Odds ratio of a coefficient 0.693 → doubles the odds.
  - Predict: unregularized LR on separable data with GD (‖w‖ grows without bound, the loss → 0).
  - Which is false: "the logistic NLL has a unique minimizer for any dataset".
  - Derivation: why does the Hessian's PSD-ness follow from $p(1-p)\ge0$?
  - Multi-label vs multi-class: which output layer for tags that can co-occur? (independent sigmoids)
  - Compare: LDA, QDA and logistic regression. Which gives a quadratic boundary, and which stays consistent when the class-conditionals aren't Gaussian? (QDA; logistic regression)
- **Figures**
  - `fund.logistic-regression/sigmoid-boundary`: 2-D data with probability contours of a fitted LR model.
  - `fund.logistic-regression/separable-divergence`: ‖w‖ and loss vs iteration on separable data.
  - `fund.logistic-regression/softmax-temperature`: class probabilities as one logit varies. **[fig-Q]**
- **Pitfalls & source disagreements**
  - Labels {0,1} (Bishop, PML1, CS229) vs {−1,+1} (ESL boosting/SVM, UML) give different-looking but equivalent loss formulas.
  - PyTorch `CrossEntropyLoss` takes logits. Passing softmax outputs is a silent bug.

---

#### 240 · `fund.loss-functions`: Loss Functions & Surrogate Losses
- **Level** core · **W2** · **prereqs** [fund.logistic-regression] · **covers** loss functions.
- **Sources**
  - Goodfellow DL §6.2.1 cost functions (learning conditional distributions with ML; conditional statistics), §6.2.2 output units: the saturation analysis for sigmoid+MSE vs CE. `dlb_ch06_mlp.txt`
  - ESL §10.6 loss functions & robustness: margin losses, population minimizers, the robustness figure (pdf p.365–369). `esl.txt`
  - Murphy PML1 §4.3.2 surrogate losses (pdf p.146), §5.1.5 regression losses (pdf p.206), §11.6 robust regression (pdf p.432). UML (SSBD) §12.3 surrogate loss functions (p.167). `pml1.txt`, `uml.txt`
  - Bishop PRML §1.5.5 loss functions for regression (pdf p.66). `bishop.txt`
  - Lin et al. 2017 focal loss (`papers/lin2017_focal_loss.txt`); Müller et al. 2019 label smoothing (`papers/muller2019_label_smoothing.txt`).
- **Subtopic map**
  1. *Loss as a modelling choice*: the NLL view (which noise model) vs the decision view (what error costs). Population minimizer = the statistic of p(y|x) being estimated.
  2. *Regression losses*, deriving each population minimizer:
     - MSE → mean.
     - MAE → median.
     - Huber (quadratic within δ, linear outside) → in between, robust.
     - Pinball/quantile → the τ-quantile.
     - Log-cosh (named).
  3. *Why not MSE for classification*: with a sigmoid output, the MSE gradient has a σ′(z) factor, which vanishes when the model is confidently wrong. CE's gradient (p − y) doesn't. MSE∘sigmoid is non-convex in w.
  4. *Margin losses* for y ∈ {±1}, as functions of m = yf:
     - 0-1 (non-convex, zero gradient).
     - Hinge max(0, 1−m).
     - Logistic log(1+e^{−m}).
     - Exponential e^{−m}.
     - Squared (1−m)².
     - Surrogates are convex upper bounds of 0-1 (for logistic, after a change of log base). Classification calibration (named; Bartlett et al.).
  5. *Population minimizers*: logistic → log-odds, exponential → ½ log-odds, hinge → sign(2p−1) (so no probability estimates), squared → 2p−1. Derive one (logistic) explicitly.
  6. *Robustness to outliers and label noise*: exponential is worst (weights grow exponentially), squared is non-monotone in m (it penalizes being "too right"), hinge and logistic grow linearly.
  7. *Imbalance and hard examples*: class-weighted CE, focal loss $-(1-p_t)^\gamma\log p_t$ (effect of γ; α-balancing; γ=0 is CE). Bias init for rare classes (pointer to fund.initialization). Compare the standard remedies interviewers ask about: resampling (under/over-sampling, SMOTE named) vs class weights vs moving the threshold. Training on rebalanced data shifts the predicted probabilities; correct them with the prior-shift formula $p(y|x)\propto\hat p(y|x)\,\pi_y/\pi'_y$ (pointer to fund.distribution-shift).
  8. *Label smoothing*: the definition and loss decomposition are in fund.regularization (one-line recap). Here, its effects: bounded logit gaps, better calibration, tighter penultimate-layer class clusters, and worse teachers for distillation (Müller).
  9. *Other common DL losses in one card*: KL/distillation loss with temperature (pointer to the LLM area), contrastive InfoNCE as CE over similarities, triplet/margin ranking loss, Dice/IoU losses for segmentation (named).
- **Question ideas**
  - Which loss's population minimizer is the conditional median? (MAE) The 0.9-quantile? (pinball τ=0.9)
  - Predict: sigmoid+MSE on a confidently wrong example has a tiny gradient. Why?
  - (calc) Focal loss with γ=2, $p_t=0.9$ → factor 0.01. With $p_t=0.1$ → 0.81.
  - (calc) Huber δ=1 on residuals {0.5, 3}: losses 0.125 and 2.5.
  - (calc) A model trained on data rebalanced from 1% to 50% positives outputs 0.8. Corrected probability under the true 1% prior: 0.016/(0.016+0.396) ≈ 0.039.
  - Which is false: "hinge loss gives calibrated class probabilities".
  - **[fig-Q]** Margin-loss plot: identify the exponential loss.
- **Figures**
  - `fund.loss-functions/margin-losses`: 0-1, hinge, logistic, exponential and squared vs m = yf. **[fig-Q]**
  - `fund.loss-functions/regression-losses`: MSE, MAE, Huber and pinball (τ=0.9) vs residual.
  - `fund.loss-functions/sigmoid-mse-vs-ce`: gradient magnitude vs logit for a positive example.
  - `fund.loss-functions/focal-vs-ce`: loss vs $p_t$ for γ ∈ {0, 1, 2, 5}.
- **Pitfalls & source disagreements**
  - ESL's "binomial deviance" = 2× NLL. The labels {0,1} vs {±1} forms.
  - Huber's δ convention (½r² inside) varies. PyTorch `HuberLoss` vs `SmoothL1Loss` differ by a factor of δ.

---

#### 245 · `fund.calibration`: Calibration & Proper Scoring Rules
- **Level** intermediate · **W2** · **prereqs** [fund.classification-metrics, fund.logistic-regression] · **covers** metrics context (probabilistic quality).
- **Sources**
  - Guo, Pleiss, Sun & Weinberger 2017 (`papers/guo2017_calibration.txt`): reliability diagrams, ECE/MCE, causes of miscalibration (depth, BN, weight decay, NLL overfitting), temperature/Platt/isotonic/histogram binning.
  - Murphy PML1 §5.1.6 probabilistic prediction & proper scoring rules (pdf p.207). `pml1.txt`. Murphy PML2 §14.2.1 proper scoring rules, §14.2.2 calibration (pdf p.612–616), §14.3 conformal prediction (pdf p.619). `pml2.txt`
  - Müller, Kornblith & Hinton 2019 label smoothing & calibration (`papers/muller2019_label_smoothing.txt`). Lakshminarayanan et al. 2017 deep ensembles (`papers/lakshminarayanan2017_deep_ensembles.txt`).
- **Subtopic map**
  1. *Definition*: perfect calibration means $P(Y=1\mid\hat p=p)=p$. Calibration ≠ accuracy (a constant base-rate predictor is calibrated but useless). Calibration vs refinement/sharpness.
  2. *Reliability diagrams* and **ECE** = $\sum_b\frac{|B_b|}N|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$. Binning sensitivity and bias. MCE.
  3. *Proper scoring rules*: a rule is strictly proper if the true distribution uniquely minimizes the expected score. Prove for log loss (via KL ≥ 0) and Brier. Hinge and accuracy are not proper. The Brier decomposition (reliability, resolution, uncertainty) is named.
  4. *Why modern nets are overconfident* (Guo): capacity, NLL overfitting after the accuracy saturates, BN, less weight decay.
  5. *Post-hoc recalibration*:
     - Temperature scaling: logits/T with one parameter fit on validation NLL. It doesn't change the argmax; for a binary model it leaves ROC-AUC unchanged (σ(z/T) is monotone in z), but one-vs-rest AUC of a multiclass softmax can shift slightly.
     - Platt scaling (logistic on scores).
     - Isotonic regression.
     - Histogram binning.
     - Vector/matrix scaling.
  6. *Train-time approaches*: label smoothing, focal loss, mixup, and ensembles (better calibrated). Calibration under distribution shift degrades.
  7. *Selective prediction / abstention* using confidence. Conformal prediction gives coverage guarantees (named, one card max).
  8. *LLM link*: token-probability calibration and RLHF's effect (named; pointer to the LLM area).
- **Question ideas**
  - (calc) ECE from a 3-bin table.
  - Temperature scaling with T > 1 on an overconfident binary classifier: what happens to accuracy, AUC and NLL? (unchanged, unchanged, typically better)
  - Which is not a strictly proper scoring rule? (hinge or accuracy)
  - Predict: a 99%-accurate classifier that always outputs 0.99. Is it calibrated? (yes, if its accuracy is exactly 99%)
  - Derivation: show that the log-score minimizer is the true probability.
- **Figures**
  - `fund.calibration/reliability-diagram`: before and after temperature scaling. **[fig-Q]**: "which model is overconfident?"
  - `fund.calibration/temperature-effect`: softmax probabilities of fixed logits for T = 0.5, 1, 2.
  - `fund.calibration/proper-score`: expected log and Brier scores vs the reported q for a true p=0.7 (both minimized at 0.7).
- **Pitfalls & source disagreements**
  - ECE depends on the number of bins and on equal-width vs equal-mass binning.
  - Top-label calibration vs full multiclass calibration definitions differ.

---

#### 250 · `fund.knn`: k-Nearest Neighbours
- **Level** core · **W2** · **prereqs** [fund.bias-variance] · **covers** kNN.
- **Sources**
  - ESL §2.3.2 nearest-neighbour methods and §2.3.3 LS vs NN (pdf p.33–37), §13.3 kNN classifiers incl. the Cover–Hart bound and examples (pdf p.482–490), §13.3.3 invariant metrics/tangent distance (pdf p.490), §13.4 adaptive NN (pdf p.494), §13.5 computation (pdf p.499). `esl.txt`
  - UML (SSBD) ch.19 §19.1–19.2.1 (kNN, 1-NN generalization bound; p.258–263), §19.3 efficient implementation. `uml.txt`
  - Bishop PRML §2.5.2 NN density estimation & the kNN classifier via Bayes (pdf p.144). `bishop.txt`
  - Murphy PML1 §16.1 incl. §16.1.3 speed/memory and §16.1.4 open-set recognition (pdf p.577–581), §16.2 learned metrics (pdf p.581). `pml1.txt`
- **Subtopic map**
  1. *Algorithm*: classification by majority vote, regression by averaging. Distance-weighted variants. Ties (odd k for binary). It is lazy and nonparametric: no training, all cost at query time.
  2. *k as complexity*: effective degrees of freedom ≈ N/k. 1-NN has zero training error (barring duplicate x with different labels). Large k → majority class. Choose k by CV.
  3. *Bias–variance for kNN regression* (recap from fund.bias-variance). Its boundary for 1-NN is a Voronoi tessellation.
  4. *Cover–Hart asymptotics*: derive the binary case. As N → ∞ the nearest neighbour's label is an independent draw from the same p(y|x), so the error → 2p(1−p) ≤ 2 min(p, 1−p), i.e. ≤ 2× Bayes. The multiclass statement.
  5. *kNN as density estimation* (Bishop): $p(x)\approx K/(NV)$. Class posteriors $K_k/K$. This connects to the Bayes classifier.
  6. *Distance metrics*: Euclidean, Manhattan, cosine. **Feature scaling is essential** (worked: metres vs mm). Mahalanobis distance. Learned metrics (LMNN/siamese, named). Invariant metrics (tangent distance).
  7. *Computation*: O(Nd) per query. KD-trees and ball trees (good in low d, degrade in high d). Approximate NN (LSH, graph-based HNSW, product quantization), named with their trade-off. Memory = the whole dataset.
  8. *Modern uses*: retrieval over learned embeddings, kNN-LM, few-shot via nearest prototype (pointer to fund.few-zero-shot), and as a sanity baseline.
- **Question ideas**
  - (calc) Bayes error 10% (binary): the asymptotic 1-NN error ≤ 0.18.
  - **[fig-Q]** Decision boundaries for k = 1, 15, 50: which is k=1?
  - Predict: an unscaled feature in mm next to others in metres (the mm feature dominates distances).
  - Which is false: "1-NN's training error estimates its test error".
  - (calc) Effective df for N=1000, k=20 ≈ 50.
- **Figures**
  - `fund.knn/decision-boundaries`: k = 1, 15, 50 on a 2-D toy set. **[fig-Q]**
  - `fund.knn/voronoi`: the 1-NN Voronoi cells of a few points.
  - `fund.knn/k-vs-error`: train and CV error vs k.
- **Pitfalls & source disagreements**
  - Cover–Hart is often quoted loosely as "2× Bayes". The binary limit is 2p(1−p). ESL presents the multiclass version.

---

#### 260 · `fund.curse-of-dimensionality`: The Curse of Dimensionality
- **Level** core · **W2** · **prereqs** [fund.knn] · **covers** curse of dimensionality.
- **Sources**
  - ESL §2.5 local methods in high dimensions: the edge-length and median-distance calculations and the extrapolation argument (pdf p.41–46); §12.3.4 SVMs and the curse (pdf p.450). `esl.txt`
  - Bishop PRML §1.4 (grid-cell counting, polynomial term growth, shell volume, Gaussian mass in high d; pdf p.53–58). `bishop.txt`
  - UML (SSBD) §19.2.2 sample complexity exponential in d (p.263). `uml.txt`
  - Goodfellow DL §5.11.1 curse, §5.11.2 local constancy & smoothness, §5.11.3 manifold learning. `dlb_ch05_ml.txt`
  - Murphy PML1 §16.1.2 (pdf p.578). `pml1.txt`
- **Subtopic map**
  1. *Volume explosion*: cells needed for a grid of resolution r in d dimensions = $r^d$ (Bishop's grid). A polynomial of degree M in d variables has $O(d^M)$ terms.
  2. *Neighbourhoods aren't local*: the edge of a sub-cube capturing fraction r of the volume is $e_d(r)=r^{1/d}$ (calc: r=0.01, d=10 → 0.63).
  3. *Sparsity*: the median distance from the origin to the nearest of N uniform points in the unit ball is $(1-2^{-1/N})^{1/d}$ (calc: N=500, d=10 → 0.52). Most points are near the boundary, so prediction becomes extrapolation (ESL).
  4. *Shell concentration*: the fraction of volume in the outer ε shell is $1-(1-\varepsilon)^d$ (calc: d=100, ε=0.01 → 0.63). Gaussian samples concentrate at radius √d (derive $E\|x\|^2=d$ and the relative spread → 0).
  5. *Distance concentration*: for iid coordinates, (max − min)/min of pairwise distances → 0, so nearest-neighbour contrast vanishes. Random vectors in high d are nearly orthogonal (cosine ~ $1/\sqrt d$).
  6. *Sample complexity*: to keep the same neighbourhood size you need $N\propto N_1^d$. UML's result: learning Lipschitz functions needs a number of samples exponential in d.
  7. *Why ML still works*: low intrinsic dimension (manifold hypothesis), smoothness and structural priors (convolution, attention), and learned representations (DLB §5.11.2–5.11.3).
  8. *Consequences*: kNN and kernel methods suffer. Use feature selection and dimensionality reduction (pointer to fund.pca). Overfitting risk with p ≫ N (pointer to fund.ridge-lasso).
  9. *"Blessings" of dimensionality* (concentration enables JL random projections; pointer to fund.nonlinear-dim-reduction).
- **Question ideas**
  - (calc) $e_{10}(0.1)$ = 0.794.
  - (calc) Expected squared norm of $x\sim N(0,I_{1000})$ = 1000, typical norm ≈ 31.6.
  - (calc) Grid with 10 bins per axis in d=20 → $10^{20}$ cells.
  - Which is false: "doubling N halves the curse of dimensionality".
  - Predict: kNN accuracy as noise dimensions are added to an informative 2-D problem.
- **Figures**
  - `fund.curse-of-dimensionality/edge-length`: $e_d(r)$ vs r for d = 1, 2, 3, 10.
  - `fund.curse-of-dimensionality/distance-concentration`: histograms of normalized pairwise distances for d = 2, 10, 100, 1000. **[fig-Q]**
  - `fund.curse-of-dimensionality/shell-volume`: outer-shell volume fraction vs d.
  - `fund.curse-of-dimensionality/gaussian-norms`: the distribution of ‖x‖ for Gaussians in d = 1, 10, 100.
- **Pitfalls & source disagreements**
  - The term covers distinct phenomena: Bellman's combinatorial blow-up, data sparsity, and distance concentration. Be explicit about which one you mean.

---

#### 270 · `fund.svm-margin-dual`: SVMs I: Max Margin, Duality & Hinge Loss
- **Level** intermediate · **W1** · **prereqs** [fund.loss-functions, fund.convexity] · **covers** SVMs.
- **Sources**
  - CS229 notes ch.6: margins intuition, functional vs geometric margins, the optimal margin classifier, Lagrange duality (pdf p.67), the dual form (pdf p.70), regularization & the non-separable case (pdf p.74) (pdf p.61–79 overall). `cs229.txt`
  - Bishop PRML §7.1 maximum margin incl. the KKT conditions, §7.1.1 overlapping classes (soft margin, ν-SVM), §7.1.2 relation to logistic regression (pdf p.346–358); Appendix E Lagrange multipliers (pdf p.727). `bishop.txt`
  - ESL §4.5.2 optimal separating hyperplanes (pdf p.151), §12.2 support vector classifier, §12.2.1 computing it (pdf p.436–442), §12.3.2 SVM as penalization (pdf p.445). `esl.txt`
  - UML (SSBD) ch.15: hard-SVM, soft-SVM & norm regularization, margin vs dimension (§15.2.2), optimality conditions, duality, SGD for soft-SVM (§15.5) (p.202–214). `uml.txt`
  - Burges 1998 (`papers/burges1998_svm_tutorial.txt`); Boyd CVX ch.5 duality & KKT (pdf p.229–270) for the optimization background.
- **Subtopic map**
  1. *Geometry*: signed distance from x to the hyperplane is $(w^\top x+b)/\|w\|$ (derive). Functional vs geometric margin. The scale invariance of (w, b) and the canonical choice $\min_iy_i(w^\top x_i+b)=1$, which gives margin 2/‖w‖.
  2. *Hard-margin primal*: $\min\frac12\|w\|^2$ s.t. $y_i(w^\top x_i+b)\ge1$. Why this is a convex QP. Why maximizing the margin is a sensible inductive bias (robustness; margin bounds don't depend on dimension, per UML §15.2.2).
  3. *Lagrangian duality crash course*: Lagrangian, dual function, weak duality, strong duality under Slater, and the KKT conditions. Keep it tied to this problem (Boyd ch.5 for background).
  4. *Deriving the dual*:
     - ∂/∂w gives $w=\sum\alpha_iy_ix_i$.
     - ∂/∂b gives $\sum\alpha_iy_i=0$.
     - Substitute back to get $\max_\alpha\sum\alpha_i-\frac12\sum_{ij}\alpha_i\alpha_jy_iy_jx_i^\top x_j$ s.t. α ≥ 0.
     - Only inner products appear, which sets up kernels.
  5. *Complementary slackness and support vectors*: $\alpha_i[y_if(x_i)-1]=0$, so α > 0 only on the margin. Compute b from any margin SV (average for stability). Removing a non-SV leaves the solution unchanged.
  6. *Soft margin*:
     - Slacks $\xi_i\ge0$ with $y_if(x_i)\ge1-\xi_i$ and objective $\frac12\|w\|^2+C\sum\xi_i$.
     - Dual box constraint 0 ≤ α ≤ C (derive).
     - Three kinds of points: α=0 (outside the margin), 0<α<C (on the margin), α=C (inside the margin or misclassified).
     - What C trades off.
  7. *Hinge loss as regularized ERM*: eliminate ξ to get $\sum\max(0,1-y_if(x_i))+\frac\lambda2\|w\|^2$ with λ ∝ 1/C. Comparison with logistic regression (similar boundaries; SVM is sparse in α and gives no probabilities).
  8. *Solving it*:
     - QP solvers.
     - SMO: why it updates two α's at a time (the equality constraint), and a coordinate-ascent sketch (CS229).
     - Primal (sub)gradient/Pegasos (UML §15.5).
     - Complexity in N.
  9. *Worked numeric example*: two or three points in 2-D, solved by hand for w, b, margin and the SVs.
- **Question ideas**
  - (calc) w = (3, 4): margin width 2/5 = 0.4.
  - Remove a point with α = 0: does the solution change? (no) Remove an SV? (it may)
  - Where does $\sum\alpha_iy_i=0$ come from? (stationarity in b)
  - Predict: C → ∞ (on separable data the solution approaches the hard margin; on non-separable data every finite C still has a solution, but slack is so expensive that the boundary chases individual violators). C → 0 (wide margin, many violations).
  - Which is false: "the SVM's solution depends on all training points".
  - (calc) Hand-solve: points (1,1) labelled +1 and (−1,−1) labelled −1 → w = (½, ½), b = 0, margin width $2\sqrt2$.
- **Figures**
  - `fund.svm-margin-dual/margin`: separable 2-D data, max-margin line, margin band, circled SVs.
  - `fund.svm-margin-dual/soft-margin-c`: the same data plus an outlier, boundaries for C = 0.1, 1, 100. **[fig-Q]**: "which C?"
  - `fund.svm-margin-dual/alpha-types`: points coloured by α=0, 0<α<C and α=C.
  - `fund.svm-margin-dual/hinge-vs-logistic`: the two losses vs yf.
- **Pitfalls & source disagreements**
  - The C vs λ conventions (sum vs mean of slacks).
  - ESL parameterizes the soft margin with ‖β‖ = 1 and margin M. CS229 and Bishop use the canonical scaling.
  - Labels must be ±1.

---

#### 280 · `fund.svm-kernels`: SVMs II: Kernels, Multiclass & Probabilities
- **Level** intermediate · **W2** · **prereqs** [fund.svm-margin-dual] · **covers** SVMs.
- **Sources**
  - CS229 notes ch.5 kernel methods: feature maps, LMS with the kernel trick, properties of kernels & Mercer (pdf p.50–60). `cs229.txt`
  - Bishop PRML §6.1 dual representations, §6.2 constructing kernels (pdf p.313–319), §7.1.3 multiclass SVMs, §7.1.4 SVR (pdf p.358–364). `bishop.txt`
  - UML (SSBD) ch.16 kernel methods: §16.2 the kernel trick & representer theorem, §16.3 kernelized soft-SVM (p.215–226). `uml.txt`
  - Murphy PML1 §17.1 Mercer kernels (pdf p.597), §17.3.4 kernel trick, §17.3.5 SVM outputs → probabilities, §17.3.7 multiclass, §17.3.8 choosing C, §17.3.9 kernel ridge, §17.3.10 SVR (pdf p.620–627). `pml1.txt`
  - ESL §12.3–12.3.4 SVMs & kernels, the curse of dimensionality for SVMs (pdf p.442–451). `esl.txt`
- **Subtopic map**
  1. *Motivation*: non-linear boundaries via a feature map φ. Explicit features blow up (degree-2 polynomial in d dims has O(d²) features). The dual only needs $\phi(x)^\top\phi(x')$.
  2. *Kernel trick*: replace $x_i^\top x_j$ with $k(x_i,x_j)$ in the dual and in prediction $f(x)=\sum\alpha_iy_ik(x_i,x)+b$. Worked example: $(x^\top z)^2$ equals the inner product of explicit degree-2 monomial features (verify in 2-D). Then the RBF kernel in 1-D: $e^{-\gamma(x-z)^2}=e^{-\gamma x^2}e^{-\gamma z^2}\sum_k\frac{(2\gamma)^k}{k!}x^kz^k$ exhibits an infinite feature map.
  3. *Valid kernels (Mercer)*: any explicit feature map gives a PSD Gram matrix, because $\sum_{ij}c_ic_jk(x_i,x_j)=\|\sum_ic_i\phi(x_i)\|^2\ge0$ (prove). Conversely, a symmetric k whose Gram matrices are all PSD has a feature map (Mercer / Moore–Aronszajn, statement). Closure rules: sums, products, positive scaling, $f(x)k f(x')$, exp of a kernel.
  4. *Common kernels*:
     - Linear.
     - Polynomial $(x^\top z+c)^p$, with feature count $\binom{d+p}{p}$.
     - RBF $\exp(-\gamma\|x-z\|^2)$: infinite-dimensional feature map, smoothness, and γ as an inverse bandwidth.
     - Sigmoid kernel (not always PSD).
     - String and graph kernels (named).
  5. *Effect of γ and C*: the bias–variance grid (large γ → bumps around each point, overfitting; small γ → nearly linear). Tuning on a log grid via CV.
  6. *Representer theorem*: for any loss + $\|f\|^2$ regularizer, the optimum lies in the span of $k(x_i,\cdot)$ (UML §16.2 proof sketch). Kernel ridge regression closed form $\alpha=(K+\lambda I)^{-1}y$ as a second example.
  7. *Scaling*: O(N²) memory for the Gram matrix and roughly O(N²–N³) training. Prediction costs O(#SV). Approximations: Nyström and random Fourier features (named).
  8. *Multiclass*: one-vs-rest (K classifiers; score calibration issues) vs one-vs-one (K(K−1)/2 classifiers with voting) vs Crammer–Singer (named).
  9. *Probabilities*: SVM scores aren't probabilities. Platt scaling (fit $\sigma(af+b)$ on held-out data). Pointer to fund.calibration.
  10. *SVR* in one card: the ε-insensitive loss gives a tube with sparse SVs.
  11. *SVMs today*: when they're still a good choice (small N, good features) and the connection to the NTK/GP view (named). Gaussian processes as the Bayesian kernel method: same Gram matrix, and the GP posterior mean equals kernel ridge regression (one line; not otherwise covered in Part A).
- **Question ideas**
  - (calc) The degree-2 kernel $(x^\top z+1)^2$ in d=100 has $\binom{102}{2}=5151$ implicit features (constant, linear and quadratic terms); the homogeneous $(x^\top z)^2$ has $\binom{101}{2}=5050$.
  - Is $k(x,z)=-\|x-z\|^2$ a valid kernel? (no: Gram matrix not PSD)
  - Predict: RBF with γ very large vs very small. **[fig-Q]** Rank three panels by γ.
  - OvO with K=10 classes → 45 classifiers.
  - Which is false: "the RBF kernel corresponds to a finite-dimensional feature map".
  - Platt scaling: what is it fit on, and why not the training set?
- **Figures**
  - `fund.svm-kernels/rbf-gamma`: decision regions for three γ values. **[fig-Q]**
  - `fund.svm-kernels/lift-to-3d`: circular 1-D/2-D data lifted with $x\mapsto(x,x^2)$ becoming linearly separable.
  - `fund.svm-kernels/ovr-vs-ovo`: OvR vs OvO regions for 3 classes, including the ambiguous zones.
- **Pitfalls & source disagreements**
  - RBF parameterization γ = 1/(2σ²) vs σ. sklearn's `gamma='scale'`.
  - "Kernel" also means convolution kernel (CNNs) and smoothing kernel (KDE). Disambiguate.

---

#### 290 · `fund.decision-trees`: Decision Trees
- **Level** core · **W1** · **prereqs** [fund.bias-variance] · **covers** decision trees.
- **Sources**
  - ESL §9.2 tree-based methods (pdf p.324–336):
    - §9.2.2 regression trees & cost-complexity pruning (pdf p.326).
    - §9.2.3 classification trees, impurity measures and the misclassification-error counterexample (pdf p.327).
    - §9.2.4 other issues: categorical predictors, loss matrix, missing values via surrogate splits, binary vs multiway splits, linear-combination splits, instability, lack of smoothness, additive structure (pdf p.329).
    - `esl.txt`
  - UML (SSBD) ch.18: §18.2.1 gain measures, §18.2.2 pruning, §18.2.3 threshold splits for real features (p.250–255). `uml.txt`
  - Murphy PML1 §18.1 CART: model definition, fitting, regularization, missing features, pros & cons (pdf p.633–637). `pml1.txt`
  - Bishop PRML §14.4 tree-based models (pdf p.683). `bishop.txt`
- **Subtopic map**
  1. *Model*: recursive axis-aligned partition with a constant prediction per leaf (mean for regression, class distribution for classification). Interpretable as rules.
  2. *Greedy growing*: optimal trees are NP-hard, so build greedily. Regression split criterion: minimize SSE with leaf means. Efficient threshold search by sorting each feature (O(pN log N) per node, then a linear scan with running sums).
  3. *Impurity measures* for class proportions $p_k$:
     - Misclassification $1-\max p_k$.
     - Gini $\sum p_k(1-p_k)$.
     - Entropy $-\sum p_k\log p_k$.
     - Derive Gini's interpretations: expected error of random labelling, and variance of one-hot indicators. Information gain = parent impurity − weighted child impurity.
  4. *Why Gini/entropy and not misclassification for growing*: strict concavity rewards purer children. Work ESL's example, where misclassification rates two splits equally while Gini/entropy prefer the one producing a pure node.
  5. *Stopping and pruning*: pre-pruning (max depth, min samples per leaf, min impurity decrease) vs post-pruning. Cost-complexity $C_\alpha(T)=\sum_mN_mQ_m+\alpha|T|$, weakest-link pruning, α by CV, and the 1-SE rule.
  6. *Categorical features*:
     - $2^{q-1}-1$ possible binary partitions.
     - For a binary target (or regression), sorting categories by mean response finds the optimal split among q−1 candidates.
     - Many-level categoricals cause overfitting and bias (C4.5 uses gain ratio).
  7. *Missing values*: surrogate splits (CART), a separate "missing" branch, or a learned default direction (XGBoost; pointer to fund.gradient-boosting).
  8. *Properties*:
     - Invariant to monotone transforms of each feature, so no scaling is needed.
     - Handles mixed types.
     - Captures interactions.
     - Piecewise-constant and axis-aligned (bad for diagonal boundaries or additive smooth structure).
  9. *High variance and instability*: small data changes can alter the top split, and the change propagates down the tree. This motivates bagging and RF (pointer to fund.bagging-random-forests).
  10. *Feature importance*: total impurity decrease (biased toward high-cardinality and continuous features) vs permutation importance (pointer to the RF lesson).
- **Question ideas**
  - (calc) Node (8+, 2−): Gini 0.32, entropy ≈ 0.722 bits.
  - (calc) Information gain for parent (5+, 5−) split into (4+, 1−) and (1+, 4−): Gini decrease 0.5 − 0.32 = 0.18.
  - Which transformation changes the learned splits? (rotation does; log of one feature doesn't)
  - Why not grow with misclassification error? (ESL example)
  - Predict: a fully grown tree's training error and test behaviour.
  - Which is false: "decision trees require feature standardization".
- **Figures**
  - `fund.decision-trees/impurity-curves`: misclassification, Gini and scaled entropy vs p (two classes).
  - `fund.decision-trees/tree-and-partition`: a small tree beside its 2-D partition. **[fig-Q]**: "which leaf contains x?"
  - `fund.decision-trees/axis-aligned`: the staircase approximation to a diagonal boundary.
  - `fund.decision-trees/pruning-cv`: CV error vs α / tree size with the 1-SE choice.
- **Pitfalls & source disagreements**
  - CART (binary splits, Gini, cost-complexity) vs ID3/C4.5 (multiway, entropy, gain ratio) differ in defaults.
  - ESL writes Gini as $\sum_{k\neq k'}p_kp_{k'}$, which is equivalent.

---

#### 300 · `fund.ensembles`: Why Ensembles Work: Voting, Averaging & Stacking
- **Level** core · **W2** · **prereqs** [fund.variance-covariance, fund.bias-variance] · **covers** ensembles.
- **Sources**
  - Bishop PRML §14.1 Bayesian model averaging vs combinations, §14.2 committees and the $E_{COM}=\frac1ME_{AV}$ derivation (pdf p.674–677). `bishop.txt`
  - ESL §8.8 model averaging & stacking (pdf p.307), §16.1 & §16.3 learning ensembles (pdf p.624–641). `esl.txt`
  - Murphy PML1 §18.2 ensemble learning, §18.2.1 stacking, §18.2.2 ensembling is not BMA (pdf p.638–639). `pml1.txt`
  - Goodfellow DL §7.11 bagging & ensembles (the expected squared error of an average with correlated errors). `dlb_ch07_regularization.txt`
  - Lakshminarayanan et al. 2017 (`papers/lakshminarayanan2017_deep_ensembles.txt`).
- **Subtopic map**
  1. *The averaging formula*: k models with error variance v and covariance c have average-error variance $\frac1kv+\frac{k-1}kc$ (DLB §7.11; derive). Equivalently $\rho\sigma^2+\frac{1-\rho}k\sigma^2$. Independent errors give a 1/k reduction. Perfect correlation gives no gain.
  2. *Committee result* (Bishop): with uncorrelated zero-mean errors, $E_{COM}=E_{AV}/M$. In general $E_{COM}\le E_{AV}$ by Jensen/Cauchy–Schwarz (derive), so averaging never hurts squared error.
  3. *Majority voting*: with independent classifiers each better than chance, the majority accuracy → 1 as k grows (Condorcet; calc with 5 voters at 0.7 → 0.837). With correlated voters the gain disappears. A worse-than-chance base learner gets worse.
  4. *Sources of diversity*: different data (bagging), features (random subspaces), initializations and hyperparameters (deep ensembles), model families. Why diversity matters more than individual accuracy past a point.
  5. *Combination rules*: hard vs soft voting, averaging probabilities vs logits, weighted averaging.
  6. *Stacking*: a meta-learner trained on **out-of-fold** base predictions. Why in-sample predictions leak. ESL §8.8's constrained-weights view.
  7. *Ensembles vs Bayesian model averaging*: BMA weights by posterior and concentrates on a single model as N grows (it's soft model selection). Ensembles *enlarge* the hypothesis class (PML1 §18.2.2; Minka's point).
  8. *Deep ensembles*: M random inits. Better accuracy, calibration and OOD uncertainty. Cost M×. Cheaper relatives: snapshot ensembles, MC dropout, weight averaging/SWA, model soups (named).
  9. *Bias vs variance*: averaging reduces variance. Boosting (sequential) reduces bias. A preview table for the next three lessons.
- **Question ideas**
  - (calc) ρ=0.3, σ²=1, k=100 → 0.307. As k → ∞ → 0.3.
  - (calc) 5 independent voters at 70% → 83.7% majority accuracy.
  - Spot the flaw: a stacking meta-learner fit on in-sample base predictions.
  - Which is false: "BMA and ensembling are the same thing in the large-data limit".
  - Predict: averaging 10 copies of the same deterministic model (no gain).
- **Figures**
  - `fund.ensembles/variance-vs-k`: ensemble variance vs k for ρ ∈ {0, 0.3, 0.7}. Notice the floor at ρσ².
  - `fund.ensembles/majority-vote`: majority accuracy vs number of voters for p = 0.55, 0.7, plus a correlated-voters curve. **[fig-Q]**
  - `fund.ensembles/stacking-diagram`: K-fold out-of-fold prediction flow into the meta-learner.
- **Pitfalls & source disagreements**
  - "Ensembles are Bayesian" is a common conflation (PML1 §18.2.2).

---

#### 310 · `fund.bagging-random-forests`: Bagging & Random Forests
- **Level** core · **W1** · **prereqs** [fund.decision-trees, fund.ensembles] · **covers** bagging, ensembles.
- **Sources**
  - ESL §8.7 bagging incl. §8.7.1 trees with simulated data and the "bagging a bad classifier" caveat (pdf p.301–307); ch.15 random forests (pdf p.606–622):
    - §15.2 definition and default m.
    - §15.3.1 OOB (pdf p.611).
    - §15.3.2 variable importance (pdf p.612).
    - §15.3.3 proximity plots.
    - §15.3.4 RF and overfitting (pdf p.615).
    - §15.4.1 variance & de-correlation (pdf p.616).
    - §15.4.2 bias (pdf p.619).
    - §15.4.3 adaptive nearest neighbours.
    - `esl.txt`
  - Breiman 1996 bagging (`papers/breiman1996_bagging.txt`): instability and when bagging helps. Breiman 2001 random forests (`papers/breiman2001_random_forests.txt`): strength/correlation bound, OOB.
  - Murphy PML1 §18.3 bagging, §18.4 random forests, §18.6 interpreting tree ensembles (pdf p.639–641, p.650). UML (SSBD) §18.3 (p.255). `pml1.txt`, `uml.txt`
- **Subtopic map**
  1. *Bootstrap aggregation*: draw B bootstrap samples, fit a model on each, average (regression) or vote (classification). Each sample has ≈ 63.2% unique points (derive via $(1-1/N)^N$).
  2. *Why bagging works*: it reduces variance through the averaging formula. Bias is roughly unchanged (the bagged estimator's expectation ≈ the base model's on bootstrap data). It helps **unstable** learners (deep trees) and barely helps stable ones (kNN with large k, linear regression; Breiman 1996).
  3. *The correlation floor*: bagged trees are correlated (they share strong splits), so variance → ρσ² as B → ∞. That limits bagging.
  4. *Random forests*: at each split, consider only m randomly chosen features. Defaults: √p (classification), ⌊p/3⌋ (regression) with min leaf 5 (ESL §15.3). Smaller m lowers ρ but raises the variance and bias of each tree. Breiman's strength–correlation bound.
  5. *OOB error*: each tree is evaluated on the ~36.8% of points it didn't see, aggregated per point. It approximates leave-one-out CV for free. When OOB is pessimistic (fewer trees vote per point).
  6. *Feature importance*:
     - Impurity-decrease (MDI) importance is biased toward continuous and high-cardinality features and is computed on training data.
     - Permutation importance on OOB/validation data is preferred, but correlated features share or mask importance.
     - Partial dependence (named).
  7. *Hyperparameters*: number of trees (more doesn't overfit; diminishing returns), m, depth/min leaf, bootstrap fraction. Sensitivity is low. RF as a strong default for tabular data.
  8. *"RFs don't overfit"*: adding trees doesn't overfit, but the limiting forest of fully grown trees can overfit noisy data (ESL §15.3.4). Use min-leaf/depth limits.
  9. *Extensions* in one paragraph: extremely randomized trees, quantile regression forests, isolation forests for anomalies (named). RF as an adaptive nearest-neighbour method (ESL §15.4.3).
  10. *RF vs GBDT preview*: parallel vs sequential, variance vs bias reduction, tuning burden (pointer to fund.gradient-boosting).
- **Question ideas**
  - (calc) OOB fraction per tree ≈ 0.368. Unique fraction ≈ 0.632.
  - (calc) Tree variance σ²=1, ρ=0.4, B=500 → ≈ 0.401. If m is lowered so ρ=0.2 but σ²=1.2 → ≈ 0.241.
  - What is RF with m = p? (bagged trees)
  - Which base learner benefits least from bagging? (kNN with large k / linear regression)
  - Which is false: "MDI importance is unbiased with respect to feature cardinality".
  - Predict: OOB error vs number of trees (decreases and plateaus, no U-shape).
- **Figures**
  - `fund.bagging-random-forests/bagged-boundary`: a single deep tree's jagged boundary vs a 100-tree bagged average.
  - `fund.bagging-random-forests/oob-vs-test`: OOB and test error vs number of trees.
  - `fund.bagging-random-forests/m-tradeoff`: test error vs m (features per split). **[fig-Q]**
  - `fund.bagging-random-forests/importance-bias`: MDI vs permutation importance with an added random high-cardinality ID feature.
- **Pitfalls & source disagreements**
  - Breiman's "RFs do not overfit" vs ESL's qualification.
  - Library defaults: sklearn `RandomForestRegressor` defaults to max_features = 1.0 (all features, i.e. bagging) since v1.1, not ESL's p/3.

---

#### 320 · `fund.boosting`: Boosting & AdaBoost
- **Level** intermediate · **W2** · **prereqs** [fund.decision-trees, fund.ensembles, fund.loss-functions] · **covers** boosting.
- **Sources**
  - ESL ch.10 (pdf p.356–372):
    - §10.1 AdaBoost.M1 (pdf p.356).
    - §10.2 boosting fits an additive model, §10.3 forward stagewise additive modelling, §10.4 exponential loss & AdaBoost derivation (pdf p.360–364).
    - §10.5 why exponential loss: population minimizer (pdf p.364).
    - §10.6 loss functions & robustness (pdf p.365).
    - `esl.txt`
  - UML (SSBD) ch.10: §10.1 weak learnability, §10.2 AdaBoost & training-error bound (Thm 10.2), §10.3 linear combinations & VC (p.130–143). `uml.txt`
  - Freund & Schapire 1999 (`papers/freund1999_boosting_intro.txt`): algorithm, training error, generalization, margins.
  - Bishop PRML §14.3 boosting, §14.3.1 minimizing exponential error, §14.3.2 error functions (pdf p.677–683). Murphy PML1 §18.5.1–18.5.4 forward stagewise, least-squares boosting, AdaBoost, LogitBoost (pdf p.642–646).
- **Subtopic map**
  1. *Weak learners and the boosting question*: can many slightly-better-than-chance learners be combined into a strong one? Weak learnability (UML §10.1).
  2. *AdaBoost algorithm*:
     - Sample weights.
     - Weighted error $\varepsilon_m$.
     - Learner weight $\alpha_m=\frac12\ln\frac{1-\varepsilon_m}{\varepsilon_m}$.
     - Reweighting $w_i\leftarrow w_i e^{-\alpha_my_ih_m(x_i)}$ then normalize.
     - Final classifier $\mathrm{sign}(\sum\alpha_mh_m)$.
     - Trace two rounds on a toy dataset.
  3. *Training-error bound*: $\prod_m2\sqrt{\varepsilon_m(1-\varepsilon_m)}\le\exp(-2\sum\gamma_m^2)$ with edge $\gamma_m=\frac12-\varepsilon_m$ (derive the first inequality via $\frac1N\sum\mathbb 1[\text{err}]\le\frac1N\sum e^{-y_iF(x_i)}=\prod Z_m$).
  4. *AdaBoost as forward stagewise additive modelling with exponential loss* (ESL §10.4): greedily add $\beta h$ to minimize $\sum e^{-y(F+\beta h)}$. This **derives** the reweighting (current weights = $e^{-y_iF(x_i)}$), the choice of h (minimize weighted error), and β = α.
  5. *Why exponential loss*: its population minimizer is ½ log-odds, so $\mathrm{sign}(F)$ is Bayes-optimal and probabilities are recoverable via σ(2F). Compare with binomial deviance (LogitBoost).
  6. *Robustness*: exponential loss puts exponentially large weight on misclassified (often mislabelled) points, so it degrades under label noise. Deviance is more robust (ESL §10.6).
  7. *Behaviour*: the test error often keeps falling after the training error hits 0. The margin explanation (Schapire) vs the additive-model/regularization view. Boosting *can* overfit eventually.
  8. *Base learners*: stumps give an additive model with no interactions. Depth-J trees capture interactions up to order J−1 (pointer to the next lesson).
  9. *Multiclass*: SAMME (named). AdaBoost for face detection (Viola–Jones, named; UML §10.4).
- **Question ideas**
  - (calc) ε=0.2 → α = ½ ln 4 ≈ 0.693, so misclassified weights ×2 and correct ×0.5 before renormalizing.
  - What if ε_m = 0.5? (α=0) ε > 0.5? (negative α, i.e. flip the learner)
  - (calc) Training-error bound after 10 rounds with every $\varepsilon_m=0.3$: $(2\sqrt{0.21})^{10}\approx0.42$.
  - Predict: AdaBoost under 10% label noise (weight concentrates on noisy points).
  - Which is false: "AdaBoost's reweighting rule is a heuristic with no loss-function interpretation".
  - Derivation step: why are the current weights proportional to $e^{-y_iF_{m-1}(x_i)}$?
- **Figures**
  - `fund.boosting/adaboost-rounds`: boundary and point weights (marker size) after rounds 1, 3 and 10. **[fig-Q]**: "which points will gain weight?"
  - `fund.boosting/train-test-rounds`: training and test error vs rounds, with the test error falling after the training error reaches 0.
  - `fund.boosting/exp-vs-deviance`: exponential and binomial deviance vs margin.
- **Pitfalls & source disagreements**
  - ESL's AdaBoost.M1 uses $\alpha=\ln\frac{1-\varepsilon}\varepsilon$ and reweights only the misclassified points. Freund–Schapire, UML and Bishop use ½ ln with symmetric updates. The classifiers are equivalent up to scaling, but the per-step numbers differ.
  - Labels ±1 throughout.

---

#### 330 · `fund.gradient-boosting`: Gradient Boosting & XGBoost
- **Level** intermediate · **W2** · **prereqs** [fund.boosting, fund.gradient-descent] · **covers** boosting. Demoted to W2 in review: its prereq fund.boosting is W2, and frontier-lab research interviews probe XGBoost internals less than optimization and probabilistic basics.
- **Sources**
  - ESL §10.9 boosting trees (pdf p.372), §10.10 numerical optimization via gradient boosting: §10.10.1 steepest descent, §10.10.2 gradient boosting, §10.10.3 implementations incl. Table 10.2 of pseudo-residuals (pdf p.377–380), §10.11 right-sized trees (pdf p.380), §10.12.1 shrinkage, §10.12.2 subsampling (pdf p.383–386), §10.13 interpretation (pdf p.386). `esl.txt`
  - Friedman 2001 (`papers/friedman2001_gbm.txt`): functional gradient descent, the LS/LAD/Huber/logistic algorithms, shrinkage.
  - Chen & Guestrin 2016 XGBoost (`papers/chen2016_xgboost.txt`): §2.1 regularized objective, §2.2 second-order approximation, optimal leaf weight & gain, §2.3 shrinkage & column subsampling, §3 exact/approximate/sparsity-aware split finding.
  - Murphy PML1 §18.5.5 gradient boosting incl. XGBoost (pdf p.646–650), §18.6 interpreting tree ensembles (pdf p.650). `pml1.txt`
- **Subtopic map**
  1. *Functional gradient descent*: treat the predictions $F(x_i)$ as parameters. The steepest-descent direction is the negative gradient (the pseudo-residuals) $r_{im}=-\partial L(y_i,F)/\partial F|_{F_{m-1}}$. Fit a regression tree to $r_{im}$ to generalize the direction to new x.
  2. *Algorithm*: initialize with a constant. Each round: compute pseudo-residuals, fit a J-leaf tree, do an optimal line search per leaf $\gamma_{jm}=\arg\min\sum_{x_i\in R_{jm}}L(y_i,F+\gamma)$, then update with shrinkage $F\leftarrow F+\nu\sum\gamma_{jm}\mathbb 1[x\in R_{jm}]$.
  3. *Pseudo-residuals for common losses* (derive):
     - Squared → y − F, i.e. fitting residuals.
     - Absolute → sign(y − F).
     - Huber → clipped residual.
     - Binomial deviance → y − p.
     - The per-leaf optimal values (mean, median, Newton step for logistic).
  4. *Regularization*: shrinkage ν (smaller ν needs more trees and generalizes better; Friedman's finding), stochastic subsampling of rows (and columns), tree size J (interaction order; 4–8 leaves typical), number of rounds via early stopping on validation (boosting overfits unlike RF).
  5. *XGBoost's second-order objective*:
     - Taylor-expand the loss to second order with $g_i,h_i$.
     - Add $\gamma T+\frac\lambda2\sum w_j^2$.
     - Per leaf, $\sum_{i\in j}(g_iw_j+\frac12h_iw_j^2)+\frac\lambda2w_j^2$ gives $w_j^*=-G_j/(H_j+\lambda)$ (derive).
     - Structure score $-\frac12\sum G_j^2/(H_j+\lambda)+\gamma T$.
     - Split gain $\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$.
     - Interpretation: it's a Newton step per leaf, and γ acts as built-in pruning.
  6. *Systems tricks*: weighted-quantile/histogram split candidates, sparsity-aware default directions for missing values, column blocks, cache awareness. LightGBM's leaf-wise growth and GOSS, CatBoost's ordered target statistics (named).
  7. *Interpretation*: relative importance (split gain, frequency) and partial dependence (ESL §10.13).
  8. *GBDT vs RF*: bias vs variance reduction, sequential vs parallel, sensitivity to hyperparameters, robustness to noise, and typical tabular performance. When to use which.
  9. *Worked numeric example*: one round of gradient boosting on 4 points with squared loss, plus one XGBoost leaf/gain computation.
- **Question ideas**
  - (calc) XGBoost leaf with G=−4, H=3, λ=1 → w* = 1, and its contribution to the score.
  - (calc) Gain: $G_L=-4,H_L=3,G_R=2,H_R=2$, λ=1, γ=0 → ½[16/4 + 4/3 − 4/6] = ½[4 + 1.333 − 0.667] ≈ 2.33.
  - Pseudo-residuals under absolute loss? (signs) Under logistic? (y − p)
  - Predict: ν = 1 vs 0.1 with early stopping (0.1 needs ~10× more trees and usually tests better).
  - Which is false: "adding more boosting rounds can never increase test error".
  - Compare: RF vs GBDT on a small noisy dataset with no tuning budget (RF is safer).
- **Figures**
  - `fund.gradient-boosting/residual-fitting`: 1-D gradient boosting on squared loss: fit and residuals at m = 1, 5, 50.
  - `fund.gradient-boosting/shrinkage-tradeoff`: test error vs number of trees for ν ∈ {1, 0.1, 0.01}. **[fig-Q]**
  - `fund.gradient-boosting/xgb-split`: a node with gradient/Hessian sums and its two candidate splits annotated with gains.
- **Pitfalls & source disagreements**
  - "Learning rate" in GBM libraries means shrinkage ν.
  - XGBoost's λ (L2 on leaf weights) and γ (per-leaf penalty) are not ESL's ν. Some texts write η for shrinkage.

---

#### 340 · `fund.clustering`: k-Means & Clustering
- **Level** intermediate · **W2** · **prereqs** [fund.variance-covariance] · **covers** clustering (k-means), unsupervised learning.
- **Sources**
  - Bishop PRML §9.1 k-means incl. the objective, the two-step proof, sequential/online k-means and k-medoids, §9.1.1 compression (pdf p.444–450). `bishop.txt`
  - ESL §14.3 cluster analysis: §14.3.1–14.3.3 proximity & dissimilarities, §14.3.6 k-means (pdf p.528), §14.3.9 vector quantization, §14.3.10 k-medoids (pdf p.534), §14.3.11 practical issues & choosing K/gap (pdf p.537), §14.3.12 hierarchical clustering & linkages (pdf p.539–547); §14.5.3 spectral clustering (pdf p.563). `esl.txt`
  - Murphy PML1 §21.1–21.3 (evaluation incl. purity/Rand/MI, hierarchical agglomerative, k-means, §21.3.4 k-means++, §21.3.5 k-medoids, §21.3.7 choosing K; pdf p.745–759), §21.5 spectral (pdf p.764). `pml1.txt`
  - MacKay ITILA ch.20 k-means & soft k-means (pdf p.296–302). UML (SSBD) ch.22 (linkage, k-means, spectral; p.307–322). CS229 notes ch.10 (pdf p.148). Arthur & Vassilvitskii 2007 (`papers/arthur2007_kmeanspp.txt`).
- **Subtopic map**
  1. *What clustering is and why it's ill-posed*: no labels, and the "right" clustering depends on the metric and the goal. Internal vs external evaluation.
  2. *k-means objective* $J=\sum_n\sum_kr_{nk}\|x_n-\mu_k\|^2$ (within-cluster sum of squares). Equivalently it minimizes the average within-cluster pairwise squared distance (ESL identity).
  3. *Lloyd's algorithm as coordinate descent*:
     - Assignment step: nearest centroid, optimal for fixed μ.
     - Update step: the mean, optimal for fixed r (derive ∂J/∂μ_k = 0).
     - **Proof** that J never increases.
     - Finite termination (finitely many partitions), so it converges to a local optimum, not the global one (NP-hard in general).
  4. *Complexity and speed-ups*: O(NKd) per iteration. Mini-batch k-means. Elkan/triangle-inequality tricks (named).
  5. *Initialization*: random restarts. **k-means++**: D² sampling, $P(x)\propto D(x)^2$, gives an O(log K)-competitive expected cost (statement).
  6. *Choosing K*: J decreases monotonically in K (why), so the elbow is a heuristic. Silhouette, gap statistic, BIC via GMM (pointer to the next lesson), stability.
  7. *Limitations*: Voronoi (linear) boundaries, assumes roughly spherical and similar-size clusters, sensitive to scaling and outliers (squared distance). Fails on rings and moons. Remedies: k-medoids (any dissimilarity, robust), kernel k-means/spectral clustering, DBSCAN (density-based; named with its two parameters).
  8. *Hierarchical agglomerative clustering*: single linkage chains, complete linkage gives compact clusters, average and Ward. Dendrogram cuts. O(N²) memory.
  9. *Evaluation*: silhouette (internal). Purity, ARI and NMI (external; invariant to label permutation).
  10. *k-means as vector quantization/compression* (image colour quantization) and as the hard-assignment limit of a GMM (preview of fund.gmm-em).
- **Question ideas**
  - (calc) 1-D {1, 2, 10, 11} with initial centroids {1, 2}: iteration 1 gives {1} | {2, 10, 11} with μ = (1, 7.67); iteration 2 gives {1, 2} | {10, 11} with μ = (1.5, 10.5); J goes 145 → 48.7 → 1.0.
  - Flashcard: prove k-means never increases J.
  - Which dataset defeats k-means? (concentric rings; very unequal cluster variances)
  - Predict: standardizing features changes the clustering? (yes, in general)
  - Which is false: "k-means always finds the global minimum of J".
  - Single vs complete linkage on two elongated clusters joined by a thin bridge (single linkage chains).
- **Figures**
  - `fund.clustering/kmeans-iterations`: four panels of Lloyd iterations with Voronoi cells and J annotated. **[fig-Q]**: "what happens in the next step?"
  - `fund.clustering/kmeans-failure`: rings/moons where k-means fails, next to a spectral-clustering result.
  - `fund.clustering/elbow-silhouette`: J vs K and silhouette vs K.
  - `fund.clustering/dendrogram`: a small dendrogram with a cut line.
- **Pitfalls & source disagreements**
  - k-means assumes Euclidean geometry, so clustering unnormalized embeddings is a common bug.
  - MacKay's soft k-means uses a stiffness β (inverse temperature) and isn't the same as GMM-EM.

---

#### 350 · `fund.gmm-em`: Gaussian Mixtures & the EM Algorithm
- **Level** intermediate · **W2** · **prereqs** [fund.clustering, fund.mle, fund.kl-divergence] · **covers** clustering, unsupervised learning (+ the ELBO foundation for gen.latent-variables-elbo).
- **Sources**
  - Bishop PRML §9.2 mixtures of Gaussians: §9.2.1 ML & singularities & identifiability, §9.2.2 EM for GMMs (pdf p.450–459); §9.3 alternative view of EM incl. §9.3.2 relation to k-means (pdf p.459–465); §9.4 EM in general: the ELBO/KL decomposition and the monotonicity proof (pdf p.470–475). `bishop.txt`
  - ESL §8.5 EM: two-component mixture, EM in general, EM as maximization–maximization (pdf p.291–298), §14.3.7 GMM as soft k-means (pdf p.529). `esl.txt`
  - CS229 notes ch.11: EM for mixtures of Gaussians, Jensen's inequality, general EM, other interpretation of the ELBO (pdf p.151–163). `cs229.txt`
  - MacKay ITILA ch.22 ML for mixtures & the fatal flaw (pdf p.312–318). Murphy PML1 §21.4 (pdf p.759), §8.7 bound optimization/MM & EM (pdf p.342–349).
- **Subtopic map**
  1. *Mixture model*: $p(x)=\sum_k\pi_kN(x|\mu_k,\Sigma_k)$ with latent $z\sim\mathrm{Cat}(\pi)$ and the generative story. It is a density model and a soft clustering.
  2. *Why direct MLE is hard*: the log-sum doesn't decouple, so there's no closed form. Label switching (K! symmetric optima). **Singularities**: a component on one point with σ → 0 sends the likelihood → ∞ (Bishop §9.2.1, MacKay §22.4). Fixes: priors, variance floors.
  3. *Responsibilities* $\gamma_{nk}=p(z=k|x_n)$ via Bayes.
  4. *EM for GMMs*: derive the M-step updates from the expected complete-data log-likelihood:
     - $N_k=\sum_n\gamma_{nk}$.
     - $\mu_k=\frac1{N_k}\sum\gamma_{nk}x_n$.
     - $\Sigma_k=\frac1{N_k}\sum\gamma_{nk}(x_n-\mu_k)(x_n-\mu_k)^\top$.
     - $\pi_k=N_k/N$.
     - Include a numeric 1-D example.
  5. *General EM*:
     - For any q(z), $\log p(x|\theta)=\underbrace{E_q[\log p(x,z|\theta)-\log q(z)]}_{\mathcal L(q,\theta)}+\mathrm{KL}(q\|p(z|x,\theta))$. Derive it, plus the Jensen derivation of the bound.
     - E-step: set q = posterior, making the bound tight.
     - M-step: maximize $\mathcal L$ in θ.
     - **Proof of monotone increase** of the log-likelihood.
     - It converges to a stationary point (local optima).
  6. *Relation to k-means*: shared covariance εI with ε → 0 gives hard assignments, i.e. Lloyd's algorithm (Bishop §9.3.2).
  7. *Practicalities*: initialize from k-means. Covariance types (full/diag/tied/spherical; parameter counts). Choose K with BIC. Monitor the log-likelihood.
  8. *Variants*: generalized EM (partial M-step), stochastic/online EM, variational EM (q restricted; pointer to gen.latent-variables-elbo), MM algorithms (EM is a special case).
  9. *Other latent-variable models* that use EM: HMMs (Baum–Welch, named), factor analysis/PPCA (pointer to fund.pca), missing-data imputation.
- **Question ideas**
  - (calc) 1-D two-component E-step for a point given parameters: compute the responsibilities.
  - Which quantity is guaranteed not to decrease in EM? (the marginal log-likelihood log p(X|θ))
  - Predict: a GMM component collapses onto one point (likelihood → ∞, a degenerate solution).
  - Derivation step: why is the bound tight after the E-step? (KL = 0)
  - (calc) Parameter count of a K=3 full-covariance GMM in d=2: 3·(2 + 3) + 2 = 17.
  - Which is false: "EM always finds the global maximum likelihood estimate".
- **Figures**
  - `fund.gmm-em/em-iterations`: GMM ellipses on 2-D data at iterations 0, 1, 5 and 20.
  - `fund.gmm-em/elbo-picture`: log-likelihood curve with the ELBO lower bound touching at θ_old and the M-step jump. **[fig-Q]**: "which curve is the ELBO?"
  - `fund.gmm-em/gmm-vs-kmeans`: anisotropic clusters where k-means fails and a full-covariance GMM succeeds.
- **Pitfalls & source disagreements**
  - "E-step computes the expected log-likelihood" (ESL/CS229) vs "E-step sets q to the posterior" (Bishop's decomposition). Same algorithm, different emphasis.
  - Bishop's 𝓛 is the ELBO. Other texts write F (free energy, negative) or Q(θ, θ_old).

---

#### 360 · `fund.pca`: Principal Component Analysis
- **Level** core · **W1** · **prereqs** [fund.variance-covariance, fund.gaussian] · **covers** dimensionality reduction.
- **Sources**
  - Bishop PRML §12.1 PCA: §12.1.1 maximum-variance formulation, §12.1.2 minimum-error formulation, §12.1.3 applications, §12.1.4 high-dimensional data (pdf p.581–590); §12.2 probabilistic PCA incl. §12.2.1 ML solution, §12.2.2 EM, §12.2.4 factor analysis (pdf p.590–606). `bishop.txt`
  - Goodfellow DL §2.12 example: PCA derived via an optimal linear decoder, §2.7–2.9 eigendecomposition, SVD, pseudo-inverse. `dlb_ch02_linear_algebra.txt`
  - ESL §14.5.1 principal components (pdf p.553–560), §3.5.1 principal components regression (pdf p.98). `esl.txt`
  - Murphy PML1 §20.1 PCA: examples, derivation, computational issues, choosing K (pdf p.687–696), §20.2.2 PPCA (pdf p.698). Shlens 2014 (`papers/shlens2014_pca_tutorial.txt`). UML (SSBD) §23.1 (p.325).
- **Subtopic map**
  1. *Goal*: find a low-dimensional linear subspace that keeps "most of the information". Two definitions, max variance and min reconstruction error, shown to coincide.
  2. *Max-variance derivation*: variance of the projection $u^\top x$ is $u^\top Su$. Maximize s.t. ‖u‖ = 1. The Lagrangian gives $Su=\lambda u$, and the variance captured equals λ, so take the top eigenvector. Subsequent components by orthogonality/induction.
  3. *Min-error derivation*: reconstruct with M orthonormal directions. The error is $\sum_{i>M}\lambda_i$ (derive). Equivalent to max variance. DLB's encoder/decoder derivation as a third route.
  4. *Computing it*:
     - Centre the data (why: otherwise PC1 points toward the mean).
     - Derive the link: with centred $X=U\Sigma V^\top$, $X^\top X=V\Sigma^2V^\top$, so the right singular vectors are the covariance eigenvectors, $\lambda_i=\sigma_i^2/(N-1)$, and the scores are UΣ.
     - Prefer the SVD numerically: forming $X^\top X$ squares the condition number (κ(X) = 10⁴ gives κ(XᵀX) = 10⁸, beyond float32's ~10⁷ precision). Costs: covariance route O(Nd² + d³), thin SVD O(Nd·min(N, d)).
     - *Inline refreshers*: the spectral theorem for symmetric matrices, and Lagrange multipliers.
     - The N < d trick: eigendecompose $XX^\top$.
     - Randomized/truncated SVD for scale (named).
  5. *Choosing M*: explained-variance ratio, scree plot, CV of reconstruction, downstream performance.
  6. *Scaling*: PCA isn't scale invariant. Use correlation-matrix PCA (standardize first) when units differ. Worked example of a large-unit feature hijacking PC1.
  7. *Interpretation and uses*: loadings, visualization, compression, denoising, decorrelation (pointer to fund.whitening), PCR (ESL §3.5.1; links ridge's SVD shrinkage to PCA's hard truncation).
  8. *Limitations*: linear only, sensitive to outliers (robust PCA named), maximal variance ≠ discriminative (LDA contrast), sign/rotation ambiguity of eigenvectors with equal eigenvalues.
  9. *Probabilistic PCA*:
     - Model $x=Wz+\mu+\varepsilon$ with $z\sim N(0,I)$ and $\varepsilon\sim N(0,\sigma^2I)$.
     - Marginal $N(\mu,WW^\top+\sigma^2I)$.
     - ML solution $W=U_M(\Lambda_M-\sigma^2I)^{1/2}R$, with $\hat\sigma^2$ = mean of the discarded eigenvalues (statement plus interpretation).
     - Rotational non-identifiability. EM for PCA.
     - Factor analysis (diagonal Ψ) and how it differs.
  10. *Link to autoencoders*: a linear AE with MSE spans the PCA subspace (pointer to fund.autoencoders).
- **Question ideas**
  - (calc) Eigenvalues [4, 2, 1, 1]: the first two PCs explain 75%. Reconstruction MSE with M=2 is 2 (the sum of the discarded eigenvalues).
  - What goes wrong with SVD-PCA without centring?
  - Why compute PCA by an SVD of X rather than an eigendecomposition of XᵀX? (stability: the condition number is squared)
  - Which is false: "PCA directions are invariant to feature scaling".
  - Predict: PC1 when one feature is measured in grams instead of kilograms.
  - Derivation step: why is the captured variance equal to the eigenvalue?
  - PPCA: what is σ̂²? (the average of the discarded eigenvalues)
- **Figures**
  - `fund.pca/pca-2d`: a correlated 2-D Gaussian with PC arrows scaled by √λ and projections onto PC1. Notice the residuals ⟂ PC1.
  - `fund.pca/scree`: explained variance per component with the cumulative curve. **[fig-Q]**: "how many components for 90%?"
  - `fund.pca/scaling-hijack`: PC1 before and after standardizing a large-unit feature.
- **Pitfalls & source disagreements**
  - Covariance 1/N vs 1/(N−1). Eigenvector sign indeterminacy.
  - ESL writes $X=UDV^\top$. Bishop uses M (latent) and D (data) for dimensions.

---

#### 370 · `fund.whitening`: Standardization & Whitening
- **Level** intermediate · **W2** · **prereqs** [fund.pca] · **covers** data whitening.
- **Sources**
  - Kessy, Lewin & Strimmer 2018 (`papers/kessy2018_whitening.txt`): the whitening family $W=Q\Sigma^{-1/2}$, ZCA, PCA, Cholesky, and the ZCA-cor/PCA-cor optimality criteria.
  - Bishop PRML §12.1.3 applications: standardization vs whitening/sphering (pdf p.585–589). `bishop.txt`
  - Murphy PML1 §10.2.8 standardization (pdf p.380), §7.4 EVD incl. whitening (pdf p.283–289). `pml1.txt`
  - ESL §11.5.3 scaling of the inputs for NNs (pdf p.417), §14.7.2 ICA & whitening as preprocessing (pdf p.579). `esl.txt`
  - Ioffe & Szegedy 2015 §2 (whitening motivation for BN; `papers/ioffe2015_batchnorm.txt`).
- **Subtopic map**
  1. *Why preprocess inputs*: GD conditioning (pointer to fund.gradient-descent), scale-sensitive methods (kNN, SVM-RBF, PCA, k-means, L2 penalties), and equal treatment of features.
  2. *Standardization* (z-scoring per feature) vs min-max scaling vs robust scaling (median/IQR). Fit on training data only (leakage; pointer to fund.model-selection).
  3. *Whitening*: find W with $\mathrm{Cov}(Wx)=I$. Show that all solutions are $W=Q\Sigma^{-1/2}$ with Q orthogonal (derive: $W\Sigma W^\top=I$).
  4. *Three canonical choices*:
     - PCA whitening $\Lambda^{-1/2}U^\top$: rotate to the eigenbasis, then scale.
     - ZCA/Mahalanobis whitening $U\Lambda^{-1/2}U^\top=\Sigma^{-1/2}$: rotate back. It is the minimum-distortion whitening (closest to the original in L2) and preserves the image-like appearance of whitened images.
     - Cholesky whitening: $L^{-1}$ with $\Sigma=LL^\top$ (lower-triangular; order dependent).
  5. *Numerical regularization*: $\Sigma+\epsilon I$ before the inverse square root, because small-eigenvalue directions are mostly noise and get amplified. Choosing ε. PCA-whitening with truncation (keep the top k).
  6. *Whitening ⇒ uncorrelated, not independent* (unless Gaussian). ICA as "whitening + a rotation to maximize non-Gaussianity" (ESL §14.7.2, named).
  7. *Whitening inside networks*: BN as per-feature standardization (not full whitening, for cost and stability reasons; Ioffe & Szegedy §2). Decorrelated BN (named).
  8. *Effects on learning*: whitened inputs make the linear-regression Hessian ∝ I, so GD converges in one well-chosen step. Why full whitening of images can amplify noise and hurt generalization (caveat; state as a claim).
- **Question ideas**
  - (calc) Σ = diag(4, 1): PCA whitening W = diag(0.5, 1). Whitened variance = 1 in both coordinates.
  - (calc) Σ = [[2,1],[1,2]] has eigenvalues 3 and 1. The ZCA matrix maps the eigenvector (1,1)/√2 to itself scaled by 1/√3.
  - PCA vs ZCA whitening: which one is closest to the original data? (ZCA)
  - Which is false: "whitened features are statistically independent".
  - Spot the flaw: fitting the whitening transform on train+test.
  - Predict: whitening with ε = 0 when Σ has a near-zero eigenvalue (noise blow-up).
- **Figures**
  - `fund.whitening/whitening-panels`: original vs PCA-whitened vs ZCA-whitened scatter. **[fig-Q]**: "which is ZCA?"
  - `fund.whitening/conditioning`: GD paths on unscaled vs standardized vs whitened least squares.
  - `fund.whitening/epsilon-effect`: whitened data with tiny vs moderate ε, showing noise amplification.
- **Pitfalls & source disagreements**
  - "Sphering" = whitening (Bishop). "Standardization" is sometimes loosely called whitening.
  - ZCA is sometimes called "Mahalanobis whitening".

---

#### 380 · `fund.nonlinear-dim-reduction`: Kernel PCA, t-SNE, UMAP & Random Projections
- **Level** intermediate · **W3** · **prereqs** [fund.pca, fund.curse-of-dimensionality] · **covers** dimensionality reduction.
- **Sources**
  - Murphy PML1 §20.4 manifold learning: the manifold hypothesis, MDS, Isomap, kernel PCA, MVU, LLE, Laplacian eigenmaps, t-SNE (pdf p.719–735). `pml1.txt`
  - Bishop PRML §12.3 kernel PCA (pdf p.606), §12.4 nonlinear latent-variable models (pdf p.611–619). ESL §14.5.4 kernel PCA (pdf p.566), §14.8 MDS (pdf p.589), §14.9 nonlinear DR & local MDS (pdf p.591). `bishop.txt`, `esl.txt`
  - van der Maaten & Hinton 2008 (`papers/vandermaaten2008_tsne.txt`): SNE → t-SNE, the crowding problem, perplexity, gradient. McInnes, Healy & Melville 2018 (`papers/mcinnes2018_umap.txt`).
  - Wattenberg et al. 2016, Distill (`web/distill2016_tsne.txt`): misreadings of t-SNE. UML (SSBD) §23.2 random projections & JL (p.329).
- **Subtopic map**
  1. *Manifold hypothesis*: data lie near a low-dimensional curved manifold, which linear PCA can't unfold (Swiss roll).
  2. *Classical MDS*: from a distance matrix via double centring. For Euclidean distances it equals PCA (sketch).
  3. *Kernel PCA*: PCA in feature space via the Gram matrix. Centring in feature space $\tilde K=K-1K-K1+1K1$. Out-of-sample projection. Pre-image problem (named).
  4. *Isomap* (geodesic distances via a kNN graph, then MDS), LLE, Laplacian eigenmaps: one line each with the key idea and the failure mode (short-circuiting).
  5. *t-SNE*:
     - High-d conditional Gaussian affinities, with $\sigma_i$ set by **perplexity** (binary search).
     - Symmetrized P.
     - Low-d Student-t (1 dof) affinities Q. The heavy tail solves the crowding problem.
     - Minimize KL(P‖Q) by gradient descent.
     - KL(P‖Q) penalizes placing true neighbours far apart, so local structure is preserved and global structure isn't.
  6. *Reading t-SNE correctly*: cluster sizes and inter-cluster distances are not meaningful, perplexity changes the picture, random seeds matter, and noise can produce apparent clusters (Distill).
  7. *UMAP*: fuzzy kNN graph plus a cross-entropy objective with attractive and repulsive terms. Faster. Claims about global-structure preservation are debated. Its main parameters (n_neighbors, min_dist).
  8. *Random projections & JL*: $k=O(\log N/\varepsilon^2)$ Gaussian random directions preserve all pairwise distances within 1±ε with high probability (statement; why it's a blessing of concentration). Data-independent and cheap.
  9. *Choosing a method*: PCA (fast, linear, interpretable) vs autoencoders (learned, nonlinear; pointer to fund.autoencoders) vs t-SNE/UMAP (visualization only) vs JL (fast distance preservation).
- **Question ideas**
  - Can you read inter-cluster distances off a t-SNE plot? (no)
  - Why a Student-t in the low-dimensional space? (crowding problem)
  - (calc) JL: the target dimension doesn't depend on the original d. N=10⁶ and ε=0.1 give k ∝ ln(10⁶)/0.01 ≈ 1,382 × the constant.
  - Which is false: "t-SNE preserves the relative distances between well-separated clusters". True distractors: classical MDS on Euclidean distances recovers the PCA embedding; the JL target dimension doesn't depend on d; kernel PCA centres the Gram matrix in feature space.
  - **[fig-Q]** The same data under three perplexities: which is perplexity 2?
- **Figures**
  - `fund.nonlinear-dim-reduction/swiss-roll`: the Swiss roll and its PCA vs Isomap embeddings.
  - `fund.nonlinear-dim-reduction/tsne-perplexity`: t-SNE at perplexity 2, 30 and 100. **[fig-Q]**
  - `fund.nonlinear-dim-reduction/t-vs-gaussian`: Gaussian vs Student-t kernels vs distance (heavier tail).
- **Pitfalls & source disagreements**
  - UMAP's global-structure claims are contested.
  - t-SNE "perplexity" is roughly the effective number of neighbours, not a model perplexity.

---
### Optimization algorithms

---

#### 410 · `fund.sgd-momentum`: SGD, Momentum & Learning-Rate Schedules
- **Level** core · **W1** · **prereqs** [fund.gradient-descent] · **covers** SGD, training loops (theory).
- **Sources**
  - Goodfellow DL §8.1.3 batch & minibatch algorithms (variance of the gradient estimate, batch-size effects), §8.3.1 SGD with its sufficient conditions on the lr, §8.3.2 momentum, §8.3.3 Nesterov. `dlb_ch08_optimization.txt`
  - Bottou, Curtis & Nocedal 2018 §4 (SGD analysis: fixed vs diminishing steps, noise ball) and §5 (noise reduction: minibatching, variance reduction) (`papers/bottou2018_large_scale_opt.txt`).
  - Murphy PML1 §8.4 SGD: finite sums, §8.4.3 step size, §8.4.4 iterate averaging, §8.4.5 variance reduction (pdf p.322–328), §8.2.4 momentum & Nesterov (pdf p.317). `pml1.txt`
  - Sutskever et al. 2013 (`papers/sutskever2013_momentum.txt`): Nesterov as look-ahead momentum. Goyal et al. 2017 (`papers/goyal2017_large_minibatch.txt`): linear scaling, warmup. Loshchilov & Hutter SGDR (`tp_loshchilov2017_sgdr.txt`). Shallue et al. 2019 batch size (`tp_shallue2019_batch_size.txt`).
  - d2l sgd, minibatch-sgd, momentum, lr-scheduler. UML (SSBD) §14.3–14.5 SGD analysis (p.191–201).
- **Subtopic map**
  1. *From GD to SGD*: the minibatch gradient is an unbiased estimator of the full gradient (prove), with variance ∝ 1/B. Cost per step vs per epoch. Why SGD is the default for large N.
  2. *Convergence with noise*:
     - Constant η converges to a noise ball of radius ∝ ησ²/μ (Bottou Thm 4.6-style statement).
     - Robbins–Monro conditions Σ η_t = ∞, Σ η_t² < ∞.
     - Rates: O(1/√T) convex, O(1/T) strongly convex.
     - Iterate averaging (Polyak).
     - Tuning-plan insertions: the 1-D stationary excess loss is ≈ ησ²/(4B), so halving η ≈ doubling B (→ link sys.tuning-steps-schedules, which derives it); the playbook's remedies for noisy end-of-training metrics are a larger batch, LR decay and Polyak averaging.
  3. *Minibatch size*: gradient variance vs hardware parallelism. Diminishing returns past the "critical batch size" (Shallue et al.; `llm_mccandlish2018_critical_batch.txt`). Generalization effects of noise (named, contested). The **linear scaling rule** (lr ∝ B) and why it breaks at large B. Keep it short and → link sys.tuning-batch-size for the McCandlish derivation and Shallue's finding that measured optimal LRs follow no simple rule (Tuning-plan insertion).
  4. *Momentum (heavy ball)*:
     - Update $v\leftarrow\beta v-\eta g$, $x\leftarrow x+v$.
     - It is an exponential moving average of gradients, with effective step η/(1−β) in consistent directions. Shallue §2.2 uses this effective LR as the quantity to compare across batch sizes (Tuning-plan insertion).
     - It damps oscillation across ravines.
     - On quadratics, optimal β gives the rate $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ (statement, with a calc comparison to GD).
  5. *Nesterov momentum*: gradient at the look-ahead point $x+\beta v$. Sutskever's formulation. Why it's more stable. The PyTorch implementation differs in form but is equivalent.
  6. *Learning-rate schedules*:
     - Step decay, exponential, 1/t.
     - Cosine annealing (with restarts, SGDR).
     - Linear warmup and why it helps: early instability, large batches, Adam's v estimates (pointer to fund.adam).
     - Warmup-stable-decay (named; `llm_hagele2024_wsd.txt`).
     - The lr range test.
     - The playbook's position: linear decay or cosine as the default; which family is best is open (→ link sys.tuning-steps-schedules; Tuning-plan insertion).
  7. *Interaction between lr, batch size, momentum and weight decay*: the "effective learning rate" view. Smith et al. "don't decay the lr, increase the batch size" (`tp_smith2018_dont_decay_lr.txt`, statement).
  8. *Gradient clipping* (by global norm) for stability (pointer to fund.rnn). Gradient accumulation to simulate large batches.
  9. *Variance reduction* (SVRG/SAG, named) and why it's rarely used in DL.
- **Question ideas**
  - (calc) Linear scaling: batch 256 at lr 0.1 → batch 8192 at lr 3.2 (with warmup).
  - (calc) Momentum β=0.9: effective step multiplier 10 in a flat consistent direction.
  - (calc) κ=100: GD rate 0.980 vs heavy-ball rate 9/11 ≈ 0.818 per step.
  - Predict: constant-lr SGD on a strongly convex problem (a stationary noise ball).
  - Which is false: "doubling the batch size always halves the number of steps needed".
  - Why warmup? (the most defensible answer among distractors)
- **Figures**
  - `fund.sgd-momentum/sgd-noise-ball`: SGD iterates with constant vs decaying lr on a 2-D quadratic.
  - `fund.sgd-momentum/momentum-ravine`: GD vs heavy-ball vs Nesterov paths in a narrow valley.
  - `fund.sgd-momentum/schedules`: step, cosine, warmup+cosine and WSD schedules. **[fig-Q]**: "which is cosine with warmup?"
  - `fund.sgd-momentum/batch-size-steps`: steps-to-target vs batch size showing perfect scaling then diminishing returns.
- **Pitfalls & source disagreements**
  - Momentum conventions: PyTorch uses v ← μv + g, p ← p − lr·v. The classical form puts lr inside v. They differ when lr changes.
  - Nesterov formulations differ across Nesterov, Sutskever and PyTorch.
  - "SGD" in DL papers means minibatch SGD, often with momentum.

---

#### 420 · `fund.adaptive-lr`: AdaGrad, RMSProp & Preconditioning
- **Level** core · **W2** · **prereqs** [fund.sgd-momentum] · **covers** Adagrad (+ RMSProp).
- **Sources**
  - Duchi, Hazan & Singer 2011 (`papers/duchi2011_adagrad.txt`): the diagonal and full-matrix algorithms, the regret bound, and the sparse-feature motivation.
  - Goodfellow DL §8.5 adaptive learning rates, §8.5.1 AdaGrad, §8.5.2 RMSProp (incl. Nesterov variant). `dlb_ch08_optimization.txt`
  - Murphy PML1 §8.4.6 preconditioned SGD: AdaGrad, RMSProp, Adadelta, Adam (pdf p.328–331). `pml1.txt`
  - d2l adagrad & rmsprop (`d2l/d2l_adagrad.txt`, `d2l/d2l_rmsprop.txt`). Ruder 2016 §4.3–4.5 (`papers/ruder2016_gd_overview.txt`).
- **Subtopic map**
  1. *Motivation*: one global lr is wrong when coordinates have very different curvature or gradient frequency (sparse features, embeddings). Idea: per-coordinate step sizes, i.e. a diagonal preconditioner.
  2. *AdaGrad*: $G_t=\sum_{s\le t}g_s^2$ (elementwise), $x\leftarrow x-\frac\eta{\sqrt{G_t}+\epsilon}g_t$.
     - Rare features keep large steps. Frequent ones shrink.
     - Full-matrix version $G^{-1/2}$ (named; Shampoo as a structured approximation).
     - Regret O(√T) in online convex optimization (statement).
  3. *AdaGrad's flaw in deep learning*: G grows monotonically, so the effective lr decays like $1/\sqrt t$ even when that's unwanted (calc). It stalls in non-convex training.
  4. *RMSProp*: replace the sum with an EMA $v_t=\rho v_{t-1}+(1-\rho)g_t^2$, which forgets old gradients. Typical ρ = 0.9–0.99. Adadelta (unit-matching argument, named).
  5. *Interpretation*:
     - $\sqrt v$ estimates the RMS gradient scale per coordinate, so updates are roughly scale-free.
     - Relation to diagonal curvature is heuristic: it's the square root of the gradient second moment, not the Hessian.
     - Sign-SGD as the limit ($v\approx g^2$ gives a step ≈ η·sign(g)).
  6. *ε's role*: numerical safety, and it sets a floor that interpolates toward SGD for tiny gradients. Precision-dependent choices.
  7. *Preconditioning view*: all of these are $x\leftarrow x-\eta P_t^{-1}g$ with diagonal P. Pointer to second-order methods (fund.second-order) and Adam (next lesson). Preconditioned optimizers (Adam, K-FAC) keep scaling perfectly to larger batch sizes than momentum SGD (Zhang et al. 2019, `tp_zhang2019_batch_size_nqm.txt`; Tuning-plan insertion).
- **Question ideas**
  - (calc) AdaGrad with a constant gradient g: the step after T updates is η/(|g|√T)·|g| = η/√T (direction sign g).
  - (calc) RMSProp with ρ=0.9 under a constant gradient g: $v_t=(1-0.9^t)g^2$, so the step ≈ η/√(1−0.9^t), which → η.
  - Predict: AdaGrad's effective lr late in a long run (→ 0).
  - Which coordinate gets the larger effective lr under AdaGrad: a frequently or a rarely updated embedding row? (rarely)
  - Which is false: "RMSProp's denominator approximates the diagonal of the Hessian".
- **Figures**
  - `fund.adaptive-lr/effective-lr`: effective lr vs step for AdaGrad vs RMSProp under a constant gradient. **[fig-Q]**: "which keeps decaying?"
  - `fund.adaptive-lr/scaled-quadratic`: SGD vs AdaGrad vs RMSProp on an axis-aligned badly scaled quadratic.
- **Pitfalls & source disagreements**
  - ε inside vs outside the square root differs across papers and frameworks.
  - RMSProp has no paper (Hinton's lecture slides). DLB and d2l present slightly different variants.

---

#### 430 · `fund.adam`: Adam & AdamW *(keeps id)*
- **Level** core · **W1** · **prereqs** [fund.adaptive-lr, fund.regularization] · **covers** Adam/AdamW.
- **Sources**
  - Kingma & Ba 2015 (`papers/kingma2014_adam.txt`): Alg. 1, §2.1 bounds on the effective step, §3 bias-correction derivation, §4 regret, §7 AdaMax.
  - Loshchilov & Hutter 2019 (`papers/loshchilov2017_adamw.txt`): Alg. 2, Prop. 1–3 (L2 ≠ WD for adaptive methods; WD = L2 for SGD up to rescaling), schedule multiplier, normalized WD.
  - Reddi, Kale & Kumar 2018 (`papers/reddi2018_amsgrad.txt`): the non-convergence counterexample and AMSGrad.
  - Goodfellow DL §8.5.3 Adam, §8.5.4 choosing an optimizer. `dlb_ch08_optimization.txt`
  - Wilson et al. 2017 (`papers/wilson2017_adaptive_marginal.txt`). d2l adam (`d2l/d2l_adam.txt`). Wortsman et al. 2023 small-scale instabilities (`sys_wortsman2023_small_scale_instabilities.txt`) for ε/β2 in practice.
- **Subtopic map**
  1. *Algorithm*:
     - $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$.
     - $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$.
     - Bias-corrected $\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$.
     - Step $\eta\hat m/(\sqrt{\hat v}+\epsilon)$.
     - Defaults (1e-3, 0.9, 0.999, 1e-8). Adam = momentum + RMSProp + bias correction.
     - NAdam: Nesterov look-ahead in the numerator (write the update; the Tuning Playbook prefers it).
  2. *Bias correction derivation*: unroll $v_t=(1-\beta_2)\sum\beta_2^{t-s}g_s^2$. Under stationarity $E[v_t]=(1-\beta_2^t)E[g^2]$. Without correction, the early steps are mis-scaled (calc: the first step is ×3.16 with the defaults; it is *larger*, because v is more biased than m).
  3. *Effective step size and invariances*: |Δ| ≲ η in typical cases (Kingma §2.1). Invariant to diagonal rescaling of gradients (e.g. multiplying the loss by c), ignoring ε. The first bias-corrected step = η·sign(g).
  4. *Hyperparameters in practice*:
     - β2 = 0.95 or 0.98 for large transformers (stability; faster adaptation to gradient-scale changes).
     - ε choice under bf16/fp16 and its interplay with tiny gradients.
     - Warmup to tame noisy early $\hat v$ (RAdam, named).
     - Memory: 2 extra fp32 states per parameter (calc for 7B params: 56 GB of optimizer state).
     - ε is coupled to the LR (Tuning-plan insertion): for $\epsilon\gg\sqrt{\hat v}$ the step is ≈ $(\eta/\epsilon)\hat m$, i.e. momentum SGD with LR η/ε (one-line derivation), so search (ε, η/ε) jointly. Choi et al. 2019 (`tp_choi2019_optimizer_comparisons.txt`): Adam contains momentum as ε → ∞, and optimizer rankings flip with the tuning protocol; all four Adam hyperparameters can matter. Gradient RMS ≈ ε silently shrinks updates (Wortsman §3.4).
  5. *L2 regularization vs weight decay*: with Adam+L2, the penalty gradient λθ is divided by $\sqrt{\hat v}$, so parameters with large gradient history are barely regularized. **AdamW** decouples: $\theta\leftarrow\theta-\eta_t(\alpha\,\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta)$. For SGD, L2 and WD coincide with λ′ = λ/α (Prop. 1).
  6. *What AdamW changes in practice*: decoupled tuning of lr and wd, better generalization in vision and transformers. Weight decay usually excludes norms and biases.
  7. *Convergence caveats*: Reddi's counterexample (rare large gradients get forgotten as v shrinks). AMSGrad uses $\max\hat v$. In practice it rarely matters.
  8. *Generalization debate*: Wilson et al. (adaptive methods can generalize worse on some vision tasks) vs Adam(W)'s dominance for transformers. Sign-descent interpretations. Successors named: Lion, Adafactor (factored second moments to save memory), Shampoo, SOAP, Muon (`llm_jordan2024_muon.txt`).
- **Question ideas**
  - (calc) First bias-corrected step for g = 0.001 and g = 1000: ≈η in both cases.
  - (calc) Without bias correction (β1=0.9, β2=0.999), the first step is $\frac{1-\beta_1}{\sqrt{1-\beta_2}}\eta\approx3.16\eta$.
  - Which is false: "AdamW and Adam+L2 are equivalent".
  - Multiply the loss by 100: what changes for SGD vs Adam? (SGD steps ×100; Adam ≈ unchanged)
  - (calc) PyTorch AdamW with lr 1e-3 and weight_decay 0.1 → per-step decay factor 1 − 1e-4.
  - (calc) Adam state memory for 7B params in fp32 (m+v): 56 GB.
  - Flashcard (Tuning-plan insertion): how to tune Adam by trial budget (<10 trials: LR only; 10–25: + β₁; 25+: + ε; ≫25: + β₂; tune the β's on a log(1−β) scale). → link sys.tuning-pipeline.
- **Figures**
  - `fund.adam/bias-correction`: raw vs corrected m_t and v_t over the first 50 steps (constant gradient).
  - `fund.adam/optimizer-race`: SGD, momentum, RMSProp and Adam on a badly scaled quadratic or Rosenbrock.
  - `fund.adam/l2-vs-wd`: effective decay per weight under Adam+L2 vs AdamW for different gradient scales. **[fig-Q]**
- **Pitfalls & source disagreements**
  - The AdamW paper scales decay by the schedule multiplier $\eta_t$ (not α). PyTorch multiplies λ by lr, so the same "weight_decay" number means different effective decay.
  - ε placement in Kingma's "efficient version" and TF's ε̂.
  - DLB writes ρ1, ρ2 and δ.

---

#### 440 · `fund.newton`: Newton's Method
- **Level** intermediate · **W2** · **prereqs** [fund.gradient-descent] · **covers** Newton's method, second-order methods.
- **Sources**
  - Boyd & Vandenberghe §9.5 Newton's method: §9.5.1 the Newton step (three interpretations, affine invariance, the Newton decrement; pdf p.498–501), §9.5.2 Newton's method with backtracking (pdf p.501), §9.5.3 convergence analysis (damped and quadratic phases; pdf p.502), §9.5.4 examples (pdf p.506–510); §9.6 self-concordance (pdf p.510). `boyd.txt`
  - Goodfellow DL §4.3.1 (Newton step from the Taylor model), §8.6.1 Newton's method (saddle-point attraction, regularized Newton). `dlb_ch04_numerical.txt`, `dlb_ch08_optimization.txt`
  - Murphy PML1 §8.3.1 Newton (pdf p.319), §10.2.6 IRLS (pdf p.376). Bishop PRML §4.3.3 IRLS (pdf p.227), §5.4.2 outer-product approximation (pdf p.271). `pml1.txt`, `bishop.txt`
  - CS229 notes ch.2 "Another algorithm for maximizing ℓ(θ)" (Newton for logistic regression; pdf p.29). Dauphin et al. 2014 (`papers/dauphin2014_saddle_free.txt`).
- **Subtopic map**
  1. *Derivation* (inline refresher: the multivariate second-order Taylor expansion): minimize the second-order Taylor model, giving $\Delta=-H^{-1}\nabla f$. Also the root-finding view (Newton–Raphson on ∇f = 0), with a 1-D picture.
  2. *Quadratics*: exact in one step. **Affine invariance**: Newton's iterates are unchanged by linear reparameterization, unlike GD (show $x=Ay$).
  3. *Convergence*: local quadratic convergence (error squares; digits double) near a strongly convex minimum with Lipschitz Hessian (statement plus a sketch). Boyd's damped phase vs pure Newton phase. The Newton decrement as a stopping criterion.
  4. *Damped Newton / line search* for global convergence. Worked 1-D example, e.g. $f(x)=e^x-2x$ from x=0: iterates 0 → ln 2 region. Compute two steps.
  5. *When quadratic convergence fails*: a singular Hessian at the optimum, e.g. $f=x^4$ gives $x\leftarrow\frac23x$, which is linear (calc).
  6. *Non-convex failure*: an indefinite Hessian makes Newton attracted to saddles and maxima (it seeks any stationary point). Fixes: Levenberg–Marquardt damping $(H+\lambda I)^{-1}$ interpolating GD↔Newton, trust regions, saddle-free Newton |H| (Dauphin).
  7. *Cost*: O(d²) memory and O(d³) per solve, infeasible for DL. Hessian-vector products in O(d) via double backprop (Pearlmutter), which leads to Hessian-free methods (next lesson).
  8. *IRLS = Newton for logistic regression*: derive $\theta\leftarrow(X^\top SX)^{-1}X^\top Sz$ with working response $z=X\theta+S^{-1}(y-p)$. Weighted least squares each step. Fisher scoring for GLMs (named).
  9. *Gauss–Newton for least squares*: $H\approx J^\top J$ (drop the residual-curvature term), PSD. Levenberg–Marquardt (pointer to fund.second-order).
- **Question ideas**
  - (calc) One Newton step on $x^2-4x$ from any start lands at x = 2.
  - (calc) Newton on $x^4$ from x = 1: iterates $(2/3)^k$, which is linear. Why?
  - (calc) $f(x)=e^x-2x$, $x_0=0$: $x_1=0-\frac{1-2}{1}=1$, $x_2=1-\frac{e-2}{e}\approx0.736$ → ln 2 ≈ 0.693.
  - Predict: Newton at a point of negative curvature (moves toward a max or saddle).
  - (calc) Dense Hessian memory at $d=10^8$: ~$10^{16}$ entries (40 PB in fp32).
  - IRLS weights for logistic regression? (p(1−p))
- **Figures**
  - `fund.newton/newton-1d`: the quadratic model at x_k and the jump to its minimum, plus a negative-curvature case moving uphill.
  - `fund.newton/newton-vs-gd`: contour paths of GD vs Newton on an ill-conditioned quadratic.
  - `fund.newton/convergence-rates`: log error vs iteration for GD (linear), Newton (quadratic) and Newton on x⁴ (linear). **[fig-Q]**
- **Pitfalls & source disagreements**
  - Root-finding Newton vs optimization Newton get conflated (applying it to f instead of f′).
  - DLB emphasizes saddle attraction. Boyd assumes convexity throughout.

---

#### 450 · `fund.second-order`: Quasi-Newton, Gauss–Newton & Natural Gradient
- **Level** advanced · **W3** · **prereqs** [fund.newton, fund.kl-divergence, fund.fisher-information] · **covers** second-order methods.
- **Sources**
  - Goodfellow DL §8.6.2 conjugate gradients (incl. the nonlinear CG caveats), §8.6.3 BFGS & L-BFGS. `dlb_ch08_optimization.txt`
  - Murphy PML1 §8.3.2 BFGS & quasi-Newton, §8.3.3 trust region (pdf p.320–322). Murphy PML2 §6.4 natural gradient descent: definition, interpretations, approximations, exponential family (pdf p.313–320); §5.1.9 KL ≈ Fisher quadratic (pdf p.267). `pml1.txt`, `pml2.txt`
  - Bottou, Curtis & Nocedal 2018 §6 second-order methods: Hessian-free Newton, stochastic quasi-Newton, Gauss–Newton, natural gradient, diagonal scaling (`papers/bottou2018_large_scale_opt.txt`).
  - Martens 2014/2020 (`papers/martens2014_natural_gradient.txt`): Fisher vs GGN equivalence, the empirical-Fisher caveat, damping. Martens & Grosse 2015 K-FAC (`papers/martens2015_kfac.txt`). Kunstner, Balles & Hennig 2019 (`papers/kunstner2019_empirical_fisher.txt`).
  - Bishop PRML §5.4 the Hessian: diagonal and outer-product approximations, inverse Hessian, finite differences, exact evaluation, fast Hessian-vector products (pdf p.269–276). `bishop.txt`
- **Subtopic map**
  1. *Why approximate curvature*: Newton's benefits (affine invariance, conditioning) without the O(d³) cost. Approaches: approximate H, approximate its inverse, or approximate the solve.
  2. *Hessian-vector products and CG*: Hv in O(d) via forward-over-reverse AD. Conjugate gradients solve Hx = −g in ≤ d iterations using only Hv products (conjugacy idea). Hessian-free optimization with damping (Martens 2010, named).
  3. *Quasi-Newton*: the secant condition $B_{k+1}s_k=y_k$. The BFGS rank-2 update (formula, and what it preserves: symmetry and PD under the curvature condition). L-BFGS stores m pairs (O(md) memory) and uses the two-loop recursion (named). It needs accurate gradients and a Wolfe line search, so it struggles with minibatch noise.
  4. *Gauss–Newton and GGN*: for $L=\frac12\|r(\theta)\|^2$, $H=J^\top J+\sum r_i\nabla^2r_i$. Drop the second term to get Gauss–Newton (PSD). Levenberg–Marquardt. The generalized Gauss–Newton $J^\top H_{\text{out}}J$ for neural nets with convex output losses.
  5. *Natural gradient*:
     - Steepest descent when distance is measured by KL between model distributions.
     - The local quadratic $\mathrm{KL}\approx\frac12\delta^\top F\delta$ gives $\Delta\propto-F^{-1}\nabla L$ (derive via a constrained step).
     - $F=E_{p_\theta}[\nabla\log p\,\nabla\log p^\top]=-E[\nabla^2\log p]$ (from fund.fisher-information).
     - Reparameterization invariance (to first order).
  6. *Fisher = GGN* for exponential-family outputs with the canonical link (Martens). **Empirical Fisher** (gradients at the *observed* labels) ≠ Fisher, and can mislead badly away from the optimum (Kunstner).
  7. *K-FAC*: per-layer block-diagonal Fisher with the Kronecker factorization $F_\ell\approx A_{\ell-1}\otimes G_\ell$, so the inverses are cheap. Damping. Practical speed-ups.
  8. *Modern practical preconditioners*: Shampoo, SOAP and Muon (orthogonalized updates, `llm_jordan2024_muon.txt`), Adam as a diagonal empirical-Fisher-like method (contested analogy).
  9. *When second-order wins*: small, deterministic, ill-conditioned problems (logistic regression, GLMs, small nets). Why it rarely wins in large-scale noisy DL.
- **Question ideas**
  - Which is the empirical Fisher? (outer products of per-example gradients at the observed labels)
  - (calc) L-BFGS memory with m=10, d=10⁸: 2·10·10⁸ = 2·10⁹ floats vs 10¹⁶ for the full Hessian.
  - Predict: L-BFGS with small noisy minibatches (unstable or poor curvature pairs).
  - Which is false: "natural-gradient steps of any finite size are exactly invariant to smooth reparameterizations of the model" (the invariance holds for the continuous-time flow, i.e. to first order in the step size).
  - Gauss–Newton: which Hessian term is dropped, and when is that accurate? (residual curvature; small residuals near a good fit)
- **Figures**
  - `fund.second-order/natural-gradient`: gradient vs natural-gradient arrows over the (μ, log σ) parameter space of a Gaussian.
  - `fund.second-order/empirical-vs-true-fisher`: preconditioned steps on a simple regression showing the empirical-Fisher pathology (after Kunstner Fig. 1). **[fig-Q]**
  - `fund.second-order/kfac-structure`: block-diagonal Fisher with Kronecker blocks.
- **Pitfalls & source disagreements**
  - "Fisher" is used loosely: many DL papers mean the empirical Fisher.
  - BFGS notation B (approximate Hessian) vs H (approximate inverse) differs by source.

---

### Deep learning

---

#### 460 · `fund.backprop`: Backpropagation & Automatic Differentiation
- **Level** core · **W1** · **prereqs** [] · **covers** backprop implementation.
- **Sources**
  - Goodfellow DL §6.5 back-propagation: §6.5.1 computational graphs, §6.5.2 chain rule, §6.5.3 recursively applying the chain rule, §6.5.5 symbol-to-symbol, §6.5.6 general back-prop (cost), §6.5.8 complications, §6.5.9 AD outside DL. `dlb_ch06_mlp.txt`
  - Baydin et al. 2018 (`papers/baydin2018_autodiff.txt`): forward vs reverse mode, dual numbers, VJP/JVP, cost.
  - Murphy PML1 §13.3.1 forward vs reverse mode, §13.3.3 VJPs for common layers, §13.3.4 computation graphs (pdf p.468–476). `pml1.txt`
  - Bishop PRML §5.3.1–5.3.4 (error backprop, simple example, efficiency, Jacobian; pdf p.262–269). `bishop.txt`
  - Chen et al. 2016 sublinear memory / gradient checkpointing (`sys_chen2016_sublinear_memory.txt`). d2l backprop (`d2l/d2l_backprop.txt`).
- **Subtopic map**
  1. *The problem*: we need ∂L/∂θ for millions of parameters. Finite differences cost O(#params) forward passes and are inexact. Symbolic differentiation suffers expression swell. AD is exact to machine precision at a cost of a few forward passes.
  2. *Computational graphs*: nodes are operations, edges are tensors. A forward pass computes values and caches what backward needs.
  3. *Chain rule, scalar → vector*: Jacobians, and the multivariate chain rule as a sum over paths. **Fan-out ⇒ gradients add** (shared weights, residual branches).
  4. *Reverse mode = backprop*: propagate adjoints $\bar v=\partial L/\partial v$ from the output backward. Each op needs only a **vector–Jacobian product** (VJP), never the full Jacobian. Work through a small scalar graph by hand (e.g. $L=(wx+b-y)^2$ with tanh), with numbers.
  5. *Forward mode*: propagate tangents via JVPs (dual numbers). Cost comparison: forward mode costs O(#inputs) passes for a full gradient and reverse mode O(#outputs). Since the loss is scalar, reverse mode wins. Jacobian-vector vs vector-Jacobian.
  6. *Common VJPs*:
     - Matmul: $\bar X=\bar YW^\top$, $\bar W=X^\top\bar Y$.
     - Elementwise ops.
     - Sum and broadcast (they are adjoints of each other).
     - Softmax+CE fused gives $p-y$.
     - Reshape/transpose.
     - Max (routes the gradient to the argmax).
  7. *Memory*: reverse mode stores activations, so memory ∝ depth × batch × width. **Gradient checkpointing** recomputes segments, trading compute for O(√L) memory (Chen et al.).
  8. *Gradient checking* (owned here; moved from fund.gradient-descent): centred differences $\frac{f(x+h)-f(x-h)}{2h}$ have O(h²) truncation error vs O(h) for forward differences (derive from Taylor), and h trades truncation against round-off (h ≈ 1e−5 in float64). Relative-error thresholds, float64, and avoiding kinks such as ReLU at 0.
  9. *Framework semantics*: define-by-run (PyTorch autograd) vs traced/compiled graphs (JAX `grad`/`jit`). `requires_grad`, `.detach()`, `no_grad`. Gradient accumulation (`.grad` adds, so zero it). Higher-order gradients (Hessian-vector products via double backprop).
  10. *Complications*: non-differentiable points (subgradients), in-place ops breaking autograd, numerical stability (log-sum-exp; `sys_blanchard2021_logsumexp.txt`).
- **Question ideas**
  - (calc) Hand backprop: $L=(\tanh(wx)-y)^2$ with w=0.5, x=2, y=0 → $\partial L/\partial w=2\tanh(1)(1-\tanh^2(1))\cdot2\approx1.28$.
  - A variable is used twice in the graph: how are its gradients combined? (summed)
  - Cost: full gradient of a scalar loss with $10^9$ params, reverse vs forward mode passes (≈1 vs 10⁹).
  - Which is false: "backprop computes and stores the full Jacobian of each layer".
  - Predict: memory with checkpointing every √L layers.
  - What does `.detach()` do to the gradient flow? (blocks it)
- **Figures**
  - `fund.backprop/graph-forward-backward`: a small computational graph with forward values above the edges and adjoints below.
  - `fund.backprop/fan-out-sum`: a node with two consumers, showing the adjoints summing.
  - `fund.backprop/forward-vs-reverse`: cost of forward vs reverse mode vs #inputs/#outputs. **[fig-Q]**
  - `fund.backprop/checkpointing`: memory vs compute for no checkpointing, √L checkpoints and full recompute.
- **Pitfalls & source disagreements**
  - "Backprop" sometimes means the whole training algorithm and sometimes just reverse-mode AD.
  - Gradient vs Jacobian layout conventions (numerator vs denominator layout) differ between textbooks.

---

#### 470 · `fund.mlp-from-scratch`: Implementing an MLP: Forward & Backward
- **Level** core · **W1** · **prereqs** [fund.backprop, fund.logistic-regression] · **covers** implementing MLP forward/backward, backprop implementation.
- **Sources**
  - CS229 notes ch.7: supervised learning with non-linear models, neural networks, modules, "Backward functions for basic modules" (pdf p.107), "Back-propagation for MLPs" (pdf p.110), "Vectorization over training examples" (pdf p.112) (pdf p.81–114 overall). `cs229.txt`
  - Goodfellow DL §6.1 learning XOR (worked solution), §6.4.1 universal approximation & depth, §6.5.4 backprop in fully connected MLPs (Algorithms 6.3–6.4), §6.5.7 MLP training example. `dlb_ch06_mlp.txt`
  - Murphy PML1 §13.2 MLPs incl. §13.2.1 XOR and §13.2.5 importance of depth (pdf p.456–466), §13.3.2 reverse mode for MLPs (pdf p.470). `pml1.txt`
  - Bishop PRML §5.1 feed-forward functions & weight-space symmetries (pdf p.247–252), §5.3.2 simple example (pdf p.265). d2l mlp & backprop.
- **Subtopic map**
  1. *Why hidden layers*: linear models can't do XOR. DLB's 2-unit ReLU solution (verify by hand). Composition of linear layers is linear without a nonlinearity.
  2. *Forward pass, vectorized*: batch-first convention $Z^{(l)}=A^{(l-1)}W^{(l)\top}+b^{(l)}$ (PyTorch layout) or column-vector convention $z=Wa+b$ (CS229/Bishop). **Shape bookkeeping** at every step. Cache Z and A for backward.
  3. *Output layer + loss*: logits → softmax-CE (fused) or sigmoid-BCE or MSE. Mean vs sum reduction and its effect on gradient scale.
  4. *Backward pass derivation* (vectorized, with shapes):
     - $\delta^{(L)}=\partial L/\partial Z^{(L)}=(P-Y)/B$ for mean CE.
     - $\partial L/\partial W^{(l)}=\delta^{(l)\top}A^{(l-1)}$.
     - $\partial L/\partial b^{(l)}=\sum_\text{batch}\delta^{(l)}$.
     - $\delta^{(l-1)}=(\delta^{(l)}W^{(l)})\odot\phi'(Z^{(l-1)})$.
     - Explain each from the VJP rules.
  5. *Writing it in numpy*: a ~30-line reference implementation outline (init, forward, loss, backward, update). Where bugs hide (transposes, broadcasting (N,) vs (N,1), forgetting the 1/B).
  6. *Verifying*: gradient check against finite differences on a tiny net. Compare with autograd.
  7. *Universal approximation & depth*: the one-hidden-layer universal approximation theorem (statement; what it doesn't promise: width, learnability). The depth-efficiency examples (DLB §6.4.1).
  8. *Weight-space symmetries*: permuting hidden units and sign flips for tanh give many equivalent minima (Bishop §5.1.1). Why the loss is non-convex and why symmetric init fails (pointer to fund.initialization).
  9. *Worked numeric example*: a 2-2-1 network with given weights. Compute the forward values, loss, and all gradients by hand. This is the canonical whiteboard exercise.
- **Question ideas**
  - Shapes: X (64×784), first layer W (256×784) (PyTorch layout) → shape of ∂L/∂W? (256×784, computed as δᵀX)
  - (calc) The 2-2-1 hand example: gradient of one specific weight.
  - Spot the bug: `dW = X.T @ delta` used with the PyTorch (out, in) layout, or the 1/B missing in a mean loss.
  - Which is false: "a one-hidden-layer network can represent any continuous function on a compact set with a fixed, small number of units".
  - (calc) Parameter count of a 784-256-128-10 MLP with biases: 200,960 + 32,896 + 1,290 = 235,146.
  - Predict: an MLP with identity activations on XOR (no better than linear).
- **Figures**
  - `fund.mlp-from-scratch/xor-solution`: XOR points in input space and in hidden-ReLU space (linearly separable).
  - `fund.mlp-from-scratch/shapes-diagram`: forward/backward data flow with tensor shapes annotated at each edge.
  - `fund.mlp-from-scratch/worked-2-2-1`: the 2-2-1 network with numbers on nodes (forward) and adjoints (backward). **[fig-Q]**
- **Pitfalls & source disagreements**
  - Weight layout: CS229/Bishop use W of shape (out, in) with column vectors. PyTorch stores (out, in) but computes $xW^\top$ on rows. DLB writes $h=g(W^\top x+c)$.
  - δ is defined as ∂L/∂z (most sources) or as ∂L/∂a (some notes).

---

#### 480 · `fund.training-loop`: The Training Loop: Implementation & Debugging
- **Level** core · **W1** · **prereqs** [fund.mlp-from-scratch, fund.sgd-momentum] · **covers** training loops with SGD.
- **Sources**
  - d2l linear-regression-scratch (data iterator, model, loss, SGD, the loop; `d2l/d2l_linear_regression_scratch.txt`), softmax regression.
  - Goodfellow DL §11.1–11.5 practical methodology: baselines, gather more data?, hyperparameters, **§11.5 debugging strategies** (visualize, fit a tiny dataset, compare back-prop to finite differences, monitor activations & gradients). `dlb_ch11_guidelines.txt`
  - Karpathy 2019 "A Recipe for Training Neural Networks" (`web/karpathy2019_training_recipe.txt`), secondary: overfit one batch, verify the loss at init, etc.
  - Goyal et al. 2017 (`papers/goyal2017_large_minibatch.txt`): warmup, the pitfalls list (weight decay scaling, momentum correction, data shuffling).
  - Google tuning playbook (`tp_tuning_playbook.md`); PyTorch AMP & gradient-clipping docs (`sys_pytorch_amp.txt`, `sys_pytorch_clip_grad_norm.txt`); HF gradient-accumulation fix (`sys_hf2024_gradient_accumulation_fix.txt`).
- **Subtopic map**
  1. *Anatomy of a loop*:
     - Data loader (shuffle each epoch, batching, drop_last).
     - Forward pass.
     - Loss.
     - `optimizer.zero_grad()`.
     - `loss.backward()`.
     - (Optional) gradient clipping.
     - `optimizer.step()`.
     - lr `scheduler.step()`.
     - Logging.
     - Explain why each line is where it is.
  2. *A minimal from-scratch SGD loop* (numpy or JAX) for linear or logistic regression, and the PyTorch equivalent. The manual parameter update under `no_grad`.
  3. *Train vs eval modes*: dropout and BN behaviour, `torch.no_grad()`/`inference_mode` for evaluation, and evaluating on a held-out split at fixed *step* intervals (not time intervals). Give zero weight to padded examples in a partial final eval batch, or the padding biases the metric (Tuning-plan insertion).
  4. *Sanity checks before tuning*:
     - The initial loss should be ≈ ln K for K balanced classes.
     - Overfit a single batch to ~0 loss.
     - Verify that shuffling labels destroys learning.
     - Check the data pipeline visually.
     - Check input normalization.
  5. *Common bugs* (each with its symptom):
     - Missing `zero_grad` (gradients accumulate; effective lr grows).
     - Softmax before `CrossEntropyLoss`.
     - Wrong reduction or loss averaging with padding.
     - Broadcasting (N,) vs (N,1) in MSE.
     - Forgetting `model.eval()`.
     - Labels misaligned after shuffling.
     - Data leakage between splits.
     - In-place ops.
     - lr too high (NaN).
     - Gradient accumulation without dividing the loss.
     - `optimizer` created before moving the model to the device / missing parameters.
     - Scheduler stepped per epoch vs per step.
  6. *Monitoring*: loss curves (train/val), gradient norms, update/parameter-norm ratios (~1e-3 heuristic), activation statistics, learning-rate traces. What instability looks like (loss spikes, NaNs) and the first fixes (lower lr, warmup, clipping, bf16 issues). Name the playbook's checks: training loss *rising* means a bug; periodic validation curves suggest train/val overlap or a shuffling bug; log the *unclipped* gradient norm (→ link sys.tuning-diagnostics; Tuning-plan insertion).
  7. *Mixed precision & clipping in the loop*: autocast + GradScaler for fp16 (bf16 needs no scaler). Clip after unscaling (named; pointer to the systems area).
  8. *Gradient accumulation*: divide the loss by the number of accumulation steps. The pitfall of token-count normalization with variable-length batches (HF fix).
  9. *Reproducibility*: seeds, deterministic flags, data order. Checkpointing (model + optimizer + scheduler + RNG state) and resuming.
  10. *Hyperparameter-tuning order* (Tuning Playbook; corrected in review): get a baseline. Don't tune the batch size against validation performance; choose it for throughput and memory. Then (re)tune the learning rate and the other optimizer hyperparameters, because the best LR depends on the batch size, then regularization. Start with a constant LR and add a schedule later (→ link sys.tuning-process, sys.tuning-batch-size).
- **Question ideas**
  - Spot the bug (code stem): missing `optimizer.zero_grad()` → gradients accumulate across steps.
  - (calc) Initial loss sanity check for 10 balanced classes ≈ 2.303.
  - Spot the bug: `loss = F.cross_entropy(F.softmax(logits, -1), y)`.
  - Predict: the data are sorted by class and the loader doesn't shuffle (the loss oscillates batch to batch, and the model drifts toward whichever class it saw most recently).
  - Gradient accumulation over 4 micro-batches without dividing the loss: what's the effective lr? (×4)
  - Which first step when loss goes NaN after 1,000 steps? (lower lr / check for inf in data / add clipping; pick the most diagnostic)
  - You double the batch size for throughput. What must you retune, and why? (the LR and other optimizer settings: the optimal LR depends on the batch size; the batch size itself isn't tuned for validation)
- **Figures**
  - `fund.training-loop/loop-diagram`: one iteration as a cycle (load → forward → loss → zero_grad → backward → clip → step → schedule).
  - `fund.training-loop/symptom-curves`: four loss-curve pathologies (lr too high, too low, overfitting, missing zero_grad / no shuffle). **[fig-Q]**: "which bug produced this curve?"
  - `fund.training-loop/overfit-one-batch`: the loss going to ~0 on one batch vs a buggy model that plateaus.
- **Pitfalls & source disagreements**
  - PyTorch zero_grad defaults (`set_to_none=True` since 2.0) vs older behaviour.
  - Scheduler step semantics (per epoch vs per iteration) vary by scheduler class.

---

#### 490 · `fund.activations`: Activation Functions
- **Level** core · **W2** · **prereqs** [fund.backprop] · **covers** activation functions.
- **Sources**
  - Goodfellow DL §6.3 hidden units: §6.3.1 ReLU & generalizations (absolute-value, leaky, PReLU, maxout), §6.3.2 sigmoid & tanh, §6.3.3 other units (softplus, hard tanh, RBF); §6.2.2 output units. `dlb_ch06_mlp.txt`
  - Murphy PML1 §13.2.3 activation functions (pdf p.458), §13.4.2–13.4.3 vanishing/exploding gradients & non-saturating activations (pdf p.477–481). `pml1.txt`
  - Glorot, Bordes & Bengio 2011 (`papers/glorot2011_relu.txt`); He et al. 2015 PReLU (`papers/he2015_prelu_init.txt`); Hendrycks & Gimpel 2016 GELU (`papers/hendrycks2016_gelu.txt`); Ramachandran et al. 2017 Swish (`papers/ramachandran2017_swish.txt`); Shazeer 2020 GLU variants (`papers/shazeer2020_glu.txt`).
  - d2l mlp (activation functions section).
- **Subtopic map**
  1. *Why nonlinearities*: without them, depth collapses to one linear map. Universal approximation needs a non-polynomial activation.
  2. *Sigmoid*: σ′ = σ(1−σ) ≤ 1/4.
     - Saturation means vanishing gradients through depth (0.25^L).
     - Not zero-centred: if all inputs to a unit are positive, the gradients of all its incoming weights share a sign, which causes zig-zag updates. Derive this.
     - Still used for gates and probabilities.
  3. *tanh*: tanh(x) = 2σ(2x) − 1, zero-centred, tanh′(0) = 1, still saturates.
  4. *ReLU*: max(0, x).
     - Non-saturating for x > 0, cheap, sparse activations.
     - Gradient 0 or 1, with the subgradient at 0 conventionally set to 0.
     - **Dying ReLU**: units stuck negative after a large update or a bad bias; once dead, they get zero gradient.
  5. *Leaky ReLU / PReLU* (learned slope), ELU, SELU (self-normalizing under specific init and input conditions; named).
  6. *Smooth modern activations*:
     - GELU = xΦ(x) (the stochastic-regularizer motivation; exact vs tanh approximation).
     - Swish/SiLU = xσ(βx) (β=1 is SiLU).
     - Both are non-monotone with a small negative dip. Compute their derivatives.
  7. *Gated units*: GLU (xW)⊙σ(xV), SwiGLU (Swish gate) and GeGLU in transformer FFNs. Parameter matching: the hidden dim is scaled by 2/3 to keep the parameter count (calc). Why they're used (empirical gains, Shazeer).
  8. *Output activations tied to the likelihood*: identity/Gaussian, sigmoid/Bernoulli, softmax/categorical, softplus or exp for positive scales, softmax temperature. Pointer to fund.loss-functions.
  9. *Interplay with initialization and normalization*: the ReLU gain √2 (pointer to fund.initialization), and why BN/LN reduce sensitivity to the activation choice.
  10. *Numerics*: stable softplus $\log(1+e^x)=\max(x,0)+\log(1+e^{-|x|})$, stable sigmoid for large |x|.
- **Question ideas**
  - (calc) Max of σ′ = 0.25. Ten stacked sigmoids give a gradient factor ≤ 0.25¹⁰ ≈ 9.5e−7.
  - **[fig-Q]** Which curve is the derivative of GELU?
  - Which activation is non-monotone? (GELU/Swish) Which is not zero-centred? (sigmoid, ReLU)
  - Predict: a large negative bias in a ReLU layer after a bad update (dead units, zero gradient).
  - (calc) SwiGLU FFN with $d_{model}=1024$ matching a 4× ReLU FFN's params: hidden ≈ 2731.
  - Derivation: why do non-zero-centred inputs force same-sign weight gradients?
- **Figures**
  - `fund.activations/activation-gallery`: sigmoid, tanh, ReLU, leaky ReLU, GELU and SiLU on one axis.
  - `fund.activations/derivatives`: their derivatives. **[fig-Q]**
  - `fund.activations/dying-relu`: a histogram of pre-activations shifted negative, with the fraction of dead units.
- **Pitfalls & source disagreements**
  - Swish vs SiLU naming (Elfwing et al. 2017 vs Ramachandran et al.). GELU exact vs approximate differs across frameworks (`approximate='tanh'`).
  - "ReLU is non-differentiable at 0" is true. Frameworks use 0 as the subgradient.

---

#### 500 · `fund.initialization`: Weight Initialization
- **Level** core · **W1** · **prereqs** [fund.activations, fund.variance-covariance] · **covers** weight initialisation.
- **Sources**
  - Glorot & Bengio 2010 (`papers/glorot2010_init.txt`): §4.2.1 the variance derivation (forward and backward conditions), eq. 16 the normalized init, and the saturation study.
  - He, Zhang, Ren & Sun 2015 (`papers/he2015_prelu_init.txt`): §2.2 the forward derivation for ReLU, Var = 2/n_l, the backward case, and why either condition suffices.
  - Saxe, McClelland & Ganguli 2014 (`papers/saxe2013_orthogonal.txt`): orthogonal init and dynamical isometry in deep linear nets.
  - Goodfellow DL §8.4 parameter initialization strategies. Murphy PML1 §13.4.5 (pdf p.482). d2l numerical stability & init (`d2l/d2l_numerical_stability_and_init.txt`). ESL §11.5.1 starting values (pdf p.416).
  - Lin et al. 2017 §3.3 prior-probability bias init (`papers/lin2017_focal_loss.txt`); Goyal et al. 2017 zero-γ residual init (`papers/goyal2017_large_minibatch.txt`).
- **Subtopic map**
  1. *Symmetry breaking*: identical initial weights give identical gradients, so units stay clones forever (prove for one layer). Biases can start at 0.
  2. *Variance propagation*: for $y=\sum_{i=1}^{n}w_ix_i$ with independent zero-mean weights, $\mathrm{Var}(y)=n\,\mathrm{Var}(w)E[x^2]$ (derive). Note $E[x^2]$ vs Var(x) when x isn't zero-mean (after a ReLU). Across L layers, the scale multiplies, so it explodes or vanishes exponentially (calc 1.1⁵⁰, 0.9⁵⁰).
  3. *Backward pass*: gradients obey the same recursion with $n_{out}$. Both directions matter.
  4. *Xavier/Glorot*:
     - In the linear regime of tanh, forward needs $n_{in}\mathrm{Var}(w)=1$ and backward needs $n_{out}\mathrm{Var}(w)=1$.
     - The compromise is $\mathrm{Var}(w)=\frac2{n_{in}+n_{out}}$.
     - The uniform version is $U(\pm\sqrt{6/(n_{in}+n_{out})})$, using Var U(−a, a) = a²/3.
     - The old heuristic $U(\pm1/\sqrt n)$ gives Var = 1/(3n), so activations shrink layer by layer (Glorot's motivation).
  5. *He/Kaiming*: ReLU zeroes half the mass, $E[\mathrm{relu}(z)^2]=\frac12\mathrm{Var}(z)$ (derive for symmetric z), so $\mathrm{Var}(w)=2/n_{in}$ (fan-in) or $2/n_{out}$ (fan-out). Leaky ReLU: $2/((1+a^2)n)$. Either condition alone keeps the product of factors bounded (He's argument).
  6. *Orthogonal init*: exactly norm-preserving for linear layers. Dynamical isometry. Useful for RNNs (eigenvalues on the unit circle) and very deep nets.
  7. *Special layers*:
     - Zero-init the last BN γ in each residual block (Goyal) or the residual branch itself (ReZero; Bachlechner et al. 2020, `tp_bachlechner2020_rezero.txt`): the network starts as the identity, so early curvature is low (Tuning-plan insertion). Or scale residual branches by $1/\sqrt{2L}$ (GPT-2; `llm_radford2019_gpt2.txt`).
     - Small final-layer logits for calibrated initial loss.
     - **Bias init for class imbalance**: $b=-\log\frac{1-\pi}\pi$ (Lin et al.; calc π=0.01 → −4.6).
     - LSTM forget-gate bias = 1 (pointer to fund.lstm-gru).
     - Embedding init scale.
  8. *Transformers*: std 0.02 conventions, μP and width-dependent scaling (named; `llm_yang2022_mup.txt`).
  9. *Framework defaults*: PyTorch `nn.Linear` uses `kaiming_uniform_(a=√5)`, which gives $U(\pm1/\sqrt{fan_{in}})$, i.e. Var = 1/(3·fan_in). That's neither Xavier nor He-for-ReLU. TF/Keras Dense defaults to Glorot uniform.
  10. *Interaction with normalization*: BN/LN make nets far less sensitive to init scale, but residual-stream growth still depends on it.
- **Question ideas**
  - (calc) Glorot uniform bound for 256→512: √(6/768) ≈ 0.088.
  - (calc) He-normal std for fan_in = 512: 0.0625.
  - (calc) 50 layers with per-layer gain 0.9 → 0.9⁵⁰ ≈ 0.005, so activations vanish.
  - Why the factor 2 for ReLU? (E[relu(z)²] = Var(z)/2)
  - All weights initialized to the same constant in an MLP: what happens? (hidden units remain identical)
  - (calc) Focal-loss bias init for π=0.01: b ≈ −4.6.
- **Figures**
  - `fund.initialization/activation-std-depth`: per-layer activation std through a 20-layer ReLU MLP under small, Xavier, He and large init. **[fig-Q]**: "which line is He init?"
  - `fund.initialization/activation-histograms`: per-layer histograms for tanh with the old heuristic vs Xavier (after Glorot Fig. 6).
  - `fund.initialization/symmetry`: two hidden units' weight trajectories under symmetric vs random init.
- **Pitfalls & source disagreements**
  - fan_in vs fan_out vs fan_avg modes.
  - PyTorch's default isn't He.
  - Glorot's derivation assumes symmetric activations in their linear regime, so it doesn't apply directly to sigmoid.

---

#### 510 · `fund.batchnorm`: Batch Normalization
- **Level** intermediate · **W1** · **prereqs** [fund.initialization] · **covers** BatchNorm.
- **Sources**
  - Ioffe & Szegedy 2015 (`papers/ioffe2015_batchnorm.txt`): §2 the whitening motivation, §3 Alg. 1 (train) & the backward equations, §3.1 inference with population statistics (Alg. 2), §3.2 convolutional BN, §3.3 higher learning rates & the scale-invariance property, §3.4 regularization.
  - Santurkar, Tsipras, Ilyas & Madry 2018 (`papers/santurkar2018_bn_help.txt`): ICS isn't the reason; smoother loss (Lipschitz and β-smoothness improvements).
  - Goodfellow DL §8.7.1 batch normalization. `dlb_ch08_optimization.txt`
  - d2l batch-norm (`d2l/d2l_batch_norm.txt`) incl. the from-scratch implementation. Murphy PML1 §14.2.4 normalization layers (pdf p.506).
- **Subtopic map**
  1. *Motivation*: deep nets are sensitive to the scale and shift of intermediate activations. The original "internal covariate shift" story. Why not full whitening (cost; backprop through the whitening).
  2. *Forward pass (training)*:
     - Per feature (per channel for conv), compute batch mean $\mu_B$ and variance $\sigma_B^2$ (biased).
     - Normalize $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$.
     - Then $y=\gamma\hat x+\beta$.
     - γ, β restore expressivity (they can recover the identity).
     - For conv, statistics run over (N, H, W).
  3. *Backward pass derivation*: gradients flow through $\mu_B$ and $\sigma_B^2$. The compact form $\partial L/\partial x=\frac\gamma{B\sigma}\big(B\,\bar y-\sum\bar y-\hat x\sum\bar y\odot\hat x\big)$ (derive). The output gradient is projected to be orthogonal to 1 and to $\hat x$.
  4. *Inference*: use running (EMA) estimates of the mean and variance (unbiased variance for the running estimate), so the layer becomes a fixed affine map that can be folded into the preceding conv/linear layer.
     - **Train/eval mismatch bugs**: forgetting `eval()`, batch size 1, and distribution shift in the running stats.
  5. *Why it helps*:
     - Allows higher lr and less init sensitivity.
     - **Scale invariance**: the output is unchanged when the preceding weights are scaled by c, and the gradient wrt them scales by 1/c, so the effective lr depends on ‖w‖ (derive). This explains the WD+BN interplay.
     - Santurkar's smoothness evidence vs the ICS claim.
  6. *Regularization side-effect*: batch noise. Dropout+BN variance shift (named).
  7. *Placement and redundancy*: BN before the nonlinearity (paper) vs after (common variants). The preceding layer's bias is redundant (β absorbs it). No weight decay on γ, β (usually).
  8. *Batch-dependence problems*:
     - Small batches give noisy statistics.
     - Non-iid batches.
     - Cross-example information leakage, e.g. in contrastive learning (shuffled BN, named).
     - Sequence models and variable lengths, hence LN for transformers (next lesson).
     - Fixes: GroupNorm (pointer), SyncBN across devices, batch renorm (named).
     - The BN batch is not the gradient batch (Tuning-plan insertion): statistics are per device unless synced; ghost BN computes them over ~64 examples (Hoffer et al. 2017, `tp_hoffer2017_ghost_bn.txt`); Goyal keeps the per-worker n = 32 fixed so the loss function doesn't change as workers scale.
  9. *Fine-tuning*: freeze vs update the BN statistics. Domain adaptation by re-estimating them (AdaBN; pointer to fund.domain-adaptation).
- **Question ideas**
  - (calc) Tensor (N=32, C=64, H=W=16): how many means does BN compute? (64) Over how many elements each? (32·16·16 = 8192)
  - Model fine in training, bad at eval → which BN issue? (running stats/eval mode/batch-size mismatch)
  - Why is a linear layer's bias redundant before BN? (the mean subtraction cancels it)
  - (calc) Scale the preceding weights by 10: BN output unchanged, weight gradient ÷10.
  - Which is false: "BN's success is fully explained by reducing internal covariate shift".
  - Derivation step: in the BN backward pass, why does the term $\sum\bar y$ appear? (the gradient through μ_B)
- **Figures**
  - `fund.batchnorm/axes-diagram`: (N, C, H, W) cube with the BN statistics axes highlighted (shared with fund.normalization).
  - `fund.batchnorm/train-vs-eval`: running mean vs per-batch means over training steps.
  - `fund.batchnorm/lr-sensitivity`: final loss vs lr with and without BN. **[fig-Q]**
- **Pitfalls & source disagreements**
  - PyTorch BN `momentum=0.1` means running ← 0.9·running + 0.1·batch, the opposite of optimizer-momentum convention.
  - Running variance is unbiased while the normalizing variance is biased.
  - Some implementations checkpoint only device 0's running statistics. The EMA is linear, so averaging across devices at checkpoint time is exact (Tuning-plan insertion).
  - The ICS explanation is contested.

---

#### 520 · `fund.normalization`: LayerNorm, RMSNorm & Where to Normalize *(keeps id)*
- **Level** intermediate · **W1** · **prereqs** [fund.batchnorm] · **covers** LayerNorm/RMSNorm (+ BatchNorm comparison).
- **Sources**
  - Ba, Kiros & Hinton 2016 (`papers/ba2016_layernorm.txt`): §3 LN definition, §4 RNN usage, §5 the invariance analysis table and geometry of parameter space.
  - Zhang & Sennrich 2019 (`papers/zhang2019_rmsnorm.txt`): RMSNorm definition, re-scaling invariance without re-centring, pRMSNorm, efficiency claims.
  - Xiong et al. 2020 (`papers/xiong2020_preln.txt`): Post-LN vs Pre-LN gradient analysis at init, warmup necessity.
  - Wu & He 2018 (`papers/wu2018_groupnorm.txt`): GN, the normalization-axes figure, small-batch behaviour.
  - Murphy PML2 §16.2.5 normalization layers (pdf p.669). Dehghani et al. 2023 QK-norm and Wortsman 2023 (`llm_dehghani2023_vit22b.txt`, `sys_wortsman2023_small_scale_instabilities.txt`) for modern variants.
- **Subtopic map**
  1. *Recap*: why BN is awkward for sequences, small batches and inference with variable inputs. This is the reason behind the playbook's "BN can often be replaced with LayerNorm": LN has no batch dependence, so there are no multi-device, small-batch or train/test discrepancies (Tuning-plan insertion).
  2. *LayerNorm*: per example, over the feature dimension: $\mu=\frac1H\sum h_i$, $\sigma^2=\frac1H\sum(h_i-\mu)^2$, $y=g\odot\frac{h-\mu}{\sqrt{\sigma^2+\epsilon}}+b$. The same computation at train and test, independent of the batch. Elementwise affine parameters.
  3. *Invariance comparison* (Ba et al. Table 1; corrected in review):
     - BN is invariant to rescaling the whole weight matrix *or any single unit's incoming weight vector*, and to rescaling or re-centring the dataset. It is not invariant to re-centring the weight matrix or to rescaling a single training case.
     - LN is invariant to rescaling and re-centring the whole weight matrix ($W\to\delta W+\mathbf 1\gamma^\top$) and to rescaling a single training case. It is *not* invariant to rescaling one unit's weight vector or to re-centring the dataset.
     - Explain what each invariance buys you: BN's per-unit scale invariance gives the weight-norm/effective-lr effect of fund.batchnorm; LN's per-case invariance makes it insensitive to the overall magnitude of each input.
  4. *RMSNorm*: $y=g\odot\frac h{\sqrt{\frac1H\sum h_i^2+\epsilon}}$. No mean subtraction and a gain only (Zhang & Sennrich keep the bias as part of the surrounding layer; LLaMA-style models drop it). It keeps re-scaling invariance and loses re-centring invariance. Cheaper. The standard in modern LLMs (LLaMA etc.).
  5. *GroupNorm / InstanceNorm*: the axes table for (N, C, H, W): BN (N,H,W), LN (C,H,W), IN (H,W), GN (C/G,H,W). GN for small-batch detection/segmentation. IN for style transfer.
  6. *Where to put the norm in a residual block*:
     - **Post-LN** (original Transformer): x + F(x), then LN.
     - **Pre-LN**: x + F(LN(x)).
     - Xiong et al.: Post-LN has large gradients near the output at init, so it needs warmup. Pre-LN has well-behaved gradients, but the residual stream grows and later layers contribute less (named).
     - Sandwich/peri-LN variants (named).
     - The Tuning Playbook lists "normalize inside the residual", $x+f(\mathrm{Norm}(x))$ rather than $\mathrm{Norm}(x+f(x))$, among its instability fixes (Tuning-plan insertion).
  7. *Normalization for stability at scale*: QK-norm (LN on queries and keys) to prevent attention-logit growth, final LN before the output head, z-loss (named).
  8. *Parameter and compute costs*: 2H params for LN (H for RMSNorm). Reductions are memory-bound, so fused kernels are used.
  9. *Weight decay on gains* (usually excluded). Init of gains at 1 (and zero-init variants such as adaLN-Zero in DiT, named; pointer to the gen area).
- **Question ideas**
  - (calc) For (N=32, T=128, H=512) under LN, how many mean/variance pairs are computed? (32·128 = 4096)
  - Which is false: "RMSNorm subtracts the mean before scaling".
  - Pre-LN vs Post-LN: which typically needs lr warmup to train stably? (Post-LN)
  - Predict: switching BN → LN in a CNN with batch size 2 (more stable statistics).
  - (calc) Vector h = (1, 2, 3, 4): LN output (with g=1, b=0, ε=0) = (−1.342, −0.447, 0.447, 1.342). RMSNorm output = h/√7.5 = (0.365, 0.730, 1.095, 1.461).
  - Which invariance does RMSNorm give up relative to LN? (re-centring)
- **Figures**
  - `fund.normalization/axes-cubes`: BN/LN/IN/GN axes cubes (after Wu & He Fig. 2). **[fig-Q]**: "which cube is LayerNorm?"
  - `fund.normalization/pre-vs-post-ln`: block diagrams of Post-LN and Pre-LN residual blocks.
  - `fund.normalization/ln-vs-rms`: a vector before and after LN vs RMSNorm (shift removed vs not).
- **Pitfalls & source disagreements**
  - "LayerNorm" over which axes in CNNs is ambiguous across codebases.
  - ε placement (inside vs outside the sqrt).
  - Pre-LN's "no warmup needed" holds at init. Large-scale practice still uses warmup.

---

#### 530 · `fund.cnn`: Convolutions: Mechanics & Inductive Bias
- **Level** core · **W2** · **prereqs** [fund.mlp-from-scratch] · **covers** CNNs.
- **Sources**
  - Goodfellow DL ch.9:
    - §9.1 the convolution operation (cross-correlation).
    - §9.2 motivation: sparse interactions, parameter sharing, equivariance.
    - §9.3 pooling and invariance.
    - §9.4 conv & pooling as an infinitely strong prior.
    - §9.5 variants: stride, zero-padding (valid/same/full), unshared/tiled, and the backprop formulas.
    - §9.8 efficient convolution (separable kernels, FFT).
    - `dlb_ch09_convnets.txt` · https://www.deeplearningbook.org/contents/convnets.html
  - Dumoulin & Visin 2016 (`papers/dumoulin2016_conv_arithmetic.txt`): output-size relationships for padding, stride, dilation and transposed convolution.
  - d2l conv-layer, padding-and-strides, channels (multi-channel and 1×1), pooling (`d2l/d2l_conv_layer.txt`, `d2l/d2l_padding_and_strides.txt`, `d2l/d2l_channels.txt`, `d2l/d2l_pooling.txt`).
  - Murphy PML1 §14.2.1 convolutional layers, §14.2.2 pooling (pdf p.498–506), §14.4 other forms of convolution (dilated, transposed, depthwise; pdf p.516). CS229 notes ch.7 convolutional modules (pdf p.~99).
- **Subtopic map**
  1. *From dense layers to convolutions*: images have locality and translation structure, while a dense layer on a 224×224×3 input has huge parameter counts and no weight sharing.
  2. *The operation*: 2-D cross-correlation $S(i,j)=\sum_{m,n}I(i+m,j+n)K(m,n)$ vs true convolution (kernel flipped). DL "conv" is cross-correlation, and the difference is immaterial when K is learned. Multi-channel input (sum over $C_{in}$) and multiple output channels.
  3. *Output size* $\lfloor(n+2p-k)/s\rfloor+1$ (derive). "Same" padding p = (k−1)/2 for odd k at stride 1. Dilation gives effective kernel $k+(k-1)(d-1)$. Transposed conv gives output $(n-1)s-2p+k$ (+ output_padding), and why it isn't a deconvolution.
  4. *Parameter count and FLOPs*: $k^2C_{in}C_{out}+C_{out}$ params. FLOPs ≈ $2H_{out}W_{out}k^2C_{in}C_{out}$. Compare with a dense layer on the same input (calc).
  5. *Receptive field*: recursion $r_l=r_{l-1}+(k_l-1)\prod_{i<l}s_i$. Two 3×3 = one 5×5 receptive field with fewer params (18C² vs 25C²) plus an extra nonlinearity (the VGG argument). Effective vs theoretical RF (named).
  6. *Inductive biases*:
     - Locality.
     - Weight sharing.
     - Translation **equivariance**: shift the input, the output shifts. Exact for stride 1 with circular padding. Broken by stride, pooling and padding (aliasing).
     - Pooling gives approximate local **invariance**.
     - "Infinitely strong prior" (DLB §9.4) and when it hurts.
  7. *Pooling*: max vs average. The max gradient routes to the argmax. Global average pooling replaces FC heads. Strided conv vs pooling for downsampling.
  8. *Pointer*: 1×1 and depthwise-separable convolutions moved to fund.cnn-architectures (review 2026-10-02), where the Inception/ResNet bottlenecks and MobileNet motivate them; this keeps this lesson at about 10 cards.
  9. *Backprop through conv*: the weight gradient correlates the input with the output gradient. The input gradient is a "full" convolution of the output gradient with the flipped kernel, i.e. a transposed conv (derive the 1-D case).
  10. *Implementation*: im2col → matmul, FFT for large kernels, Winograd (named). Memory layouts NCHW vs NHWC.
- **Question ideas**
  - (calc) Input 32×32, k=5, p=2, s=2 → 16×16.
  - (calc) Params of a 3×3 conv 64→128 with bias: 73,856.
  - (calc) Receptive field of three stacked 3×3 stride-1 convs: 7. With a stride-2 second layer? (3, then 3+2·1=5, then 5+2·2=9)
  - Which is false: "convolutional layers are translation invariant". (They're equivariant; invariance is approximate via pooling.)
  - Predict: the output when the input image is shifted by 2 pixels with a stride-2 conv (not a clean shift; aliasing).
- **Figures**
  - `fund.cnn/conv-sliding`: a 5×5 input, 3×3 kernel and 3×3 output, with one dot product highlighted.
  - `fund.cnn/receptive-field`: RF growth over three layers drawn as nested squares. **[fig-Q]**
  - `fund.cnn/padding-stride`: output grids for valid, same and stride 2.
- **Pitfalls & source disagreements**
  - Convolution vs cross-correlation naming.
  - "Same" padding with even kernels is asymmetric (framework-dependent).
  - NCHW (PyTorch) vs NHWC (TF).

---

#### 540 · `fund.cnn-architectures`: CNN Architectures, Residual Connections & Efficient Convolutions
- **Level** core · **W2** · **prereqs** [fund.cnn, fund.batchnorm] · **covers** CNNs.
- **Sources**
  - He, Zhang, Ren & Sun 2016 ResNet (`papers/he2016_resnet.txt`): the degradation problem, residual learning, identity vs projection shortcuts, bottleneck blocks, depth experiments.
  - Murphy PML1 §14.3 common architectures: LeNet, AlexNet, GoogLeNet/Inception, ResNet, DenseNet, NAS (pdf p.509–516); §14.4 other forms of convolution incl. depthwise (pdf p.516); §13.4.4 residual connections (pdf p.481). `pml1.txt`
  - Goodfellow DL §9.8 efficient convolution (separable kernels). `dlb_ch09_convnets.txt`
  - d2l resnet (`d2l/d2l_resnet.txt`), channels (1×1 convs). Goodfellow DL §9.11 history.
  - Murphy PML2 §16.2.4 residual connections (pdf p.669). `pml2.txt`
- **Subtopic map**
  1. *The canonical progression and what each step contributed*:
     - LeNet: conv-pool stacks.
     - AlexNet: ReLU, dropout, GPUs, data augmentation.
     - VGG: deep stacks of 3×3.
     - Inception: multi-branch, 1×1 bottlenecks.
     - ResNet: residuals.
     - DenseNet: concatenation.
     - EfficientNet: compound scaling (named).
     - ConvNeXt (named).
  2. *The degradation problem*: deeper *plain* nets get higher **training** error, so this is optimization, not overfitting. That's the motivation for residuals.
  3. *Residual block*: $y=x+F(x)$. Easy to represent the identity (F = 0). Gradient $\partial y/\partial x=I+\partial F/\partial x$, so gradients have a direct path. Unrolled view: an ensemble of paths of many lengths (named).
  4. *Block designs*:
     - Basic (3×3, 3×3).
     - Bottleneck (1×1 reduce → 3×3 → 1×1 expand); param/FLOP comparison (calc).
     - Projection shortcuts when the shape changes (1×1 stride-2 conv).
     - Pre-activation ResNet (BN-ReLU-conv; named).
  5. *1×1 and depthwise-separable convolutions* (moved from fund.cnn): a 1×1 conv is per-pixel channel mixing (a dense layer over channels), used for the bottlenecks above. Depthwise k×k per channel + pointwise 1×1 (MobileNet/Xception, named), with cost ratio ≈ $1/C_{out}+1/k^2$ vs a standard conv (derive; calc).
  6. *Normalization and init in ResNets*: BN after each conv. Zero-init the last γ in each block, or the whole residual branch (ReZero), so the network starts as the identity (pointer to fund.initialization; Tuning-plan insertion).
  7. *Design principles*: downsample by stride while doubling channels (keeps per-layer compute roughly constant), global average pooling + linear head, depth vs width vs resolution trade-offs.
  8. *Residuals beyond CNNs*: transformers' residual stream, highway networks and LSTM's additive cell as precursors (pointer to fund.lstm-gru).
  9. *Inductive bias vs data scale*: CNNs vs ViTs (named; pointer to the vision/LLM area).
- **Question ideas**
  - Why do deeper plain nets have higher *training* error? (optimization difficulty, not overfitting)
  - (calc) Bottleneck block 256→64→64→256 (1×1, 3×3, 1×1) params ≈ 69.6k vs two 3×3 256→256 convs ≈ 1.18M.
  - (calc) Depthwise-separable cost ratio for k=3, C_out=256 ≈ 0.115.
  - Derivation: the gradient of y = x + F(x) wrt x, and why it helps.
  - Which is false: "ResNets work because the residual branch learns the full mapping more easily than a plain block".
  - Predict: remove the shortcut connections from a 100-layer ResNet (training degrades).
- **Figures**
  - `fund.cnn-architectures/degradation`: training error vs iteration for 20- vs 56-layer plain nets and their ResNet counterparts (schematic after He et al.). **[fig-Q]**
  - `fund.cnn-architectures/residual-blocks`: basic vs bottleneck block diagrams with channel counts.
  - `fund.cnn-architectures/depthwise-separable`: standard vs depthwise + pointwise block diagram with cost annotations (moved from fund.cnn; replaces a decorative architecture-timeline figure).
- **Pitfalls & source disagreements**
  - "ResNets solve vanishing gradients" vs the degradation/optimization framing (He et al. note that BN already handles vanishing gradients).

---

#### 550 · `fund.rnn`: RNNs, BPTT & Vanishing Gradients
- **Level** intermediate · **W2** · **prereqs** [fund.backprop, fund.activations] · **covers** RNNs.
- **Sources**
  - Goodfellow DL ch.10:
    - §10.1 unfolding graphs.
    - §10.2 RNNs: the canonical equations, §10.2.1 teacher forcing, §10.2.2 gradient computation (BPTT equations).
    - §10.3 bidirectional, §10.4 encoder–decoder.
    - §10.7 the challenge of long-term dependencies (eigenvalue analysis).
    - §10.11.1 gradient clipping.
    - `dlb_ch10_rnn.txt`
  - Pascanu, Mikolov & Bengio 2013 (`papers/pascanu2013_rnn_difficulty.txt`): exploding/vanishing analysis (spectral-radius conditions), the dynamical-systems view, norm clipping, the vanishing-gradient regularizer.
  - Bengio, Simard & Frasconi 1994 (`sys_bengio1994_long_term_deps.txt`): the original long-term-dependency difficulty.
  - Murphy PML1 §15.2 RNNs: vec2seq, seq2vec, seq2seq, §15.2.4 teacher forcing, §15.2.5 BPTT, §15.2.6 vanishing/exploding (pdf p.533–542). `pml1.txt`
  - d2l rnn & bptt (`d2l/d2l_rnn.txt`, `d2l/d2l_bptt.txt`); d2l clipping (`sys_d2l_rnn_scratch_clipping.txt`); Zhang et al. 2020 why clipping helps (`sys_zhang2020_why_clipping.txt`).
- **Subtopic map**
  1. *Sequence tasks*: one-to-many, many-to-one, many-to-many (aligned and seq2seq). Why weight sharing across time (variable length, generalization across positions).
  2. *Vanilla RNN*: $h_t=\tanh(Wh_{t-1}+Ux_t+b)$, $o_t=Vh_t+c$. Unrolled graph. Parameter count independent of T (calc).
  3. *Training*: the per-step loss is summed. **BPTT** is backprop on the unrolled graph. Derive $\frac{\partial L}{\partial h_t}$ recursively and the weight gradient as a sum over time.
  4. *Why gradients vanish or explode*:
     - $\frac{\partial h_t}{\partial h_k}=\prod_{i=k+1}^t\mathrm{diag}(\phi'(a_i))W$.
     - With $\|\mathrm{diag}\phi'\|\le\gamma$, the norm is ≤ $(\gamma\sigma_{\max}(W))^{t-k}$.
     - σ_max(W) < 1/γ is *sufficient* for vanishing. σ_max(W) > 1/γ is *necessary* for exploding (Pascanu; γ = 1 for tanh, ¼ for sigmoid). Pascanu et al. phrase it with the largest eigenvalue, but the proof bounds norms, i.e. singular values.
     - The eigenvalue picture for the linear case (DLB §10.7).
     - Consequence: long-range credit assignment fails.
  5. *Gradient clipping*: global norm $g\leftarrow g\cdot\min(1,\tau/\|g\|)$ preserves direction, vs value clipping. The cliff picture (Pascanu). Why clipping is standard even in transformers.
  6. *Truncated BPTT*: carry the hidden state, backprop only k steps. A biased gradient that misses dependencies longer than k. Memory O(kH).
  7. *Teacher forcing*: feed ground-truth previous outputs during training. Exposure bias at test time. Scheduled sampling (named; `llm_bengio2015_scheduled_sampling.txt`).
  8. *Architectural variants*: stacked/deep RNNs, bidirectional (needs the full sequence), encoder–decoder with a fixed-size bottleneck, which motivated attention (pointer to the LLM area).
  9. *Mitigations other than gating*: orthogonal/identity init (IRNN), skip connections through time, leaky units. Gating (next lesson) is the main fix.
  10. *Cost and why transformers won*: O(T) sequential dependency (no parallelism over time in training) and path length O(T) between distant tokens. SSM/linear-RNN revival (named; pointer to the LLM area).
- **Question ideas**
  - (calc) Vanilla RNN params with input 100, hidden 256 (no output layer): 256·256 + 256·100 + 256 = 91,392.
  - (calc) Global norm clipping with ‖g‖=10, τ=5 → scale 0.5.
  - Predict: a 100-step dependency with σ_max(W)=0.5 and tanh (the gradient ~ 0.5¹⁰⁰, vanished).
  - Which is false: "with tanh, σ_max(W) < 1 is necessary for vanishing gradients" (it's sufficient).
  - Truncated BPTT with k=20: can the model learn a dependency of 50 steps? (not via gradients; maybe via the carried state)
  - Exposure bias: what is it, and what's one mitigation?
- **Figures**
  - `fund.rnn/unrolled`: a rolled RNN cell and its unrolled graph over 4 steps, with the gradient paths drawn.
  - `fund.rnn/gradient-norm-vs-lag`: ‖∂h_t/∂h_k‖ vs t−k for σ_max = 0.9, 1.0 and 1.1. **[fig-Q]**
  - `fund.rnn/clipping-cliff`: a loss surface with a cliff and clipped vs unclipped steps (after Pascanu Fig. 6).
- **Pitfalls & source disagreements**
  - Sufficient vs necessary conditions get misquoted.
  - DLB's symbols (U input, W recurrent, V output) differ from Pascanu's and d2l's.

---

#### 560 · `fund.lstm-gru`: LSTMs & GRUs
- **Level** intermediate · **W2** · **prereqs** [fund.rnn] · **covers** LSTMs.
- **Sources**
  - Hochreiter & Schmidhuber 1997 (`papers/hochreiter1997_lstm.txt`): the constant error carousel, the original gates (no forget gate).
  - Gers, Schraudolph & Schmidhuber 2002 (`papers/gers2002_lstm_peephole_timing.txt`): the forget gate (introduced in Gers et al. 2000) and peephole connections.
  - Greff et al. 2017 LSTM search-space odyssey (`papers/greff2015_lstm_odyssey.txt`): ablations (forget gate and output activation most critical; coupled gates fine).
  - Jozefowicz, Zaremba & Sutskever 2015 (`papers/jozefowicz2015_rnn_empirical.txt`): forget-bias = 1 and GRU vs LSTM comparisons. Chung et al. 2014 GRU (`papers/chung2014_gru.txt`).
  - Goodfellow DL §10.10.1 LSTM, §10.10.2 other gated RNNs. Murphy PML1 §15.2.7 gating & long-term memory (pdf p.542). d2l lstm & gru. Olah 2015 (`web/olah2015_lstm.txt`), secondary.
- **Subtopic map**
  1. *Idea*: add a memory cell updated **additively** and controlled by learned gates, so information and gradients can persist.
  2. *LSTM equations* (modern, with forget gate):
     - $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$; likewise $i_t$ and $o_t$.
     - $\tilde c_t=\tanh(W_c[\cdot]+b_c)$.
     - $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$.
     - $h_t=o_t\odot\tanh(c_t)$.
     - The role of each gate in words.
  3. *Gradient flow*: $\partial c_t/\partial c_{t-1}=\mathrm{diag}(f_t)$ along the direct path (plus indirect terms through the gates). With f ≈ 1 gradients persist (the constant error carousel). Compare with the vanilla RNN's $W^\top\mathrm{diag}(\phi')$ product. The gates *learn* when to forget.
  4. *Forget-gate bias init = 1* (Jozefowicz), so the cell starts by remembering. The 1997 LSTM had no forget gate, and cell states grew unboundedly on long streams (the reason Gers added it).
  5. *Parameter count*: $4(H(H+D)+H)$ for one bias vector. PyTorch keeps two bias vectors, giving $4H(H+D)+8H$ (calc).
  6. *Variants*: peepholes (gates see c), coupled input-forget gates, projection layers (named). Greff's finding that the vanilla LSTM is hard to beat and that the forget gate and output activation matter most.
  7. *GRU*: update gate z, reset gate r, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ with $\tilde h_t=\tanh(W[r_t\odot h_{t-1},x_t])$. No separate cell. 3 blocks instead of 4. Performance comparable (Chung, Jozefowicz).
  8. *Practical training*: clipping, dropout on non-recurrent connections, variational/recurrent dropout (named), layer norm in RNNs (pointer to fund.normalization).
  9. *Where LSTMs stand now*: replaced by transformers for most NLP (parallelism, long-range access). Still used in streaming and small-footprint settings. Their gating ideas live on in SSMs/xLSTM (named; pointer to the LLM area).
- **Question ideas**
  - (calc) LSTM params with input 100, hidden 256: 365,568 (one bias) or 366,592 (PyTorch, two biases).
  - Which gate, if fixed to 1, turns the cell into an accumulator? (forget gate f=1 with input gate open)
  - Why initialize the forget bias to 1?
  - Which is false: "the original 1997 LSTM included a forget gate".
  - GRU vs LSTM: which has fewer parameters and why? (3 vs 4 gate blocks, no output gate or separate cell)
  - **[fig-Q]** LSTM cell diagram with one path highlighted: which equation does it compute?
- **Figures**
  - `fund.lstm-gru/lstm-cell`: data-flow diagram of the cell with the gates, the c path (additive) and h.
  - `fund.lstm-gru/gru-cell`: GRU data flow.
  - `fund.lstm-gru/gradient-persistence`: gradient norm vs lag for a vanilla RNN vs an LSTM with f ≈ 0.95 vs f ≈ 1 (simulated). **[fig-Q]**
- **Pitfalls & source disagreements**
  - LSTM equation variants (peepholes, biases, concatenated vs separate matrices) differ across sources.
  - DLB uses s for the cell state and different gate letters.

---

#### 570 · `fund.autoencoders`: Autoencoders
- **Level** intermediate · **W2** · **prereqs** [fund.pca, fund.mlp-from-scratch] · **covers** autoencoders.
- **Sources**
  - Goodfellow DL ch.14:
    - §14.1 undercomplete AEs (the linear AE = PCA subspace).
    - §14.2 regularized AEs: §14.2.1 sparse (a latent-variable interpretation), §14.2.2 denoising, §14.2.3 penalizing derivatives.
    - §14.3 capacity and depth.
    - §14.4 stochastic encoders/decoders.
    - §14.5 DAE & §14.5.1 estimating the score.
    - §14.6 manifolds.
    - §14.7 contractive AE.
    - §14.9 applications.
    - `dlb_ch14_autoencoders.txt`
  - Murphy PML1 §20.3 autoencoders: bottleneck, denoising, contractive, sparse, VAE intro (pdf p.709–719). `pml1.txt`
  - Bishop PRML §12.4.2 auto-associative networks & the PCA connection (pdf p.612). `bishop.txt`
  - Vincent et al. 2010 stacked DAE (`papers/vincent2010_denoising_ae.txt`); Vincent 2011 DAE ↔ score matching (`papers/vincent2011_dsm.txt`).
- **Subtopic map**
  1. *Setup*: encoder $z=f(x)$, decoder $\hat x=g(z)$, minimize the reconstruction loss. Why learn to copy: the bottleneck or regularizer forces a useful representation.
  2. *Linear AE with MSE*: the optimal solution spans the top-k principal subspace (Eckart–Young argument), but the learned basis needn't be orthogonal or ordered. A nonlinear AE generalizes PCA to curved manifolds.
  3. *Overcomplete/high-capacity AEs* can learn the identity, so regularization is needed (DLB §14.3).
  4. *Reconstruction losses as likelihoods*: MSE ↔ Gaussian decoder, BCE ↔ Bernoulli decoder. Choosing per data type.
  5. *Sparse AE*: L1 on the code or a KL(ρ‖ρ̂) sparsity penalty. DLB's interpretation as approximate MAP in a latent-variable model with a sparse prior. Modern sparse AEs for interpretability (overcomplete with TopK/L1; `llm_bricken2023_monosemanticity.txt`), named.
  6. *Denoising AE*:
     - Corrupt with $\tilde x\sim C(\tilde x|x)$ and reconstruct the clean x.
     - The learned vector field $r(\tilde x)-\tilde x$ points toward the data manifold.
     - For small Gaussian noise, $(r(x)-x)/\sigma^2\approx\nabla\log p(x)$ (Vincent 2011; DLB §14.5.1): DAEs estimate the **score**. This is the bridge to diffusion (pointer to gen.ebm-score-matching).
  7. *Contractive AE*: penalty $\|\partial f/\partial x\|_F^2$ makes the encoding insensitive to off-manifold directions. Related to DAEs with small noise.
  8. *Uses*:
     - Dimensionality reduction and visualization.
     - Pretraining (historical greedy layer-wise).
     - Anomaly detection by reconstruction error, with the caveat that AEs can reconstruct anomalies too.
     - Compression.
     - Masked autoencoders (MAE) as modern denoising AEs (named).
  9. *Why a plain AE isn't a generative model*: there's no prior on z, so decoding random codes gives garbage (holes in latent space). That motivates VAEs (pointer to gen.latent-variables-elbo).
- **Question ideas**
  - True/false nuance: a linear AE with k-dim code and MSE learns the same *subspace* as top-k PCA (subspace yes; orthonormal ordered basis no).
  - Predict: an overcomplete AE with no regularization (learns the identity, useless code).
  - Denoising AE: what does $r(\tilde x)-\tilde x$ approximate? (σ² × score)
  - Why can't you sample from a vanilla AE by decoding N(0, I) codes?
  - Which is false: "AE reconstruction error reliably flags all anomalies".
  - Contractive penalty: write it, and say what it encourages.
- **Figures**
  - `fund.autoencoders/architecture`: encoder → bottleneck → decoder diagram.
  - `fund.autoencoders/dae-vector-field`: 2-D data on a circle with the learned denoising vector field pointing to the manifold. **[fig-Q]**
  - `fund.autoencoders/latent-holes`: a 2-D AE latent space with encoded training points and decodes of random points.
  - `fund.autoencoders/linear-ae-vs-pca`: learned linear AE basis vectors vs PCA directions spanning the same plane.
- **Pitfalls & source disagreements**
  - "Sparse autoencoder" means different things in classic DL vs interpretability work.
  - DLB covers DAEs twice (§14.2.2 and §14.5).
  - Whether a VAE is "an autoencoder" depends on framing (Kingma & Welling present it as a latent-variable model with amortized inference).

---

#### 580 · `fund.gradient-estimators`: Gradients Through Sampling: REINFORCE vs Reparameterization
- **Level** advanced · **W3** · **prereqs** [fund.backprop, fund.monte-carlo] · **covers** prerequisite for Gumbel-Softmax (also used in VAEs and RL).
- **Sources**
  - Murphy PML2 §6.3.3 SGD for parameters of a distribution, §6.3.4 score-function estimator (REINFORCE) with control variates, §6.3.5 reparameterization trick, §6.3.7 stochastic computation graphs (pdf p.307–313). `pml2.txt`
  - Kingma & Welling 2014 §2.3–2.4 (SGVB estimator and the reparameterization trick; `papers/kingma2013_vae.txt`). Kingma & Welling 2019 §2.4 (`papers/kingma2019_vae_intro.txt`).
  - Goodfellow DL §20.9 back-propagation through random operations, §20.9.1 discrete stochastic operations (REINFORCE, baselines). `dlb_ch20_generative_models.txt`
  - Huijben et al. 2022 §2 (estimator taxonomy; `papers/huijben2021_gumbel_review.txt`). Schulman et al. 2015 GAE (`llm_schulman2015_gae.txt`) for the RL connection (named).
- **Subtopic map**
  1. *The problem*: compute $\nabla_\phi E_{z\sim q_\phi}[f(z)]$. The expectation's distribution depends on φ, so we can't push the gradient inside naively. Examples: VAE encoder, policy gradient, discrete latent choices.
  2. *Score-function / REINFORCE estimator*: derive $\nabla_\phi E_q[f]=E_q[f(z)\nabla_\phi\log q_\phi(z)]$ via the log-derivative trick. It works for discrete z and non-differentiable f. It is unbiased.
  3. *Variance problem and baselines*: $E_q[\nabla\log q]=0$ (prove), so subtracting any constant or state-dependent baseline b keeps it unbiased. The optimal-baseline idea: a baseline is a control variate, whose optimal coefficient was derived in fund.monte-carlo (recap in one line). A worked 1-D example showing the variance drop.
  4. *Reparameterization (pathwise) estimator*: write $z=g(\phi,\varepsilon)$ with ε ~ p(ε) independent of φ. Then $\nabla_\phi E[f(g(\phi,\varepsilon))]=E[\nabla_zf\cdot\partial g/\partial\phi]$. Gaussian example $z=\mu+\sigma\varepsilon$. Requirements: continuous z, differentiable f and g.
  5. *Variance comparison*: a worked example (e.g. $f(z)=z^2$, $z\sim N(\mu,1)$) computing both estimators' variance. Reparameterization is usually much lower, though not always.
  6. *What to do for discrete z*:
     - REINFORCE with baselines.
     - Continuous relaxations (Gumbel-Softmax, next lesson).
     - Straight-through estimators (biased).
     - Marginalizing small discrete spaces exactly.
  7. *Connections*: policy gradients in RL (pointer to the LLM RLHF area), VAEs (pointer to gen.vae), implicit reparameterization and score-function estimators in black-box VI (named).
- **Question ideas**
  - Flashcard: show $E_q[\nabla_\phi\log q_\phi(z)]=0$.
  - (calc) $z\sim N(\mu,1)$, $f=z^2$: true gradient 2μ. Reparam estimator 2(μ+ε) has variance 4. Compare with the score-function estimator's variance at μ=0 ($E[z^6]=15$, so variance 15).
  - Which estimator applies to a non-differentiable black-box reward? (score function)
  - Which is false: "subtracting a baseline biases the REINFORCE gradient".
  - Predict: reparameterization for a Bernoulli latent (not directly applicable; needs relaxation).
- **Figures**
  - `fund.gradient-estimators/stochastic-graph`: computation graph before and after reparameterization (the stochastic node moved to an input ε).
  - `fund.gradient-estimators/variance-comparison`: histograms of the two estimators' samples for the $z^2$ example. **[fig-Q]**
- **Pitfalls & source disagreements**
  - "Score function" here means $\nabla_\phi\log q_\phi$ (w.r.t. parameters), not the diffusion score $\nabla_x\log p(x)$ (w.r.t. data). Flag this clash explicitly.

---

#### 590 · `fund.gumbel-softmax`: Gumbel-Max, Gumbel-Softmax & Straight-Through
- **Level** advanced · **W3** · **prereqs** [fund.gradient-estimators] · **covers** Gumbel-Softmax.
- **Sources**
  - Jang, Gu & Poole 2017 (`papers/jang2016_gumbel_softmax.txt`): §2 Gumbel-Softmax, temperature, the straight-through variant, experiments.
  - Maddison, Mnih & Teh 2017 (`papers/maddison2016_concrete.txt`): the Concrete distribution, its density, rounding property, temperature ≤ 1/(n−1) for log-convexity.
  - Huijben, Kool, Paulus & van Sloun 2022 (`papers/huijben2021_gumbel_review.txt`): proofs of Gumbel-max, Gumbel-top-k, the estimator landscape.
  - Murphy PML2 §6.3.6 Gumbel-softmax, §6.3.8 straight-through (pdf p.311–313). Bengio, Léonard & Courville 2013 STE (`papers/bengio2013_straight_through.txt`). van den Oord et al. 2017 VQ-VAE (STE usage; `papers/vandenoord2017_vqvae.txt`).
- **Subtopic map**
  1. *Goal*: sample from Cat(π) inside a network and still backprop. Sampling via argmax isn't differentiable.
  2. *Gumbel distribution*: $G=-\log(-\log U)$ with U ~ U(0,1). CDF $\exp(-e^{-g})$.
  3. *Gumbel-max trick*: $\arg\max_i(\log\pi_i+G_i)\sim\mathrm{Cat}(\pi)$. **Prove it**, either by integrating the Gumbel CDF or via exponential races ($-\log U_i/\pi_i$ are exponential with rates π_i, and the minimum's index has probability ∝ π_i). Works with unnormalized logits (shift invariance).
  4. *Gumbel-Softmax / Concrete relaxation*: $y=\mathrm{softmax}((\log\pi+G)/\tau)$.
     - τ → 0 gives one-hot argmax samples but high gradient variance. Large τ gives near-uniform, smooth, biased samples.
     - Annealing schedules.
     - The density (Maddison), and log-convexity for τ ≤ 1/(n−1).
  5. *Bias*: the relaxed sample isn't a categorical sample, so gradients are biased w.r.t. the true discrete objective. The downstream network sees soft inputs at train time and hard ones at test time.
  6. *Straight-through (ST) Gumbel*: forward the hard one-hot, backprop the soft gradient. Implementation `y_hard - y_soft.detach() + y_soft`. Biased but practical. The general STE for binarization and quantization (copy gradients; VQ-VAE).
  7. *Gumbel-top-k*: sampling k items without replacement via the top-k of the perturbed logits (review).
  8. *Uses*: discrete-latent VAEs (dVAE in DALL·E 1, named), hard attention, differentiable architecture search, discrete actions, token-level relaxations.
  9. *Numerics*: clamp U to avoid log 0. Compute in log space. Temperature placement.
- **Question ideas**
  - (calc) Sampling Gumbel by inverse CDF: U = 0.5 → G = −log(−log 0.5) ≈ 0.367.
  - Gumbel-max: with logits log(0.7) and log(0.3), what is P(argmax = class 1)? (0.7, exactly)
  - Predict: τ → 0 vs τ → ∞ behaviour of the samples and of the gradient variance.
  - What does `y_hard - y_soft.detach() + y_soft` produce in the forward and backward passes?
  - Which is false: "Gumbel-Softmax samples are exact samples from the categorical distribution for any τ > 0".
- **Figures**
  - `fund.gumbel-softmax/temperature-samples`: relaxed samples on the 3-simplex for τ = 0.1, 0.5, 1, 5. **[fig-Q]**: "which panel has the smallest τ?"
  - `fund.gumbel-softmax/gumbel-max-histogram`: empirical frequencies from Gumbel-max vs π.
  - `fund.gumbel-softmax/st-diagram`: forward (hard) and backward (soft) paths of the straight-through estimator.
- **Pitfalls & source disagreements**
  - Jang ("Gumbel-Softmax") and Maddison ("Concrete") describe the same distribution. Maddison writes λ for temperature and α for location parameters.
  - Inputs should be logits or log-probabilities, not probabilities.

---

### Transfer & distribution shift

---

#### 600 · `fund.transfer-learning`: Transfer Learning & Fine-Tuning
- **Level** intermediate · **W2** · **prereqs** [fund.cnn-architectures, fund.regularization] · **covers** transfer learning.
- **Sources**
  - Goodfellow DL §15.2 transfer learning & domain adaptation (shared representations, which layers transfer), §15.1 greedy layer-wise pretraining (historical). `dlb_ch15_representation.txt`
  - Murphy PML1 §19.2 transfer learning: §19.2.1 fine-tuning, §19.2.2 adapters, §19.2.3 supervised pretraining, §19.2.4 self-supervised pretraining (pdf p.658–667). `pml1.txt`
  - Yosinski, Clune, Bengio & Lipson 2014 (`papers/yosinski2014_transferable.txt`): layer-wise transferability, fragile co-adaptation, specificity.
  - Kumar et al. 2022 (`papers/kumar2022_lpft.txt`): fine-tuning distorts features, LP-FT.
  - CS229 notes ch.15 "Linear probe and finetuning", LoRA (pdf p.193–196). Hu et al. 2021 LoRA (`llm_hu2021_lora.txt`). Houlsby et al. 2019 adapters (`llm_houlsby2019_adapters.txt`). d2l fine-tuning (`d2l/d2l_fine_tuning.txt`).
- **Subtopic map**
  1. *Why transfer*: labelled data is scarce, pretrained features encode general structure, and compute is cheap at fine-tuning time. Source/target task terminology.
  2. *What transfers* (Yosinski): early layers are general (edges, textures), later layers specific. Performance drops from co-adaptation when splitting mid-network. Transfer even from distant tasks beats random init.
  3. *Strategies*:
     - Feature extraction / linear probe (freeze the backbone).
     - Full fine-tuning.
     - Partial fine-tuning (top k layers).
     - Discriminative (layer-wise) learning rates.
     - Gradual unfreezing.
     - Parameter-efficient methods: adapters, LoRA ($\Delta W=BA$, rank r, B initialized to 0 so training starts at the pretrained model; calc params), prompt/prefix tuning (named).
  4. *Choosing a strategy*: the 2×2 of target data size × domain similarity, and compute/memory constraints.
  5. *LP-FT*: fine-tuning with a randomly initialized head distorts pretrained features (large early gradients), hurting OOD accuracy. Linear-probe first, then fine-tune (Kumar et al.'s argument in words).
  6. *Practicalities*:
     - Lower lr for the backbone than for the head.
     - Warmup.
     - Keep or replace the BN statistics (pointer to fund.batchnorm).
     - Regularize toward the pretrained weights (L2-SP, named).
     - Early stopping.
     - Data augmentation.
     - Catastrophic forgetting of source capabilities.
  7. *Pretraining sources*: supervised (ImageNet), self-supervised (contrastive, masked modelling), multimodal (CLIP), with a one-line description each (pointer to the LLM and representation areas).
  8. *Negative transfer* and when pretraining doesn't help (very different domains, abundant target data).
- **Question ideas**
  - Small target dataset, similar domain: best first strategy? (linear probe or fine-tune the top layers)
  - (calc) LoRA on a 4096×4096 matrix with r=8: 65,536 params vs 16.8M (0.39%).
  - Why initialize LoRA's B to zero? (ΔW = 0 at the start, so you begin exactly from the pretrained model)
  - LP-FT motivation: what goes wrong with full fine-tuning from a random head? (feature distortion, hurting OOD accuracy)
  - Which is false: "lower layers of a CNN are more task-specific than higher layers".
  - Predict: fine-tuning with a high lr on all layers with tiny data (overfitting and forgetting).
- **Figures**
  - `fund.transfer-learning/strategy-grid`: the 2×2 (data size × similarity) with the recommended strategy in each cell. **[fig-Q]**
  - `fund.transfer-learning/yosinski-curve`: accuracy vs layer index at which features are frozen or transferred (schematic after Yosinski Fig. 2).
  - `fund.transfer-learning/lora-diagram`: W frozen plus a low-rank BA side path.
- **Pitfalls & source disagreements**
  - DLB treats domain adaptation as a case of transfer. PML2 frames it via distribution shift. Keep the vocabulary clear (next two lessons).

---

#### 610 · `fund.few-zero-shot`: Few-Shot & Zero-Shot Learning
- **Level** intermediate · **W2** · **prereqs** [fund.transfer-learning] · **covers** few-/zero-shot learning.
- **Sources**
  - Snell, Swersky & Zemel 2017 (`papers/snell2017_protonets.txt`): episodic training, prototypes, the Bregman-divergence argument, the linear-model equivalence, zero-shot extension.
  - Finn, Abbeel & Levine 2017 (`papers/finn2017_maml.txt`): the MAML bi-level objective, second-order terms, first-order approximation.
  - Tian et al. 2020 (`papers/tian2020_good_embedding_fewshot.txt`): a good embedding + linear classifier beats meta-learning.
  - Radford et al. 2021 CLIP (`papers/radford2021_clip.txt`): §2.3 contrastive pretraining (pseudocode), §3.1 zero-shot transfer incl. prompt engineering & ensembling, robustness analysis.
  - Brown et al. 2020 GPT-3 (`papers/brown2020_gpt3.txt`): definitions of zero-, one- and few-shot in-context learning. Murphy PML1 §19.5 meta-learning & MAML, §19.6 few-shot & matching networks (pdf p.681–685). Goodfellow DL §15.2 (one-shot and zero-shot via shared representations). CS229 notes ch.17 in-context learning & zero-shot prompting (pdf p.217–219).
- **Subtopic map**
  1. *Definitions and the vocabulary clash*:
     - N-way K-shot episodes with support/query sets (meta-learning).
     - "Few-shot" in GPT-3 = examples in the prompt, no gradient updates.
     - "Few-shot fine-tuning" = updating on K examples per class.
     - Zero-shot: no target-class examples at all.
  2. *Metric-based meta-learning*:
     - Prototypical networks: prototype $c_k$ = mean embedding of the support set, classify by softmax over $-\|f(x)-c_k\|^2$.
     - Derive the equivalence to a linear classifier with $w_k=2c_k$, $b_k=-\|c_k\|^2$.
     - Why squared Euclidean distance (Bregman argument).
     - Episodic training. Matching networks (named).
  3. *Optimization-based meta-learning*: MAML.
     - Inner step $\theta'_i=\theta-\alpha\nabla L_{T_i}(\theta)$.
     - Outer objective $\sum_iL_{T_i}(\theta'_i)$.
     - The meta-gradient involves Hessian-vector products. First-order MAML drops them.
     - Intuition: find an init that is a few steps from good solutions for many tasks.
  4. *Strong baselines*: pretrain a good embedding (supervised or self-supervised), then fit a linear/logistic classifier on the K shots. This matches or beats meta-learning (Tian et al.). This shifted the field toward pretraining.
  5. *Classic zero-shot learning*: map inputs and class descriptions (attributes, word embeddings) to a shared space, and classify unseen classes by similarity (DLB §15.2). Generalized ZSL (seen + unseen classes at test time) and hubness (named).
  6. *CLIP zero-shot*:
     - Symmetric InfoNCE over an N×N batch of image–text pairs.
     - At test time, embed prompts "a photo of a {class}". The normalized text embeddings act as classifier weights, with logit scale/temperature.
     - Prompt engineering and ensembling give accuracy gains.
     - Robustness claims under distribution shift.
     - Caveat: test classes may appear in web pretraining data, so it isn't zero-shot in the classic sense.
  7. *In-context learning*: few-shot prompting in LLMs. Sensitivity to example order and format. Whether ICL is implicit fine-tuning (named; pointer to the LLM area).
  8. *Evaluation pitfalls*: class overlap between pretraining and "unseen" classes, transductive vs inductive settings, variance across episodes (report CIs).
- **Question ideas**
  - (calc) Prototypes: class A support embeddings (0,0), (2,0) → c_A = (1,0); class B (0,4), (0,2) → c_B = (0,3); query (1,1) → class A (squared distances 1 vs 5).
  - Is GPT-3's "few-shot" setting a gradient-based adaptation? (no)
  - In CLIP zero-shot classification, what plays the role of the classifier weights? (normalized text embeddings of the prompts)
  - Which is false: "MAML's exact meta-gradient requires only first derivatives".
  - Predict: Tian et al.'s finding when comparing ProtoNets with a linear probe on a good embedding.
  - Why is CLIP's "zero-shot" contested as a term?
- **Figures**
  - `fund.few-zero-shot/prototypes`: 2-D embeddings with class prototypes and the Voronoi decision regions for a query. **[fig-Q]**
  - `fund.few-zero-shot/maml-landscape`: a shared init θ with one-step adaptation arrows to task optima.
  - `fund.few-zero-shot/clip-zero-shot`: an image embedding scored against prompt embeddings (diagram).
- **Pitfalls & source disagreements**
  - The three meanings of "few-shot".
  - The "zero-shot" contamination caveat (CLIP vs classic ZSL definitions).

---

#### 620 · `fund.distribution-shift`: Distribution Shift: Covariate, Label & Concept
- **Level** advanced · **W3** · **prereqs** [fund.learning-setups, fund.bayes-theorem, fund.monte-carlo] · **covers** domain adaptation (foundations).
- **Sources**
  - Murphy PML2 ch.19 beyond iid:
    - §19.2 distribution shift incl. §19.2.2 the causal view and §19.2.3 the four main types (pdf p.769–774).
    - §19.3 detecting shifts via two-sample tests (pdf p.774–779).
    - §19.5.2 weighted ERM for covariate shift (pdf p.782).
    - §19.5.4 unsupervised label-shift techniques (pdf p.784).
    - `pml2.txt`
  - d2l environment & distribution shift (`d2l/d2l_environment_and_distribution_shift.txt`): covariate/label/concept shift examples, covariate shift correction via a domain classifier, label shift correction via the confusion matrix, nonstationary environments.
  - Lipton, Wang & Smola 2018 BBSE (`papers/lipton2018_label_shift.txt`): assumptions, the estimator, consistency.
  - Murphy PML2 §2.7.5 density-ratio estimation via binary classifiers (pdf p.95). `pml2.txt`
- **Subtopic map**
  1. *Why iid fails in deployment*: time drift, new populations, sensor changes, feedback loops. Training/test notation $p_S(x,y)$ vs $p_T(x,y)$.
  2. *Taxonomy* via factorization:
     - Covariate shift: p(x) changes, p(y|x) fixed.
     - Label/prior shift: p(y) changes, p(x|y) fixed.
     - Concept shift: p(y|x) changes.
     - Conditional shift (named).
     - A real example of each.
  3. *Causal view* (PML2 §19.2.2): covariate shift is natural when x causes y. Label shift is natural when y causes x (disease → symptoms).
  4. *Covariate shift correction*:
     - Derive $E_T[\ell]=E_S[w(x)\ell]$ with $w=p_T(x)/p_S(x)$ (importance sampling, from fund.monte-carlo).
     - Estimate w with a domain classifier: $w(x)=\frac{P(T|x)}{P(S|x)}\cdot\frac{n_S}{n_T}$ (derive via Bayes).
     - Requirements: support overlap, plus variance blow-up and effective sample size $(\sum w)^2/\sum w^2$.
     - Clipping/normalizing weights.
     - It helps most for misspecified models (Shimodaira, statement).
  5. *Label shift correction*:
     - Posterior adjustment $p_T(y|x)\propto p_S(y|x)\,p_T(y)/p_S(y)$ (derive).
     - Estimating $p_T(y)$ without labels: **BBSE** solves $C\,w=\hat\mu_T$, where $C_{ij}=p_S(\hat y=i,y=j)$ and $\hat\mu_T$ is the predicted-label distribution on target. Its assumptions.
     - EM-based prior re-estimation (Saerens, named).
  6. *Concept shift*: you need new labels (continual learning, periodic retraining). It can't be fixed without target supervision.
  7. *Detecting shift*:
     - Two-sample tests (MMD, KS on features).
     - **Classifier two-sample test**: if a classifier separates source from target above chance, there is a shift.
     - Monitoring prediction distributions in production.
  8. *Robustness vs adaptation*: DRO and invariance-based methods (named), vs adapting with unlabelled target data (next lesson).
- **Question ideas**
  - Classify: hospital with different disease prevalence, same symptom distribution given disease → label shift. New camera with different colour response → covariate shift. Spam definitions change → concept shift.
  - (calc) Domain classifier P(T|x) = 0.8, equal sample sizes → w = 4.
  - (calc) Label-shift correction: source prior 0.5/0.5, target $p_T(y{=}1)=0.9$, $p_S(y=1|x)=0.5$ → $p_T(y=1|x)=0.9$.
  - (calc) Effective sample size for weights (1, 1, 1, 9): 144/84 ≈ 1.71.
  - Which is false: "importance weighting can correct covariate shift even when the target has regions with zero source density".
- **Figures**
  - `fund.distribution-shift/shift-types`: three panels showing p(x) and p(y|x) for covariate, label and concept shift. **[fig-Q]**: "which shift is this?"
  - `fund.distribution-shift/importance-weights`: source vs target densities and the weight function w(x), blowing up in the tails.
  - `fund.distribution-shift/bbse`: confusion-matrix-based prior estimation as a small linear-system diagram.
- **Pitfalls & source disagreements**
  - Terminology varies: dataset shift, domain shift, covariate shift. "Concept drift" sometimes covers all of them.

---

#### 630 · `fund.domain-adaptation`: Unsupervised Domain Adaptation: Theory & Methods
- **Level** advanced · **W3** · **prereqs** [fund.distribution-shift, fund.kl-divergence] · **covers** domain adaptation.
- **Sources**
  - Ben-David et al. 2010 (`papers/bendavid2010_da_theory.txt`): Thm 1 (L1/TV bound), the $\mathcal H\Delta\mathcal H$-divergence, Thm 2 (finite-sample bound with λ), combining source and target data.
  - Ganin et al. 2016 DANN (`papers/ganin2016_dann.txt`): §3.2 the proxy A-distance (PAD, from Ben-David et al. 2006), the domain-adversarial objective, gradient reversal layer, the λ schedule, the link to the HΔH bound.
  - Murphy PML2 §19.5.3 UDA for covariate shift, §19.5.5 test-time adaptation, §19.6.2 domain generalization (pdf p.783–792), §26.7.6 adversarial domain adaptation (pdf p.958). `pml2.txt`
  - Murphy PML1 §19.2.5 domain adaptation (pdf p.667), §19.3.1–19.3.2 self-training & entropy minimization (pdf p.668–672). `pml1.txt`
  - Goodfellow DL §15.2. `dlb_ch15_representation.txt`
- **Subtopic map**
  1. *Setting*: labelled source, **unlabelled** target, the same label space. Contrast with supervised DA (some target labels; fine-tuning) and domain generalization (no target data).
  2. *Ben-David bound*: $\varepsilon_T(h)\le\varepsilon_S(h)+\frac12d_{\mathcal H\Delta\mathcal H}(D_S,D_T)+\lambda$ with $\lambda=\min_h[\varepsilon_S(h)+\varepsilon_T(h)]$.
     - Explain each term: source error; how distinguishable the domains are *to the hypothesis class*; whether any hypothesis is good on both.
     - The proxy A-distance $2(1-2\epsilon_{dom})$ from a domain classifier's error.
  3. *What the bound suggests*: learn a representation where the domains are indistinguishable (small divergence) while keeping the source error low, and hope λ stays small.
  4. *DANN*:
     - Feature extractor $G_f$, label predictor $G_y$, domain classifier $G_d$.
     - Minimax objective.
     - **Gradient reversal layer**: identity in the forward pass, gradient × −λ in the backward pass.
     - The training schedule for λ.
     - Equivalence to minimizing a proxy HΔH divergence.
  5. *Failure mode*: aligning marginal feature distributions can **misalign classes** when label proportions differ across domains, so λ grows. Fixes: class-conditional alignment and label-shift-aware weighting (named).
  6. *Other UDA families*:
     - Moment matching (MMD, CORAL: align second-order statistics).
     - Self-training / pseudo-labelling with confidence thresholds.
     - Entropy minimization.
     - BN-statistics re-estimation (AdaBN).
     - Test-time adaptation (TENT: entropy minimization on BN affine params).
  7. *Domain generalization*: train on multiple source domains to generalize to unseen ones (invariance methods such as IRM, and DRO, named). Strong ERM baselines are hard to beat (named).
  8. *Evaluation pitfalls*: hyperparameter selection without target labels is the hidden difficulty (reverse validation, named). Reporting oracle-tuned numbers inflates results.
- **Question ideas**
  - Name each term of the Ben-David bound. Which one can a representation-learning method reduce directly? (the divergence)
  - (calc) Domain classifier error 0.3 → proxy A-distance 2(1 − 0.6) = 0.8.
  - What does the gradient reversal layer do in the forward vs backward pass?
  - Spot the flaw: "aligning feature marginals guarantees good target accuracy" (λ can grow; label shift).
  - Predict: TENT on a batch from a shifted domain (adapts the BN affine params by entropy minimization).
  - Which is false: "UDA methods can be tuned on target validation labels in a realistic deployment".
- **Figures**
  - `fund.domain-adaptation/dann-architecture`: feature extractor feeding a label head and a domain head through the GRL. **[fig-Q]**
  - `fund.domain-adaptation/feature-alignment`: t-SNE-style 2-D features of source and target before and after alignment, including the class-misalignment failure.
  - `fund.domain-adaptation/bound-terms`: stacked bar of source error + divergence + λ for three representations.
- **Pitfalls & source disagreements**
  - Ben-David has two theorems with different divergences (TV/L1 vs HΔH) and constants. Cite which one.
  - DANN's λ (adversarial weight) vs Ben-David's λ (joint optimal error) is a symbol clash.

---

---

# Part B: Generative Modeling (`area: genmodels`, prefix `gen.`)

## B.1 Topic sequence (22 lessons)

| order | id | title | level | wave | prereqs | status |
|---|---|---|---|---|---|---|
| 10 | gen.overview | Generative Models: Taxonomy & Trade-offs | core | W2 | fund.mle, fund.kl-divergence | new |
| 20 | gen.autoregressive | Autoregressive Models: PixelCNN & WaveNet | core | W2 | gen.overview, fund.cnn | new |
| 30 | gen.latent-variables-elbo | Latent-Variable Models, Variational Inference & the ELBO | core | **W1** | fund.kl-divergence, fund.mle | new |
| 40 | gen.vae | Variational Autoencoders | core | **W1** | gen.latent-variables-elbo, fund.autoencoders | new |
| 50 | gen.vae-variants | β-VAE, Posterior Collapse, IWAE & VQ-VAE | intermediate | W2 | gen.vae | new |
| 60 | gen.gans | Generative Adversarial Networks | core | **W1** | gen.overview, fund.kl-divergence | new |
| 70 | gen.wgan | Wasserstein GANs, Lipschitz Constraints, Gradient Penalties & f-GANs | intermediate | W2 | gen.gans | new |
| 80 | gen.flows | Normalizing Flows | intermediate | W2 | gen.overview, fund.probability-basics | new |
| 90 | gen.ebm-score-matching | Energy-Based Models, Score Matching & Tweedie | advanced | W2 | gen.overview, fund.autoencoders | new |
| 100 | gen.ddpm-forward-reverse | Diffusion I: Forward Noising & the Reverse Posterior | core | **W1** | gen.latent-variables-elbo, fund.gaussian | new |
| 110 | gen.ddpm-objective | Diffusion II: The ELBO & the ε-Prediction Loss | core | **W1** | gen.ddpm-forward-reverse | new |
| 120 | gen.diffusion-parameterizations | Diffusion III: Parameterizations, SNR & Noise Schedules | core | **W1** | gen.ddpm-objective | new (split from ddpm-objective) |
| 130 | gen.score-sde | Score-Based Diffusion & the SDE/ODE View | advanced | W2 | gen.ebm-score-matching, gen.diffusion-parameterizations | new |
| 140 | gen.ddim | Fast Sampling I: DDIM | advanced | W2 | gen.score-sde | new (was gen.ddim-solvers) |
| 150 | gen.diffusion-solvers | Fast Sampling II: ODE Solvers & the EDM Design Space | advanced | W2 | gen.ddim | new (split from ddim-solvers) |
| 160 | gen.distillation | Diffusion Distillation & Consistency Models | advanced | W2 | gen.diffusion-solvers | new (W3 → W2 in review: few-step generation is a standard probe at image/video labs) |
| 170 | gen.guidance | Classifier & Classifier-Free Guidance | intermediate | **W1** | gen.diffusion-parameterizations | new |
| 180 | gen.flow-matching | Flow Matching | advanced | **W1** | gen.flows, gen.diffusion-parameterizations | new |
| 190 | gen.rectified-flow | Rectified Flow & the Diffusion–Flow Equivalence | advanced | W2 | gen.flow-matching, gen.score-sde | new |
| 200 | gen.latent-diffusion | Latent Diffusion & Conditioning | intermediate | W2 | gen.vae-variants, gen.guidance | new |
| 210 | gen.diffusion-transformers | Diffusion Backbones: U-Net, DiT & MM-DiT | intermediate | W2 | gen.latent-diffusion | new (added in review) |
| 220 | gen.evaluation | Evaluating Generative Models | intermediate | W2 | gen.overview, gen.gans | new (moved to the end) |

W1 count: 8 of 22 (36%).

Prereq notes:
- `gen.latent-variables-elbo` no longer requires `fund.gmm-em` (W2). EM is a cross-link, not a gate: the ELBO lesson must be fully
  self-contained, because the reader may meet the ELBO here first.
- `gen.vae` teaches the reparameterization trick inline. `fund.gradient-estimators` (W3) and `fund.gumbel-softmax` (W3) are
  pointers from `gen.vae`/`gen.vae-variants`, not prereqs, so no W1/W2 lesson depends on a W3 lesson.
- `gen.flow-matching` (W1) keeps `gen.flows` (W2) as a prereq, but it must recap CNFs and the instantaneous change of
  variables inline (its items 1–2), so it can be written and read before `gen.flows`.
- Evaluation moved to the end because its content (mode collapse, guidance trade-off curves, diffusion memorization) refers
  to lessons that come later in the sequence.

## B.2 Coverage of the requested draft topics
| draft topic | lessons |
|---|---|
| gen.overview (taxonomy, trade-offs, evaluation) | gen.overview + gen.evaluation (split; evaluation now last) |
| gen.vae (LVMs, ELBO both forms, reparam, Gaussian KL, amortization gap, posterior collapse, β-VAE, VQ-VAE, IWAE) | gen.latent-variables-elbo → gen.vae → gen.vae-variants |
| gen.gans (minimax, optimal D/JSD, non-saturating, mode collapse, WGAN/KR duality, clipping/GP/R1/SN, f-GAN, conditional, why diffusion replaced GANs) | gen.gans → gen.wgan |
| gen.flows (change of variables, coupling/RealNVP/Glow, MAF vs IAF, CNFs) | gen.flows (CNFs revisited in gen.flow-matching) |
| gen.autoregressive | gen.autoregressive |
| gen.ddpm (forward closed form, reverse posterior, ELBO → ε-loss, weighting, variances) | gen.ddpm-forward-reverse → gen.ddpm-objective |
| parameterizations (ε/x0/v/score), SNR view, schedules, loss weightings | gen.diffusion-parameterizations |
| gen.score-based (score matching, DSM = Tweedie, Langevin, NCSN, SDE, reverse SDE, PF-ODE) | gen.ebm-score-matching → gen.score-sde |
| gen.fast-sampling (DDIM, ODE solvers, DPM-Solver/Heun/EDM incl. preconditioning, distillation, consistency) | gen.ddim → gen.diffusion-solvers → gen.distillation |
| gen.guidance (incl. CFG derivation, over-saturation and its fixes) | gen.guidance |
| gen.flow-matching (CFM, marginal-field proof, OT/rectified, relation to diffusion, v-prediction) | gen.flow-matching → gen.rectified-flow |
| gen.latent-diffusion (AE latent space, LDM/SD, cross-attention, why latent) | gen.latent-diffusion |
| diffusion backbones (U-Net, DiT/adaLN-Zero, MM-DiT; linked to the vision area) | gen.diffusion-transformers (the vision area should link here for DiT, and own ViT and video diffusion) |
| discrete / masked generative models (MaskGIT, D3PM, LLaDA) | named only in gen.overview (deferred, as in the LLM plan) |

## B.3 Additional sources for Part B (all in `$SRC`)
- **Textbook spine**:
  - Murphy PML2 (`pml2.txt`): ch.20 overview & evaluation (pdf p.807–823), ch.21 VAEs (pdf p.825–853), ch.22 AR (pdf p.855–862), ch.23 flows (pdf p.863–881), ch.24 EBMs & score matching (pdf p.883–899), ch.25 diffusion (pdf p.901–925), ch.26 GANs (pdf p.927–958), §10.1–10.2 VI (pdf p.473–488).
  - Goodfellow DL ch.20 (`dlb_ch20_generative_models.txt`).
  - CS229 notes ch.14 diffusion (pdf p.181–191): "The diffusion process" (p.181), "Parameterizing the reverse process" (p.184), "Training diffusion models by maximizing the ELBO" (p.185), "Continuous-time view of reverse diffusion" (p.189). Ch.11 EM/ELBO/VAE (pdf p.151–167).
- **Tutorial monograph**: Luo 2022 "Understanding Diffusion Models: A Unified Perspective" (`papers/luo2022_diffusion_unified.txt`). Its derivations are careful and match Ho et al.'s notation. **Luo's sections are unnumbered**: cite them by title and page, e.g. `Luo 2022, "Variational Diffusion Models" (p.6–14)`. Pages: "Evidence Lower Bound" p.2, "Variational Autoencoders" p.4, "Hierarchical Variational Autoencoders" p.5, "Variational Diffusion Models" p.6, "Learning Diffusion Noise Parameters" p.14, "Three Equivalent Interpretations" p.15 (incl. Tweedie), "Score-based Generative Models" p.17, "Guidance" p.20–21.
- **Secondary blogs** (intuition only; `web/`): Weng on VAEs, GANs, flows and diffusion; Y. Song on score-based models; Dieleman on perspectives and guidance; Gao et al. 2024 on diffusion ↔ flow matching.
- **Not cached**: Bishop & Bishop 2024 chs. 16–20 (viewer only). Anderson 1982 reverse-time SDE (paywalled; use Song et al. 2021 App. A and PML2 §25.4.3).
- **Not cached but needed** (open arXiv papers; the coordinator should cache them before the W2 lessons that use them are written; arXiv ids to be confirmed on fetch). Until then the items they support stay "named, with a one-line reason", never derived from memory:
  - Peebles & Xie 2023, *Scalable Diffusion Models with Transformers* (DiT), arXiv 2212.09748 → gen.diffusion-transformers.
  - Mescheder, Geiger & Nowozin 2018, *Which Training Methods for GANs do actually Converge?* (R1/R2, Dirac-GAN), arXiv 1801.04406 → gen.wgan, gen.gans. (PML2 §26.3.5 covers the Dirac-GAN example and is cached.)
  - Saharia et al. 2022, Imagen (dynamic thresholding), arXiv 2205.11487 → gen.guidance.
  - Lin et al. 2024, *Common Diffusion Noise Schedules and Sample Steps are Flawed* (zero terminal SNR, CFG rescale), arXiv 2305.08891 → gen.diffusion-parameterizations, gen.guidance.
  - Kynkäänniemi et al. 2024, *Applying Guidance in a Limited Interval*, arXiv 2404.07724 → gen.guidance.

## B.4 Notation clash table (put a version of this in the relevant lessons)
Diffusion notation is the main source of errors in this area. Every diffusion lesson should state which convention it uses,
and convert with this table. Formulas were checked against the cached papers.

| quantity | Ho et al. 2020 (DDPM) | Song et al. DDIM 2021 | NCSN 2019 / Song SDE 2021 | Kingma VDM 2021 / Salimans & Ho 2022 / Ho & Salimans CFG 2022 | Karras EDM 2022 | Lipman FM 2023 / Liu RF 2023 | Esser SD3 2024 |
|---|---|---|---|---|---|---|---|
| noisy sample | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ | $x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$ (**their $\alpha_t$ = Ho's $\bar\alpha_t$**, App. C.2) | VE: $x+\sigma z$; VP: as Ho | $z_t=\alpha_tx+\sigma_t\epsilon$ (**$\alpha_t$ = Ho's $\sqrt{\bar\alpha_t}$**), VP: $\alpha_t^2+\sigma_t^2=1$ | $x=s(t)\,(y+\sigma(t)n)$, usually $s=1$, $\sigma(t)=t$ | $x_t=tx_1+(1-t)x_0$ (σ_min→0) | $z_t=(1-t)x_0+t\epsilon$ |
| clean data symbol | $x_0$ | $x_0$ | $x$ / $x(0)$ | $x$ | $y$ (and $x$ = noisy!) | **$x_1$** ($x_0$ = noise) | $x_0$ |
| time direction | t=0 data → t=T noise | same | t=0 data → t=1 noise (SDE) | t=0 data → t=1 noise | σ→0 data, σ_max noise | **t=0 noise → t=1 data** | t=0 data → t=1 noise |
| noise-level index | t=1 least noisy | same | **NCSN: σ_1 largest** (σ_1>…>σ_L); **Song SDE: σ_1 smallest** (σ_min=σ_1<…<σ_N) | continuous t | σ directly | continuous t | continuous t |
| log-SNR symbol | (none; SNR = $\bar\alpha_t/(1-\bar\alpha_t)$) | (none) | (none) | VDM: $\gamma(t)=-\log\mathrm{SNR}$; Salimans & Ho, Ho & Salimans: $\lambda=\log\mathrm{SNR}=\log(\alpha^2/\sigma^2)$ | uses σ; log-normal on $\ln\sigma$ | (none) | $\lambda_t=\log(a_t^2/b_t^2)$ = log-SNR |
| VP schedule | discrete β_1=1e-4 … β_T=0.02, T=1000 | inherits Ho's | VP SDE: $\beta(t)=\bar\beta_{min}+t(\bar\beta_{max}-\bar\beta_{min})$, 0.1 → 20 (= T × Ho's β) | learned or cosine | n/a | n/a | n/a |
| regression target "velocity" | n/a | n/a | n/a | Salimans & Ho **v = αε − σx** ($=dz/d\phi$ for α=cos φ, σ=sin φ; points toward noise) | $F_\theta$ target $=(y-c_{skip}x)/c_{out}$ | **$u=x_1-x_0$ = data − noise** | **$v=\epsilon-x_0$ = noise − data** (opposite sign) |
| guidance | n/a | n/a | n/a | Ho & Salimans eq. 6: $\tilde\epsilon=(1+w)\epsilon_c-w\epsilon_u$ (**w=0 unguided**) | n/a | n/a | SD-style "CFG scale" s: $\epsilon_u+s(\epsilon_c-\epsilon_u)$ (**s=1 unguided**, s = 1+w); SD3 reports scales 1.0–5.0 (App. B.3) |

Also: **DPM-Solver (Lu et al. 2022) uses λ = log(α/σ) = ½ log-SNR**, half of VDM/Salimans–Ho's λ. Dhariwal & Nichol's classifier-guidance scale s multiplies $\nabla\log p_\phi(y|x_t)$, and is applied on top of a
conditional or unconditional model. Ho's ELBO terms are $L_T$, $L_{t-1}$ (1<t≤T) and $L_0$; Luo and others index the
denoising-matching terms as $L_t$. Networks often output ε, v or σ·score rather than the score itself.

---

## B.5 Lesson plans

---

#### 10 · `gen.overview`: Generative Models: Taxonomy & Trade-offs
- **Level** core · **W2** · **prereqs** [fund.mle, fund.kl-divergence].
- **Sources**
  - Murphy PML2 §20.1–20.3 types and goals of generative models (pdf p.807–816), §20.5 training objectives (pdf p.822). `pml2.txt`
  - Goodfellow 2016 NIPS GAN tutorial §2 (`papers/goodfellow2016_gan_tutorial.txt`): why study generative models, and the taxonomy of maximum-likelihood models (explicit tractable, explicit approximate, implicit).
  - Song & Kingma 2021 (`papers/song2021_train_ebm.txt`): EBMs and the three training routes (MLE+MCMC, score matching, NCE).
  - Goodfellow DL ch.20 intro and §20.10.2 differentiable generator networks. `dlb_ch20_generative_models.txt`
  - Dieleman 2023 "Perspectives on diffusion" (`web/dieleman2023_perspectives.txt`), secondary, for the unifying views.
- **Subtopic map**
  1. *What a generative model is*: learning p(x) (or p(x|c)), and the uses: sampling, density evaluation, representation, imputation, compression, simulation. Unconditional vs conditional.
  2. *Taxonomy by how p(x) is represented*:
     - Explicit tractable: autoregressive, normalizing flows.
     - Explicit approximate: VAEs and diffusion (ELBO).
     - Unnormalized: energy-based models, $p\propto e^{-E}$, with an intractable partition function.
     - Implicit: GANs (a sampler only).
     - Score-based: model ∇log p.
  3. *What each optimizes*: forward KL / MLE (AR, flows, VAE bound, diffusion bound) vs adversarial divergences (JS, Wasserstein, f-divergences) vs score matching. Mode covering vs mode seeking, and the consequences (blur vs dropped modes; pointer to fund.kl-divergence).
  4. *The trilemma*: sample quality, coverage/diversity, sampling speed. GANs are fast and sharp but weak on coverage. VAEs are fast with coverage but blurry. Diffusion has quality and coverage but is slow. Also: exact likelihood, a latent space, controllability, training stability.
  5. *Latent-variable models*: $p(x)=\int p(x|z)p(z)dz$, why the marginal is intractable, and what posterior inference means (preview of gen.latent-variables-elbo).
  6. *EBMs and discrete models in brief*: an EBM's MLE gradient needs samples from the model, which is why it is hard to train (one sentence; the derivation lives in gen.ebm-score-matching, don't duplicate it). Discrete/masked generative models (VQ tokens + AR or masked transformers such as MaskGIT; discrete diffusion, PML2 §25.7; LLaDA, `llm_nie2025_llada.txt`) named in one paragraph as where the field is heading. Not developed further (deferred, as in the LLM plan).
  7. *Sampling cost*: AR is O(D) sequential steps; diffusion is O(#steps) network evaluations; GAN/VAE/flow is one pass.
  8. *Unifying views*: diffusion as a hierarchical VAE with a fixed encoder, as a score model, as a continuous flow. Flow matching with Gaussian paths ≈ diffusion. AR transformers over discrete tokens (pointer to the LLM area).
  9. *Historical arc*: RBMs/DBNs → VAE & GAN (2014) → flows (2015–18) → diffusion (2020–) → latent diffusion, DiTs and flow matching. What drove each shift.
- **Question ideas**
  - Which families give exact log-likelihoods? (AR, flows; diffusion via the probability-flow ODE, but trained on a bound)
  - Match each property to a family: "sampler only, no density" → GAN; "intractable partition function" → EBM.
  - Predict: a model trained with a mode-seeking objective on multimodal data (drops modes).
  - Predict: a GAN, a VAE and a diffusion model trained on the same 8-mode 2-D mixture. Which drops modes, which blurs between them, and which is slowest to sample?
  - Rank the sampling cost for a 1024-token image: AR vs a 50-step diffusion vs a GAN.
- **Figures**
  - `gen.overview/taxonomy-tree`: tree of families split by how p(x) is represented. **[fig-Q]**: "where does a VQ-VAE + transformer prior sit?"
  - `gen.overview/trilemma`: a triangle (quality, diversity, speed) with families placed.
  - `gen.overview/mode-cover-vs-seek`: a 2-D mixture of 8 Gaussians, with a "VAE-like" blurry fit vs a "GAN-like" mode-dropping fit.
- **Pitfalls & source disagreements**
  - Taxonomies differ (Goodfellow's explicit/implicit split vs PML2's).
  - "Diffusion is likelihood-based" is only partly true: it is trained on a reweighted ELBO.
  - "VAE blur comes from the KL term" is contested; the Gaussian decoder likelihood is a main cause.

---

#### 20 · `gen.autoregressive`: Autoregressive Models: PixelCNN & WaveNet
- **Level** core · **W2** · **prereqs** [gen.overview, fund.cnn].
- **Sources**
  - Murphy PML2 ch.22: §22.1 introduction, §22.2 NADE, §22.3 causal CNNs (1-D and PixelCNN), §22.4 transformers (pdf p.855–862). `pml2.txt`
  - van den Oord, Kalchbrenner & Kavukcuoglu 2016 PixelRNN (`papers/vandenoord2016_pixelrnn.txt`): masked convolutions (type A/B), the 256-way softmax over pixel values, RGB channel ordering.
  - van den Oord et al. 2016 Gated PixelCNN (`papers/vandenoord2016_pixelcnn_decoders.txt`): the blind spot, vertical and horizontal stacks, gated activations, conditioning.
  - van den Oord et al. 2016 WaveNet (`papers/vandenoord2016_wavenet.txt`): dilated causal convolutions, receptive field, μ-law quantization, gated units, global/local conditioning.
  - Goodfellow DL §20.10.7–20.10.10 auto-regressive networks, linear and neural AR, NADE. `dlb_ch20_generative_models.txt`
- **Subtopic map**
  1. *Chain-rule factorization* $p(x)=\prod_ip(x_i|x_{<i})$: exact likelihood with no approximation. Any ordering is valid, but the choice matters (raster scan for images). Each conditional is a classification (or density) problem.
  2. *Training vs sampling asymmetry*: training is parallel via masking (teacher forcing: all conditionals in one pass), sampling is sequential with D network evaluations (calc for 32×32×3).
  3. *Parameter sharing across conditionals*: from fully visible belief nets to NADE (shared weights, O(D·H)) to masked networks (MADE, named), causal convolutions and causal attention.
  4. *PixelCNN*:
     - Masked convolution kernels: mask **A** (first layer, excludes the current pixel) vs **B** (later layers, includes it). Explain why A must come first.
     - RGB within-pixel ordering.
     - The receptive-field **blind spot** and its fix with vertical + horizontal stacks (Gated PixelCNN).
  5. *Output distributions*: a 256-way softmax per sub-pixel (no ordinal assumption, but it learns smoothness) vs the discretized logistic mixture (PixelCNN++, named).
  6. *WaveNet*:
     - Causal 1-D convs with dilations doubling 1, 2, 4, …, 512, so the receptive field grows exponentially (calc 1024).
     - μ-law companding to 256 classes.
     - Gated activation tanh ⊙ σ.
     - Residual and skip connections.
     - Conditioning.
  7. *Transformers as AR models*: causal masks, KV caching for faster sampling. Discrete tokens from VQ-VAEs enable AR image models (pointer to gen.vae-variants and the LLM area).
  8. *Sampling controls and failure modes*: temperature, top-k/top-p (pointer to the LLM area), exposure bias, error accumulation.
  9. *Strengths and weaknesses*: the best likelihoods among classic models, but slow sampling, no global latent, and order dependence.
- **Question ideas**
  - Why mask A in the first layer and mask B afterwards?
  - (calc) WaveNet with kernel 2 and dilations 1…512: receptive field 1024 samples.
  - (calc) Sequential network evaluations to sample a 32×32×3 image with PixelCNN: 3072.
  - Training is parallel but sampling is sequential. Why?
  - **[fig-Q]** Receptive-field diagram of a stacked PixelCNN with the blind spot: which pixels can't influence the current one?
  - Which is false: "autoregressive models require approximate inference to compute log p(x)".
- **Figures**
  - `gen.autoregressive/masks-ab`: 5×5 kernel masks A and B.
  - `gen.autoregressive/blind-spot`: the receptive field of stacked masked convs showing the blind spot. **[fig-Q]**
  - `gen.autoregressive/dilated-stack`: WaveNet's dilated causal convolution stack.
- **Pitfalls & source disagreements**
  - PixelCNN vs Gated PixelCNN vs PixelCNN++ differ in masks, outputs and architecture. Specify which one you mean.

---

#### 30 · `gen.latent-variables-elbo`: Latent-Variable Models, Variational Inference & the ELBO
- **Level** core · **W1** · **prereqs** [fund.kl-divergence, fund.mle]. Cross-link (not a prereq): fund.gmm-em, which derives the same decomposition for EM. This lesson must not assume the reader has done EM.
- **Card budget** (about 9): problem + tiny example · refreshers (Jensen, KL ≥ 0, importance sampling) · Jensen derivation · exact identity · consequences & EM link · the two rewritings · variational families & amortization · worked linear-Gaussian example · probes + key results. Gradient estimation (item 8) is one paragraph that hands off to gen.vae. Mean-field coordinate-ascent updates are out of scope (behaviour only).
- **Sources**
  - Kingma & Welling 2019 "An Introduction to VAEs" (`papers/kingma2019_vae_intro.txt`): §1.7–1.8 deep latent-variable models and their intractabilities (there is no §1.9); §2.1–2.2 the encoder/inference model and the ELBO; §2.2.1 "Two for One" (the ELBO as both approximate MLE and approximate inference; the gap = posterior KL); amortized inference.
  - Murphy PML2 §10.1 VI: §10.1.1 the variational objective, §10.1.2 form of q, §10.1.3 variational EM, §10.1.5 amortized VI (pdf p.473–479). `pml2.txt`
  - Bishop PRML §9.4 EM in general (the ELBO/KL decomposition; pdf p.470), §10.1 variational inference incl. §10.1.1 mean field and §10.1.2 properties of factorized approximations (KL(q‖p) behaviour) (pdf p.482–490). `bishop.txt`
  - CS229 notes ch.11: Jensen, general EM, "other interpretation of ELBO", variational inference & VAE (pdf p.154–167). `cs229.txt`
  - Luo 2022, "Evidence Lower Bound" (p.2–4) and "Variational Autoencoders" (p.4–5) (`papers/luo2022_diffusion_unified.txt`; unnumbered sections). Weng VAE blog (`web/weng2018_vae.txt`), secondary.
- **Subtopic map**
  1. *The problem*: we believe the data come from hidden causes z, giving $p_\theta(x)=\int p_\theta(x|z)p(z)dz$. With a neural decoder this integral has no closed form, and naive Monte Carlo with $z\sim p(z)$ has enormous variance (almost no z explain a given x). The same intractability hits the posterior $p(z|x)=p(x|z)p(z)/p(x)$. Make this concrete with a tiny example.
  2. *Refreshers, inline*: Jensen's inequality for concave log (one-line convexity argument and the chord picture), KL ≥ 0 (from Jensen), and importance sampling ($E_p[f]=E_q[f\,p/q]$, and why a proposal close to the target has low variance). Pointers to fund.probability-basics and fund.kl-divergence, but the lesson must work without them.
  3. *ELBO derivation 1 (Jensen)*: $\log p(x)=\log E_{q(z|x)}[\frac{p(x,z)}{q(z|x)}]\ge E_q[\log p(x,z)-\log q(z|x)]$. Explain why we introduce q (importance-sampling view: a proposal that concentrates on plausible z).
  4. *ELBO derivation 2 (exact identity)*: $\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q(z|x)\|p(z|x))$ (derive line by line). Consequences:
     - The gap is exactly the posterior KL.
     - Maximizing the ELBO over q does approximate inference. Over θ it does approximate MLE.
     - The bound is tight iff q = the true posterior. That's EM's E-step (link fund.gmm-em).
  5. *Rewriting the ELBO*: $E_q[\log p(x|z)]-\mathrm{KL}(q(z|x)\|p(z))$, i.e. reconstruction minus prior-matching. Interpret each term. Also the "energy + entropy" form $E_q[\log p(x,z)]+H(q)$.
  6. *Variational families*: mean-field/factorized, Gaussian. The reverse-KL behaviour: q under-estimates posterior variance and is mode-seeking (Bishop §10.1.2).
  7. *Amortized inference*: instead of optimizing a separate $q_i$ per datapoint (classical VI), learn an encoder $q_\phi(z|x)$ shared across data. Cost and benefit. The amortization gap (preview of gen.vae).
  8. *Optimizing the ELBO*: gradients wrt θ are easy (Monte Carlo of $\nabla_\theta\log p_\theta(x,z)$). Gradients wrt φ need the reparameterization or score-function estimators (pointer to fund.gradient-estimators). This sets up the VAE.
  9. *Worked example*: a 1-D linear-Gaussian model ($z\sim N(0,1)$, $x|z\sim N(z,\sigma^2)$) with a Gaussian q. Compute the ELBO, the true log-likelihood and the gap for a couple of q's, showing the gap → 0 at the true posterior $N(\frac{x}{1+\sigma^2},\frac{\sigma^2}{1+\sigma^2})$.
- **Question ideas**
  - The ELBO gap equals? ($\mathrm{KL}(q(z|x)\|p(z|x))$)
  - Derivation step: in the Jensen derivation, which function's concavity is used? (log)
  - (calc) The worked linear-Gaussian example: for x=1, σ²=1, the posterior is N(0.5, 0.5) and $\log p(x)=\log N(1;0,2)\approx-1.516$. The ELBO at q = N(0, 1) (= the prior) is ≈ −1.919, so the gap ≈ 0.403 = KL(N(0,1)‖N(0.5,0.5)). At the posterior the ELBO equals −1.516 (gap 0).
  - Derivation step: in $\log p(x)=\mathrm{ELBO}+\mathrm{KL}(q\|p(z|x))$, which identity lets you write $\log p(x)=\log p(x,z)-\log p(z|x)$ for every z, and why can you then take $E_q$ of both sides?
  - Which is false: "maximizing the ELBO wrt q minimizes KL(p(z|x)‖q)" (it's the reverse KL).
  - Why amortize inference? (one encoder pass vs per-datapoint optimization; generalizes to new x)
  - Predict: posterior variance under a mean-field Gaussian q for a correlated true posterior (underestimated).
- **Figures**
  - `gen.latent-variables-elbo/elbo-gap`: bar or curve showing log p(x) = ELBO + KL for several q's in the linear-Gaussian example. Notice the gap closing at the true posterior. **[fig-Q]**
  - `gen.latent-variables-elbo/graphical-model`: plate diagram of z → x (generative) with the q(z|x) inference arrow dashed.
  - `gen.latent-variables-elbo/meanfield-underestimate`: a correlated 2-D Gaussian posterior with the reverse-KL mean-field fit inside it.
- **Pitfalls & source disagreements**
  - ELBO sign and naming: ELBO, variational free energy (negative), "VAE loss" (negative ELBO).
  - Bishop's 𝓛(q, θ) vs Kingma's $\mathcal L_{\theta,\phi}(x)$ notation.

---

#### 40 · `gen.vae`: Variational Autoencoders
- **Level** core · **W1** · **prereqs** [gen.latent-variables-elbo, fund.autoencoders]. The reparameterization trick is taught inline here. fund.gradient-estimators (W3) is a pointer for the REINFORCE comparison, not a prereq.
- **Sources**
  - Kingma & Welling 2014 (`papers/kingma2013_vae.txt`): §2.3 the SGVB estimator (two forms), §2.4 the reparameterization trick, §3 the VAE with Gaussian encoder, Appendix B the Gaussian KL in closed form.
  - Kingma & Welling 2019 (`papers/kingma2019_vae_intro.txt`): §2.3–2.5 the reparameterization-based estimator and factorized Gaussian posteriors, §2.5.1 full-covariance posteriors; §2.6 marginal-likelihood estimation by importance sampling (§2.7 is "Marginal Likelihood and ELBO as KL Divergences"); ch.3 beyond Gaussian posteriors (named).
  - Murphy PML2 §21.2 VAE basics: modelling assumptions, model fitting, comparison with autoencoders, "VAEs optimize in an augmented space" (pdf p.825–830). `pml2.txt`
  - Cremer, Li & Duvenaud 2018 (`papers/cremer2018_amortization_gap.txt`): approximation gap vs amortization gap.
  - Goodfellow DL §20.10.3 (`dlb_ch20_generative_models.txt`). CS229 notes ch.11 "Variational inference and variational auto-encoder" (pdf p.163). Weng blog (`web/weng2018_vae.txt`), secondary.
- **Subtopic map**
  1. *Model*: prior $p(z)=N(0,I)$, decoder $p_\theta(x|z)$ (Gaussian with mean $\mu_\theta(z)$, or Bernoulli/categorical), encoder $q_\phi(z|x)=N(\mu_\phi(x),\mathrm{diag}\,\sigma^2_\phi(x))$. Parameterize log σ² for positivity and stability.
  2. *Objective*: per-datapoint ELBO $E_{q_\phi}[\log p_\theta(x|z)]-\mathrm{KL}(q_\phi(z|x)\|p(z))$ (from the previous lesson).
  3. *Reparameterization trick*: $z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon$ with $\varepsilon\sim N(0,I)$ moves the randomness out of the parameter path, so ∇φ flows through μ and σ (pointer to fund.gradient-estimators for why REINFORCE would be too noisy).
  4. *Closed-form KL*: derive $\mathrm{KL}(N(\mu,\sigma^2)\|N(0,1))=\frac12(\mu^2+\sigma^2-1-\log\sigma^2)$ from $E_q[\log q-\log p]$ using $E_q[(z-\mu)^2]=\sigma^2$ and $E_q[z^2]=\mu^2+\sigma^2$; then sum over dimensions for a diagonal Gaussian (independence makes the KL additive). This is a deterministic, low-variance term. State the general $\mathrm{KL}(N(\mu_1,\sigma_1^2)\|N(\mu_2,\sigma_2^2))$ too, since diffusion reuses it.
  5. *Reconstruction term*:
     - With a Gaussian decoder of fixed variance σ_x², $-\log p=\frac{\|x-\mu_\theta(z)\|^2}{2\sigma_x^2}+$ const, so σ_x² sets the reconstruction/KL balance (an implicit β).
     - Bernoulli decoder → BCE.
     - One Monte Carlo sample of z per datapoint is enough in practice.
  6. *Training loop*: encoder → (μ, log σ²) → sample ε → z → decoder → loss = recon + KL → backprop. Pseudocode with shapes.
  7. *Generation*: sample z ~ N(0, I) and decode (use the decoder mean, or sample). Latent interpolation and arithmetic. The holes / prior-hole problem: the aggregate posterior $q(z)=E_x q(z|x)$ ≠ p(z).
  8. *Why samples are blurry*: a Gaussian likelihood with a factorized decoder averages over plausible outputs (MSE → mean). Remedies: more expressive decoders (AR, which risks posterior collapse), perceptual/adversarial losses (VQGAN, named), hierarchical latents.
  9. *Inference quality*:
     - Approximation gap (family too simple) vs amortization gap (encoder not optimal for each x; Cremer).
     - Fixes: richer q (flows/IAF, named; pointer to gen.flows), semi-amortized refinement.
     - Estimating log p(x) by importance sampling with q as the proposal.
  10. *AE vs VAE*: a stochastic encoder plus the KL regularizer gives a smooth, sampleable latent space. A VAE with zero KL weight is just a noisy AE.
- **Question ideas**
  - (calc) KL(N(μ=1, σ=1)‖N(0,1)) = 0.5. KL(N(0, σ=0.5)‖N(0,1)) = ½(0.25 − 1 − ln 0.25) ≈ 0.318.
  - Why sample ε instead of z directly? (the gradient path through μ and σ)
  - Predict: the effect of setting the decoder variance σ_x² very small (reconstruction dominates; acts like small β).
  - Which is false: "the VAE's KL term must be estimated by Monte Carlo".
  - Amortization gap vs approximation gap: define each, and give one fix for each.
  - Spot the flaw: summing the reconstruction loss over pixels but averaging the KL over latent dimensions (an implicit re-weighting).
- **Figures**
  - `gen.vae/architecture`: encoder → (μ, σ) → z = μ + σ⊙ε → decoder, with the ε input on the side and the gradient path highlighted.
  - `gen.vae/latent-space`: a 2-D latent space of a toy VAE with encoded classes and a decoded grid.
  - `gen.vae/aggregate-posterior-holes`: the aggregate posterior q(z) vs the prior N(0, I) contours, showing regions of the prior with no data. **[fig-Q]**
- **Pitfalls & source disagreements**
  - Sum vs mean reductions in recon and KL silently change the effective β. This is very common in code.
  - "Reparameterization trick" (Kingma) = "pathwise derivative" (stats literature).
  - Some texts present the VAE as an autoencoder plus noise. Kingma frames it as amortized VI for a deep LVM.

---

#### 50 · `gen.vae-variants`: β-VAE, Posterior Collapse, IWAE & VQ-VAE
- **Level** intermediate · **W2** · **prereqs** [gen.vae]. Straight-through and Gumbel-softmax are explained inline in one or two lines each; fund.gumbel-softmax (W3) is a pointer.
- **Sources**
  - Higgins et al. 2017 β-VAE (`papers/higgins2017_beta_vae.txt`): the constrained-optimization derivation, disentanglement claims, the metric.
  - Bowman et al. 2016 (`papers/bowman2015_posterior_collapse.txt`): posterior collapse with RNN decoders, KL annealing, word dropout.
  - Burda, Grosse & Salakhutdinov 2016 IWAE (`papers/burda2015_iwae.txt`): the K-sample bound, monotonicity, active units.
  - van den Oord, Vinyals & Kavukcuoglu 2017 VQ-VAE (`papers/vandenoord2017_vqvae.txt`): the codebook, nearest-neighbour quantization, straight-through gradient, the three-term loss, the learned prior.
  - Murphy PML2 §21.3.1 β-VAE, §21.4 avoiding posterior collapse (KL annealing, free bits, skip connections), §21.5 hierarchical VAEs incl. the very deep VAE and the AR connection, §21.6 VQ-VAE incl. VQ-VAE-2, dVAE, VQ-GAN (pdf p.830–853). `pml2.txt`
- **Subtopic map**
  1. *β-VAE*: objective $E_q[\log p(x|z)]-\beta\,\mathrm{KL}$. Derivation as a Lagrangian of "maximize reconstruction subject to KL < ε". β > 1 tightens the information bottleneck (claimed disentanglement, worse reconstructions). Rate–distortion view: −ELBO = D + R. Disentanglement claims are contested (named).
  2. *Posterior collapse*:
     - The KL → 0 for some or all latent dimensions, and the decoder ignores z.
     - Why: powerful autoregressive decoders can model x without z, and the KL term pushes q to the prior early in training ("information preference").
     - Diagnose with per-dimension KL / "active units".
  3. *Fixes*: KL annealing/warmup (β ramps 0 → 1), free bits (KL floor per dimension), word/input dropout, weaker or local decoders, skip connections, δ-VAE/rate lower bounds, better inference (PML2 §21.4).
  4. *IWAE*:
     - $\mathcal L_K=E\big[\log\frac1K\sum_{k=1}^K\frac{p(x,z_k)}{q(z_k|x)}\big]$.
     - $\mathcal L_1$ = ELBO ≤ $\mathcal L_K$ ≤ $\mathcal L_{K+1}$ ≤ log p(x) (prove the ordering via Jensen and symmetry). As K → ∞ it converges to log p(x).
     - It gives a tighter bound and more active units, at K× compute.
     - The encoder-gradient signal-to-noise issue at large K (named).
  5. *Hierarchical VAEs*: multiple latent layers with top-down inference (ladder VAE, NVAE, VDVAE, named). Link: diffusion = a hierarchical VAE with a fixed Gaussian encoder (pointer to gen.ddpm-objective).
  6. *VQ-VAE*:
     - Encoder output $z_e(x)$ is quantized to the nearest codebook vector $e_k$.
     - The decoder reconstructs from $z_q$.
     - Gradients: **straight-through** copies $\nabla_{z_q}$ to $z_e$ (argmin has zero/undefined gradient; in code, $z_q=z_e+\mathrm{sg}[z_q-z_e]$).
     - Loss $-\log p(x|z_q)+\|\mathrm{sg}[z_e]-e\|^2+\beta\|z_e-\mathrm{sg}[e]\|^2$ (the paper prints "log p" but means the reconstruction loss, i.e. −log p): codebook loss moves the codes, commitment loss (β = 0.25; results robust for 0.1–2.0, van den Oord §3.2) keeps the encoder near them.
     - **Who gets which gradient** (a common probe): the decoder gets only the reconstruction term; the encoder gets reconstruction (via straight-through) + commitment; the codebook gets only the codebook term (or EMA updates instead).
     - Why there's no KL term: a uniform prior over K codes and a deterministic posterior give a constant KL = log K.
  7. *VQ-VAE in practice*: codebook collapse and dead codes. EMA codebook updates and code resets. A learned AR prior over code indices (PixelCNN/transformer) for generation. VQ-VAE-2 hierarchy. dVAE with Gumbel-softmax (DALL·E 1). VQGAN (perceptual + adversarial losses) as the tokenizer behind latent/AR image models (pointer to gen.latent-diffusion).
- **Question ideas**
  - Predict: β-VAE with β=10 vs β=1 (blurrier reconstructions, more factorized latents, lower rate).
  - Diagnose: a text VAE with an LSTM decoder has KL ≈ 0 throughout training. What is it, and what are two fixes?
  - IWAE with K=1 equals? (the ELBO) As K → ∞? (log p(x))
  - VQ-VAE: where does the encoder's gradient come from? (straight-through from the decoder loss + commitment loss)
  - Which is false: "the VQ-VAE β plays the same role as β in β-VAE".
  - (calc) VQ-VAE with K=512 codes and a uniform prior: the KL term = log 512 ≈ 6.24 nats per latent position (a constant).
- **Figures**
  - `gen.vae-variants/rate-distortion`: rate vs distortion curve with β = 0.5, 1, 4 points.
  - `gen.vae-variants/kl-per-dim`: per-latent-dimension KL bars for a collapsed vs a healthy model. **[fig-Q]**
  - `gen.vae-variants/vq-diagram`: encoder → nearest codebook lookup → decoder, with the straight-through gradient arrow and the sg[·] operators.
  - `gen.vae-variants/iwae-bounds`: $\mathcal L_K$ vs K approaching log p(x) in a toy model.
- **Pitfalls & source disagreements**
  - The β in β-VAE (KL weight) vs the β in VQ-VAE (commitment weight) vs KL-annealing β are different.
  - "Posterior collapse" is sometimes used for partial collapse (some dimensions only).

---

#### 60 · `gen.gans`: Generative Adversarial Networks
- **Level** core · **W1** · **prereqs** [gen.overview, fund.kl-divergence].
- **Sources**
  - Goodfellow et al. 2014 (`papers/goodfellow2014_gan.txt`): the value function, Alg. 1, Prop. 1 (optimal D), Thm 1 (global optimum and the JSD), Prop. 2 (convergence in function space), the −log D heuristic.
  - Goodfellow 2016 tutorial (`papers/goodfellow2016_gan_tutorial.txt`): §3.2 cost functions (minimax, non-saturating, maximum-likelihood game), §3.3 DCGAN, §5 research frontiers incl. non-convergence and mode collapse.
  - Arjovsky & Bottou 2017 (`papers/arjovsky2017_principled_gan.txt`): perfect discriminators on disjoint/low-dimensional supports, vanishing gradients, the instability of −log D.
  - Murphy PML2 §26.1–26.3 (learning by comparison: density-ratio estimation with classifiers; GAN loss functions; gradient descent dynamics; challenges; improving optimization; convergence) (pdf p.927–946), §26.4 conditional GANs (pdf p.946), §26.6 architectures incl. regularization (pdf p.948–952). `pml2.txt`
  - Salimans et al. 2016 (`papers/salimans2016_improved_gan_is.txt`); Mirza & Osindero 2014 cGAN (`papers/mirza2014_cgan.txt`). Goodfellow DL §20.10.4. Weng GAN blog (`web/weng2017_gan.txt`), secondary.
- **Subtopic map**
  1. *Idea*: train a sampler G(z) without a likelihood by playing against a discriminator D(x) that tries to tell real from fake. Density-ratio estimation by classification (PML2 §26.2): an optimal classifier recovers $p_{data}/p_g$.
  2. *Minimax objective*: $\min_G\max_DV=E_{p_{data}}\log D(x)+E_{p_z}\log(1-D(G(z)))$. The log-likelihood of a binary classifier.
  3. *Optimal discriminator*: maximize pointwise $a\log D+b\log(1-D)$, giving $D^*=\frac{p_{data}}{p_{data}+p_g}$ (derive).
  4. *Plug in*: $C(G)=-\log4+2\,\mathrm{JSD}(p_{data}\|p_g)$ (derive). The global minimum is at $p_g=p_{data}$ with $D^*=\frac12$. This is the theoretical justification, which assumes an optimal D and infinite capacity.
  5. *Non-saturating generator loss* $-\log D(G(z))$: early on, D rejects fakes confidently and $\log(1-D)$ saturates (gradient ≈ 0). The alternative has strong gradients and the same fixed point. Its implicit divergence (Arjovsky & Bottou: KL(p_g‖p_data) − 2 JSD) explains mode-seeking and instability.
  6. *Training dynamics*: alternating/simultaneous gradient steps on a game, not a minimization. The bilinear example min_x max_y xy cycles or spirals (show it). Its GAN instance is the **Dirac-GAN** (data = δ₀, $G_\theta=\theta$, $D_\psi(x)=\psi x$; PML2 §26.3.5, pdf p.942–946): unregularized gradient descent circles the equilibrium instead of converging, which motivates gradient penalties (pointer to gen.wgan). There's no single loss that tracks progress. Two-timescale updates (TTUR, named).
  7. *Failure modes*:
     - Mode collapse: G maps many z to a few modes. Min–max vs max–min ordering intuition.
     - Vanishing gradients when D is too good.
     - Disjoint low-dimensional supports make JSD constant (Arjovsky & Bottou).
     - Oscillation.
     - Remedies (Salimans): feature matching, minibatch discrimination, historical averaging, label smoothing (one-sided).
  8. *Conditional GANs*: feed y to G and D (concatenation, projection discriminators). Pix2pix and CycleGAN in one line (named).
  9. *Architectures and tricks*: DCGAN guidelines, spectral norm and gradient penalties (pointer to gen.wgan), progressive growing, StyleGAN, the truncation trick (fidelity vs diversity).
  10. *Where GANs stand*: largely displaced by diffusion for image synthesis (stability, coverage, scaling), but adversarial losses survive in VQGAN tokenizers, super-resolution and adversarial diffusion distillation (pointer to gen.distillation).
- **Question ideas**
  - Derive D*(x) (flashcard). At $p_g=p_{data}$, D* = ? (½) V = ? (−log 4)
  - Why the non-saturating loss? (gradient strength when D rejects fakes confidently)
  - Predict: supports disjoint and D near-perfect under the minimax loss (generator gradient → 0).
  - Simultaneous GD on min_x max_y xy from (1, 1): what happens? (spirals outward / cycles)
  - Mode collapse in precision/recall terms? (high precision, low recall)
  - Which is false: "the GAN objective with an optimal discriminator minimizes the forward KL from data to model".
- **Figures**
  - `gen.gans/optimal-discriminator-1d`: two 1-D densities ($p_{data}$, $p_g$) with $D^*(x)$ overlaid. Notice D* = ½ where the densities are equal. **[fig-Q]**: "where is D* above ½?"
  - `gen.gans/saturating-vs-nonsaturating`: generator loss vs D(G(z)) for log(1−D) and −log D. Notice the flat gradient near D=0.
  - `gen.gans/bilinear-dynamics`: the spiral trajectory of simultaneous GD on xy.
  - `gen.gans/mode-collapse`: an 8-Gaussian ring with a collapsed generator covering 2 modes.
- **Pitfalls & source disagreements**
  - −log 4 uses natural logs.
  - Code often implements the non-saturating loss as BCE with fake samples labelled "real". It is mislabelled as "minimax" in many tutorials.

---

#### 70 · `gen.wgan`: Wasserstein GANs, Lipschitz Constraints, Gradient Penalties & f-GANs
- **Level** intermediate · **W2** · **prereqs** [gen.gans].
- **Sources**
  - Arjovsky, Chintala & Bottou 2017 WGAN (`papers/arjovsky2017_wgan.txt`): §2 different distances incl. the parallel-lines example, Thm 1 (W continuity), §3 Kantorovich–Rubinstein duality, Alg. 1 with weight clipping.
  - Gulrajani et al. 2017 WGAN-GP (`papers/gulrajani2017_wgan_gp.txt`): problems with clipping, the gradient penalty on interpolates (λ=10), no BN in the critic.
  - Miyato et al. 2018 spectral normalization (`papers/miyato2018_spectral_norm.txt`): the Lipschitz bound via layer spectral norms, power iteration.
  - Nowozin, Cseke & Tomioka 2016 f-GAN (`papers/nowozin2016_fgan.txt`): f-divergences, Fenchel conjugates, the variational lower bound, recovering GAN/KL objectives.
  - Murphy PML2 §26.2.3 bounds on f-divergences, §26.2.4 integral probability metrics, §26.2.5 moment matching (pdf p.932–936), §2.7.2–2.7.3 IPMs & MMD (pdf p.91–95), §6.8 optimal transport (pdf p.348). Weng GAN→WGAN blog (`web/weng2017_gan.txt`).
- **Subtopic map**
  1. *Why change the divergence*: JS (and KL) are discontinuous or saturated when supports don't overlap. Arjovsky's parallel-lines example: $P_0$ vs $P_\theta$ on lines at x=0 and x=θ have JS = log 2 and KL = ∞ for θ ≠ 0, but W1 = |θ| (compute).
  2. *Wasserstein-1 / earth mover's distance*: $W_1(P,Q)=\inf_{\gamma\in\Pi(P,Q)}E_\gamma\|x-y\|$. The transport-plan intuition, with a 1-D example where W1 = the area between the CDFs.
  3. *Kantorovich–Rubinstein duality*: $W_1=\sup_{\|f\|_L\le1}E_Pf-E_Qf$ (statement plus intuition). It turns W1 into an optimization over 1-Lipschitz "critics". It is an integral probability metric.
  4. *WGAN*: the critic $f_w$ maximizes $E_{data}f-E_gf$ and the generator minimizes $-E_{z}f(G(z))$. No sigmoid or log. The critic loss correlates with sample quality. Train the critic more steps (n_critic = 5).
  5. *Enforcing Lipschitz*:
     - **Weight clipping** (crude; capacity underuse and exploding/vanishing gradients depending on c).
     - **Gradient penalty** $\lambda E_{\hat x}(\|\nabla_{\hat x}f(\hat x)\|_2-1)^2$ on random interpolates $\hat x=\epsilon x+(1-\epsilon)\tilde x$. Why interpolates (the optimal critic has unit-norm gradients along transport lines). Why no BN in the critic (the per-sample penalty).
     - **Spectral normalization** $W/\sigma(W)$ per layer, with σ estimated by power iteration. Lipschitz of a composition ≤ the product of the layers' constants (1-Lipschitz activations).
     - **R1 / R2 zero-centred penalties** (Mescheder et al. 2018; not cached, see B.3): $\frac\gamma2E_{p_{data}}\|\nabla_xD(x)\|^2$ (R1, on real data) or on fakes (R2). Used with the standard non-saturating GAN loss, not only WGAN (StyleGAN family). Contrast with WGAN-GP: zero-centred vs one-centred, real data vs interpolates, and the motivation (local convergence of the game, shown on the Dirac-GAN, vs enforcing a Lipschitz critic). PML2 §26.6.5 (pdf p.951–952) surveys GAN regularizers and is the cached citation until the paper is fetched.
  6. *f-GAN*: refresher on the Fenchel conjugate $f^*(t)=\sup_u(tu-f(u))$ with one worked example ($f(u)=u\log u\Rightarrow f^*(t)=e^{t-1}$). Then $D_f(P\|Q)\ge\sup_TE_P[T]-E_Q[f^*(T)]$ (derive the bound from $f(u)=\sup_t(tu-f^*(t))$ applied pointwise to $u=p/q$). Choosing f recovers the original GAN (a JS variant), KL, reverse KL, etc.
  7. *Other IPMs*: MMD with kernels (MMD-GAN, named). Comparison: f-divergences need overlapping support, IPMs don't.
  8. *What WGAN did and didn't fix*: more stable training and a meaningful loss, but not a full solution to mode coverage. Its legacy: SN and GP are used widely, including in other discriminators.
- **Question ideas**
  - (calc) Parallel lines at distance θ = 0.5: JS = log 2, W1 = 0.5.
  - (calc) 1-D: W1 between point masses at 0 and 3 is 3. Between U(0,1) and U(2,3) it is 2.
  - Where is the WGAN-GP penalty evaluated, and what's its target norm? (interpolates; 1) Compare R1. (real data; 0)
  - (calc) Fenchel conjugate of $f(u)=u\log u$ at t=1: $f^*(1)=e^0=1$.
  - (calc) Spectral norm of diag(3, 1) is 3, so the normalized matrix is diag(1, 1/3).
  - Which is false: "weight clipping enforces exactly 1-Lipschitz critics without side effects".
  - Derivation: write the Fenchel lower bound for an f-divergence.
- **Figures**
  - `gen.wgan/parallel-lines`: JS, KL and W1 vs θ for the parallel-lines example. **[fig-Q]**: "which curve gives useful gradients?"
  - `gen.wgan/transport-plan`: 1-D histograms with arrows showing mass moved, and W1 = area between the CDFs.
  - `gen.wgan/critic-vs-discriminator`: an optimal critic (linear-ish, unbounded) vs a saturated discriminator on separated 1-D distributions.
- **Pitfalls & source disagreements**
  - "Critic" (WGAN) vs "discriminator" (GAN): the critic outputs an unbounded score, not a probability.
  - The WGAN-GP λ (=10) is unrelated to other λs in this area.

---

#### 80 · `gen.flows`: Normalizing Flows
- **Level** intermediate · **W2** · **prereqs** [gen.overview, fund.probability-basics].
- **Sources**
  - Papamakarios et al. 2021 review (`papers/papamakarios2021_flows_review.txt`): §2 definition & change of variables, expressivity; §3 constructions: autoregressive (affine, MAF/IAF), coupling, linear (LU, 1×1), residual (contractive, Sylvester); §4 continuous flows; §5 tradeoffs.
  - Dinh, Sohl-Dickstein & Bengio 2017 RealNVP (`papers/dinh2016_realnvp.txt`): affine coupling, triangular Jacobian, masking schemes, multi-scale architecture, BN in flows.
  - Kingma & Dhariwal 2018 Glow (`papers/kingma2018_glow.txt`): actnorm, invertible 1×1 convolution and its log-det, the LU parametrization, the full step.
  - Papamakarios, Pavlakou & Murray 2017 MAF (`papers/papamakarios2017_maf.txt`); Kingma et al. 2016 IAF (`papers/kingma2016_iaf.txt`).
  - Murphy PML2 ch.23 (§23.1 preliminaries & training, §23.2 affine, elementwise, coupling, autoregressive, residual and continuous-time flows, §23.3 applications; pdf p.863–881). Weng flow blog (`web/weng2018_flows.txt`), secondary.
- **Subtopic map**
  1. *Idea*: transform a simple base density by an invertible, differentiable map, and get exact likelihoods and exact sampling.
  2. *Change of variables* (refresher from fund.probability-basics): $x=g(z)$, $f=g^{-1}$, $\log p_x(x)=\log p_z(f(x))+\log|\det J_f(x)|$. Compositions add log-dets. 1-D worked example.
  3. *The cost problem*: a general determinant is O(D³). Design layers with triangular Jacobians, whose determinant is the product of the diagonal.
  4. *Affine coupling (RealNVP)*:
     - Split $x=(x_a,x_b)$: $y_a=x_a$, $y_b=x_b\odot\exp(s(x_a))+t(x_a)$.
     - Inverse in closed form. log-det = Σ s(x_a).
     - s and t are arbitrary networks that need not be invertible themselves.
     - Alternate the partitions or permute (checkerboard/channel masks), since a single coupling leaves half the dims unchanged.
  5. *Autoregressive flows*:
     - MAF: $x_i=z_i\exp(\alpha_i)+\mu_i$ with $(\mu_i,\alpha_i)$ functions of $x_{<i}$. Density evaluation is one parallel pass (MADE masks). Sampling needs D sequential passes.
     - IAF: parameters depend on $z_{<i}$, so sampling is parallel and the density of external x is sequential. Used as a flexible VAE posterior.
     - Coupling = a 2-block special case.
  6. *Glow components*:
     - Actnorm (per-channel affine with data-dependent init).
     - Invertible 1×1 conv: log-det = H·W·log|det W|, made cheap via the LU parametrization.
     - Multi-scale "factor out half the dims".
  7. *Training*: maximize the exact log-likelihood. Dequantization for discrete pixels. Report bits/dim (pointer to gen.evaluation).
  8. *Limitations*:
     - Dimension can't change (no bottleneck), so many layers are needed and memory is high.
     - Topology: a homeomorphism can't map a unimodal Gaussian onto disconnected supports without thin low-density bridges.
     - Sample quality has lagged diffusion.
  9. *Residual flows* (invertibility via Lip < 1, power-series log-det, named). *Continuous normalizing flows*: $dz/dt=v(z,t)$, $\frac{d\log p}{dt}=-\mathrm{tr}(\partial v/\partial z)$. Hutchinson trace estimation (FFJORD). Simulation-heavy training, which is the motivation for flow matching (pointer to gen.flow-matching).
- **Question ideas**
  - (calc) z ~ N(0,1), x = 2z + 1: $p_x(1)=\phi(0)/2\approx0.199$.
  - (calc) Affine coupling with s-outputs (0.5, −0.2, 0.1) on the transformed half → log-det = 0.4.
  - MAF vs IAF: which is fast for density evaluation, and which for sampling?
  - (calc) Glow 1×1 conv on a 32×32×c feature map: log-det = 1024·log|det W|.
  - Why can't a flow use a low-dimensional bottleneck?
  - Which is false: "the networks s and t inside an affine coupling layer must be invertible".
- **Figures**
  - `gen.flows/change-of-variables`: a 1-D Gaussian pushed through a monotone function, showing the density stretching and compressing.
  - `gen.flows/coupling-diagram`: data flow of an affine coupling layer (forward and inverse).
  - `gen.flows/maf-vs-iaf`: dependency diagrams with the arrows for the density and sampling directions. **[fig-Q]**
  - `gen.flows/toy-flow`: a 2-D two-moons density learned by a stack of coupling layers, with intermediate transformed grids.
- **Pitfalls & source disagreements**
  - Direction conventions: some papers define f: data→noise (the "normalizing" direction), others g: noise→data. The log-det sign follows from that.
  - MAF/IAF naming confusion.

---

#### 90 · `gen.ebm-score-matching`: Energy-Based Models, Score Matching & Tweedie
- **Level** advanced · **W2** · **prereqs** [gen.overview, fund.autoencoders].
- **Sources**
  - Song & Kingma 2021 "How to Train Your EBM" (`papers/song2021_train_ebm.txt`): MLE with MCMC (Langevin, contrastive divergence), score matching (explicit/implicit, sliced, denoising), NCE, connections.
  - Hyvärinen 2005 (`papers/hyvarinen2005_score_matching.txt`): the score-matching objective via integration by parts, consistency.
  - Vincent 2011 (`papers/vincent2011_dsm.txt`): denoising score matching equivalence and the DAE connection.
  - Efron 2011 (`papers/efron2011_tweedie.txt`): Tweedie's formula.
  - Murphy PML2 ch.24: §24.1 EBMs & computational difficulties, §24.2 MLE with gradient-based MCMC & contrastive divergence, §24.3 score matching (basic, DSM, sliced, connection to CD), §24.4 NCE (pdf p.883–897). `pml2.txt`
  - Song & Ermon 2019 §2–3 (`papers/song2019_ncsn.txt`): Langevin dynamics and the pitfalls of score-based modelling (manifold hypothesis, low-density regions). Y. Song blog (`web/song2021_score_blog.txt`), secondary.
- **Subtopic map**
  1. *EBMs*: $p_\theta(x)=e^{-E_\theta(x)}/Z(\theta)$. A flexible architecture, but Z is intractable, so likelihoods and naive sampling are hard. Products of experts (named).
  2. *MLE for EBMs*: $\nabla\log p=-\nabla E(x)+E_{p_\theta}[\nabla E]$ (derive via $\nabla\log Z$). The second term needs model samples: MCMC/Langevin, contrastive divergence (short chains from data), persistent chains. The bias and cost.
  3. *Score function*: $s(x)=\nabla_x\log p(x)=-\nabla_xE(x)$ doesn't depend on Z. Score of a Gaussian (calc). The score field as a vector field pointing to high density.
  4. *Langevin dynamics*: $x_{k+1}=x_k+\frac\eta2\nabla\log p(x_k)+\sqrt\eta\,\xi_k$ samples p as η → 0 and K → ∞ (statement with intuition: gradient ascent plus noise to avoid collapsing to the mode). A worked 1-D example.
  5. *Explicit score matching*: $\frac12E_p\|s_\theta(x)-\nabla\log p(x)\|^2$ needs the unknown true score. **Hyvärinen's trick**: integrate by parts to get $E_p[\mathrm{tr}(\nabla_xs_\theta(x))+\frac12\|s_\theta(x)\|^2]$ + const (derive in 1-D with boundary conditions). The trace of the Jacobian is costly in high dimensions, which motivates **sliced** score matching (random projections).
  6. *Denoising score matching*:
     - Perturb $\tilde x=x+\sigma\epsilon$.
     - Objective $E\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x|x)\|^2$ with target $-(\tilde x-x)/\sigma^2=-\epsilon/\sigma$.
     - It equals score matching on the *noised marginal* $q_\sigma(\tilde x)$ up to a constant (proof sketch: expand the square and use $\nabla\log q_\sigma(\tilde x)=E[\nabla\log q_\sigma(\tilde x|x)\mid\tilde x]$).
  7. *Tweedie's formula*: $E[x|\tilde x]=\tilde x+\sigma^2\nabla\log q_\sigma(\tilde x)$ (derive for Gaussian noise). Optimal denoiser ↔ score. That's why denoising autoencoders estimate scores (pointer to fund.autoencoders), and it underpins diffusion.
  8. *Pitfalls of single-scale score models* (Song & Ermon): the score is undefined or inaccurate off the data manifold and in low-density regions, and Langevin mixes poorly between modes (wrong mode weights). This motivates noise at multiple scales (pointer to gen.score-sde).
  9. *NCE* (one paragraph): learn E by classifying data vs noise samples. Its relation to GANs and to score matching (named).
- **Card budget** (about 10): EBMs + MLE gradient · score & Langevin · Hyvärinen's trick (+ sliced, one paragraph) · DSM and its equivalence proof · Tweedie (derivation + the Gaussian-data example) · single-scale pitfalls · NCE paragraph folded into the connections card · probes · key results. This is the lesson diffusion leans on for "denoiser = score", so the DSM proof and Tweedie get full derivations; trim NCE first if it overflows.
- **Question ideas**
  - (calc) Score of N(μ, σ²) at x: $-(x-\mu)/\sigma^2$. At x=3 for N(1, 4): −0.5.
  - (calc) Tweedie: prior x ~ N(0,1), $\tilde x=x+n$ with n ~ N(0,1). Then $E[x|\tilde x]=\tilde x/2$. Check via the marginal score $-\tilde x/2$.
  - (calc) General Gaussian data $x\sim N(0,s^2)$, $\tilde x=x+\sigma n$: the optimal denoiser is $D(\tilde x)=\frac{s^2}{s^2+\sigma^2}\tilde x$ (shrinkage toward the mean). This reappears as EDM's $c_{skip}$ (gen.diffusion-solvers).
  - Why doesn't the score depend on Z? (the gradient of a constant is 0)
  - DSM target for Gaussian corruption? ($-\epsilon/\sigma$)
  - Which is false: "Langevin dynamics with an accurate score samples correctly regardless of how far apart the modes are".
  - Derivation step: in Hyvärinen's integration by parts, which term vanishes and why? (the boundary term, since p → 0)
- **Figures**
  - `gen.ebm-score-matching/score-field`: the score vector field of a 2-component 2-D mixture over density contours.
  - `gen.ebm-score-matching/langevin-chains`: Langevin trajectories converging to the modes. **[fig-Q]**: "what happens with too large a step?"
  - `gen.ebm-score-matching/low-density-error`: true vs estimated score magnitude error vs density (large error in low-density regions).
  - `gen.ebm-score-matching/tweedie`: noisy observations and the posterior-mean denoised estimates moving toward the data.
- **Pitfalls & source disagreements**
  - "Score" here is $\nabla_x\log p$, not the statistics score $\nabla_\theta\log p$ (fund.fisher-information and fund.gradient-estimators). Flag the clash.
  - Langevin step conventions: some write ε/2 on the gradient and √ε on the noise, others ε and √(2ε).

---

#### 100 · `gen.ddpm-forward-reverse`: Diffusion I: Forward Noising & the Reverse Posterior
- **Level** core · **W1** · **prereqs** [gen.latent-variables-elbo, fund.gaussian].
- **Sources**
  - Ho, Jain & Abbeel 2020 (`papers/ho2020_ddpm.txt`): §2 background, eq. (2) forward process, eq. (4) closed-form $q(x_t|x_0)$, eqs. (6)–(7) the posterior $q(x_{t-1}|x_t,x_0)$, Alg. 2 sampling, §3.2 reverse-process variances (β_t vs β̃_t).
  - Luo 2022, "Variational Diffusion Models" (p.6–14; unnumbered section) (`papers/luo2022_diffusion_unified.txt`): the forward process as a fixed encoder, the step-by-step derivations of $q(x_t|x_0)$ and of the posterior (completing the square).
  - Sohl-Dickstein et al. 2015 (`papers/sohldickstein2015_diffusion.txt`): the original nonequilibrium-thermodynamics framing.
  - Murphy PML2 §25.2.1 encoder (forward diffusion), §25.2.2 decoder (reverse diffusion) (pdf p.902–904). CS229 notes ch.14 "The diffusion process", "Parameterizing the reverse process" (pdf p.181–185). `pml2.txt`, `cs229.txt`
  - Weng diffusion blog (`web/weng2021_diffusion.txt`), secondary.
- **Subtopic map**
  1. *Intuition*: destroying structure by adding noise is easy. Learn to reverse it step by step, where each reverse step is a small denoising problem. Contrast with a VAE (here the encoder is fixed and many latent layers share one network).
  1b. *Gaussian refresher, inline* (pointer to fund.gaussian, but self-contained): (i) if $a\sim N(0,s_1^2I)$ and $b\sim N(0,s_2^2I)$ are independent, $a+b\sim N(0,(s_1^2+s_2^2)I)$; (ii) a product of Gaussian densities in the same variable is Gaussian with **precisions adding** and a precision-weighted mean; (iii) completing the square: $-\frac12(Ax^2-2Bx)$ ⇒ variance $1/A$, mean $B/A$. Items 3 and 6 use exactly these.
  2. *Forward process*: $q(x_t|x_{t-1})=N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)$ with a small increasing β schedule and T ≈ 1000. Why scale by $\sqrt{1-\beta_t}$: it keeps the variance at 1 if Var(x_{t−1}) = 1 (variance-preserving; show it).
  3. *Closed-form marginal*: define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s\le t}\alpha_s$. Derive $q(x_t|x_0)=N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$ by induction, merging two Gaussian noise terms (the variances add: $\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)=1-\bar\alpha_t$). Consequence: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ in one shot, which makes training cheap. As $\bar\alpha_T\to0$, $x_T\approx N(0,I)$.
  4. *Signal-to-noise ratio*: SNR(t) = $\bar\alpha_t/(1-\bar\alpha_t)$ decreases monotonically. A worked numerical table for the linear schedule (β from 1e-4 to 0.02, T=1000; calc): t=1: $\bar\alpha\approx0.9999$; t=250: $\bar\alpha\approx0.524$, SNR ≈ 1.10; t=500: $\bar\alpha\approx0.079$, SNR ≈ 0.085; t=1000: $\bar\alpha\approx4.0\times10^{-5}$. SNR = 1 is crossed near t ≈ 259, so roughly the last half of the chain is mostly noise (motivates the cosine schedule in gen.diffusion-parameterizations).
  5. *The reverse process we want*: $q(x_{t-1}|x_t)$ is intractable (it needs the data distribution), but for small β it's approximately Gaussian (Feller/Sohl-Dickstein). Model it as $p_\theta(x_{t-1}|x_t)=N(\mu_\theta(x_t,t),\Sigma_\theta)$.
  6. *Tractable posterior given x0*: derive $q(x_{t-1}|x_t,x_0)=N(\tilde\mu_t,\tilde\beta_tI)$ with $\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$ and $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$, via Bayes and completing the square. **Derive it fully, line by line** (this is the most common diffusion whiteboard question): (a) $q(x_{t-1}|x_t,x_0)\propto q(x_t|x_{t-1})\,q(x_{t-1}|x_0)$, using the Markov property $q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})$; (b) both factors are Gaussian in $x_{t-1}$; (c) precisions add: $1/\tilde\beta_t=\alpha_t/\beta_t+1/(1-\bar\alpha_{t-1})$, which simplifies to the $\tilde\beta_t$ above; (d) mean $=\tilde\beta_t\big(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\big)$, which simplifies to $\tilde\mu_t$. Interpret $\tilde\mu$ as a weighted blend of the clean image and the current noisy one. This is the training target in the next lesson.
  7. *Ancestral sampling* (Alg. 2): start from $x_T\sim N(0,I)$ and iterate $x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz$. The choice $\sigma_t^2\in\{\beta_t,\tilde\beta_t\}$ (the two extremes that are optimal for different data assumptions, Ho §3.2). T network evaluations, so sampling is slow (pointer to gen.ddim).
  8. *Network* (one sentence): a time-conditioned network that outputs a tensor the shape of $x_t$ (U-Net in Ho et al.). Architecture details live in gen.diffusion-transformers.
- **Card budget** (about 9): intuition · Gaussian refresher · forward process & variance preservation · closed-form marginal (induction) · SNR table · reverse process & why $q(x_{t-1}|x_t)$ is intractable · posterior derivation (may take 1.5 cards) · ancestral sampling & variance choice · probes + key results (the toy walkthrough rides on the forward-2d figure).
  9. *A 2-D toy walkthrough*: Swiss-roll data diffused forward over t, and the learned reverse trajectories.
- **Question ideas**
  - (calc) $\bar\alpha_t=0.25$, $x_0=2$, ε=1 → $x_t=0.5\cdot2+\sqrt{0.75}\approx1.866$.
  - (calc) With constant β=0.02, $\bar\alpha_{100}=0.98^{100}\approx0.133$, so SNR ≈ 0.153.
  - Why the factor $\sqrt{1-\beta_t}$ in the forward step? (variance preservation)
  - Which posterior variance is $\tilde\beta_t$? (q(x_{t−1}|x_t, x_0)) Is it larger or smaller than β_t? (smaller)
  - Derivation step: in completing the square for $q(x_{t-1}|x_t,x_0)$, what is the coefficient of $x_{t-1}^2$ (the posterior precision)? ($\alpha_t/\beta_t+1/(1-\bar\alpha_{t-1})$) Why does $q(x_t|x_{t-1})$ not depend on $x_0$? (Markov)
  - Which is false: "$q(x_{t-1}|x_t)$ is available in closed form for any data distribution".
  - **[fig-Q]** A row of images at increasing t: which corresponds to SNR ≈ 1?
- **Figures**
  - `gen.ddpm-forward-reverse/forward-2d`: a 2-D Swiss roll at t = 0, 100, 300 and 1000 under the forward process. Notice its convergence to N(0, I).
  - `gen.ddpm-forward-reverse/schedule-curves`: $\sqrt{\bar\alpha_t}$, $\sqrt{1-\bar\alpha_t}$ and log-SNR vs t for the linear schedule.
  - `gen.ddpm-forward-reverse/markov-chain`: the graphical model $x_0\to x_1\to\cdots\to x_T$ with q arrows forward and $p_\theta$ arrows back.
  - `gen.ddpm-forward-reverse/posterior-blend`: the coefficients of $x_0$ and $x_t$ in $\tilde\mu_t$ as functions of t. **[fig-Q]**
- **Pitfalls & source disagreements**
  - **α notation clash**: Ho's $\alpha_t=1-\beta_t$ vs DDIM's $\alpha_t$ = Ho's $\bar\alpha_t$ vs VDM's $\alpha_t$ = Ho's $\sqrt{\bar\alpha_t}$ (see table B.4).
  - Time indexing (t = 0 data vs t = T noise; continuous t ∈ [0,1] in later papers).

---

#### 110 · `gen.ddpm-objective`: Diffusion II: The ELBO & the ε-Prediction Loss
- **Level** core · **W1** · **prereqs** [gen.ddpm-forward-reverse].
- **Card budget** (about 9): the diffusion model as an LVM · ELBO derivation (2 cards: the Bayes-on-$x_0$ trick, then telescoping into $L_T,L_{t-1},L_0$) · Gaussian KL → mean MSE · ε-parameterization and its weight · the weight in SNR form · $L_{\text{simple}}$ and Alg. 1 · reverse variances and likelihood evaluation · probes + key results. Parameterizations, schedules and weightings moved to gen.diffusion-parameterizations (split in review; the original lesson had about 12 cards of material).
- **Sources**
  - Ho et al. 2020 (`papers/ho2020_ddpm.txt`): §2 eq. (5) the ELBO decomposition $L_T+\sum_{t>1}L_{t-1}+L_0$ and App. A (its derivation), §3.1 $L_T$ is constant, §3.2 eq. (8) the mean-matching form and the ε-parameterization (eqs. 11–12, with the weight $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$), §3.3 the discrete decoder $L_0$ (eq. 13), §3.4 $L_{\text{simple}}$ (eq. 14) and why it down-weights small t, §4 Table 2 (ε vs μ̃ prediction, fixed vs learned variance, true bound vs $L_{\text{simple}}$).
  - Luo 2022, "Variational Diffusion Models" (p.6–14; unnumbered) (`papers/luo2022_diffusion_unified.txt`): the full ELBO derivation (prior matching, denoising-matching and reconstruction terms), including the step that conditions the forward transitions on $x_0$.
  - CS229 notes ch.14 "Training diffusion models by maximizing the ELBO" (pdf p.185–189). `cs229.txt`
  - Nichol & Dhariwal 2021 (`papers/nichol2021_improved_ddpm.txt`): §3.1 learned variances (interpolating log β and log β̃) and $L_{\text{hybrid}}=L_{\text{simple}}+\lambda L_{\text{vlb}}$ with λ = 0.001, §3.3 importance-sampled t to reduce the variance of $L_{\text{vlb}}$.
  - Kingma et al. 2021 VDM §4 (`papers/kingma2021_vdm.txt`): the discrete-time loss written as $(\mathrm{SNR}(s)-\mathrm{SNR}(t))\|x-\hat x_\theta\|^2$, which is the SNR form of the weight below. Murphy PML2 §25.2.3 model fitting (pdf p.904–906). `pml2.txt`
- **Subtopic map**
  1. *ELBO for the diffusion LVM*: treat $x_{1:T}$ as latents with the fixed encoder q (recall the ELBO from gen.latent-variables-elbo). Derive $-\mathrm{ELBO}=\underbrace{\mathrm{KL}(q(x_T|x_0)\|p(x_T))}_{L_T}+\sum_{t>1}\underbrace{E_q\mathrm{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}}\underbrace{-E_q\log p_\theta(x_0|x_1)}_{L_0}$. Show the key trick: rewriting $q(x_t|x_{t-1})=q(x_t|x_{t-1},x_0)=\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}$ (Bayes, using the Markov property), after which the $q(x_t|x_0)$ ratios telescope (Luo). Say why this beats the naive form: each term is a KL between Gaussians, computed in closed form instead of by high-variance Monte Carlo (Ho calls this Rao–Blackwellized). Interpret each term. $L_T$ has no parameters (calc: ≈ 2.0e-5 nats per dimension for $x_0=1$ under the linear schedule).
  2. *Each $L_{t-1}$ is a KL between Gaussians*. With fixed $\Sigma=\sigma_t^2I$, the general Gaussian KL (from gen.vae) reduces to $\frac1{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2$ + const (Ho eq. 8).
  3. *ε-parameterization*:
     - Substitute $x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde\mu_t$ to get $\tilde\mu_t=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon)$ (derive).
     - Parameterize $\mu_\theta$ the same way with $\epsilon_\theta(x_t,t)$. The sampler of the previous lesson becomes Alg. 2.
     - The KL becomes a weighted $\|\epsilon-\epsilon_\theta\|^2$ (derive the weight $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$).
  4. *The weight in SNR form*: with $\sigma_t^2=\tilde\beta_t$, the same term written on $x_0$-error is $\frac12(\mathrm{SNR}(t-1)-\mathrm{SNR}(t))\|x_0-\hat x_0\|^2$, and on ε-error is $\frac12\big(\frac{\mathrm{SNR}(t-1)}{\mathrm{SNR}(t)}-1\big)\|\epsilon-\hat\epsilon\|^2$ (derive in 3 lines; calc-checked; VDM §4 writes the same form). So the true ELBO weights each noise level by how much SNR drops across that step. This is the bridge to the weighting view in gen.diffusion-parameterizations.
  5. *$L_{\text{simple}}$*: drop the weights and sample t uniformly. $E_{t,x_0,\epsilon}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2$. It is a reweighted ELBO that down-weights small t (easy denoising) relative to the true bound, giving better samples and worse likelihood (Ho §3.4, Table 2). The training algorithm (Alg. 1) is about 5 lines.
  6. *Reverse variance choices*: fixed β_t vs β̃_t (the upper and lower extremes, optimal for $x_0\sim N(0,I)$ and for a single point respectively; Ho §3.2) vs learned interpolation (Nichol: $\Sigma=\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$, trained with $L_{\text{vlb}}$ in $L_{\text{hybrid}}$). This improves likelihood and allows fewer sampling steps.
  7. *Likelihood evaluation*: the discretized decoder $L_0$ for 8-bit data, bits/dim reporting, importance sampling of t to reduce the variance of $L_{\text{vlb}}$ (Nichol §3.3).
  8. *Why it works so well* (one card of connections): hierarchical VAE with a fixed encoder (pointer to gen.vae-variants); denoiser at each noise level ↔ score (pointer to gen.diffusion-parameterizations and gen.ebm-score-matching); stable regression training vs adversarial training.
- **Question ideas**
  - Which loss weighting does $L_{\text{simple}}$ correspond to relative to the true ELBO? (uniform ε-MSE, which drops the t-dependent weights)
  - Derivation step: in the ELBO derivation, why can we replace $q(x_t|x_{t-1})$ by $q(x_t|x_{t-1},x_0)$? (Markov property) What does this buy? (tractable Gaussian posteriors and closed-form KLs)
  - (calc) Show the ε-weight equals $\frac12(\mathrm{SNR}(t-1)/\mathrm{SNR}(t)-1)$ when $\sigma_t^2=\tilde\beta_t$, and evaluate it at t=500 of the linear schedule (≈ 0.0055).
  - Predict: training with the true ELBO weights vs $L_{\text{simple}}$ (better bits/dim vs better FID).
  - Which is false: "$L_T$ must be minimized with respect to θ" (it has no parameters when β is fixed).
  - Which is false: "the DDPM training loss $L_{\text{simple}}$ is exactly the negative ELBO".
- **Figures**
  - `gen.ddpm-objective/elbo-terms`: a bar decomposition of the ELBO into $L_T$, the $L_{t-1}$ (by t) and $L_0$ for a toy model (computed, e.g. 1-D Gaussian data where the optimal $\epsilon_\theta$ is known in closed form). Notice that most of the likelihood comes from small t.
  - `gen.ddpm-objective/elbo-vs-simple-weights`: the true ELBO's per-t weight (SNR form) vs the flat $L_{\text{simple}}$ weight across t. **[fig-Q]**: "which curve down-weights small t?"
  - `gen.ddpm-objective/training-loop`: the 5-line Alg. 1 annotated next to a sampled $x_t$.
- **Pitfalls & source disagreements**
  - **"The DDPM loss is the ELBO"** is false for $L_{\text{simple}}$. The notation clash table (B.4).
  - Nichol uses $L_{\text{vlb}}$ for the ELBO terms. VDM uses continuous t and α/σ notation. Ho's term indices ($L_{t-1}$) are shifted relative to Luo's ($L_t$).
  - The SNR form of the weight holds for $\sigma_t^2=\tilde\beta_t$; with $\sigma_t^2=\beta_t$ the constant differs.

---

#### 120 · `gen.diffusion-parameterizations`: Diffusion III: Parameterizations, SNR & Noise Schedules
- **Level** core · **W1** · **prereqs** [gen.ddpm-objective]. Split from the original ddpm-objective in review: interviews at image/video labs probe "ε vs x0 vs v vs score" and "what does the schedule actually change" as their own chain.
- **Card budget** (about 8–9): α/σ/SNR notation and conversions · the four targets and their conversions · score ↔ ε (Tweedie in two lines) · loss equivalences as SNR weightings · numerical conditioning at the two ends (why v) · schedules (linear, cosine) in log-SNR terms · what the schedule does and doesn't change (VDM invariance) · schedule pitfalls (terminal SNR, resolution) · probes + key results.
- **Sources**
  - Salimans & Ho 2022 §4 & App. D (`papers/salimans2022_progressive_distillation.txt`): eq. (9) ε-MSE = SNR-weighted x-MSE, why ε-prediction breaks as SNR → 0, the x / (x, ε) / v options, v ≡ αε − σx and $\hat x=\alpha z_t-\sigma\hat v$.
  - Luo 2022, "Three Equivalent Interpretations" (p.15–17; incl. Tweedie's formula) (`papers/luo2022_diffusion_unified.txt`).
  - Kingma et al. 2021 VDM (`papers/kingma2021_vdm.txt`): §3.1–3.2 forward process in α/σ notation and SNR(t) = α²/σ², the learned schedule; §4 discrete-time loss; §5.1 the continuous-time ELBO is invariant to the schedule except at its endpoints; §5.2 weighted losses.
  - Kingma & Gao 2023 (`papers/kingma2023_diffusion_objectives.txt`): Table 1 (common objectives as weightings w(λ) of one loss, incl. ε, v, EDM and FM-OT, App. D), §4 monotonic weightings equal the ELBO under Gaussian-noise data augmentation.
  - Nichol & Dhariwal 2021 §3.2 the cosine schedule (s = 0.008, β clipped at 0.999) (`papers/nichol2021_improved_ddpm.txt`). Ho et al. 2020 §4 the linear schedule.
  - Esser et al. 2024 SD3 §2 (`papers/esser2024_sd3_rectified_flow.txt`): the unified weighted-ε form of all objectives; §5.3.2 resolution-dependent timestep shift. Karras et al. 2022 Table 1 (`papers/karras2022_edm.txt`) as a notation bridge. Murphy PML2 §25.2.3–25.2.4 (pdf p.904–907). Gao et al. 2024 (`web/gao2024_diffusion_meets_flow_matching.txt`), secondary.
- **Subtopic map**
  1. *Notation*: write the VP forward process as $x_t=\alpha_tx_0+\sigma_t\epsilon$ with $\alpha_t=\sqrt{\bar\alpha_t}$, $\sigma_t=\sqrt{1-\bar\alpha_t}$, $\alpha_t^2+\sigma_t^2=1$; SNR = $\alpha_t^2/\sigma_t^2$ and log-SNR λ. Convert to Ho's notation explicitly (table B.4). From here on the noise level is λ, not t.
  2. *Four targets, one piece of information*: the network can predict $x_0$, ε, the score, or v. Conversions: $\hat x_0=(x_t-\sigma_t\hat\epsilon)/\alpha_t$; $v=\alpha_t\epsilon-\sigma_tx_0$ with $\hat x_0=\alpha_tx_t-\sigma_t\hat v$ and $\hat\epsilon=\sigma_tx_t+\alpha_t\hat v$ (derive using $\alpha^2+\sigma^2=1$).
  3. *Score ↔ ε*: $\nabla\log q_t(x_t)=E[\nabla\log q(x_t|x_0)\mid x_t]=-E[\epsilon|x_t]/\sigma_t$, so the optimal ε-network is $-\sigma_t\times$ score. Equivalently Tweedie: $E[x_0|x_t]=(x_t+\sigma_t^2\nabla\log q_t(x_t))/\alpha_t$. Two-line derivation here; the full DSM proof is in gen.ebm-score-matching (pointer).
  4. *Losses as weightings*: $\|\epsilon-\hat\epsilon\|^2=\mathrm{SNR}\,\|x_0-\hat x_0\|^2$; $\|v-\hat v\|^2=(1+\mathrm{SNR})\|x_0-\hat x_0\|^2$; score-MSE $=\|\epsilon-\hat\epsilon\|^2/\sigma_t^2$ (derive each). With the sampling density of t, every objective is $\int p(\lambda)\,w(\lambda)\,\|x_0-\hat x_0\|^2$; the ELBO's own weight is the SNR drop per step from gen.ddpm-objective (continuous: $-\frac12\,d\mathrm{SNR}/dt$). Kingma & Gao: monotone weightings are ELBOs of noise-augmented data. Min-SNR / P2 weightings (named).
  5. *Numerical conditioning*: ε-prediction amplifies errors in $\hat x_0$ by $\sigma_t/\alpha_t$, which blows up as SNR → 0 (and its x-space weight goes to 0 there), so it fails at the highest noise levels and in few-step distillation (Salimans & Ho). x0-prediction at low noise forces the network to re-emit the whole input, so its errors land directly on the output instead of on a small residual (which is why EDM adds a skip path). v-prediction has a unit-variance target for unit-variance data ($\mathrm{Var}(v)=\alpha^2+\sigma^2=1$) and behaves at both ends. EDM's preconditioning generalizes this (pointer to gen.diffusion-solvers).
  6. *Noise schedules*:
     - Linear β ∈ [1e-4, 0.02] (Ho).
     - Cosine $\bar\alpha_t=\frac{f(t)}{f(0)}$, $f(t)=\cos^2(\frac{t/T+s}{1+s}\frac\pi2)$, s = 0.008, β clipped at 0.999 (Nichol §3.2). (calc) At t=500: cosine $\bar\alpha\approx0.494$ vs linear ≈ 0.079; SNR = 1 is crossed near t ≈ 496 vs ≈ 259.
     - In log-SNR terms a schedule is just a choice of which noise levels are visited (and how densely), so plot schedules against λ, not t.
  7. *What the schedule does and doesn't change*: the continuous-time ELBO depends only on the endpoint SNRs (VDM §5.1, derive by the change of variable t → SNR), so for the true ELBO the schedule changes only the Monte-Carlo variance; for weighted losses with uniform t ($L_{\text{simple}}$, FM) it changes the effective weighting; and it changes the discretization error at sampling time.
  8. *Schedule pitfalls*:
     - Non-zero terminal SNR (linear schedule: $\bar\alpha_T\approx4\times10^{-5}$, log-SNR ≈ −10.1) means training never sees pure noise but sampling starts from it. Zero-terminal-SNR fixes need v- or x-prediction, because ε-prediction is degenerate at SNR = 0 (Lin et al. 2024, named; not cached).
     - Resolution dependence: the same per-pixel σ destroys less global information at higher resolution (neighbouring pixels average out the noise), so high-resolution models shift the schedule toward more noise (SD3 §5.3.2; named shift formula).
- **Question ideas**
  - (calc) Convert ε prediction to score at $\bar\alpha_t=0.36$: $s=-\epsilon/0.8$.
  - (calc) v-prediction: $\bar\alpha_t=0.64$, $x_0=1$, ε=0.5 → v = 0.8·0.5 − 0.6·1 = −0.2.
  - (calc) At SNR = 1, by what factor does the v-loss weight the $x_0$-error relative to the ε-loss? (v: 1+SNR = 2; ε: SNR = 1)
  - Predict: which parameterization's implied $\hat x_0$ blows up as SNR → 0, and why does that matter for 1-step distillation? (ε)
  - Why the cosine schedule? (the linear schedule reaches near-pure noise too early, so many steps contribute little, especially at low resolution)
  - Which is false: "changing the noise schedule changes the optimum of the continuous-time ELBO".
  - Derivation step: show $\hat\epsilon=\sigma_tx_t+\alpha_t\hat v$. Which identity is needed? ($\alpha^2+\sigma^2=1$)
- **Figures**
  - `gen.diffusion-parameterizations/schedules`: $\bar\alpha_t$ and log-SNR for the linear vs cosine schedules. **[fig-Q]**: "which is cosine?"
  - `gen.diffusion-parameterizations/weightings`: implied weighting $w(\lambda)$ on the $x_0$-error vs log-SNR for ε-, x0-, v-prediction, the ELBO and FM-OT (Kingma & Gao Table 1), computed from the formulas.
  - `gen.diffusion-parameterizations/error-amplification`: the factor $\sigma/\alpha$ by which an ε-error maps to an $x_0$-error, vs the bounded factor for v-prediction, across log-SNR.
- **Pitfalls & source disagreements**
  - λ conventions: VDM's γ = −log SNR; Salimans & Ho's λ = log SNR; DPM-Solver's λ = ½ log SNR (B.4).
  - "α" means Ho's $\alpha_t=1-\beta_t$, DDIM's $\bar\alpha_t$, or VDM's $\sqrt{\bar\alpha_t}$ (B.4).
  - Salimans & Ho's v and flow matching's velocity agree only up to sign and scale (gen.rectified-flow).

---

#### 130 · `gen.score-sde`: Score-Based Diffusion & the SDE/ODE View
- **Level** advanced · **W2** · **prereqs** [gen.ebm-score-matching, gen.diffusion-parameterizations].
- **Card budget** (about 9–10): multi-scale NCSN · DDPM equivalence · SDE refresher · forward SDEs (VE/VP; VP derived from the DDPM step) · reverse SDE · PF-ODE derivation via Fokker–Planck · PC sampling + conditional generation · SDE vs ODE · probes + key results.
- **Sources**
  - Song & Ermon 2019 NCSN (`papers/song2019_ncsn.txt`): multiple noise levels, a noise-conditional network, the λ(σ) = σ² weighting, annealed Langevin, the geometric σ schedule.
  - Song et al. 2021 SDE (`papers/song2021_sde.txt`):
    - §2.1–2.2 SMLD and DDPM recap (note: here σ_1 is the *smallest* noise, the opposite of NCSN's indexing).
    - §3.1–3.2 forward SDE and the reverse-time SDE (Anderson). §3.3 time-conditional DSM.
    - §3.4 VE, VP and sub-VP SDEs as limits of SMLD and DDPM (App. B: VP with $\beta(t)=\bar\beta_{min}+t(\bar\beta_{max}-\bar\beta_{min})$, 0.1 → 20).
    - §4.1 reverse-diffusion samplers, §4.2 predictor–corrector samplers.
    - §4.3 the probability-flow ODE and exact likelihood.
    - §5 controllable generation.
    - App. B–D proofs and details.
  - Murphy PML2 §25.3 SGMs incl. §25.3.3 equivalence to DDPM (pdf p.908–911), §25.4 continuous-time models: forward/reverse SDEs and ODEs, comparison (pdf p.911–915). `pml2.txt`
  - CS229 notes ch.14 "Continuous-time view of reverse diffusion" (pdf p.189–191). `cs229.txt`
  - Luo 2022, "Score-based Generative Models" (p.17–20; unnumbered) (`papers/luo2022_diffusion_unified.txt`). Y. Song blog (`web/song2021_score_blog.txt`), secondary.
- **Subtopic map**
  1. *Multi-scale fix* (from gen.ebm-score-matching):
     - Perturb with noise levels $\sigma_1>\dots>\sigma_L$ (geometric, σ_max ~ the largest data distance).
     - Train one noise-conditional score network $s_\theta(x,\sigma)$ with DSM weighted by λ(σ) = σ², which makes the terms comparable (why).
     - Sample by annealed Langevin from large to small σ.
  2. *Equivalence with DDPM*: $\epsilon_\theta(x_t,t)=-\sqrt{1-\bar\alpha_t}\,s_\theta(x_t,t)$. Both learn denoisers at many noise levels. NCSN = variance exploding, DDPM = variance preserving.
  2b. *SDE refresher, inline* (the reader may never have used SDEs): Brownian motion, with increments $dW\sim N(0,dt\,I)$ (so noise scales as $\sqrt{dt}$, not dt); Euler–Maruyama $x_{t+\Delta}=x_t+f\Delta+g\sqrt\Delta\,z$; the Fokker–Planck equation $\partial_tp=-\nabla\cdot(fp)+\frac12g^2\Delta p$ for scalar g (stated, with the intuition "drift transports mass, diffusion spreads it"). No Itô calculus beyond this.
  3. *Continuous-time limit*: the forward SDE $dx=f(x,t)dt+g(t)dW$.
     - VE: f = 0, $g=\sqrt{d\sigma^2/dt}$.
     - VP: $f=-\frac12\beta(t)x$, $g=\sqrt{\beta(t)}$ (derive VP as the limit of the DDPM step).
     - sub-VP (named).
     - Marginals $p_t$ and the perturbation kernels in closed form.
  4. *Reverse-time SDE* (Anderson 1982, statement): $dx=[f-g^2\nabla_x\log p_t(x)]dt+g\,d\bar W$, run backward in time. The only unknown is the score, which we learn with time-conditional DSM. Euler–Maruyama discretization ≈ ancestral sampling.
  5. *Probability-flow ODE*: $dx=[f-\frac12g^2\nabla\log p_t]dt$ has the **same marginals** $p_t$ as the SDE. **Derive it** (three lines): write $\frac12g^2\Delta p=\nabla\cdot(\frac12g^2p\nabla\log p)$, so Fokker–Planck becomes a continuity equation $\partial_tp=-\nabla\cdot[(f-\frac12g^2\nabla\log p)p]$ with no diffusion term, i.e. the ODE with that drift moves mass identically. The reverse SDE follows the same way (add and subtract $\frac12g^2\nabla\log p$).
     - Deterministic: a bijection between noise and data.
     - Exact likelihoods via the instantaneous change of variables (link to gen.flows).
     - Latent encoding and interpolation.
     - Enables fast ODE solvers (pointer to gen.ddim and gen.diffusion-solvers).
  6. *Predictor–corrector sampling*: a reverse-SDE step (predictor) plus a few Langevin steps (corrector) at the same noise level.
  7. *Conditional generation and inverse problems*: $\nabla\log p_t(x|y)=\nabla\log p_t(x)+\nabla\log p_t(y|x)$. Inpainting, colourization and other inverse problems (pointer to gen.guidance).
  8. *SDE vs ODE samplers*: stochasticity corrects accumulated errors (better at many steps), while ODEs are better at few steps and deterministic (Karras's observation; pointer to gen.diffusion-solvers).
- **Question ideas**
  - (calc) VP SDE: derive the drift from $x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}z$ with $\sqrt{1-\beta}\approx1-\beta/2$.
  - Probability-flow ODE vs reverse SDE: same marginals? Which is deterministic?
  - Why the λ(σ) = σ² weighting in NCSN? (it makes $\sigma^2\|s\|^2$ terms O(1) across scales)
  - Which is false: "the reverse SDE requires knowing the data distribution in closed form". (It needs the score, which is learned.)
  - Predict: annealed Langevin with only the smallest σ (poor mixing; wrong mode weights).
  - (calc) ε ↔ score conversion at σ_t = 0.5: s = −2ε.
  - (calc) VE PF-ODE for Gaussian data $N(0,s^2)$: $dx/d\sigma=\sigma x/(s^2+\sigma^2)$, so $x(\sigma)\propto\sqrt{s^2+\sigma^2}$. With s=2, starting at x=3 at σ=10 and integrating to σ=0 gives $x=3\cdot2/\sqrt{104}\approx0.588$ (checked numerically). Each trajectory is a rescaling: the ODE is a deterministic bijection.
  - Derivation step: in the PF-ODE derivation, which term of the Fokker–Planck equation is rewritten as a drift? (the Laplacian term, via $\Delta p=\nabla\cdot(p\nabla\log p)$)
- **Figures**
  - `gen.score-sde/sde-trajectories`: 1-D density evolving forward (heat map over t) with forward-SDE sample paths and reverse-ODE paths overlaid. **[fig-Q]**: "which paths are ODE (smooth) vs SDE?"
  - `gen.score-sde/noise-scales`: perturbed densities for σ = 0.1, 1, 10 on a bimodal 1-D distribution.
  - `gen.score-sde/ve-vs-vp`: the marginal std over t for VE and VP.
- **Pitfalls & source disagreements**
  - Noise-level indexing: NCSN orders σ_1 > … > σ_L, while Song et al. 2021 order σ_1 < … < σ_N (B.4).
  - VE/VP naming. Time runs from 0 (data) to 1 or T (noise). The reverse SDE is written with dt < 0 in some sources and with re-parameterized time in others.
  - Score scaling conventions (networks often output σ·score or ε).

---

#### 140 · `gen.ddim`: Fast Sampling I: DDIM
- **Level** advanced · **W2** · **prereqs** [gen.score-sde]. Split from the original `gen.ddim-solvers` in review: DDIM's derivation alone fills a lesson.
- **Card budget** (about 8): the cost problem · the key observation · constructing $q_\sigma$ and proving the marginals match · the update and its reading · η and step skipping · DDIM as an ODE solver · consequences (determinism, interpolation, inversion) · probes + key results.
- **Sources**
  - Song, Meng & Ermon 2021 DDIM (`papers/song2020_ddim.txt`):
    - §3.1 non-Markovian forward processes $q_\sigma$ with the same marginals, the family indexed by σ; §3.2 the generative process and Thm 1 (the objective $J_\sigma$ equals $L_\gamma$ up to a constant).
    - §4.1 the generative process, eq. (12), the σ that recovers DDPM, and σ = 0 (DDIM).
    - §4.2 accelerated sampling with sub-sequences τ (details in App. C.1).
    - §4.3 the ODE interpretation, eqs. (13)–(14).
    - §5 experiments incl. consistency and interpolation.
    - App. C.2 the notation note (their α = Ho's ᾱ).
  - Gao et al. 2024 (`web/gao2024_diffusion_meets_flow_matching.txt`), secondary: DDIM = the flow-matching Euler sampler, and its invariance to linear rescaling of the schedule.
  - Murphy PML2 §25.5.1 DDIM sampler (pdf p.916). `pml2.txt`
- **Subtopic map**
  1. *The cost*: ~1000 network evaluations (NFE) for DDPM ancestral sampling. NFE is the cost metric. Error comes from discretizing an SDE/ODE.
  2. *DDIM's key observation*: $L_{\text{simple}}$ depends only on the marginals $q(x_t|x_0)$, never on the joint, so any inference process with the same marginals, including non-Markovian ones, shares the trained network.
  3. *Construct the family*: $q_\sigma(x_{t-1}|x_t,x_0)=N\big(\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2I\big)$ (Ho notation). **Show** that it preserves $q(x_{t-1}|x_0)=N(\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I)$ by induction, using the Gaussian marginalization identity (means compose linearly; variances: $(1-\bar\alpha_{t-1}-\sigma_t^2)+\sigma_t^2$). The choice $\sigma_t^2=\tilde\beta_t$ gives back the DDPM posterior.
  4. *DDIM update*:
     - $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}$.
     - $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta+\sigma_tz$ (eq. 12, in Ho's notation).
     - Interpretation: predict the clean image, then re-noise it to the next level, reusing the predicted noise direction.
  5. *η and step skipping*: $\sigma_t=\eta\sqrt{\tilde\beta_t}$; η = 1 recovers DDPM-like stochasticity, η = 0 is deterministic. Use a sub-sequence τ of timesteps (e.g. 50 of 1000) without retraining (the same marginal argument).
  6. *Deterministic DDIM = an ODE solver*: with η = 0, divide eq. 12 by $\sqrt{\bar\alpha_{t-1}}$ to get $\bar x_{t-1}=\bar x_t+(\bar\sigma_{t-1}-\bar\sigma_t)\epsilon_\theta$ with $\bar x=x/\sqrt{\bar\alpha}$, $\bar\sigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$: an Euler step of $d\bar x=\epsilon_\theta\,d\bar\sigma$, which is the PF-ODE in these variables (derive). This is DPM-Solver-1 (first-order exponential integrator) and the flow-matching Euler sampler (Gao et al.).
  7. *Consequences*: the same $x_T$ gives the same image (consistency); meaningful latent interpolation (slerp in $x_T$); DDIM inversion for editing, which is approximate because it evaluates ε at $x_t$ instead of the unknown next point. Link to SDEdit-style partial noising (pointer to gen.latent-diffusion).
- **Question ideas**
  - DDIM with η = 0: is sampling stochastic given x_T? (no)
  - (calc) DDIM step: $\bar\alpha_t=0.25$, $\bar\alpha_{t-1}=0.64$, $x_t=1.0$, $\epsilon_\theta=0.5$, η=0 → $\hat x_0=(1-0.866\cdot0.5)/0.5\approx1.134$, $x_{t-1}=0.8\cdot1.134+0.6\cdot0.5\approx1.207$.
  - Why can DDIM reuse a DDPM-trained network? (the loss only sees the marginals q(x_t|x_0))
  - Derivation step: in the marginal-preservation induction, which two variances add to $1-\bar\alpha_{t-1}$?
  - Which is false: "DDIM requires retraining the model with a different objective".
  - Predict: DDIM inversion followed by DDIM sampling with 10 vs 100 steps (reconstruction error larger at 10, since the "ε at the current point" approximation is worse with big steps).
- **Figures**
  - `gen.ddim/ddim-step`: geometry of one DDIM step (predict x̂0, then re-noise to t−1) in 2-D.
  - `gen.ddim/trajectories`: 2-D toy trajectories for DDPM (stochastic) vs DDIM (deterministic) at 10 and 50 steps, from the same $x_T$. **[fig-Q]**
- **Pitfalls & source disagreements**
  - DDIM's α = Ho's ᾱ (App. C.2). η semantics.
  - "DDIM" names both the deterministic sampler and the whole σ-family.

---

#### 150 · `gen.diffusion-solvers`: Fast Sampling II: ODE Solvers & the EDM Design Space
- **Level** advanced · **W2** · **prereqs** [gen.ddim].
- **Card budget** (about 9): EDM's common framework · truncation error and the ρ step schedule · Heun · exponential integrators / DPM-Solver · stochastic sampling (churn) and SDE vs ODE · preconditioning derivation · training-noise distribution and loss weighting · guidance interactions · probes + key results.
- **Sources**
  - Karras, Aittala, Aila & Laine 2022 EDM (`papers/karras2022_edm.txt`):
    - §2 and Table 1 the common framework (σ(t), s(t); the ODE $dx/d\sigma=(x-D(x;\sigma))/\sigma$ for s=1, σ=t); App. B.1–B.3.
    - §3 deterministic sampling: Heun's 2nd-order method (Alg. 1), the step schedule with ρ=7; App. B.4.
    - §4 stochastic sampling (churn); App. B.5.
    - §5 preconditioning ($c_{skip},c_{out},c_{in},c_{noise}$), loss weighting, log-normal σ sampling ($P_{mean}=-1.2$, $P_{std}=1.2$, $\sigma_{data}=0.5$); **App. B.6 the first-principles derivation of the c's**.
  - Lu et al. 2022 DPM-Solver (`papers/lu2022_dpm_solver.txt`): §2.2 diffusion ODEs, §3.1 the exact solution via variation of constants with λ = log(α/σ) (**half** log-SNR), §3.2 high-order solvers; DPM-Solver-1 = DDIM.
  - Murphy PML2 §25.5 speeding up diffusion models (pdf p.916–918). Dieleman perspectives (`web/dieleman2023_perspectives.txt`), Gao et al. 2024 (churn as partial undoing of a DDIM step), secondary.
- **Subtopic map**
  1. *EDM's common framework*: $x=s(t)(y+\sigma(t)n)$; with s = 1 and σ(t) = t the PF-ODE is $dx/d\sigma=-\sigma\nabla\log p(x;\sigma)=(x-D(x;\sigma))/\sigma$, where D is the optimal denoiser (Tweedie). Place DDPM/VP and NCSN/VE in this table (Table 1). Sampling cost = NFE.
  2. *Truncation error and step placement*: Euler's local error grows with curvature; curvature is largest at low σ. EDM's $\sigma_i=\big(\sigma_{max}^{1/\rho}+\frac{i}{N-1}(\sigma_{min}^{1/\rho}-\sigma_{max}^{1/\rho})\big)^\rho$ with ρ = 7 concentrates steps at low σ.
  3. *Heun's 2nd-order method*: Euler predictor plus a trapezoidal corrector; NFE = 2N − 1 because the last step skips the correction (18 steps → 35 NFE). Global error O(h²) vs Euler's O(h).
  4. *Exponential integrators*: the VP PF-ODE is semi-linear, $dx/dt=f(t)x+g(t)\epsilon_\theta$. Solve the linear part exactly (variation of constants), change variable to λ = log(α/σ), and Taylor-expand $\epsilon_\theta$ in λ. First order = DDIM; second/third order = DPM-Solver-2/3; multistep DPM-Solver++(2M) and UniPC (named). 10–20 NFE for good quality.
  5. *Stochastic sampling*: "churn" adds a little noise (raising σ) then takes an ODE step down, which partially undoes the step and lets later predictions correct earlier errors. SDE samplers are more forgiving at many steps; deterministic ones win at few steps. Quality vs NFE curves.
  6. *Preconditioning* (derive, App. B.6): write $D=c_{skip}x+c_{out}F_\theta(c_{in}x;c_{noise})$ and require (i) unit-variance network input ⇒ $c_{in}=1/\sqrt{\sigma^2+\sigma_{data}^2}$; (ii) unit-variance effective training target; (iii) $c_{out}$ as small as possible so network errors are amplified least ⇒ $c_{skip}=\sigma_{data}^2/(\sigma^2+\sigma_{data}^2)$, $c_{out}=\sigma\sigma_{data}/\sqrt{\sigma^2+\sigma_{data}^2}$; loss weight $\lambda(\sigma)=1/c_{out}^2$; $c_{noise}=\frac14\ln\sigma$. Read it: at low σ, D ≈ x (skip); at high σ, D ≈ the network (predict the clean image). $c_{skip}$ is exactly the Gaussian-data optimal denoiser's shrinkage (gen.ebm-score-matching), and the target resembles v-prediction (gen.diffusion-parameterizations).
  7. *Training noise distribution*: $\ln\sigma\sim N(-1.2,1.2^2)$, because the loss can only be reduced at intermediate σ (at tiny σ the task is trivial; at huge σ the answer is the dataset mean).
  8. *Guidance interactions*: CFG doubles NFE per step (pointer to gen.guidance). High guidance scales make the ODE stiffer, so few-step solvers degrade; thresholding helps.
- **Question ideas**
  - (calc) EDM Heun with 18 steps: NFE = 35 (2 per step minus 1).
  - (calc) EDM preconditioning at σ = σ_data = 0.5: $c_{skip}=0.5$, $c_{out}\approx0.354$, $c_{in}\approx1.414$, loss weight $1/c_{out}^2=8$.
  - Predict: an ODE sampler vs an ancestral sampler at 10 NFE (the ODE sampler is usually better at low NFE).
  - Derivation step: why must $c_{skip}\to1$ and $c_{out}\to0$ as σ → 0? (the noisy input already is the answer; the network should only add a small correction)
  - Which is false: "DPM-Solver's λ is the log-SNR used in VDM" (it is half of it).
  - Predict: EDM step schedule with ρ = 1 vs ρ = 7 at 18 steps (ρ = 1 wastes steps at high σ; worse FID).
- **Figures**
  - `gen.diffusion-solvers/edm-steps`: σ_i for ρ = 1, 3 and 7 over 18 steps.
  - `gen.diffusion-solvers/precond-coefficients`: $c_{skip}$, $c_{out}$, $c_{in}$ vs σ on a log axis with σ_data = 0.5. **[fig-Q]**: "which curve is $c_{skip}$?"
  - `gen.diffusion-solvers/quality-vs-nfe`: error vs NFE for Euler, Heun and DPM-Solver-2 on a toy where the exact PF-ODE solution is known (e.g. Gaussian or Gaussian-mixture data), so the curve is computed, not schematic.
- **Pitfalls & source disagreements**
  - EDM's t = σ and EDM's x is the *noisy* sample (data is y). NFE vs "steps" (Heun uses 2 NFE per step).
  - DPM-Solver orders vs NFE accounting; λ = ½ log-SNR (B.4).

---

#### 160 · `gen.distillation`: Diffusion Distillation & Consistency Models
- **Level** advanced · **W2** · **prereqs** [gen.diffusion-solvers].
- **Sources**
  - Salimans & Ho 2022 (`papers/salimans2022_progressive_distillation.txt`): §3 the progressive distillation algorithm (a student matches two teacher DDIM steps), §4 parameterizations (why ε fails at low SNR; v and x-prediction), the target derivation, results at 4–8 steps.
  - Song, Dhariwal, Chen & Sutskever 2023 consistency models (`papers/song2023_consistency.txt`): §3 the definition (self-consistency along PF-ODE trajectories), the boundary condition via $c_{skip}/c_{out}$, multistep sampling (Alg. 1); §4 consistency distillation (Def. 1, Thm 1 error bound); §5 consistency training (Thm 2).
  - Liu, Gong & Liu 2023 rectified flow §2.2, the "Reflow" and "Distillation" paragraphs (p.7–8) (`papers/liu2022_rectified_flow.txt`), as the flow-based counterpart. (§2.3 is the nonlinear extension, not distillation.)
  - Murphy PML2 §25.5.3 distillation (pdf p.917). Karras et al. 2022 (preconditioning reused by consistency models; `papers/karras2022_edm.txt`).
- **Subtopic map**
  1. *Why distil*: even good ODE solvers need ~10–50 NFE, but deployment wants 1–4. Idea: train a student to jump along the teacher's deterministic (PF-ODE/DDIM) trajectories.
  2. *Progressive distillation*:
     - The student initialized from the teacher learns to match **two** teacher DDIM steps with **one**.
     - Repeat, halving the steps each round: 1024 → 512 → … → 4 (calc: the number of rounds).
     - Derive the student's target $\tilde x$ that makes one student step land where two teacher steps land.
     - Why ε-prediction breaks at few steps: at low SNR, x̂0 from ε is ill-conditioned. Hence v- or x-prediction.
  3. *Consistency models*:
     - Learn $f_\theta(x_t,t)$ mapping any point on a PF-ODE trajectory to its origin $x_\epsilon$.
     - Self-consistency $f(x_t,t)=f(x_{t'},t')$ for points on the same trajectory.
     - Boundary condition $f(x,\epsilon)=x$, enforced by the parameterization $c_{skip}(t)x+c_{out}(t)F_\theta$ with $c_{skip}(\epsilon)=1$ and $c_{out}(\epsilon)=0$.
  4. *Consistency distillation (CD)*: use one teacher ODE-solver step from $x_{t_{n+1}}$ to get $\hat x_{t_n}$, then minimize $d(f_\theta(x_{t_{n+1}}),f_{\theta^-}(\hat x_{t_n}))$ with an EMA target network θ⁻. *Consistency training (CT)*: no teacher, using the unbiased score estimate from $x_0$ (the same noise pair). One-step generation with optional multistep refinement.
  5. *Rectified-flow reflow* as distillation by straightening (pointer to gen.rectified-flow).
  6. *Adversarial and distribution-matching distillation* (ADD, DMD, named): adding GAN-style or score-distillation losses for 1–4 step sampling.
  7. *Trade-offs*:
     - Some loss of diversity and fidelity at 1 step.
     - Training cost.
     - Interaction with guidance (guidance distillation, named).
     - Losing the ability to trade NFE for quality, partly restored by multistep consistency sampling.
- **Question ideas**
  - (calc) Progressive distillation from 1024 to 4 steps: 8 rounds.
  - Why does progressive distillation switch away from ε-prediction?
  - Consistency-model boundary condition: what must $c_{skip}(\epsilon)$ and $c_{out}(\epsilon)$ be? (1 and 0)
  - CD vs CT: which needs a pretrained diffusion teacher? (CD)
  - Which is false: "consistency models can only generate in exactly one step".
- **Figures**
  - `gen.distillation/progressive-halving`: teacher 2-step vs student 1-step arrows on a trajectory, repeated across rounds. **[fig-Q]**
  - `gen.distillation/consistency-map`: PF-ODE trajectories with all points mapping to the same endpoint under f.
- **Pitfalls & source disagreements**
  - The consistency-model time convention follows EDM (σ = t, with a small ε > 0 lower limit).
  - "Distillation" covers trajectory and distribution-matching approaches with different guarantees.

---

#### 170 · `gen.guidance`: Classifier & Classifier-Free Guidance
- **Level** intermediate · **W1** · **prereqs** [gen.diffusion-parameterizations] (needs score = −ε/σ).
- **Card budget** (about 9–10): why conditioning alone is weak · Bayes on scores · classifier guidance · CFG training and sampling · the implicit-classifier derivation · the trade-off · over-saturation: mechanism and fixes · what guidance is not (and costs/variants) · probes + key results.
- **Sources**
  - Dhariwal & Nichol 2021 (`papers/dhariwal2021_guided_diffusion.txt`): §4.1 conditional reverse process (proof in App. H), §4.2 conditional DDIM, Alg. 1 & 2, §4.3 scaling classifier gradients (the $p(y|x)^s/Z$ interpretation), the fidelity–diversity trade-off; architecture improvements (§3).
  - Ho & Salimans 2022 CFG (`papers/ho2022_cfg.txt`): §3.2 joint training with a null label ($p_{uncond}$), eq. (6) $\tilde\epsilon=(1+w)\epsilon(z,c)-w\epsilon(z)$, the implicit-classifier interpretation, FID/IS vs w.
  - Song et al. 2021 SDE §5 controllable generation (`papers/song2021_sde.txt`). Murphy PML2 §25.6.1–25.6.3 conditional diffusion, classifier guidance, CFG (pdf p.919–920). `pml2.txt`
  - Luo 2022, "Guidance" (p.20–22; unnumbered) (`papers/luo2022_diffusion_unified.txt`).
  - Dieleman 2022 "Guidance: a cheat code" (`web/dieleman2022_guidance.txt`), secondary: the temperature/sharpening interpretation and the caveats. Rombach et al. 2022 (CFG in LDM; `papers/rombach2022_ldm.txt`).
  - Not cached (see B.3): Saharia et al. 2022 (dynamic thresholding), Lin et al. 2024 (CFG rescale), Kynkäänniemi et al. 2024 (guidance interval). Fetch before writing item 6; until then give the mechanism in one line each with the paper named.
- **Subtopic map**
  1. *Conditioning a diffusion model*: train $\epsilon_\theta(x_t,t,c)$ directly (class embedding, text cross-attention). Why plain conditioning often gives weakly class-consistent samples (the model hedges).
  2. *Bayes on scores*: $\nabla\log p_t(x|y)=\nabla\log p_t(x)+\nabla\log p_t(y|x)$, because $p(y)$ has no x-dependence (derive). The guidance term pushes samples toward regions a classifier thinks are class y.
  3. *Classifier guidance*:
     - Train a classifier $p_\phi(y|x_t,t)$ **on noisy inputs** at all noise levels (why a clean-image classifier fails).
     - Modified noise prediction $\hat\epsilon=\epsilon_\theta(x_t)-s\sqrt{1-\bar\alpha_t}\nabla\log p_\phi(y|x_t)$, i.e. a mean shift by $s\Sigma\nabla\log p_\phi$ in the ancestral form.
     - A scale s > 1 approximates sampling from $p(x|y)\,p(y|x)^{s-1}\propto p(x)p(y|x)^s$: a sharpened classifier, trading diversity for fidelity.
     - Downsides: an extra noisy classifier to train, and adversarial-gradient concerns.
  4. *Classifier-free guidance*:
     - Train one network on both conditional and unconditional tasks by dropping c with probability $p_{uncond}$ (0.1–0.2), replacing it with a null token.
     - At sampling, $\tilde\epsilon=\epsilon_u+(1+w)(\epsilon_c-\epsilon_u)=(1+w)\epsilon_c-w\epsilon_u$.
     - **Derive the implicit classifier** (Ho & Salimans §3.2): by Bayes, $\nabla\log p_t(c|x)=\nabla\log p_t(x|c)-\nabla\log p_t(x)=-(\epsilon_c-\epsilon_u)/\sigma_t$. Classifier guidance with weight w on this implicit classifier, applied to the conditional model, gives $\epsilon_c-w\sigma_t\nabla\log p(c|x)=(1+w)\epsilon_c-w\epsilon_u$; equivalently the unconditional model with scale 1+w. The (heuristic) target is $p(x|c)^{1+w}p(x)^{-w}\propto p(x)\,p(c|x)^{1+w}$. Note Ho & Salimans's caveat: $\epsilon_\theta$ is not the gradient of any actual classifier, so this is an analogy, not an identity.
  5. *The guidance-scale trade-off*:
     - As w increases, IS and precision rise, recall falls, and FID is U-shaped (Ho Fig.).
     - Text-to-image systems typically use a CFG scale ≈ 5–10 (state as typical, not universal).
     - Class-conditional ImageNet models want much less: Ho & Salimans §4.2 sweep w ∈ {0, 0.1, …, 4} and get the best FID at w ≈ 0.1–0.3 (s ≈ 1.1–1.3), with IS still rising up to w = 4. SD3 evaluates at scale 5.0 (App. B.3).
  6. *Over-saturation at high scales: mechanism and fixes*:
     - Mechanism: the guided prediction extrapolates past $\epsilon_c$, so the implied $\hat x_0=(x_t-\sigma_t\tilde\epsilon)/\alpha_t$ leaves the data range (e.g. pixels beyond [−1, 1]), the sample's mean and contrast inflate, and colours saturate. The effect compounds over steps and is worst at high noise, where $\sigma_t/\alpha_t$ is large.
     - Static thresholding: clip $\hat x_0$ to [−1, 1] at each step (cheap; flattens detail).
     - Dynamic thresholding (Saharia et al. 2022): clip $\hat x_0$ to its p-th percentile s (if s > 1) and divide by s.
     - CFG rescale (Lin et al. 2024): rescale the guided prediction so its per-sample std matches the conditional prediction's, then blend.
     - Guidance interval (Kynkäänniemi et al. 2024): apply guidance only at intermediate noise levels; at high noise it collapses diversity, at low noise it does little.
  7. *What guidance does and doesn't do*: the guided "distribution" isn't the sampling of a well-defined product at every t (the score of a tempered mixture ≠ the tempered score; Dieleman's discussion). It is a heuristic that works.
  8. *Costs and variants*: CFG doubles NFE (batch the cond and uncond passes). Negative prompts replace $\epsilon_u$ with $\epsilon_{neg}$. Guidance distillation (named). Guidance with flow-matching velocities works the same way (pointer to gen.flow-matching).
  9. *Other conditioning*: inpainting and inverse problems via likelihood guidance (named), and conditioning mechanisms (pointer to gen.latent-diffusion).
- **Question ideas**
  - (calc) Ho's w=3 equals which "CFG scale" s in $\epsilon_u+s(\epsilon_c-\epsilon_u)$? (s=4)
  - What does w=0 (Ho) / s=1 give? (plain conditional sampling)
  - (calc) $\epsilon_u=0.2$, $\epsilon_c=0.5$, s=3 → $\tilde\epsilon=1.1$.
  - Why must the classifier in classifier guidance be trained on noisy inputs?
  - Predict: diversity and FID as the guidance scale goes from 1 to 20.
  - Which is false: "CFG requires training a separate classifier".
  - Derivation step: in the implicit-classifier derivation, why does $\log p(c)$ drop out of the gradient? (no x-dependence)
  - Predict: at w = 8 the generated images have blown-out colours. Which quantity left its training range, and name one fix. ($\hat x_0$; dynamic thresholding / CFG rescale / guidance interval)
- **Figures**
  - `gen.guidance/score-composition`: 2-D vector fields for the unconditional score, the classifier gradient, and their scaled sum, with samples concentrating on one class.
  - `gen.guidance/tradeoff-toy`: computed on a toy (2-D two-class Gaussian mixture with exact scores, sampled with the guided PF-ODE): class consistency (precision-like) rises and sample spread / mode coverage (recall-like) falls as w grows. **[fig-Q]**. Only reproduce Ho & Salimans's FID/IS-vs-w curve if labelled "redrawn from Ho & Salimans Fig." with values read from the paper.
  - `gen.guidance/oversaturation`: on the same toy (data in [−1, 1]), the distribution of $\hat x_0$ at a mid-noise step for w = 0, 3, 10, with the fraction outside [−1, 1] marked.
  - `gen.guidance/toy-sharpening`: a 1-D mixture of two classes, with the guided density $\propto p(x)p(y|x)^s$ for s = 1, 3, 10.
- **Pitfalls & source disagreements**
  - **w vs s convention clash** (Ho & Salimans's (1+w) vs the SD/Imagen-style scale where s=1 means no guidance).
  - The classifier-guidance "scale" in Dhariwal multiplies the gradient. Sign conventions between ε- and score-space.

---

#### 180 · `gen.flow-matching`: Flow Matching
- **Level** advanced · **W1** · **prereqs** [gen.flows, gen.diffusion-parameterizations]. Written in W1 before gen.flows (W2), so items 1–2 must recap CNFs, divergence and the continuity equation inline.
- **Card budget** (about 9): recap & motivation · continuity equation · FM objective & why it's intractable · conditional paths + Thm 1 derivation · CFM gradient proof · Gaussian paths & the OT path · training/sampling algorithm · conditional straightness vs marginal curvature · probes + key results.
- **Sources**
  - Lipman, Chen, Ben-Hamu, Nickel & Le 2023 (`papers/lipman2023_flow_matching.txt`):
    - §2 CNFs and the continuity equation.
    - §3 the flow-matching objective.
    - §3.1 constructing $p_t,u_t$ from conditional paths, with Thm 1 (the marginal field generates the marginal path).
    - §3.2 conditional flow matching, with Thm 2 (∇CFM = ∇FM).
    - §4 Gaussian conditional paths: Thm 3, the general $u_t(x|x_1)$ formula; §4.1 special instances: diffusion paths (VE/VP) and the OT path $\mu_t=tx_1$, $\sigma_t=1-(1-\sigma_{min})t$.
    - App. A proofs.
  - Chen et al. 2018 neural ODEs (`papers/chen2018_neural_ode.txt`) and Grathwohl et al. 2019 FFJORD (`papers/grathwohl2018_ffjord.txt`): CNF background and why maximum-likelihood CNF training is expensive.
  - Tong et al. 2024 (`papers/tong2023_ot_cfm.txt`): the conditional-FM family, minibatch OT couplings. Albergo, Boffi & Vanden-Eijnden 2023 stochastic interpolants (`papers/albergo2023_stochastic_interpolants.txt`).
  - Kingma & Gao 2023 App. D.3 (`papers/kingma2023_diffusion_objectives.txt`): FM-OT as a weighted diffusion loss.
  - Murphy PML2 §25.4.7 flow matching (pdf p.915). Gao et al. 2024 (`web/gao2024_diffusion_meets_flow_matching.txt`), secondary.
- **Subtopic map**
  1. *Recap and motivation*: a CNF $dx/dt=v_\theta(x,t)$ transports a simple $p_0$ to the data $p_1$ (flows lesson). Maximum-likelihood training requires simulating the ODE and computing traces, which is expensive. Goal: train v by **simulation-free regression**.
  2. *Probability paths and the continuity equation*: a time-dependent density $p_t$ is generated by a vector field $u_t$ iff $\partial_tp_t+\nabla\cdot(p_tu_t)=0$. Inline refresher: divergence as net outflow; 1-D intuition (mass in an interval changes only by flux through its ends). Pointer back to the PF-ODE derivation in gen.score-sde, which uses the same equation.
  3. *Flow-matching objective*: $E_{t,x\sim p_t}\|v_\theta(x,t)-u_t(x)\|^2$. This is intractable because neither $p_t$ nor $u_t$ is known for the marginal path.
  4. *Conditional paths*: choose simple per-sample paths $p_t(x|x_1)$ (e.g. Gaussian, from $N(0,I)$ at t=0 to near $\delta_{x_1}$ at t=1) with known $u_t(x|x_1)$. The marginal path is $p_t(x)=\int p_t(x|x_1)q(x_1)dx_1$, and the **marginal field** is $u_t(x)=E[u_t(x|x_1)\mid x_t=x]$ (Bayes-weighted average of the conditional fields: $u_t(x)=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1$). **Derive Thm 1** in three lines: differentiate the mixture under the integral, apply each conditional continuity equation, and pull the divergence out, giving $\partial_tp_t=-\nabla\cdot(p_tu_t)$.
  5. *Conditional FM*: $\mathcal L_{CFM}=E_{t,x_1,x\sim p_t(\cdot|x_1)}\|v_\theta(x,t)-u_t(x|x_1)\|^2$. **Prove** that $\nabla\mathcal L_{CFM}=\nabla\mathcal L_{FM}$: expand both squares; the $\|v\|^2$ terms match; the cross terms match because $u_t(x)$ is the conditional expectation; the rest is constant in θ. The regression-to-the-conditional-mean intuition.
  6. *Gaussian conditional paths*:
     - $x_t=\mu_t(x_1)+\sigma_t(x_1)\epsilon$ has $u_t(x|x_1)=\frac{\sigma_t'}{\sigma_t}(x-\mu_t)+\mu_t'$ (derive from the flow map $\psi_t(\epsilon)$).
     - **OT path**: $\mu_t=tx_1$, $\sigma_t=1-(1-\sigma_{min})t$, giving $u_t(x|x_1)=\frac{x_1-(1-\sigma_{min})x}{1-(1-\sigma_{min})t}$. As σ_min → 0, $x_t=(1-t)x_0+tx_1$ with target $x_1-x_0$.
     - Diffusion paths (VP/VE) are special cases, with time reversed relative to diffusion conventions.
  7. *Training algorithm*: sample t, $x_0\sim N(0,I)$, $x_1\sim$ data. Form $x_t$. Regress $v_\theta(x_t,t)$ onto the target. About 5 lines.
  8. *Sampling*: integrate the ODE from t=0 to 1 with Euler/Heun/adaptive solvers. Exact likelihood via the instantaneous change of variables. Guidance applies to velocities.
  9. *Conditional straightness vs marginal curvature*: each conditional path is straight, but the marginal trajectories bend because conditional paths cross and get averaged. That's why few-step sampling still has error. Minibatch OT couplings (Tong) and reflow (next lesson) straighten them.
- **Question ideas**
  - OT-path target with σ_min=0: given $x_0$ (noise) and $x_1$ (data), what does the network regress onto? ($x_1-x_0$, constant along the path)
  - Why is CFM a valid surrogate for FM? (equal gradients; the marginal field is a conditional expectation)
  - (calc) Linear path at t=0.25 with $x_0=(0,0)$, $x_1=(4,8)$: $x_t=(1,2)$, target (4, 8).
  - Do straight conditional paths imply straight marginal trajectories? (no)
  - Derivation step: in the ∇CFM = ∇FM proof, the cross term $E_{p_t(x)}\langle v_\theta,u_t(x)\rangle$ is rewritten as $E_{q(x_1)p_t(x|x_1)}\langle v_\theta,u_t(x|x_1)\rangle$. Which definition makes this valid? (the marginal field as the Bayes-weighted average of conditional fields)
  - Predict: at a point x midway between two data points at t = 0.5 (OT path, σ_min = 0), what does the optimal $v_\theta$ output? (the posterior-weighted average of the two conditional targets, which is shorter than either: the source of marginal curvature)
  - Which is false: "flow-matching training requires simulating the ODE during training".
- **Figures**
  - `gen.flow-matching/conditional-vs-marginal`: 2-D straight conditional paths from noise samples to data points vs the curved learned marginal trajectories. **[fig-Q]**
  - `gen.flow-matching/probability-path`: a 1-D density evolving from N(0,1) to a bimodal target over t (heat map).
  - `gen.flow-matching/training-diagram`: the CFM training step (sample t, x0, x1 → x_t → regress onto the target).
- **Pitfalls & source disagreements**
  - **Time direction**: Lipman and Liu use t=0 noise → t=1 data. DDPM/score-SDE and SD3 use t=0 data (table B.4).
  - "OT path" means the conditional (per-pair) OT displacement, not the OT map between the two distributions.

---

#### 190 · `gen.rectified-flow`: Rectified Flow & the Diffusion–Flow Equivalence
- **Level** advanced · **W2** · **prereqs** [gen.flow-matching, gen.score-sde]. Builds on gen.diffusion-parameterizations for the conversions; this lesson adds the velocity to that family.
- **Sources**
  - Liu, Gong & Liu 2023 (`papers/liu2022_rectified_flow.txt`): §2.1 rectified flow (the linear interpolation $X_t=tX_1+(1-t)X_0$, the loss, the non-crossing property), §2.2 main results: marginal preservation (Thm 3.3), reduced convex transport costs (Thm 3.5), reflow and its straightening (Thm 3.7), the "Distillation" paragraph (p.8); §2.3 the nonlinear extension (no straightening guarantee).
  - Gao, Hoogeboom, Heek, De Bortoli, Murphy & Salimans 2024 (`web/gao2024_diffusion_meets_flow_matching.txt`): Gaussian flow matching ≡ diffusion; the equivalence of samplers, weightings and network outputs; the schedule's effect.
  - Esser et al. 2024 SD3 (`papers/esser2024_sd3_rectified_flow.txt`): §2 a unified view of the objectives as weighted ε-losses (following Kingma & Gao), §3 flow trajectories (RF, EDM, cosine, LDM-linear), §3.1 tailored SNR samplers (logit-normal, mode sampling with heavy tails, CosMap), §5.1 the comparison (rf/lognorm(0, 1) wins), §5.3.2 resolution-dependent timestep shifting.
  - Kingma & Gao 2023 (`papers/kingma2023_diffusion_objectives.txt`): weighting functions; FM's implied weighting. Salimans & Ho 2022 (v-prediction; `papers/salimans2022_progressive_distillation.txt`).
  - Albergo et al. 2023 (`papers/albergo2023_stochastic_interpolants.txt`), for the unifying interpolant view (named).
- **Subtopic map**
  1. *Rectified flow*: the same objective as FM with the linear path (σ_min = 0), $\min E\|X_1-X_0-v(X_t,t)\|^2$. Liu's framing as learning a transport map between *any* two distributions (noise→data, or domain→domain).
  2. *Non-crossing and reflow*:
     - Paths of the learned ODE don't cross, so the induced coupling $(Z_0,Z_1)$ is deterministic.
     - **Reflow**: retrain on pairs generated by the current flow, which gives straighter trajectories (Thm 3.7: the best straightness measure among the first K rectified flows is O(1/K)) and does not increase any convex transport cost (Thm 3.5; statement with the Jensen intuition). Liu advises against many reflow rounds: each one accumulates the previous flow's errors.
     - Straight paths make few-step (even 1-step Euler) sampling accurate.
     - Distillation of the reflowed model (pointer to gen.distillation).
  3. *Gaussian FM ≡ diffusion* (Gao et al.):
     - For a Gaussian path $x_t=\alpha_tx_1+\sigma_t\epsilon$, the velocity $v=\dot\alpha_tx_1+\dot\sigma_t\epsilon$ is a linear combination of the x̂1- and ε-predictions.
     - So a velocity network, an ε-network and a score network are interconvertible.
     - The FM ODE is the probability-flow ODE of a diffusion with that schedule.
     - The DDIM sampler is identical to the FM Euler sampler for the linear schedule, and DDIM is invariant to linearly rescaling (α, σ), whereas Euler on the PF-ODE is not (Gao et al.).
  4. *Weightings and schedules*: for the linear path with data coefficient α_t and noise coefficient σ_t (α+σ = 1), derive $\|\hat v-v\|^2=\|\hat\epsilon-\epsilon\|^2/\alpha_t^2=\|\hat x_{data}-x_{data}\|^2/\sigma_t^2$, i.e. FM is the ε-loss weighted by $(1+\mathrm{SNR}^{-1/2})^2$ (calc: a factor 4 at SNR = 1; calc-checked numerically). Choosing the time-sampling density changes the effective weighting further. SD3's logit-normal(0, 1) sampling puts most mass near t = 0.5 (intermediate noise), and mode sampling with heavy tails keeps the endpoints. Resolution-dependent timestep shifts (SD3 §5.3.2).
  5. *v-prediction connection*: with the trigonometric schedule (α = cos, σ = sin), Salimans & Ho's v equals the velocity up to sign and scale. Show it.
  6. *Practice at scale*: rectified-flow transformers (SD3, Flux-style, named) in latent space (pointer to gen.latent-diffusion). Few-step sampling. CFG on velocities.
  7. *Choosing between "diffusion" and "flow matching"*: mostly a choice of path/schedule, parameterization and weighting, plus deterministic vs stochastic sampling. The same family underneath (Gao et al.'s conclusion). Stochastic interpolants as the general framework (named).
- **Question ideas**
  - (calc) Reflowed, perfectly straight flow: one Euler step from t=0 to 1 with velocity v gives $x_1=x_0+v$. Exact? (yes, if the flow is exactly straight)
  - Convert: the linear path $x_t=(1-t)x_0+tx_1$ (x0 noise) with velocity v and $x_t$ known → $\hat x_1=x_t+(1-t)v$ and $\hat\epsilon=x_t-tv$ (derive).
  - Why does SD3 sample t from a logit-normal? (to emphasize intermediate noise levels where learning is hardest)
  - (calc) Linear path at SNR = 1 (t = 0.5): by what factor does the FM loss weight the ε-error? (4)
  - Which is false: "flow matching with Gaussian paths and diffusion models learn fundamentally different generative processes".
  - Predict: few-step Euler sampling before vs after reflow.
- **Figures**
  - `gen.rectified-flow/reflow-straightening`: 2-D trajectories before and after one reflow iteration. **[fig-Q]**: "which is after reflow?"
  - `gen.rectified-flow/equivalence-diagram`: boxes for ε-, x̂1-, v- and score-prediction with the linear conversion maps on the arrows.
  - `gen.rectified-flow/timestep-densities`: uniform vs logit-normal vs mode-shifted t sampling densities.
- **Pitfalls & source disagreements**
  - Liu's RF and Lipman's FM-OT share a loss but emphasize different things (reflow vs Gaussian-path theory).
  - SD3 uses t=1 noise, the opposite of Liu and Lipman.

---

#### 200 · `gen.latent-diffusion`: Latent Diffusion & Conditioning
- **Level** intermediate · **W2** · **prereqs** [gen.vae-variants, gen.guidance].
- **Sources**
  - Rombach, Blattmann, Lorenz, Esser & Ommer 2022 LDM (`papers/rombach2022_ldm.txt`):
    - §3.1 the perceptual compression stage (an autoencoder with KL- or VQ-regularization, perceptual + patch-adversarial losses), the downsampling factor f; §3.2 latent diffusion.
    - §3.3 conditioning via cross-attention.
    - §4.1 perceptual compression trade-offs (f), §4.2–4.5 generation, conditional/text-to-image (with CFG), super-resolution, inpainting.
    - App. G (p.29) the autoencoder details, including rescaling KL-regularized latents by $1/\hat\sigma$ estimated from the first batch (VQ latents are left unscaled).
  - Murphy PML2 §25.5.4 latent-space diffusion, §25.6.4 generating high-resolution images (pdf p.918–920), §21.6.6 VQ-GAN (pdf p.853). `pml2.txt`
  - van den Oord et al. 2017 VQ-VAE (`papers/vandenoord2017_vqvae.txt`) for the discrete-latent route. Esser et al. 2024 SD3 (`papers/esser2024_sd3_rectified_flow.txt`): MM-DiT, latent channels, rectified flow in latent space.
  - Ho & Salimans 2022 (`papers/ho2022_cfg.txt`) for CFG in text-to-image. Backbones (U-Net, DiT, MM-DiT) are in gen.diffusion-transformers.
- **Subtopic map**
  1. *Why latent*: pixel-space diffusion spends most of its compute on imperceptible high-frequency detail. Separate **perceptual compression** (an autoencoder) from **semantic generation** (diffusion in latent space). The number of spatial positions drops by $f^2$ (calc: 512² at f = 8 → 64² latents), and attention cost by up to $f^4$.
  2. *The first stage autoencoder*:
     - Encoder E downsamples by f ∈ {4, 8, 16} to $h\times w\times c$ latents. Decoder D reconstructs.
     - Training losses: L1/perceptual (LPIPS) + patch-GAN adversarial + a light KL toward N(0, I) or VQ regularization. Why not a plain VAE: blurry reconstructions (pointer to gen.vae).
     - The trade-off in f: small f is expensive, large f loses detail (Rombach's ablation).
  3. *Latent scaling*: rescale latents to roughly unit variance before diffusion (Rombach App. G: divide by the std estimated on the first batch; the SD v1 code hard-codes 0.18215, which is a code constant, not a number in the paper). Why: the noise schedule and the SNR values assume unit-variance data, so a latent with std ≈ 5 would see a much higher effective SNR than intended.
  4. *The second stage*: a standard diffusion (or flow-matching) model on z = E(x). Sample z, then decode with D. Training is the same as pixel diffusion with ε/v targets.
  5. *Conditioning mechanisms*:
     - Concatenation (for spatially aligned conditions such as masks and low-res images).
     - Adaptive normalization (class or time embeddings; adaLN in DiTs).
     - **Cross-attention**: queries from U-Net features, keys and values from text-encoder tokens ($\mathrm{softmax}(QK^\top/\sqrt d)V$; pointer to the LLM area).
     - Text encoders (CLIP/T5) and pooled vs token embeddings.
  6. *CFG in text-to-image*: conditional dropout of the text during training, guidance at sampling, negative prompts (pointer to gen.guidance).
  7. *Architectures* (one paragraph): U-Net with attention vs diffusion transformers on latent patches; details in gen.diffusion-transformers.
  8. *Applications in one card*: inpainting (mask conditioning), super-resolution, image-to-image (SDEdit-style partial noising, named), editing via DDIM inversion (pointer to gen.ddim), video latents (named; vision area).
  9. *Limitations*: decoder artifacts (text, faces), latent-space bias, memorization concerns (pointer to gen.evaluation).
- **Question ideas**
  - (calc) 512×512×3 image with f=8 and c=4 latents: 64×64×4 = 16,384 latent values vs 786,432 pixels (48× fewer).
  - Why rescale latents (e.g. ×0.18215 in SD v1)? (the noise schedule assumes roughly unit-variance data)
  - Predict: forgetting the latent scaling with a latent std of about 5. (the effective SNR at every t is ~25× higher than designed, so the model rarely sees near-pure noise and sampling from N(0, I) is mismatched)
  - Which conditioning mechanism suits a text prompt vs an inpainting mask? (cross-attention vs concatenation)
  - Predict: f = 32 vs f = 4 (faster but blurrier detail vs slow but high fidelity).
  - Which is false: "the LDM autoencoder is trained jointly with the diffusion model end-to-end".
  - Cross-attention: which side provides the queries? (the image/U-Net features)
- **Figures**
  - `gen.latent-diffusion/two-stage`: pipeline diagram (encoder → latent diffusion with text cross-attention → decoder).
  - `gen.latent-diffusion/f-tradeoff`: schematic reconstruction quality vs compute for f = 1…32. **[fig-Q]**
  - `gen.latent-diffusion/cross-attention`: queries from spatial latents attending to text tokens, with an attention-map illustration.
- **Pitfalls & source disagreements**
  - The scaling factor differs by model and VAE (SD v1 vs SDXL vs SD3) and is a property of the code release, not of the method.
  - "Latent diffusion" vs "Stable Diffusion" (a specific trained LDM).
  - KL-reg vs VQ-reg first stages.

---

#### 210 · `gen.diffusion-transformers`: Diffusion Backbones: U-Net, DiT & MM-DiT
- **Level** intermediate · **W2** · **prereqs** [gen.latent-diffusion]. Pointers: the LLM area for attention and transformer blocks (`llm.*` attention lessons), the vision area for ViT. Added in review: labs doing image and video generation ask about the backbone (how t and c get in, patch size vs compute, why adaLN-Zero), and gen.latent-diffusion had only a named mention. The vision area should link here for DiT rather than duplicate it, and own ViT and video diffusion.
- **Card budget** (about 8): what the backbone must do · the diffusion U-Net · conditioning injection (add, AdaGN/adaLN, cross-attention, in-context tokens) · DiT: patchify, blocks, adaLN-Zero, unpatchify · patch size and compute · MM-DiT · scaling behaviour + video extension (pointer) · probes + key results.
- **Sources**
  - Ho et al. 2020 §4 and App. B (`papers/ho2020_ddpm.txt`): the PixelCNN++-style U-Net, group norm, sinusoidal time embedding, self-attention at 16×16.
  - Dhariwal & Nichol 2021 §3 (`papers/dhariwal2021_guided_diffusion.txt`): architecture ablations (depth vs width, more attention heads and resolutions, BigGAN-style up/down-sampling blocks), adaptive group normalization (AdaGN).
  - Rombach et al. 2022 §3.3 (`papers/rombach2022_ldm.txt`): cross-attention inside the U-Net.
  - Esser et al. 2024 SD3 §4 (`papers/esser2024_sd3_rectified_flow.txt`): MM-DiT (2×2 patching of latents, separate weights for text and image tokens, joint attention over both, modulation from timestep + pooled text, RMSNorm on Q and K for stability), §5.3 scaling study (validation loss tracks sample quality), and the appendix experiment extending MM-DiT to video.
  - **Not cached, needed** (B.3): Peebles & Xie 2023 DiT (patchify, the four conditioning variants, adaLN-Zero, Gflops vs FID scaling). Until it is cached, DiT-specific claims (the conditioning ablation, "Gflops predict FID better than parameters") are named with the paper, not stated as checked facts.
- **Subtopic map**
  1. *What the backbone must do*: map $(x_t,t,c)$ to an output of the same shape (ε, v or x̂0) at every noise level, so it must be told the noise level and must handle global structure (high noise) and fine detail (low noise).
  2. *The diffusion U-Net*: encoder–decoder with skip connections, residual blocks with group norm, self-attention at low resolutions. Why it suits denoising: multi-scale processing, and the skips carry high-frequency detail the decoder would otherwise have to regenerate.
  3. *Getting t and c in*: sinusoidal embedding of t (as in transformer positional encodings) → MLP → added to the features, or used to predict a per-channel scale and shift of the normalized features (AdaGN; Dhariwal ablation shows it helps). Class labels join the same embedding. Text: cross-attention (gen.latent-diffusion) or joint attention (MM-DiT).
  4. *DiT*: patchify the latent into $p\times p$ patches → $(H/p)(W/p)$ tokens with 2-D positional embeddings; a stack of ViT blocks; conditioning by adaLN: regress scale/shift (and a residual gate) from the t + c embedding. **adaLN-Zero** initializes the gate to zero so every block starts as the identity, which stabilizes training (the same idea as zero-initializing the last layer of a residual branch). Unpatchify with a linear decoder back to the latent shape.
  5. *Patch size and compute*: (calc) a 32×32×4 latent (256² image at f = 8) gives 256 tokens at p = 2 and 64 at p = 4. Halving p multiplies tokens by 4, attention FLOPs by about 16 and MLP FLOPs by about 4. Smaller patches mean better samples at higher cost, so compute is set by tokens, not by parameter count alone.
  6. *MM-DiT* (SD3): text and image tokens keep separate weights but attend jointly, so text can attend to image as well as the reverse (unlike cross-attention, where only the image queries). Timestep and pooled text drive the modulation. QK-normalization keeps attention logits bounded at scale.
  7. *Scaling and why transformers took over*: predictable scaling (SD3: lower validation loss tracks better human-preference and benchmark scores), a uniform architecture shared with LLMs, and easy extension to other token layouts. Video (pointer to the vision area, named only): spatiotemporal latents from a 3-D autoencoder, spacetime patches, and the cost of full attention growing with the square of (frames × spatial tokens), which motivates factorized spatial/temporal attention.
- **Question ideas**
  - (calc) Tokens for a 64×64×4 latent with p = 2. (1024) Relative attention cost vs p = 4? (16×)
  - Why initialize the adaLN gate to zero? (each block starts as the identity, so the deep network starts as a shallow, stable function)
  - Compare: cross-attention vs MM-DiT joint attention. Which lets text tokens be updated by image content? (joint attention)
  - Predict: removing the U-Net's skip connections. (blurrier outputs; fine detail must pass through the bottleneck)
  - Which is false: "the timestep only needs to be given to the first layer of the network".
- **Figures**
  - `gen.diffusion-transformers/unet-vs-dit`: side-by-side schematics (U-Net with skips and attention at low resolution; DiT as patchify → N blocks → unpatchify).
  - `gen.diffusion-transformers/adaln-zero-block`: a DiT block with the t + c embedding feeding scale, shift and gate, the gate initialized at 0.
  - `gen.diffusion-transformers/patch-size-compute`: token count and relative attention FLOPs vs p for 32×32 and 64×64 latents. **[fig-Q]**
- **Pitfalls & source disagreements**
  - "DiT" names the specific Peebles & Xie model and, loosely, any diffusion transformer.
  - adaLN vs AdaGN vs FiLM are the same idea (feature-wise affine modulation) applied to different norms.

---

#### 220 · `gen.evaluation`: Evaluating Generative Models
- **Level** intermediate · **W2** · **prereqs** [gen.overview, gen.gans]. Placed last so it can refer to every family (mode collapse, guidance curves, diffusion memorization).
- **Sources**
  - Murphy PML2 §20.4 evaluating generative models: likelihood, distances in feature space (FID/KID), precision & recall, statistical tests, pretrained-classifier issues, overfitting/memorization, human evaluation (pdf p.816–822). `pml2.txt`
  - Theis, van den Oord & Bethge 2016 (`papers/theis2016_note_on_evaluation.txt`): likelihood vs sample quality, Parzen-window failure, the mixture argument.
  - Heusel et al. 2017 FID (`papers/heusel2017_fid.txt`); Salimans et al. 2016 IS (`papers/salimans2016_improved_gan_is.txt`); Chong & Forsyth 2020 FID bias (`papers/chong2020_unbiased_fid.txt`).
  - Kynkäänniemi et al. 2019 precision/recall (`papers/kynkaanniemi2019_gen_precision_recall.txt`); Carlini et al. 2023 memorization (`papers/carlini2023_diffusion_memorization.txt`).
  - Goodfellow DL §20.14 evaluating generative models. Ho et al. 2020 §3.3 & §4 (discrete decoder, bits/dim reporting; `papers/ho2020_ddpm.txt`).
- **Subtopic map**
  1. *Why evaluation is hard*: no single number captures fidelity, diversity, novelty and likelihood together. Different metrics disagree.
  2. *Likelihood metrics*:
     - Test NLL, and bits/dim = NLL/(D ln 2) (calc).
     - Discrete data vs continuous densities: uniform **dequantization** gives a lower bound on the discrete log-likelihood, and densities on a [0,1] rescaling carry a D log 256 offset.
     - Discretized decoders (Ho et al.).
  3. *Likelihood ≠ sample quality* (Theis), with both directions shown:
     - A 0.99·noise + 0.01·good-model mixture loses at most log 100 ≈ 4.6 nats in total, so its bits/dim barely change while 99% of samples are garbage.
     - A model that memorizes training images has great samples and terrible test likelihood.
     - Parzen-window estimates are unreliable in high dimensions.
  4. *Inception Score*: $\exp E_x\mathrm{KL}(p(y|x)\|p(y))$. High when samples are confidently classified *and* class-diverse. It ignores real data and intra-class diversity, and it's ImageNet-specific and gameable.
  5. *FID*:
     - Fit Gaussians to Inception-V3 pool features of real and generated sets: $\|\mu_r-\mu_g\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$.
     - It is the Fréchet/W2 distance between Gaussians (the formula's origin).
     - Biased upward at finite N, so compare at fixed N (50k) or extrapolate (Chong & Forsyth).
     - **Its assumptions**: (i) features are Gaussian, so only the first two moments count (two very different distributions with equal mean and covariance get FID 0); (ii) Inception-V3 ImageNet features define "perceptual" similarity, so FID is most sensitive to ImageNet-class content and can be moved without improving images (feature-extractor bias; DINOv2-based variants named).
     - Sensitive to preprocessing (resize implementation, JPEG). KID (an MMD with a polynomial kernel) as an unbiased alternative with no Gaussian assumption.
     - Video: FVD replaces Inception with a video network (named; vision area).
  6. *Precision and recall for generative models*: fidelity (are samples on the data manifold?) vs coverage (is the data manifold covered?) via k-NN radii (Kynkäänniemi). Mode collapse = high precision, low recall. Density/coverage variants (named).
  7. *Memorization and novelty*: nearest-neighbour checks (pixel vs feature space). Extraction attacks on diffusion models (Carlini). Duplicated training images get memorized more.
  8. *Conditional and text-to-image evaluation*: CLIP score for alignment and human preference studies (named). Guidance-scale trade-off curves (FID vs CLIP score; pointer to gen.guidance).
  9. *Classifier-based and downstream evaluation*: train a classifier on synthetic data and test it on real data. Two-sample tests (MMD/KID).
- **Question ideas**
  - (calc) 3072-dim image with NLL 6,000 nats → ≈2.82 bits/dim.
  - Which metric doesn't use real data? (IS)
  - (calc) FID between identical Gaussians = 0. Which term detects a variance (diversity) mismatch? (the trace term)
  - Predict: FID computed with 5k vs 50k samples from the same model (5k is higher on average, by bias).
  - Theis's mixture: what does it show? (calc) The penalty is at most log 100 ≈ 4.6 nats per image, i.e. ≈ 0.002 bits/dim for a 3072-dim image.
  - Mode collapse in precision/recall terms? (high precision, low recall)
  - Which is false: "FID = 0 implies the generated and real image distributions are identical". (only the Gaussian fits of the Inception features match)
- **Figures**
  - `gen.evaluation/precision-recall-manifolds`: real and generated point clouds with k-NN balls, showing a high-precision/low-recall case. **[fig-Q]**: "which case is mode collapse?"
  - `gen.evaluation/fid-vs-n`: FID vs the number of samples, showing the bias and the extrapolation to ∞.
  - `gen.evaluation/likelihood-vs-quality`: a 2×2 of likelihood (good/bad) × samples (good/bad) with an example model in each cell.
- **Pitfalls & source disagreements**
  - bits/dim conventions (dequantization; [0,255] vs [0,1] scaling).
  - FID implementations (TF vs PyTorch Inception weights, resizing) give different numbers. Comparing FID across datasets is meaningless.

---


*End of plan. Total: 63 fundamentals lessons + 22 generative-modeling lessons. All canonical URLs in `$SRC/INDEX.md` were checked and returned HTTP 200 on 2026-10-02.*
