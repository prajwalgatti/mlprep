# Source cache index: Deep Learning Tuning Playbook (tp_*)

Built 2026-10-02 for the `sys.tuning-*` lessons and the insertions table in `docs/content-plan-tuning.md`.
PDF-derived files have page markers like `=== [shallue2019_batch_size p.15] ===` (pdf page). Equations from PDFs are mangled
(subscripts on separate lines); rebuild maths from the PDF in `pdf/tp/`. The playbook itself is raw markdown with LaTeX intact.

## The playbook

| File | Source | URL | Licence | How to find sections |
|---|---|---|---|---|
| `tp_tuning_playbook.md` (120 KB) | Godbole, Dahl, Gilmer, Shallue & Nado, *Deep Learning Tuning Playbook*, v1.0 (2023), README.md on `main` fetched 2026-10-02 | https://github.com/google-research/tuning_playbook | CC-BY 4.0: attribute, adapt in own words | `grep -n '^##' tp_tuning_playbook.md`. Cite as `Tuning Playbook §"<heading>"`, e.g. `§"Choosing the batch size"`, `§"Identifying scientific, nuisance, and fixed hyperparameters"`, `FAQ "How should Adam's hyperparameters be tuned?"` |

Main sections (line numbers): Starting a new project 113 (architecture 131, optimizer 151, batch size 187–388, initial config 390) ·
Scientific approach 425 (incremental 440, explore/exploit 485, goals 517, scientific/nuisance/fixed 542, studies 690, budget 753,
insight checklist 800, search-space boundaries 844, training curves 901, isolation plots 996, adopting changes 1057, after exploration 1111) ·
Steps per run 1149 (not compute-bound 1186, LR-sweep algorithm 1228, compute-bound 1255, Round 1/2 1290/1339) ·
Pipeline 1368 (input 1370, eval 1400, checkpoints 1508, tracking 1529, batch norm 1549, multi-host 1573) ·
FAQs 1587 (schedules 1589–1643, Adam 1645, quasi-random 1663–1798, instability/warmup/clipping 1800–1985, metaparameter 1987,
batch size vs validation 2014, update rules 2028).

The playbook's figures (bad/good search space, isolation plot, bootstrap trial budget, stride instability, warmup, clipping) are PNGs on GitHub
(`assets/`); they are not cached. Our lessons redraw the ideas from synthetic data.

## Supporting papers (pdf in `pdf/tp/`)

| File | Paper | URL | Use it for |
|---|---|---|---|
| `tp_shallue2019_batch_size.txt` | Shallue, Lee, Antognini, Sohl-Dickstein, Frostig & Dahl, *Measuring the Effects of Data Parallelism on Neural Network Training*, JMLR 2019 (CC-BY) | https://arxiv.org/abs/1811.03600 | Steps-to-result vs batch size (perfect scaling → diminishing returns → maximal parallelism, §4.1); momentum extends perfect scaling (§4.4); label smoothing helps more at large batch (§4.6); optimal effective LR $\eta/(1-\gamma)$ does not follow linear/sqrt rules (§2.2, §4.7); step vs epoch budgets (§4.8, §5) |
| `llm_mccandlish2018_critical_batch.txt` (already cached by the LLM curator) | McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training*, 2018 | https://arxiv.org/abs/1812.06162 | Derivation of $\epsilon_{opt}(B)=\epsilon_{max}/(1+B_{noise}/B)$, $B_{noise}=\mathrm{tr}(H\Sigma)/G^\top HG$, $B_{simple}=\mathrm{tr}\Sigma/|G|^2$ (§2.2); steps/examples tradeoff $(S/S_{min}-1)(E/E_{min}-1)=1$, $B_{crit}=E_{min}/S_{min}$ (§2.3); caveats (§2.4) |
| `papers/goyal2017_large_minibatch.txt` (already cached) | Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017 | https://arxiv.org/abs/1706.02677 | Linear scaling rule and its "k small steps ≈ one big step" argument (§2.1); gradual warmup over 5 epochs (§2.2); BN per-worker n=32 (§2.3); momentum correction, loss normalisation by kn (§3); zero-init last BN γ (§5) |
| `tp_bergstra2012_random_search.txt` | Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 2012 | https://www.jmlr.org/papers/v13/bergstra12a.html | Low effective dimensionality, grid vs random (§1, Fig. 1); GP-based evidence (§3); low-discrepancy (Sobol/Halton) vs grid vs random, $1-(1-v/V)^T$ (§4) |
| `tp_choi2019_optimizer_comparisons.txt` | Choi, Shallue, Nado, Lee, Maddison & Dahl, *On Empirical Comparisons of Optimizers for Deep Learning*, 2019 | https://arxiv.org/abs/1910.05446 | Inclusion hierarchy (SGD ⊆ Momentum ⊆ RMSProp/Adam), Adam → Momentum as $\epsilon\to\infty$ (App. A); tuning protocol decides rankings; search over $(\epsilon, \alpha_0/\epsilon)$ because ε and LR are coupled (§4) |
| `tp_gilmer2021_curvature_instability.txt` | Gilmer, Ghorbani, Garg et al., *A Loss Curvature Perspective on Training Instability in Deep Learning*, 2021 | https://arxiv.org/abs/2110.04369 | Divergence when $\lambda_1 > 2/\eta$; warmup lowers $\lambda_1$ gradually; warmup ≈ as good as BN/MetaInit for stability; pre- vs post-LN (§1, §3–4) |
| `tp_cohen2021_edge_of_stability.txt` | Cohen, Kaur, Li, Kolter & Talwalkar, *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021 | https://arxiv.org/abs/2103.00065 | Progressive sharpening and the $2/\eta$ threshold for full-batch GD (§1–3) |
| `tp_nado2021_large_batch_benchmark.txt` | Nado, Gilmer, Shallue, Anil & Dahl, *A Large Batch Optimizer Reality Check*, 2021 | https://arxiv.org/abs/2102.06356 | Well-tuned Nesterov/Adam match LARS/LAMB at large batch; tuning protocol matters |
| `tp_zhang2019_batch_size_nqm.txt` | Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue & Grosse, *Which Algorithmic Choices Matter at Which Batch Sizes? (NQM)*, NeurIPS 2019 | https://arxiv.org/abs/1907.04164 | Preconditioning (Adam, K-FAC) and momentum raise the critical batch size; noisy quadratic model |
| `tp_smith2018_dont_decay_lr.txt` | Smith, Kindermans, Ying & Le, *Don't Decay the Learning Rate, Increase the Batch Size*, ICLR 2018 | https://arxiv.org/abs/1711.00489 | SGD noise scale $g\approx\eta N/B$ (with momentum $\eta N/(B(1-m))$); decaying LR ≈ growing batch (§2–3) |
| `tp_loshchilov2017_sgdr.txt` | Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017 | https://arxiv.org/abs/1608.03983 | Cosine annealing formula (§3) |
| `tp_wu2018_short_horizon_bias.txt` | Wu, Ren, Liao & Grosse, *Understanding Short-Horizon Bias in Stochastic Meta-Optimization*, ICLR 2018 | https://arxiv.org/abs/1803.02021 | Why LR schedules tuned on short horizons decay too fast (§1, §3) |
| `tp_hoffer2017_ghost_bn.txt` | Hoffer, Hubara & Soudry, *Train Longer, Generalize Better*, NeurIPS 2017 | https://arxiv.org/abs/1705.08741 | Ghost Batch Norm (Alg. 1, §4); "generalization gap" is a too-few-updates effect (§5) |
| `tp_bachlechner2020_rezero.txt` | Bachlechner et al., *ReZero is All You Need*, 2020 | https://arxiv.org/abs/2003.04887 | Residual branches initialised to zero ($x + \alpha F(x)$, $\alpha=0$) |
| `tp_kaplan2020_scaling.txt` | Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 | https://arxiv.org/abs/2001.08361 | Critical batch size grows as loss falls, $B_{crit}(L)$ (§5.1) |

Other cached files these lessons use: `papers/ioffe2015_batchnorm.txt`, `papers/muller2019_label_smoothing.txt`, `papers/xiong2020_preln.txt`,
`papers/kingma2014_adam.txt`, `papers/loshchilov2017_adamw.txt`, `sys_wortsman2023_small_scale_instabilities.txt` (Adam ε vs gradient RMS §3.4,
qk-layernorm, z-loss), `sys_zhang2020_why_clipping.txt`, `sys_pytorch_clip_grad_norm.txt`, `llm_hoffmann2022_chinchilla.txt`.
