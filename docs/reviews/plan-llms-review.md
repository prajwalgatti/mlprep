# Review: content plan for LLMs (`docs/content-plan-llms.md`)

Reviewer: adversarial plan review, 2026-10-02. Standard: `docs/writing-brief.md`. I edited the plan directly. Each entry below is
tagged **blocking**, **major** or **minor** and marked **[changed]** or **[flagged]** (flagged = not changed in the plan).
Verification scripts are in the session scratchpad (`llmreview/check.py`).

Net structure after the review: **61 lessons** (it was 60). Wave 1 has 23, wave 2 has 30, wave 3 has 8. Every wave-1 lesson's hard prereqs
are now wave 1 or cross-area. A script checked that the topic table, the section headers (order, wave, ★) and every `llm.*` id referenced
in the file agree. The only references to a removed id are the deliberate "merged" notes for `llm.arithmetic-intensity`.

## 1. Structure, scoping and ordering

- **blocking — no RL area exists [changed].** The plan said `rl.*` (claimed to be "planned in `docs/content-plan.md`") owns the
  policy-gradient theorem, PPO and GAE, and that `llm.rlhf-ppo` only recaps them. No `rl.*` plan exists in any `docs/` file. `PLAN.md`
  lists RL as an area but has no plan for it. The only generic coverage is `fund.gradient-estimators` (wave 3). With 10 items already,
  `llm.rlhf-ppo` could not also teach PG, PPO and GAE from scratch in 6–10 cards.
  Fix: added **`llm.policy-gradients`** (order 475, wave 1). It covers the LM-as-policy setup, the REINFORCE derivation, why a baseline is
  unbiased, RLOO, reward-to-go, importance ratios and the surrogate, the PPO clip case analysis, and GAE.
  `llm.rlhf-ppo` was trimmed to the KL-regularized objective, per-token shaping, the four models, KL estimators, PPO-ptx, RLOO and
  practicalities, and now has prereqs `llm.reward-modeling` + `llm.policy-gradients`. The cross-area and Unit M notes were rewritten.
  If an RL area is planned later, it should link to this lesson, not duplicate it.
- **major — `llm.arithmetic-intensity` no longer earns a standalone lesson [changed: merged away].** Under the confirmed ownership rule,
  `sys.roofline` and `sys.gpu-basics` own items 1–4, 6 and 8 (regimes, intensity, ridge point, matmul intensity ≈ B, elementwise ops,
  memory hierarchy, overhead). What was left (decode vs prefill intensity, the 42 ms decode floor) already appeared almost verbatim in
  `llm.kv-cache` items 3–4. The transformer-specific parts moved as follows:
  - Into `llm.kv-cache`: new item 3 (decode matmul intensity ≈ B). New item 4 derives decode-attention intensity ≈ $h/h_{kv}$, shows it
    is independent of batch, and adds the MLA MQA-mode value ≈ 242 FLOPs/byte (Python-checked). Also a roofline figure, the
    $[16,8192]\times[8192,8192]$ question (≈16 FLOPs/byte) and the "batch raises decode-attention intensity" which-is-false.
  - Into `llm.flash-attention` item 1: the A100 SRAM/HBM numbers from FA §2.1, verified at pdf p.3 (192 KB × 108 SMs at ~19 TB/s;
    40–80 GB at 1.5–2.0 TB/s), and a worked 128 MiB-per-head $P$ for N = 8192.
  - The prereqs of `llm.flash-attention` and `llm.kv-cache` now point to `sys.roofline`. This also removed two wave-1 → wave-2
    prereq violations.
- **major — `llm.positional-encodings` would overflow [changed: split].** It had 11 items: learned, sinusoidal derivation, four-term
  expansion, Shaw, the Transformer-XL derivation, T5, ALiBi, NoPE and a comparison table. With the brief's probe and recap cards that is
  about 12–13 cards. Split into:
  - `llm.positional-encodings` (wave 1, keeps its id): why position matters, learned, sinusoidal, the four-term expansion, NoPE,
    cache compatibility.
  - New **`llm.relative-position`** (order 55, wave 2): Shaw, Transformer-XL $u$/$v$, T5 buckets, ALiBi, comparison table.
  - `llm.rope` stays on `llm.positional-encodings`, with `llm.relative-position` recommended. RoPE item 10 recaps ALiBi/T5 in one
    paragraph so it stands alone in wave 1.
  - Existing relative-PE items in the old YAML need new ids in the new file.
- **major — wave 1 depended on wave-2 lessons [changed].**
  - `llm.sampling` and `llm.scaling-laws` depended on `llm.pretraining-objective` (W2). Promoted pretraining-objective to wave 1.
  - `llm.lora` and `llm.reward-modeling` depended on `llm.sft` (W2). Promoted SFT to wave 1; it is also the backbone of the user's
    "finetuning" item.
  - `llm.rlvr-grpo` depended on `llm.chain-of-thought` (W2). Made it a recommended prereq.
  - `llm.flash-attention` and `llm.kv-cache` depended on arithmetic-intensity (W2). Fixed by the merge.
  - `llm.architecture-comparison` (W1) depended on the wave-2 chain `linear-attention` → `ssm` → `mamba`. **Moved to wave 2**, with a
    note to write unit L as a block at the start of wave 2. The basic transformer-vs-RNN rows are already in wave-1
    `llm.self-attention` item 7. *Coordinator call:* if the user wants "LLM vs RNN vs S4" in the first batch, promote `llm.ssm` and
    `llm.mamba` too (+2 wave-1 lessons) rather than writing a self-contained comparison that duplicates them.
- **minor — decoding order [changed].** `llm.decoding` said "⇒ sampling (next lesson)", but sampling came *before* it. Swapped the order
  numbers (decoding 270, sampling 280) and moved the sections to match. The pedagogy is better too: the failure of maximization
  motivates sampling. Decoding's prereq changed from `llm.causal-attention` to `llm.pretraining-objective`.
- **minor — stale pointer [changed].** `llm.long-context` item 1 said "position extrapolation (previous lesson)", but the previous lesson
  is `llm.mot-mod`. It now points to `llm.context-extension`.
- **minor — overflow watch [flagged].** These lessons are at the top of the 6–10 range once the probe and recap cards are counted:
  - `llm.self-attention` (now with the backward pass; trimmed numerics and additive attention to one line each)
  - `llm.kv-cache` (11 items after the merge, but items 1, 11 and some others are one-liners)
  - `llm.context-extension`, `llm.transformer-block`, `llm.tokenization`
  Writers should report overflow rather than compress.
- **minor — thin lesson [flagged].** `llm.long-context-eval` has almost no derivation core: NIAH/RULER definitions, lost-in-the-middle,
  recipes. It could merge into `llm.long-context` plus `llm.evaluation-pitfalls`, but that would overflow `llm.long-context`. Left as is.

## 2. Ownership rule applied (trims and links)

The rule is: `sys.*` owns the generic or hardware derivation; `llm.*` keeps the transformer-specific numbers and links back by id. It is
written into the cross-area section of the plan, with one bullet per overlap.

- **major — `llm.params-flops` [changed].**
  - Removed or linked: the $2mkn$ / backward = 2× / $6N$ derivation (now a one-card recap), MFU/HFU, the days-to-train rules of thumb and
    question (`sys.flops-mfu` already has 70B-on-15T), and the full memory item (now one numbers-only card, `sys.memory-anatomy`).
    Dropped PaLM App. B and Megatron as sources.
  - Kept or added: per-layer GQA/SwiGLU counts, contrast with `sys.memory-anatomy`'s $12Ld^2+Vd$, the Kaplan attention term, the
    crossover reconciliation, the Llama-3 numbers, and a vocab-share item (≈13% of Llama-3-8B; ≈31% for GPT-2-small shapes).
  - New questions: GQA doesn't cut score FLOPs; why Kaplan's attention term is $2n_{ctx}d$. New per-layer breakdown figure.
- **major — `llm.training-stability` [changed].** Items 3–5 (QK-norm, z-loss, soft-cap) became a recipe view that links to
  `sys.stability-tricks` items 1–3 for the numerics. Item 6 links to `sys.tuning-diagnostics`, `sys.gradient-clipping` and `fund.adam`.
  Item 8 is now a pointer. Replaced the z-loss and soft-cap figures, which duplicated the sys `softcap` figure, with an init-variance
  simulation and a PaLM rewind timeline. Replaced the z-loss derivation question with a $1/\sqrt{2L}$ derivation.
- **major — `llm.pretraining-optimization` [changed].** The $B_{simple}$ derivation, the steps–examples hyperbola (figure and question)
  and the noise-floor argument now link to `sys.tuning-batch-size` and `sys.tuning-steps-schedules`. Kept the LLM recipes: Kaplan's
  $B_{crit}(L)$ (as the tuning-plan insertion asks), DeepSeek-V3's 3072 → 15360 ramp over 469B tokens, Llama 3's 4M → 8M → 16M, WSD.
- **major — `llm.quantization-advanced` [changed].** E4M3/E5M2 and FP8-training recipes (Micikevicius, DeepSeek-V3 §3.3) now link to
  `sys.fp-formats` and `sys.fp8-training`. Kept the inference-side MXFP4: E2M1 elements and an E8M0 scale per 32 values give 4.25 bits,
  verified in Rouhani Table 1 and gpt-oss pdf p.5. Replaced the E4M3/E5M2 figure and question.
- **major — `llm.long-context` ring attention [changed].** The plan claimed to keep "the online-softmax merge and the overlap
  condition", but `sys.sequence-context-parallel` derives both. Reduced to one paragraph plus a transformer-specific worked number:
  GQA-8 cuts per-hop K/V traffic 8×, 256 MiB vs 2 GiB per layer per hop for Llama-3-70B at 1M tokens over 16 devices (Python-verified).
  Replaced the ring-schedule figure and the derivation question.
- **minor — genmodels/fundamentals links [changed].** `llm.cross-attention` item 3 now links `gen.latent-diffusion` and `gen.guidance`.
  `llm.distillation` item 5 links `fund.kl-divergence`.

## 3. Coverage vs real interviews

- **major — attention backward pass was wave-2 only [changed].** It is now `llm.self-attention` item 9 (wave 1): $dV$, $dP$,
  $dS=P\odot(dP-D)$, the $D_i=dO_i\cdot O_i$ identity, $dQ$, $dK$, with source FA App. B.2 (verified pdf p.18), plus a new question.
  `llm.flash-attention-2-3` item 1 now recaps it and explains why $D_i$ makes tiling possible.
- **major — GRPO objective given only in words [changed].** `llm.rlvr-grpo` item 2 now has the full objective: the $1/G$, $1/|o_i|$,
  clip and per-token k3 terms, verified in DeepSeekMath §4.1 (pdf p.11–14). The $1/|o_i|$ and $1/\mathrm{std}$ factors are flagged as
  the hook for Dr. GRPO.
- **major — load-balancing "minimum at uniform" was asserted, not argued [changed].** Switch §2.2 (pdf p.7) only asserts it. For fixed
  $f$, $\sum f_iP_i$ is minimized by a one-hot $P$. Added the Cauchy–Schwarz argument ($N\sum P_i^2\ge1$ with $f\approx P$) to
  `llm.moe-load-balancing` item 2.
- **major — Muon under-covered for 2026 [changed].** `llm.pretraining-optimization` item 7 now derives Muon as steepest descent under
  the spectral norm ⇒ $UV^\top$. Added the Newton–Schulz quintic $(3.4445,-4.7750,2.0315)$ (Jordan blog), the 2-D-only scope, and
  Liu 2025's weight-decay and RMS-matching fixes. New singular-value figure and derivation question.
- **minor — vocab-size trade-off [changed].** Made it quantitative in `llm.tokenization` item 9: Llama 2 → 3 gives ≈20% fewer tokens for
  ≈0.79B extra embedding+head parameters at d = 4096 (Python-checked).
- **Checked and adequate (no change):** KV-cache formula; speculative acceptance proof and expected tokens; the DPO derivation chain
  (Gibbs form verified correct); Chinchilla allocation; the RoPE relative proof; the BPE algorithm with a worked trace; Bradley–Terry;
  GAE; online softmax; NTK-aware base change (formula matches YaRN App. A.2).
- **minor — not added [flagged]:**
  - Agents/tool use and multi-turn agentic RL. Excluded by the plan, but increasingly asked at 2026 frontier labs. Suggest an optional
    wave-3 lesson later; Lambert ch. 13 is cached.
  - The softmax bottleneck (LM-head rank limit). Not cached.
  - The subtlety that differentiating the k3 estimator in the loss does not give an unbiased KL gradient. Not found in the cache, so not
    added.

## 4. Sources and recency (spot-checks: about 45 claims grepped; all supported unless noted)

**Recent items (2025–26), all verified in the cached text:**
- **DeepSeek V4:**
  - It is a *preview* (arXiv 2606.19348v1, 26 Apr 2026). Pro is 1.6T/49B. Flash is 13B active, with **284B** total in the paper (p.1)
    but **285B** in the model card (p.3). The plan now says to cite which. **[changed, minor]**
  - CSA/HCA mechanics are on pdf p.9–12. Core attention is **shared-KV MQA, not MLA**. The plan previously implied an MLA lineage;
    `llm.mqa-gqa-mla` and `llm.sparse-attention` now note this. **[changed, major]**
  - At 1M tokens: 27% of V3.2's per-token FLOPs and 10% of its KV cache (p.1, 5).
  - mHC with $n_{hc}=4$ and 20 Sinkhorn iterations (p.8, 24); Muon; MTP kept.
- **DeepSeek-V3.2 DSA:** FP8 lightning indexer with ReLU, top-**2048** tokens, 1000-step dense warm-up, built on MLA in MQA mode
  (p.3–4, 20). Added to the plan. **[changed]**
- **mHC:** Xie et al. arXiv 2512.24880 (Birkhoff polytope, Sinkhorn). Item text expanded.
- **gpt-oss:** 128-token banded windows alternating with dense layers; learned sink in the softmax denominator; 128/32 experts, top-4;
  MXFP4 at 4.25 bits (p.5).
- **Qwen3:** 235B-A22B; 128 experts, 8 active, no shared experts; QK-Norm; QKV-bias removed (p.3).
- **Kimi:**
  - K2: 1.04T/32B, 384 experts top-8, 64 heads; MuonClip/QK-Clip. QK-Norm is inapplicable to MLA because keys aren't materialized;
    the clip applies only to unshared head components (p.3, 6–7). Added to the plan. **[changed]**
  - Kimi Linear: 3:1 KDA:MLA (p.2, 8). The plan said "check ratio"; now filled in. **[changed]**
- **Raschka 2026:** MLA in Kimi K2.5, GLM-5, Ling 2.5 and Sarvam 105B (Sarvam 30B uses GQA); Qwen3.5 397B-A17B; Qwen3-Next 3:1
  Gated DeltaNet : gated attention; the "data and recipe matter more than architecture" conclusion. All verified.
- **Gemma 2/3:** soft-cap 50/30 (Gemma 2 p.2); Gemma 3 5:1 local:global, 1024 window, QK-norm replacing soft-capping (p.1–2).
- **RL fixes:**
  - DAPO: four techniques; ε_low 0.2, ε_high 0.28 (text line ~804); KL removed.
  - Dr. GRPO: length bias and difficulty bias from $1/|o_i|$ and $1/\mathrm{std}$ (p.6).
  - GSPO: sequence-level, length-normalized ratio; MoE stability; GRPO needs Routing Replay (p.3–6).
  - Yue 2025: base models win pass@k at large k. Correctly flagged as contested.
- **HRM/TRM:** HRM 27M params with ~1000 examples; TRM 7M with 2 layers, 45% ARC-AGI-1 and 8% ARC-AGI-2 (TRM p.1). Added with a note that
  these are per-task models, not LMs, and that the gains are contested. **[changed, minor]**
- **Geiping:** log-normal Poisson loop sampling verified (Fig. 3, p.4); the plan's "verify" replaced. 3.5B params, 800B tokens.
- **R1:**
  - ~600k reasoning + ~200k non-reasoning ≈ 800k SFT samples (App. B.3.3).
  - The unsuccessful MCTS/PRM attempts are App. G.2 (pdf p.63) in the cached revised version. The plan's "check which version" was
    resolved. **[changed]**
  - The eval sampling setting (T = 0.6, top-p = 0.95, pdf p.40) and DAPO's (T = 1.0, top-p = 0.7, p.8) replace the "cite only if a
    source states it" placeholder in `llm.sampling`. **[changed]**

**Classic items, verified:**
- Vaswani footnote on $\mathrm{Var}(q\cdot k)=d_k$ (p.4).
- Chinchilla constants (p.25) and Besiroglu's refit (p.2).
- Llama 3: 128,000 vocab in Table 3; 3.17 → 3.94 chars/token; RoPE base 500k; 3.8e25 FLOPs; 15.6T tokens; batch schedule (p.14); six
  long-context stages and ~800B tokens (p.14).
- Leviathan's formulas and Theorem 3.11; Switch α = 10⁻²; ST-MoE c_z = 10⁻³.
- InstructGPT: β = 0.02 and γ = 27.8 (p.42), 16 SFT epochs, 72.6% agreement.
- QLoRA 0.373 bits; DeepSeek-V3 d_c = 512, d_c′ = 1536, d_h^R = 64, bias γ = 0.001, 3072 → 15360 ramp, 2K warmup steps.
- PaLM 46.2% MFU (p.66) and its rewind/skip (p.11).
- YaRN $0.1\ln s+1$ and α = 1, β = 32; PI ~600× and 1000 steps; Mixtral 47B/13B; Mistral W = 4096 ≈ 131K; DeepSeekMoE 120 vs 4.4e9
  combinations; Griffin c = 8, 2 recurrent : 1 local-MQA, 1024 window; Jamba 1:7.
- ALiBi slopes; T5 buckets to 128; Mamba Theorem 1; vLLM 20.4–38.2% and 2–4×; min-p; RULER "half at 32K"; MT-Bench >80%.
- MoT 55.8/37.2/47.2%; MoD 12.5%; Snell 4× and 14×; aux-free $b_i\leftarrow b_i+u\,\mathrm{sign}(e_i)$; MXFP4 E2M1/E8M0/32.
- K2's MLA-only clipping; Muon's Newton–Schulz coefficients.
- Every `llm_*`, `papers/` and `d2l/` file cited in the plan exists in the cache.

## 5. Correctness (recomputed in Python)

- **major — GB vs GiB in the KV memory budget [changed].** `llm.kv-cache` item 6 said "80 GB GPU − 16 GB weights → 64 GiB → 524,288
  tokens → 64 sequences at 8k". But 64 GB is 59.6 GiB, which gives ≈488k tokens, i.e. **59** sequences. The same error was in the
  `llm.mqa-gqa-mla` question. Both are fixed, and the item now teaches the trap. The brief explicitly lists GB/GiB consistency as a
  common critic finding.
- **major — mislabelled attention FLOPs [changed].** `llm.params-flops` gave "≈10.7 GFLOPs/token at full 8k". $2Ln_{ctx}hd_h$ is the
  *causal average* over an 8k sequence; the token at position 8k costs ≈21.5. Relabelled.
- **major — "recheck the crossover constant" left open [changed: reconciled].** Non-gated 4d MLP: n = 6d non-causal, 12d
  causal-averaged (= Kaplan's $d_{model}>n_{ctx}/12$, p.7). The Scaling Book's T > 8D assumes a gated MLP with F = 4D. Written into the
  `llm.self-attention` pitfall and `llm.params-flops` item 4.
- **major — MCQ with a true "which is false" stem [changed].** `llm.mqa-gqa-mla`'s "MLA caches one latent vector per token per layer
  plus a small RoPE key" is true. It was replaced by a genuinely false statement about absorption surviving RoPE.
- **minor — PPO/DAPO clipping described as a hard cap [changed].** "Ratio cap 1.2 ⇒ 0.012" and "max reachable probability" were
  reworded: clipping removes the gradient incentive beyond $1+\epsilon$, but a step can still overshoot.
- **Recomputed and correct (no change):**
  - Llama-3-70B: 151.0M attention, 704.6M FFN, 855.7M per layer, 68.45B for the layers, 70.55B total.
  - KV: 320 KiB/token; 40 GiB at 128k; 80 GiB for 32×8k; 10 GiB at 32k; MHA 2.5 MiB/token; 8B at 128 KiB/token.
  - MLA: 576 vs 32,768 elements (57×), 68.6 KiB/token, equivalent to GQA with 2.25 groups.
  - Compute and memory: 405B → 3.79e25 FLOPs; decode floor 41.8 ms vs 0.14 ms (~295×); 70B full fine-tune 1.12 TB; 13B 208 GB.
  - Chinchilla: published constants give 32.2B / 2.98T (92.6 tokens/param); Besiroglu's give 72.2B / 1.33T (18.4); the 20× rule at
    1e24 gives 91.3B / 1.83T.
  - RoPE: λ_min 5.44e4; NTK-aware base at s = 4 is 4.09e4.
  - Temperature table; speculative decoding E = 1.8 / 2.44 / 3.36 / 4.33 and speedups 1.71 / 2.22 / 2.80 / 3.09; β = 0.9 with residual
    (0, 1, 0).
  - LoRA: 131,072 params (0.78%). MoE: Mixtral 46.7B / 12.9B; distinct experts ≈ 163; all-to-all 0.94 GB; Switch loss example 1.14.
  - GRPO advantages +1.73 / −0.58 (sample-std variant 1.62 / −0.54).
  - pass@k 0.917 and 0.6; majority-of-5 0.683; SE ±1.45 pts; BT 0.818; DPO loss 0.554; BPB 0.79; LSH 0.9996; exact match 0.349 / 0.599.
  - NF4 70B ≈ 36 GB; int4-g64 4.25 bits; Perceiver 98× fewer score entries; copy bits 15.6k vs 262k.

## 6. Pedagogy

- **major — inline refreshers not identified [changed].** Added an "Inline refreshers" section listing, per lesson, the prerequisites a
  newcomer may lack, with links to `fund.*` and `sys.*` lessons where they exist. Examples: the softmax Jacobian and matrix calculus;
  Euler/rotations; Lagrange substitution; rejection sampling and TV distance; the log-derivative trick; the Gibbs variational principle;
  the matrix exponential and FFT; associative scans; OBS's second-order expansion.
- **minor [changed].** The new and trimmed lessons each start from the problem. `llm.policy-gradients` opens with "why we can't backprop
  through r". The trimmed `llm.positional-encodings` ends on the four-term expansion as the question the next two lessons answer.
- **Checked:** most other syllabi already open with the problem and build in order. `llm.mup`, `llm.ssm` and `llm.dpo` are well
  sequenced.

## 7. Questions and figures

- Fixed the false-stem MCQ (§5). Replaced the figures that duplicated `sys.*` figures: soft-cap, z-loss, the McCandlish hyperbola,
  E4M3/E5M2, and the ring schedule. Moved the PPO clip figure to `llm.policy-gradients`.
- **minor [flagged].** `llm.in-context-learning`'s "Which composition type does the induction head's key use?" is pure recall; convert
  it to a predict or ablation question when writing.
- **minor [flagged].** Several figures are labelled "schematic". That is acceptable, but writers should prefer generating curves from
  the stated functional forms (Gao, Schaeffer, Muennighoff), as the plan already suggests.
