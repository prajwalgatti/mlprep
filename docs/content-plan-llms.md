# Content plan: Large Language Models (area `llms`, prefix `llm.`)

Planned 2026-10-02 for the study app. Read it with [writing-brief.md](writing-brief.md) (how to write) and
[CONTENT_GUIDE.md](../CONTENT_GUIDE.md) (file format). The general-ML and generative-modelling plan is in
[content-plan.md](content-plan.md), and the systems plan is in [content-plan-applied.md](content-plan-applied.md).

**Shape of the plan.** 61 short lessons in 15 units. Each lesson is about 6–10 explainer cards, so a big area becomes
an ordered sequence linked by `prereqs`. That follows the brief's rule: depth comes from the sequence, not from bloating one lesson.
**Wave 1** (★, 23 lessons) holds the most interview-critical material; write it first. Every wave-1 lesson's hard prereqs
are themselves wave 1 (or a cross-area `sys.*`/`fund.*` lesson), so wave 1 can be studied before wave 2 exists.
Wave 2 is the core intermediate material (30 lessons). Wave 3 (8 lessons) is advanced or optional.
Revised 2026-10-02 after an adversarial review (log: [reviews/plan-llms-review.md](reviews/plan-llms-review.md)): `llm.arithmetic-intensity`
was merged into `llm.kv-cache` and `llm.flash-attention`; `llm.relative-position` was split out of `llm.positional-encodings`;
`llm.policy-gradients` was added because no RL-area plan exists; overlaps with `sys.*` were trimmed under the ownership rule below.

**Each lesson below lists:**
- **Primary sources**, with exact sections and PDF pages, canonical URLs, and cache filenames.
- **Subtopic map (must cover)**: the full syllabus. It covers every derivation, variant, design choice and failure mode
  an interviewer could reasonably probe, and what the reader must understand for each. Writers should treat it as a checklist.
  If a lesson overflows 10 cards, tell the coordinator so it can be split. Don't compress derivations to make it fit.
- **Figures**: 2–4 ideas, each saying what to notice.
- **Question ideas**, including figure-based MCQs.
- **Pitfalls / checks**: contested points, and numbers to verify against the primary source before writing.
  Numbers marked *Python-verified* were recomputed during planning. Writers must still recompute anything they use.

**Source cache.** All files live in the session scratchpad at `…/scratchpad/sources/`. The index is `INDEX-llm.md`.
LLM files are prefixed `llm_`: text in `llm_*.txt`, PDFs in `llm_pdf/`, raw HTML in `llm_html/`, and auto-extracted
section outlines in `llm_toc/` (these are noisy but useful for finding a section's page). Page markers `=== [name p.N] ===`
are **PDF page numbers**, not printed page numbers. Cite the section number plus "pdf p.N".
Files from the other curator's cache are cited by their own paths, for example `papers/xiong2020_preln.txt`.

**Existing files.** `llm.self-attention` (order 10) and `llm.positional-encodings` (order 20 → **50**) keep their ids.
The rewrites narrow their scope. Causal masking, the KV cache and MQA/GQA move to `llm.causal-attention` and `llm.mqa-gqa-mla`.
Relative encodings (Shaw, Transformer-XL, T5 buckets, ALiBi) move to `llm.relative-position`; RoPE and context extension move to
`llm.rope` and `llm.context-extension`. Any existing item whose meaning moves gets a new id in the new file
(see "IDs and revisions" in the brief).

**Cross-area boundaries.**
- **RL.** `PLAN.md` lists an RL area, but no `rl.*` plan exists yet (checked `content-plan.md`, `content-plan-applied.md`, `content-plan-tuning.md`
  on 2026-10-02). The only generic coverage is `fund.gradient-estimators` (wave 3: score-function/REINFORCE vs reparameterization).
  So this plan teaches what RLHF needs itself: `llm.policy-gradients` (token-level MDP, REINFORCE with baselines, importance ratios, PPO's clipped
  surrogate, GAE) in the LLM setting. If an `rl.*` area is planned later, its generic PPO/GAE lesson should link to this one (or replace items 3–7
  of it), not duplicate it.
- The applied/systems area owns data, tensor, pipeline, expert and sequence parallelism, collectives, mixed precision,
  checkpointing and profiling. LLM lessons link to it and keep only transformer-specific arithmetic.
- The vision area (listed in `PLAN.md`, no plan yet) owns ViT, CLIP, DiT and multimodal tokenizers. `llm.cross-attention` covers only the conditioning mechanism.
- The genmodels area owns diffusion (`gen.latent-diffusion` owns cross-attention conditioning inside the U-Net; `gen.guidance` owns CFG), and
  `fund.kl-divergence` owns forward vs reverse KL. LLM lessons link to them. Discrete diffusion LMs are deferred (see "Excluded").
- **Ownership rule for overlaps with the systems plans (confirmed by the coordinator; applied in this revision).** The `sys.*` lesson owns the
  generic or hardware derivation; the `llm.*` lesson keeps only transformer-specific worked numbers and links back by id. Applied as follows:
  - `sys.roofline`, `sys.gpu-basics` own rooflines, arithmetic intensity, the ridge point, the memory hierarchy and the matmul-intensity ≈ batch
    derivation. `llm.arithmetic-intensity` was therefore **merged away**: its transformer-specific parts (decode matmul vs decode attention
    intensity, why batching doesn't help attention over the KV cache, the 70B decode floor) went to `llm.kv-cache`, and the SRAM/HBM
    motivation for fusion went to `llm.flash-attention`.
  - `sys.flops-mfu` owns $2mkn$, backward = 2× forward, $6N$, MFU/HFU and training-time estimates; `sys.memory-anatomy` and
    `sys.activation-checkpointing` own bytes/param, activation memory and recompute. `llm.params-flops` keeps per-layer counts for GQA/SwiGLU
    blocks, the attention-FLOPs term and its conventions, and the Llama-3 worked numbers.
  - `sys.stability-tricks` owns the numerics of z-loss, soft-capping, QK-norm and Adam ε (derivations, gradients, the soft-cap figure);
    `sys.mixed-precision`, `sys.fp-formats` own precision; `sys.gradient-clipping`, `sys.tuning-diagnostics` own clipping and warmup protocol.
    `llm.training-stability` keeps the LLM-recipe view (which failure, which model adopted which fix, init, data spikes, MuonClip, mHC).
  - `sys.fp-formats`, `sys.fp8-training` own E4M3/E5M2, scaling recipes and FP8 *training* (incl. DeepSeek-V3's tile scaling).
    `llm.quantization-advanced` keeps inference-side block formats (NF4, MXFP4 weights in gpt-oss) and KV-cache quantization.
  - `sys.sequence-context-parallel` owns sequence/context parallelism, ring attention, the online-softmax merge across shards, the overlap
    condition and zig-zag balancing. `llm.long-context` keeps one summary paragraph plus a transformer-specific number (GQA shrinks the K/V
    traffic) and links.
  - `sys.tuning-batch-size`, `sys.tuning-steps-schedules` own the gradient-noise-scale derivation, the steps–examples trade-off, the noise-floor
    argument for decay, and schedule families. `llm.pretraining-optimization` keeps published LLM recipes (β2 = 0.95, WSD/cooldowns, batch ramps,
    DeepSeek LLM's LR/batch power laws, Muon).
- The Applied plan hands two subjects to this plan: MoE systems → `llm.moe-systems` (with `llm.moe-load-balancing` for capacity/dropping)
  and quantization → `llm.quantization` plus `llm.quantization-advanced`. The Scaling Book series (`sb.*`) overlaps on inference and transformer math.
  That's acceptable, but LLM lessons stay framework-neutral.

## Topic table

| order | id | title | level | wave | prereqs |
|---|---|---|---|---|---|
| | **Unit A** | | | | |
| 10 | `llm.self-attention` ★ | Scaled dot-product & multi-head attention | core | 1 | — |
| 20 | `llm.causal-attention` ★ | Causal attention, teacher forcing & the KV cache | core | 1 | `llm.self-attention` |
| 30 | `llm.lm-objectives` | Decoder-only, encoder-only and encoder–decoder models | intermediate | 2 | `llm.causal-attention` |
| 40 | `llm.cross-attention` | Cross-attention, encoder–decoder conditioning & Perceiver | intermediate | 2 | `llm.self-attention`, `llm.lm-objectives` |
| | **Unit B** | | | | |
| 50 | `llm.positional-encodings` ★ | Why position matters; learned and sinusoidal absolute encodings | core | 1 | `llm.self-attention` |
| 55 | `llm.relative-position` | Relative position: Shaw, Transformer-XL, T5 buckets, ALiBi | intermediate | 2 | `llm.positional-encodings` |
| 60 | `llm.rope` ★ | Rotary position embedding (RoPE) | core | 1 | `llm.positional-encodings` (recommended: `llm.relative-position`) |
| 70 | `llm.context-extension` | Extending RoPE context: PI, NTK-aware, YaRN, ABF | intermediate | 2 | `llm.rope` |
| | **Unit C** | | | | |
| 80 | `llm.transformer-block` ★ | The modern decoder block | core | 1 | `llm.self-attention` |
| 90 | `llm.params-flops` ★ | Counting parameters and FLOPs for real transformer configs | core | 1 | `llm.transformer-block` (cross-area: `sys.flops-mfu`, `sys.memory-anatomy`) |
| 100 | `llm.training-stability` | Initialization and training instabilities | intermediate | 2 | `llm.transformer-block`, `llm.params-flops` |
| | **Unit D** | | | | |
| 110 | `llm.tokenization` ★ | Subword tokenization: BPE, WordPiece, Unigram | core | 1 | — |
| 120 | `llm.tokenization-effects` | What tokenization does to models (and tokenizer-free models) | intermediate | 2 | `llm.tokenization`, `llm.pretraining-objective` |
| | **Unit E** | | | | |
| 130 | `llm.pretraining-objective` ★ | Next-token prediction, perplexity and compression | core | 1 | `llm.causal-attention` |
| 140 | `llm.pretraining-data` | Pretraining data: filtering, deduplication, mixtures | intermediate | 2 | `llm.pretraining-objective` |
| 150 | `llm.pretraining-optimization` | Optimizers, LR schedules and batch size for LLMs | intermediate | 2 | `llm.pretraining-objective` (cross-area: `sys.tuning-batch-size`, `sys.tuning-steps-schedules`) |
| | **Unit F** | | | | |
| 160 | `llm.scaling-laws` ★ | Scaling laws and compute-optimal training | core | 1 | `llm.params-flops`, `llm.pretraining-objective` |
| 170 | `llm.scaling-laws-practice` | Scaling laws in practice: Kaplan vs Chinchilla, over-training, data limits, emergence | intermediate | 2 | `llm.scaling-laws` |
| 180 | `llm.mup` | μP and hyperparameter transfer across width | advanced | 3 | `llm.scaling-laws`, `llm.pretraining-optimization` |
| | **Unit G** | | | | |
| 190 | `llm.mqa-gqa-mla` ★ | Shrinking the KV cache: MQA, GQA and MLA | core | 1 | `llm.causal-attention`, `llm.params-flops` (restates the KV formula; `llm.kv-cache` deepens it) |
| 200 | `llm.sparse-attention` | Local, strided and learned sparse attention | intermediate | 2 | `llm.self-attention`, `llm.causal-attention` |
| 210 | `llm.attention-sinks` | Attention sinks, massive activations and softmax variants | advanced | 3 | `llm.sparse-attention`, `llm.training-stability` |
| | **Unit H** | | | | |
| 230 | `llm.flash-attention` ★ | FlashAttention: online softmax, tiling and IO complexity | core | 1 | `llm.self-attention` (cross-area: `sys.roofline`, `sys.gpu-basics`) |
| 240 | `llm.flash-attention-2-3` | Tiled backward pass, FA2, FA3 and decode kernels | intermediate | 2 | `llm.flash-attention` |
| | **Unit I** | | | | |
| 250 | `llm.kv-cache` ★ | KV-cache arithmetic, prefill vs decode, and latency | core | 1 | `llm.causal-attention`, `llm.params-flops` (cross-area: `sys.roofline`) |
| 260 | `llm.serving-systems` | Serving: continuous batching, PagedAttention, prefix caching | intermediate | 2 | `llm.kv-cache` |
| 270 | `llm.decoding` | Search-based decoding: greedy, beam search and its failure modes | intermediate | 2 | `llm.pretraining-objective` |
| 280 | `llm.sampling` ★ | Sampling: temperature, top-k, top-p, min-p and penalties | core | 1 | `llm.pretraining-objective` (recommended: `llm.decoding`) |
| 290 | `llm.speculative-decoding` ★ | Speculative decoding | intermediate | 1 | `llm.kv-cache`, `llm.sampling` |
| 300 | `llm.quantization` | Quantization fundamentals and LLM outliers | intermediate | 2 | `llm.kv-cache` |
| 310 | `llm.quantization-advanced` | GPTQ, NF4, KV-cache quantization and low-precision formats | advanced | 3 | `llm.quantization` |
| | **Unit J** | | | | |
| 320 | `llm.moe` ★ | MoE fundamentals: gating, sparsity and active parameters | core | 1 | `llm.transformer-block`, `llm.params-flops` |
| 330 | `llm.moe-load-balancing` ★ | Load balancing, capacity and router stability | intermediate | 1 | `llm.moe` |
| 340 | `llm.moe-systems` | Fine-grained & shared experts, expert parallelism and MoE inference | intermediate | 2 | `llm.moe-load-balancing`, `llm.kv-cache` |
| 350 | `llm.mot-mod` | Beyond token-level MoE: Mixture-of-Transformers and Mixture-of-Depths | advanced | 2 | `llm.moe` |
| | **Unit K** | | | | |
| 360 | `llm.long-context` | Long-context architectures: segment recurrence, ring attention, KV compression | intermediate | 2 | `llm.context-extension`, `llm.kv-cache`, `llm.flash-attention` |
| 370 | `llm.long-context-eval` | Does the model use its context? Evaluation and training for long context | intermediate | 2 | `llm.long-context` |
| | **Unit L** | | | | |
| 380 | `llm.linear-attention` | Linear attention: the kernel view and attention as an RNN | intermediate | 2 | `llm.self-attention`, `llm.causal-attention` |
| 390 | `llm.ssm` | State space models: S4 from continuous dynamics to convolution | intermediate | 2 | `llm.linear-attention` |
| 400 | `llm.mamba` | Mamba: selective state spaces and the SSD duality | intermediate | 2 | `llm.ssm` |
| 410 | `llm.modern-rnns-hybrids` | Modern RNNs and hybrids: RWKV, RetNet, Griffin, Gated DeltaNet, Jamba | intermediate | 2 | `llm.mamba` |
| 420 | `llm.architecture-comparison` | Transformers vs RNNs vs SSMs: a head-to-head | intermediate | 2 (write unit L first in wave 2) | `llm.mamba`, `llm.kv-cache` (recommended: `llm.modern-rnns-hybrids`) |
| 430 | `llm.looped-transformers` | Universal, looped and recurrent-depth transformers | advanced | 3 | `llm.architecture-comparison` (recommended after `llm.chain-of-thought`) |
| | **Unit M** | | | | |
| 440 | `llm.sft` ★ | Supervised fine-tuning and instruction tuning | core | 1 | `llm.pretraining-objective` |
| 450 | `llm.lora` ★ | LoRA: low-rank adaptation | core | 1 | `llm.sft` |
| 460 | `llm.qlora-peft` | QLoRA and the PEFT zoo | intermediate | 2 | `llm.lora`, `llm.quantization` |
| 470 | `llm.reward-modeling` ★ | Reward models: Bradley–Terry, preference data, overoptimization | core | 1 | `llm.sft` |
| 475 | `llm.policy-gradients` ★ | Policy gradients for LMs: REINFORCE, baselines, PPO and GAE | core | 1 | `llm.pretraining-objective` (recommended: `fund.gradient-estimators`) |
| 480 | `llm.rlhf-ppo` ★ | RLHF with PPO: KL-regularized objective, four models, practicalities | core | 1 | `llm.reward-modeling`, `llm.policy-gradients` |
| 490 | `llm.dpo` ★ | Direct Preference Optimization | core | 1 | `llm.rlhf-ppo` |
| 500 | `llm.preference-variants` | IPO, KTO, ORPO, SimPO and online vs offline preference learning | intermediate | 2 | `llm.dpo` |
| | **Unit N** | | | | |
| 510 | `llm.chain-of-thought` | Chain-of-thought and self-consistency | core | 2 | `llm.sampling` |
| 520 | `llm.test-time-compute` | Scaling test-time compute: best-of-N, verifiers, PRMs and search | intermediate | 2 | `llm.chain-of-thought`, `llm.reward-modeling` |
| 530 | `llm.rlvr-grpo` ★ | RL with verifiable rewards: GRPO and the R1 recipe | core | 1 | `llm.rlhf-ppo` (recommended: `llm.chain-of-thought`) |
| 540 | `llm.rlvr-practice` | GRPO in practice: biases, fixes and open questions | advanced | 2 | `llm.rlvr-grpo` |
| | **Unit O** | | | | |
| 550 | `llm.distillation` | Knowledge distillation for LLMs | intermediate | 2 | `llm.pretraining-objective`, `llm.sft` |
| 560 | `llm.evaluation-metrics` | Evaluating LLMs: metrics, benchmarks and statistics | intermediate | 2 | `llm.pretraining-objective` |
| 570 | `llm.evaluation-pitfalls` | Contamination, LLM judges and benchmark validity | intermediate | 3 | `llm.evaluation-metrics` |
| 580 | `llm.in-context-learning` | In-context learning and induction heads | advanced | 3 | `llm.transformer-block`, `llm.self-attention` |
| 590 | `llm.mech-interp` | Mechanistic interpretability basics | advanced | 3 | `llm.in-context-learning` |
| 600 | `llm.rag` | Retrieval-augmented generation | intermediate | 3 | `llm.cross-attention`, `llm.long-context-eval` |

## Coverage of the user's required items

| Required item | Where it is covered |
|---|---|
| Flash Attention | `llm.flash-attention` (online softmax, tiling, IO complexity), `llm.flash-attention-2-3` (tiled backward pass, FA2, FA3, decode kernels); attention backward derived in `llm.self-attention`; roofline background in `sys.roofline` |
| LoRA | `llm.lora`; QLoRA and other PEFT methods in `llm.qlora-peft` |
| Transformer-XL | four-term expansion of the absolute-PE logit in `llm.positional-encodings`; the relative form ($u$/$v$ biases, $W_{k,R}$) in `llm.relative-position`; segment-level recurrence in `llm.long-context` |
| Griffin (Hawk/Griffin, RG-LRU + local attention) | `llm.modern-rnns-hybrids`; compared in `llm.architecture-comparison` |
| Perceiver / Perceiver IO | `llm.cross-attention` (latent bottleneck, $O(MN)$ cost, output queries); Flamingo's Perceiver Resampler too |
| Scaling laws | `llm.scaling-laws` (Kaplan, Chinchilla derivation), `llm.scaling-laws-practice` (Kaplan-vs-Chinchilla resolution, over-training, data limits, emergence); MoE scaling in `llm.moe-systems`; test-time scaling in `llm.test-time-compute`; RM overoptimization laws in `llm.reward-modeling` |
| Mixture of Experts | `llm.moe`, `llm.moe-load-balancing`, `llm.moe-systems`, `llm.mot-mod` (MoT and MoD) |
| "LLM scaling factor" (ambiguous) | Covered under each reasonable reading: (1) the attention $1/\sqrt{d_k}$ factor, derived in `llm.self-attention`; (2) RoPE/context-extension scale factor $s$, the NTK-aware base $b'=b\,s^{d/(d-2)}$ and YaRN's temperature in `llm.context-extension`; (3) μP width multipliers and the $1/d$ attention scale in `llm.mup`. The related scale factors LoRA $\alpha/r$ (`llm.lora`) and residual init $1/\sqrt{2L}$ (`llm.training-stability`) are covered too. Ask the user which reading they meant; all are covered regardless |
| RoPE | `llm.rope` (derivation, frequencies, implementation); `llm.context-extension`; decoupled RoPE in `llm.mqa-gqa-mla` |
| Sinusoidal embeddings | `llm.positional-encodings` (derivation of the shift-as-rotation property) |
| Relative positional embeddings (Shaw, T5 buckets, Transformer-XL) | `llm.relative-position` (Shaw, Transformer-XL, T5 buckets, ALiBi, comparison table); RoPE in `llm.rope` |
| LLM vs RNN vs S4 | `llm.architecture-comparison` (dedicated lesson, wave 2 because it needs the unit-L chain); mechanics in `llm.linear-attention`, `llm.ssm` (S4), `llm.mamba`, `llm.modern-rnns-hybrids`. The basic transformer-vs-RNN rows (parallelism, path length, Vaswani Table 1) are already in wave-1 `llm.self-attention` item 7 |
| Tokenisation | `llm.tokenization`, `llm.tokenization-effects` |
| Pretraining | `llm.pretraining-objective`, `llm.pretraining-data`, `llm.pretraining-optimization`, `llm.training-stability` |
| Finetuning | `llm.sft`, `llm.lora`, `llm.qlora-peft`; `llm.distillation` |
| RLHF | `llm.reward-modeling`, `llm.policy-gradients`, `llm.rlhf-ppo`, `llm.dpo`, `llm.preference-variants`; reasoning RL in `llm.rlvr-grpo` and `llm.rlvr-practice` |
| Decoding techniques | `llm.decoding` (greedy, beam, constrained), `llm.sampling`, `llm.speculative-decoding` |
| Causal attention (masking, train/inference asymmetry, KV caching) | `llm.causal-attention`; KV arithmetic in `llm.kv-cache` |
| Cross attention | `llm.cross-attention` |
| Original draft extras: MoT, looped transformers, test-time compute, quantization, long context, evaluation | `llm.mot-mod`, `llm.looped-transformers`, `llm.test-time-compute`, `llm.quantization(-advanced)`, `llm.long-context(-eval)`, `llm.evaluation-metrics/-pitfalls` |
| Suggested extras: multi-token prediction, distillation, RAG, ICL theory, mech interp | MTP in `llm.pretraining-objective` and `llm.speculative-decoding`; `llm.distillation`; `llm.rag`; `llm.in-context-learning`; `llm.mech-interp` |

## Recent developments checked (2024 to Oct 2026)

These sources were read or skimmed during planning, and the plan places each item as follows.
Consensus items are taught as standard. Recent items are marked as such in the lessons.
- **Attention/KV efficiency.** MLA (DeepSeek-V2/V3) is now adopted beyond DeepSeek (Kimi K2/K2.5, GLM-5 and Sarvam 105B, per Raschka 2026).
  GQA remains the default. Sliding-window/global interleaving is mainstream (Gemma 3 at 5:1 with a 1024-token window; gpt-oss alternates 128-token banded and dense layers).
  Learned sparse attention: NSA (2025), DeepSeek Sparse Attention in V3.2 (Dec 2025; an FP8 lightning indexer, still $O(L^2)$ but cheap, picks the
  top-2048 KV tokens per query; instantiated on MLA in its MQA mode — V3.2 §2.1, pdf p.3–4) and
  **DeepSeek V4** (a *preview* release, 26 Apr 2026, arXiv 2606.19348), which interleaves compressed sparse attention (CSA: compress every $m$ tokens' KV into
  one entry, then DSA top-k selection) and heavily compressed attention (HCA: compress every $m'\gg m$ tokens, attend densely), each with a sliding-window
  branch, and runs the core attention as **shared-KV MQA**, not MLA (V4 §2.3, pdf p.9–12). At 1M tokens V4-Pro needs 27% of V3.2's per-token FLOPs and 10% of
  its KV cache (pdf p.1, p.5). V4 also has **mHC** residual connections (expansion $n_{hc}=4$, 20 Sinkhorn–Knopp iterations) and the **Muon** optimizer, keeps MTP,
  and comes as V4-Pro (1.6T total / 49B active) and V4-Flash (13B active; 284B total in the paper, 285B in the model card — cite which). → `llm.mqa-gqa-mla`, `llm.sparse-attention`, `llm.training-stability`, `llm.pretraining-optimization` (recent, verify before teaching).
- **Attention sinks and softmax fixes.** StreamingLLM sinks; gpt-oss uses learned per-head sink logits; gated attention (Qiu et al. 2025, used in Qwen3-Next) and differential attention. → `llm.attention-sinks`.
- **Hybrid linear-attention models.** Gated DeltaNet (ICLR 2025) in Qwen3-Next and Qwen3.5 (Feb 2026, now Qwen's main line); Kimi Linear (KDA, 3:1 KDA:MLA — Kimi Linear §2, pdf p.2, 8); Qwen3-Next/3.5 at 3:1 Gated DeltaNet:gated attention (Raschka 2026); Nemotron-H/3. → `llm.linear-attention`, `llm.modern-rnns-hybrids`, `llm.architecture-comparison`.
- **MoE everywhere at the frontier of open models.** Fine-grained experts with shared experts, auxiliary-loss-free balancing, sigmoid gating, and very sparse configurations
  (Kimi K2 with 384 experts; Qwen3.5-397B-A17B). → MoE unit. MoT (TMLR 2025) and MoD → `llm.mot-mod`.
- **Training stability and optimizers.** QK-norm is now common (OLMo 2, Gemma 3, Qwen3); MuonClip/QK-Clip (Kimi K2); WSD schedules; multi-stage LR schedules (DeepSeek-V3). → `llm.training-stability`, `llm.pretraining-optimization`.
- **Multi-token prediction.** Gloeckle 2024 and DeepSeek-V3/V4 MTP, also used for speculation. → `llm.pretraining-objective`, `llm.speculative-decoding`.
- **Reasoning and RL.** DeepSeek-R1 (the cached arXiv version is the revised one); GRPO and its 2025 fixes (Dr. GRPO, DAPO, GSPO); the Kimi k1.5 recipe;
  the **contested** "does RL add capability beyond the base model?" debate (Yue et al. 2025); s1 budget forcing; Snell's test-time compute-optimal scaling; RLVR reward hacking. → `llm.rlvr-grpo`, `llm.rlvr-practice`, `llm.test-time-compute`.
- **Latent/recurrent reasoning.** Recurrent-depth models (Geiping 2025), Coconut, looped transformers' expressivity (Saunshi 2025),
  and HRM/TRM (2025, **contested** claims). → `llm.looped-transformers` (wave 3).
- **Scaling-law corrections.** The Chinchilla replication (Besiroglu 2024: the published Approach-3 constants imply about 93 tokens/param, while the refit gives about 18–20);
  the Kaplan–Chinchilla reconciliation (Pearce & Song 2024; Porian 2024); inference-aware over-training. → scaling unit.
- **Tokenizer-free models.** The Byte Latent Transformer (entropy patching, Dec 2024) is still research. → `llm.tokenization-effects`.
- **Low precision.** FP8 training (DeepSeek-V3), MXFP4 weights (gpt-oss), FlashAttention-3 FP8. → `llm.quantization-advanced`.
- **Raschka's 2026 conclusion**, noted as context: architecture differences among open models matter less than data and training recipe.
  Efficiency tweaks (MLA, DSA, hybrid linear attention, SWA) dominate architectural change.

## Inline refreshers (prerequisites a newcomer may lack)

The brief asks for a short inline refresher wherever a lesson leans on a prerequisite the reader might be shaky on. Writers should budget
two to five sentences (or one half-card) for each item below, and link the fundamentals lesson where one exists.
- `llm.self-attention`: variance of a sum of independent zero-mean terms; the softmax Jacobian $\mathrm{diag}(p)-pp^\top$; matrix-calculus rules for $Y=XW$ (`fund.backprop`).
- `llm.positional-encodings`, `llm.rope`: angle-addition identities; 2×2 rotation matrices ($R_a^\top R_b=R_{b-a}$); complex numbers and Euler's formula.
- `llm.params-flops`: why a matmul costs $2mkn$ FLOPs (one line, then link `sys.flops-mfu`).
- `llm.tokenization`: UTF-8 byte lengths; EM and Viterbi for the unigram model (`fund.gmm-em` for EM).
- `llm.pretraining-objective`: cross-entropy = entropy + KL (`fund.information-theory`, `fund.kl-divergence`); arithmetic coding in one paragraph.
- `llm.scaling-laws`: constrained minimization by substitution (or a Lagrange multiplier); reading power laws on log–log axes.
- `llm.flash-attention`: log-sum-exp and the max-subtraction trick (`sys.stable-numerics`); SRAM vs HBM (`sys.gpu-basics`).
- `llm.kv-cache`: arithmetic intensity and the ridge point (`sys.roofline`); GB vs GiB.
- `llm.speculative-decoding`: rejection sampling; total-variation distance; the geometric series and truncated geometric expectation.
- `llm.moe-load-balancing`: why a non-differentiable count can multiply a differentiable term; Cauchy–Schwarz for the "minimum at uniform" step.
- `llm.lora`: matrix rank, SVD and low-rank approximation (Eckart–Young, one line).
- `llm.reward-modeling`: logistic regression as MLE (`fund.logistic-regression`); the Gumbel/logistic-noise derivation of Bradley–Terry.
- `llm.policy-gradients`: the log-derivative trick (`fund.gradient-estimators`); importance sampling; why a baseline leaves the gradient unbiased.
- `llm.dpo`: KL divergence and the Gibbs variational principle (minimizer of $\mathbb{E}_\pi[-r]+\beta\mathrm{KL}(\pi\|\pi_{ref})$).
- `llm.ssm`: solution of a linear ODE with a matrix exponential; discrete convolution and the FFT; eigenvalues and stability.
- `llm.mamba`: associative operators and the parallel prefix scan.
- `llm.linear-attention`: the kernel trick (feature maps $\phi$); associativity of matrix products; outer-product associative memories.
- `llm.mup`: how the scale of a sum of $n$ terms depends on correlation ($\sqrt n$ vs $n$); Adam's update is roughly sign-like in magnitude (`fund.adam`).
- `llm.quantization-advanced`: second-order Taylor expansion of a quadratic loss; inverse of a matrix after removing a row/column (for the OBS update).
- `llm.distillation`: forward vs reverse KL (`fund.kl-divergence`); softmax temperature.
- `llm.test-time-compute`, `llm.evaluation-metrics`: binomial and hypergeometric probabilities; Jensen's inequality (for the biased pass@k plug-in).

## Excluded, deferred, or left to other areas
- **Parallelism, collectives, mixed-precision training, activation checkpointing, profiling**: the applied/systems plan covers these.
  LLM lessons link to it. The ring-attention and expert-parallel arithmetic stays here because it is transformer-specific.
- **Multimodal LLM architectures** (vision encoders, image tokenizers, VLM training): the vision area covers these. Cross-attention and Perceiver conditioning stay here.
- **Discrete diffusion language models** (LLaDA, Gemini Diffusion, Mercury): a real 2025–26 trend, but not yet standard interview material, and they overlap the genmodels area.
  Deferred to a possible later advanced lesson. LLaDA is cached as `llm_nie2025_llada.txt`.
- **Agents and tool use**: fast-moving and light on derivation. Lambert's RLHF book ch. 13 is cached if a lesson is wanted later.
- **Safety evaluations, red-teaming, jailbreaks, watermarking**: outside the research-scientist technical core for this app. They get only brief mentions in the evaluation lessons.
- **Model merging and weight averaging**: mentioned in `llm.sft` and `llm.pretraining-optimization` only.
- **Prompt engineering beyond CoT/self-consistency**: excluded as low-signal for interviews.
- **Hallucination and factuality** as a standalone topic: partly covered by calibration (`llm.sampling`), RAG and evaluation. It has no derivation core, so it is not a lesson.
- **Historical encoder models in depth** (BERT variants, ELECTRA, etc.): only the objectives comparison in `llm.lm-objectives`.

---

## Unit A: Attention foundations

### 10 · `llm.self-attention` · Scaled dot-product & multi-head attention
core · wave 1 · prereqs: none · **existing file: rewrite in place, keep ids where meaning is unchanged**

Scope: one attention layer, end to end, for a reader who has only seen MLPs/RNNs. Causal masking, the KV cache and MQA/GQA move out to `llm.causal-attention` and `llm.mqa-gqa-mla` (the current file's q/f items on those should be retired or moved, with new ids there).

**Primary sources**
- Vaswani et al. 2017, *Attention Is All You Need*, §3.2.1 Scaled Dot-Product Attention (p.4, incl. footnote 4 on the variance argument), §3.2.2 Multi-Head (p.4–5), §4 Why Self-Attention + Table 1 (p.6). https://arxiv.org/abs/1706.03762 · `llm_vaswani2017_attention.txt`
- d2l.ai §11.1–11.3 (attention as kernel regression; §11.3.3 scaled dot product; §11.3.2.1 masked softmax), §11.5 Multi-Head Attention, §11.6.1–11.6.2 (self-attention; CNN vs RNN vs self-attention complexity/path-length table). https://d2l.ai/chapter_attention-mechanisms-and-transformers/ · `llm_d2l_attention_scoring.txt`, `llm_d2l_multihead_attention.txt`, `llm_d2l_self_attention_posenc.txt`
- Jurafsky & Martin SLP3 (Aug 2026 draft) ch. 7 §7.1 Attention, §7.1.1 Attention more formally, §7.3 Parallelizing computation using a single matrix X (pdf p.3–15). https://web.stanford.edu/~jurafsky/slp3/7.pdf · `llm_slp3_ch7.txt`
- JAX Scaling Book Part 4 "All the Transformer Math You Need to Know", sections *Counting Dots*, *Attention*, *Fractional cost of attention with context length*. https://jax-ml.github.io/scaling-book/transformers/ · `llm_scalingbook_transformers.txt`
- Dao et al. 2022 (FlashAttention) App. B.2 memory-efficient backward pass: $dV$, $dP$, $dS=P\odot(dP-D)$ with $D_i=\mathrm{rowsum}(dO_i\odot O_i)$ (pdf p.18–19, used again in `llm.flash-attention-2-3`). `llm_dao2022_flashattention.txt`

**Subtopic map (must cover)**
1. *The problem attention solves.* Fixed-size RNN state as a bottleneck; long path length between distant tokens; no parallelism over time. Attention = content-addressed, differentiable lookup. Start from Nadaraya–Watson kernel regression (d2l §11.2): output = similarity-weighted average of values; softmax of dot products is one kernel choice.
2. *Queries, keys, values.* Define $X\in\mathbb{R}^{n\times d}$, $Q=XW_Q$, $K=XW_K$, $V=XW_V$; shapes of every tensor; why separate Q and K projections (asymmetric relation "i looks for j" ≠ "j looks for i"; $W_QW_K^\top$ is a rank-$d_k$ bilinear form). Score matrix $n\times m$; softmax is **row-wise over keys**.
3. *Tiny worked example.* 3 tokens, $d_k=2$: compute scores, softmax, output by hand (numbers recomputed in Python).
4. *Why divide by $\sqrt{d_k}$ (derive).* Under i.i.d. zero-mean unit-variance components, $\mathrm{Var}(q\cdot k)=d_k$. Then the softmax Jacobian $\partial p/\partial z=\mathrm{diag}(p)-pp^\top$: when logits have std $\sqrt{d_k}\gg1$, $p$ is near one-hot and every Jacobian entry → 0, so gradients vanish. Scaling restores unit-variance logits at init. State the assumption honestly: it holds at initialization, not necessarily after training (logit growth is a known instability; links to `llm.training-stability`). Note μP prefers $1/d_k$ (see `llm.mup`); this is one of the "scaling factor" readings the user asked about.
5. *Multi-head attention.* $h$ heads with $d_k=d/h$; concat then $W_O$. Parameter count $4d^2$ (no biases) independent of $h$; FLOPs same as one wide head. What multiple heads buy: several attention distributions per token (one head can only produce one convex combination per query). Equivalent view: $W_O$ splits into per-head blocks, so MHA = sum over heads of $\mathrm{softmax}(\cdot)\,X W_V^{(i)} W_O^{(i)}$ (the "heads are independent and additive" view used in interpretability).
6. *Permutation equivariance (prove).* For a permutation matrix $P$: $\mathrm{Attn}(PX)=P\,\mathrm{Attn}(X)$, because $PQK^\top P^\top$ and row-wise softmax commutes with permutations. Consequence: order must be injected (→ `llm.positional-encodings`); a causal mask breaks the symmetry partially.
7. *Cost.* FLOPs: projections $8nd^2$ (Q,K,V,O, 2 FLOPs per MAC), $QK^\top$ $2n^2d$, $AV$ $2n^2d$. Memory: the $h\times n\times n$ score tensor. Compare with the MLP ($16nd^2$ at $4d$ hidden). Attention matmuls exceed the linear layers when $4n^2d>24nd^2$, i.e. $n>6d$; recompute with the Scaling Book's per-token formulation. Path length $O(1)$ vs $O(n)$ for RNNs, $O(\log_k n)$ for dilated CNNs (Vaswani Table 1).
8. *Self- vs cross-attention* (one paragraph; full treatment in `llm.cross-attention`). Fold in one sentence on additive (Bahdanau) vs dot-product scoring (d2l §11.3.4): dot product won because it is a matmul.
9. *Backward pass of one head (derive; a standard whiteboard question).* With $S=QK^\top/\sqrt{d_k}$, $P=\mathrm{softmax}_{\text{row}}(S)$, $O=PV$ and upstream $dO$: $dV=P^\top dO$; $dP=dO\,V^\top$; row-wise softmax Jacobian gives $dS_{ij}=P_{ij}(dP_{ij}-D_i)$ with $D_i=\sum_jP_{ij}dP_{ij}$, and show $D_i=dO_i\cdot O_i$ (so the backward never needs a full row of $dP$ to get $D$); $dQ=dS\,K/\sqrt{d_k}$, $dK=dS^\top Q/\sqrt{d_k}$; then $dW_Q=X^\top dQ$ etc. Count the cost (about 2× the forward matmuls) and note that the $n\times n$ matrix $P$ must be stored or recomputed — the hook for `llm.flash-attention`. Source: Dao 2022 App. B.2 (pdf p.18–19). Numerics (one line): max-subtraction in the softmax; mask handling lives in `llm.causal-attention`.

**Figures**
- Heatmap of a 6×6 attention matrix for a toy sentence with row-sums = 1 annotated; reader notices softmax is per row (per query).
- Softmax entropy vs logit standard deviation (synthetic: draw $q,k\sim\mathcal N(0,I_{d_k})$ for $d_k\in\{16,64,256\}$, with and without $1/\sqrt{d_k}$); reader sees unscaled attention collapse to near one-hot as $d_k$ grows.
- Data-flow diagram: $X\to Q,K,V\to$ scores $\to$ softmax $\to$ weighted sum $\to$ concat heads $\to W_O$, with tensor shapes on every arrow.
- FLOP share of attention vs linear layers as a function of $n/d$ (crossing at $n=6d$).

**Question ideas**
- Compute: given $q=(1,0)$, keys $(1,0),(0,1),(1,1)$, values scalars — output of scaled attention.
- Predict: double $d_k$ without rescaling; what happens to max attention weight and gradient norm at init? (figure-based: pick the curve for $d_k=256$ unscaled).
- Which is false: "MHA with $h$ heads has $h$ times the parameters of single-head attention."
- Spot the flaw: a derivation of $\mathrm{Var}(q\cdot k)=d_k$ that silently assumes $q$ and $k$ are correlated / non-zero-mean.
- Compare: at what sequence length do attention FLOPs equal projection+MLP FLOPs for $d=4096$?
- Open: "Prove self-attention is permutation-equivariant and say what breaks if you add a causal mask."
- Derivation step: given $dP$ and $P$ for one row, compute $dS$ and check that $\sum_j dS_{ij}=0$ (why must it be?).

**Pitfalls / checks**
- Variance argument assumes independence and unit variance at init; don't present it as holding during training.
- The crossover depends on conventions; state them. Reconciled during review (Python-checked): with a non-gated $4d$ MLP and MHA, linear layers cost $24d^2$ FLOPs per token (forward), full non-causal attention $4nd$ ⇒ $n=6d$; averaging over causal positions halves the attention term ⇒ $n=12d$, which is Kaplan's "$d_{model}>n_{ctx}/12$" (Kaplan §2.1, pdf p.7). The Scaling Book's $T>8D$ (Part 4, *Fractional cost of attention*) uses a gated MLP with $F=4D$ (three matrices ⇒ $32d^2$ per token) and non-causal attention. Llama-style SwiGLU with $F\approx\tfrac83d$ is back to $24d^2$ (plus GQA's smaller K/V projections).
- Current file says MHA params "4d²" — correct only without biases; say so.

---

### 20 · `llm.causal-attention` · Causal attention, teacher forcing & the KV cache
core · wave 1 · prereqs: `llm.self-attention`

**Primary sources**
- SLP3 ch. 7 §7.3 (masking the upper triangle; the $QK^\top$ matrix figure), §7.5–7.6 (LM head; autoregressive generation) (pdf p.12–25). `llm_slp3_ch7.txt`
- Vaswani 2017 §3.2.3 (masking in the decoder, p.5). `llm_vaswani2017_attention.txt`
- JAX Scaling Book Part 7 "All About Transformer Inference", *The Basics of Transformer Inference* (prefill vs generate, KV cache). https://jax-ml.github.io/scaling-book/inference/ · `llm_scalingbook_inference.txt`; and Part 4 *Key-Value (KV) caching*. `llm_scalingbook_transformers.txt`
- Bengio et al. 2015, *Scheduled Sampling*, §1–2.4 (exposure bias; p.1–4). https://arxiv.org/abs/1506.03099 · `llm_bengio2015_scheduled_sampling.txt`
- Llama 3 paper §3.2 (document attention mask within packed sequences, p.6). https://arxiv.org/abs/2407.21783 · `llm_dubey2024_llama3.txt`

**Subtopic map**
1. *Autoregressive factorization.* $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$; exact chain rule, no assumption. Training objective = sum of $T$ next-token cross-entropies.
2. *Why a mask makes training parallel.* Without masking, position $t$ could read $x_{t+1}$ (trivial copying). The causal mask $M_{ij}=-\infty$ for $j>i$ gives all $T$ conditionals in one forward pass = teacher forcing. Derive that row $i$ of the masked softmax only depends on $x_{\le i}$ at every layer (induction over layers).
3. *Train/inference asymmetry.* Training: one parallel pass over ground-truth prefixes. Inference: sequential, each token conditioned on the model's own samples. Exposure bias: train-time inputs never contain model errors; scheduled sampling as one fix and why it is rarely used for LLMs (breaks parallelism; RL-based post-training partly addresses it).
4. *The KV cache (derive why it is valid).* Because of the causal mask, keys/values of position $j$ at every layer depend only on $x_{\le j}$, so they are unchanged when new tokens are appended. Cache K,V, not Q (old queries are never reused). Cost of generating $T$ tokens: without cache $\sum_t O(t\,d^2 + t^2 d)$ = $O(T^2d^2+T^3d)$; with cache $O(Td^2+T^2d)$. Bidirectional encoders cannot do this (every position changes when a token is added).
5. *Prefill vs decode.* Prefill processes the prompt in parallel and fills the cache (compute-bound, like training). Decode adds one token per step (memory-bound; detail in `llm.kv-cache`).
6. *Mask bookkeeping with a cache.* New query at position $t$ attends to all cached keys $0..t$ (no mask needed for single-token decode; for chunked prefill the mask is offset by cache length). RoPE must use absolute positions of the new tokens.
7. *Padding.* Right vs left padding for batched generation; why left padding is standard for decoder-only generation; position ids must skip pads; key-padding masks combine with the causal mask; fully masked rows.
8. *Packing.* Concatenating documents to fill a context; cross-document attention leakage; block-diagonal causal (document) masks (Llama 3 uses them); loss masking at boundaries.
9. *Other mask shapes.* Prefix-LM (bidirectional over the prefix, causal after), sliding-window causal (preview of `llm.sparse-attention`), encoder (no mask). Draw them.
10. *Implementation pitfalls.* `triu` vs `tril`, mask before softmax, $-\infty$ in fp16, mask dtype broadcasting.

**Figures**
- Four mask patterns side by side as $8\times8$ grids: bidirectional, causal, prefix-LM, causal + document mask for 3 packed docs. Reader notices which entries each query can see.
- Generation timeline: prefill block then decode steps, with the KV cache growing column by column; the new row (query) per step is a single row.
- Cumulative FLOPs to generate $T$ tokens with vs without a cache (log scale), showing $T^3$ vs $T^2$ growth for the attention term.

**Question ideas**
- Predict: an engineer removes the causal mask during pretraining but keeps it at inference. What happens to training loss and to generations?
- Which is false: "BERT can use a KV cache for incremental encoding."
- Spot the bug: chunked prefill of 4 new tokens with a cache of 10, mask built as a $4\times4$ lower triangle.
- Compute: number of attention score entries computed to generate 100 tokens after a 900-token prompt, with and without cache.
- Figure MCQ: which mask diagram corresponds to a prefix-LM with prefix length 3?
- Open: "Explain why caching K and V is exact, and why we never cache Q."

**Pitfalls / checks**
- KV-cache validity relies on causal masking at every layer; sliding-window caches can evict, but bidirectional layers break caching entirely.
- Left padding + RoPE: positions must be computed from the attention mask, not the raw index.

---

### 30 · `llm.lm-objectives` · Decoder-only, encoder-only and encoder–decoder models
intermediate · wave 2 · prereqs: `llm.causal-attention`

**Primary sources**
- Devlin et al. 2018 (BERT), §3.1 masked LM (15% selection, 80/10/10 replacement, p.4), NSP; §5.1 ablation. https://arxiv.org/abs/1810.04805 · `llm_devlin2018_bert.txt`
- Raffel et al. 2019 (T5), §3.2 Architectures (Fig. 3–4 mask patterns: encoder-decoder, LM, prefix LM; p.15–19), §3.3 Unsupervised objectives (span corruption; p.20–). https://arxiv.org/abs/1910.10683 · `llm_raffel2019_t5.txt`
- Wang et al. 2022, *What LM Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?*, §2.1–2.2 and results §4 (p.3–10). https://arxiv.org/abs/2204.05832 · `llm_wang2022_lm_arch_objective.txt`
- Radford et al. 2019 (GPT-2), §2 Approach (LM as unsupervised multitask learner). `llm_radford2019_gpt2.txt`
- SLP3 ch. 9 Masked Language Models: §9.2.1 bidirectional architecture, §9.3.1 masking words (80/10/10), §9.3.2 next sentence prediction, §9.3.3 training regimes (pdf p.3–8). https://web.stanford.edu/~jurafsky/slp3/9.pdf · `llm_slp3_ch9.txt`; ch. 13 §13.3 encoder–decoder details: `llm_slp3_ch13.txt`

**Subtopic map**
1. Three architecture families and their masks; which conditionals each can model; parameter sharing between encoder and decoder.
2. CLM objective (recap) vs MLM: what MLM estimates ($p(x_i\mid x_{\setminus M})$, not a joint), why it is not a generative model without tricks; the 80/10/10 rule and why (train/test mismatch of `[MASK]`); only ~15% of tokens give a loss signal → sample efficiency argument.
3. Span corruption (T5): sentinel tokens, mean span length 3, 15% corruption; output only the spans → shorter targets.
4. Prefix LM and UL2-style mixtures (mention); bidirectional prefix + causal continuation.
5. Why decoder-only dominates at scale: one objective for every token, KV-cacheable generation, simplicity, emergent zero/few-shot (GPT-2/3); Wang 2022 findings (causal decoder best after pure unsupervised pretraining for zero-shot; encoder–decoder with MLM best after multitask finetuning) — present as evidence, not law.
6. Where encoders still win: embeddings/retrieval, classification; encoder–decoder in translation/speech (Whisper) and T5-family.
7. Compute comparison: encoder–decoder with $L$+$L$ layers vs decoder-only with $2L$ layers at matched params/FLOPs (T5 §3.2.2 discussion).
8. Fill-in-the-middle as a reordering trick that lets a causal LM infill (brief).

**Figures**
- Three mask diagrams + which tokens contribute loss (shaded) for CLM, MLM, span corruption on the same 10-token sentence.

**Question ideas**
- Compute: fraction of tokens receiving a gradient per sequence under BERT MLM vs CLM.
- Predict: replace every selected token with `[MASK]` (no 10/10); effect at fine-tuning time.
- Which is false: "An MLM's per-token conditionals define a valid joint distribution by the chain rule."
- Compare: encoder–decoder vs decoder-only at matched FLOPs for summarization.

**Pitfalls / checks**
- Wang 2022 conclusions depend on adaptation regime; do not state "decoder-only is better" unconditionally.
- Check T5 span-corruption defaults (15%, mean span 3) against T5 §3.3.4 before writing.

---

### 40 · `llm.cross-attention` · Cross-attention, encoder–decoder conditioning & Perceiver
intermediate · wave 2 · prereqs: `llm.self-attention`, `llm.lm-objectives`

**Primary sources**
- Vaswani 2017 §3.2.3 (encoder–decoder attention, p.5). `llm_vaswani2017_attention.txt`
- Jaegle et al. 2021 (Perceiver), §3.1 architecture: latent array, cross-attention cost $O(MN)$, latent transformer $O(N^2)$ in latent count, weight sharing (p.3–4). https://arxiv.org/abs/2103.03206 · `llm_jaegle2021_perceiver.txt`
- Jaegle et al. 2021 (Perceiver IO), §3.1–3.2 encode/process/decode with an output query array (p.4–5). https://arxiv.org/abs/2107.14795 · `llm_jaegle2021_perceiver_io.txt`
- Alayrac et al. 2022 (Flamingo), §2.1 Perceiver Resampler, §2.2 gated cross-attention (tanh gate initialised at 0), §2.3 per-image masking (p.5–6). https://arxiv.org/abs/2204.14198 · `llm_alayrac2022_flamingo.txt`
- Rombach et al. 2022 (Latent Diffusion), §3.3 Conditioning Mechanisms (cross-attention in the U-Net). Shared cache: `papers/rombach2022_ldm.txt`

**Subtopic map**
1. Definition: $Q$ from the target stream $X\in\mathbb{R}^{n\times d}$, $K,V$ from the source $Z\in\mathbb{R}^{m\times d_z}$; output shape $n\times d$; cost $O(nm d)$; no causal mask on the source side; permutation-invariant in the source.
2. Encoder–decoder transformer block: self-attn (causal) → cross-attn → FFN; the encoder output is computed once and its K/V can be cached for all decode steps (static cross-attention KV cache).
3. Conditioning generative models (one card; `gen.latent-diffusion` owns the LDM details and `gen.guidance` owns CFG — link, don't re-teach): text-to-image diffusion (U-Net/DiT tokens attend to text embeddings); compare with alternatives (concatenation/in-context tokens as in MM-DiT/SD3 joint attention, FiLM/adaLN modulation).
4. Perceiver: problem = quadratic cost on huge inputs (pixels, audio); a small learned latent array of $N\ll M$ queries cross-attends to the input, then self-attends in latent space; total $O(MN + LN^2)$ instead of $O(LM^2)$; iterative cross-attention and weight sharing; position encodings (Fourier features) must be attached to inputs.
5. Perceiver IO: decode by cross-attending from an output query array → arbitrary output structure.
6. Flamingo: Perceiver Resampler (fixed number of visual tokens), gated xattn-dense layers interleaved into a frozen LM, tanh(α) gate initialised at 0 so the LM starts unchanged; per-image masks.
7. Q-Former/learned query bottlenecks (BLIP-2) as the same idea (mention only; not cached).
8. Cross-attention vs self-attention over a concatenated sequence: expressivity and cost trade-off (concat lets the source also attend to the target; cost $(n+m)^2$).

**Figures**
- Block diagram: decoder layer with causal self-attn then cross-attn arrows from encoder outputs; shapes.
- Perceiver cost plot: FLOPs vs input size $M$ for full self-attention vs latent bottleneck with $N=256$.
- Flamingo gate: $\tanh(\alpha)$ at init 0 → output of the gated layer is exactly the frozen LM's.

**Question ideas**
- Compute: Perceiver with $M=50{,}000$ inputs, $N=512$ latents, $d=1024$: cross-attention score entries vs full self-attention.
- Predict: Flamingo gate initialised at 1 instead of 0 — effect on early training.
- Which is false: "In cross-attention the number of output tokens equals the number of source tokens."
- Compare: cross-attention conditioning vs joint self-attention over concatenated text+image tokens.

**Pitfalls / checks**
- Perceiver complexity notation: paper uses $M$ (input) and $N$ (latents); keep consistent.
## Unit B: Position information

### 50 · `llm.positional-encodings` · Why position matters; learned and sinusoidal absolute encodings
core · wave 1 · prereqs: `llm.self-attention` · **existing file: rewrite in place; relative encodings move to `llm.relative-position`, RoPE and context extension to `llm.rope` and `llm.context-extension`**

Split during review: the original 11-item syllabus (absolute + Shaw + Transformer-XL + T5 + ALiBi + NoPE + comparison) would have run to
12–13 cards once the brief's "interviewers probe" and "key results" cards are added. This lesson ends on the four-term expansion, which is the
question that `llm.relative-position` and `llm.rope` answer.

**Primary sources**
- Vaswani 2017 §3.5 Positional Encoding (sinusoids; the "linear function of $PE_{pos}$" claim, p.6). `llm_vaswani2017_attention.txt`
- d2l §11.6.3 Positional Encoding: §11.6.3.1 absolute, §11.6.3.2 relative (derives the 2×2 rotation that maps $PE_p$ to $PE_{p+\delta}$). `llm_d2l_self_attention_posenc.txt`
- Kazemnejad et al. 2023 (NoPE) §5.1 Theorem 1 (a causal decoder without PE can recover absolute position; p.5–6, proof App. C.1). https://arxiv.org/abs/2305.19466 · `llm_kazemnejad2023_nope.txt`
- Dai et al. 2019 (Transformer-XL) §3.3 (the four-term expansion of the absolute-PE logit, p.4–5) for item 5. `llm_dai2019_transformerxl.txt`

**Subtopic map**
1. *Why position is needed:* recap of permutation equivariance; "dog bites man". What a causal mask already leaks (NoPE: a token can count how many tokens precede it, e.g. by attending uniformly to a BOS marker), and why explicit PE still helps.
2. *Design axes:* absolute vs relative; added to input vs injected into attention logits; fixed vs learned; extrapolation beyond training length. (This axis list is the map for the next two lessons.)
3. *Learned absolute* (BERT/GPT-2): table of $L_{\max}$ vectors; cannot represent unseen positions.
4. *Sinusoidal (derive the shift property):* $PE_{p,2i}=\sin(p\omega_i)$, $PE_{p,2i+1}=\cos(p\omega_i)$, $\omega_i=10000^{-2i/d}$. Show $[PE_{p+\delta,2i},PE_{p+\delta,2i+1}]$ = 2×2 rotation by $\delta\omega_i$ applied to $[PE_{p,2i},PE_{p,2i+1}]$ (angle-addition identities), so a fixed linear map converts absolute to shifted position. Geometric wavelengths $2\pi$ to $10000\cdot2\pi$ = multi-scale clock; dot product $PE_p\cdot PE_{p+\delta}=\sum_i\cos(\delta\omega_i)$ depends only on $\delta$ and decays (noisily) with $|\delta|$. Why adding (not concatenating) works in high dimension.
5. *Why absolute-add is not truly relative:* expand $(x_i+p_i)^\top W_QW_K^\top(x_j+p_j)$ into four terms — content-content, content-position, position-content, position-position. Only the last depends on both positions, and even it is not a function of $i-j$ alone once $W_QW_K^\top$ sits between the sinusoids. This motivates relative schemes (next lesson) and RoPE.
6. *NoPE:* theorem sketch and empirical length-generalization claims (small-scale); modern use as interleaved NoPE layers (e.g. SmolLM3 / Llama 4 per Raschka) — mark as practice, not consensus.
7. *KV-cache compatibility:* absolute-add PE enters once at the input, so cached keys stay valid; contrast with schemes that must re-encode on a shift.

**Figures**
- Sinusoidal PE heatmap (positions × dims) plus the dot-product $PE_0\cdot PE_\delta$ vs $\delta$ curve.
- Rotation picture: one 2-D sin/cos pair as a point on a circle advancing by $\omega_i$ per position, for a fast and a slow frequency.
- Four-term expansion diagram: the logit split into four coloured terms, with the one(s) carrying position marked.

**Question ideas**
- Derivation step: which identity turns $PE_{p+\delta}$ into a rotation of $PE_p$?
- Predict: a learned-absolute model trained at 2k evaluated at 4k.
- Which is false: "With additive sinusoidal PE, the attention logit between positions $i$ and $j$ depends only on $i-j$." (False: the content–position cross terms and $W_QW_K^\top$ break it.)
- Compute: wavelength of the slowest sinusoid pair for $d=512$ ($2\pi\cdot10000^{510/512}\approx6.0\times10^4$ positions; Python-verified).
- Open: "Expand the attention logit with additive absolute PE and say which terms carry position."

**Pitfalls / checks**
- Sinusoidal pairing convention: the paper interleaves sin/cos by even/odd index; many codebases use half-split. Say so.
- NoPE results are from small models on synthetic tasks; mark the scale caveat.

---

### 55 · `llm.relative-position` · Relative position: Shaw, Transformer-XL, T5 buckets, ALiBi
intermediate · wave 2 · prereqs: `llm.positional-encodings` · new lesson (split from the old positional-encodings scope; existing relative-PE items in the old file get new ids here)

**Primary sources**
- Shaw, Uszkoreit & Vaswani 2018, §3.1–3.3 relation-aware attention, clipping distance $k$ (p.2–3). https://arxiv.org/abs/1803.02155 · `llm_shaw2018_relative.txt`
- Dai et al. 2019 (Transformer-XL) §3.3 Relative Positional Encodings: the four-term decomposition (a)–(d), global bias vectors $u,v$, $W_{k,R}$ (p.4–5); App. B efficient computation (p.12). https://arxiv.org/abs/1901.02860 · `llm_dai2019_transformerxl.txt`
- Raffel 2019 (T5) §2.1 Model: 32 learned scalar biases per head, log-spaced buckets up to offset 128, shared across layers (p.5). `llm_raffel2019_t5.txt`
- Press et al. 2021 (ALiBi) §3 (head slopes = geometric sequence starting at $2^{-8/n}$; p.5), §4 extrapolation results. https://arxiv.org/abs/2108.12409 · `llm_press2021_alibi.txt`

**Subtopic map**
1. *Goal:* make the logit depend on $j-i$ rather than on absolute indices; two places to inject it — inside the key/value vectors, or as an additive bias on the logit.
2. *Shaw et al.:* learned vectors $a^K_{ij}, a^V_{ij}$ indexed by clipped offset $\mathrm{clip}(j-i,-k,k)$, added to keys (and values) inside attention; memory-efficient implementation; $2k+1$ embeddings.
3. *Transformer-XL relative form (derive from the four-term expansion):* replace absolute $p_j$ by a sinusoidal $R_{i-j}$; replace the query-side position term by learned global vectors $u$ (content bias) and $v$ (position bias); separate $W_{k,E}$ and $W_{k,R}$. Interpret the four terms (a) content addressing, (b) content-dependent position bias, (c) global content bias, (d) global position bias. Pairs with segment recurrence (`llm.long-context`), which is why it needed relative positions (cached states from a previous segment would otherwise reuse the same absolute positions). Efficient computation via the shift trick (App. B), one paragraph.
4. *T5 buckets:* one learned scalar per (head, bucket) added to logits; exact buckets for small offsets, log-spaced up to 128, one bucket beyond; shared across layers; deeper layers can still compose longer ranges.
5. *ALiBi:* subtract $m_h\,(i-j)$ from logits; slopes geometric ($\tfrac12,\tfrac14,\dots,\tfrac1{256}$ for 8 heads); no learned params; recency bias; good extrapolation in perplexity. Limitation: the linear penalty makes far tokens hard to retrieve (weak on long-range retrieval tasks).
6. *Comparison table:* parameters, where injected, relative?, extrapolation, KV-cache compatibility (all cache-friendly), cost, fused-kernel friendliness (additive biases need kernel support; RoPE does not — forward pointer to `llm.rope`).

**Figures**
- ALiBi bias matrices for two heads (slopes 1/2 and 1/256) as heatmaps; reader notices steep vs nearly flat recency.
- T5 bucket assignment: offset (0–300) → bucket id, showing linear then log spacing and the cap at 128.
- Transformer-XL's four terms (a)–(d) as a 2×2 grid of content/position × query/global, with the learned $u$, $v$ replacing the query-position terms.

**Question ideas**
- Compute: ALiBi slopes for 16 heads (geometric from $2^{-8/16}=2^{-1/2}$).
- Which is false: "T5's relative bias makes each layer sensitive to offsets beyond 128."
- Figure MCQ: identify which heatmap is the ALiBi head with the smallest slope.
- Predict: Shaw-style clipping at $k=16$ — what can a single layer distinguish between offsets 20 and 200?
- Open: "Show where Transformer-XL's $u$ and $v$ come from, starting from the four-term expansion."

**Pitfalls / checks**
- ALiBi slopes for non-power-of-2 head counts follow a special rule (ALiBi §3); check before quoting.

---

### 60 · `llm.rope` · Rotary position embedding (RoPE)
core · wave 1 · prereqs: `llm.positional-encodings` (recommended: `llm.relative-position`; item 10 gives a one-paragraph recap of ALiBi/T5 so the lesson stands alone in wave 1)

**Primary sources**
- Su et al. 2021 (RoFormer) §3.2.1 2-D case (complex form), §3.2.2 general form (block-diagonal $R^d_{\Theta,m}$, $\theta_i=10000^{-2(i-1)/d}$), §3.3 properties, §3.4.1 derivation, §3.4.2 efficient computation, §3.4.3 long-term decay (p.4–8). https://arxiv.org/abs/2104.09864 · `llm_su2021_roformer.txt`
- EleutherAI blog, *Rotary Embeddings: A Relative Revolution* (2021): intuitive derivation, complex-number view, implementation. https://blog.eleuther.ai/rotary-embeddings/ · `llm_eleuther2021_rotary.txt`
- Peng et al. 2023 (YaRN) §2.1 RoPE recap and §3.2 wavelength view $\lambda_d=2\pi b^{2d/|D|}$ (p.2–5). `llm_peng2023_yarn.txt`
- Llama 3 §3.2 (RoPE base 500,000, p.7). `llm_dubey2024_llama3.txt`

**Subtopic map**
1. *Goal stated as an equation:* find $f(x,m)$ with $\langle f_q(x_m,m), f_k(x_n,n)\rangle = g(x_m,x_n,m-n)$. Motivation: relative positions inside the dot product without extra parameters, and compatible with a KV cache (keys are rotated once, at their own position).
2. *2-D solution:* identify $\mathbb{R}^2$ with $\mathbb{C}$; $f(x,m)=xW e^{\mathrm{i}m\theta}$; then $\mathrm{Re}[q e^{\mathrm{i}m\theta}\,\overline{k e^{\mathrm{i}n\theta}}]=\mathrm{Re}[q\bar k e^{\mathrm{i}(m-n)\theta}]$. Same thing as a real rotation: $(R_m q)^\top(R_n k)=q^\top R_{n-m}k$ using $R_m^\top R_n=R_{n-m}$ (rotations compose, are orthogonal).
3. *General $d$:* split into $d/2$ pairs, each with its own frequency $\theta_i=b^{-2i/d}$ ($b=10000$ originally); block-diagonal rotation; norm-preserving (orthogonal) so it doesn't change $\|q\|,\|k\|$.
4. *Efficient implementation:* elementwise $x\odot\cos(m\theta) + \mathrm{rot}(x)\odot\sin(m\theta)$, never a $d\times d$ matmul; interleaved-pairs (paper/GPT-J) vs half-split (GPT-NeoX/Llama) layouts are different permutations — weights are not interchangeable across conventions.
5. *Applied to Q and K only, not V;* applied after projection, at every layer (unlike additive PE, which enters once at the input).
6. *Long-term decay (sketch):* the RoFormer bound via Abel summation — the relative upper bound of the sum of $q_ik_i^*e^{\mathrm{i}(m-n)\theta_i}$ decays as $|m-n|$ grows, for $\theta_i$ geometric. Be honest that it is an upper bound on a worst case, not a guaranteed decay of actual scores.
7. *Frequency/wavelength picture:* wavelength $\lambda_i=2\pi/\theta_i$; high-frequency pairs encode fine local order, low-frequency pairs rotate less than once over the training context and act almost like absolute position. This is the key to `llm.context-extension`.
8. *Base $b$:* larger base → longer wavelengths; why Llama 3 uses $b=500{,}000$ and long-context models raise it further.
9. *Variants:* partial RoPE (rotate only a fraction of head dims; GPT-NeoX/GPT-J, MiniMax-M2 per Raschka), 2-D/3-D RoPE for images/video (M-RoPE) — mention; decoupled RoPE in MLA (`llm.mqa-gqa-mla`).
10. *RoPE vs ALiBi vs T5:* parameter-free, multiplicative, relative, cache-friendly; extrapolation without modification is poor (unseen rotation angles), unlike ALiBi's perplexity extrapolation.

**Figures**
- Two 2-D vectors $q,k$ rotated by $m\theta$ and $n\theta$; the angle between them depends only on $m-n$ (draw for $(m,n)=(2,5)$ and $(7,10)$ side by side).
- Wavelength per pair index for $b=10^4$ and $b=5\times10^5$ vs a horizontal line at the training context (4k / 8k); reader sees which pairs complete a full rotation.
- $\cos(\delta\theta_i)$ summed over pairs vs offset $\delta$ (the decay-like curve), for $b=10^4$.

**Question ideas**
- Derivation step: which property of rotation matrices gives $(R_mq)^\top(R_nk)=q^\top R_{n-m}k$?
- Compute: wavelength of the lowest-frequency pair for $d_{head}=128$, $b=10^4$: $2\pi\cdot10^{4\cdot126/128}\approx5.4\times10^4$ tokens (Python-verified) — far beyond a 4096 training context.
- Predict: apply RoPE to V as well; does the output still depend only on relative position? (No: the weighted sum of rotated values carries absolute phase.)
- Which is false: "RoPE changes the norm of query vectors as position grows."
- Spot the bug: half-split RoPE kernel applied to a checkpoint trained with interleaved pairs.
- Open: "Derive RoPE from the requirement that the score depends only on $m-n$."

**Pitfalls / checks**
- Index convention: RoFormer writes $\theta_i=10000^{-2(i-1)/d}$ for $i=1..d/2$; code uses $i=0..d/2-1$. Same set.
- Long-term decay is a bound; don't call it a property of trained models.

---

### 70 · `llm.context-extension` · Extending RoPE context: PI, NTK-aware, YaRN, ABF
intermediate · wave 2 · prereqs: `llm.rope`

This lesson carries the "**RoPE scaling factor**" reading of the user's "LLM scaling factor" item.

**Primary sources**
- Chen et al. 2023 (Position Interpolation) §2 method (scale $m\to m/s$), §2.3 Theorem 2.1 interpolation bound (~600× tighter than extrapolation for LLaMA-7B, p.2–5), fine-tuning ≤1000 steps. https://arxiv.org/abs/2306.15595 · `llm_chen2023_pi.txt`
- Peng et al. 2023 (YaRN) §3.1 NTK-aware (base change; formula in App. A.2), §3.2 NTK-by-parts (ramp $\gamma(r)$ with $\alpha=1,\beta=32$ for Llama), §3.3 Dynamic scaling, §3.4 attention temperature $\sqrt{1/t}=0.1\ln s+1$ (p.4–6). https://arxiv.org/abs/2309.00071 · `llm_peng2023_yarn.txt`
- Xiong et al. 2023 (Effective Long-Context Scaling) §3 / Fig. on RoPE ABF (base 10,000 → 500,000) and continual pretraining recipe (p.6–8). https://arxiv.org/abs/2309.16039 · `llm_xiong2023_long_context_scaling.txt`
- Ding et al. 2024 (LongRoPE) — non-uniform per-dimension rescaling found by search; progressive extension. https://arxiv.org/abs/2402.13753 · `llm_ding2024_longrope.txt`
- Llama 3 §3.4.2 long-context pretraining (six stages, 8K → 128K). `llm_dubey2024_llama3.txt`

**Subtopic map**
1. *Why naive extrapolation fails:* at positions $>L$, low-frequency pairs see rotation angles never seen in training; attention scores blow up / become erratic (PI Fig. 2 argument). Perplexity explodes just past $L$.
2. *Position Interpolation:* use $m/s$ with $s=L'/L$; all angles stay inside the trained range. Theorem 2.1 intuition: interpolated scores are smooth between trained integer positions, so error is bounded, unlike extrapolation. Cost: high-frequency pairs get compressed, so neighbouring tokens become harder to distinguish → needs fine-tuning; degrades beyond $s\approx8$ (YaRN §3.1).
3. *NTK-aware scaling (derive the base change):* keep the highest frequency $\theta_0=1$ fixed, make the lowest frequency $\theta_{d/2-1}=b^{-(d-2)/d}$ shrink by $s$: solve $b'^{-(d-2)/d}=b^{-(d-2)/d}/s$ ⇒ $b'=b\,s^{d/(d-2)}$. Interpretation: spread "interpolation pressure" — little for high frequencies, full for low. Works partly without fine-tuning. Weakness: some dimensions end up slightly extrapolated.
4. *NTK-by-parts:* per-pair ratio $r_i=L/\lambda_i$ (number of full rotations in the training context). Ramp: if $r_i>\beta$ don't interpolate; if $r_i<\alpha$ interpolate fully ($\theta_i/s$); linear blend between. Why: pairs with many rotations encode only relative information (leave them); pairs with < 1 rotation behave as absolute (must interpolate).
5. *Dynamic scaling:* choose $s=\max(1, \ell/L)$ from the current length $\ell$ at inference so short prompts are unaffected; caveat with KV caches (keys rotated with an old $s$).
6. *YaRN temperature:* scale logits by $1/t$ with $\sqrt{1/t}=0.1\ln s+1$; reason: interpolation raises attention entropy at long lengths; empirical fit for Llama. YaRN = NTK-by-parts + temperature.
7. *ABF / base raising during continued pretraining:* Code Llama ($b=10^6$), Xiong 2023 ($b=5\times10^5$), Llama 3 ($b=5\times10^5$ from the start). Same idea as NTK-aware but trained.
8. *LongRoPE:* searched non-uniform factors; progressive 256k → 2M (mention).
9. *Data and training side:* extension needs long documents and continued pretraining; progressive length schedules (Llama 3 six stages). Short-context quality regressions and how they are checked.
10. *ALiBi extrapolation vs RoPE interpolation:* ALiBi extrapolates perplexity without tuning but is weaker at retrieving far tokens; perplexity is not evidence of using long context (→ `llm.long-context-eval`).

**Figures**
- Rotation angle $m\theta_i$ vs position for one low- and one high-frequency pair, shading the trained range $[0,L]$; overlay extrapolation, PI and NTK-aware curves.
- Per-pair scale factor vs pair index for PI (flat $1/s$), NTK-aware (smooth), NTK-by-parts (ramp).
- YaRN temperature $1/t$ vs $s$.

**Question ideas**
- Compute: NTK-aware base for $s=4$, $d=128$, $b=10^4$: $b'=10^4\cdot4^{128/126}\approx4.09\times10^4$ (Python-verified).
- Predict: apply PI with $s=16$ and no fine-tuning; which tokens get confused (neighbours)?
- Which is false: "NTK-by-parts interpolates the highest-frequency dimensions most strongly."
- Compare: dynamic NTK vs static YaRN when serving with a KV cache.
- Figure MCQ: which curve is NTK-by-parts?

**Pitfalls / checks**
- The NTK-aware formula is in YaRN App. A.2 (original is a Reddit post by bloc97 — blocked from this network; cite YaRN).
- Ramp parameters $\alpha=1,\beta=32$ are Llama-specific (YaRN §3.2).
- "Interpolation bound ~600×" is for LLaMA-7B setting (PI §2.3).
## Unit C: The transformer block, counting, and stability

### 80 · `llm.transformer-block` · The modern decoder block
core · wave 1 · prereqs: `llm.self-attention`

**Primary sources**
- Vaswani 2017 §3.1 (post-LN sublayer: $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$), §3.3 FFN ($d_{ff}=4d$) (p.3–5). `llm_vaswani2017_attention.txt`
- Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture*, Theorem 1 (p.5–6): last-layer gradient norm $O(d\sqrt{\ln d})$ for Post-LN vs $O(d\sqrt{\ln d/L})$ for Pre-LN; warmup discussion. Shared cache: `papers/xiong2020_preln.txt` (file contains NUL bytes; use `grep -a`).
- Shazeer 2020, *GLU Variants Improve Transformer* §2 (FFN$_{\mathrm{SwiGLU}}$; $d_{ff}$ cut from 3072 to 2048 to match params, p.2). Shared cache: `papers/shazeer2020_glu.txt`
- Zhang & Sennrich 2019 (RMSNorm). Shared cache: `papers/zhang2019_rmsnorm.txt`; Ba et al. 2016 LayerNorm: `papers/ba2016_layernorm.txt`
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*, sections "Virtual Weights and the Residual Stream as a Communication Channel", "Attention Heads are Independent and Additive". https://transformer-circuits.pub/2021/framework/index.html · `llm_elhage2021_circuits_framework.txt`
- SLP3 ch. 7 §7.2 Transformer Blocks (residual stream view, layer norm, §7.2.3 putting it together; pdf p.8–12). `llm_slp3_ch7.txt`
- d2l §11.7 The Transformer Architecture: §11.7.2 position-wise FFN, §11.7.3 residual connection and layer normalization (LayerNorm vs BatchNorm), §11.7.4–11.7.5 encoder/decoder blocks. https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html · `llm_d2l_transformer.txt`
- Chowdhery et al. 2022 (PaLM) §2 Model Architecture: SwiGLU, parallel layers, MQA, RoPE, shared input–output embeddings, no biases, 256k SentencePiece vocab (p.5–6). https://arxiv.org/abs/2204.02311 · `llm_chowdhery2022_palm.txt`
- Raschka, *The Big LLM Architecture Comparison* (2025–26): §2.1 norm placement (OLMo 2 post-norm-inside-residual), §3.2 Gemma 3 pre+post norm, per-model configs. https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison · `llm_raschka_big_arch_comparison.txt`

**Subtopic map**
1. *Residual stream:* $x_{l+1}=x_l+F_l(\cdot)$; each sublayer reads from and writes to a shared $d$-dimensional channel; the identity path gives a direct gradient route (derive $\partial x_L/\partial x_l = I + \sum(\dots)$). Interpretation from Elhage: sublayers communicate through linear subspaces.
2. *LayerNorm vs RMSNorm:* formulas, what each removes (mean & scale vs scale only), learned gain (and bias), cost; RMSNorm is standard (Llama, etc.). Normalization is per token (contrast BatchNorm: why BN is unsuitable for variable-length autoregressive training/inference).
3. *Post-LN vs Pre-LN (derive the intuition, cite Xiong's theorem):* Post-LN puts a norm on the residual path, so gradients near the output are large at init and need warmup; Pre-LN keeps an identity path, gradients scale as $1/\sqrt{L}$ at the last layer, trainable without warmup but the residual stream norm grows with depth and later layers contribute less ("representation collapse"/ diminishing updates). Variants: sandwich/peri-norm (Gemma 2/3: norm before and after each sublayer), OLMo 2 (norm on sublayer output inside the residual), final norm before the LM head. Mark which is consensus (pre-norm family) vs contested details.
4. *FFN:* two-layer MLP with $4d$ hidden; role as key–value memory (mention); activations ReLU → GELU → SiLU.
5. *Gated FFNs:* GLU family $(\sigma(xW)\odot xV)W_2$; SwiGLU uses Swish/SiLU. Three matrices ⇒ to match a $4d$ two-matrix FFN's $8d^2$ params, set hidden $=\tfrac{8}{3}d$ ($3\cdot d\cdot\tfrac83 d = 8d^2$); Llama rounds to a multiple of 256 (Llama-3-70B: 28,672 = 3.5·8192, so not exactly 8/3 — note that modern models choose the ratio freely).
6. *Embeddings and LM head:* tied vs untied (GPT-2/PaLM tie; Llama 3 unties); embedding scaling ($\sqrt d$ multiplier in some models); vocab-size share of params at small scale.
7. *Biases removed* (PaLM, Llama): stability and throughput; QKV bias retained in some (Qwen2) — a design choice.
8. *Parallel block* (GPT-J/PaLM): $x + \mathrm{Attn}(\mathrm{LN}(x)) + \mathrm{MLP}(\mathrm{LN}(x))$; fuses the input matmuls (~15% faster at scale per PaLM), quality-neutral at 540B per PaLM, slightly worse at small scale.
9. *The "Llama-style" recipe as a checklist:* pre-RMSNorm, RoPE, SwiGLU, GQA, no biases, untied embeddings; and 2025–26 additions: QK-norm (OLMo 2, Gemma 3, Qwen3), sliding-window/global interleave (Gemma 3), MoE FFNs, hyper-connections/mHC widening the residual stream (DeepSeek V4; advanced, contested — see `llm.training-stability`).
10. *Depth vs width* (gpt-oss vs Qwen3 comparison in Raschka §9.1): what changes in latency/parallelism; no clean rule.

**Figures**
- Side-by-side block diagrams: Post-LN (original), Pre-LN, sandwich/peri-norm, parallel block; highlight where the identity path is interrupted.
- Synthetic experiment: residual-stream RMS vs layer index for Pre-LN with random init (numpy forward pass through random sublayers) — reader sees growth ∝ $\sqrt{l}$.
- SwiGLU vs GELU vs ReLU activation curves (and SiLU derivative).

**Question ideas**
- Compute: hidden size of a SwiGLU FFN that matches a $4d$ GELU FFN at $d=4096$.
- Predict: switch a working Pre-LN model to Post-LN, same LR, no warmup.
- Which is false: "RMSNorm subtracts the mean of the activations before scaling."
- Spot the flaw: "Pre-LN removes the need for normalization at the output" (forgets the final norm).
- Figure MCQ: which diagram keeps a clean identity path from input to output?
- Open: "Walk through one Llama-style block and justify each design choice."

**Pitfalls / checks**
- Xiong's theorem is a bound at initialization under assumptions; don't overstate.
- Exact norm placements of OLMo 2 / Gemma 3 / Gemma 4: check against Raschka §2.1, §3.2 and the Gemma 3 report (`llm_gemma2025_gemma3.txt`).

---

### 90 · `llm.params-flops` · Counting parameters and FLOPs for real transformer configs
core · wave 1 · prereqs: `llm.transformer-block` · cross-area: `sys.flops-mfu` (owns $2mkn$, backward = 2× forward, $6N$, MFU/HFU, training-time estimates), `sys.memory-anatomy` and `sys.activation-checkpointing` (own bytes/param, activation memory, recompute)

Scope after review (ownership rule): this lesson recaps the generic results in one card with links, then spends its budget on what is
transformer-specific — per-layer counts for GQA + SwiGLU blocks, the attention-FLOPs term and its competing conventions, and worked numbers
for Llama-3 configs. MFU, days-to-train, bytes/param and activation memory are taught in the `sys.*` lessons and only linked here.

**Primary sources**
- Kaplan et al. 2020 §2.1 and Table 1 (per-op params and forward FLOPs; $N=2d_{model}n_{layer}(2d_{attn}+d_{ff})$; $C_{fwd}\approx2N+2n_{layer}n_{ctx}d_{attn}$; $C\approx6N$ per training token; the "$d_{model}>n_{ctx}/12$" remark) (pdf p.6–7). https://arxiv.org/abs/2001.08361 · `llm_kaplan2020_scaling.txt`
- JAX Scaling Book Part 4: *Counting Dots*, *Global FLOPs and Params Calculation* (MLPs, Attention, General rule of thumb, Fractional cost of attention with context length). `llm_scalingbook_transformers.txt`
- Llama 3 Table 3 (pdf p.7) for the worked config; §3.2 (405B trained with $3.8\times10^{25}$ FLOPs on 15.6T tokens; pdf p.1, 7). `llm_dubey2024_llama3.txt`
- EleutherAI, *Transformer Math 101* (2023) for the cross-check of $C\approx6PD$. `llm_eleuther2023_transformer_math.txt`

**Subtopic map**
1. *Recap card (link `sys.flops-mfu`):* a matmul costs $2mkn$ FLOPs; backward ≈ 2× forward; so $\approx6$ FLOPs per parameter per training token ⇒ $C\approx6ND$. State it, give the one-line reason, and link — don't re-derive.
2. *Per-layer parameter count:* attention $d\cdot(h d_h) + 2d\cdot(h_{kv}d_h) + (h d_h)\cdot d$; FFN $2d d_{ff}$ (or $3d d_{ff}$ gated); norms $O(d)$. Worked: Llama-3-70B ($L=80,d=8192,h=64,h_{kv}=8,d_h=128,d_{ff}=28672$, vocab 128,256 in the released config vs "128,000" in Table 3 — use the table's rounding with a note): attention ≈151.0M, FFN ≈704.6M, per layer ≈855.7M, ×80 ≈68.45B; untied embeddings + head ≈2×1.05B ⇒ ≈70.55B total (Python-verified). Contrast with the textbook $12Ld^2+Vd$ (MHA, $4d$ MLP; `sys.memory-anatomy` item 3): GQA shrinks the K/V projections 8×, SwiGLU at $3.5d$ grows the MLP.
3. *Forward FLOPs per token:* $\approx 2N_{\text{non-emb}}$ plus the attention-score term, plus the LM head $2dV$. Kaplan's $2n_{layer}n_{ctx}d_{attn}$: for one token at position $t$, $QK^\top$ and $AV$ each cost $2t\,d_{attn}$, i.e. $4t\,d_{attn}$ per layer; averaged over causal positions (mean $t=n_{ctx}/2$) it is $2n_{ctx}d_{attn}$ (Kaplan Table 1, row "Attention: Mask"). Note GQA does *not* reduce score FLOPs (they scale with query heads $h$). Worked (Python-verified): 70B, 8k context — linear ≈137 GFLOPs/token vs attention ≈10.7 GFLOPs/token *averaged over an 8k causal sequence* ($2Ln_{ctx}hd_h$); the single token at position 8k pays ≈21.5 GFLOPs.
4. *When attention matters (reconcile the constants):* with a non-gated $4d$ MLP, attention FLOPs equal linear FLOPs at $n=6d$ (non-causal) or $n=12d$ (causal average — Kaplan's $n_{ctx}/12$); the Scaling Book's $T>8D$ assumes a gated MLP with $F=4D$ and non-causal attention. Same convention table as `llm.self-attention`'s pitfall; the point is that every quoted crossover carries its convention.
5. *Total training compute:* worked: Llama-3-405B, $15.6$T tokens: $6\cdot405\text{e9}\cdot15.6\text{e12}\approx3.79\times10^{25}$, matching the paper's $3.8\times10^{25}$ (Python-verified); add the attention term for 8k sequences to show it is a few-percent correction at this width.
6. *MoE adjustment:* use active params for FLOPs, total params for memory (forward pointer to `llm.moe`).
7. *Inference per token (forward pointer to `llm.kv-cache`):* $2N$ FLOPs per generated token; every decode step reads all weights once.
8. *Memory, one summary card (link `sys.memory-anatomy`):* weights 2 bytes/param in bf16; mixed-precision Adam ≈16 bytes/param before activations ⇒ 70B full fine-tune ≈1.12 TB; activations scale with $B\cdot T\cdot d\cdot L$. Numbers only; the derivations live in `sys.memory-anatomy`.
9. *Vocabulary share:* embedding + head $=2Vd$; worked: Llama-3-8B, $2\cdot128{,}256\cdot4096\approx1.05$B of ≈8.0B (≈13%); for a 125M model with a 50k vocab and $d=768$, tied embeddings are ≈38.6M (≈31%). Links to the vocab-size trade-off in `llm.tokenization`.

**Figures**
- Stacked bar of parameter share (embeddings, attention, FFN) for a 125M, 8B and 70B model — reader sees embeddings dominate small models.
- Attention FLOP fraction vs context length for $d=4096$ and $d=8192$, causal-averaged, with the $n=12d$ crossover marked.
- Per-layer parameter breakdown of a Llama-3-70B block (Q, K, V, O, gate, up, down) as bars — reader sees GQA's tiny K/V and the MLP's dominance.

**Question ideas**
- Compute: params of a block with $d=4096$, 32 heads, 8 KV heads, SwiGLU 14,336 (Llama-3-8B layer: ≈218.1M without norms; Python-verified).
- Derivation step: why Kaplan's attention term is $2n_{ctx}d_{attn}$ and not $4n_{ctx}d_{attn}$.
- Which is false: "6ND counts the attention score FLOPs."
- Which is false: "Switching from MHA to GQA-8 reduces attention-score FLOPs by 8×." (False: only the K/V projections and cache shrink.)
- Predict: doubling context from 8k to 16k for a 7B model — % change in FLOPs/token.
- Open: "Estimate from scratch the parameter count and training FLOPs of Llama-3-70B on 15T tokens, stating every convention." (Days-to-train and MFU questions belong to `sys.flops-mfu`.)

**Pitfalls / checks**
- Llama 3 vocabulary: Table 3 says 128,000; the released tokenizer has 128,256 entries. Use one and say which.
- Conventions differ (FLOPs vs MACs; whether embeddings count in $N$; Kaplan excludes embeddings; causal vs non-causal attention). Make conventions explicit in every worked number.
- The 125M example assumes GPT-2-small shapes ($V=50{,}257$, $d=768$, tied); recompute if using another config.

---

### 100 · `llm.training-stability` · Initialization and training instabilities
intermediate · wave 2 · prereqs: `llm.transformer-block`, `llm.params-flops` (cross-area: `sys.stability-tricks`, `sys.gradient-clipping`, `sys.tuning-diagnostics`)

Scope split with `sys.stability-tricks` (Applied plan): that lesson owns z-loss, soft-capping, QK-norm and Adam ε as *numerics*; this lesson owns the *training-recipe* view (what fails in LLM pretraining, which model reports adopted which fix and why, init, data-side spikes, MuonClip, mHC) and links to it for the numerical derivations. Gradient clipping → `sys.gradient-clipping`; tuning workflow → `sys.tuning-*`.

**Primary sources**
- Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities*: §3.1 attention-logit growth and qk-layernorm, §3.2 output-logit divergence and z-loss ($10^{-4}\log^2Z$), LR sensitivity metric, §3.4+ interventions (warmup, weight decay, μParam, AdamW ε). https://arxiv.org/abs/2309.14322 · `llm_wortsman2023_instabilities.txt`
- Chowdhery 2022 (PaLM) §5 Training Setup: z-loss $10^{-4}\log^2 Z$ (p.10), loss spikes and the rewind-and-skip-batches fix (§5.1). `llm_chowdhery2022_palm.txt`
- Dehghani et al. 2023 (ViT-22B) §2 QK normalization motivation (attention logit divergence). https://arxiv.org/abs/2302.05442 · `llm_dehghani2023_vit22b.txt`
- Gemma 2 report §2 logit soft-capping (50 for attention, 30 for final logits) (p.2). https://arxiv.org/abs/2408.00118 · `llm_team2024_gemma2.txt`
- OLMo 2 §3 Pretraining stability: §3.1 repeated n-grams, §3.2 initialization, §3.3 RMSNorm/reordered norm/QK-norm/z-loss, §3.4 weight decay on embeddings (p.12–17). https://arxiv.org/abs/2501.00656 · `llm_olmo2024_olmo2.txt`
- Kimi K2 report: MuonClip / QK-Clip (p.1–5). https://arxiv.org/abs/2507.20534 · `llm_kimi2025_k2.txt`; Xie et al. 2025 mHC (hyper-connections constrained to doubly-stochastic mixing). https://arxiv.org/abs/2512.24880 · `llm_xie2025_mhc.txt`
- GPT-2 §2.3 (residual-layer weights scaled by $1/\sqrt{N}$ at init, p.4). `llm_radford2019_gpt2.txt`

**Subtopic map**
1. *What instability looks like:* loss spikes, divergence, slow recovery; why it matters more at scale (cost of a rewind). Proxies: Wortsman's LR-sensitivity curves at small scale.
2. *Initialization:* std 0.02 (GPT-2 style) vs $1/\sqrt{d}$ fan-in; residual-branch output scaling $1/\sqrt{2L}$ (derive: variance of the residual stream after $2L$ additions stays O(1)); small-init embeddings; why output layer init matters (logits at init should be near uniform, loss ≈ $\ln V$).
3. *Attention logit growth (recipe view):* $q\cdot k$ grows as weights grow → softmax saturates (entropy collapse); seen in ViT-22B and in Wortsman's proxies. Fix adopted: QK-norm (OLMo 2, Gemma 3, Qwen3), applied before RoPE. The numerics (why the logit bound holds, learnable scale) are derived in `sys.stability-tricks` item 1 — link, one-sentence recap only.
4. *Output logit divergence (recipe view):* $\log Z$ drifts; PaLM and Wortsman add z-loss with $\lambda=10^{-4}$; OLMo 2 keeps it. The gradient derivation and worked numbers are in `sys.stability-tricks` item 2 — link, don't repeat.
5. *Logit soft-capping (who and why):* Gemma 2 caps attention logits at 50 and final logits at 30 (Gemma 2 §2, pdf p.2); Gemma 3 replaced attention soft-capping with QK-norm (Gemma 3 §2, pdf p.2), partly because additive/tanh logit transforms complicate fused attention kernels. The function and its gradient are in `sys.stability-tricks` item 3.
6. *Optimizer-side choices in published recipes:* β2 = 0.95 rather than 0.999 (faster adaptation to gradient-scale changes), small AdamW ε (Wortsman §3.4: ε near the gradient RMS shrinks updates), clipping at 1.0, warmup, weight decay on/off for embeddings (OLMo 2 §3.4). One line each with the model that uses it; the generic protocols are in `sys.tuning-diagnostics` (warmup, LR sweep), `sys.gradient-clipping` and `fund.adam`.
7. *Data-side spikes:* repeated n-grams and bad batches (OLMo 2 §3.1); PaLM's rewind ~100 steps and skip 200–500 batches.
8. *Precision (pointer only):* bf16 vs fp16 and fp32 accumulation live in `sys.mixed-precision`; FP8 training in `sys.fp8-training`.
9. *Muon and QK-Clip (2025):* Muon = orthogonalized momentum update via Newton–Schulz; K2's QK-Clip rescales $W_Q,W_K$ when max logit exceeds a threshold — same goal as QK-norm without changing the forward pass. Why it exists: K2 notes QK-norm is not applicable to MLA because the keys are never fully materialized at inference (absorption), so K2 clips weights instead, and for MLA only the unshared head-specific components (K2 §2.1, pdf p.3). Muon itself is taught in `llm.pretraining-optimization`.
10. *Residual-stream widening (2024–26, contested):* hyper-connections widen the residual stream into $n$ parallel streams mixed by learned matrices, which breaks the identity map; DeepSeek's mHC projects the residual mixing matrix onto the doubly-stochastic matrices (Birkhoff polytope) with Sinkhorn–Knopp, restoring an identity-like, norm-preserving map (Xie et al., arXiv 2512.24880 §1, §3). Adopted in DeepSeek V4 with $n_{hc}=4$ and 20 Sinkhorn iterations (V4 pdf p.8, 24). Include as "frontier, verify before teaching as standard".
11. *μP as a stability tool* (pointer to `llm.mup`).

**Figures**
- Simulated attention entropy vs training step with and without QK-norm (toy: weights scaled up over time).
- Init-variance experiment (numpy): residual-stream RMS vs depth with and without the $1/\sqrt{2L}$ output scaling — reader sees growth ∝ $\sqrt{L}$ vs O(1).
- Timeline of a loss spike with PaLM's rewind-and-skip fix (schematic, labelled), annotated with the ~100-step rewind and 200–500 skipped batches (PaLM §5.1, pdf p.11).

**Question ideas**
- Derivation: residual-branch scaling — show that $2L$ independent unit-variance additions need a $1/\sqrt{2L}$ factor to keep the stream's variance O(1).
- Compute: initial loss of a randomly initialised 128k-vocab model (≈ $\ln 128000$ ≈ 11.76).
- Predict: removing QK-norm and raising LR 3× in a small proxy model (Wortsman Fig. 1 style).
- Which is false: "Soft-capping changes the argmax of the logits."
- Compare: QK-norm vs QK-Clip.

**Pitfalls / checks**
- PaLM rewind details verified (PaLM §5.1, pdf p.11): restart ~100 steps before the spike and skip ~200–500 data batches.
- Gemma 3 replaced Gemma 2's soft-capping with QK-norm (verified: Gemma 3 report §2, pdf p.2).
- mHC/hyper-connections: recent; present as a development, not interview canon.
## Unit D: Tokenization

### 110 · `llm.tokenization` · Subword tokenization: BPE, WordPiece, Unigram
core · wave 1 · prereqs: none

**Primary sources**
- SLP3 ch. 2 §2.3 Unicode (code points, UTF-8), §2.4 Subword Tokenization: BPE — §2.4.1 training, §2.4.2 encoder, §2.4.3 in practice; §2.6.9 regexes for BPE pre-tokenization (pdf p.7–15, 23). https://web.stanford.edu/~jurafsky/slp3/2.pdf · `llm_slp3_ch2.txt`
- Sennrich, Haddow & Birch 2015, §3.2 Byte Pair Encoding (merge algorithm, Algorithm 1; p.3–4). https://arxiv.org/abs/1508.07909 · `llm_sennrich2015_bpe.txt`
- Radford et al. 2019 (GPT-2) §2.2 Input Representation: byte-level BPE, 256 base symbols, no merges across character categories (p.4). `llm_radford2019_gpt2.txt`
- Kudo 2018, *Subword Regularization*, §3.2 unigram LM (EM, Viterbi, pruning), §3.3 subword sampling (p.3–4). https://arxiv.org/abs/1804.10959 · `llm_kudo2018_unigram.txt`
- Kudo & Richardson 2018, *SentencePiece* §3.1 lossless tokenization (whitespace as ▁), §3.2 (p.2–3). https://arxiv.org/abs/1808.06226 · `llm_kudo2018_sentencepiece.txt`
- Bostrom & Durrett 2020, *BPE is Suboptimal for LM Pretraining* (unigram vs BPE segmentations). https://arxiv.org/abs/2004.03720 · `llm_bostrom2020_bpe_suboptimal.txt`

**Subtopic map**
1. *Problem:* word-level vocabularies → OOV and huge tables; characters → very long sequences ($O(n^2)$ attention, more decode steps). Subwords trade these off. Define compression (chars or bytes per token) and fertility (tokens per word).
2. *Unicode and bytes:* code points vs UTF-8 bytes (1–4 bytes each); why byte-level base vocab (256) guarantees no OOV.
3. *BPE training (worked example):* start from characters/bytes; count adjacent pairs; merge the most frequent; repeat until the vocabulary target. Do 3–4 merges by hand on a toy corpus ("low lower lowest newer wider"), Python-checked.
4. *BPE encoding:* apply merges in learned order (not greedy longest match); complexity; determinism; the same string can tokenize differently with a leading space.
5. *Pre-tokenization:* regex splitting (GPT-2/tiktoken patterns: contractions, letters, digits, whitespace) prevents merges across categories; digit grouping choices (single digits, up to 3 digits).
6. *WordPiece:* merge the pair maximizing $\frac{c(ab)}{c(a)c(b)}$ (likelihood gain) rather than raw frequency; `##` continuation marker; greedy longest-match encoding.
7. *Unigram LM (derive):* model a segmentation $\mathbf{x}=(x_1..x_M)$ with $p(\mathbf{x})=\prod p(x_i)$; best segmentation by Viterbi; train by EM over latent segmentations starting from a large seed vocabulary, then prune tokens whose removal least reduces likelihood. Subword regularization: sample segmentations during training (and BPE-dropout analogue).
8. *SentencePiece:* language-agnostic, treats input as raw Unicode, whitespace as `▁`, lossless detokenization; implements BPE or unigram.
9. *Vocabulary size trade-offs (an interview favourite; make it quantitative):* embedding + unembedding params $2Vd$ (large share for small models: ≈13% of Llama-3-8B, ≈31% of a GPT-2-small-shaped model — worked in `llm.params-flops` item 9), LM-head cost $2Vd$ FLOPs per token and a $B\times T\times V$ logit tensor, sequence length shrinks with larger $V$ (fewer FLOPs and decode steps per character; more text per context window), rare tokens get few gradient updates (under-trained embeddings, `llm.tokenization-effects`), and multilingual fertility improves with more vocab. Worked: Llama 2 → Llama 3 (32k → 128k) raised English compression 3.17 → 3.94 chars/token, i.e. ≈20% fewer tokens for the same text, for ≈0.8B extra embedding+head parameters at $d=4096$ (Python-checked). Typical sizes: 32k (Llama 2), 128k (Llama 3: 100k tiktoken + 28k multilingual), 256k (PaLM, Gemma). Llama 3 compression 3.17 → 3.94 chars/token on English (Table/§3.2).
10. *Special tokens and chat templates:* BOS/EOS, padding, role markers; tokenizer and template mismatches as a real bug source.

**Figures**
- BPE merge trace on the toy corpus: tokens after each merge, with the merged pair highlighted.
- Unigram lattice for one word with candidate segmentations and the Viterbi path.
- Tokens per 1,000 characters vs vocabulary size (synthetic but generated by actually training BPE on a small text with the `tokenizers`-free pure-Python implementation).

**Question ideas**
- Compute: next merge given pair counts.
- Which is false: "BPE encoding picks the longest vocabulary match at each position."
- Predict: growing vocab from 32k to 128k for a 1B model — effect on params, tokens per document, FLOPs per character.
- Compare: unigram vs BPE when the same word appears with different morphology.
- Open: "Explain how a unigram tokenizer is trained and why it can sample segmentations."

**Pitfalls / checks**
- WordPiece's exact criterion is under-documented (Google never published code); present the likelihood-ratio form as the standard description and attribute it.
- Llama 3 compression numbers from Llama 3 §3.2 (p.7).

---

### 120 · `llm.tokenization-effects` · What tokenization does to models (and tokenizer-free models)
intermediate · wave 2 · prereqs: `llm.tokenization`, `llm.pretraining-objective`

**Primary sources**
- Singh & Strouse 2024, *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*: left-to-right vs right-to-left 3-digit chunking (§3, p.3–4). https://arxiv.org/abs/2402.14903 · `llm_singh2024_tokenization_arithmetic.txt`
- Petrov et al. 2023, *Language Model Tokenizers Introduce Unfairness Between Languages*: tokenizer parity, cost/latency/context implications (§3–5, p.3–8). https://arxiv.org/abs/2305.15425 · `llm_petrov2023_tokenizer_unfairness.txt`
- Land & Bartolo 2024, *Fishing for Magikarp*: under-trained/"glitch" tokens and detection via embedding norms. https://arxiv.org/abs/2405.05417 · `llm_land2024_magikarp.txt`
- Xue et al. 2021 (ByT5) and Pagnoni et al. 2024 (Byte Latent Transformer) §2 patching schemes, §2.3 entropy patching, §3 architecture (local encoder/decoder + latent global transformer) (p.2–6). https://arxiv.org/abs/2105.13626 · `llm_xue2021_byt5.txt`; https://arxiv.org/abs/2412.09871 · `llm_pagnoni2024_blt.txt`
- DeepSeek-V3 §4.1 (tokenizer with tokens combining punctuation and line breaks; random splitting to reduce boundary bias; p.21–22). `llm_deepseek2024_v3.txt`

**Subtopic map**
1. *Arithmetic:* inconsistent digit chunking (e.g. "1234" → "123"+"4" vs "1"+"234"); L2R vs R2L chunking and error patterns; single-digit tokenization (Llama, PaLM) as a fix.
2. *Character-level tasks:* spelling, counting letters, reversing — the model never sees characters inside a token.
3. *Multilinguality:* fertility differs by language/script (up to large factors); cost, latency and effective context penalties; vocabulary allocation; byte fallback.
4. *Under-trained tokens:* tokens present in the tokenizer corpus but rare in training data; anomalous behaviour; detection by embedding statistics.
5. *Token boundary artefacts:* prompt ending with a space; "token healing"; DeepSeek-V3's random splitting of combined punctuation tokens.
6. *Comparing models across tokenizers:* perplexity per token is not comparable; bits-per-byte / bits-per-character (derive conversion: $\text{BPB}=\frac{\text{total NLL in nats}}{\ln 2\cdot\#\text{bytes}}$).
7. *Compression ↔ compute:* better compression = more text per FLOP; but compression is not the only goal (Bostrom: unigram segmentations align better with morphology).
8. *Tokenizer-free models:* ByT5 (bytes, longer sequences, deeper encoder), MegaByte-style patching, BLT entropy patching (patch boundaries where next-byte entropy is high; compute allocated adaptively); trade-offs and current status (research, not yet mainstream).
9. *Changing a tokenizer after pretraining:* vocabulary extension and embedding initialization (mean of sub-token embeddings); why it is hard.

**Figures**
- Bar chart: tokens per sentence for the same sentence in several languages under a toy English-trained BPE (generated by training a small BPE on English text and encoding translations; or use numbers from Petrov Table with citation).
- Digit chunking L2R vs R2L for "1234567", aligned with place values.

**Question ideas**
- Compute: convert 2.0 nats/token to bits-per-byte at 4 bytes/token.
- Predict: switching digits to single-digit tokens — effect on sequence length and arithmetic accuracy.
- Which is false: "Two models with the same per-token perplexity on a corpus have the same bits-per-byte."
- Spot the flaw: comparing Llama 2 and Llama 3 by validation perplexity.

**Pitfalls / checks**
- Fertility ratios differ by tokenizer and corpus; quote specific numbers only with their source table.
- BLT claims (matching BPE at scale) are from the paper's own experiments; flag as recent.

## Unit E: Pretraining

### 130 · `llm.pretraining-objective` · Next-token prediction, perplexity and compression
core · wave 1 (promoted in review: it is a hard prereq of wave-1 `llm.sampling`, `llm.scaling-laws`, `llm.sft` and `llm.policy-gradients`) · prereqs: `llm.causal-attention`

**Primary sources**
- SLP3 ch. 3 §3.3 perplexity, §3.3.1 weighted branching factor, §3.7 perplexity and entropy (pdf p.9–10, 18–21). `llm_slp3_ch3.txt`; ch. 7 §7.7 Pretraining Transformer LLMs, §7.7.1 perplexity (pdf p.25–26). `llm_slp3_ch7.txt`
- Delétang et al. 2023, *Language Modeling Is Compression* §2 (arithmetic coding; log-loss = code length). https://arxiv.org/abs/2309.10668 · `llm_deletang2023_compression.txt`
- Gloeckle et al. 2024, *Better & Faster LLMs via Multi-token Prediction* §2 method (shared trunk, $n$ heads), §3.1–3.2 (benefits grow with size; self-speculative decoding), §5 speculation on why (p.2–7). https://arxiv.org/abs/2404.19737 · `llm_gloeckle2024_mtp.txt`
- DeepSeek-V3 §2.2 Multi-Token Prediction (sequential MTP modules keeping the causal chain; λ weights; p.10–11). `llm_deepseek2024_v3.txt`
- Shared cache for MLE/KL background: d2l information theory (`d2l/d2l_information_theory.txt`), PML1 (`pml1.txt`).

**Subtopic map**
1. *Objective:* minimize $-\frac1T\sum_t\log p_\theta(x_t\mid x_{<t})$ = MLE; equals cross-entropy $H(p_{data},p_\theta)=H(p_{data})+\mathrm{KL}(p_{data}\|p_\theta)$; the entropy term is the irreducible loss (connects to $E$ in Chinchilla's law).
2. *Logits → softmax → CE gradient:* $\partial\ell/\partial z = p - e_y$ (derive); the LM head cost $O(Vd)$ per token; memory of $B\times T\times V$ logits and chunked/fused CE.
3. *Perplexity:* $\exp$ of mean NLL; weighted branching factor interpretation; worked example (uniform over $k$ ⇒ PPL $=k$).
4. *Bits per token / byte / character:* conversions; why BPB is tokenizer-agnostic (forward link to `llm.tokenization-effects`).
5. *LM = compressor:* arithmetic coding achieves code length ≈ $-\log_2 p$; total bits = cumulative log-loss; implications ("prequential" view; compression rates of LLMs on text/images in Delétang).
6. *Sequence packing and loss masking:* packing efficiency; document masks; masking padding and (in SFT) prompts.
7. *Why not label smoothing:* it biases the calibrated distribution; LLM pretraining uses plain CE (with z-loss as an auxiliary, see `llm.training-stability`).
8. *Multi-token prediction:* $n$ independent heads on a shared trunk (Gloeckle) vs sequential modules (DeepSeek-V3); training signal densification; benefits reported at larger sizes and on code; use as a speculative drafter at inference.
9. *What the loss does and doesn't tell you:* loss vs downstream ability; memorization; train/val gap ~0 in single-epoch regime.

**Figures**
- CE gradient picture: bar chart of $p$ and $p-e_y$ over a 6-token vocab.
- Perplexity vs NLL curve with anchors (uniform over 50k vs a good model).
- MTP architectures diagram: parallel heads (Gloeckle) vs chained modules (DeepSeek-V3).

**Question ideas**
- Compute: model gets loss 2.3 nats/token, tokenizer averages 4.2 bytes/token — BPB?
- Derivation: gradient of softmax CE wrt logits.
- Which is false: "Lower perplexity on a held-out set implies better downstream reasoning."
- Compare: Gloeckle parallel heads vs DeepSeek sequential MTP.
- Open: "Explain why next-token prediction is equivalent to compression."

**Pitfalls / checks**
- Gloeckle's gains are size- and domain-dependent (stronger on code, can hurt small models) — check §3.1/§3.7 before generalizing.

---

### 140 · `llm.pretraining-data` · Pretraining data: filtering, deduplication, mixtures
intermediate · wave 2 · prereqs: `llm.pretraining-objective`

**Primary sources**
- Penedo et al. 2024 (FineWeb) §3.2 text extraction, §3.3–3.5 base filtering and deduplication (MinHash details; per-dump vs global dedup finding), §3.6 heuristic filters, FineWeb-Edu classifier (p.3–9). https://arxiv.org/abs/2406.17557 · `llm_penedo2024_fineweb.txt`
- Lee et al. 2021, *Deduplicating Training Data Makes LMs Better* §4.1 exact substring (suffix arrays) and §4.1.2 MinHash approximate matching (Jaccard) (p.3–5). https://arxiv.org/abs/2107.06499 · `llm_lee2021_dedup.txt`
- Li et al. 2024 (DCLM) — model-based quality filtering (fastText classifier) benchmark. https://arxiv.org/abs/2406.11794 · `llm_li2024_dclm.txt`
- Xie et al. 2023 (DoReMi) §2 minimax domain reweighting with a proxy model (p.3–5). https://arxiv.org/abs/2305.10429 · `llm_xie2023_doremi.txt`
- Muennighoff et al. 2023, *Scaling Data-Constrained LMs* §3 (effective data with repetition; parametric fit), §6–7 (p.3–9). https://arxiv.org/abs/2305.16264 · `llm_muennighoff2023_data_constrained.txt`
- Llama 3 §3.1 pretraining data (web curation, dedup levels, heuristic and model-based filters, data mix, annealing on high-quality data). `llm_dubey2024_llama3.txt`; OLMo 2 §4 mid-training (Dolmino mix, annealing, checkpoint soups; p.18–25). `llm_olmo2024_olmo2.txt`

**Subtopic map**
1. *Pipeline overview:* crawl (Common Crawl WARC/WET) → HTML text extraction (trafilatura vs WET) → language ID → URL/blocklists → heuristic filters (Gopher/C4 rules: length, symbol ratios, repetition) → deduplication → quality classifiers → PII/toxicity → tokenization.
2. *Deduplication:* exact (hash, suffix-array substring), near-duplicate with MinHash + LSH. Derive: $P(\text{minhash match})=J(A,B)$ for a random permutation; with $b$ bands of $r$ rows, $P(\text{candidate})=1-(1-J^r)^b$ (an S-curve; worked numbers). Why dedup helps (less memorization, better perplexity per FLOP), FineWeb's finding that per-snapshot dedup beat global dedup.
3. *Model-based quality filtering:* classifiers trained on reference "good" text (Wikipedia-linked, OpenHermes/ELI5 in DCLM) or LLM-annotated educational value (FineWeb-Edu); risk of narrowing diversity.
4. *Data mixtures:* domain weights (web, code, math, books, multilingual); upsampling; DoReMi's minimax excess-loss reweighting; heuristic mix search with small proxy models (Llama 3).
5. *Repetition and data constraints:* Muennighoff — up to ~4 epochs ≈ as good as fresh data, value of repeated tokens decays (effective-data formula $D'=U_D+U_DR^*_D(1-e^{-R_D/R^*_D})$; check notation in §3.1), filling with code.
6. *Annealing / mid-training:* upsample high-quality and task-like data during the LR decay; checkpoint soups (OLMo 2); why the decay phase is where data quality matters most (link to WSD in `llm.pretraining-optimization`).
7. *Synthetic data:* rephrased web text, textbook-style data; model-collapse concerns (contested at scale).
8. *Contamination:* benchmark leakage into crawls; n-gram decontamination (link `llm.evaluation-pitfalls`).
9. *Data for long context, code, multilinguality* (brief pointers).

**Figures**
- MinHash LSH S-curve: $1-(1-J^r)^b$ vs $J$ for $(b,r)=(20,5),(14,8)$ (recheck the FineWeb settings).
- Pipeline funnel: tokens remaining after each stage (illustrative, labelled as such, or FineWeb's numbers with citation).
- Data mixture pie for Llama 3 (≈50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual; Llama 3 §3.1.2).

**Question ideas**
- Compute: probability two docs with Jaccard 0.8 become LSH candidates with $b=20,r=5$.
- Predict: training 4 epochs on 250B unique tokens vs 1 epoch on 1T tokens (Muennighoff).
- Which is false: "MinHash similarity estimates cosine similarity between TF-IDF vectors."
- Compare: heuristic filters vs classifier filters: failure modes.

**Pitfalls / checks**
- FineWeb MinHash parameters (n-gram size, number of hashes, bands) — read §3.4 before quoting.
- Llama 3 mixture verified (§3.1.2 data-mix summary): ~50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual.

---

### 150 · `llm.pretraining-optimization` · Optimizers, LR schedules and batch size for LLMs
intermediate · wave 2 · prereqs: `llm.pretraining-objective` (cross-area: `sys.tuning-batch-size`, `sys.tuning-steps-schedules`, `fund.adam`)

Scope split: the generic tuning workflow, batch-size and schedule tuning live in `sys.tuning-batch-size`, `sys.tuning-steps-schedules` (`content-plan-tuning.md`); this lesson keeps the LLM-pretraining specifics (β2=0.95, WSD and cooldowns in published recipes, critical batch size and batch ramps in LLM reports, Muon in K2/V4) and links there.

**Primary sources**
- Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*: §2 cosine background, §3 constant LR + cooldown (WSD), cooldown length/shape, §4 SWA/schedule-free, §5 implications for scaling-law research (p.2–9). https://arxiv.org/abs/2405.18392 · `llm_hagele2024_wsd.txt`
- Hu et al. 2024 (MiniCPM) WSD scheduler section and batch/LR experiments. https://arxiv.org/abs/2404.06395 · `llm_hu2024_minicpm.txt`
- McCandlish et al. 2018, *An Empirical Model of Large-Batch Training* §2.2 gradient noise scale, $B_{\text{simple}}=\mathrm{tr}(\Sigma)/|G|^2$, §2.3 time/data trade-off $(S/S_{min}-1)(E/E_{min}-1)=1$ (p.5–8). https://arxiv.org/abs/1812.06162 · `llm_mccandlish2018_critical_batch.txt`
- DeepSeek LLM 2024 §3.1 scaling laws for batch size and LR vs compute (p.8–9). https://arxiv.org/abs/2401.02954 · `llm_bi2024_deepseek_llm.txt`
- DeepSeek-V3 §4.2 training hyper-parameters (AdamW β=(0.9,0.95), wd 0.1, LR warmup→constant→cosine→constants, batch ramp 3072→15360 sequences; p.22–23). `llm_deepseek2024_v3.txt`
- Keller Jordan 2024, *Muon* blog (Newton–Schulz orthogonalization). https://kellerjordan.github.io/posts/muon/ · `llm_jordan2024_muon.txt`; Liu et al. 2025, *Muon is Scalable for LLM Training* §2.2 (weight decay, update-RMS matching to AdamW), §3.2 scaling law (p.3–7). https://arxiv.org/abs/2502.16982 · `llm_liu2025_muon.txt`
- Shared cache: AdamW (`papers/loshchilov2017_adamw.txt`), Adam (`papers/kingma2014_adam.txt`), d2l LR schedules (`d2l/d2l_lr_scheduler.txt`).

**Subtopic map**
1. *AdamW recap* (pointer to fundamentals area) and LLM-specific settings: β2=0.95 (why: faster tracking of changing gradient scale, fewer spikes), ε, decoupled weight decay 0.1, gradient clipping 1.0, no dropout in modern pretraining (single-epoch regime).
2. *Warmup in LLM recipes:* typical lengths in published runs (1–2k steps; DeepSeek-V3's 2k) and why transformers need it (noisy second-moment estimates early; Post-LN gradients). The generic warmup protocol is in `sys.tuning-diagnostics`; one line + link.
3. *Cosine decay to ~10%:* requires knowing the total steps in advance; consequence for scaling-law experiments (Chinchilla's correction of Kaplan partly came from this).
4. *WSD / trapezoidal:* warmup–stable–decay; loss plateaus during the stable phase and drops sharply during cooldown; can branch cooldowns from one run to get many training lengths cheaply; cooldown length (~10–20%) and shape (1−sqrt) (Hägele §3); DeepSeek-V3's multi-stage schedule as an example. Why decay helps at all (the noise-floor argument $\approx\eta\sigma^2/(4B)$) is derived in `sys.tuning-steps-schedules` item 2 — recap in one sentence; the river-valley picture is a heuristic, label it so.
5. *Batch size in LLM reports:* the gradient-noise-scale derivation ($B_{noise}$, $B_{simple}$, the $(S/S_{min}-1)(E/E_{min}-1)=1$ trade-off) belongs to `sys.tuning-batch-size` items 3–4 — recap the result in two sentences and link. LLM-specific content: Kaplan's $B_{crit}(L)$ power law (critical batch grows as loss falls; Kaplan §5.1, pdf p.5) as the reason recipes ramp the batch — DeepSeek-V3 3072 → 15360 sequences over the first 469B tokens (§4.2, pdf p.23); Llama 3 4M → 8M → 16M tokens (pdf p.14).
6. *Scaling of LR and batch with compute* (DeepSeek LLM power laws) vs μP (pointer to `llm.mup`).
7. *Muon (derive the core idea; recent and much-discussed in 2026):* for a weight matrix the natural norm is the spectral (operator) norm; the steepest-descent step under a spectral-norm constraint is $-\eta\,UV^\top$ where $M=U\Sigma V^\top$ is the momentum matrix (all singular values set to 1 — "orthogonalized momentum"); computing $UV^\top$ exactly needs an SVD, so Muon runs ~5 Newton–Schulz iterations of a tuned quintic $\varphi(x)=ax+bx^3+cx^5$, $(a,b,c)=(3.4445,-4.7750,2.0315)$ (Jordan blog; Liu 2025 §2, pdf p.2–3). Only 2-D hidden matrices; embeddings, LM head and norm gains stay on AdamW. Scaling fixes from Liu 2025 §2.2: add weight decay (weight/output RMS otherwise grows) and match the update RMS to AdamW's so AdamW hyperparameters transfer. Adopted by Kimi K2 (MuonClip) and DeepSeek V4. Present as recent and still being evaluated.
8. *Mixed precision:* pointer only (`sys.mixed-precision`).
9. *Checkpoint averaging / EMA / SWA* as an alternative to decay (Hägele §4).

**Figures**
- LR vs step for cosine, WSD and DeepSeek-V3's schedule (piecewise, from §4.2 numbers).
- Loss vs tokens for cosine vs WSD with branching cooldowns (schematic but generated from a simple noisy-quadratic simulation).
- Muon picture: singular values of a momentum matrix before and after 5 Newton–Schulz steps (numpy) — reader sees them pushed toward 1.

**Question ideas**
- Predict: stop a cosine run at 50% of planned steps vs a WSD run cooled down at that point.
- Derivation step: why setting all singular values of the momentum to 1 is the steepest-descent direction under a spectral-norm step-size constraint.
- Which is false: "Muon replaces AdamW for embedding and output layers."
- Compare: why a scaling-law study using one cosine schedule for all horizons is biased.

**Pitfalls / checks**
- Llama 3 batch schedule verified (pdf p.14): 4M tokens (sequence length 4,096) → 8M (8,192-token sequences) after 252M tokens → 16M after 2.87T tokens.
- Muon specifics (Newton–Schulz coefficients, which params) — follow the blog/paper exactly.
## Unit F: Scaling laws and hyperparameter transfer

### 160 · `llm.scaling-laws` · Scaling laws and compute-optimal training
core · wave 1 · prereqs: `llm.params-flops`, `llm.pretraining-objective`

**Primary sources**
- Kaplan et al. 2020 §1.1–1.2 summary of power laws $L(N)$, $L(D)$, $L(C_{min})$; §2 compute accounting; §6 optimal allocation ($N_{opt}\propto C^{0.73}$) (p.2–5, 6–7, 13–18). https://arxiv.org/abs/2001.08361 · `llm_kaplan2020_scaling.txt`
- Hoffmann et al. 2022 (Chinchilla) §3.1 Approach 1 (fix $N$, vary $D$; envelope), §3.2 Approach 2 (IsoFLOP profiles; $a=0.49,b=0.51$), §3.3 Approach 3 (parametric fit $E=1.69,A=406.4,B=410.7,\alpha=0.34,\beta=0.28$; $a=0.46,b=0.54$), Table 2–3 (p.4–8); App. D.2 derivation of the closed form (p.~25). https://arxiv.org/abs/2203.15556 · `llm_hoffmann2022_chinchilla.txt`
- Besiroglu et al. 2024, *Chinchilla Scaling: A replication attempt* (refit $E=1.8172, A=482.01, B=2085.43, \alpha=0.3478, \beta=0.3658$; Approach 3 inconsistency; p.1–4). https://arxiv.org/abs/2404.10102 · `llm_besiroglu2024_chinchilla_replication.txt`
- JAX Scaling Book Part 4 *General rule of thumb for Transformer FLOPs* (for $C=6ND$). `llm_scalingbook_transformers.txt`
- Llama 3 §3.2.1 Scaling Laws (IsoFLOP experiments $6\times10^{18}$–$10^{22}$ FLOPs, two-stage downstream prediction; p.7–8). `llm_dubey2024_llama3.txt`

**Subtopic map**
1. *What a scaling law is:* empirical power law $L(x)\approx (x_c/x)^{\alpha_x}$ (+ irreducible term) over many orders of magnitude; why log–log straight lines; what is held fixed (architecture family, data distribution, tuned hyperparameters).
2. *Kaplan 2020 findings:* $L(N)$, $L(D)$, $L(C)$ exponents; weak dependence on shape (depth/width) at fixed $N$; larger models are more sample-efficient; their optimal allocation $N\propto C^{0.73}$ ⇒ "grow the model, not the data".
3. *Chinchilla setup:* $C\approx6ND$; three approaches; IsoFLOP curve = loss vs $N$ at fixed $C$, minimum gives $N_{opt}(C)$.
4. *Derive compute-optimal allocation:* minimize $E+A N^{-\alpha}+B D^{-\beta}$ s.t. $6ND=C$. Substitute $D=C/(6N)$, set derivative to 0: $\alpha A N^{-\alpha}=\beta B D^{-\beta}$ ⇒ $N_{opt}=G\,(C/6)^{\beta/(\alpha+\beta)}$, $D_{opt}=G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$, $G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$. Interpretation: at the optimum the marginal loss reduction per FLOP is equal for params and data. With $\alpha\approx\beta$ both exponents ≈ 0.5 ⇒ scale $N$ and $D$ equally ⇒ a constant tokens/param ratio (~20 per Approaches 1–2).
5. *Worked numbers (Python-verified):* with Chinchilla's published Approach-3 constants at Chinchilla's budget $5.76\times10^{23}$ FLOPs the formula gives $N\approx32$B, $D\approx3.0$T (~93 tokens/param), *not* 70B/1.4T; Besiroglu's refit gives $N\approx72$B, $D\approx1.33$T (~18 tokens/param), consistent with the 20× rule. Teach this as a lesson in how fragile fitted constants are.
6. *Interpreting $E$:* irreducible loss ≈ entropy of text under this tokenizer; loss differences near $E$ are small in nats but large in capability.
7. *Practical rule:* ~20 tokens per parameter for compute-optimal *training* (Chinchilla Table 3); Chinchilla (70B, 1.4T tokens) beat Gopher (280B, ~300B tokens) at the same compute ($5.76\times10^{23}$ FLOPs).
8. *Using scaling laws in practice:* fit on small runs ($10^{18}$–$10^{22}$ FLOPs) then extrapolate; Llama 3 predicted the flagship's downstream accuracy via loss → NLL-on-task → accuracy; 405B chosen as roughly compute-optimal for $3.8\times10^{25}$ FLOPs.
9. *What scaling laws don't promise:* downstream tasks can be non-smooth (→ emergence debate in the next lesson); laws are distribution- and recipe-specific.
10. *Other axes (pointer):* data-constrained, inference-aware, MoE, test-time compute (`llm.scaling-laws-practice`, `llm.moe-systems`, `llm.test-time-compute`).

**Figures**
- IsoFLOP curves generated from the Besiroglu-fitted $L(N,D)$ at 5 compute budgets, with minima joined (the compute-optimal frontier).
- $N_{opt}$ and $D_{opt}$ vs $C$ (log–log) for Kaplan's 0.73 exponent vs Chinchilla's ~0.5.
- Bar: Gopher vs Chinchilla params and tokens at equal compute.

**Question ideas**
- Derivation step: which condition holds at the optimum (equal marginal returns)?
- Compute: compute-optimal $N$ and $D$ for $C=10^{24}$ with the 20 tokens/param rule ($6\cdot20N^2=C$ ⇒ $N\approx91$B, $D\approx1.8$T; Python-verified).
- Figure MCQ: on IsoFLOP curves, which marked point is compute-optimal for budget 3?
- Which is false: "If $\alpha>\beta$, the optimal model size grows faster than the optimal data size." (False: $N_{opt}\propto C^{\beta/(\alpha+\beta)}$, so $\alpha>\beta$ makes the $N$ exponent below 0.5 and data grows faster.)
- Spot the flaw: using Chinchilla's Approach-3 constants to justify "20 tokens per parameter".
- Open: "Derive the Chinchilla compute-optimal allocation from $L(N,D)$."

**Pitfalls / checks**
- The ~20 tokens/param rule comes from Approach 1 (Table 3: 10B ↔ 205.1B tokens, 67B ↔ 1.5T, 175B ↔ 3.7T) and Approach 2. Approach 3's published constants are inconsistent with it (Besiroglu §1/§4: the refit gives ~20 tokens/param); see item 5.
- Exponent naming: Chinchilla writes $N_{opt}\propto C^a$, $D_{opt}\propto C^b$; $a=\beta/(\alpha+\beta)$.

---

### 170 · `llm.scaling-laws-practice` · Scaling laws in practice: Kaplan vs Chinchilla, over-training, data limits, emergence
intermediate · wave 2 · prereqs: `llm.scaling-laws`

**Primary sources**
- Pearce & Song 2024, *Reconciling Kaplan and Chinchilla Scaling Laws* (non-embedding vs total params; small-scale; p.1–3). https://arxiv.org/abs/2406.12907 · `llm_pearce2024_reconcile.txt`
- Porian et al. 2024, *Resolving Discrepancies in Compute-Optimal Scaling* (three factors: last-layer FLOPs, warmup duration, scale-dependent optimizer tuning incl. AdamW β2). https://arxiv.org/abs/2406.19146 · `llm_porian2024_discrepancies.txt`
- Sardana et al. 2023, *Beyond Chinchilla-Optimal: Accounting for Inference*. https://arxiv.org/abs/2401.00448 · `llm_sardana2023_beyond_chinchilla.txt`
- Muennighoff et al. 2023 §3 data-constrained law, §5–7 allocation (p.3–9). `llm_muennighoff2023_data_constrained.txt`
- Wei et al. 2022, *Emergent Abilities of LLMs* (definition; Fig. 2). https://arxiv.org/abs/2206.07682 · `llm_wei2022_emergent.txt`; Schaeffer et al. 2023, *Are Emergent Abilities a Mirage?* §2 (metric nonlinearity argument with per-token accuracy $p^L$). https://arxiv.org/abs/2304.15004 · `llm_schaeffer2023_mirage.txt`
- Hägele 2024 §5 (WSD makes scaling-law sweeps cheaper). `llm_hagele2024_wsd.txt`

**Subtopic map**
1. *Why Kaplan and Chinchilla disagree:* (a) Kaplan used non-embedding params, which at small scale biases the exponent (Pearce); (b) fixed LR schedule length not matched to each run's horizon (models stopped early look worse) and warmup too long for small runs; (c) last-layer FLOPs; (d) optimizer settings not tuned per scale (Porian). Explain each mechanism, not just the list.
2. *Over-training for inference:* lifetime cost = training $6ND$ + inference $2N\cdot D_{inf}$; minimizing total cost for a target loss shifts to smaller $N$, larger $D$ (Sardana). Examples: Llama 3 8B on ~15T tokens (~1,900 tokens/param), Llama 3 70B (~214). Loss keeps improving log-linearly well past 20 tokens/param.
3. *Data-constrained scaling:* repeated tokens have diminishing value; up to ~4 epochs ≈ fresh; effective data/params formulas; optimal strategy when unique data is limited (more epochs and somewhat larger models; add code; light filtering).
4. *Emergence debate:* abrupt jumps on some benchmarks vs Schaeffer's argument that nonlinear/discontinuous metrics (exact match over $L$ tokens ≈ $p^L$) turn smooth per-token improvement into apparent jumps; what remains contested (some abilities still look sharp with continuous metrics; in-context learning phase changes).
5. *Predicting downstream performance:* Llama 3's two-stage method; sigmoidal accuracy vs loss.
6. *Scaling law for hyperparameters:* LR/batch vs compute power laws (DeepSeek LLM §3.1) vs μP (next lesson).
7. *Scaling laws for other things* (pointers): MoE (`llm.moe-systems`), test-time compute (`llm.test-time-compute`), reward-model overoptimization (`llm.reward-modeling`), precision/quantization laws (mention only).
8. *Methodology pitfalls:* fitting in log space vs Huber on log loss, choosing the fit range, confidence intervals (Besiroglu's critique of narrow CIs).

**Figures**
- Inference-aware optimum: total cost vs $N$ for a fixed target loss at three inference volumes.
- Schaeffer's mirage: per-token accuracy $p$ rising smoothly vs exact-match $p^{L}$ for $L=1,5,20$ — the jump appears only for large $L$.
- Effective data vs epochs (Muennighoff decay curve).

**Question ideas**
- Compute: per-token accuracy 0.9 vs 0.95 — exact-match on a 10-token answer.
- Predict: a team expects to serve $10^{13}$ tokens; should they train a Chinchilla-optimal model?
- Which is false: "Kaplan's smaller exponent for data arose because larger models were trained for too many tokens."
- Compare: repeating data 8 epochs vs adding lower-quality fresh data.

**Pitfalls / checks**
- Present emergence as contested; the mirage paper does not show that all emergence is an artefact.
- Llama 3 8B token count (~15T, §3.1/§3.4) — verify.

---

### 180 · `llm.mup` · μP and hyperparameter transfer across width
advanced · wave 3 · prereqs: `llm.scaling-laws`, `llm.pretraining-optimization`

This lesson carries the "**μP width-scaling multipliers**" reading of the user's "LLM scaling factor" item, including the $1/d$ vs $1/\sqrt d$ attention scale.

**Primary sources**
- Yang, Hu et al. 2022, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*: §1–3 (motivation; SP's LR optimum drifts with width), Table 3 μP vs SP for SGD/Adam (init variance, multipliers, per-layer LR), the transformer-specific $1/d$ attention scaling, §4 μTransfer procedure (p.1–6). https://arxiv.org/abs/2203.03466 · `llm_yang2022_mup.txt`
- EleutherAI & Cerebras, *The Practitioner's Guide to the Maximal Update Parameterization* (blog): "A Simple Approach to the μP Math" (controlled activation magnitudes, per-op derivations), coordinate check, μTransfer test. https://blog.eleuther.ai/mutransfer/ · `llm_eleuther_mutransfer.txt`
- Everett et al. 2024, *Scaling Exponents Across Parameterizations and Optimizers* (per-layer LR prescriptions; standard param with tuned per-layer LRs can match μP; ε matters). https://arxiv.org/abs/2407.05872 · `llm_everett2024_scaling_exponents.txt`
- Wortsman 2023 §3.4 (μParam as an LR-sensitivity intervention). `llm_wortsman2023_instabilities.txt`

**Subtopic map**
1. *Problem:* tuning LR/init on the target model is impossible at scale; in standard parameterization (SP) the optimal LR shifts with width, so small-model sweeps don't transfer.
2. *Desiderata:* as width $n\to\infty$, activations stay $\Theta(1)$ per coordinate at init, and *every* layer's features change by $\Theta(1)$ per step (maximal feature learning, not lazy/NTK and not exploding).
3. *Derive the core scaling (Adam, hidden matrix):* for $y=Wx$ with $x$ having $\Theta(1)$ coordinates, an update $\Delta W$ whose entries are $\Theta(\eta)$ and aligned with $x$ (low-rank gradient structure) changes $y$ by $\Theta(\eta\, n)$ ⇒ need $\eta\propto1/n$ for hidden layers. Contrast SGD (gradient entries already scale as $1/n$). Use the EleutherAI blog's step-by-step version.
4. *Output layer:* logits would blow up/grow with width; μP uses an output multiplier $\propto1/n$ (or init variance $1/n^2$) so the readout learns at the right rate.
5. *Input/embedding layer:* LR $\Theta(1)$ under Adam; init $\Theta(1)$.
6. *Attention scale:* μP uses $1/d_k$ instead of $1/\sqrt{d_k}$ because, after training, $q$ and $k$ become correlated (aligned), so $q\cdot k$ scales like $d_k$, not $\sqrt{d_k}$ (contrast with the independence argument in `llm.self-attention`).
7. *μTransfer procedure:* define base width, parameterize, sweep HPs on a small proxy, transfer to large width; coordinate check (activation RMS vs width should be flat).
8. *What transfers and what doesn't:* LR, init scale, multipliers transfer across width; depth needs extra rules (depth-μP, residual scaling $1/\sqrt{L}$ — mention), batch size and training length don't transfer automatically; regularization HPs partially.
9. *Contested/practice:* Everett 2024 — SP with per-layer LR tuned scaling can match; adoption in industry (Cerebras-GPT, some labs) vs many open models using SP + LR scaling laws (DeepSeek LLM). Present both.

**Figures**
- Synthetic "LR sweep" curves: training loss vs log LR for widths 256–4096 under SP (optimum drifts) vs μP (optimum aligned). Generate by simulating a small MLP in numpy, or clearly label as schematic if not simulated.
- Coordinate-check plot: activation RMS vs width for SP vs μP after a few Adam steps (simulated).

**Question ideas**
- Derivation step: why an Adam update to a width-$n$ hidden matrix changes outputs by $\Theta(\eta n)$.
- Predict: double width under SP keeping LR fixed — which layer's updates blow up first?
- Which is false: "μP makes the optimal learning rate independent of depth as well as width."
- Compare: attention $1/\sqrt{d}$ vs $1/d$ — justify each.

**Pitfalls / checks**
- Table 3 has several equivalent formulations (multiplier vs init vs LR); pick one and state it. Check the exact entries before writing.
## Unit G: Efficient attention variants

### 190 · `llm.mqa-gqa-mla` · Shrinking the KV cache: MQA, GQA and MLA
core · wave 1 · prereqs: `llm.causal-attention`, `llm.params-flops` (restates the KV formula; `llm.kv-cache` deepens it)

**Primary sources**
- Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need* §2–3 (incremental decoding is memory-bandwidth bound; memory-to-compute ratio analysis; MQA) (p.1–5). https://arxiv.org/abs/1911.02150 · `llm_shazeer2019_mqa.txt`
- Ainslie et al. 2023 (GQA) §2.1 uptraining (mean-pool K/V heads, ~5% of pretraining compute), §2.2 grouped-query attention (p.1–3). https://arxiv.org/abs/2305.13245 · `llm_ainslie2023_gqa.txt`
- DeepSeek-V2 §2.1.1 standard MHA, §2.1.2 low-rank KV joint compression, §2.1.3 decoupled RoPE, §2.1.4 KV-cache comparison table (MLA ≈ GQA with 2.25 groups), App. C full MLA formulas, App. D attention ablations (p.6–9, appendix). https://arxiv.org/abs/2405.04434 · `llm_deepseek2024_v2.txt`
- DeepSeek-V3 §2.1.1 MLA recap and §4.2 hyper-parameters ($n_h=128$, $d_h=128$, $d_c=512$, $d'_c=1536$, $d^R_h=64$; p.6–7, 22). `llm_deepseek2024_v3.txt`
- Raschka, *Big LLM Architecture Comparison* §1.1 MLA (figures, MLA vs GQA vs MHA). `llm_raschka_big_arch_comparison.txt`; JAX Scaling Book Part 4 *Key-Value (KV) caching*. `llm_scalingbook_transformers.txt`
- Touvron et al. 2023 (Llama 2) §2.2 and App. A.2.1 (GQA adopted for the larger models for inference scalability; attention-variant ablation). https://arxiv.org/abs/2307.09288 · `llm_touvron2023_llama2.txt`

**Subtopic map**
1. *Why the KV cache dominates decode:* per-token cache bytes $2\cdot L\cdot h_{kv}\cdot d_h\cdot b$; decode step reads weights + whole cache; Shazeer's ratio argument (memory traffic per FLOP ∝ $1/b + n/d$ terms; check the paper's expression).
2. *MQA:* all $h$ query heads share one K/V head ⇒ cache ÷ $h$; quality and training-stability costs; tensor-parallel replication issue (one KV head can't be split across devices).
3. *GQA:* $g$ groups each sharing a K/V head; interpolates MHA ($g=h$) and MQA ($g=1$); uptraining by mean-pooling heads; standard choice (Llama 2 70B / Llama 3: 8 KV heads).
4. *Worked KV numbers (Python-verified):* Llama-3-70B (80 layers, 8 KV heads, $d_h=128$, bf16): 320 KiB/token; 40 GiB for one 128k-token sequence; 80 GiB for 32 × 8k sequences — vs 140 GB of bf16 weights. The MHA counterfactual (64 KV heads) is 8× larger (2.5 MiB/token).
5. *MLA (derive):* compress $h_t$ into a latent $c^{KV}_t=W^{DKV}h_t\in\mathbb{R}^{d_c}$, reconstruct per-head keys/values $k=W^{UK}c$, $v=W^{UV}c$; cache only $c$. *Absorption trick:* $q^\top k = (W^{UQ}c^Q)^\top W^{UK}c^{KV} = c^{Q\top}(W^{UQ\top}W^{UK})c^{KV}$, so $W^{UK}$ folds into the query side and $W^{UV}$ into $W^O$ — attention runs directly on latents at inference.
6. *Why RoPE breaks absorption, and decoupled RoPE:* a position-dependent rotation sits between $W^{UQ}$ and $W^{UK}$, so the product is no longer a fixed matrix; MLA adds a small shared RoPE key $k^R_t$ ($d^R_h=64$) and per-head RoPE queries, concatenated with the non-rotary parts. Cache per token per layer $=d_c+d^R_h=576$ elements.
7. *Worked MLA numbers (Python-verified):* DeepSeek-V3: 576 elements/layer vs MHA's $2\cdot128\cdot128=32{,}768$ (≈57× smaller); ×61 layers × 2 bytes ≈ 68.6 KiB/token. DeepSeek-V2's comparison: equal to GQA with 2.25 groups, yet reported stronger than MHA in their ablation.
8. *Query compression* ($d'_c=1536$) reduces activation memory in training, not the cache.
9. *Trade-offs:* MLA is compute-heavier per byte (higher arithmetic intensity in decode — good on memory-bound hardware), more complex kernels; at inference it runs in "MQA mode" (one shared latent per token, all heads attend to it; DeepSeek-V3.2 App. A, pdf p.20). Adoption 2025–26 (Kimi K2/K2.5, GLM-5, Ling 2.5, Sarvam 105B per Raschka 2026) vs GQA remaining the default. Note the counter-trend: DeepSeek V4 (preview, Apr 2026) runs shared-KV MQA over *compressed* KV entries (CSA/HCA) instead of MLA (V4 §2.3, pdf p.9–12; detail in `llm.sparse-attention`).
10. *Other cache reducers (pointers):* sliding windows (`llm.sparse-attention`), cross-layer KV sharing, KV quantization (`llm.quantization-advanced`), eviction (`llm.long-context`).

**Figures**
- Head-sharing diagram: MHA / GQA ($g=2$) / MQA / MLA (latent box) with arrows from K/V heads to query heads.
- KV-cache GiB vs context length for Llama-3-70B under MHA, GQA-8, MQA, and an MLA-style 576-element cache (same layer count), with a horizontal line at 80 GB HBM.
- MLA data-flow: $h\to c^{KV}$ (cached) $\to$ per-head $k,v$; the RoPE side branch.

**Question ideas**
- Compute: KV bytes/token for Llama-3-8B (32 layers, 8 KV heads, $d_h=128$, bf16) → 128 KiB.
- Compute: max batch at 8k context on one 80 GB GPU after 16 GB of weights (8B model): 64 GB ≈ 59.6 GiB free ⇒ ≈488k tokens at 128 KiB/token ⇒ 59 sequences (Python-verified; ignores activations and fragmentation — and watch GB vs GiB).
- Derivation step: which matrix product lets MLA skip materializing per-head keys?
- Which is false: "MLA's absorption trick still works if RoPE is applied directly to the up-projected keys $W^{UK}c^{KV}_t$." (False: the position-dependent rotation sits between $W^{UQ}$ and $W^{UK}$, which is why decoupled RoPE exists. The true options can include "MLA caches one latent vector per token per layer plus a small shared RoPE key".)
- Predict: converting an MHA checkpoint to GQA by keeping only the first K/V head of each group vs mean-pooling (GQA paper finds mean-pooling best).
- Open: "Explain why RoPE complicates MLA and how DeepSeek solved it."

**Pitfalls / checks**
- MLA notation differs between V2 and V3 papers; pick V2 App. C formulas.
- KV numbers assume bf16 and no quantization; say so.

---

### 200 · `llm.sparse-attention` · Local, strided and learned sparse attention
intermediate · wave 2 · prereqs: `llm.self-attention`, `llm.causal-attention`

**Primary sources**
- Child et al. 2019, *Sparse Transformers* §4 factorized (strided/fixed) attention patterns, $O(n\sqrt n)$ (p.3–5). https://arxiv.org/abs/1904.10509 · `llm_child2019_sparse_transformer.txt`
- Beltagy et al. 2020 (Longformer) §3 sliding window, dilated window, global tokens. https://arxiv.org/abs/2004.05150 · `llm_beltagy2020_longformer.txt`
- Jiang et al. 2023 (Mistral 7B) §2 sliding-window attention ($W=4096$, receptive field ≈ $L\cdot W$), rolling buffer cache (p.2–3). https://arxiv.org/abs/2310.06825 · `llm_jiang2023_mistral7b.txt`
- Gemma 3 report §2 / §5.2 local:global 5:1, local span 1024, KV-memory motivation (p.1–6). https://arxiv.org/abs/2503.19786 · `llm_gemma2025_gemma3.txt`; gpt-oss model card §2 (alternating banded window of 128 tokens and dense layers; p.5). https://arxiv.org/abs/2508.10925 · `llm_openai2025_gptoss.txt`
- Yuan et al. 2025 (Native Sparse Attention) §2 critique of inference-only sparsity, §3.3 compression / selection / sliding-window branches with gates, §3.4 kernel design (p.3–8). https://arxiv.org/abs/2502.11089 · `llm_yuan2025_nsa.txt`
- DeepSeek-V3.2 §2.1 DeepSeek Sparse Attention: lightning indexer + top-k token selection, continued pretraining (dense warm-up then sparse), §2.3 inference costs (p.3–5). https://arxiv.org/abs/2512.02556 · `llm_deepseek2025_v32.txt`; DeepSeek-V4 report (CSA = compressed + DSA; HCA = heavy compression, dense). https://arxiv.org/abs/2606.19348 · `llm_deepseek2026_v4.txt`

**Subtopic map**
1. *Why sparsify:* $O(n^2)$ compute and $O(n)$-per-token KV memory; most attention mass is local or on a few tokens.
2. *Sliding-window attention:* each query sees the last $W$ keys; cost $O(nW)$; receptive field grows by $W$ per layer ⇒ $\approx L\cdot W$ after $L$ layers (derive; contrast "can in principle" vs effective use); rolling-buffer KV cache of size $W$.
3. *Interleaving local and global layers:* Gemma 2/3 (5 local : 1 global in Gemma 3, local span 1024), gpt-oss (alternating banded-128/dense), Llama 4 / Command-A style chunked + global; KV-memory arithmetic: worked example of cache saved by 5:1 at 128k.
4. *Fixed sparse patterns:* strided + local (Sparse Transformer), dilated windows and global tokens (Longformer/BigBird); why these lost to dense + FlashAttention for mid-length contexts (GPU efficiency of block-sparse vs irregular).
5. *Learned/dynamic sparsity:* query-dependent selection of key blocks. NSA: three branches — compressed coarse tokens, top-k selected blocks (chosen using the compressed scores), sliding window — combined by learned gates; trained natively, blockwise for hardware. DSA (V3.2): a cheap FP8 "lightning indexer" (few heads, ReLU) scores all previous tokens, each query attends to the top-$k$ tokens ($k=2048$), cost of the indexer still $O(n^2)$ but with a tiny constant; trained by a short dense warm-up of the indexer (1000 steps) then sparse continued training (V3.2 §2.1, pdf p.3–4). V4's CSA/HCA (preview, Apr 2026): CSA compresses the KV of every $m$ tokens into one entry and applies DSA-style top-$k$ selection over compressed entries; HCA compresses every $m'\gg m$ tokens and attends densely; both add a sliding-window branch for local detail and run the core attention as shared-KV MQA; layers interleave CSA and HCA. Reported at 1M tokens: 27% of V3.2's per-token FLOPs and 10% of its KV cache for V4-Pro (V4 §2.3 and pdf p.1, 5, 9–12).
6. *Training vs inference sparsity:* methods that sparsify only at inference (eviction) vs natively trained sparsity (NSA's argument); compatibility with MLA/GQA (NSA groups queries to share selected blocks).
7. *Quality failure modes:* retrieval of a specific far token requires a global path; sliding-only models fail needle tests beyond $L\cdot W$ in practice.

**Figures**
- Mask gallery ($32\times32$): sliding window, strided (Sparse Transformer), Longformer local+global, block-selected (NSA-like) for one query row.
- Receptive field growth for SWA with $W=4$ across 3 layers (which inputs can influence the last token).
- KV cache GiB vs context for all-global vs 5:1 local:global (local span 1024) for a Gemma-3-27B-like config (state the config used).

**Question ideas**
- Compute: theoretical receptive field of a 32-layer SWA model with $W=4096$ (≈131k).
- Predict: replace every global layer of a 5:1 model with local — effect on a 100k-token needle test.
- Which is false: "DSA's indexer makes attention selection linear in context length."
- Figure MCQ: which mask corresponds to Longformer with two global tokens?
- Compare: NSA vs StreamingLLM-style eviction.

**Pitfalls / checks**
- Gemma 3 numbers (5:1, 1024) from the report (p.1); gpt-oss bandwidth 128 (p.5).
- DSA/CSA/HCA details are 2025–26; verify specifics in the cached reports before writing; mark as recent.

---

### 210 · `llm.attention-sinks` · Attention sinks, massive activations and softmax variants
advanced · wave 3 · prereqs: `llm.sparse-attention`, `llm.training-stability`

**Primary sources**
- Xiao et al. 2023 (StreamingLLM) §3.1 failure of window attention and attention sinks, §3.2 rolling KV cache with sinks, §3.3 pretraining with a sink token (p.4–6). https://arxiv.org/abs/2309.17453 · `llm_xiao2023_attention_sinks.txt`
- Sun et al. 2024, *Massive Activations in LLMs* (huge activations in a few dims/tokens acting as fixed biases; link to sinks). https://arxiv.org/abs/2402.17762 · `llm_sun2024_massive_activations.txt`
- gpt-oss model card §2: learned per-head bias in the softmax denominator ("similar to off-by-one attention and attention sinks", p.5). `llm_openai2025_gptoss.txt`
- Qiu et al. 2025, *Gated Attention for LLMs* §2 (sigmoid gate after SDPA), §4.1 non-linearity in the low-rank $W_VW_O$ map, §4.2 input-dependent sparsity, §5.2 attention-sink link (p.2–9). https://arxiv.org/abs/2505.06708 · `llm_qiu2025_gated_attention.txt`
- Ye et al. 2024 (Differential Transformer) §2 differential attention $(\mathrm{softmax}(Q_1K_1^\top)-\lambda\,\mathrm{softmax}(Q_2K_2^\top))V$, λ re-parameterization, headwise norm. https://arxiv.org/abs/2410.05258 · `llm_ye2024_diff_transformer.txt`

**Subtopic map**
1. *Observation:* many heads put large weight on the first token(s) regardless of content; evicting them (pure sliding-window cache) collapses perplexity.
2. *Why (mechanism):* softmax weights must sum to 1, so a head with "nothing to do" needs a place to dump mass; the first token is visible to all queries (causal) and becomes a no-op sink (its value vector ≈ small); massive activations provide the fixed bias.
3. *StreamingLLM fix:* keep the first ~4 tokens + a recent window; positions re-indexed within the cache; enables "infinite" streaming but *not* long-range memory.
4. *Architectural fixes:* dedicated learnable sink token at pretraining; softmax with an extra "+1" in the denominator (off-by-one / softmax₁); per-head learned sink logit (gpt-oss); gated attention output ($\sigma(XW_g)\odot\mathrm{SDPA}$) removes sinks and massive activations, adds non-linearity (Qwen3-Next uses it).
5. *Differential attention:* subtracting two softmax maps cancels common-mode "noise" attention; claims on hallucination/long context (single-paper evidence; mark as such).
6. *Related logit controls:* QK-norm, soft-capping (pointer to `llm.training-stability`).
7. *Practical implications:* quantization outliers (massive activations ↔ LLM.int8 outliers, `llm.quantization`), KV eviction policies must protect sinks, interpretability of "no-op" heads.

**Figures**
- Simulated attention map with a strong first-column sink (generated from a toy model or schematic labelled as such).
- Perplexity vs tokens for: dense, window-only, window+4 sinks (re-plot StreamingLLM Fig. numbers with citation, or schematic).
- Softmax vs softmax with an extra 1 in the denominator: max weight vs logit scale.

**Question ideas**
- Predict: sliding-window KV eviction that drops the first token after 4k tokens.
- Derivation: with an off-by-one softmax, what is the total attention mass when all logits are very negative?
- Which is false: "StreamingLLM lets a model recall a fact from 1M tokens ago."
- Compare: learned sink logit vs output gate as fixes.

**Pitfalls / checks**
- Differential attention and gated attention: one or two papers each; report as findings, not consensus.

## Unit H: Kernels and FlashAttention

*(Order 220, `llm.arithmetic-intensity`, was removed in review. Its generic content — three regimes, intensity, ridge point, matmul intensity ≈ batch, memory hierarchy, overhead — is owned by `sys.roofline` and `sys.gpu-basics`. Its transformer-specific content moved to `llm.kv-cache` items 3–5 and `llm.flash-attention` item 1. Its question ideas moved with it.)*

### 230 · `llm.flash-attention` · FlashAttention: online softmax, tiling and IO complexity
core · wave 1 · prereqs: `llm.self-attention` · cross-area: `sys.roofline` (intensity, ridge point), `sys.gpu-basics` (SMs, SRAM vs HBM)

**Primary sources**
- Milakov & Gimelshein 2018, *Online normalizer calculation for softmax*: Algorithm 1–3 (naive, safe, online safe softmax), Theorem 1 correctness proof (p.1–3). https://arxiv.org/abs/1805.02867 · `llm_milakov2018_online_softmax.txt`
- Dao et al. 2022 (FlashAttention) §2.2 standard attention implementation (HBM reads/writes), §3.1 tiling + recomputation (Algorithm 1; softmax decomposition $m(x)$, $\ell(x)$), §3.2 IO complexity Theorem 2 ($\Theta(Nd+N^2)$ vs $\Theta(N^2d^2M^{-1})$) and Proposition 3 lower bound, §3.3 block-sparse; App. B.1–B.4 forward/backward details, App. C proofs (p.3–6, 17–25). https://arxiv.org/abs/2205.14135 · `llm_dao2022_flashattention.txt`
- JAX Scaling Book Part 4 Appendix A "How does Flash Attention work?". `llm_scalingbook_transformers.txt`

**Subtopic map**
1. *Standard attention's IO problem:* one-card recap of the hardware (link `sys.gpu-basics`, `sys.roofline`): on an A100, HBM is 40–80 GB at 1.5–2.0 TB/s while on-chip SRAM is 192 KB per SM × 108 SMs at ~19 TB/s (FA §2.1, pdf p.3); elementwise ops (softmax, mask, dropout) have O(1) FLOPs per byte, so they are memory-bound and only fusion helps. Then: standard attention materializes $S=QK^\top$ and $P=\mathrm{softmax}(S)$ in HBM: $\Theta(N^2)$ reads/writes per head, dominating time for typical $d$ even though FLOPs are unchanged. Worked (Python): for $N=8192$, one head's $P$ in bf16 is 128 MiB of writes and reads per pass, versus 2 MiB for $Q$ ($d=128$).
2. *Safe softmax:* subtract the max for numerical stability ⇒ needs a pass to find the max before normalizing (3 passes naive).
3. *Online softmax (derive):* maintain running max $m_j$ and running denominator $\ell_j$; on a new element/block with max $\tilde m$: $m_{new}=\max(m,\tilde m)$, $\ell_{new}=e^{m-m_{new}}\ell+e^{\tilde m-m_{new}}\tilde\ell$. Prove by induction that $\ell_j=\sum_{i\le j}e^{x_i-m_j}$. One pass for max+sum.
4. *Extend to the output (the key step):* keep an unnormalized accumulator $o$; when the max changes, rescale $o\leftarrow e^{m-m_{new}}o + e^{\tilde m-m_{new}}\tilde P\tilde V$; divide by $\ell$ at the end. So softmax-times-$V$ can be computed block by block without ever storing a full row of $P$.
5. *Tiling algorithm (FA1 Algorithm 1):* outer loop over K/V blocks, inner over Q blocks (FA1 order); block sizes $B_c=\lceil M/4d\rceil$, $B_r=\min(B_c,d)$ to fit SRAM; per-block matmuls on tensor cores.
6. *IO complexity (proof sketch):* each K/V block loaded once per outer iteration, Q/O blocks re-read $T_c=N/B_c=\Theta(Nd/M)$ times ⇒ $\Theta(N^2d^2/M)$ HBM accesses vs $\Theta(Nd+N^2)$; for $d=64$–128 and $M\approx100$ KB this is many times fewer. Lower bound: no exact algorithm does asymptotically better for all $M$ (Prop. 3).
7. *Memory:* $O(N)$ extra (store $m,\ell$ — or just the logsumexp $L=m+\log\ell$ — per row), not $O(N^2)$.
8. *Backward via recomputation:* don't store $P$; recompute it blockwise from $Q,K$ and the saved logsumexp; more FLOPs, fewer bytes, net faster. (The gradient formulas were derived in `llm.self-attention` item 9; the tiled version is in `llm.flash-attention-2-3`.)
9. *Causal masking and block skipping:* blocks entirely above the diagonal are skipped (≈2× saving for causal).
10. *What FA does not change:* FLOPs ($O(N^2d)$), exactness (it is exact attention up to floating-point reordering), quadratic time.

**Figures**
- Tiling diagram: $S$ matrix divided into $B_r\times B_c$ tiles; one Q-tile row sweeping across K/V tiles, with SRAM vs HBM shading; causal-skipped tiles greyed.
- Online softmax trace: process a 12-element vector in 3 blocks, showing $m$, $\ell$ and the rescale factor after each block (numbers computed in Python, final equals full softmax).
- HBM traffic vs sequence length for standard attention vs FlashAttention ($d=128$, $M=$ 100–200 KB) from the Theorem 2 expressions (constants dropped, labelled).

**Question ideas**
- Compute: running $(m,\ell)$ after blocks [1,3] then [2,5].
- Derivation step: why multiply the old accumulator by $e^{m_{old}-m_{new}}$?
- Which is false: "FlashAttention reduces the FLOPs of attention from $O(N^2d)$ to $O(Nd)$."
- Predict: halve SRAM size — effect on HBM accesses (∝ $1/M$).
- Figure MCQ (from the old `llm.arithmetic-intensity`): on an H100 roofline, which point is the unfused softmax kernel and which is the FlashAttention forward?
- Spot the bug: online softmax that rescales $\ell$ but forgets to rescale the output accumulator.
- Open: "Derive online softmax and show how FlashAttention uses it to avoid materializing the attention matrix."

**Pitfalls / checks**
- FA1 loop order (K/V outer) differs from FA2 (Q outer); don't mix them.
- Block-size formulas from Algorithm 1 — copy exactly.

---

### 240 · `llm.flash-attention-2-3` · Tiled backward pass, FA2, FA3 and decode kernels
intermediate · wave 2 · prereqs: `llm.flash-attention`

**Primary sources**
- Dao 2023 (FlashAttention-2) §2.3.2 backward pass, §3.1 algorithm tweaks (fewer non-matmul FLOPs: unscaled output, store only logsumexp), §3.2 parallelism over sequence length, §3.3 work partitioning between warps (split Q, not K: avoids shared-memory reduction), §4 (≈2× over FA1, 50–73% of peak forward on A100) (p.3–11). https://arxiv.org/abs/2307.08691 · `llm_dao2023_flashattention2.txt`
- Shah et al. 2024 (FlashAttention-3) §3.1 producer–consumer warp specialization + ping-pong scheduling, §3.2 intra-warpgroup overlap of GEMMs and softmax, §3.3 FP8 (block quantization, incoherent processing with Hadamard), §4 (up to 740 TFLOP/s FP16 = 75% of H100; ~1.2 PFLOP/s FP8; 2.6× lower FP8 error than baseline) (p.1–10). https://arxiv.org/abs/2407.08608 · `llm_shah2024_flashattention3.txt`
- Dao 2022 App. B.2–B.4 (memory-efficient backward; gradient of softmax: $dS=P\odot(dP-D)$ with $D_i=\mathrm{rowsum}(dO_i\odot O_i)$) (p.18–21). `llm_dao2022_flashattention.txt`
- JAX Scaling Book inference part (*What about attention?*) for decode-time attention bottlenecks. `llm_scalingbook_inference.txt`

**Subtopic map**
1. *Attention backward, tiled:* recap the formulas from `llm.self-attention` item 9 ($dV=P^\top dO$, $dP=dO\,V^\top$, $dS_{ij}=P_{ij}(dP_{ij}-D_i)$, $D_i=dO_i\cdot O_i$) in one card, then show why the identity $D_i=dO_i\cdot O_i$ matters for tiling: $D$ is computed from $O$ and $dO$ (both $N\times d$) before the loop, so no tile ever needs a full row of $dP$.
2. *Recomputation in backward:* recompute $P$ tile by tile from $Q,K$ and the logsumexp; accumulate $dK,dV$ per K/V block, $dQ$ via atomic adds (FA2) — why.
3. *FA2 changes:* (a) don't rescale the output every block — keep it unnormalized and divide once; store only logsumexp; (b) loop order Q-outer so each thread block owns output rows; parallelize across sequence blocks (not just batch × heads) — matters for long sequences with small batch; (c) within a block, split Q across warps so warps don't need to communicate through shared memory. Why non-matmul FLOPs matter: tensor-core matmul throughput is ~16× higher than non-matmul FP32 on A100 (check the paper's figure).
4. *FA3 (Hopper):* asynchronous TMA loads + warp-specialized producers/consumers; ping-pong between two warpgroups so one's softmax overlaps the other's GEMM; intra-warpgroup pipelining; FP8 with per-block scales and Hadamard "incoherent processing" to spread outliers.
5. *Decode-time kernels:* one query per sequence ⇒ FA's parallelism over queries is useless; split-KV / FlashDecoding: partition the KV cache across thread blocks, compute partial softmax stats, combine with the same log-sum-exp merge as online softmax (derive the merge).
6. *Paged KV + FlashAttention* (pointer to `llm.serving-systems`); variable-length (packed) batches with cumulative sequence offsets.
7. *Numerics:* fp32 accumulation of $m,\ell$; determinism issues from atomic adds in backward.

**Figures**
- Loop-order comparison: FA1 (K/V outer) vs FA2 (Q outer) arrows over the score matrix.
- Ping-pong timeline: two warpgroups alternating GEMM and softmax (Gantt-style, schematic).
- Split-KV decode: one query row, KV cache split into 4 chunks, partial $(m,\ell,o)$ merged.

**Question ideas**
- Derivation: which saved per-row statistics (logsumexp, $D_i$) does the tiled backward need, and why is $D_i$ computable before the loop?
- Compute: merge two partial softmax results $(m_1,\ell_1,o_1)$, $(m_2,\ell_2,o_2)$.
- Which is false: "FA2 speeds up attention mainly by reducing matmul FLOPs."
- Predict: batch 1, 128k tokens, 32 heads on 132 SMs — why FA1-style parallelism over (batch, heads) underutilizes the GPU.
- Compare: FlashAttention vs FlashDecoding — what is parallelized.

**Pitfalls / checks**
- FA2 headline numbers are A100; FA3 numbers are H100. Keep hardware attached to every number.
## Unit I: Inference

### 250 · `llm.kv-cache` · KV-cache arithmetic, prefill vs decode, and latency
core · wave 1 · prereqs: `llm.causal-attention`, `llm.params-flops` · cross-area: `sys.roofline` (intensity, ridge point, matmul intensity ≈ batch), `sys.flops-mfu`

Applies `sys.roofline` / `sys.flops-mfu` to prefill vs decode rather than re-teaching them. Absorbs the transformer-specific half of the removed `llm.arithmetic-intensity` (items 3–5 below); open with a two-sentence recap of intensity and the ridge point and link `sys.roofline`. Overlaps the Scaling Book series (`sb.*`) on inference math; keep framework-neutral (no JAX/TPU specifics beyond quoted hardware numbers).

**Primary sources**
- JAX Scaling Book Part 7: *What do we actually want to optimize?*, *Linear operations: what bottlenecks us?* (decode matmuls memory-bound until batch ≳ critical intensity), *What about attention?*, *Theoretical estimates for LLM latency and throughput*, *What about memory?*, *Modeling throughput and latency for LLaMA 2-13B*; Part 8 *Serving LLaMA 3-70B* (*Thinking about throughput*, *What about prefill?*). https://jax-ml.github.io/scaling-book/inference/ · `llm_scalingbook_inference.txt`, `llm_scalingbook_applied-inference.txt`
- kipply, *Transformer Inference Arithmetic* (KV cache size, FLOPs vs memory per token, latency estimates). `llm_kipply2022_inference_arithmetic.txt`
- Pope et al. 2022, *Efficiently Scaling Transformer Inference* §2.1 trade-offs (latency, throughput, MFU), §2.2 setup, §3.3 multi-query attention's effect on memory (p.3–7). https://arxiv.org/abs/2211.05102 · `llm_pope2022_inference_scaling.txt`
- Lilian Weng, *Large Transformer Model Inference Optimization* (2023), sections on methods overview and KV-cache cost. `llm_weng2023_inference_optimization.txt`

**Subtopic map**
1. *Two phases:* prefill (prompt of $P$ tokens processed in parallel; FLOPs $\approx2NP$ + attention; compute-bound for $P$ beyond a few hundred) vs decode (1 token per sequence per step; memory-bound).
2. *KV formula:* bytes $=2\cdot L\cdot h_{kv}\cdot d_h\cdot b_{bytes}\cdot T\cdot B$; worked: Llama-3-70B 320 KiB/token; 40 GiB at 128k; 80 GiB for 32×8k; Llama-3-8B 128 KiB/token (all Python-verified).
3. *Decode matmuls are memory-bound (apply `sys.roofline`):* in a decode step each weight matrix multiplies a $[B,d]$ activation, so its intensity is ≈ $B$ FLOPs/byte in bf16 — at batch 1 that is ~300× below the H100 ridge (≈295). Prefill multiplies $[B\cdot P,d]$, so a prompt of a few hundred tokens is already compute-bound.
4. *Decode attention is memory-bound and batching doesn't help it (derive):* per layer, reading the K cache costs $2\,T h_{kv}d_h$ bytes and the scores cost $2\,T h\,d_h$ FLOPs, so the intensity is ≈ $h/h_{kv}$ FLOPs/byte (MHA: 1; Llama-3-70B GQA-8: 8), independent of $B$ because every sequence has its own cache. MLA in its absorbed MQA mode shares one 576-element latent across 128 heads: ≈ $2\cdot128\cdot(576+512)/(2\cdot576)\approx242$ FLOPs/byte (Python-checked; bf16, DeepSeek-V3 dims), close to the ridge — the real reason MLA is decode-friendly beyond cache size.
5. *Decode step time model:* $t \gtrsim \max\big(\frac{\text{weight bytes}+\text{KV bytes}}{\text{BW}},\ \frac{2NB}{\text{FLOP/s}}\big)$. Worked: 70B in bf16 at batch 1: ≥ 140 GB / 3.35 TB/s ≈ 42 ms/token on one H100-class device (memory floor, ignoring that 140 GB doesn't fit one 80 GB GPU — so in practice TP across ≥2 GPUs; say so) vs 0.14 ms of compute ⇒ ~300× idle compute.
6. *Batching:* weights are read once per step for the whole batch, so throughput rises ~linearly with batch until (a) compute-bound at batch ≈ ridge point (~300 on H100, 240 on TPU v5e) or (b) KV cache fills memory. KV reads grow with batch × context, so long contexts hit (b) first.
7. *Latency vs throughput:* TTFT (time to first token; prefill-dominated), TPOT/ITL (decode), throughput (tokens/s/GPU); Pareto frontier; why larger batches hurt per-user latency only mildly while memory-bound.
8. *Memory budget worked example:* one 80 GB GPU, 8B bf16 model (16 GB weights); 64 GB ≈ 59.6 GiB is left for the cache; at 128 KiB/token that holds ≈488k tokens, i.e. 59 sequences at 8k (Python-verified; ignores activations, CUDA context and fragmentation). Use this to teach the GB-vs-GiB trap: treating the 64 GB as 64 GiB gives 524,288 tokens and 64 sequences, an 8% overestimate.
9. *Prefill attention vs linear cost:* attention FLOPs $\propto P^2$; when long prompts dominate.
10. *Levers:* fewer KV bytes (GQA/MLA, KV quantization, sliding windows), fewer weight bytes (weight quantization; MoE active params), more tokens per weight read (batching, speculative decoding), parallelism (TP lowers per-device bytes; link applied).
11. *MoE at inference (one line; forward pointer):* memory holds all experts; per-token FLOPs use active params; at small batch each token touches different experts, so bytes read per step can approach total params (forward pointer to `llm.moe-systems`).

**Figures**
- KV cache GiB vs context length for 8B and 70B (GQA-8) with the 80 GB line; shaded region = weights.
- Roofline (H100 bf16) with points for decode matmul at batch 1/64/512, prefill matmul, and decode attention for MHA, GQA-8 and MLA — reader sees attention stuck left of the ridge regardless of batch, MLA closest to it.
- Decode latency and throughput vs batch size (roofline-derived model for a 70B model on 4 GPUs with TP), showing the memory-bound plateau and the knee.
- Timeline of one request: prefill block then decode steps; TTFT and TPOT labelled.

**Question ideas**
- Compute: KV bytes for one 32k-token sequence on Llama-3-70B (10 GiB).
- Compute: minimum per-token decode latency for a 7B model in int4 on 1 TB/s bandwidth (3.5 GB ⇒ 3.5 ms, ignoring KV and scales).
- Compute (from the old `llm.arithmetic-intensity`): intensity of a $[16,8192]\times[8192,8192]$ bf16 matmul (≈16 FLOPs/byte) — memory- or compute-bound on H100?
- Which is false: "Increasing batch size raises the arithmetic intensity of decode attention over the KV cache."
- Predict: double the batch from 8 to 16 at 2k context — effect on throughput and per-token latency.
- Which is false: "Prefill of a 4k-token prompt is memory-bound like decode."
- Figure MCQ: on the latency–throughput plot, which point is batch 256?
- Open: "Estimate tokens/s for serving a 70B model on 8 H100s at batch 64, 4k context."

**Pitfalls / checks**
- Ridge-point numbers: H100 ≈ 295 (Scaling Book GPU part), TPU v5e 240 (inference part).
- State bf16 and whether TP is used in every worked latency.

---

### 260 · `llm.serving-systems` · Serving: continuous batching, PagedAttention, prefix caching
intermediate · wave 2 · prereqs: `llm.kv-cache`

**Primary sources**
- Kwon et al. 2023 (vLLM / PagedAttention) §2.2 batching techniques (iteration-level scheduling), §3 memory challenges (only 20.4–38.2% of KV memory holds actual token states in prior systems), §4.1–4.5 PagedAttention, block tables, copy-on-write for parallel sampling/beam search, scheduling and preemption (swap/recompute), §6 results (2–4× throughput) (p.2–13). https://arxiv.org/abs/2309.06180 · `llm_kwon2023_pagedattention.txt`
- JAX Scaling Book Part 7 *Designing an Effective Inference Engine*: *Continuous batching*, *Prefix caching*, disaggregated prefill/generate; *Distributing Inference*: prefill, generation, sharding the KV cache. `llm_scalingbook_inference.txt`
- Pope 2022 §3 partitioning for inference (weight-stationary vs gathered layouts; p.3–8). `llm_pope2022_inference_scaling.txt`

**Subtopic map**
1. *Static batching problem:* requests finish at different times; padding waste; head-of-line blocking.
2. *Continuous (iteration-level) batching:* scheduler adds/removes sequences every decode step (Orca idea; vLLM §2.2); mixing prefill and decode in a step.
3. *KV memory fragmentation:* contiguous per-request max-length reservations → internal fragmentation (reserved but unused), external fragmentation; measured utilization 20.4–38.2% (vLLM Fig. 2).
4. *PagedAttention:* KV stored in fixed-size blocks (e.g. 16 tokens) addressed via a per-sequence block table, like OS virtual memory; waste only in the last block per sequence; attention kernel gathers blocks.
5. *Sharing:* parallel sampling and beam search share prompt blocks with reference counts and copy-on-write; prefix caching across requests (system prompts; radix-tree caches like SGLang's RadixAttention — mention).
6. *Preemption:* when memory runs out, evict whole sequences (swap to CPU or recompute later); all-or-nothing per sequence.
7. *Chunked prefill:* split long prompts into chunks interleaved with decode steps to bound TPOT spikes.
8. *Prefill/decode disaggregation:* separate pools tuned for compute-bound prefill vs memory-bound decode; KV transfer cost.
9. *Multi-GPU serving* (pointer to applied area): TP for latency, DP replicas for throughput, EP for MoE.
10. *Metrics and SLOs:* TTFT, TPOT, goodput.

**Figures**
- Memory layout: contiguous reservation per request (with internal fragmentation shaded) vs paged blocks with a block table.
- Continuous vs static batching Gantt chart over 10 steps for 4 requests of different lengths.
- Copy-on-write: two samples sharing prompt blocks then diverging.

**Question ideas**
- Compute: with block size 16 and 100 sequences of random lengths, expected wasted slots (≤15 each; mean ≈7.5).
- Predict: turning off prefix caching for a chatbot with a 2k-token system prompt — TTFT effect.
- Which is false: "PagedAttention reduces attention FLOPs."
- Compare: swap vs recompute preemption.

**Pitfalls / checks**
- The vLLM paper reports near-zero waste and 2–4× throughput; the often-quoted "<4% waste" comes from the vLLM blog, not the paper — don't attribute it to the paper.

---

### 270 · `llm.decoding` · Search-based decoding: greedy, beam search and its failure modes
intermediate · wave 2 · prereqs: `llm.pretraining-objective` (ordered before `llm.sampling` in review: maximization's failure is the motivation for sampling)

**Primary sources**
- d2l §10.8 Beam Search (greedy vs exhaustive vs beam; complexity; length-normalized score). https://d2l.ai/chapter_recurrent-modern/beam-search.html · `llm_d2l_beam_search.txt`
- SLP3 ch. 13 §13.4 decoding in MT with beam search (pdf p.10–15). `llm_slp3_ch13.txt`; ch. 7 §7.6.1 greedy decoding (pdf p.21–22). `llm_slp3_ch7.txt`
- Meister, Vieira & Cotterell 2020, *If Beam Search is the Answer, What was the Question?* §2.1–2.2 beam search and length normalization, §4 UID bias (p.3–6). https://arxiv.org/abs/2010.02650 · `llm_meister2020_beam.txt`
- Holtzman et al. 2019 §4.3 "natural language does not maximize probability", §5.3 repetition (p.7–9). https://arxiv.org/abs/1904.09751 · `llm_holtzman2019_nucleus.txt`
- Willard & Louf 2023, *Efficient Guided Generation* (regex/grammar → FSM → token masks). https://arxiv.org/abs/2307.09702 · `llm_willard2023_structured_generation.txt`

**Subtopic map**
1. *Decoding as search:* MAP sequence $\arg\max_y\sum_t\log p(y_t\mid y_{<t})$ is intractable ($V^T$); greedy = myopic; example where greedy is suboptimal (worked 2-step tree, Python-checked).
2. *Beam search:* keep top-$k$ partial hypotheses by cumulative log-prob; cost $O(kTV)$ scoring; end-of-sequence handling.
3. *Length bias and normalization:* log-probs are negative ⇒ shorter hypotheses favoured; normalize by $|y|^\alpha$ (GNMT-style) or $1/|y|$; trade-offs.
4. *The beam search curse / inadequacy of the mode:* exact MAP often returns empty or degenerate outputs; larger beams can hurt quality; Meister's view that beam search's implicit bias (uniform information density) is why it works in MT.
5. *Open-ended generation:* maximization → repetition loops and bland text (Holtzman: human text is not high-probability under the model) ⇒ sampling (next lesson).
6. *Where search still fits:* closed-ended tasks (MT, ASR, code with tests), reasoning with verifiers (step-level beam search, `llm.test-time-compute`), MBR decoding (choose the candidate with max expected utility against samples; mention).
7. *Constrained/structured decoding:* mask logits to tokens consistent with a grammar/JSON schema/regex; FSM over tokens (Willard & Louf); subtle distortion: masking changes the distribution vs conditioning on the constraint (greedy masking ≠ sampling from $p(y\mid y\in\mathcal{C})$).
8. *Stopping criteria and EOS calibration.*

**Figures**
- Search tree over 3 steps with probabilities where greedy picks a worse full sequence than beam size 2.
- BLEU/quality vs beam width (schematic or cite numbers) and output length vs beam width.

**Question ideas**
- Compute: greedy vs beam-2 sequence probabilities on a given tree.
- Predict: increase beam from 5 to 100 on open-ended story generation.
- Which is false: "Exact MAP decoding of a well-trained LM produces the best-quality text."
- Spot the flaw: JSON-constrained decoding claimed to sample from the model's conditional distribution over valid JSON.

**Pitfalls / checks**
- Length-penalty formulas vary (GNMT uses $((5+|y|)/6)^\alpha$); state which.

---

### 280 · `llm.sampling` · Sampling: temperature, top-k, top-p, min-p and penalties
core · wave 1 · prereqs: `llm.pretraining-objective` (recommended: `llm.decoding`, which motivates why we sample rather than maximize)

**Primary sources**
- Holtzman et al. 2019 (nucleus sampling) §3.1 nucleus sampling definition, §3.2 top-k and its failure for flat vs peaked distributions, §3.3 temperature, §5 distributional evaluation (p.4–9). https://arxiv.org/abs/1904.09751 · `llm_holtzman2019_nucleus.txt`
- SLP3 ch. 7 §7.6 sampling: random sampling, temperature, top-k, top-p (pdf p.21–25). `llm_slp3_ch7.txt`
- Nguyen et al. 2024 (min-p) §3 method ($p_{\text{scaled}}=p_{\text{base}}\cdot p_{\max}$; keep tokens with $p\ge p_{\text{scaled}}$; p.3) and high-temperature results. https://arxiv.org/abs/2407.01082 · `llm_nguyen2024_minp.txt`
- Meister et al. 2022, *Locally Typical Sampling* (keep tokens whose surprisal is close to the conditional entropy). https://arxiv.org/abs/2202.00666 · `llm_meister2022_typical.txt`
- Keskar et al. 2019 (CTRL) §4.1 penalized sampling (repetition penalty θ≈1.2 dividing logits of previously generated tokens). https://arxiv.org/abs/1909.05858 · `llm_keskar2019_ctrl.txt`
- GPT-4 technical report, calibration figure (pre-trained model well calibrated; post-training reduces calibration). https://arxiv.org/abs/2303.08774 · `llm_openai2023_gpt4.txt`; Kadavath et al. 2022 (calibration of LMs). `llm_kadavath2022_know_what_they_know.txt`

**Subtopic map**
1. *Ancestral sampling* from $p$: unbiased but the long tail of low-probability tokens accumulates over many steps (probability of at least one tail token in $T$ steps).
2. *Temperature (derive limits):* $p_T(x)\propto\exp(z_x/T)$; $T\to0$ → argmax, $T\to\infty$ → uniform; entropy is monotone increasing in $T$ (show $dH/dT\ge0$ via variance of logits under $p_T$). Worked: logits (2,1,0) at T=0.5/1/2 → (0.867,0.117,0.016), (0.665,0.245,0.090), (0.506,0.307,0.186) (Python-verified). Equivalent to sampling from $p^{1/T}$ renormalized.
3. *Top-k:* keep $k$ highest then renormalize; fails because the right $k$ depends on context entropy (flat vs peaked).
4. *Top-p (nucleus):* smallest set with cumulative mass ≥ p; adapts to entropy; worked example.
5. *Min-p:* threshold relative to the top token's probability; behaves better at high temperature; order of operations (temperature before/after truncation) matters and differs across libraries.
6. *Typical sampling* (brief) and η/ε-sampling (mention).
7. *Repetition controls:* CTRL repetition penalty (divide positive logits / multiply negative ones by θ — note sign handling), presence and frequency penalties (OpenAI-style additive), n-gram blocking; side effects (penalizing necessary repeats like names/code).
8. *Interaction with post-training:* RLHF/RL models are sharper/less calibrated; typical serving defaults; reasoning models are evaluated with sampling, not greedy decoding: DeepSeek-R1 reports pass@1 from $k$ samples at temperature 0.6, top-p 0.95 (R1 cached revised version, pdf p.40), DAPO evaluates at T=1.0, top-p 0.7 (DAPO experimental setup, pdf p.8).
9. *Diversity vs quality trade-off;* best-of-N with sampling (pointer to `llm.test-time-compute`); determinism (T=0 is not bitwise deterministic on GPUs due to batching/non-associative reductions).
10. *Structured outputs:* logit masking (pointer to `llm.decoding`).

**Figures**
- One next-token distribution (sorted bars) shown under T=0.5/1/2, with the top-k (k=5), top-p (p=0.9) and min-p (0.1) cut-offs marked. Reader sees top-p and min-p adapt while top-k doesn't.
- Two contexts (peaked vs flat) with the same top-k cut — shows over/under-truncation.
- Entropy of $p_T$ vs T.

**Question ideas**
- Compute: top-p=0.8 set for probs (0.5,0.2,0.15,0.1,0.05).
- Compute: min-p 0.1 set for the same probs (threshold 0.05 → keep all with p ≥ 0.05).
- Predict: T=1.5 with top-p=0.95 vs T=1.5 with min-p=0.1 on a flat distribution.
- Which is false: "Lowering temperature can change which token is the argmax."
- Figure MCQ: which panel shows nucleus truncation at p = 0.9?
- Open: "Why does top-p adapt to context where top-k doesn't? Give an example."

**Pitfalls / checks**
- Order of temperature vs truncation differs between implementations (HF applies processors in a configured order); state the convention.
- CTRL's penalty divides logits by θ; for negative logits libraries multiply instead — check CTRL §4.1 and say what the paper defines.

---

### 290 · `llm.speculative-decoding` · Speculative decoding
intermediate · wave 1 · prereqs: `llm.kv-cache`, `llm.sampling`

**Primary sources**
- Leviathan, Kalman & Matias 2022, §2.3 speculative sampling (accept with prob. $\min(1,p/q)$, else sample from $\mathrm{norm}(\max(0,p-q))$), App. A.1 correctness proof, Definition 3.1/Theorem 3.5 ($\beta=1-D_{LK}(p,q)=\sum_x\min(p,q)$), §3.1 expected tokens per iteration $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, Theorem 3.8 walltime improvement $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$, Theorem 3.11 extra arithmetic, §3.6 approximation models (p.2–5). https://arxiv.org/abs/2211.17192 · `llm_leviathan2022_speculative.txt`
- Chen et al. 2023 (DeepMind), *Accelerating LLM Decoding with Speculative Sampling*: Algorithm 2, Theorem 1 (modified rejection sampling recovers the target) with proof (p.3–4, 10); 2–2.5× on Chinchilla 70B. https://arxiv.org/abs/2302.01318 · `llm_chen2023_speculative_sampling.txt`
- Cai et al. 2024 (Medusa) §2.1 extra heads + tree attention, §2.2 training (Medusa-1/2), typical acceptance. https://arxiv.org/abs/2401.10774 · `llm_cai2024_medusa.txt`; Li et al. 2024 (EAGLE) §3 feature-level drafting. https://arxiv.org/abs/2401.15077 · `llm_li2024_eagle.txt`
- JAX Scaling Book Part 7 Appendix D *Speculative Sampling*. `llm_scalingbook_inference.txt`; Gloeckle 2024 §3.2 self-speculative decoding with MTP heads. `llm_gloeckle2024_mtp.txt`

**Subtopic map**
1. *Motivation:* decode is memory-bound, so verifying $\gamma+1$ tokens in one target forward costs ≈ the same as generating one; a cheap drafter proposes $\gamma$ tokens.
2. *Algorithm:* draft $x_1..x_\gamma\sim q$; one target pass gives $p$ at all $\gamma+1$ positions; accept sequentially; on first rejection resample from the residual; if all accepted, sample one bonus token from $p$.
3. *Correctness proof (derive):* $P(\text{output}=x)=q(x)\min(1,\frac{p(x)}{q(x)}) + (1-\beta)\frac{\max(0,p(x)-q(x))}{1-\beta}=\min(q,p)+\max(0,p-q)=p(x)$, where $\beta=\sum_x\min(p,q)$ is the acceptance probability and $1-\beta=\sum_x\max(0,p-q)$ (prove this identity).
4. *Acceptance rate:* $\beta=1-\mathrm{TV}(p,q)$; greedy special case (accept iff argmaxes match).
5. *Expected tokens per target call (derive):* with i.i.d. acceptance α, number of accepted drafts is a truncated geometric ⇒ $E=\sum_{i=0}^{\gamma}\alpha^i=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. Worked (Python): α=0.8: γ=1→1.8, 2→2.44, 4→3.36, 8→4.33; with drafter cost ratio c=0.05 the speedup $E/(\gamma c+1)$ is 1.71, 2.22, 2.80, 3.09 — diminishing returns, optimum γ depends on α and c.
6. *Cost side:* wasted target FLOPs on rejected tokens (Theorem 3.11); helps latency at small batch, hurts or helps little at large batch where decode is already compute-bound.
7. *Drafters:* smaller same-family model (shared tokenizer required), n-gram/lookup (prompt lookup), extra decoding heads (Medusa; tree of candidates verified with tree attention masks), feature-level autoregressive head (EAGLE), MTP heads (Gloeckle; DeepSeek-V3 MTP used for speculation), self-speculation via early exit.
8. *Tree verification:* verify multiple branches in one pass with a tree-shaped causal mask; acceptance of the longest valid branch.
9. *KV-cache handling:* roll back rejected positions in both caches.
10. *Lossy variants:* Medusa's "typical acceptance" trades exactness for speed — say clearly it no longer samples exactly from $p$.

**Figures**
- Step diagram: draft 4 tokens, target scores, accept 2, reject 3rd, resample from residual.
- Residual distribution picture: bars of $p$, $q$, $\min(p,q)$ and normalized $\max(0,p-q)$ on a 6-token vocab.
- Expected tokens and speedup vs γ for α ∈ {0.6, 0.8, 0.9} at c = 0.05.

**Question ideas**
- Compute: $\beta$ for $p=(0.5,0.3,0.2)$, $q=(0.6,0.2,0.2)$ (=0.9) and the residual distribution ((0, 1, 0)).
- Derivation step: which term accounts for tokens produced after a rejection?
- Predict: drafter = target model itself (c = 1): speedup?
- Which is false: "Speculative sampling is exact only for greedy decoding; with temperature sampling it approximates the target." (False: the residual correction makes it exact for any drafter; drafter quality only affects speed.)
- Figure MCQ: identify the residual distribution among four bar charts.
- Open: "Prove speculative sampling preserves the target distribution."

**Pitfalls / checks**
- The i.i.d.-α assumption behind the closed form is a simplification (Leviathan §3.1).
- Draft and target must share tokenization and sampling temperature/filters (apply top-p etc. to both $p$ and $q$ consistently).
---

### 300 · `llm.quantization` · Quantization fundamentals and LLM outliers
intermediate · wave 2 · prereqs: `llm.kv-cache`

Owns inference/weight quantization as handed over by the Applied plan. Floating-point format basics (`sys.fp-formats`) and FP8 *training* (`sys.fp8-training`) live there; link rather than re-teach.

**Primary sources**
- Dettmers et al. 2022, *LLM.int8()* §2 background (absmax and zero-point quantization formulas), §3.1 vector-wise quantization, §3.2 mixed-precision decomposition (outlier dims in fp16, threshold 6.0), §4 emergent outlier features from ~6.7B parameters (p.3–9). https://arxiv.org/abs/2208.07339 · `llm_dettmers2022_int8.txt`
- Xiao et al. 2022 (SmoothQuant) §3 activation outliers are per-channel and persistent, §4 smoothing $s_j=\max|X_j|^\alpha/\max|W_j|^{1-\alpha}$ with $\alpha=0.5$ default (p.3–5). https://arxiv.org/abs/2211.10438 · `llm_xiao2022_smoothquant.txt`
- Lin et al. 2023 (AWQ) §3.1 ~1% salient weights matter, §3.2 activation-aware per-channel scaling and its error analysis (p.2–5). https://arxiv.org/abs/2306.00978 · `llm_lin2023_awq.txt`
- JAX Scaling Book Part 7 (quantized weights shift the decode roofline). `llm_scalingbook_inference.txt`
- Lilian Weng, *Large Transformer Model Inference Optimization* — quantization section (PTQ vs QAT; outliers). `llm_weng2023_inference_optimization.txt`

**Subtopic map**
1. *Why quantize:* decode is memory-bound ⇒ fewer bytes per weight → proportionally faster decode and bigger batches; int8/int4 tensor-core throughput for compute-bound prefill (W8A8).
2. *Uniform quantization (derive):* symmetric absmax $q=\mathrm{round}(127\,x/\max|x|)$; asymmetric with scale and zero-point; dequantization; quantization error ≈ uniform with variance $\Delta^2/12$; worked example on a small vector (Python).
3. *Granularity:* per-tensor, per-channel (rows of $W$), per-token (rows of $X$), group-wise (e.g. 64/128 weights share a scale); cost of storing scales (bits/weight overhead: worked, 16-bit scale per 128 weights = +0.125 bit).
4. *Weight-only vs weight+activation:* W4A16 for memory-bound decode; W8A8 needs activation quantization ⇒ outlier problem.
5. *Outlier features (LLM.int8):* a few hidden dimensions with magnitudes ≫ others, systematic across tokens, appear at scale; per-tensor activation int8 then wastes range; mixed-precision decomposition keeps ~0.1% outlier dims in fp16. Link to massive activations / attention sinks.
6. *SmoothQuant:* $XW=(X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W)$; migrate difficulty from activations to weights; derive why it is exact and how $\alpha$ balances ranges.
7. *AWQ:* weight importance depends on activation magnitude; scale salient input channels of $W$ up before quantization (and inputs down) so their relative rounding error shrinks; grid search for scales; no backprop.
8. *PTQ vs QAT:* post-training methods (round-to-nearest, GPTQ, AWQ) vs quantization-aware training (Gemma 3 QAT checkpoints; straight-through estimator recap from fundamentals).
9. *Evaluation:* perplexity degradation hides task-specific damage (long-context, math); check on downstream tasks.

**Figures**
- Histogram of a synthetic activation matrix with 2 outlier channels; per-tensor vs per-channel int8 grids overlaid — reader sees most values collapse to a few levels under per-tensor scaling.
- Uniform quantizer staircase with error band.
- SmoothQuant before/after channel max-magnitude bars for $X$ and $W$ (synthetic).

**Question ideas**
- Compute: int8 absmax quantization of [0.1, −0.5, 2.0, 0.03]; reconstruction error.
- Compute: bits/weight for int4 with an fp16 scale per group of 64 (4.25).
- Predict: per-tensor int8 activations in a 13B model vs a 1B model (outliers).
- Which is false: "SmoothQuant changes the model's function slightly, so it requires fine-tuning."
- Open: "Why is weight-only int4 enough to speed up decode but not prefill?"

**Pitfalls / checks**
- LLM.int8's outlier threshold (6.0) and the 6.7B emergence point — verify §3.2/§4 wording.

---

### 310 · `llm.quantization-advanced` · GPTQ, NF4, KV-cache quantization and low-precision formats
advanced · wave 3 · prereqs: `llm.quantization` · cross-area: `sys.fp-formats` (E4M3/E5M2 bit layouts, range/precision), `sys.fp8-training` (scaling recipes, DeepSeek-V3 tile scaling, MX for training)

**Primary sources**
- Frantar et al. 2022 (GPTQ) §3 background (layer-wise objective $\|WX-\hat WX\|^2$, Optimal Brain Quantization: greedy one-at-a-time quantization with inverse-Hessian updates), §4 GPTQ algorithm (arbitrary order, lazy batch updates, Cholesky reformulation), 3–4 bits for 175B in ~4 GPU hours (p.3–6). https://arxiv.org/abs/2210.17323 · `llm_frantar2022_gptq.txt`
- Dettmers et al. 2023 (QLoRA) §3 NF4 (quantiles of $\mathcal N(0,1)$, information-theoretically optimal for normal weights), double quantization, paged optimizers (p.3–5). https://arxiv.org/abs/2305.14314 · `llm_dettmers2023_qlora.txt`
- Liu et al. 2024 (KIVI) — keys per-channel, values per-token, 2-bit KV cache (p.1–4). https://arxiv.org/abs/2402.02750 · `llm_liu2024_kivi.txt`
- Rouhani et al. 2023, *Microscaling Data Formats* (MX: shared E8M0 scale per block of 32 elements; MXFP4/6/8) — for the inference-side MXFP4 weights only. https://arxiv.org/abs/2310.10537 · `llm_rouhani2023_microscaling.txt`. (FP8 formats and FP8 training sources — Micikevicius 2022, DeepSeek-V3 §3.3 — are owned by `sys.fp-formats` / `sys.fp8-training`.)
- gpt-oss §2.1 (MoE weights post-trained/quantized to MXFP4, 4.25 bits/param; p.5). `llm_openai2025_gptoss.txt`

**Subtopic map**
1. *Layer-wise reconstruction objective:* quantize $W$ to minimize $\|WX-\hat WX\|_F^2$ on calibration data; Hessian $H=2XX^\top$.
2. *OBS/OBQ step (derive):* quantizing weight $w_q$ incurs error; optimal compensation of the remaining weights $\delta=-\frac{w_q-\mathrm{quant}(w_q)}{[H^{-1}]_{qq}}H^{-1}_{:,q}$, and the error term $\frac{(w_q-\mathrm{quant}(w_q))^2}{[H^{-1}]_{qq}}$. GPTQ's insights: fixed column order for all rows (shares $H^{-1}$), lazy blocked updates, Cholesky for numerical stability.
3. *NF4:* construct 16 levels at quantiles of the standard normal (normalized to [−1,1], with an exact zero); blockwise absmax normalization; double quantization of the scales (saves ~0.37 bits/param — check QLoRA §3).
4. *KV-cache quantization:* keys have outlier channels ⇒ per-channel; values per-token; residual recent tokens kept in full precision; 2-bit with small loss (KIVI); interaction with attention numerics.
5. *Low-precision float formats for inference weights:* one-line recap of E4M3/E5M2 with a link to `sys.fp-formats` (FP8 *training* is `sys.fp8-training`); then the inference-specific part: MXFP4 block format (FP4 E2M1 elements, 32 elements share an 8-bit power-of-two E8M0 scale ⇒ 4 + 8/32 = 4.25 bits/param) and its use for gpt-oss's MoE weights, which are 90+% of parameters and let the 120B model fit one 80 GB GPU (gpt-oss §2.1, pdf p.5); FP8 (W8A8) serving as a throughput choice.
6. *Rotation-based methods* (Hadamard incoherence: QuIP#/QuaRot; FA3's incoherent processing) — mention with pointer.
7. *Choosing a scheme:* bandwidth-bound decode → weight-only 4-bit; high-throughput serving → W8A8/FP8; long context → KV quantization.

**Figures**
- NF4 level positions vs uniform int4 levels on top of a Gaussian density.
- MXFP4 block: 32 FP4 values sharing one power-of-two scale, with an outlier forcing the scale up (synthetic numpy) — reader sees why block size trades overhead against outlier damage.
- GPTQ column sweep diagram: quantize column $j$, push error into columns $>j$.

**Question ideas**
- Derivation step: OBQ update for remaining weights.
- Compute: memory for a 70B model in NF4 with double quantization vs bf16.
- Which is false: "Values in the KV cache are best quantized per-channel."
- Compute: bits/param of MXFP4 (4.25) vs int4 with an fp16 scale per 128 weights (4.125) — and gpt-oss-120b's MoE-weight memory at 4.25 bits.

**Pitfalls / checks**
- QLoRA's double-quantization saving (0.373 bits/param, verified pdf p.5) and NF4 construction details; gpt-oss's 4.25 bits/param verified (pdf p.5).

## Unit J: Mixture of Experts

### 320 · `llm.moe` · MoE fundamentals: gating, sparsity and active parameters
core · wave 1 · prereqs: `llm.transformer-block`, `llm.params-flops`

**Primary sources**
- Shazeer et al. 2017, *Outrageously Large Neural Networks* §2 the MoE layer, §2.1 gating network (softmax gating; noisy top-k gating $H(x)_i=(xW_g)_i+\epsilon\cdot\mathrm{softplus}((xW_{noise})_i)$), §3.1 shrinking batch problem, §4 balancing (importance loss) (p.3–6). https://arxiv.org/abs/1701.06538 · `llm_shazeer2017_moe.txt`
- Fedus et al. 2021 (Switch Transformer) §2.1 simplifying sparse routing (top-1), §2.2 efficient sparse routing (capacity, aux loss), §2.4 training techniques (selective fp32 router, smaller init, expert dropout) (p.5–11). https://arxiv.org/abs/2101.03961 · `llm_fedus2021_switch.txt`
- Lepikhin et al. 2020 (GShard) §2.2 position-wise MoE layer (top-2 gating, random second-expert routing, group-level capacity) (p.4–6). https://arxiv.org/abs/2006.16668 · `llm_lepikhin2020_gshard.txt`
- Jiang et al. 2024 (Mixtral) §2 architecture (8 experts, top-2; "47B total, 13B active"). https://arxiv.org/abs/2401.04088 · `llm_jiang2024_mixtral.txt`
- HF blog, *Mixture of Experts Explained* (history, terminology, fine-tuning issues). https://huggingface.co/blog/moe · `llm_hf_moe_explained.txt`; JAX Scaling Book Part 4 *Sparsity and Mixture-of-Experts*. `llm_scalingbook_transformers.txt`
- Komatsuzaki et al. 2022, *Sparse Upcycling* §3 (initialize experts from a dense checkpoint). https://arxiv.org/abs/2212.05055 · `llm_komatsuzaki2022_upcycling.txt`

**Subtopic map**
1. *Motivation:* decouple parameter count (capacity/knowledge) from per-token compute; conditional computation.
2. *The MoE layer:* replace the FFN with $E$ expert FFNs and a router $g(x)=\mathrm{softmax}(xW_r)$; output $y=\sum_{i\in\mathrm{TopK}}g_i(x)\,E_i(x)$; renormalize the top-k weights or not (variants).
3. *Gating formulations:* softmax-then-top-k vs top-k-then-softmax; noisy top-k (Shazeer) for exploration; top-1 (Switch: simpler, less communication; the gate value still scales the expert output so the router gets gradient); top-2 (GShard, Mixtral); sigmoid affinities (DeepSeek-V3). Why the router gets gradients only through the selected experts' gate values.
4. *Active vs total parameters (worked):* Mixtral 8×7B: ~47B total, ~13B active (not 56B: attention and embeddings are shared — compute the split from the config); DeepSeek-V3: 671B total, 37B active; FLOPs/token ≈ $2N_{active}$, memory ∝ $N_{total}$.
5. *Dense vs MoE at equal FLOPs:* MoE reaches a given loss faster in training (Switch's speedups); why: more parameters per FLOP; diminishing returns with more experts.
6. *Expert specialization:* what experts specialize on (token/syntax-level more than topic-level in Mixtral's analysis); routing is per token per layer.
7. *Shrinking batch problem:* each expert sees only $kB/E$ tokens ⇒ low arithmetic intensity unless batches are large (pointer to `llm.moe-systems`).
8. *Training tricks:* router in fp32, small init, router z-loss (next lesson), expert dropout for fine-tuning; MoE fine-tuning overfits more easily (Switch/ST-MoE observations).
9. *Upcycling:* copy a dense FFN into all experts, add a fresh router, continue training — cheap MoE start; symmetry-breaking concern.

**Figures**
- MoE layer diagram: tokens → router scores → top-2 → experts → weighted sum, with one token's path highlighted.
- Bar: total vs active parameters for Mixtral 8×7B, DeepSeek-V3, Qwen3-235B-A22B, gpt-oss-120b (configs from cached reports; verify each).
- Router probability simplex/heatmap for a batch of tokens across 8 experts.

**Question ideas**
- Compute: total and active params of a 32-layer MoE with $d=4096$, 8 experts of SwiGLU size 14,336, top-2, plus attention as in Llama-3-8B.
- Predict: switch from top-2 to top-1 at fixed experts — FLOPs, communication, quality.
- Which is false: "With top-1 routing the router receives no gradient."
- Compare: dense 13B vs MoE 47B/13B active — inference memory and latency at batch 1.
- Open: "Explain how MoE decouples parameters from FLOPs, and what it costs."

**Pitfalls / checks**
- Mixtral numbers: paper abstract says 47B/13B; the often-quoted 46.7B/12.9B is from the Mistral blog — cite accordingly.
- Shazeer's noisy gating formula: verify §2.1 notation.

---

### 330 · `llm.moe-load-balancing` · Load balancing, capacity and router stability
intermediate · wave 1 · prereqs: `llm.moe`

**Primary sources**
- Fedus 2021 (Switch) §2.2: expert capacity $=\frac{\text{tokens per batch}}{\text{experts}}\times$ capacity factor; token dropping; differentiable load-balancing loss $\alpha N\sum_i f_iP_i$ with $\alpha=10^{-2}$; Fig. 3 (p.6–8). `llm_fedus2021_switch.txt`
- Zoph et al. 2022 (ST-MoE) §3 stabilizing training: router z-loss $L_z=\frac1B\sum_j(\log\sum_i e^{x_{ij}})^2$, $c_z=10^{-3}$; App. B (p.5–8, 31). https://arxiv.org/abs/2202.08906 · `llm_zoph2022_stmoe.txt`
- Zhou et al. 2022, *Expert Choice Routing* §3.1 pitfalls of token choice, §3.2 experts choose top-$k$ tokens, §3.3 capped variant (p.3–5). https://arxiv.org/abs/2202.09368 · `llm_zhou2022_expert_choice.txt`
- Wang et al. 2024, *Auxiliary-Loss-Free Load Balancing* §2.2 interference of aux loss, §3 bias-adjusted top-k (bias used for selection only; $b_i\leftarrow b_i+u\,\mathrm{sign}(e_i)$), §5.2 future-token leakage of expert choice (p.3–8). https://arxiv.org/abs/2408.15664 · `llm_wang2024_auxfree.txt`
- DeepSeek-V3 §2.1.2 aux-loss-free strategy + complementary sequence-wise balance loss ($\alpha=10^{-4}$), node-limited routing, no token dropping (p.8–10, 22–23). `llm_deepseek2024_v3.txt`
- Gale et al. 2022 (MegaBlocks) §3 token dropping vs capacity, §4 dropless block-sparse expert computation (p.3–5). https://arxiv.org/abs/2211.15841 · `llm_gale2022_megablocks.txt`

**Subtopic map**
1. *Routing collapse:* positive feedback — popular experts get trained more, get chosen more; dead experts.
2. *Switch auxiliary loss (derive):* $f_i$ = fraction of tokens dispatched to expert $i$ (non-differentiable count), $P_i$ = mean router probability for $i$ (differentiable). $L=\alpha N\sum_i f_iP_i$. Show that under uniform routing $f_i=P_i=1/N$ ⇒ $L=\alpha$; gradient flows through $P_i$ weighted by $f_i$, pushing probability away from overloaded experts. Be precise about "minimized at uniform" (Switch §2.2 asserts it, pdf p.7): for *fixed* $f$, $\sum_if_iP_i$ is minimized by moving all of $P$ onto the least-loaded expert, so the uniform optimum is a statement about the joint fixed point; with $f\approx P$, $N\sum_iP_i^2\ge(\sum_iP_i)^2=1$ by Cauchy–Schwarz, with equality iff $P$ is uniform. That is the argument an interviewer wants. Why multiply by $N$ (keeps the loss scale constant as $N$ changes). Relation to Shazeer's importance (CV²) loss and GShard's version.
3. *Capacity factor and token dropping:* fixed buffers per expert for static shapes; overflowing tokens skip the layer (residual passes through); CF 1.0–1.25 typical in Switch; trade-off compute/communication vs dropped tokens; dropless MoE (MegaBlocks: block-sparse matmuls, variable-size groups).
4. *Router z-loss (derive):* penalize large router logits' log-sum-exp; reduces round-off error in exp under bf16 and stabilizes training; coefficient $10^{-3}$; relation to the output-layer z-loss in `llm.training-stability`.
5. *Expert-choice routing:* each expert picks its top-$k$ tokens (perfect balance by construction, variable experts per token); causality problem for autoregressive decoding (selection depends on other tokens in the batch/sequence = future-token leakage).
6. *Auxiliary-loss-free balancing (DeepSeek):* add a per-expert bias to scores only for top-k selection, not for gate values; update the bias by ±u according to over/under-load after each step; avoids the gradient interference that aux losses cause; V3 still adds a tiny sequence-wise loss.
7. *Sequence-, device- and node-level balance:* why balance at batch level can hide per-sequence imbalance; device-limited/node-limited routing to bound communication.
8. *Instabilities specific to MoE:* bf16 router logits, fine-tuning instability, expert dropout; ST-MoE's recommendations.
9. *Soft MoE (contrast):* every slot is a weighted average of all tokens ⇒ no dropping, fully differentiable, but not causal (vision only) (Puigcerver 2023; `llm_puigcerver2023_soft_moe.txt`).

**Figures**
- Token counts per expert over training without vs with balancing (simulated router with rich-get-richer dynamics in numpy).
- Capacity diagram: 4 experts, CF 1.0 vs 1.5, overflow tokens marked (re-draw of Switch Fig. 3 idea).
- Aux loss value $N\sum f_iP_i$ for uniform vs skewed routing (bar), plus the bias-update trace for aux-free balancing.

**Question ideas**
- Compute: Switch loss for $N=4$, $f=(0.4,0.3,0.2,0.1)$, $P=(0.35,0.3,0.2,0.15)$ (α omitted).
- Derivation step: why $f_i$ can appear in the loss even though it is not differentiable.
- Predict: capacity factor 1.0 → 2.0 — effect on dropped tokens, memory, step time.
- Which is false: "Expert-choice routing can be used unchanged for autoregressive decoding."
- Compare: aux loss vs bias-based balancing — what each does to the language-modelling gradient.
- Open: "Derive the Switch load-balancing loss and explain each factor."

**Pitfalls / checks**
- Switch's $f_i$ and $P_i$ definitions (fractions over tokens in the batch) — copy exactly.
- DeepSeek-V3 bias speed γ=0.001 for 14.3T tokens then 0 (§4.2).

---

### 340 · `llm.moe-systems` · Fine-grained & shared experts, expert parallelism and MoE inference
intermediate · wave 2 · prereqs: `llm.moe-load-balancing`, `llm.kv-cache`

Owns the MoE-systems material handed over by the Applied plan (`content-plan-applied.md`): expert parallelism, all-to-all cost, capacity/token dropping, load imbalance. Link to `sys.collectives` / `sys.parallelism-composition` for collective algorithms rather than re-deriving them.

**Primary sources**
- Dai et al. 2024 (DeepSeekMoE) §3.1 fine-grained expert segmentation (split each expert into $m$ smaller ones, activate $mK$: $\binom{16}{2}=120$ vs $\binom{64}{8}\approx4.4\times10^9$ combinations), §3.2 shared expert isolation, §3.3 expert- and device-level balance losses (p.5–7). https://arxiv.org/abs/2401.06066 · `llm_dai2024_deepseekmoe.txt`
- DeepSeek-V3 §2.1.2 (1 shared + 256 routed, top-8, expert hidden 2048), §3.2 DualPipe and cross-node all-to-all, node-limited routing (≤4 nodes), §3.4 inference deployment (prefill/decode expert parallel sizes, redundant experts) (p.8–20). `llm_deepseek2024_v3.txt`
- JAX Scaling Book GPUs part *Expert Parallelism* roofline. `llm_scalingbook_gpus.txt`; Part 4 *Sparsity and Mixture-of-Experts*. `llm_scalingbook_transformers.txt`
- Krajewski et al. 2024, *Scaling Laws for Fine-Grained MoE* §4 granularity, §5 parametric law with granularity. https://arxiv.org/abs/2402.07871 · `llm_krajewski2024_fine_grained_moe_scaling.txt`
- Raschka *Big LLM Architecture Comparison* §9.2 few large vs many small experts; §12.1 Qwen3-Next expert sizes. `llm_raschka_big_arch_comparison.txt`

**Subtopic map**
1. *Fine-grained experts:* same total/active params, more and smaller experts with larger top-k ⇒ combinatorially more expert mixtures; granularity as a scaling-law variable (Krajewski).
2. *Shared experts:* always-on expert(s) capture common knowledge so routed experts specialize; DeepSeekMoE/V3 (1 shared), Qwen3-MoE excludes shared experts (Qwen3 report §2, pdf p.3: 128 experts, 8 active) — design not settled.
3. *Expert parallelism:* experts sharded across devices; each MoE layer needs all-to-all dispatch and combine; communication volume per layer ≈ $2\cdot B\cdot T\cdot k\cdot d\cdot$bytes (dispatch + return; derive; worked: 4096 tokens, k=8, d=7168, bf16 ≈ 0.94 GB per layer, Python-verified); overlap with compute (DualPipe).
3b. *Why EP stays intra-node (bandwidth arithmetic):* cross-node InfiniBand gives ~50 GB/s per GPU (H100 node: ~400 GB/s egress shared by 8 GPUs, Scaling Book GPU part; DeepSeek-V3 §3.2.2: IB 50 GB/s vs NVLink 160 GB/s on H800, ≈3.2×) vs ~450 GB/s NVLink per H100 inside a node. Compute the all-to-all time for the worked example at both bandwidths vs the expert matmul time; hence EP within the NVLink domain, and DeepSeek's node-limited routing (each token to ≤4 nodes) plus dedup of cross-node sends.
3c. *Capacity, token dropping and load imbalance at the systems level:* the slowest (most loaded) expert/device sets the step time, so imbalance costs throughput even without drops; capacity factor fixes buffer sizes for static shapes (recap from `llm.moe-load-balancing`); dropless block-sparse kernels (MegaBlocks); redundant/hot-expert replication at inference (DeepSeek-V3 §3.4).
4. *Interplay with other parallelism* (pointer to applied area): EP × DP × TP × PP.
5. *MoE inference economics:* memory for all experts; at batch 1 each token reads only its active experts' weights, but across a batch the union of experts touched grows quickly ⇒ bytes read per step approach total params at moderate batch; high throughput needs large batches (DeepSeek-V3 deploys large EP groups); decode latency vs dense model of equal active params.
6. *MoE scaling laws:* loss vs total params at fixed active compute; granularity optimum grows with budget (Krajewski) — present carefully as one paper's fit.
7. *Configs to compare (verify each from cached reports):* Mixtral 8×7B (8 experts, top-2), DeepSeek-V3 (256+1, top-8), Qwen3-235B-A22B (128 experts, top-8), gpt-oss-120b (128 experts, top-4), Kimi K2 (384 experts, top-8). Each number must be checked against its report.
8. *Upcycling and dense-to-MoE conversion* (recap), MoE distillation into dense (mention).

**Figures**
- Fine-grained segmentation diagram (DeepSeekMoE Fig. 2 idea): conventional top-2 of 16 vs fine-grained top-8 of 64 vs + shared expert.
- All-to-all dispatch/combine across 4 devices for 8 tokens.
- Expected number of distinct experts touched vs batch size for E=256, k=8 under uniform routing: $E(1-(1-k/E)^B)$ (derive; Python-plot).

**Question ideas**
- Compute: distinct experts touched at batch 32 with E=256, k=8, uniform routing (≈163, Python-verified).
- Compute: all-to-all bytes per MoE layer for 4096 tokens, k=8, d=7168, bf16.
- Predict: replace 1 shared + 255 routed with 256 routed — what changes.
- Which is false: "At batch size 1, an MoE model reads all expert weights every decode step."
- Open: "Why do MoE models need large serving batches to be cost-effective?"

**Pitfalls / checks**
- Model configs (expert counts, top-k, shared experts) — verify from `llm_yang2025_qwen3.txt`, `llm_openai2025_gptoss.txt`, `llm_kimi2025_k2.txt`, `llm_deepseek2024_v3.txt`.
- Bandwidths are hardware-generation specific (H800 NVLink 160 GB/s in DeepSeek-V3 vs H100 ~450 GB/s); always attach the GPU type.

---

### 350 · `llm.mot-mod` · Beyond token-level MoE: Mixture-of-Transformers and Mixture-of-Depths
advanced · wave 2 · prereqs: `llm.moe`

**Primary sources**
- Liang et al. 2024/2025 (Mixture-of-Transformers, TMLR) §2.2 modality-specific parameter decoupling (separate FFN, attention projections and norms per modality; global self-attention over the full mixed sequence), §3 results (Chameleon setting: dense quality at 55.8% of FLOPs; speech at 37.2%; Transfusion 7B: image quality at 47.2% wall-clock), §4 leave-one-out analysis, §6 systems aspects (p.1–6 and later sections). https://arxiv.org/abs/2411.04996 · `llm_liang2024_mot.txt`
- Raposo et al. 2024 (Mixture-of-Depths) §3.1 compute budget via capacity, §3.2 routing around blocks (residual skip), §3.3 expert-choice routing scheme, §3.5 sampling (non-causal top-k; auxiliary predictor/loss for autoregressive decoding), §4.1 isoFLOP results (e.g. every-other-block routing with 12.5% capacity), §4.3 MoDE (p.3–10). https://arxiv.org/abs/2404.02258 · `llm_raposo2024_mod.txt`
- Graves 2016 (ACT) as background for adaptive depth (shared with `llm.looped-transformers`). `llm_graves2016_act.txt`

**Subtopic map**
1. *Three axes of conditional compute:* which parameters (MoE: per token, learned router), which modality's parameters (MoT: deterministic by modality), how much depth (MoD: whether a token passes through a block).
2. *MoT design:* each modality (text, image, speech) gets its own copy of all non-embedding transformer weights (QKV/O projections, FFN, norms); attention is computed jointly over all tokens, so modalities still interact; routing is by known modality label — no load balancing, no learned router. FLOPs per token = dense; parameter count × number of modalities.
3. *Why it helps:* reduces interference between modalities with different statistics; reported FLOP savings to match dense quality (55.8% / 37.2% / 47.2% wall-clock — quote with setting); leave-one-out analysis; combining MoT with MoE.
4. *MoT vs MoE comparison:* deterministic vs learned routing; MoE experts are FFN-only, MoT decouples attention projections too; systems implications (grouping tokens by modality; no all-to-all imbalance).
5. *MoD design:* per block, a router scores tokens; only top-$k$ (capacity $C<T$) are processed by attention+MLP, the rest skip via the residual; static compute graph because $k$ is fixed; output multiplied by router weight so the router learns.
6. *MoD's causality problem:* top-$k$ over the sequence uses future tokens; fixes: auxiliary binary classifier on router outputs or a small predictor MLP, used at sampling time.
7. *MoD results and MoDE:* isoFLOP gains, faster steps; combined with MoE.
8. *Related:* early exit, CoLT5 conditional routing, adaptive computation (pointer).

**Figures**
- Three-panel diagram: MoE (per-token FFN routing), MoT (per-modality full-block weights with shared global attention), MoD (some tokens skip a block).
- MoD capacity picture: sequence of 16 tokens, 4 selected per block, across 4 blocks (which tokens get compute).

**Question ideas**
- Compute: parameter count of an MoT with 3 modalities built from a 7B dense model (non-embedding ×3) and its FLOPs/token.
- Predict: MoD with capacity 12.5% every other block — FLOPs per forward relative to dense.
- Which is false: "MoT needs an auxiliary load-balancing loss."
- Compare: MoT vs a 3-expert MoE routed by a learned router.
- Open: "Why is top-k token selection in MoD problematic for autoregressive sampling, and how is it fixed?"

**Pitfalls / checks**
- MoT FLOP-savings figures are setting-specific (Chameleon 7B, Transfusion); quote with context. MoT is TMLR 04/2025 (arXiv 2411.04996).
## Unit K: Long context

### 360 · `llm.long-context` · Long-context architectures: segment recurrence, ring attention, KV compression
intermediate · wave 2 · prereqs: `llm.context-extension`, `llm.kv-cache`, `llm.flash-attention`

**Primary sources**
- Dai et al. 2019 (Transformer-XL) §3.2 segment-level recurrence with state: cache previous segment's hidden states with stop-gradient $\mathrm{SG}(h^{n-1}_{\tau})$, concatenate as extra keys/values; effective dependency length $O(N\times L)$; §3.3 why relative positions are needed; §4.3 relative effective context length; §4.5 evaluation speed (p.3–8). https://arxiv.org/abs/1901.02860 · `llm_dai2019_transformerxl.txt`
- (Background only; owned by `sys.sequence-context-parallel`) Liu, Zaharia & Abbeel 2023 (Ring Attention) §3 blockwise attention distributed over a ring of devices, overlap of KV-block communication with compute; condition block size ≥ FLOPs/bandwidth ratio (p.3–5). https://arxiv.org/abs/2310.01889 · `llm_liu2023_ring.txt`
- Xiao 2023 (StreamingLLM) §3.2 rolling cache with sinks. `llm_xiao2023_attention_sinks.txt`; Zhang et al. 2023 (H2O) heavy-hitter eviction. https://arxiv.org/abs/2306.14048 · `llm_zhang2023_h2o.txt`
- Llama 3 §3.4.2 long-context pre-training (8K→128K in six stages, ~800B tokens; check), and §3.3 context parallelism (all-gather-based CP). `llm_dubey2024_llama3.txt`
- Weng, *The Transformer Family v2*: sections on context memory, recurrence, sparse/efficient attention (survey). https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/ · `llm_weng2023_transformer_family_v2.txt`

**Subtopic map**
1. *What limits context:* (a) attention compute $O(n^2)$ in prefill/training, (b) KV memory $O(n)$ per sequence at decode, (c) position extrapolation (`llm.context-extension`), (d) long training data, (e) whether the model actually *uses* far context (next lesson).
2. *Transformer-XL segment recurrence (derive):* process segment $\tau+1$ with keys/values from $[\mathrm{SG}(h_\tau^{n-1});h_{\tau+1}^{n-1}]$; the dependency length grows by one segment per layer ⇒ $O(NL)$; stop-gradient keeps training cost per segment constant; requires relative positions (absolute indices would collide across segments); evaluation speed-up from reusing states. Compressive Transformer as an extension (mention).
3. *Memory-augmented / retrieval-based context* (kNN memory, RETRO — pointer to `llm.rag`).
4. *Sequence/context parallelism (summary + link; owned by `sys.sequence-context-parallel`):* one paragraph — split the sequence across devices; ring attention rotates K/V blocks and merges partial results with the online-softmax statistics from `llm.flash-attention`; Llama 3 uses all-gather-based CP instead (§3.3). The merge derivation, the overlap condition and zig-zag balancing are derived in the sys lesson; don't repeat them. Transformer-specific number to keep: the K/V block each device ships per layer per hop is $2c\,h_{kv}d_h$ elements, so GQA-8 cuts CP traffic 8× vs MHA — worked: Llama-3-70B, 1M tokens over 16 devices ($c=65{,}536$), bf16: 256 MiB per layer per hop vs 2 GiB with 64 KV heads (Python-verified).
5. *KV compression at inference:* eviction policies (window + sinks; heavy hitters by accumulated attention; risks: evicted information is gone), quantization, cross-layer sharing, latent compression (MLA), compressed/sparse attention (DSA, V4's CSA/HCA). Compare by what they lose.
6. *Architectural long-context designs in 2025–26 models:* local/global interleaves (Gemma 3), hybrids with linear attention (Qwen3-Next/3.5, Kimi Linear), DeepSeek V4 1M-token context via compressed attention — mark specifics as recent.
7. *Cost model:* worked example — prefill FLOPs and KV memory for a 1M-token prompt on a 70B GQA model (Python; show attention dominates).

**Figures**
- Transformer-XL dependency graph: segments × layers with arrows from cached segment; receptive field widening per layer (re-draw of their Fig. 2 idea).
- KV-compression gallery: one 64-token strip per method (sliding window, window + 4 sinks, heavy-hitter eviction, compressed blocks + top-k selection) showing which tokens survive — reader sees what each method throws away. (Ring-attention schedule figures belong to `sys.sequence-context-parallel`.)
- Prefill FLOP split (linear vs attention) vs context length up to 1M for a 70B GQA model.

**Question ideas**
- Compute: Transformer-XL with 16 layers and segment length 512 — maximum dependency length (≈8k).
- Compute: per-hop K/V bytes per layer for context parallelism on Llama-3-70B at 1M tokens over 16 devices, GQA-8 vs MHA.
- Predict: H2O-style eviction on a task that needs a specific early token after 50k tokens.
- Which is false: "Transformer-XL backpropagates through the cached previous segment."
- Open: "Walk through the bottlenecks of serving a 1M-token context and the fix for each."

**Pitfalls / checks**
- Llama 3 long-context stage details (number of stages, tokens) — verify §3.4.2.

---

### 370 · `llm.long-context-eval` · Does the model use its context? Evaluation and training for long context
intermediate · wave 2 · prereqs: `llm.long-context`

**Primary sources**
- Liu et al. 2023, *Lost in the Middle* §2 multi-document QA, §3 key–value retrieval; U-shaped accuracy vs position; §4 why (architecture, instruction tuning, query-aware contextualization) (p.2–8). https://arxiv.org/abs/2307.03172 · `llm_liu2023_lost_middle.txt`
- Hsieh et al. 2024 (RULER) §3 task categories (retrieval incl. multi-key/multi-value NIAH, multi-hop tracing, aggregation, QA), effective vs claimed length (only about half of models claiming ≥32K performed satisfactorily at 32K) (p.1–6). https://arxiv.org/abs/2404.06654 · `llm_hsieh2024_ruler.txt`
- Fu et al. 2024, *Data Engineering for Scaling LMs to 128K Context* (per-source length upsampling, ~5B tokens of continual pretraining). https://arxiv.org/abs/2402.10171 · `llm_fu2024_data_128k.txt`
- Xiong et al. 2023 §4–5 (long-context continual pretraining, data mix, short-task retention). `llm_xiong2023_long_context_scaling.txt`
- Press 2021 App. B (sliding-window evaluation and the early-token curse) for perplexity-evaluation pitfalls. `llm_press2021_alibi.txt`

**Subtopic map**
1. *Perplexity is not enough:* long-context perplexity improves mostly from local context; sliding-window evaluation artefacts.
2. *Needle-in-a-haystack:* setup (depth × length grid), why it saturates; multi-needle and distractor variants.
3. *RULER:* synthetic task families, effective context length definition (threshold vs a baseline), findings.
4. *Lost in the middle:* U-shaped position effect (primacy/recency); dependence on model and instruction tuning; mitigation (reordering retrieved docs, training on position-diverse data).
5. *Real tasks:* long-document QA, multi-hop over long inputs, repository-level code; LongBench-style suites (mention).
6. *Training recipes:* continual pretraining with upsampled long documents while preserving domain mix (Fu), progressive length increase, RoPE base change, synthetic long-range tasks; mixing short data to retain short-task quality.
7. *Long-context vs RAG trade-off* (pointer to `llm.rag`).

**Figures**
- NIAH heatmap (context length × needle depth) for a synthetic model with mid-context degradation (schematic, labelled).
- Accuracy vs answer position (U-shape) — replot from Liu 2023 numbers with citation or schematic.

**Question ideas**
- Predict: a model passes single-needle NIAH at 128k but fails RULER multi-hop tracing at 32k — what does that say?
- Which is false: "Lower perplexity at 64k tokens shows the model uses information 50k tokens back."
- Compare: upsampling long documents vs concatenating short documents to reach 128k.

**Pitfalls / checks**
- RULER's "half of models" claim refers to their tested set at publication time.

## Unit L: Recurrent alternatives and architecture comparison

### 380 · `llm.linear-attention` · Linear attention: the kernel view and attention as an RNN
intermediate · wave 2 · prereqs: `llm.self-attention`, `llm.causal-attention`

**Primary sources**
- Katharopoulos et al. 2020, *Transformers are RNNs* §3.1 generalized attention with a similarity function, §3.2 linearized attention $V'_i=\frac{\phi(Q_i)^\top\sum_j\phi(K_j)V_j^\top}{\phi(Q_i)^\top\sum_j\phi(K_j)}$, feature map $\mathrm{elu}+1$, §3.3 causal masking as running sums ($S_i=S_{i-1}+\phi(K_i)V_i^\top$, $Z_i=Z_{i-1}+\phi(K_i)$), §3.4 transformers are RNNs (p.2–5). https://arxiv.org/abs/2006.16236 · `llm_katharopoulos2020_linear.txt`
- Choromanski et al. 2020 (Performer) — positive random features approximating the softmax kernel (FAVOR+). https://arxiv.org/abs/2009.14794 · `llm_choromanski2020_performer.txt`
- Sun et al. 2023 (RetNet) §2.1 retention: parallel, recurrent and chunkwise forms with decay $\gamma$ (p.3–5). https://arxiv.org/abs/2307.08621 · `llm_sun2023_retnet.txt`
- Yang et al. 2024 (Gated DeltaNet) §2.1 Mamba2 as linear attention with decay, §2.2 DeltaNet (delta rule as online regression: $S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top$), §3.1 gated delta rule, §3.2 S-NIAH case study (p.2–5). https://arxiv.org/abs/2412.06464 · `llm_yang2024_gated_deltanet.txt`
- d2l §11.2 (attention pooling as kernel regression) for the kernel-smoother view. Raschka *Big Arch Comparison* §14 (linear-attention revival, Kimi Delta Attention) for recent adoption. `llm_raschka_big_arch_comparison.txt`

**Subtopic map**
1. *Attention as a kernel smoother:* $o_i=\sum_j \frac{\kappa(q_i,k_j)}{\sum_l\kappa(q_i,k_l)}v_j$ with $\kappa=\exp(q^\top k/\sqrt d)$.
2. *Linearization (derive):* if $\kappa(q,k)=\phi(q)^\top\phi(k)$, associativity gives $\phi(q_i)^\top(\sum_j\phi(k_j)v_j^\top)$ ⇒ $O(n d_\phi d)$ instead of $O(n^2d)$.
3. *Causal case = RNN (derive):* state $S_t=S_{t-1}+\phi(k_t)v_t^\top\in\mathbb{R}^{d_\phi\times d}$, normalizer $z_t$; $O(1)$ per-token decode cost and fixed memory — no KV cache growth.
4. *Feature maps:* elu+1, random features approximating softmax (Performer; unbiased positive features), learned maps; approximation quality issues.
5. *The fixed-state bottleneck:* state is a $d_\phi\times d$ matrix — it can store only ~$d_\phi$ key–value associations without interference (superposition/collision argument); recall degrades as $n$ grows. Contrast with softmax attention's growing cache.
6. *Decay and gating:* RetNet $S_t=\gamma S_{t-1}+k_tv_t^\top$ (exponential forgetting; parallel form with a decay mask $D_{nm}=\gamma^{n-m}$), data-dependent gates (GLA, Mamba-2 as scalar data-dependent decay).
7. *Delta rule (derive):* treat $S$ as a linear associative memory trained online to map $k_t\mapsto v_t$; one SGD step on $\frac12\|Sk_t-v_t\|^2$ gives $S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top$ — overwrites the old value for a repeated key instead of adding. Gated DeltaNet combines decay + delta rule.
8. *Chunkwise-parallel training:* inter-chunk recurrence + intra-chunk attention-like computation; why this is needed for GPU efficiency.
9. *2025–26 adoption:* hybrid stacks — Qwen3-Next/3.5: Gated DeltaNet : gated full attention = 3:1 (Raschka 2026, §on Qwen3-Coder-Next/Qwen3.5); Kimi Linear: KDA : MLA = 3:1, chosen by ablation (Kimi Linear §2 and Table 1, pdf p.2, 8). Ratios verified in review; mark as recent.

**Figures**
- Compute/memory comparison: softmax attention ($n\times n$ matrix) vs linear attention (running $d\times d$ state) as data-flow diagrams.
- Recall experiment (numpy): store $n$ random key–value pairs in a $d\times d$ outer-product memory vs delta-rule memory; retrieval error vs $n$ for $d=64$ — reader sees interference beyond ~$d$ pairs and delta rule's advantage.
- Parallel vs recurrent forms of retention: decay mask heatmap.

**Question ideas**
- Derivation: write causal linear attention as a recurrence.
- Compute: state size per layer for $d_\phi=d=128$ and 32 heads (fp16) vs a KV cache at 32k tokens.
- Predict: associative recall with 1,000 pairs in a single-head linear attention layer with $d=64$.
- Which is false: "Linear attention with the elu+1 feature map computes exactly the same outputs as softmax attention."
- Open: "Derive the delta rule as online regression and explain why it helps recall."

**Pitfalls / checks**
- Hybrid ratios verified in review (3:1 for both); recheck if a newer report is cited.

---

### 390 · `llm.ssm` · State space models: S4 from continuous dynamics to convolution
intermediate · wave 2 · prereqs: `llm.linear-attention`

**Primary sources**
- Gu, Goel & Ré 2021 (S4) §2.1 continuous SSM $x'=Ax+Bu$, $y=Cx+Du$; §2.2 HiPPO matrix for long-range memory; §2.3 discretization (bilinear) → recurrence; §2.4 convolutional view with kernel $\bar K=(C\bar B, C\bar A\bar B,\dots)$; §3.1–3.3 diagonalization problems, NPLR parameterization, complexity (p.3–7). https://arxiv.org/abs/2111.00396 · `llm_gu2021_s4.txt`
- Gu et al. 2020 (HiPPO) — online function approximation with Legendre measures (motivation only). https://arxiv.org/abs/2008.07669 · `llm_gu2020_hippo.txt`
- Orvieto et al. 2023 (LRU) §3 linear diagonal complex recurrences match S4; stable exponential parameterization; normalization (p.5–9). https://arxiv.org/abs/2303.06349 · `llm_orvieto2023_lru.txt`
- Gu & Dao 2023 (Mamba) §2 SSM background (ZOH discretization $\bar A=\exp(\Delta A)$, $\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\Delta B$; LTI ⇒ convolution) (p.2–4). `llm_gu2023_mamba.txt`

**Subtopic map**
1. *Continuous-time linear SSM:* state $x(t)\in\mathbb{R}^N$ driven by input $u(t)$; why continuous: resolution invariance, principled initialization.
2. *Discretization (derive ZOH):* solve $x'=Ax+Bu$ with $u$ constant over $[t,t+\Delta]$: $x_{k}=e^{\Delta A}x_{k-1}+A^{-1}(e^{\Delta A}-I)Bu_k$. Bilinear alternative (S4). Interpretation of $\Delta$ as step size / how much to integrate the input vs keep the state.
3. *Recurrent view:* $x_k=\bar Ax_{k-1}+\bar Bu_k$, $y_k=Cx_k$ — $O(N)$ per step inference.
4. *Convolutional view (derive):* unroll ⇒ $y_k=\sum_j C\bar A^{j}\bar Bu_{k-j}$ ⇒ $y=\bar K*u$; train in parallel with FFT in $O(L\log L)$. Only valid because parameters are time-invariant (LTI).
5. *Long-range memory and HiPPO:* random $A$ forgets quickly (eigenvalues inside unit circle → exponential decay); HiPPO $A$ gives optimal polynomial projection of history; stability conditions (eigenvalue real parts negative).
6. *Making it fast:* computing $\bar K$ naively needs powers of $\bar A$; diagonal-plus-low-rank (NPLR) + Cauchy kernels in S4; diagonal variants (S4D/DSS) and LRU show diagonal complex recurrences suffice with good init.
7. *Deep SSM block:* SSM per channel (SISO) + mixing MLP/gating; H3 (mention).
8. *Limitation:* LTI ⇒ the same filter for every input — cannot select or ignore content (e.g. selective copying fails) ⇒ motivates Mamba.

**Figures**
- Impulse response kernels $\bar K$ for a few diagonal SSM channels with different decay/oscillation.
- Recurrent vs convolutional computation diagram on a length-8 sequence.
- Eigenvalues of $\bar A$ in the complex plane for stable vs unstable parameterizations.

**Question ideas**
- Derivation: ZOH discretization for scalar $a<0$: $\bar a=e^{\Delta a}$, $\bar b=(e^{\Delta a}-1)b/a$.
- Compute: kernel $\bar K$ for scalar $\bar a=0.5$, $\bar b=1$, $c=1$, first 4 taps.
- Predict: increasing $\Delta$ for a channel — memory horizon.
- Which is false: "The convolutional form remains valid when $\bar B$ depends on the current input."
- Open: "Explain how an SSM can be trained like a CNN and run like an RNN."

**Pitfalls / checks**
- S4 uses bilinear discretization; Mamba uses ZOH. Say which formula belongs to which paper.

---

### 400 · `llm.mamba` · Mamba: selective state spaces and the SSD duality
intermediate · wave 2 · prereqs: `llm.ssm`

**Primary sources**
- Gu & Dao 2023 (Mamba) §3.1 selection as compression (selective copying, induction heads tasks), §3.2 input-dependent $\Delta,B,C$ (Algorithm 2), §3.3 selective scan: hardware-aware kernel (state kept in SRAM, recomputation), §3.4 Mamba block, §3.5.1 connection to gating ($\Delta$ ↔ RNN gate; Theorem 1), §4 results (p.5–12). https://arxiv.org/abs/2312.00752 · `llm_gu2023_mamba.txt`
- Dao & Gu 2024 (Mamba-2, *Transformers are SSMs*) §2.4 overview of structured state space duality, §3 SSMs as semiseparable matrices (§3.1 matrix form $y=Mx$ with $M_{ji}=C_j^\top A_{j:i}B_i$), §4 structured masked attention, §5 SSD (scalar-times-identity $A$ ⇒ equivalence with masked linear attention with decay mask), §6 hardware-efficient chunked algorithm (p.3–20). https://arxiv.org/abs/2405.21060 · `llm_dao2024_mamba2.txt`
- Raschka, *Beyond Standard LLMs* (2025) — sections on linear-attention hybrids and SSMs in current models. https://magazine.sebastianraschka.com/p/beyond-standard-llms · `llm_raschka_beyond_standard_llms.txt`

**Subtopic map**
1. *Why selection:* LTI models can't do content-based filtering; tasks: selective copying, associative recall/induction heads.
2. *Selective SSM:* make $\Delta_t,B_t,C_t$ functions of $x_t$ (linear projections; $\Delta$ via softplus); interpretation: large $\Delta$ ⇒ reset state and focus on current token, small $\Delta$ ⇒ ignore token and persist state (derive from $\bar A=e^{\Delta A}$).
3. *Consequence:* time-varying ⇒ no convolution ⇒ need a scan. *Parallel scan (derive):* the recurrence $h_t=a_th_{t-1}+b_t$ composes associatively: $(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\ a_2b_1+b_2)$ ⇒ Blelloch scan in $O(\log L)$ depth, $O(L)$ work.
4. *Hardware-aware kernel:* expanded state ($D\times N$ per token) never written to HBM; fuse discretization+scan+output in SRAM; recompute in backward (same IO idea as FlashAttention).
5. *Mamba block:* in-projection, short causal conv, SiLU gate, selective SSM, out-projection; replaces attention+MLP; no positional encoding needed.
6. *Connection to gating (Theorem 1):* with $N=1$, $A=-1$, $B=1$, the selective SSM reduces to a gated RNN $h_t=(1-g_t)h_{t-1}+g_tx_t$ with $g_t=\sigma(\cdot)$.
7. *Mamba-2 / SSD:* restrict $A_t=a_t I$ (scalar); then the sequence map is $y=(L\circ CB^\top)x$ with a 1-semiseparable decay mask $L_{ji}=\prod_{k=i+1}^{j}a_k$ — exactly causal linear attention with data-dependent decay. Two algorithms: quadratic (attention-like, matmul-friendly) and linear (recurrent); chunked hybrid for tensor cores; larger state sizes (e.g. 64–256).
8. *Results and limits:* strong perplexity per FLOP at small/medium scale; weaker on recall/copying-heavy tasks (→ `llm.architecture-comparison`); inference: constant memory, high throughput.

**Figures**
- Selective $\Delta_t$ trace over a sequence with "important" tokens: state reset vs persist.
- Parallel scan tree for 8 elements with the associative operator.
- SSD duality picture: the semiseparable matrix $M$ (lower-triangular with decay structure) = masked attention matrix.

**Question ideas**
- Derivation: show the operator $(a,b)$ composition is associative.
- Compute: $h_4$ for $a=(0.5,0.5,1,0)$, $b=(1,1,1,1)$, $h_0=0$, and explain the effect of $a_4=0$.
- Predict: making only $B$ and $C$ input-dependent but keeping $\Delta$ fixed.
- Which is false: "Mamba requires positional encodings to know token order."
- Open: "Explain the SSD duality between Mamba-2 and linear attention."

**Pitfalls / checks**
- Theorem 1's exact parameter setting — copy from §3.5.1.

---

### 410 · `llm.modern-rnns-hybrids` · Modern RNNs and hybrids: RWKV, RetNet, Griffin, Gated DeltaNet, Jamba
intermediate · wave 2 · prereqs: `llm.mamba`

**Primary sources**
- De et al. 2024 (Griffin/Hawk) §2.1 residual block, §2.3 temporal-mixing blocks (MQA local attention, window 1024 by default; recurrent block with short conv + RG-LRU), §2.4 RG-LRU equations (recurrence gate, input gate, $a_t=a^{c\,r_t}$ with $a=\sigma(\Lambda)$, $c=8$; $\sqrt{1-a_t^2}$ input normalization), §3 scaling vs transformers, §4 efficient training, §5–6 inference and long-context/copying results (p.2–13). https://arxiv.org/abs/2402.19427 · `llm_de2024_griffin.txt`
- Peng et al. 2023 (RWKV) §3.1 token shift, WKV operator with time decay, output gating, §3.3 RNN-like inference (p.3–4). https://arxiv.org/abs/2305.13048 · `llm_peng2023_rwkv.txt`
- Sun et al. 2023 (RetNet) §2.1–2.2 retention and multi-scale decay, §3.3–3.4 training/inference cost (p.3–8). `llm_sun2023_retnet.txt`
- Lieber et al. 2024 (Jamba) §2 architecture (blocks of $l=8$ layers, attention:Mamba = 1:7, MoE every other layer), §6.1–6.2 why hybrids help (in-context learning needs attention) (p.3–11). https://arxiv.org/abs/2403.19887 · `llm_lieber2024_jamba.txt`
- Kimi Linear tech report §2.2 gated delta rule, §4 architecture (KDA:MLA hybrid ratio), §5 synthetic tests and scaling (p.3–10). https://arxiv.org/abs/2510.26692 · `llm_kimi2025_linear.txt`; Yang 2024 (Gated DeltaNet) §3. `llm_yang2024_gated_deltanet.txt`

**Subtopic map**
1. *Family tree:* linear attention (+decay) ↔ linear RNNs with diagonal recurrence ↔ SSMs; all have a fixed-size state and parallelizable training (scan or chunked).
2. *RWKV:* time-mixing with per-channel exponential decay $w$ and a bonus $u$ for the current token (WKV), token shift, receptance gate; parallel training, RNN inference; versions evolved toward data-dependent decay (mention).
3. *RetNet:* retention = linear attention with exponential decay $\gamma$ (per head, multi-scale); three equivalent forms (parallel with decay mask, recurrent, chunkwise); derive equivalence of parallel and recurrent forms on a 3-token example.
4. *Griffin/Hawk:* RG-LRU equations; why the $\sqrt{1-a_t^2}$ factor (keeps hidden-state variance bounded); Hawk = pure recurrent; Griffin = recurrent blocks interleaved with local MQA attention (2 recurrent : 1 attention); findings: matches transformer scaling, better length extrapolation, worse at copying/retrieval beyond the local window unless fine-tuned (§6).
5. *Gated DeltaNet and Kimi Delta Attention* (recap from `llm.linear-attention`): decay + delta rule; KDA's finer-grained gating.
6. *Hybrids:* motivation — attention layers provide precise retrieval; recurrent layers give cheap long-range state and small caches. Ratios: Jamba 1:7 (attention:Mamba), Griffin 1:2, Qwen3-Next 1:3 (full attention:Gated DeltaNet), Kimi Linear 3:1 (KDA:MLA), Nemotron-H (Mamba-2 heavy; mention). KV-cache saving ∝ fraction of attention layers (worked).
7. *Training/inference systems differences:* scans and chunked kernels; state caching for prefix reuse is harder than KV caching (state must be snapshotted per prefix); speculative decoding requires state rollback.

**Figures**
- Layer-stack diagrams of Jamba block (1 attention + 7 Mamba, MoE every other), Griffin (R, R, A), Qwen3-Next-style (3 linear : 1 full).
- KV/state memory vs context for pure transformer, 1:3 hybrid, 1:7 hybrid, pure SSM (configs stated).
- RG-LRU gate behaviour: $a_t$ vs $r_t$ for $a=0.9, 0.99$ with $c=8$.

**Question ideas**
- Compute: KV cache of a 1:7 hybrid vs the same-depth pure transformer at 256k context.
- Derivation: RetNet recurrent form from its parallel form.
- Predict: Hawk vs Griffin on a phone-book lookup task at 8k tokens.
- Which is false: "Hybrid models need no KV cache at all."
- Compare: RWKV vs Mamba — where the data dependence enters.

**Pitfalls / checks**
- RG-LRU constants and gating — verify §2.4 equations before writing.
- Hybrid ratios of 2025–26 models change between versions; cite the specific report.

---

### 420 · `llm.architecture-comparison` · Transformers vs RNNs vs SSMs: a head-to-head
intermediate · wave 2 (moved from wave 1 in review: its prereq chain `llm.linear-attention` → `llm.ssm` → `llm.mamba` is wave 2, so as a wave-1 lesson it would have had to teach S4 and Mamba from scratch; write unit L as a block at the start of wave 2) · prereqs: `llm.mamba`, `llm.kv-cache` (recommended: `llm.modern-rnns-hybrids`)

This is the user's explicit "LLM vs RNN vs S4" item. The basic transformer-vs-RNN rows (parallelism, path length, per-layer cost; Vaswani Table 1) are already taught in wave-1 `llm.self-attention` item 7.

**Primary sources**
- Vaswani 2017 §4 + Table 1 (complexity per layer, sequential operations, maximum path length for self-attention, recurrent, convolutional) (p.6). `llm_vaswani2017_attention.txt`; d2l §11.6.2 comparing CNNs, RNNs and self-attention. `llm_d2l_self_attention_posenc.txt`
- Jelassi et al. 2024, *Repeat After Me: Transformers are Better than SSMs at Copying* §2.2 (Theorem 2.3: depth-2 transformer copies exponentially long strings), §2.3 (Theorem 2.7: a generalized SSM with state space $S$ can't copy strings longer than ~$\log|S|$ bits' worth), §3 experiments (p.2–6). https://arxiv.org/abs/2402.01032 · `llm_jelassi2024_repeat_after_me.txt`
- Arora et al. 2023 (Zoology) §3 associative recall explains most of the perplexity gap between gated-convolution models and attention, §3.2 MQAR task, §4.2 capacity analysis (p.4–8). https://arxiv.org/abs/2312.04927 · `llm_arora2023_zoology.txt`
- Merrill, Petty & Sabharwal 2024, *The Illusion of State in State-Space Models* §2.3 transformers in TC⁰, §3 state tracking as monoid word problems (e.g. $S_5$ permutation composition), §4 SSMs also in TC⁰ (p.2–6). https://arxiv.org/abs/2404.08819 · `llm_merrill2024_illusion_state.txt`; Merrill & Sabharwal 2023, *Expressive Power of Transformers with Chain of Thought* (CoT steps extend expressivity). https://arxiv.org/abs/2310.07923 · `llm_merrill2023_cot_expressivity.txt`
- Gu & Dao 2023 (Mamba) §4.2–4.5 (inference throughput, scaling comparisons). `llm_gu2023_mamba.txt`; Griffin §5–6 (sampling efficiency; copying/retrieval). `llm_de2024_griffin.txt`

**Subtopic map**
1. *Comparison table (derive every row):*
   - training parallelism over time: RNN/LSTM sequential ($O(n)$ dependent steps); transformer fully parallel; linear RNN/SSM parallel via scan/convolution/chunks.
   - training cost: transformer $O(n^2d)$ attention + $O(nd^2)$; SSM/linear RNN $O(nd\cdot N)$ or $O(n d^2)$.
   - inference cost per token: transformer $O(nd)$ attention read (grows with context); RNN/SSM $O(1)$ in $n$.
   - inference memory: KV cache $O(n)$ vs fixed state $O(1)$.
   - maximum path length: $O(1)$ attention vs $O(n)$ recurrence.
   - recall/copying: exact retrieval from any position (attention) vs lossy compression into a fixed state.
2. *Why classic RNNs lost:* sequential training, vanishing gradients (link to fundamentals RNN/LSTM topic), poor hardware utilization — not fundamentally worse per parameter on all tasks.
3. *Why linear RNNs/SSMs came back:* parallel training + RNN inference; good perplexity per FLOP; long sequences (audio, DNA).
4. *Recall/copying limitation (proof idea):* a model with $b$ bits of state can distinguish at most $2^b$ histories, so it cannot copy an arbitrary string of more than $b$ bits (pigeonhole); transformers store the whole context in the KV cache. Zoology: associative recall accounts for most of the quality gap; MQAR results.
5. *Expressivity/state tracking (contested but important):* both (fixed-precision, log-depth) transformers and diagonal/linear SSMs sit in TC⁰ and provably can't do some sequential state-tracking problems ($S_5$ word problem) in one forward pass unless TC⁰ = NC¹; nonlinear RNNs can; CoT adds serial steps (Merrill & Sabharwal). Present as complexity-theory results with assumptions.
6. *Length generalization:* RNN/SSMs extrapolate in perplexity more gracefully; transformers depend on PE (pointers to `llm.context-extension`).
7. *Practical verdict in 2025–26:* frontier models are transformers (often MoE, with efficient attention); hybrids appear in efficiency-focused releases; pure SSM LMs are rare at the frontier. Say what is consensus vs trend.
8. *Interview framing:* "when would you choose an SSM?" — long streams, memory-bound edge inference, throughput; "when not?" — retrieval-heavy, in-context learning, copying.

**Figures**
- Summary table figure (rendered as a matplotlib table) with the rows above.
- Per-token decode cost and memory vs context length for transformer vs SSM vs 1:3 hybrid.
- Copy-task accuracy vs string length (schematic from Jelassi Fig. idea, labelled) or a toy numpy experiment with a linear RNN memory of fixed size.

**Question ideas**
- Which is false: "An SSM's per-token inference cost grows linearly with context length."
- Compute: bits needed to copy a 1,000-token string over a 50k vocabulary (~15.6 kbits) vs a state of 16×1024 fp16 numbers (~262 kbits) — does capacity alone forbid it? (Teach the nuance: capacity bound vs learnability.)
- Predict: Mamba vs transformer of equal size on phone-book lookup at 4k tokens.
- Compare: path length and gradient flow in LSTM vs transformer.
- Open: "Compare transformers, RNNs and S4/Mamba on training, inference, memory and capability."

**Pitfalls / checks**
- Complexity-theoretic claims depend on precision/uniformity assumptions; quote them with assumptions.
- Don't claim SSMs "can't do in-context learning" — they can, but recall is weaker.

---

### 430 · `llm.looped-transformers` · Universal, looped and recurrent-depth transformers
advanced · wave 3 · prereqs: `llm.architecture-comparison` (recommended after `llm.chain-of-thought`)

**Primary sources**
- Dehghani et al. 2018 (Universal Transformers) §2 model: shared transition applied recurrently over depth, per-position ACT halting; Turing-completeness argument (p.2–5). https://arxiv.org/abs/1807.03819 · `llm_dehghani2018_universal.txt`
- Graves 2016 (Adaptive Computation Time) §2: halting unit $h^n_t$, halting at $N(t)=\min\{n:\sum h\ge1-\epsilon\}$, remainder $R(t)$, ponder cost $\rho=N+R$, mean-field state update (p.2–6). https://arxiv.org/abs/1603.08983 · `llm_graves2016_act.txt`
- Giannou et al. 2023, *Looped Transformers as Programmable Computers* (a fixed 13-layer transformer looped can emulate a general-purpose computer / SUBLEQ; p.1–6). https://arxiv.org/abs/2301.13196 · `llm_giannou2023_looped.txt`
- Saunshi et al. 2025, *Reasoning with Latent Thoughts: On the Power of Looped Transformers* (k-layer looped L times ≈ kL-layer model on reasoning; looping simulates CoT steps). https://arxiv.org/abs/2502.17416 · `llm_saunshi2025_looped_reasoning.txt`
- Geiping et al. 2025, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*: prelude → recurrent block (random iteration count during training, truncated backprop) → coda; 3.5B model on 800B tokens; test-time scaling by more iterations; adaptive exit (p.1–10). https://arxiv.org/abs/2502.05171 · `llm_geiping2025_recurrent_depth.txt`
- Hao et al. 2024 (Coconut) continuous latent thoughts fed back as input embeddings. https://arxiv.org/abs/2412.06769 · `llm_hao2024_coconut.txt`; Wang 2025 (HRM) and Jolicoeur-Martineau 2025 (TRM) — small recursive reasoners on ARC/Sudoku. `llm_wang2025_hrm.txt`, `llm_jolicoeur2025_trm.txt`

**Subtopic map**
1. *Motivation:* depth is fixed per token in a standard transformer; some problems need more serial computation; parameter sharing across depth decouples compute from parameters.
2. *Universal Transformer:* the same block applied $T$ times with timestep embeddings; per-position dynamic halting.
3. *ACT (derive the mechanics):* halting probabilities per step, cumulative sum to $1-\epsilon$, remainder, weighted average state/output, ponder cost added to loss (and why it's non-differentiable in $N$ but differentiable in $R$).
4. *Expressivity of looping:* looped transformers emulate programs/iterative algorithms (Giannou); looping $L$ times ≈ depth $kL$ with $k$ layers' parameters; connection to CoT as externalized serial steps (Saunshi; Merrill & Sabharwal).
5. *Recurrent-depth LMs (Geiping 2025):* architecture; training with randomly sampled loop counts (a log-normal Poisson distribution, verified: Geiping Fig. 3, pdf p.4) and truncated backprop through the last k iterations; test-time compute scaling by unrolling more; KV-cache sharing across iterations; per-token adaptive exit.
6. *Latent reasoning vs token CoT:* Coconut — feed the last hidden state back as the next input embedding (breadth-first-ish search claims); pros (bandwidth, no verbalization) and cons (interpretability, training difficulty).
7. *HRM/TRM (2025, contested):* HRM (27M params, ~1000 training examples per task) reports strong Sudoku/Maze/ARC-AGI results; TRM (a single 2-layer, 7M-param network recursed) reports 45% on ARC-AGI-1 and 8% on ARC-AGI-2 (TRM abstract, pdf p.1). These are task-specific models trained per benchmark, not LMs; what drives the gains (recursion, deep supervision, data augmentation) is debated — present as contested and do not compare directly with LLM scores.
8. *Training challenges:* stability of deep unrolls, gradient truncation, halting collapse.
9. *Relation to MoD/early exit* (pointer to `llm.mot-mod`).

**Figures**
- Unrolled diagram: prelude → $r$ iterations of a shared block → coda, with input re-injection.
- ACT halting: cumulative halting probability crossing $1-\epsilon$ with the remainder shaded.
- Accuracy vs test-time iterations (schematic from Geiping's main figure, labelled).

**Question ideas**
- Compute: ACT with halting probs (0.3, 0.4, 0.5), ε=0.01: $N$, $R$, ponder cost.
- Predict: train with fixed 4 loops, test with 16 — what can go wrong (vs randomized loop counts)?
- Which is false: "A looped transformer with k layers looped L times has kL layers' worth of parameters."
- Compare: latent recurrence vs chain-of-thought tokens for test-time compute.

**Pitfalls / checks**
- Geiping training details (iteration sampling distribution, backprop depth) — verify; HRM/TRM claims are recent and debated.
## Unit M: Post-training

Cross-area note: no `rl.*` plan exists yet (see "Cross-area boundaries"), so `llm.policy-gradients` teaches the RL machinery RLHF needs — in the LLM setting, from scratch. If an RL area is planned later, coordinate so its generic PPO/GAE lesson and `llm.policy-gradients` link rather than duplicate.

### 440 · `llm.sft` · Supervised fine-tuning and instruction tuning
core · wave 1 (promoted in review: hard prereq of wave-1 `llm.lora` and `llm.reward-modeling`, and the backbone of the user's "finetuning" item) · prereqs: `llm.pretraining-objective`

**Primary sources**
- SLP3 ch. 8 §8.1 instruction tuning (§8.1.1 instructions as training data, §8.1.2 evaluation), §8.2 parameter-efficient fine-tuning (pdf p.2–7). https://web.stanford.edu/~jurafsky/slp3/8.pdf · `llm_slp3_ch8.txt`
- Wei et al. 2021 (FLAN) §2 instruction tuning on templated tasks; zero-shot gains scale with model size (p.2–6). https://arxiv.org/abs/2109.01652 · `llm_wei2021_flan.txt`
- Ouyang et al. 2022 (InstructGPT) §3.1 methodology, §3.5 SFT details (16 epochs, overfits on val loss but helps RM score — check) (p.6–9). https://arxiv.org/abs/2203.02155 · `llm_ouyang2022_instructgpt.txt`
- Zhou et al. 2023 (LIMA) — 1,000 curated examples; "superficial alignment hypothesis". https://arxiv.org/abs/2305.11206 · `llm_zhou2023_lima.txt`
- Lambert, *RLHF Book* ch. 4 Instruction Fine-Tuning (§4.1 chat templates, §4.2 best practices, §4.3 implementation; pdf p.33–37). https://rlhfbook.com · https://arxiv.org/abs/2504.12501 · `llm_lambert2025_rlhf_book.txt`
- Biderman et al. 2024, *LoRA Learns Less and Forgets Less* (full FT vs LoRA: forgetting of source-domain abilities). `llm_biderman2024_lora_forgets_less.txt`

**Subtopic map**
1. *Where SFT sits:* pretraining → SFT (instruction/chat) → preference tuning / RL; what each stage changes (format and behaviour vs knowledge).
2. *Objective:* next-token CE on responses only — mask the loss on prompt/system tokens (derive the masked loss; why masking matters for short answers after long prompts).
3. *Chat templates and special tokens:* roles, turn separators, EOS; template mismatch at inference as a common failure; multi-turn loss masking.
4. *Data:* templated NLP tasks (FLAN), human-written demonstrations (InstructGPT), distilled/synthetic data from stronger models; quality vs quantity (LIMA; superficial alignment hypothesis — contested); diversity of prompts; packing with per-example masks.
5. *Hyperparameters:* LR ~10× lower than pretraining peak, few epochs, overfitting signs; InstructGPT's observation that val loss rises while RM score/human preference still improves (check).
6. *Catastrophic forgetting:* mechanisms (distribution shift, large updates); mitigations — lower LR, mixing in pretraining data (replay), regularize toward the base (L2/KL), PEFT (forgets less but learns less), model merging/averaging.
7. *Memory cost of full fine-tuning* (from `llm.params-flops`): ~16 bytes/param for mixed-precision Adam ⇒ motivates PEFT.
8. *Rejection-sampling fine-tuning / best-of-N SFT* as a bridge to RL (Lambert ch. 9; pointer).
9. *Evaluation of SFT models:* instruction-following benchmarks, LLM judges (pointer to evaluation lessons).

**Figures**
- Token-level loss mask over a chat example (system, user, assistant turns; only assistant tokens shaded).
- Pipeline diagram: base → SFT → preference/RL, with what data feeds each stage.

**Question ideas**
- Predict: training on prompt+response tokens with no masking for a dataset of long prompts and 1-word answers.
- Which is false: "SFT on 1,000 curated examples can't change a model's style."
- Compute: memory for full fine-tuning a 13B model with Adam in mixed precision (≈208 GB before activations).
- Compare: replay vs LoRA for avoiding forgetting.

**Pitfalls / checks**
- LIMA's claim is contested; present as hypothesis.
- InstructGPT SFT epoch/overfitting details — verify §3.5/App. C.

---

### 450 · `llm.lora` · LoRA: low-rank adaptation
core · wave 1 · prereqs: `llm.sft`

**Primary sources**
- Hu et al. 2021 (LoRA) §2 problem statement, §4.1 $W_0+\Delta W=W_0+BA$, $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $A$ random Gaussian, $B=0$ at init, scaling $\alpha/r$; no added inference latency after merging; §4.2 applying to attention weights; §7 understanding the low-rank updates (which matrices, optimal rank, subspace similarity, $\Delta W$ amplifies directions not emphasized in $W$) (p.2–12). https://arxiv.org/abs/2106.09685 · `llm_hu2021_lora.txt`
- Aghajanyan et al. 2020, *Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning* (low intrinsic dimension; decreases with pretraining and model size). https://arxiv.org/abs/2012.13255 · `llm_aghajanyan2020_intrinsic_dim.txt`
- Kalajdzievski 2023 (rsLoRA): scaling $\alpha/\sqrt r$ keeps update magnitude stable as $r$ grows. https://arxiv.org/abs/2312.03732 · `llm_kalajdzievski2023_rslora.txt`
- Biderman et al. 2024 (LoRA Learns Less and Forgets Less): LoRA underperforms full FT on code/math continued pretraining, forgets less; full-FT updates are high-rank. https://arxiv.org/abs/2405.09673 · `llm_biderman2024_lora_forgets_less.txt`
- SLP3 ch. 8 §8.2 PEFT (LoRA description). `llm_slp3_ch8.txt`

**Subtopic map**
1. *Problem:* full FT of $N$ params per task — memory (optimizer states) and storage (a full copy per task).
2. *Hypothesis:* the fine-tuning update has low intrinsic rank (motivated by Aghajanyan's intrinsic-dimension results).
3. *Method:* $h=W_0x+\frac{\alpha}{r}BAx$; freeze $W_0$; train $A,B$. Parameters: $r(d+k)$ vs $dk$ — worked example: $d=k=4096$, $r=16$ ⇒ 131,072 vs 16.8M (0.78%) per matrix; whole-model counts for Llama-3-8B attention-only vs all-linear.
4. *Initialization (why):* $B=0$ so $\Delta W=0$ at start — the model begins exactly at the pretrained function; $A$ random so gradients to $B$ are non-zero (derive $\partial L/\partial B=\frac{\alpha}{r}\,\delta\,(Ax)^\top$, $\partial L/\partial A=\frac{\alpha}{r}B^\top\delta x^\top=0$ initially).
5. *The α/r scaling:* makes the effective update scale roughly independent of $r$ under Adam so LR need not be retuned when changing $r$; rsLoRA argues $\alpha/\sqrt r$ for stability at large $r$ — present both.
6. *Where to apply:* original paper — $W_q,W_v$ best for a fixed budget; modern practice — all linear layers (attention + MLP) often better (QLoRA paper finding).
7. *Merging:* $W=W_0+\frac{\alpha}{r}BA$ ⇒ zero inference overhead; or keep separate for multi-adapter serving (batched LoRA kernels, many tenants).
8. *Memory savings:* no optimizer states for frozen weights; gradients only for adapters; activations still needed (worked: 7B model full FT vs LoRA memory).
9. *What LoRA can't do well:* high-rank updates (continued pretraining on new domains, math/code at scale) — Biderman; LoRA forgets less (regularization view).
10. *Variants (brief, details next lesson):* DoRA (magnitude/direction split), LoRA+ (different LRs for A and B), VeRA, AdaLoRA.

**Figures**
- Diagram: frozen $W_0$ path plus $x\to A\to B$ low-rank path summed; shapes labelled.
- Parameter count vs rank for one 4096×4096 matrix (with full-matrix line).
- Singular value spectrum of a synthetic "full fine-tuning update" vs rank-r truncation error (numpy).

**Question ideas**
- Compute: LoRA params for all q,k,v,o and MLP matrices of Llama-3-8B at r=8 (use GQA shapes for k,v).
- Derivation: gradient of $A$ at initialization is zero — does training get stuck? (No: $B$ moves first.)
- Predict: initialize both $A$ and $B$ randomly.
- Which is false: "A merged LoRA model is slower at inference than the base model."
- Compare: LoRA vs full FT for continued pretraining on code (Biderman).
- Open: "Explain LoRA, its initialization and the α/r scaling, and when it falls short."

**Pitfalls / checks**
- The original LoRA paper's α/r convention vs PEFT library defaults; rsLoRA's claim is about large r.

---

### 460 · `llm.qlora-peft` · QLoRA and the PEFT zoo
intermediate · wave 2 · prereqs: `llm.lora`, `llm.quantization`

**Primary sources**
- Dettmers et al. 2023 (QLoRA) §3: 4-bit NormalFloat, double quantization, paged optimizers; backprop through frozen 4-bit weights into LoRA adapters (BF16 compute); §4–5 Guanaco results; LoRA on all layers needed to match full FT (p.3–8). https://arxiv.org/abs/2305.14314 · `llm_dettmers2023_qlora.txt`
- Houlsby et al. 2019 (Adapters) §2 bottleneck adapters inserted after sublayers. https://arxiv.org/abs/1902.00751 · `llm_houlsby2019_adapters.txt`
- Li & Liang 2021 (Prefix-Tuning) §4 trainable prefix activations at every layer (reparameterized by an MLP). https://arxiv.org/abs/2101.00190 · `llm_li2021_prefix_tuning.txt`; Lester et al. 2021 (Prompt Tuning) — soft prompts at the input; gap closes with scale. https://arxiv.org/abs/2104.08691 · `llm_lester2021_prompt_tuning.txt`
- Liu et al. 2024 (DoRA) §3–4 weight decomposition $W=m\frac{V}{\|V\|_c}$, LoRA on the direction. https://arxiv.org/abs/2402.09353 · `llm_liu2024_dora.txt`
- Lambert RLHF book / SLP3 §8.2 for overview. `llm_slp3_ch8.txt`

**Subtopic map**
1. *QLoRA recipe:* base weights stored in NF4 (blockwise, block 64), dequantized to bf16 on the fly for matmuls; LoRA adapters in bf16 receive gradients; gradients flow through the frozen quantized weights to earlier adapters.
2. *NF4 (recap/derive):* equal-probability bins for a standard normal, normalized by block absmax; why it suits normally distributed weights.
3. *Double quantization:* quantize the fp32 block scales to 8-bit with a second-level scale; memory saving per parameter (verify §3 number).
4. *Paged optimizers:* unified memory paging of optimizer states to survive memory spikes.
5. *Memory worked example:* 65B model fine-tuned on one 48 GB GPU (QLoRA's headline) — recompute: 65e9 × ~0.5 bytes ≈ 33 GB + adapters + activations.
6. *Adapters:* bottleneck MLP $x+W_{up}\sigma(W_{down}x)$; adds inference latency (extra sequential layers) unlike merged LoRA.
7. *Prefix and prompt tuning:* learned key/value prefixes per layer vs input soft prompts; parameter counts; take context length; weaker at small scale.
8. *DoRA:* decompose into magnitude vector and direction; LoRA updates the direction; claimed to match full-FT update patterns better.
9. *Choosing a PEFT method:* table of params, inference overhead, mergeability, quality.
10. *Quantization-aware merge caveat:* merging adapters into a quantized base requires re-quantization (accuracy drift).

**Figures**
- Memory stack bars for fine-tuning a 65B model: full FT (16-bit + Adam), LoRA (16-bit base), QLoRA (NF4 base) — weights, grads, optimizer, adapters.
- Placement diagram: where adapters, LoRA, prefix and prompt tuning attach in a transformer block.

**Question ideas**
- Compute: bytes per parameter for NF4 with block size 64 and fp32 scales vs with double quantization.
- Which is false: "Adapter layers can be merged into the base weights after training."
- Predict: QLoRA with LoRA only on $W_q,W_v$ vs all linear layers (QLoRA finding).
- Compare: prefix tuning vs LoRA at 1B vs 70B scale.

**Pitfalls / checks**
- QLoRA's double-quant saving is 0.373 bits/param (32/64 → 8/64 + 32/(64·256); verified pdf p.5); block size 64 for weights, 256 for the second-level scales.

---

### 470 · `llm.reward-modeling` · Reward models: Bradley–Terry, preference data, overoptimization
core · wave 1 · prereqs: `llm.sft`

**Primary sources**
- Ouyang et al. 2022 (InstructGPT) §3.4–3.5: comparisons of K=4–9 responses giving $\binom K2$ pairs per prompt trained in one batch element; 6B RM; RM loss $-\frac{1}{\binom K2}\mathbb{E}\log\sigma(r(x,y_w)-r(x,y_l))$; labeler agreement ≈72.6% (p.7–9). `llm_ouyang2022_instructgpt.txt`
- Stiennon et al. 2020, *Learning to summarize from human feedback* §3.4 RM training, §4.3 understanding the RM (optimizing against it eventually decreases true quality) (p.5–8). https://arxiv.org/abs/2009.01325 · `llm_stiennon2020_summarize.txt`
- Lambert, *RLHF Book* ch. 5 Reward Modeling: §5.1 Bradley–Terry RM, §5.2 ORMs, §5.3 PRMs, §5.4 RM types vs value functions; ch. 11 preference data; ch. 14 over-optimization (pdf p.39–55, 133–145, 171–178). `llm_lambert2025_rlhf_book.txt`
- Gao, Schulman & Hilton 2022, *Scaling Laws for Reward Model Overoptimization* §2 setup (gold RM labels a proxy RM), §3.1 functional forms $R_{bon}(d)=d(\alpha_{bon}-\beta_{bon}d)$, $R_{RL}(d)=d(\alpha_{RL}-\beta_{RL}\log d)$ with $d=\sqrt{\mathrm{KL}(\pi\|\pi_{init})}$, §3.2–3.6 scaling with RM size/data, KL penalty effect, §4.2 Goodhart taxonomy (p.2–10). https://arxiv.org/abs/2210.10760 · `llm_gao2022_rm_overoptimization.txt`
- Christiano et al. 2017, *Deep RL from Human Preferences* §2.2 fitting the reward function (Bradley–Terry over trajectory segments). https://arxiv.org/abs/1706.03741 · `llm_christiano2017_rlhf.txt`
- Weng 2024, *Reward Hacking in Reinforcement Learning* (taxonomy, RLHF-specific hacking: sycophancy, length). `llm_weng2024_reward_hacking.txt`

**Subtopic map**
1. *Why a reward model:* we can't write a reward for "helpful"; humans compare more reliably than they score; a learned RM generalizes comparisons to new outputs.
2. *Bradley–Terry (derive):* assume latent scores with Gumbel noise ⇒ $P(y_w\succ y_l)=\sigma(r(x,y_w)-r(x,y_l))$; MLE gives the logistic loss; only differences are identified (reward is defined up to a per-prompt constant) — why that matters for PPO (normalization) and for DPO (cancellation).
3. *Architecture:* SFT model with the LM head replaced by a scalar head on the final token; why initialize from the SFT model; size choices (InstructGPT 6B vs 175B policy).
4. *Data:* pairwise or K-way rankings ($\binom{K}{2}$ pairs, batched per prompt to avoid overfitting — InstructGPT); annotator agreement ~73% sets a ceiling on RM accuracy; preference strength / ties; Plackett–Luce for rankings (mention).
5. *Training details:* one epoch (overfits fast), reward normalization (mean 0 on SFT samples), margins, ensembles for uncertainty.
6. *ORM vs PRM* (pointer to `llm.test-time-compute`); generative RMs / LLM-as-judge RMs (mention).
7. *Overoptimization (Goodhart):* proxy reward rises while gold reward peaks then falls as the policy moves away (KL grows); Gao's functional forms (best-of-n vs RL), bigger RMs and more RM data delay the peak; KL penalty doesn't change the gold-vs-KL frontier much in their setup (check §3.6 wording).
8. *Reward hacking in LLMs:* length bias, sycophancy, formatting, exploiting RM blind spots; mitigations — KL penalty, RM ensembles, length penalties/normalization, iterated RLHF with fresh comparisons.
9. *Evaluation of RMs:* pairwise accuracy on held-out comparisons; benchmarks (RewardBench — mention); correlation with downstream policy quality is imperfect.

**Figures**
- Sigmoid of reward difference with data points of labelled comparisons; annotate the 73% agreement ceiling.
- Gao-style curves: proxy and gold reward vs $\sqrt{\mathrm{KL}}$ for RL, generated from the functional forms with illustrative coefficients (state that coefficients are illustrative).
- K-way ranking → pairs diagram (K=4 → 6 pairs).

**Question ideas**
- Compute: BT probability for reward difference 1.5 (σ(1.5) ≈ 0.818).
- Derivation: show adding $c(x)$ to all rewards leaves the BT likelihood unchanged.
- Predict: train the RM for 3 epochs instead of 1 (overfitting; InstructGPT/Stiennon observation).
- Which is false: "Scaling up the reward model eliminates overoptimization." (False: in Gao et al. larger RMs and more RM data delay and soften the gold-reward peak but do not remove it.)
- Figure MCQ: on the proxy-vs-gold plot, where should you stop RL?
- Open: "Derive the reward-model loss from the Bradley–Terry model and explain overoptimization."

**Pitfalls / checks**
- Gao's forms use $d=\sqrt{\mathrm{KL}}$; don't write KL.
- InstructGPT batching of all $\binom K2$ pairs from a prompt as one element (to prevent overfitting) — verify §3.5.

---

### 475 · `llm.policy-gradients` · Policy gradients for LMs: REINFORCE, baselines, PPO and GAE
core · wave 1 · prereqs: `llm.pretraining-objective` (recommended: `fund.gradient-estimators`) · **new lesson added in review** because no `rl.*` plan exists, and `llm.rlhf-ppo` (10 items) could not also teach the policy-gradient theorem, PPO and GAE from scratch within 6–10 cards.

Scope: only the RL a reader needs for RLHF/RLVR, set up directly on sequences of tokens. No general MDP theory, no Bellman optimality, no Q-learning.

**Primary sources**
- Lambert, *RLHF Book* §6.2.1 Deriving the Policy Gradient (pdf p.59), §6.2.2–6.2.3 vanilla PG and REINFORCE (p.62), §6.2.4 RLOO (p.63), §6.2.5–6.2.6 PPO and understanding its objective (p.64–70), §6.2.7 value functions and PPO (p.70–72). `llm_lambert2025_rlhf_book.txt`
- Schulman et al. 2017 (PPO) §2 background (policy-gradient estimator, TRPO's surrogate), §3 clipped surrogate $L^{CLIP}$, §5 algorithm (p.1–5). https://arxiv.org/abs/1707.06347 · `llm_schulman2017_ppo.txt`
- Schulman et al. 2015 (GAE) §2 preliminaries (advantage, baselines), §3 $\hat A_t^{GAE(\gamma,\lambda)}=\sum_l(\gamma\lambda)^l\delta_{t+l}$ and the bias–variance role of λ (p.2–5). https://arxiv.org/abs/1506.02438 · `llm_schulman2015_gae.txt`
- Ahmadian et al. 2024 (RLOO) §3 (REINFORCE with a leave-one-out baseline for sequence-level rewards). `llm_ahmadian2024_rloo.txt`

**Subtopic map**
1. *Setup as an LM problem:* prompt $x$, response $y=(y_1..y_T)\sim\pi_\theta(\cdot|x)$, scalar reward $r(x,y)$ at the end; goal $\max_\theta J=\mathbb{E}_{y\sim\pi_\theta}[r(x,y)]$. Why we can't backprop through $r$ (sampling is discrete; $r$ may be a program or a human). Token-level view: state = prompt + prefix, action = next token, deterministic transitions, $\gamma=1$.
2. *Policy gradient (derive):* log-derivative trick $\nabla_\theta\mathbb{E}_{\pi_\theta}[r]=\mathbb{E}[r\,\nabla\log\pi_\theta(y|x)]$ and $\log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|x,y_{<t})$ ⇒ REINFORCE. Read it as "reward-weighted maximum likelihood on your own samples" (link SFT's loss).
3. *Baselines (derive why unbiased):* $\mathbb{E}_{y}[b(x)\nabla\log\pi(y|x)]=b(x)\nabla\sum_y\pi(y|x)=0$; variance drops when $b\approx\mathbb{E}[r|x]$. Advantage $A=r-b$. Leave-one-out baseline over $k$ samples (RLOO) as an unbiased, critic-free choice — the bridge to GRPO.
4. *Credit assignment and causality:* a token can't affect rewards earned before it ⇒ reward-to-go; with only a terminal reward every token gets the same return, so per-token advantages need a value function $V(s_t)$.
5. *Importance sampling and the surrogate:* reusing a batch for several gradient steps means the data come from $\pi_{old}$; $\nabla J\approx\mathbb{E}_{\pi_{old}}[\rho_t\nabla\log\pi_\theta\,A_t]$ with $\rho_t=\pi_\theta/\pi_{old}$ per token; the surrogate $\mathbb{E}[\rho_tA_t]$ has the right gradient at $\theta=\theta_{old}$ but is only trustworthy nearby.
6. *PPO's clipped objective (derive the behaviour):* $\min(\rho_tA_t,\mathrm{clip}(\rho_t,1\pm\epsilon)A_t)$; case analysis for $A>0$ and $A<0$ shows the gradient vanishes once ρ leaves $[1-\epsilon,1+\epsilon]$ in the improving direction, and the min keeps the objective a pessimistic bound; several epochs of minibatch updates per rollout batch. Note what clipping does *not* do: it does not hard-cap how far one optimizer step can move ρ.
7. *Value function and GAE (derive):* TD error $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$; $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ telescopes to Monte-Carlo return minus $V$ at λ=1 and to one-step TD at λ=0 — λ trades variance for bias. Value loss (and value clipping).
8. *Where each piece reappears:* RLHF-PPO (`llm.rlhf-ppo`: value head, KL shaping), GRPO (`llm.rlvr-grpo`: group baseline instead of a critic), DPO (`llm.dpo`: no sampling at all).

**Figures**
- PPO clipped objective vs ratio ρ for A>0 and A<0 (two panels) — reader sees the flat regions where the gradient is zero.
- Variance of the REINFORCE gradient estimate with no baseline, a constant baseline and a leave-one-out baseline on a toy 5-token bandit (numpy) — reader sees the baseline cut variance without moving the mean.
- GAE weights $(\gamma\lambda)^l$ over future TD errors for λ = 0, 0.9, 1.

**Question ideas**
- Derivation step: why does $\mathbb{E}[b(x)\nabla\log\pi(y|x)]=0$?
- Compute: GAE advantages for a 3-token episode with given rewards/values, γ=1, λ=0.95.
- Figure MCQ: which region of the clipped objective has zero gradient when A<0?
- Which is false: "Subtracting a baseline that depends on the sampled response $y$ itself leaves the policy gradient unbiased." (False in general; the baseline must not depend on the action being scored — the reason RLOO leaves the sample out.)
- Predict: set ε very large in PPO and run 10 epochs per batch — what happens?
- Open: "Derive the policy gradient for a language model and explain what PPO's clipping is for."

**Pitfalls / checks**
- Sign conventions: PPO *maximizes* the surrogate; most code minimizes its negative. Say which.
- Use per-token ratios here; sequence-level ratios (GSPO) come in `llm.rlvr-practice`.

---

### 480 · `llm.rlhf-ppo` · RLHF with PPO: KL-regularized objective, four models, practicalities
core · wave 1 · prereqs: `llm.reward-modeling`, `llm.policy-gradients`

**Primary sources**
- Ouyang 2022 (InstructGPT) §3.5 RL: objective $\mathbb{E}[r_\theta(x,y)-\beta\log\frac{\pi^{RL}(y|x)}{\pi^{SFT}(y|x)}]+\gamma\,\mathbb{E}_{pretrain}[\log\pi^{RL}(x)]$ (PPO-ptx), value function initialized from the RM, per-token KL penalty (p.9); App. C.4 hyperparameters (β = 0.02, γ = 27.8; pdf p.42). `llm_ouyang2022_instructgpt.txt`
- Ziegler et al. 2019, *Fine-Tuning LMs from Human Preferences* §2 (reward $R(x,y)=r(x,y)-\beta\,\mathrm{KL}$, adaptive KL controller). https://arxiv.org/abs/1909.08593 · `llm_ziegler2019_finetuning_prefs.txt`
- Lambert, *RLHF Book* §6.1 role of RL, §6.4 auxiliary topics (KL estimators, value heads); ch. 15 regularization (pdf p.56–58, 79–96, 179–183). `llm_lambert2025_rlhf_book.txt`
- Schulman, *Approximating KL Divergence* (k1, k2, k3 estimators). http://joschu.net/blog/kl-approx.html · `llm_schulman2020_kl_approx.txt`
- Ahmadian et al. 2024, *Back to Basics: REINFORCE-style optimization (RLOO)*. https://arxiv.org/abs/2402.14740 · `llm_ahmadian2024_rloo.txt`

**Subtopic map**
1. *The RLHF objective:* $\max_\pi\mathbb{E}_{y\sim\pi}[r(x,y)]-\beta\,\mathrm{KL}(\pi(\cdot|x)\|\pi_{ref}(\cdot|x))$; why KL: keep the policy where the RM is accurate (Goodhart, from `llm.reward-modeling`), preserve fluency/diversity.
2. *Per-token form (derive):* sequence KL = expected sum of per-token log-ratios (chain rule of KL) ⇒ shaped per-token reward $r_t=-\beta\log\frac{\pi(y_t|\cdot)}{\pi_{ref}(y_t|\cdot)}$ plus the RM score at the last token; adaptive β controllers (Ziegler).
3. *Putting PPO on it:* advantages from GAE over the shaped rewards; value head initialized from the RM (InstructGPT); one PPO iteration step by step (recap from `llm.policy-gradients`, no re-derivation).
4. *Four models in memory:* policy, reference, reward, value (+ optimizer states for the two trainable ones) — worked memory estimate; motivates GRPO/RLOO (no critic) and DPO (no RM at train time).
5. *KL estimators:* k1 $=\log\frac{\pi}{\pi_{ref}}$ (unbiased, high variance, can be negative per sample), k3 $=\frac{\pi_{ref}}{\pi}-1-\log\frac{\pi_{ref}}{\pi}$ (unbiased, non-negative, lower variance); KL in reward (InstructGPT) vs KL in loss (GRPO).
6. *PPO-ptx and the alignment tax:* mixing pretraining gradients (γ = 27.8 in InstructGPT) to reduce regressions on NLP benchmarks.
7. *REINFORCE/RLOO alternative:* with sequence-level reward and good baselines, PPO's machinery may be unnecessary for LLMs (Ahmadian) — present as an argument with evidence.
8. *Practicalities:* rollout generation dominates wall time; reward normalization/whitening; entropy; length growth; instability signs; reward hacking signatures (pointer to `llm.reward-modeling`).

**Figures**
- RLHF loop diagram: prompts → policy rollouts → RM score + KL to reference → advantages (value head) → PPO update.
- Per-token reward timeline: KL penalties on every token, RM score at EOS, GAE advantages computed backwards (numbers from a toy example in Python).
- Memory stack for PPO-RLHF on a 7B policy: four models' weights + two optimizer states (bars), next to GRPO and DPO.

**Question ideas**
- Derivation: why the sequence-level KL equals the sum over tokens of expected log-ratios.
- Predict: set β=0 — what happens to reward and to text quality over training?
- Which is false: "The k1 KL estimator is always non-negative per sample."
- Compute: memory for PPO-RLHF with a 7B policy, 7B reference, 7B RM and 7B value model in bf16, with Adam states for policy and value (state the bytes/param convention).
- Open: "Walk through one PPO iteration in RLHF, naming every model involved."

**Pitfalls / checks**
- InstructGPT KL coefficient β = 0.02 and pretraining-mix coefficient γ = 27.8 (both verified, App. C, pdf p.42).
- Some implementations put KL in the reward (InstructGPT) and others in the loss (GRPO); be explicit.

---

### 490 · `llm.dpo` · Direct Preference Optimization
core · wave 1 · prereqs: `llm.rlhf-ppo`

**Primary sources**
- Rafailov et al. 2023 (DPO) §3 preliminaries (RLHF pipeline, Eq. 3 KL-constrained objective), §4 derivation (Eq. 4 optimal policy $\pi_r(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)\exp(\frac1\beta r(x,y))$; Eq. 5 reward reparameterization; Eq. 7 DPO loss), "What does the DPO update do?" gradient analysis, §5.1 equivalence classes of rewards (Lemma 1–2, Theorem 1), §5.2 actor-critic instability; App. A.1 optimum derivation, A.2 BT derivation, A.4 gradient (p.3–7, 15–18). https://arxiv.org/abs/2305.18290 · `llm_rafailov2023_dpo.txt`
- Lambert, *RLHF Book* ch. 8 Direct Alignment Algorithms: §8.1 DPO derivation, §8.2 numerical concerns and alternatives, §8.3 implementation (caching reference log-probs) (pdf p.108–118). `llm_lambert2025_rlhf_book.txt`
- SLP3 ch. 8 §8.3–8.4 learning from preferences; preference-based alignment (pdf p.7–15). `llm_slp3_ch8.txt`
- Azar et al. 2023 (IPO) §4.1–4.2: DPO overfits when preferences are (near-)deterministic because the BT logit goes to infinity and the KL term can't stop it (p.4–5). https://arxiv.org/abs/2310.12036 · `llm_azar2023_ipo.txt`

**Subtopic map**
1. *Start from the RLHF objective* (recap): $\max_\pi\mathbb{E}[r]-\beta\mathrm{KL}(\pi\|\pi_{ref})$ per prompt.
2. *Closed-form optimum (derive):* rewrite as $-\beta\,\mathrm{KL}(\pi\,\|\,\frac1Z\pi_{ref}e^{r/\beta})+\beta\log Z$ (Gibbs variational principle); minimized KL = 0 ⇒ $\pi^*=\frac1{Z(x)}\pi_{ref}e^{r/\beta}$. Explain $Z(x)$ and why it's intractable (sum over all sequences).
3. *Invert:* $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$.
4. *Plug into Bradley–Terry:* $Z(x)$ cancels in the difference ⇒ $P(y_w\succ y_l)=\sigma\big(\beta\log\frac{\pi(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi(y_l)}{\pi_{ref}(y_l)}\big)$ ⇒ DPO loss = negative log-likelihood of preferences under the implicit reward.
5. *Gradient (derive and interpret):* $-\beta\,\sigma(\hat r(y_l)-\hat r(y_w))[\nabla\log\pi(y_w)-\nabla\log\pi(y_l)]$ — weighted by how wrong the implicit RM is; without the weight (unlikelihood) training degenerates.
6. *Role of β:* strength of KL regularization; small β → aggressive deviation; effect on implicit reward scale.
7. *"Your LM is secretly a reward model":* implicit reward $\hat r=\beta\log\frac{\pi}{\pi_{ref}}$; reward equivalence classes (differ by $f(x)$) induce the same optimal policy (Lemma/Theorem).
8. *Practical recipe:* SFT first (or use $\pi_{ref}$ = SFT model); precompute reference log-probs; sum (not mean) of token log-probs per sequence; data from the SFT policy's own samples is better (distribution shift).
9. *Known failure modes:* likelihood of *both* chosen and rejected can decrease; overfitting with deterministic preferences (IPO's argument: BT logit → ∞ makes $\pi(y_l)\to0$ regardless of β); off-policy data; length exploitation; sensitivity to β.
10. *DPO vs PPO:* what is gained (no RM at training, no sampling, stable supervised loss) and lost (no on-policy exploration, implicit RM generalization) — pointer to the next lesson's debate.

**Figures**
- Derivation map: RLHF objective → Gibbs optimum → reward inversion → BT → DPO loss (flow diagram with each equation).
- DPO gradient weight $\sigma(\hat r_l-\hat r_w)$ vs implicit reward margin.
- Toy experiment (numpy over a 5-response "vocabulary"): $\pi^*$ for several β from a fixed reward and reference — shows interpolation between $\pi_{ref}$ and argmax.

**Question ideas**
- Derivation step: which term vanishes when substituting into BT, and why?
- Compute: DPO loss for β=0.1 with log-ratio 2.0 (chosen) and −1.0 (rejected): $-\log\sigma(0.3)$.
- Predict: β → 0 and β → ∞ effects on the learned policy.
- Which is false: "DPO requires sampling from the policy during training."
- Spot the flaw: an implementation averaging token log-probs instead of summing them (it becomes a length-normalized variant, like SimPO).
- Open: "Derive DPO from the KL-regularized RLHF objective."

**Pitfalls / checks**
- Equation numbering differs between arXiv versions; cite sections, not equation numbers.

---

### 500 · `llm.preference-variants` · IPO, KTO, ORPO, SimPO and online vs offline preference learning
intermediate · wave 2 · prereqs: `llm.dpo`

**Primary sources**
- Azar et al. 2023 (IPO / ΨPO) §4 general objective with Ψ, §5.1–5.2 IPO squared loss $\big(h_\pi(y_w,y_l)-\frac{1}{2\tau}\big)^2$ (p.3–6). `llm_azar2023_ipo.txt`
- Ethayarajh et al. 2024 (KTO) §3 prospect theory and HALOs, §4 KTO loss from unpaired desirable/undesirable labels with reference point $z_0$ (KL estimate) and $\lambda_D,\lambda_U$ (p.3–7). https://arxiv.org/abs/2402.01306 · `llm_ethayarajh2024_kto.txt`
- Hong et al. 2024 (ORPO) §3 role of SFT, §4 odds-ratio loss added to SFT NLL, no reference model; §4.3 gradient (p.3–5). https://arxiv.org/abs/2403.07691 · `llm_hong2024_orpo.txt`
- Meng et al. 2024 (SimPO) §2.2 length-normalized average log-prob reward (reference-free), §2.3 target margin γ, §4.2 length normalization prevents length exploitation (p.3–8). https://arxiv.org/abs/2405.14734 · `llm_meng2024_simpo.txt`
- Xu et al. 2024, *Is DPO Superior to PPO for LLM Alignment?* (well-tuned PPO beats DPO; DPO's out-of-distribution weakness). https://arxiv.org/abs/2404.10719 · `llm_xu2024_dpo_vs_ppo.txt`
- Singhal et al. 2023, *A Long Way to Go: Length Correlations in RLHF* (length drives much of RLHF reward gains). https://arxiv.org/abs/2310.03716 · `llm_singhal2023_length_rlhf.txt`; Bai et al. 2022, *Constitutional AI* (RLAIF; critique-revise SL stage + AI preference labels). https://arxiv.org/abs/2212.08073 · `llm_bai2022_constitutional.txt`

**Subtopic map**
1. *A unifying view:* losses of the form $f(\beta[\log\frac{\pi(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi(y_l)}{\pi_{ref}(y_l)}])$ — DPO uses $-\log\sigma$, IPO squared error to a target, hinge variants (SLiC); what changes is the implied regularization.
2. *IPO:* regress the log-ratio gap to $1/(2\tau)$ — bounded target prevents the deterministic-preference blow-up; derive why squared loss keeps the KL regularization effective.
3. *KTO:* needs only per-example good/bad labels; value function from prospect theory (loss aversion, reference point); useful when pairs are unavailable or data is imbalanced.
4. *ORPO:* single-stage — SFT NLL on chosen + λ·(−log σ(log-odds ratio of chosen vs rejected)); no reference model; odds vs probability ratio argument.
5. *SimPO:* reward = average log-prob per token (aligned with generation metric), margin γ, no reference model; length normalization combats verbosity; sensitivity to hyperparameters.
6. *Length bias across methods:* longer responses win preference labels and judges; DPO/RLHF can inflate length; mitigations — length normalization (SimPO), length-controlled evaluation (LC AlpacaEval), explicit penalties.
7. *Online vs offline:* offline DAAs train on a fixed dataset (from other policies) ⇒ distribution shift; online/iterative DPO (sample from current policy, label with RM/judge, repeat) recovers much of PPO's advantage; Xu 2024's finding that tuned PPO can beat DPO — debate status (contested; depends on tuning and task).
8. *RLAIF / Constitutional AI:* AI-generated preference labels guided by principles; critique–revision SFT stage; reduces human labelling cost.
9. *Choosing a method in practice:* table of required data (pairs vs labels), reference model (yes/no), online sampling, compute.

**Figures**
- Loss vs implicit-reward margin for DPO (logistic), IPO (squared to target), SLiC (hinge), with the IPO target marked.
- Table figure: method × {needs pairs, needs reference, needs RM, online?}.
- Response length vs training step for DPO vs SimPO (schematic, labelled) or from SimPO §4.2 numbers with citation.

**Question ideas**
- Compute: IPO loss for gap 3.0 with τ=0.1 (target 5) → (3−5)² = 4.
- Predict: running DPO for many epochs on preferences where one response always wins.
- Which is false: "ORPO requires a frozen reference model."
- Compare: offline DPO vs iterative online DPO with an RM.
- Open: "Why might PPO outperform DPO, and what does online DPO change?"

**Pitfalls / checks**
- IPO's notation ($\tau$ vs β) and target value — copy from §5.2.
- Present PPO-vs-DPO as contested.
## Unit N: Reasoning and test-time compute

### 510 · `llm.chain-of-thought` · Chain-of-thought and self-consistency
core · wave 2 · prereqs: `llm.sampling`

**Primary sources**
- Wei et al. 2022, *Chain-of-Thought Prompting* §2 method, §3 arithmetic results (gains emerge with scale), §3.3 ablations (equation-only, variable compute via dots, reasoning-after-answer), App. A.1 why scale helps (p.2–7, 16). https://arxiv.org/abs/2201.11903 · `llm_wei2022_cot.txt`
- Wang et al. 2022, *Self-Consistency* §2 sample diverse reasoning paths and marginalize (majority vote over final answers), weighting variants, §3 results (p.2–8). https://arxiv.org/abs/2203.11171 · `llm_wang2022_self_consistency.txt`
- Merrill & Sabharwal 2023, *The Expressive Power of Transformers with Chain of Thought* (linear/polynomial CoT steps strictly extend what fixed-depth transformers can compute). https://arxiv.org/abs/2310.07923 · `llm_merrill2023_cot_expressivity.txt`
- Lilian Weng 2025, *Why We Think* — sections on CoT as variable compute, parallel sampling vs sequential revision, faithfulness. https://lilianweng.github.io/posts/2025-05-01-thinking/ · `llm_weng2025_why_we_think.txt`

**Subtopic map**
1. *Phenomenon:* few-shot exemplars with intermediate steps; zero-shot "let's think step by step"; gains depend on scale and task (math, symbolic).
2. *Why it can help (three views):* (a) more serial computation per answer — a fixed-depth transformer can do only $O(L)$ sequential steps per token, CoT tokens add steps (Merrill & Sabharwal; links to TC⁰ discussion); (b) decomposition into easier conditional predictions; (c) training-data distribution contains worked solutions.
3. *Wei's ablations:* equation-only and "dots" (same token count without reasoning) don't recover gains ⇒ not just more compute tokens (for those models).
4. *Self-consistency (derive):* treat reasoning path $z$ as latent; answer $a$; approximate $\arg\max_a p(a|x)=\arg\max_a\sum_z p(a,z|x)$ by sampling paths and voting; majority vote vs probability-weighted; why diversity (temperature) matters; worked example; accuracy vs number of samples saturates.
5. *Majority vote math:* if each sample is correct w.p. $p>0.5$ and errors are spread, majority accuracy → 1 as $n$ grows (Condorcet); when wrong answers concentrate it fails — compute a small example.
6. *Faithfulness:* stated reasoning may not reflect the computation (post-hoc rationalization); implications for monitoring (mention).
7. *Beyond prompting:* training on CoT (STaR-style bootstrapping; reasoning RL in the next lessons); length vs accuracy.

**Figures**
- Accuracy vs model scale with/without CoT (schematic from Wei Fig. 4, labelled) — or omit if no real numbers.
- Self-consistency: 10 sampled paths → answer histogram → vote.
- Majority-vote accuracy vs n for p=0.4/0.6 per-sample accuracy (binomial, Python).

**Question ideas**
- Compute: majority-of-5 accuracy with per-sample accuracy 0.6 and binary answers (≈0.683).
- Predict: self-consistency with temperature 0.
- Which is false: "Replacing the reasoning with an equal number of filler tokens gives the same gains (in Wei et al.'s experiments)."
- Open: "Give two complementary explanations of why CoT helps."

**Pitfalls / checks**
- Later work found filler tokens can help when models are trained to use them; Wei's ablation is about prompting pretrained models — be precise.

---

### 520 · `llm.test-time-compute` · Scaling test-time compute: best-of-N, verifiers, PRMs and search
intermediate · wave 2 · prereqs: `llm.chain-of-thought`, `llm.reward-modeling`

**Primary sources**
- Snell et al. 2024, *Scaling LLM Test-Time Compute Optimally* §2 proposer/verifier view, §3 compute-optimal strategy by estimated difficulty, §5 search against a PRM (best-of-N weighted, beam search, lookahead), §6 revisions, §7 test-time vs pretraining FLOPs (>4× efficiency over best-of-N; can beat a 14× larger model on easy/medium questions) (p.1–13). https://arxiv.org/abs/2408.03314 · `llm_snell2024_test_time.txt`
- Lightman et al. 2023, *Let's Verify Step by Step* §2.5 ORMs, §2.6 PRMs (step-level labels, PRM800K; solution score = product of step correctness probabilities), §3 large-scale results, §4 small-scale process vs outcome (p.3–10). https://arxiv.org/abs/2305.20050 · `llm_lightman2023_prm.txt`
- Brown et al. 2024, *Large Language Monkeys* §2 coverage (pass@k) grows log-linearly with samples over orders of magnitude; §3.1 exponentiated power law fit; verification bottleneck (p.3–9). https://arxiv.org/abs/2407.21787 · `llm_brown2024_monkeys.txt`
- Muennighoff et al. 2025 (s1) §3 budget forcing (append "Wait" to extend, force end-of-thinking to cap), sequential vs parallel scaling (p.3–6). https://arxiv.org/abs/2501.19393 · `llm_muennighoff2025_s1.txt`
- Chen et al. 2021 (Codex) §2.1 unbiased pass@k estimator $1-\binom{n-c}{k}/\binom nk$ (p.3). https://arxiv.org/abs/2107.03374 · `llm_chen2021_codex.txt`; Lambert RLHF book ch. 7 reasoning & inference-time scaling (pdf p.97–107). `llm_lambert2025_rlhf_book.txt`

**Subtopic map**
1. *Two knobs:* parallel (sample N, aggregate) and sequential (longer reasoning, revision); plus search guided by a verifier.
2. *Coverage vs selection:* pass@k — probability at least one of k samples is correct; independent-sample formula $1-(1-p)^k$ (worked: p=0.2 → k=10: 0.89, k=100: ≈1); unbiased estimator from $n\ge k$ samples (derive $1-\binom{n-c}{k}/\binom nk$; worked: n=10, c=3, k=5 → 0.917). Selection without an oracle needs voting or a verifier — the "verification gap".
3. *Best-of-N with a reward model:* expected quality vs N; overoptimization against an imperfect verifier (Gao's BoN form; KL of BoN ≈ $\log N-\frac{N-1}{N}$ — derive/cite).
4. *ORM vs PRM:* outcome labels vs step labels; PRM scoring (product or min over steps); PRM800K; process supervision more reliable in Lightman's setting; cost of step labels; automatic step labels via Monte-Carlo rollouts (Math-Shepherd; mention).
5. *Search:* step-level beam search with a PRM, lookahead, MCTS (why it was less successful for LLM reasoning: DeepSeek-R1's "Unsuccessful Attempts" on PRMs and MCTS — App. G.2, pdf p.63, in the cached revised version; token-level search space too large, value model hard to train).
6. *Compute-optimal allocation (Snell):* easy questions benefit from sequential revisions, hard from parallel search; allocate by predicted difficulty; exchange rate between test-time and pretraining compute depends on difficulty and inference load.
7. *Sequential scaling in reasoning models:* longer thinking via RL-trained models; budget forcing; diminishing returns; overthinking.
8. *Cost accounting:* tokens × FLOPs per token; latency implications; matching FLOPs between a small model with heavy sampling and a large model.

**Figures**
- Coverage pass@k vs k (log x) for per-sample p ∈ {0.01, 0.05, 0.2} plus a "majority vote" curve that plateaus — shows the verification gap.
- PRM vs ORM scoring of a 5-step solution with one wrong middle step.
- Snell-style schematic: accuracy vs test-time budget for easy vs hard bins under parallel vs sequential strategies (schematic, labelled).

**Question ideas**
- Compute: unbiased pass@3 with n=5, c=1 (=0.6).
- Predict: best-of-1000 against a small RM vs majority vote on math — which improves more with N, and why might BoN degrade?
- Which is false: "A PRM gives a training signal only at the final answer."
- Compare: spending 10× compute on a 7B model's sampling vs using a 70B model greedily.
- Open: "How would you allocate a fixed inference budget across questions of different difficulty?"

**Pitfalls / checks**
- Snell's headline numbers are for their PaLM-2 setting; keep the setting attached.

---

### 530 · `llm.rlvr-grpo` · RL with verifiable rewards: GRPO and the R1 recipe
core · wave 1 · prereqs: `llm.rlhf-ppo` (recommended: `llm.chain-of-thought`; one sentence on what a reasoning trace is suffices for wave 1)

**Primary sources**
- Shao et al. 2024 (DeepSeekMath) §4.1 GRPO: objective with clipped ratio, group of G outputs per question, outcome-supervision advantage $\hat A_{i,t}=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}$, process-supervision variant, KL to reference added to the loss with the unbiased k3 estimator, iterative RL with RM refresh; §5.2.2 unified paradigm (SFT, RFT, DPO, PPO, GRPO as gradient coefficients) (p.11–16). https://arxiv.org/abs/2402.03300 · `llm_shao2024_deepseekmath.txt`
- DeepSeek-AI 2025 (DeepSeek-R1; cached arXiv version is the revised one): §2.1 GRPO, §2.2 reward design (rule-based accuracy and format rewards; no neural RM for reasoning to avoid hacking), §2.3 R1-Zero (response length growth, "aha moment" Table 2), §3 multi-stage pipeline (cold start, RL with language-consistency reward, rejection sampling ~600k reasoning + ~200k non-reasoning ⇒ ~800k SFT samples, second RL), App. A.3 GRPO vs PPO, App. B.3 data recipe (p.2–8, 14–27). https://arxiv.org/abs/2501.12948 · `llm_deepseek2025_r1.txt`
- Lambert RLHF book §6.2 (GRPO, RLOO in the policy-gradient family), ch. 7 RLVR (pdf p.58–107). `llm_lambert2025_rlhf_book.txt`
- Kimi k1.5 report (long-context RL, length penalty, no MCTS/value function). https://arxiv.org/abs/2501.12599 · `llm_kimi2025_k15.txt`
- Raschka, *A Technical Tour of the DeepSeek Models from V3 to V3.2* (R1 → V3.2 post-training changes incl. scaling GRPO). `llm_raschka_deepseek_v3_to_v32.txt`; DeepSeek-V3.2 §3.1 scaling GRPO (p.7–9). `llm_deepseek2025_v32.txt`

**Subtopic map**
1. *RLVR setting:* prompts with automatically checkable answers (math final answers, unit tests, format checks); reward $r\in\{0,1\}$ (+format); why verifiable rewards resist hacking better than learned RMs (but not perfectly).
2. *From PPO to GRPO (derive):* drop the value network; sample $G$ responses per prompt; baseline = group mean; advantage normalized by group std and assigned to every token of the response; clipped ratio objective as in PPO; KL penalty in the loss (not reward) using k3 per token. Write the objective out in full (interviewers ask for it): $J=\mathbb{E}\big[\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}\big(\min(\rho_{i,t}\hat A_{i,t},\mathrm{clip}(\rho_{i,t},1\pm\epsilon)\hat A_{i,t})-\beta\,\hat D_{i,t}\big)\big]$, with $\rho_{i,t}=\pi_\theta(o_{i,t}|q,o_{i,<t})/\pi_{old}(\cdot)$, $\hat A_{i,t}=(r_i-\mathrm{mean}(r))/\mathrm{std}(r)$ and $\hat D_{i,t}=\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$ per token (DeepSeekMath §4.1, pdf p.11–14; cite the section, not equation numbers). Flag the $1/|o_i|$ and $1/\mathrm{std}$ factors here — `llm.rlvr-practice` shows they cause length and difficulty biases. Memory/compute savings vs PPO.
3. *Why group baselines work:* Monte-Carlo estimate of $V(x)$ from same-prompt samples; relation to RLOO (leave-one-out mean, unbiased) — show GRPO's mean includes the sample itself (small bias) and std scaling changes weighting.
4. *Zero-advantage groups:* if all G answers are right or all wrong, advantages are 0 ⇒ no signal (motivates curriculum/dynamic sampling, next lesson).
5. *R1-Zero:* GRPO directly on the base model with accuracy+format rewards and a thinking template; response length and accuracy grow together; emergent reflection ("aha"); problems: readability, language mixing.
6. *R1 pipeline:* cold-start long-CoT SFT → reasoning RL (+ language-consistency reward) → rejection sampling + general SFT (~800k samples) → RL for all scenarios (rule rewards + preference RMs). Distillation of R1 traces into smaller dense models (Qwen/Llama) outperforms RL on small models directly (R1's finding).
7. *Unified view (DeepSeekMath §5.2.2):* SFT, RFT, DPO, PPO, GRPO differ in data source (offline/online), reward function, and gradient coefficient — useful interview framing.
8. *Compute profile:* rollouts dominate (long generations); inference engine inside the RL loop; off-policyness between rollout and update.
9. *Outcome vs process rewards in RL:* outcome rewards are sparse but robust; PRMs in RL risk hacking (R1 discussion).

**Figures**
- GRPO step diagram: one prompt → G=8 samples → rewards (0/1) → normalized advantages → token-level loss.
- Advantage values for a group with 3/8 correct: correct = +, incorrect = − (numbers computed); and the all-correct group with zero advantages.
- R1 pipeline flowchart (four stages with data sizes).

**Question ideas**
- Compute: GRPO advantages for rewards (1,0,0,1,0,0,0,0): mean 0.25, population std 0.433 ⇒ +1.73 for correct, −0.58 for incorrect (Python-verified; sample std gives slightly different values).
- Derivation: show that subtracting the group mean is a valid baseline (unbiased for $\nabla\log\pi$ terms of other samples; note the self-inclusion subtlety).
- Predict: G=2 vs G=16 on hard prompts where p(correct)=0.05.
- Which is false: "GRPO needs a learned value function to compute advantages."
- Compare: PPO with a critic vs GRPO for long reasoning responses.
- Open: "Explain the DeepSeek-R1 training pipeline and why each stage exists."

**Pitfalls / checks**
- The cached R1 PDF is a revised version whose section layout differs from v1 (Jan 2025); cite sections from the cached file and say so.
- GRPO uses population vs sample std depending on implementation; state which.

---

### 540 · `llm.rlvr-practice` · GRPO in practice: biases, fixes and open questions
advanced · wave 2 · prereqs: `llm.rlvr-grpo`

**Primary sources**
- Liu et al. 2025, *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO): GRPO's per-response length normalization $1/|o_i|$ and std normalization introduce length and difficulty biases; remove both (p.1–6). https://arxiv.org/abs/2503.20783 · `llm_liu2025_drgrpo.txt`
- Yu et al. 2025 (DAPO): Clip-Higher (decoupled $\epsilon_{low}=0.2$, $\epsilon_{high}=0.28$) against entropy collapse, Dynamic Sampling (drop groups with all-0/all-1 rewards), token-level policy-gradient loss, overlong reward shaping (p.2–8). https://arxiv.org/abs/2503.14476 · `llm_yu2025_dapo.txt`
- Zheng et al. 2025 (GSPO): sequence-level importance ratio (length-normalized) and sequence-level clipping; stability for MoE RL (p.1–4). https://arxiv.org/abs/2507.18071 · `llm_zheng2025_gspo.txt`
- Yue et al. 2025, *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* (pass@k at large k: base models catch up/overtake RL models; RL narrows the sampling distribution). https://arxiv.org/abs/2504.13837 · `llm_yue2025_rl_beyond_base.txt`
- Weng 2024 *Reward hacking* (RLVR-specific hacking: test tampering, format exploits). `llm_weng2024_reward_hacking.txt`; Kimi k1.5 (length penalty, curriculum). `llm_kimi2025_k15.txt`

**Subtopic map**
1. *Loss aggregation choices:* sequence-mean-then-batch-mean (GRPO original: each response weighted equally regardless of length ⇒ per-token weight ∝ $1/|o_i|$ ⇒ long wrong answers are penalized less per token ⇒ length growth) vs token-level mean over the batch (DAPO) vs constant normalizer (Dr. GRPO). Derive the per-token weights under each.
2. *Std normalization bias:* dividing by group std up-weights prompts that are nearly all-right or all-wrong (low std) — difficulty bias; Dr. GRPO removes it.
3. *Entropy collapse and clipping:* symmetric clipping limits probability increases of low-probability tokens more (ratio bound on small π); Clip-Higher; entropy bonuses vs not.
4. *Dynamic sampling / curriculum:* filter prompts with zero-variance rewards; keep effective batch size; difficulty targeting.
5. *Overlong responses:* truncation gives false negatives; soft length penalties; length budgets (Kimi k1.5).
6. *Sequence-level vs token-level importance ratios:* GSPO's argument that token-level ratios with sequence-level rewards are mismatched and noisy (especially for MoE routing changes); sequence-level clipping.
7. *KL or no KL:* many RLVR recipes drop the KL term (DAPO) — when the policy must move far from the base; risks.
8. *Off-policy and asynchronous RL:* staleness between generator and learner; importance correction; inference/training numerical mismatch (mention).
9. *Does RL add capability? (contested):* Yue 2025 — RL improves pass@1 but base models can match at large k ⇒ RL mainly re-weights existing solutions; counter-arguments (longer training, harder tasks, ProRL-style); spurious-reward results (mention, not cached). Present both sides.
10. *Reward hacking with verifiers:* weak tests, answer-format exploits, modifying tests in agentic coding; mitigations.

**Figures**
- Per-token weight vs response length under the three aggregation schemes.
- Clipping region for a low-probability token with ε=0.2 vs ε_high=0.28 (the probability beyond which the clipped objective gives no further incentive: $1.2\,\pi_{old}$ vs $1.28\,\pi_{old}$).
- pass@k curves for a base vs RL model crossing at large k (schematic from Yue Fig. 1, labelled).

**Question ideas**
- Compute: per-token gradient weight for responses of length 100 and 1,000 in the same group under sequence-mean aggregation.
- Predict: with symmetric ε=0.2 and a positive advantage, how far does the clipped objective push a token with π_old = 0.01? (No gradient incentive beyond ratio 1.2, i.e. π ≈ 0.012; an optimizer step can still overshoot, since clipping zeroes the gradient rather than hard-capping π — DAPO §3.1 makes the upper-clip argument, pdf p.4.)
- Which is false: "Dropping groups whose samples are all correct throws away useful gradient signal." (False: such groups have zero group-relative advantage, so they contribute no policy gradient; DAPO drops them to keep the effective batch full.)
- Compare: GRPO vs GSPO ratios.
- Open: "Argue for and against the claim that RLVR teaches new reasoning skills."

**Pitfalls / checks**
- These are 2025 results; mark as recent; check exact DAPO formulas.

## Unit O: Distillation, evaluation, interpretability, retrieval

### 550 · `llm.distillation` · Knowledge distillation for LLMs
intermediate · wave 2 · prereqs: `llm.pretraining-objective`, `llm.sft`

**Primary sources**
- Hinton, Vinyals & Dean 2015 §2 distillation with temperature $T$, gradient $\frac{1}{T}(q_i-p_i)$ and the $T^2$ scaling; high-temperature limit ≈ logit matching (p.2–3). https://arxiv.org/abs/1503.02531 · `llm_hinton2015_distillation.txt`
- Kim & Rush 2016, *Sequence-Level Knowledge Distillation* §3 word-level vs sequence-level KD (train on teacher's beam outputs), sequence-level interpolation. https://arxiv.org/abs/1606.07947 · `llm_kim2016_seq_kd.txt`
- Gu et al. 2023 (MiniLLM) §2: reverse KL $\mathrm{KL}(q_\theta\|p)$ for generative distillation (mode-seeking), policy-gradient optimization. https://arxiv.org/abs/2306.08543 · `llm_gu2023_minillm.txt`
- Agarwal et al. 2023 (GKD, on-policy distillation) §3: train on student-generated sequences with teacher token-level feedback; generalized JSD family; train–inference mismatch argument. https://arxiv.org/abs/2306.13649 · `llm_agarwal2023_gkd.txt`
- Gemma 2 §3 pretraining with distillation (student trained on teacher's next-token distributions; >50× compute-optimal tokens; check wording). `llm_team2024_gemma2.txt`; Gemma 3 §5.4 small vs large teacher. `llm_gemma2025_gemma3.txt`; DeepSeek-R1 distilled models (SFT on R1 traces). `llm_deepseek2025_r1.txt`; Lambert RLHF book §12.2–12.3 (distillation with synthetic data; on-policy teacher–student). `llm_lambert2025_rlhf_book.txt`

**Subtopic map**
1. *Why distill:* deploy smaller/faster models; teacher's soft distributions carry more information per example than one-hot labels ("dark knowledge").
2. *Hinton KD (derive):* $L=T^2\,\mathrm{KL}(p_T^{teacher}\|p_T^{student})$ (+ hard-label CE); gradient wrt student logit $z_i$: $\frac1T(q_i-p_i)$; why multiply by $T^2$ (keeps gradient magnitude comparable as T changes); $T\to\infty$ ⇒ matching centred logits.
3. *Token-level KD for LMs:* sum of per-position KL between teacher and student next-token distributions on a fixed corpus (forward KL, teacher-forced). Storage/compute: teacher logits over a large vocab (top-k truncation).
4. *Sequence-level KD:* train on teacher-generated outputs (beam or samples) with standard CE — this is what "distilling R1 into Qwen" is; approximates the teacher's mode.
5. *Forward vs reverse KL (derive the behaviours; the general asymmetry is owned by `fund.kl-divergence` — recap in two sentences, then the LLM-specific consequences):* forward $\mathrm{KL}(p\|q)$ is mode-covering (student spreads mass over all teacher modes, may put mass where teacher has little after capacity limits); reverse $\mathrm{KL}(q\|p)$ is mode-seeking (student commits to high-probability teacher regions) — illustrate with a bimodal teacher and unimodal student.
6. *On-policy distillation:* exposure bias of teacher-forced KD; GKD samples from the student and gets token-level teacher distributions on those prefixes; connection to RL with dense rewards (teacher log-probs as reward).
7. *Distillation in pretraining:* Gemma 2/3 train small models on teacher distributions for many more tokens than compute-optimal; choice of teacher size (Gemma 3 §5.4).
8. *Capacity gap and pitfalls:* too-strong teachers can hurt small students; tokenizer mismatch between teacher/student (needs alignment); legal/ToS issues of distilling API models (mention).

**Figures**
- Teacher distribution softened at T=1, 2, 5.
- Forward vs reverse KL fits of a single Gaussian to a bimodal mixture (computed numerically).
- On-policy vs off-policy distillation data-flow diagram.

**Question ideas**
- Derivation: gradient of the softened KD loss wrt a student logit.
- Compute: soft targets for logits (3,1,0) at T=2.
- Predict: distilling with reverse KL vs forward KL into a much smaller student — diversity of outputs.
- Which is false: "Sequence-level KD requires access to the teacher's logits."
- Open: "Compare token-level, sequence-level and on-policy distillation."

**Pitfalls / checks**
- Gemma 2 trains its 2B/9B models with distillation on more than 50× the compute-optimal token count (verified, abstract/§1, pdf p.1); details in §3.

---

### 560 · `llm.evaluation-metrics` · Evaluating LLMs: metrics, benchmarks and statistics
intermediate · wave 2 · prereqs: `llm.pretraining-objective`

**Primary sources**
- Chen et al. 2021 (Codex) §2.1 functional correctness, unbiased pass@k estimator and numerically stable computation (p.2–3). `llm_chen2021_codex.txt`
- Miller 2024, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations* (CLT standard errors, clustered SEs for related questions, paired differences, power analysis; p.1–10). https://arxiv.org/abs/2411.00640 · `llm_miller2024_error_bars.txt`
- Biderman et al. 2024, *Lessons from the Trenches on Reproducible Evaluation of LMs* (prompt sensitivity, log-likelihood vs generation scoring, normalization choices, lm-eval-harness). https://arxiv.org/abs/2405.14782 · `llm_biderman2024_lm_eval_lessons.txt`
- Chiang et al. 2024 (Chatbot Arena) §4–5 Bradley–Terry ratings from pairwise human votes, confidence intervals, sampling. https://arxiv.org/abs/2403.04132 · `llm_chiang2024_chatbot_arena.txt`
- SLP3 ch. 7 §7.7.1 perplexity; ch. 8 §8.1.2, §8.4.3 evaluation of instruction-tuned and aligned models. `llm_slp3_ch7.txt`, `llm_slp3_ch8.txt`; Raschka, *Understanding the 4 Main Approaches to LLM Evaluation* (linked from the architecture article; not cached — optional).

**Subtopic map**
1. *What to evaluate:* base-model capability (perplexity/BPB, few-shot MC benchmarks), instruction-following and chat quality, reasoning, code (functional correctness), safety, long context.
2. *Scoring formats:* multiple-choice via log-likelihood of options (length normalization choices: per token, per character, unconditional normalization) vs generate-then-parse; why they disagree; few-shot prompt formats and sensitivity.
3. *pass@k (derive the unbiased estimator):* sample $n$, count $c$ correct; $\widehat{\text{pass@}k}=1-\binom{n-c}{k}/\binom nk$; why $1-(1-\hat p)^k$ is biased (Jensen, concave in $\hat p$); worked numbers.
4. *Statistics:* standard error of accuracy $\sqrt{p(1-p)/n}$ (worked: 1,000 questions at 70% ⇒ ±1.45 pts SE ⇒ 95% CI ≈ ±2.8); clustered questions (shared passages) inflate SE; paired comparisons between models reduce variance; seed/sampling variance; reporting CIs.
5. *Pairwise human evaluation:* Chatbot Arena; Bradley–Terry MLE for ratings (link to `llm.reward-modeling`), Elo as online approximation; CI via bootstrap; style/length confounds.
6. *LLM-as-a-judge (overview; biases in next lesson):* pairwise vs single-answer grading; agreement with humans.
7. *Benchmark lifecycle:* saturation, held-out/private test sets, dynamic benchmarks.

**Figures**
- pass@k unbiased estimator vs naive plug-in estimate for n=20 and varying c (shows the bias).
- Accuracy with 95% CIs for two models on 500 vs 5,000 questions — overlapping vs separated.
- Bradley–Terry rating fit from a small synthetic pairwise win matrix.

**Question ideas**
- Compute: SE and 95% CI for 820/1000 correct.
- Compute: pass@1 and pass@10 from n=20, c=4.
- Predict: switching MC scoring from sum log-prob to per-token average — which options gain?
- Which is false: "Plugging $\hat p=c/n$ into $1-(1-\hat p)^k$ gives an unbiased pass@k."
- Open: "How would you decide whether a 1.5-point benchmark gain is real?"

**Pitfalls / checks**
- Miller's recommendations (clustered SEs, paired tests) — cite sections precisely.

---

### 570 · `llm.evaluation-pitfalls` · Contamination, LLM judges and benchmark validity
intermediate · wave 3 · prereqs: `llm.evaluation-metrics`

**Primary sources**
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* §3 judge types, §3.3 limitations: position bias, verbosity bias, self-enhancement bias, limited math/reasoning grading; mitigations (swap positions, few-shot, reference-guided) (p.3–7). https://arxiv.org/abs/2306.05685 · `llm_zheng2023_mtbench.txt`
- Dubois et al. 2024, *Length-Controlled AlpacaEval* (regression-based debiasing of length). https://arxiv.org/abs/2404.04475 · `llm_dubois2024_length_controlled.txt`
- Oren et al. 2023, *Proving Test Set Contamination in Black-Box LMs* (exchangeability test: canonical order vs shuffled likelihoods). https://arxiv.org/abs/2310.17623 · `llm_oren2023_contamination.txt`
- Llama 3 §5.1 contamination analysis (n-gram overlap-based). `llm_dubey2024_llama3.txt`; GPT-4 report App. (contamination checks via substring match). `llm_openai2023_gpt4.txt`
- Schaeffer 2023 (metric choice can create apparent emergence). `llm_schaeffer2023_mirage.txt`

**Subtopic map**
1. *Contamination:* benchmark items in pretraining data; types (verbatim, paraphrased, answer leakage); detection — n-gram overlap (choice of n, thresholds), membership-inference style likelihood tests, Oren's exchangeability test (derive: if the dataset wasn't seen, any ordering is equally likely ⇒ canonical order shouldn't have higher likelihood than shuffles; permutation-test p-value); mitigation — decontamination, private/held-out sets, time-split benchmarks.
2. *LLM-as-judge biases:* position (swap and average), verbosity/length (length-controlled win rates via regression), self-preference, sensitivity to formatting; agreement with humans (~80% on MT-Bench, similar to human–human — check).
3. *Goodhart on benchmarks:* overfitting to leaderboards, prompt engineering per benchmark, cherry-picked settings.
4. *Metric validity:* exact match vs semantic correctness; discontinuous metrics and apparent emergence.
5. *Reproducibility:* harness differences, tokenization of answer options, few-shot exemplars, sampling settings; report full configs.
6. *Safety and capability evals* (brief): red-teaming, dangerous-capability evals — pointer only.

**Figures**
- Position-bias illustration: judge preference for answer A when A/B swapped (2×2 table with synthetic counts).
- Exchangeability test: histogram of log-likelihoods of shuffled orders with the canonical order marked.

**Question ideas**
- Predict: judge prefers the first answer 65% of the time — how to correct.
- Which is false: "An 8-gram overlap check catches paraphrased contamination."
- Compare: n-gram decontamination vs Oren's test — what each needs access to.
- Open: "Design an evaluation protocol for a new chat model that controls for judge biases and contamination."

**Pitfalls / checks**
- MT-Bench agreement figures — verify §4.

---

### 580 · `llm.in-context-learning` · In-context learning and induction heads
advanced · wave 3 · prereqs: `llm.transformer-block`, `llm.self-attention`

**Primary sources**
- Brown et al. 2020 (GPT-3) — few-shot in-context learning definitions and scaling (shared cache: `papers/brown2020_gpt3.txt`, §1–2).
- Olsson et al. 2022, *In-context Learning and Induction Heads* (induction head = previous-token head + head that attends to the token after an earlier occurrence of the current token and copies it; phase change in training coinciding with ICL score improvement; six lines of evidence). https://arxiv.org/abs/2209.11895 · `llm_olsson2022_induction.txt`
- Elhage et al. 2021, *A Mathematical Framework*: "Two-Layer Attention-Only Transformers", "Three Kinds of Composition" (Q-, K-, V-composition), "Induction Heads" (function, mechanism). `llm_elhage2021_circuits_framework.txt`
- Xie et al. 2021, *An Explanation of ICL as Implicit Bayesian Inference* (pretraining on mixtures of latent concepts ⇒ prompts let the model infer the concept). https://arxiv.org/abs/2111.02080 · `llm_xie2021_icl_bayesian.txt`
- von Oswald et al. 2022, *Transformers Learn In-Context by Gradient Descent* (construction: one linear self-attention layer = one GD step on in-context linear regression). https://arxiv.org/abs/2212.07677 · `llm_vonoswald2022_icl_gd.txt`; Garg et al. 2022, *What Can Transformers Learn In-Context?* (function classes; near-optimal least squares). https://arxiv.org/abs/2208.01066 · `llm_garg2022_icl_linear.txt`

**Subtopic map**
1. *Phenomenon:* few-shot prompting without weight updates; zero/one/few-shot; dependence on scale, format, label correctness (mention).
2. *Induction heads (mechanism):* sequence "... A B ... A" → predict B. Two-step circuit: layer-1 previous-token head writes "the previous token was A" into B's residual stream; layer-2 head's query (current A) matches keys carrying "previous token = A" (K-composition) and its OV circuit copies B. Draw it.
3. *Composition types:* Q-, K-, V-composition between heads across layers; why one-layer attention-only models can only do skip-trigrams.
4. *Phase change evidence:* induction heads form abruptly early in training; in-context loss (loss at token 500 minus token 50) drops at the same time; ablations.
5. *Theoretical views of ICL:* (a) implicit Bayesian inference over latent concepts (Xie); (b) implicit optimization — attention implements GD-like updates on in-context examples for linear regression (von Oswald; Garg's empirical near-optimality); limits of each view for real LLMs.
6. *Practical implications:* example order sensitivity, recency, label-space vs mapping, long-context many-shot ICL (mention).

**Figures**
- Induction circuit diagram on the sequence "the cat sat ... the cat" showing previous-token head and induction head attention arrows.
- Synthetic in-context loss vs training step with a phase-change drop (schematic, labelled).
- Linear-regression ICL: transformer prediction error vs number of in-context examples compared to least squares (schematic from Garg, labelled) — or omit.

**Question ideas**
- Which composition type does the induction head's key use? (K-composition)
- Predict: ablate previous-token heads — effect on repeated-random-token loss.
- Which is false: "A one-layer attention-only transformer can implement an induction head."
- Compare: Bayesian vs gradient-descent explanations of ICL.

**Pitfalls / checks**
- Olsson's claims are strongest for small attention-only models; correlational for large models — say so.

---

### 590 · `llm.mech-interp` · Mechanistic interpretability basics
advanced · wave 3 · prereqs: `llm.in-context-learning`

**Primary sources**
- Elhage et al. 2021, *A Mathematical Framework*: residual stream as communication channel, virtual weights, QK and OV circuits, path expansion. `llm_elhage2021_circuits_framework.txt`
- Elhage et al. 2022, *Toy Models of Superposition* (more features than dimensions when features are sparse; phase diagram; interference). https://arxiv.org/abs/2209.10652 · `llm_elhage2022_superposition.txt`
- Bricken et al. 2023, *Towards Monosemanticity* (sparse autoencoders on MLP activations: $\hat x=W_d\,\mathrm{ReLU}(W_e(x-b_d)+b_e)+b_d$, L2 reconstruction + L1 sparsity; feature interpretability, dead features). https://transformer-circuits.pub/2023/monosemantic-features/index.html · `llm_bricken2023_monosemanticity.txt`
- Meng et al. 2022 (ROME) §2 causal tracing / activation patching to localize factual recall in mid-layer MLPs. https://arxiv.org/abs/2202.05262 · `llm_meng2022_rome.txt`
- Belrose et al. 2023, *Tuned Lens* (logit lens and its learned affine correction). https://arxiv.org/abs/2303.08112 · `llm_belrose2023_tuned_lens.txt`
- SLP3 ch. 10 Interpretability (currently a stub chapter: contextual embeddings, probing basics; 10 pdf pages). `llm_slp3_ch10.txt`

**Subtopic map**
1. *Goals and levels:* behavioural vs representational vs mechanistic; features and circuits.
2. *Residual-stream algebra:* logits = sum of contributions of each component (direct logit attribution); QK circuit $W_E^\top W_QW_K^\top W_E$ decides where to attend; OV circuit $W_UW_OW_VW_E$ decides what is written.
3. *Logit lens / tuned lens:* decode intermediate residual states with the unembedding (plus a learned affine map); what it shows (iterative refinement) and its biases.
4. *Probing:* linear probes for properties; correlation vs causation; control tasks.
5. *Causal interventions:* activation patching (clean → corrupted runs), causal tracing (ROME), path patching; ablations (zero vs mean vs resample).
6. *Superposition:* when features are sparse, a model can store $m>d$ features in $d$ dims with small interference; polysemantic neurons; toy-model phase diagram.
7. *Sparse autoencoders:* objective, dictionary size, L1 coefficient, dead/ultra-low-density features, evaluating features (interpretability, reconstruction loss in downstream CE); scaling to production models (Templeton 2024 — not cached; mention).
8. *Limits and open problems:* faithfulness of explanations, SAE reconstruction error, feature splitting.

**Figures**
- Superposition toy: 5 sparse features in 2-D — learned directions (pentagon) from actually training the toy model in numpy.
- Activation patching heatmap over layers × token positions (synthetic, labelled).
- SAE diagram: activation → encoder (overcomplete) → ReLU sparse code → decoder.

**Question ideas**
- Which is false: "A linear probe with high accuracy shows the model uses that feature."
- Predict: increasing feature sparsity in the toy model — more or fewer features in superposition?
- Compute: SAE with $d=512$, expansion 8 — dictionary size and parameter count.
- Open: "How would you test whether a head is responsible for a behaviour?"

**Pitfalls / checks**
- Keep claims about large-model SAEs qualitative unless sourced.

---

### 600 · `llm.rag` · Retrieval-augmented generation
intermediate · wave 3 · prereqs: `llm.cross-attention`, `llm.long-context-eval`

**Primary sources**
- Lewis et al. 2020 (RAG) §2 RAG-Sequence vs RAG-Token marginalization over retrieved documents $p(y|x)\approx\sum_{z\in\text{top-}k}p_\eta(z|x)p_\theta(y|x,z)$, DPR retriever (MIPS), joint fine-tuning of query encoder and generator (p.2–4). https://arxiv.org/abs/2005.11401 · `llm_lewis2020_rag.txt`
- SLP3 ch. 11 Information Retrieval and RAG: §11.1 ranked retrieval (tf-idf/BM25), §11.3 IR with dense vectors (bi-encoders), RAG sections and QA datasets (pdf p.3–20). https://web.stanford.edu/~jurafsky/slp3/11.pdf · `llm_slp3_ch11.txt`
- Liu 2023 (lost in the middle — retrieved-document position effects). `llm_liu2023_lost_middle.txt`

**Subtopic map**
1. *Why retrieve:* knowledge freshness, attribution, smaller models, long-tail facts; alternative to stuffing everything into context.
2. *Retrievers:* sparse (BM25) vs dense bi-encoders (DPR; contrastive training with in-batch negatives — derive the InfoNCE loss), maximum inner product search and ANN indexes; rerankers (cross-encoders).
3. *RAG model (derive):* latent document $z$; RAG-Sequence marginalizes once per sequence, RAG-Token per token; training end-to-end through $p_\eta(z|x)$ with top-$k$ approximation.
4. *Modern practice:* retrieve-then-read with frozen LLMs, chunking, query rewriting, citations; long-context vs RAG trade-offs.
5. *Retrieval inside the model* (mention): kNN-LM interpolation, RETRO chunked cross-attention.
6. *Failure modes:* retrieval misses, distractors, position effects, conflicts between parametric and retrieved knowledge.
7. *Evaluation:* recall@k, answer accuracy, faithfulness/attribution.

**Figures**
- RAG pipeline: query → retriever (top-k) → generator with marginalization.
- Recall@k vs k for BM25 vs dense (schematic, labelled).

**Question ideas**
- Derivation: RAG-Sequence vs RAG-Token likelihoods for k=2 documents.
- Predict: increasing k from 5 to 50 for a model with lost-in-the-middle behaviour.
- Which is false: "Dense retrieval always outperforms BM25."
- Open: "When would you choose RAG over a longer context window?"

**Pitfalls / checks**
- Lower priority (wave 3).
