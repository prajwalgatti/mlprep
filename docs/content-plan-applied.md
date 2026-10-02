# Content plan: Applied ML & Systems (area `applied`, ids `sys.*`)

This plan covers the **Applied ML & Systems** area: numerics, gradients, memory, communication, parallelism, performance,
compilation and frameworks. Writers follow [writing-brief.md](writing-brief.md) ("Teach it, don't summarise it") and the
format in [CONTENT_GUIDE.md](../CONTENT_GUIDE.md). Sources are cached in the scratchpad source cache
(`sources/`);
their index is `INDEX-sys.md` there (shared files from `INDEX.md` and `INDEX-llm.md` are also used and are marked "shared" below).

**How to use a topic section.** Each topic has: the problem the lesson answers; primary sources with exact sections; a
**subtopic map**, which is the syllabus the lesson must cover (every item is something a strong interviewer could probe,
with what the reader must be able to *do*); **worked computations** whose numbers were recomputed in Python for this plan
(recompute them again when writing); question ideas; pitfalls and claims to verify; figure ideas. A lesson is about 6–10
explainer cards. Where a syllabus is larger, it has already been split into a sequence of topics linked by `prereqs`.

**Reader assumption.** A strong ML PhD who trains models but may never have thought about bits, collectives or sharding.
Every lesson starts from a concrete problem ("a 7B model does not fit on an 80 GB GPU, why, and what do we do?"), defines
every symbol, and derives each formula before using it.

**Boundaries with other plans (do not duplicate).**
- `docs/content-plan-tuning.md` (`sys.tuning-*`, Google Deep Learning Tuning Playbook): hyperparameter-tuning
  methodology, batch-size choice and step budgets, training-curve diagnostics, the clipping-threshold rule
  (`sys.tuning-diagnostics`) and the multi-host checklist (`sys.tuning-pipeline`). Here, `sys.gradient-accumulation` covers
  only the mechanics and correctness of accumulation and links to the tuning lessons for "what batch size should I use".
  The tuning plan's provisional ids map to: `sys.data-parallel†` → `sys.ddp` / `sys.gradient-accumulation`; `sys.memory†` →
  `sys.memory-anatomy` / `sys.gradient-accumulation`; `sys.checkpointing†` → `sys.reliability-checkpointing`;
  `sys.input-pipeline†` → `sys.profiling`.
- `docs/content-plan-llms.md` (`llm.*`). All `llm.*` ids below come from the current draft of that plan and are marked
  "id to confirm" where used; re-check them when the LLM plan is final. Owners decided:
  - MoE and expert-parallel systems (all-to-all dispatch, capacity, EP meshes, DualPipe overlap): `llm.moe-systems`.
    Here EP appears only as one mesh axis and as the all-to-all exception in the bandwidth rule.
  - Quantization (int8/int4, outliers, NF4, inference FP8/MX): `llm.quantization`, `llm.quantization-advanced`.
  - KV cache and decode-time rooflines: `llm.kv-cache`, `llm.arithmetic-intensity`. `sys.roofline` is the general model.
  - Training-recipe stability (warmup, β₂, QK-Clip, data spikes): `llm.training-stability`. `sys.stability-tricks` keeps
    only the numerics of large logits.
  - Parameter/FLOP counting for budgets: `llm.params-flops`, `llm.scaling-laws`. `sys.flops-mfu` owns MFU/HFU and training time.
  - FlashAttention (`llm.flash-attention`) and ring attention in `llm.long-context`: used here only as fusion and
    context-parallel examples. LoRA/QLoRA (`llm.lora`, `llm.qlora-peft`) appear only as a memory lever.
- `scalingbook` area (`sb.*`) follows the JAX Scaling Book chapter by chapter. The `sys.*` lessons stay framework-neutral
  and syllabus-driven, cite the book as a source, add "see also" lines (`sb.roofline-matmul`, `sb.collective-costs`,
  `sb.fsdp`, `sb.tensor-parallelism`, `sb.gpu-chip`; ids from the scalingbook plan) and must agree with its numbers.
- `fundamentals` (`fund.*`, `docs/content-plan.md`) owns these basics; `sys.*` lessons recap them in at most one card and
  link by id:
  - `fund.rnn` (Jacobian product, Pascanu's σ_max conditions, eigenvalue picture, norm vs value clipping, cliffs, truncated
    BPTT; figures `fund.rnn/gradient-norm-vs-lag`, `fund.rnn/clipping-cliff`) and `fund.lstm-gru` (gating);
  - `fund.initialization` (variance propagation, 0.9⁵⁰/1.1⁵⁰), `fund.activations` (sigmoid saturation), `fund.normalization`;
  - `fund.backprop` (item 7: Chen's √L checkpointing; figure `fund.backprop/checkpointing`);
  - `fund.training-loop` (items 7–9: AMP/GradScaler usage, the HF gradient-accumulation bug, reproducibility, resume);
  - `fund.logistic-regression` (item 7: log-sum-exp) and `fund.variance-covariance` (variance computation);
  - `fund.adam` (optimizer, ε).
  No `sys.*` figure redraws a fund figure.

## Conventions for all `sys.*` lessons

**Symbols** (use these consistently; define them again in each lesson):

| Symbol | Meaning |
|---|---|
| $N$ | number of model parameters (the ZeRO paper writes $\Psi$; say so when citing it) |
| $P$ | number of devices taking part in one collective (the ring size); $d, t, p$ = data-, tensor-, pipeline-parallel degrees |
| $b, s, h, a, L, V$ | micro-batch size (sequences), sequence length, hidden size, attention heads, layers, vocabulary (Korthikanti/Megatron notation) |
| $m$ | number of micro-batches per pipeline flush; $v$ = virtual stages (model chunks) per device |
| $B$ | tokens per step (global batch × sequence length) unless stated otherwise |
| $C$ | peak FLOP/s of one accelerator; $W$ = bandwidth (bytes/s) of the relevant link |
| $\alpha, \beta$ | per-message latency (s) and inverse bandwidth (s/byte) in the α–β cost model |

**Units.** 1 GB = $10^9$ bytes, 1 GiB = $2^{30}$ bytes; say which one you use (nvidia-smi and PyTorch report MiB/GiB).
FLOPs = floating-point operations (a multiply-add counts as 2); FLOP/s = rate. Network bandwidths are quoted in bytes/s
per direction unless stated; vendors often quote bidirectional totals or Gb/s (bits), so convert explicitly.

**Hardware constants** used in worked examples (all from `sys_scalingbook_gpus.txt` "Summary of GPU specs" / "Networking",
`sys_scalingbook_roofline.txt`, `sys_korthikanti2022_seqpar_recompute.txt` footnote 5):

| Quantity | Value | Use |
|---|---|---|
| H100 SXM dense bf16 | ≈ 989 TFLOP/s (Scaling Book rounds to 9.9e14) | rooflines, MFU |
| H100 dense fp8 | ≈ 1.98e15 FLOP/s (Scaling Book table rounds to 2.0e15) | fp8 ridge point |
| H100 HBM bandwidth / capacity | 3.35 TB/s (table rounds to 3.4e12), 80 GB | rooflines, memory |
| H100 NVLink (within an 8-GPU node, via NVSwitch) | 450 GB/s per GPU per direction (≈370 GB/s achieved in practice) | intra-node collectives |
| H100 node egress over InfiniBand | 8 × 400 Gb/s NICs = 400 GB/s per node (50 GB/s per NIC) | cross-node AG/RS/AR: use 400 GB/s per node; 50 GB/s per GPU only for all-to-all |
| A100 dense bf16 / HBM | 312 TFLOP/s / ≈2.0 TB/s (80 GB) | older papers' MFU numbers |

**Bandwidth rule for collectives** (taught in `sys.collectives` item 6; every lesson uses it). Inside a node, use per-GPU
NVLink egress (450 GB/s). Across nodes, all-reduce / reduce-scatter / all-gather use all 8 NICs at once (one ring per NIC,
or reduce inside the node first), so the effective rate is the node egress, 400 GB/s: $T_{AG/RS} \approx \text{bytes}/400\text{e}9$,
all-reduce twice that without SHARP (Scaling Book Part 12 "Cross-node collectives"). 50 GB/s per GPU is right only when each
GPU's whole buffer must leave through its own NIC: cross-node all-to-all, or a TP group spread one GPU per node. A "flat ring
at 50 GB/s per GPU" estimate for cross-node all-reduce is ≈ 8× too pessimistic.

**Gradient dtype in communication examples.** Worked examples assume **bf16 gradients** (bf16 params or a bf16 comm hook)
and say so. With standard `torch.autocast` + DDP the params, hence `.grad`, are fp32, so DDP all-reduces fp32 gradients:
comm time and the compute-bound threshold double (≈ 4400 / 4950 tokens/GPU instead of 2200 / 2475).

Say in the text that peak numbers are vendor or Scaling-Book figures and that achieved FLOP/s are typically 80–85% of peak
on H100 even for large matmuls (Scaling Book Part 1, side-note).

## Topic table

Orders are in gaps of 10. **W1** = wave 1 (most interview-critical; write first). W2 = second wave; W3 = advanced or
specialised. "Cards" is an estimate of explainer cards. † = prereq later in the order (W3 lesson, written after it).
Revised after the plan review (`docs/reviews/plan-applied-review.md`): roofline/FLOPs and a new GPU-basics lesson moved
before the parallelism block and into W1; vanishing/exploding and clipping re-scoped against `fund.*` and moved to W2; the
ring derivation moved into `sys.collectives`; `sys.reliability-checkpointing` added.

| Order | id | Title | Level | Prereqs | Wave | Cards |
|---|---|---|---|---|---|---|
| 10 | `sys.fp-formats` | Floating-point formats: bits, range and precision (fp32, fp16, bf16, tf32, fp8) | core | — | **W1** | 8 |
| 20 | `sys.fp-error` | Rounding error, cancellation and non-associativity | intermediate | sys.fp-formats | W2 | 8 |
| 30 | `sys.stable-numerics` | Numerically stable building blocks: log-sum-exp, softmax, cross-entropy, variance | core | sys.fp-error | W2 | 9 |
| 40 | `sys.mixed-precision` | Mixed-precision training: autocast, master weights, loss scaling | core | sys.fp-formats | **W1** | 9 |
| 50 | `sys.fp8-training` | FP8 and block-scaled low-precision training | advanced | sys.mixed-precision, sys.roofline† | W3 | 8 |
| 60 | `sys.stability-tricks` | The numerics of large logits: z-loss, soft-capping and QK-norm | advanced | sys.stable-numerics, sys.mixed-precision | W3 | 5 |
| 70 | `sys.vanishing-exploding` | Vanishing and exploding gradients in deep networks and transformers | core | fund.initialization, fund.rnn | W2 | 8 |
| 80 | `sys.gradient-clipping` | Gradient clipping in practice | core | sys.vanishing-exploding, sys.mixed-precision, fund.rnn | W2 | 7 |
| 90 | `sys.memory-anatomy` | Where GPU memory goes when training | core | sys.mixed-precision | **W1** | 9 |
| 100 | `sys.gpu-basics` | How a GPU executes a training step (SMs, HBM/SRAM, tensor cores, warps, streams) | core | — | **W1** | 8 |
| 110 | `sys.roofline` | The roofline model and arithmetic intensity | core | sys.gpu-basics | **W1** | 8 |
| 120 | `sys.flops-mfu` | Counting FLOPs, MFU/HFU and estimating training time | core | sys.roofline | **W1** | 8 |
| 130 | `sys.profiling` | Profiling training: compute-, memory- and overhead-bound | intermediate | sys.roofline, sys.gpu-basics | W2 | 9 |
| 140 | `sys.activation-checkpointing` | Activation checkpointing (rematerialisation) | intermediate | sys.memory-anatomy, sys.flops-mfu | W2 | 7 |
| 150 | `sys.gradient-accumulation` | Gradient accumulation: equivalence and where it breaks | core | sys.memory-anatomy | W2 | 7 |
| 160 | `sys.collectives` | Communication primitives and the ring all-reduce | core | — | **W1** | 9 |
| 170 | `sys.collective-algorithms` | Latency, trees, hierarchy and interconnects | intermediate | sys.collectives | W2 | 7 |
| 180 | `sys.ddp` | Distributed Data Parallel (DDP) | core | sys.collectives, sys.memory-anatomy, sys.roofline | **W1** | 9 |
| 190 | `sys.zero` | ZeRO: sharding optimizer state, gradients and parameters | core | sys.ddp, sys.memory-anatomy | **W1** | 9 |
| 200 | `sys.fsdp` | FSDP in practice: units, prefetching, FSDP2, HSDP | intermediate | sys.zero | W2 | 9 |
| 210 | `sys.tensor-parallel` | Tensor parallelism (Megatron-style) | core | sys.collectives, sys.memory-anatomy, sys.roofline | **W1** | 9 |
| 220 | `sys.sequence-context-parallel` | Sequence parallelism and context parallelism | advanced | sys.tensor-parallel, sys.activation-checkpointing, sys.stable-numerics | W3 | 8 |
| 230 | `sys.pipeline-parallel` | Pipeline parallelism: GPipe, the bubble, 1F1B | core | sys.memory-anatomy, sys.collectives, sys.ddp | **W1** | 9 |
| 240 | `sys.pipeline-schedules` | Advanced pipeline schedules: interleaved, zero-bubble, DualPipe, async | advanced | sys.pipeline-parallel | W3 | 8 |
| 250 | `sys.parallelism-composition` | Composing parallelism (3D/4D/5D) and choosing a configuration | advanced | sys.zero, sys.tensor-parallel, sys.pipeline-parallel, sys.sequence-context-parallel, sys.flops-mfu | W3 | 9 |
| 260 | `sys.reliability-checkpointing` | Checkpointing and fault tolerance at scale | advanced | sys.fsdp | W3 | 8 |
| 270 | `sys.graph-capture` | JIT compilation I: tracing, graph capture, retracing and graph breaks | intermediate | — | W2 | 9 |
| 280 | `sys.fusion-codegen` | JIT compilation II: operator fusion, XLA, Inductor/Triton, CUDA graphs | intermediate | sys.graph-capture, sys.roofline | W2 | 8 |
| 290 | `sys.jax-model` | The JAX programming model: purity, PRNG keys, transformations, sharding | intermediate | sys.graph-capture | W2 | 9 |
| 300 | `sys.frameworks` | JAX vs PyTorch vs TensorFlow | core | sys.graph-capture | W2 | 7 |

Wave 1 is 11 of 30 topics, in dependency order: fp-formats → mixed-precision → memory-anatomy → gpu-basics →
roofline → flops-mfu → collectives → ddp → zero → tensor-parallel → pipeline-parallel.
Suggested wave-2 order: stable-numerics, activation-checkpointing, gradient-accumulation, profiling, graph-capture,
frameworks, vanishing-exploding, gradient-clipping, collective-algorithms, fsdp, fp-error, fusion-codegen, jax-model.

## Coverage of the required items

| Required item | Main topic | Also covered in |
|---|---|---|
| Tensor parallelism | `sys.tensor-parallel` | sys.sequence-context-parallel, sys.parallelism-composition |
| FSDP | `sys.zero` (theory: memory, comm volume) → `sys.fsdp` (implementation) | sys.parallelism-composition, sys.reliability-checkpointing |
| DDP | `sys.ddp` | sys.collectives (all-reduce), sys.gradient-accumulation (`no_sync`) |
| Pipeline parallelism | `sys.pipeline-parallel` → `sys.pipeline-schedules` | sys.parallelism-composition |
| Communication primitives | `sys.collectives` (incl. ring all-reduce) → `sys.collective-algorithms` | every parallelism lesson |
| Mixed precision training | `sys.mixed-precision` | sys.fp8-training, sys.memory-anatomy |
| Gradient checkpointing | `sys.activation-checkpointing` | sys.memory-anatomy, sys.sequence-context-parallel (selective recompute); basics in fund.backprop |
| Gradient accumulation | `sys.gradient-accumulation` | sys.ddp (`no_sync`), sys.zero/fsdp (sharded grads), sys.pipeline-parallel (micro-batches); basics in fund.training-loop |
| Profiling | `sys.profiling` | sys.gpu-basics, sys.roofline, sys.flops-mfu |
| Gradient clipping | `sys.gradient-clipping` | sys.mixed-precision (unscale before clip), sys.fsdp (sharded norm); basics in fund.rnn |
| Numerical precision tricks | `sys.stable-numerics`, `sys.stability-tricks` | sys.fp-error (Kahan, stochastic rounding), sys.mixed-precision |
| Exploding / vanishing gradients | `sys.vanishing-exploding` (deep-net/transformer view) | sys.gradient-clipping; basics in fund.rnn, fund.initialization |
| Floating point representation | `sys.fp-formats` | sys.fp-error, sys.fp8-training |
| JIT compiling | `sys.graph-capture` → `sys.fusion-codegen` | sys.jax-model, sys.frameworks |
| JAX vs PyTorch vs TensorFlow | `sys.frameworks` | sys.jax-model, sys.graph-capture |

---

# Part A: Numerics

## sys.fp-formats — Floating-point formats: bits, range and precision

Level core · prereqs — · **W1** · ~8 cards

**Problem the lesson answers.** "Why does training in fp16 overflow when bf16 doesn't, and why is bf16 still worse for
some things?" The reader should be able to decode any format from its (sign, exponent, mantissa) bit counts and compute
its range, precision and failure modes by hand.

**Primary sources**
- Goldberg 1991, "Rounding Error › Floating-point Formats" and "Relative Error and Ulps"; "The IEEE Standard › Formats and
  Operations", "Special Quantities", "Denormalized Numbers" — `sys_goldberg1991_floating_point.txt`
  (https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).
- Micikevicius et al. 2022 (FP8 formats) §3 (E4M3 and E5M2 definitions, the single NaN pattern, max 448 / 57344) —
  `sys_micikevicius2022_fp8.txt` (https://arxiv.org/abs/2209.05433).
- Kalamkar et al. 2019 (bf16) §2–3 — `sys_kalamkar2019_bf16.txt` (https://arxiv.org/abs/1905.12322).
- PyTorch note "Numerical accuracy", section "TensorFloat-32 (TF32) on Nvidia Ampere (and later) devices", and "CUDA
  semantics › TensorFloat-32" — `sys_pytorch_numerical_accuracy.txt`, `sys_pytorch_cuda_semantics.txt`.
- Cross-check tables only: `sys_wiki_half_precision.txt`, `sys_wiki_bfloat16.txt`.

**Subtopic map**
1. *Why floating point.* Fixed point wastes range; scientific notation in base 2 gives roughly constant *relative*
   precision over a huge range. Value $(-1)^S \cdot 1.M \cdot 2^{E - \text{bias}}$ for normal numbers.
2. *Encoding.* Sign bit; biased exponent (bias $= 2^{e-1}-1$); implicit leading 1; reserved exponent patterns (all-zeros
   for zero/subnormals, all-ones for inf/NaN in IEEE formats). Reader must decode a given 16-bit pattern by hand.
3. *Derive the key numbers from (e, m)*: max normal $(2 - 2^{-m})\,2^{E_{\max}}$ with $E_{\max} = 2^e - 2 - \text{bias}$;
   min normal $2^{1-\text{bias}}$; min subnormal $2^{1-\text{bias}-m}$; machine epsilon (gap above 1) $= 2^{-m}$; unit
   roundoff $u = 2^{-(m+1)}$ (half an ulp at 1); decimal digits ≈ $(m+1)\log_{10}2$.
4. *The format zoo* (table the reader can reproduce): fp32 (1,8,23), fp16 (1,5,10), bf16 (1,8,7), tf32 (1,8,10, a
   tensor-core internal format, 19 bits used), fp8 E4M3 (1,4,3) and E5M2 (1,5,2). Range is set by exponent bits,
   precision by mantissa bits. bf16 = fp32 with the mantissa truncated, so conversion is a shift and the range matches fp32.
5. *Ulps and spacing.* Spacing doubles every binade: at magnitude $x$ the gap is about $x \cdot 2^{-m}$. Integers are
   exact only up to $2^{m+1}$ (fp16: 2048; bf16: 256), so `256 + 1 = 256` in bf16. Consequence for counters, token ids,
   positions and step counts stored in low precision.
6. *Subnormals (gradual underflow).* What they are, why they exist (x − y = 0 iff x = y), their reduced precision, and
   that some hardware flushes them to zero (FTZ/DAZ) for speed. fp16's subnormals reach $2^{-24} \approx 5.96\times10^{-8}$.
7. *Special values.* ±0, ±inf, NaN (quiet/signalling), NaN propagation, `inf - inf`, `0 * inf`. E4M3 has **no inf**
   and one NaN mantissa pattern, which buys the extra binade up to 448 (Micikevicius 2022 §3); E5M2 keeps IEEE specials.
8. *Rounding on conversion.* Round-to-nearest-even is the default; saturating vs non-saturating casts to fp8; truncation
   (bf16 via chopping) vs proper rounding.
9. *Which format where in DL*: weights/activations/gradients/optimizer state; why gradients of small magnitude underflow
   in fp16 (motivates loss scaling, next lessons); why bf16 needs no loss scaling but loses precision (motivates fp32
   master weights); tf32 as the default matmul precision knob in PyTorch (`allow_tf32`, `set_float32_matmul_precision`).
10. *Integer quantization* (int8/int4 with scales and zero-points, outliers) is owned by `llm.quantization` /
    `llm.quantization-advanced` (ids from the current LLM plan draft; confirm); give one line contrasting floating vs
    integer formats and link.
11. *What interviewers probe*: "compute bf16 epsilon", "max fp16 value", "why bf16 for training and fp16 for inference
    sometimes", "what is tf32", "what does E4M3 vs E5M2 trade".

**Worked computations (verified with `torch.finfo` and the formulas)**

| Format | (e, m) | bias | max normal | min normal | min subnormal | eps $=2^{-m}$ |
|---|---|---|---|---|---|---|
| fp32 | 8, 23 | 127 | 3.40e38 | 1.18e-38 | 1.40e-45 | 1.19e-7 |
| fp16 | 5, 10 | 15 | 65504 | 6.10e-5 | 5.96e-8 | 9.77e-4 |
| bf16 | 8, 7 | 127 | 3.39e38 | 1.18e-38 | 9.18e-41 | 7.81e-3 |
| tf32 | 8, 10 | 127 | ≈3.40e38 | 1.18e-38 | — (internal) | 9.77e-4 |
| E5M2 | 5, 2 | 15 | 57344 | 6.10e-5 | 1.53e-5 | 0.25 |
| E4M3 | 4, 3 | 7 | 448 (no inf; see note) | $2^{-6}$ = 0.0156 | $2^{-9}$ ≈ 0.00195 | 0.125 |

- E4M3 note: the max-normal formula $E_{\max} = 2^e - 2 - \text{bias}$ assumes IEEE conventions and would give 240. OCP E4M3
  keeps normals in the all-ones exponent ($E_{\max} = 15 - 7 = 8$) and reserves only mantissa 111 there for NaN, so the
  largest value is S.1111.110 $= 1.75\cdot2^8 = 448$ (Micikevicius 2022 §3). Good compute MCQ: "what would E4M3's max be
  under IEEE conventions?" (240).
- Decode by hand: fp16 max = sign 0, exponent 11110 (=30, unbiased 15), mantissa all ones → $(2-2^{-10})\cdot 2^{15} = 65504$.
- bf16 swamping: `1.0 + 0.001` rounds to `1.0` (half-ulp at 1 is $2^{-8} \approx 0.0039$); `1.0 + 0.004` → `1.0078125`.
- fp16: `1.0 + 1e-4` → `1.0`; `2048 + 1` → `2048`; Adam's default ε = 1e-8 is below fp16's smallest subnormal and becomes 0.
- `exp(x)` overflows fp16 for x > ln 65504 ≈ 11.09, fp32 for x > 88.7.

**Question ideas**
- Compute: "A format has 6 exponent bits and 9 mantissa bits with IEEE conventions. What are its max value and epsilon?"
- Decode: given the bit pattern `0 01111 0100000000` (fp16), what number is it? (1.25)
- Predict: a model trained in bf16 stores the step counter in bf16; what happens to the LR schedule after step 256?
- Which is false: "bf16 and fp32 have the same dynamic range" / "bf16 has more precision than fp16" (false) / "fp16 can
  represent 1e-6 only as a subnormal" / "E4M3 cannot represent infinity".
- Compare: why E4M3 for forward activations/weights and E5M2 for gradients.

**Pitfalls / verify**
- Machine epsilon vs unit roundoff: some texts (and numpy `finfo.eps`) use the gap $2^{-m}$; numerical analysts use
  $u = 2^{-(m+1)}$. State the convention.
- tf32 is not a storage dtype; tensors stay fp32 in memory, inputs are rounded to 10 mantissa bits inside the tensor core,
  accumulation is fp32.
- E4M3 max 448 assumes the OCP/NVIDIA variant (no inf). Other fp8 variants exist (e.g. "fnuz" types with different bias);
  mention only as a caveat.
- Do not say "bf16 has 3 significant digits"; it has about 2.4 decimal digits (8 bits of significand).

**Figures**
- `bit-layouts`: horizontal bars showing sign/exponent/mantissa widths for fp32, tf32, bf16, fp16, E5M2, E4M3, aligned on the binary point.
- `number-line`: log-scale number line with representable ranges (min subnormal → max) for fp16, bf16, E4M3, E5M2, with a
  shaded band for typical gradient magnitudes (illustrative) to show what underflows in fp16.
- `spacing`: ulp spacing vs magnitude (staircase) for fp16 and bf16 on log–log axes.

---

## sys.fp-error — Rounding error, cancellation and non-associativity

Level intermediate · prereqs sys.fp-formats · W2 · ~8 cards

**Problem.** "Two identical training runs give different losses on different GPU counts; summing a million small numbers
in bf16 gives garbage; subtracting nearly equal numbers loses all digits." The lesson gives the error model that explains
all three and the standard fixes (higher-precision accumulation, pairwise/Kahan summation, stochastic rounding).

**Primary sources**
- Goldberg 1991: "Relative Error and Ulps", "Guard Digits", "Cancellation" (benign vs catastrophic; Theorem 3 area
  formula example), "Exactly Rounded Operations", "Systems Aspects › Optimizers › Theorem 8 (Kahan Summation Formula)" —
  `sys_goldberg1991_floating_point.txt`.
- Wikipedia "Kahan summation algorithm" (algorithm, error bound vs naive and pairwise) — `sys_wiki_kahan.txt`.
- Gupta et al. 2015 §3 (stochastic rounding: definition and unbiasedness) — `sys_gupta2015_limited_precision.txt`.
- Zamirai et al. 2020 (why nearest rounding stalls bf16 weight updates; Kahan and stochastic-rounding fixes) —
  `sys_zamirai2020_revisiting_bf16.txt` (https://arxiv.org/abs/2010.06192).
- PyTorch notes "Numerical accuracy" (batched vs sliced results; reduced-precision reductions in fp16/bf16 GEMMs) and
  "Reproducibility" (nondeterministic algorithms, `torch.use_deterministic_algorithms`) — `sys_pytorch_numerical_accuracy.txt`,
  `sys_pytorch_randomness.txt`.

**Subtopic map**
1. *The standard model*: $\mathrm{fl}(x \circ y) = (x \circ y)(1+\delta)$, $|\delta| \le u$. Reader can state it and
   use it on two operations.
2. *Absolute vs relative error; ulps.* Why relative error is the natural measure in floating point.
3. *Catastrophic cancellation.* Subtracting nearly equal *already-rounded* numbers: the subtraction is exact but exposes
   earlier rounding errors. Derive the relative error blow-up factor $\approx |x|/|x-y|$. Benign cancellation (exact
   inputs) vs catastrophic. Examples: $E[x^2]-E[x]^2$, $1-\cos x$, the quadratic formula; rewrite tricks (algebraic
   reformulation, `log1p`, `expm1`) foreshadow `sys.stable-numerics`.
4. *Non-associativity.* $(a+b)+c \ne a+(b+c)$; `(1e8 + 1) - 1e8 = 0` but `1e8 - 1e8 + 1 = 1` in fp32. Consequences:
   results depend on reduction order → GPU nondeterminism (atomic adds in scatter/index_add, split-K GEMMs, different
   all-reduce trees and ring orders, different batch sizes / tensor-parallel degrees giving different kernels).
   The same row can give different outputs when computed inside a different batch, because kernels choose different
   reduction splits by shape (state this without a coined name; the term "batch invariance" comes from an uncached blog).
5. *Error growth in summation.* Naive recursive sum error bound $\approx (n-1)u\sum|x_i|$; pairwise (tree) summation
   $O(u \log n)$; why GPU reductions are naturally tree-shaped; why matmuls accumulate in fp32 even with bf16 inputs.
6. *Swamping / lost updates.* Adding a small number to a large accumulator: if $|y| < \tfrac12\mathrm{ulp}(x)$ it vanishes.
   Show the naive fp16 running sum of 10,000 × 0.1 stalls at 256 and bf16 at 32. Link to weight updates
   $w \leftarrow w - \eta g$ in pure bf16 (Zamirai 2020) and to fp32 master weights.
7. *Kahan (compensated) summation.* Derive the compensation term $c = (t - s) - y$, why it captures the low-order bits,
   error bound $O(u)$ independent of $n$ (to first order); cost (4 flops/element, extra state); use in bf16 optimizers.
8. *Stochastic rounding.* Round up with probability proportional to distance: $\mathbb{E}[\mathrm{SR}(x)] = x$. Derive
   the unbiasedness; explain why it rescues tiny updates in expectation; variance cost; hardware support.
9. *Reproducibility in practice*: seeds are not enough; deterministic algorithms flags; fixed reduction order; cost.
10. *Interview probes*: "why does my loss differ between 8 and 16 GPUs with the same global batch", "why accumulate in
    fp32", "what is Kahan summation", "why is stochastic rounding unbiased".

**Worked computations (verified in PyTorch 2.1)**
- fp16 naive running sum of 10,000 copies of 0.1 → 256.0; bf16 → 32.0; fp16 Kahan sum → 1000.0. Caveat: fp16(0.1) = 0.0999756,
  so the exact sum of the stored values is 999.756; Kahan returns 1000.0 because that is the correctly rounded fp16 result
  (spacing 0.5 near 1000). Kahan removes accumulation error, not the representation error of 0.1.
- fp32: `(1e8 + 1) - 1e8 = 0.0`, `1e8 - 1e8 + 1 = 1.0`.
- Variance by $E[x^2]-E[x]^2$ for {1e4+4, 1e4+7, 1e4+13, 1e4+16} in fp32 gives 24.0; the two-pass formula gives the true 22.5.
- Unit roundoff: fp32 $2^{-24} \approx 5.96\times10^{-8}$, bf16 $2^{-8} \approx 0.0039$, fp16 $2^{-11} \approx 4.9\times10^{-4}$.
- Naive-sum bound for $n = 4096$ terms in bf16: $(n-1)u \approx 16$, i.e. the bound is useless, which is why the
  accumulator must be fp32.

**Question ideas**
- Predict: running mean of the loss accumulated in a bf16 scalar over an epoch; what goes wrong and when?
- Debug: "same seed, same data, different results on every run" — which ops could be responsible (atomics, cuDNN autotuning
  picking different algorithms, all-reduce order)?
- Derive: show $\mathbb{E}[\mathrm{SR}(x)] = x$ for $x$ between neighbours $a < x < b$.
- Which is false: "pairwise summation has error growing like log n" / "Kahan summation needs higher-precision hardware" (false).
- Compute: smallest $\Delta$ such that $1 + \Delta \ne 1$ in fp16 under round-to-nearest-even (just above $2^{-11}$).

**Pitfalls / verify**
- "Floating-point addition is not commutative" is false; it is commutative but not associative.
- Cancellation is not caused by the subtraction itself (which is exact by Sterbenz's lemma when inputs are within a factor 2).
- Kahan summation can be destroyed by compilers that reassociate (fast-math); Goldberg discusses this under "Optimizers".

**Figures**
- `running-sum`: running sum of 0.1 in fp32 / bf16 / fp16 vs the exact line, showing the plateaus at 32 and 256 and the Kahan curve on the exact line.
- `cancellation`: relative error of $E[x^2]-E[x]^2$ vs two-pass as the data offset grows from $10^0$ to $10^6$ (fp32).
- `stochastic-rounding`: histogram of SR outcomes for a value between two bf16 neighbours with the mean marked.

---

## sys.stable-numerics — Numerically stable building blocks

Level core · prereqs sys.fp-error · W2 · ~9 cards

**Problem.** "softmax([1000, 1001, 1002]) returns NaN; log(sigmoid(x)) returns -inf; LayerNorm variance comes out
negative." The lesson derives the stable form of every common DL primitive.

**Overlap with fund.** `fund.logistic-regression` (item 7) introduces log-sum-exp and stable softmax regression, and the
fund variance lessons introduce Welford. Here, recap those in one card each and spend the cards on the full catalogue
(log-softmax, BCE-with-logits, softplus, log1p/expm1, Chan's merge, ε placement, online softmax) and the low-precision angle.

**Primary sources**
- Goodfellow DL §4.1 "Overflow and Underflow" (softmax stabilisation by subtracting the max) — shared `dlb_ch04_numerical.txt`.
- Blanchard, Higham & Higham 2021 §2–3 (shifted log-sum-exp and softmax, error analysis) — `sys_blanchard2021_logsumexp.txt`.
- Wikipedia "Algorithms for calculating variance" (naive, two-pass, Welford, Chan's parallel merge) — `sys_wiki_variance_algorithms.txt`.
  Secondary source: use it as a cross-check only and derive Welford's update and Chan's merge in full in the lesson.
- PyTorch `torch.amp` "Autocast Op Reference" (ops that autocast to fp32: softmax, log_softmax, layer_norm, losses; "Prefer
  binary_cross_entropy_with_logits over binary_cross_entropy") — `sys_pytorch_amp.txt`.
- Milakov & Gimelshein 2018 (online softmax: running max and rescaled sum in one pass) — shared `llm_milakov2018_online_softmax.txt`.

**Subtopic map**
1. *Log-sum-exp.* $\mathrm{LSE}(z) = m + \log\sum_i e^{z_i - m}$, $m = \max_i z_i$. Derive the identity, show the
   largest term becomes $e^0 = 1$ so the sum is in $[1, n]$ (no overflow, no log(0)). Gradient of LSE is softmax.
2. *Softmax and log-softmax.* Stable softmax by max-subtraction (invariance to shifts, derived); log-softmax as
   $z_i - \mathrm{LSE}(z)$ rather than `log(softmax(z))`, which underflows to `-inf` for very negative logits.
3. *Cross-entropy from logits.* $\ell = \mathrm{LSE}(z) - z_y$; gradient $\mathrm{softmax}(z) - e_y$ (derive); why
   frameworks fuse `log_softmax + nll` and why passing probabilities into a CE expecting logits is a classic bug.
4. *Sigmoid, BCE and softplus.* $\log\sigma(x) = -\mathrm{softplus}(-x)$; stable softplus $\max(x,0) + \log1p(e^{-|x|})$;
   BCE-with-logits form $\max(x,0) - xy + \log(1+e^{-|x|})$ (derive from the two branches). Why `BCELoss` after a
   separate sigmoid is unsafe in fp16 (it is blocked under autocast in PyTorch).
5. *log1p / expm1.* Why `log(1+x)` loses everything for $|x| < u$ and how `log1p` avoids forming $1+x$. Uses: softplus,
   log-probabilities near 1, KL terms, `log(1 - p)` = `log1p(-p)`.
6. *Variance.* One-pass $E[x^2]-E[x]^2$ cancels catastrophically; two-pass is stable; Welford's online update (derive
   $M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$); Chan's merge for parallel/blocked reductions (how fused
   LayerNorm kernels and BatchNorm across GPUs combine partial statistics).
7. *Epsilon placement.* LayerNorm/RMSNorm: $x/\sqrt{\sigma^2 + \epsilon}$ (inside the sqrt) vs Adam's
   $\hat m/(\sqrt{\hat v} + \epsilon)$ (outside). Effect on gradients at zero variance; ε must be representable in the
   compute dtype (1e-8 underflows in fp16; 1e-5/1e-6 typical for norms).
8. *Online softmax.* Single pass with running max $m$ and rescaled denominator $d \leftarrow d\,e^{m_{old}-m_{new}} + e^{z-m_{new}}$
   (derive); this is the core of FlashAttention (link `llm.flash-attention`, id to confirm).
9. *Other traps*: `sqrt` at 0 has infinite gradient (norms of zero vectors: add ε or use `torch.linalg.vector_norm`
   carefully); `log` of probabilities after clamping; division by token counts that can be zero; `exp` in temperature
   scaling; integer overflow in index arithmetic.
10. *Interview probes*: "implement a stable softmax", "why does max-subtraction not change the output", "derive the
    gradient of cross-entropy wrt logits", "why prefer BCEWithLogits".

**Worked computations (verified)**
- softmax([1000, 1001, 1002]): naive gives NaN (inf/inf); stable gives [0.0900, 0.2447, 0.6652].
- LSE([1000, 1001, 1002]) = 1002 + log(1 + e^{-1} + e^{-2}) = 1002.4076.
- `log(1 + 1e-17)` = 0.0 in fp64, `log1p(1e-17)` = 1e-17.
- Welford on {4, 7, 13, 16} step by step (means 4, 5.5, 8, 10; $M_2$ = 0, 4.5, 42, 90; population variance 22.5).

**Question ideas**
- Code-reading: a custom loss computes `torch.log(torch.softmax(z, -1))[range(n), y]`; when does it return inf and what is the fix?
- Derive: the gradient of $\mathrm{LSE}$; then show the CE gradient is bounded in $[-1, 1]$ per logit.
- Predict: computing LayerNorm statistics in bf16 with $E[x^2]-E[x]^2$ on activations with mean 300 and std 1.
- Which is false: "subtracting the max changes the softmax output" (false) / "log-softmax is computed as z − LSE(z)" / ...
- Compare: ε inside vs outside the square root.

**Pitfalls / verify**
- Max-subtraction prevents overflow but tiny probabilities still underflow to 0 in softmax; that is why log-softmax exists.
- In mixed precision, autocast already upcasts softmax/log_softmax/norms to fp32 (check the op list for the current version).
- Welford's update in the literature has several equivalent forms; pick one and verify on the example.

**Figures**
- `softplus`: naive `log(1+exp(x))` in fp32 vs stable softplus over x in [-100, 100] showing overflow/underflow regions.
- `lse-shift`: bar chart of $e^{z_i}$ vs $e^{z_i - m}$ on a log axis (one bar pinned at 1).
- `welford`: running mean and variance estimates on a stream with large offset, naive vs Welford, in fp32.

---

## sys.mixed-precision — Mixed-precision training

Level core · prereqs sys.fp-formats · **W1** · ~9 cards

**Problem.** "Half-precision matmuls run on Tensor Cores (H100: ≈ 990 TFLOP/s bf16 vs ≈ 66 TFLOP/s on the CUDA cores,
Scaling Book Part 12) and halve activation memory, but naively training in fp16 diverges
or stalls. What exactly must stay in fp32, and why?"

**Primary sources**
- Micikevicius et al. 2018 §3.1 (fp32 master copy; the update-to-weight ratio argument), §3.2 (loss scaling; for SSD, many activation-gradient
  values fell below fp16's range and a loss scale of 8 was enough), §3.3 (fp32 accumulation; which ops in fp32) —
  `sys_micikevicius2018_mixed_precision.txt` (https://arxiv.org/abs/1710.03740).
- NVIDIA *Train With Mixed Precision* §2 (loss scaling, choosing a scale factor, dynamic scaling), §3 (AMP), §4
  (Tensor Core shape rules) — `sys_nvidia_mixed_precision.txt`.
- PyTorch `torch.amp` "Autocasting", "Gradient Scaling", "Autocast Op Reference" (GradScaler defaults: init_scale 65536,
  growth_factor 2, backoff_factor 0.5, growth_interval 2000) — `sys_pytorch_amp.txt`; AMP examples "Working with Unscaled
  Gradients › Gradient clipping", "Gradient accumulation" — `sys_pytorch_amp_examples.txt`.
- Kalamkar et al. 2019 (bf16 trains without loss scaling) — `sys_kalamkar2019_bf16.txt`.
- Ultra-Scale Playbook "Mixed Precision Training" (lines ≈4585–5147) — `sys_ultrascale2025_playbook.txt`.

**Subtopic map**
1. *Why bother.* Tensor-core throughput (H100 bf16 ≈ 989 TFLOP/s dense vs ≈ 66 TFLOP/s on CUDA cores; one-line definition of a
   Tensor Core here, full treatment in `sys.gpu-basics`), half the
   bytes for activations and communication. Mixed precision does **not** by itself reduce weight+optimizer memory
   (derive: 2 + 2 + 4 + 4 + 4 = 16 bytes/param vs 4 + 4 + 8 = 16 for fp32 Adam).
2. *The three failure modes of pure fp16*: (a) overflow (activations/gradients > 65504), (b) gradient underflow
   (values below ≈ 2.98e-8, half the smallest subnormal, round to 0; values below 6.1e-5 survive only as subnormals
   with reduced precision), (c) lost updates ($\eta g$ below half an ulp of $w$).
3. *Fix (c): fp32 master weights.* Keep an fp32 copy, cast to half for forward/backward, apply the update in fp32. Derive
   the condition: an update is lost when $|\eta g| < \tfrac12 \mathrm{ulp}(w) \approx |w|\,2^{-(m+1)}$, up to a factor 2 within a
   binade (worst case just above a power of two); for fp16 that is a relative update below about $2.4$–$4.9\times10^{-4}$, typical
   late in training. Subtlety: $w - \delta$ at $w = 1$ rounds using the smaller spacing below 1 (half-ulp $2.44\times10^{-4}$).
4. *Fix (b): loss scaling.* Multiply the loss by $S$; by the chain rule every gradient is multiplied by $S$; unscale
   (divide by $S$) in fp32 before the optimizer. Static scaling vs dynamic scaling (halve and skip the step on inf/NaN;
   double after $k$ clean steps). Why skipping a step is safe. Interaction with gradient clipping (unscale first) and
   with gradient accumulation (unscale once after the last micro-batch).
5. *Fix (a) and precision-sensitive ops: autocast.* Matmuls/convs in half with fp32 accumulation; reductions, softmax,
   log, exp, norms, losses in fp32 (the op lists); "promote to widest" ops. Why the accumulator matters (sum of $K$
   products in a dot product, link `sys.fp-error`).
6. *bf16 vs fp16.* bf16 has fp32's exponent range, so gradients rarely underflow and loss scaling is unnecessary, but
   only 8 significant bits; fp32 master weights and fp32 optimizer state remain essential. When fp16 is still used
   (older GPUs, inference).
7. *What lives in which dtype* (table): params (bf16 working copy + fp32 master), grads (bf16 or fp32; fp32
   accumulation for stability), Adam m and v (fp32), activations (bf16), communication (bf16 grads all-reduce vs fp32
   reduce; FSDP `reduce_dtype`). With standard `torch.autocast` + DDP the parameters, hence `.grad`, are fp32, so DDP
   all-reduces **fp32** gradients unless the model is cast to bf16 or a bf16 compression comm hook is registered.
8. *Memory accounting* for a 7B model: 16 bytes/param → 112 GB for weights+grads+optimizer before activations, vs
   80 GB per H100 (motivates ZeRO). Variant with fp32 gradient accumulation buffer: 20 bytes/param (Playbook table).
9. *Tensor Core practicalities*: dimensions multiple of 8 (fp16/bf16) for efficient kernels (NVIDIA §4); padding vocab.
10. *Interview probes*: "why do we need master weights if bf16 has fp32's range", "what does GradScaler do on an inf",
    "where must you call unscale_", "does mixed precision save memory".

**Worked computations (verified)**
- A gradient of 1e-8 casts to 0 in fp16; scaled by $2^{16}$ it becomes ≈ 6.55e-4 (representable).
- Lost update: $w = 1.0$, $\eta g = 10^{-4}$ in fp16 → no change; in bf16 even $10^{-3}$ is lost.
- 7B: 16 × 7e9 = 112 GB; with an extra fp32 gradient buffer 20 × 7e9 = 140 GB (matches the Playbook table: 112 / 140 GB).
- Dynamic scaling trace: start $2^{16}$; overflow at step 3 → $2^{15}$, step skipped; 2000 clean steps → $2^{16}$.

**Question ideas**
- Predict: training in fp16 with loss scaling but without fp32 master weights; which symptom appears and when (loss
  plateaus late as updates vanish)?
- Debug: GradScaler's scale keeps halving every few steps until it reaches tiny values; what does that indicate (a real
  inf/NaN in the forward, not underflow)?
- Code order: where do `scaler.unscale_(opt)` and `clip_grad_norm_` go relative to `scaler.step`? What breaks if you clip scaled grads?
- Which is false: "bf16 training typically needs dynamic loss scaling" (false) / "autocast runs softmax in fp32" / ...
- Compute: bytes per parameter for AdamW with bf16 params, fp32 master weights, fp32 grads, fp32 m and v.
- Explain: Micikevicius needed a loss scale of only 8 for SSD, yet GradScaler starts at 65536 — why start high and back off?
- Predict: DDP + autocast on a 1.3B model — are the all-reduced gradients bf16 or fp32, and what does that do to comm time?

**Pitfalls / verify**
- Do not claim mixed precision halves total memory; it halves activations and comm volume.
- Some stacks keep gradients in fp32 (Playbook note on nanotron); others accumulate in bf16. State which you assume.
- Overlap: `fund.training-loop` (items 7–9) introduces AMP/GradScaler usage; this lesson owns the why (formats, failure modes,
  memory, comm dtype).
- GradScaler defaults: verify against the current docs before quoting.
- `torch.cuda.amp.*` is deprecated in favour of `torch.amp.*`; mention neutrally.

**Figures**
- `grad-histogram`: synthetic log-scale histogram of gradient magnitudes with the fp16 representable range shaded and a
  second copy shifted right by $\log_2 S$ (illustrating loss scaling; label as illustrative, modelled on Micikevicius Fig. 3).
- `dtype-flow`: diagram of one training step: fp32 master → cast → bf16 forward/backward → grads → unscale/clip → fp32 update.
- `dynamic-scale`: step plot of the loss scale over 10k steps with overflow events.

---

## sys.fp8-training — FP8 and block-scaled low-precision training

Level advanced · prereqs sys.mixed-precision, sys.roofline (later in order; W3, so written after it) · W3 · ~8 cards

**Problem.** "FP8 doubles tensor-core throughput again, but E4M3 has only 18 binades and 3 mantissa bits. How do you train
in it without diverging?" Answer: per-tensor (or per-block) scaling factors, the right format per tensor, higher-precision
accumulation, and keeping sensitive ops out of fp8.

**Primary sources**
- Micikevicius et al. 2022 §2 (scaling factors; why loss-scaling-style global scaling is not enough), §3 (formats), §4
  (results; which layers kept in higher precision) — `sys_micikevicius2022_fp8.txt`.
- NVIDIA Transformer Engine FP8 primer (E4M3 forward / E5M2 backward "HYBRID", just-in-time vs delayed scaling, amax
  history, MXFP8 with one scale per 32 values) — `sys_nvidia_te_fp8_primer.txt`.
- DeepSeek-V3 §3.3 (FP8 mixed-precision framework, fine-grained 1×128 activation tiles and 128×128 weight blocks,
  increased accumulation precision) — shared `llm_deepseek2024_v3.txt`.
- Rouhani et al. 2023 (OCP Microscaling MX formats) — shared `llm_rouhani2023_microscaling.txt`. Overlap: the FP8/MX part of
  `llm.quantization-advanced` (id from the current LLM plan draft; confirm) covers formats for inference; this lesson owns training.
- TorchTitan (Float8 training in PyTorch) — `sys_liang2024_torchtitan.txt`.

**Subtopic map**
1. Motivation: H100 fp8 ≈ 2× bf16 FLOP/s (ridge point doubles to ≈ 591 FLOPs/byte); memory and comm halve again.
2. Why fp8 needs scaling: 3 mantissa bits mean relative error up to 1/16; 17–18 binades cannot cover the spread of
   gradient and activation magnitudes in one tensor across training. A scale $s$ maps $\max|x|$ near the format max.
3. Per-tensor scaling: $x_8 = \mathrm{cast}(x \cdot s)$, $s = \text{fmt\_max}/\text{amax}$ (with margin); GEMM on fp8,
   output rescaled by $1/(s_A s_B)$; accumulation in fp32.
4. Just-in-time (current) scaling vs delayed scaling (amax history window, e.g. 16 steps, `max` reduction); why delayed
   scaling is faster (no extra pass) and its risk (sudden spikes saturate).
5. Format choice: E4M3 for weights and activations (precision), E5M2 for gradients (range); all-E4M3 recipes.
6. What stays in higher precision: master weights, optimizer state, softmax/norms, the first and last layers or
   embeddings, attention score computation in many recipes; the reader should justify each.
7. Block / fine-grained scaling: outliers in one channel force a tiny global scale and flush everything else to zero;
   per-block scales (MXFP8: 32 elements with a shared power-of-two scale; DeepSeek-V3 tiles) fix this. Trade-offs:
   scale storage, kernel support.
8. Accumulation precision: limited-precision accumulation inside fp8 tensor cores and DeepSeek-V3's periodic promotion to
   fp32 (§3.3.2); why long $K$ dimensions are the risk.
9. Distributed details: amax must be agreed across tensor-parallel ranks; fp8 all-gather of weights in FSDP.
10. Probes: "why not just use loss scaling for fp8", "E4M3 vs E5M2 for gradients", "what is MX", "what breaks with outliers".

**Worked computations**
- E4M3 max 448, min normal $2^{-6}$, min subnormal $2^{-9}$; E5M2 max 57344, min subnormal $2^{-16}$ (verified formulas).
- Scale example: amax = 3.2 → $s = 448/3.2 = 140$; a value 0.001 maps to 0.14, which E4M3 stores as 0.140625 (verified
  with `torch.float8_e4m3fn`). With one outlier making amax = 3200, $s = 0.14$ and 0.001 maps to $1.4\times10^{-4}$, below
  half the smallest subnormal ($2^{-10} \approx 9.8\times10^{-4}$), so it rounds to 0.
- E4M3 spans $\log_2(448/2^{-9}) \approx 17.8$ binades including subnormals.
- H100 ridge point: dense fp8 1.979e15 / 3.35e12 ≈ 591 FLOPs/byte vs ≈ 295 (bf16); the Scaling Book's rounded 2.0e15 would give
  597 — quote "≈ 2 × 295 ≈ 590".

**Question ideas**
- Compute: given amax history [2, 3, 8, 3] and E4M3, what scale does a `max` delayed recipe pick, and what happens if the
  current step's amax is 20?
- Which is false: "E5M2 represents infinities" / "E4M3 has more range than E5M2" (false) / ...
- Predict: one activation channel has values 100× larger than the rest; per-tensor vs per-block scaling outcomes.
- Explain: why master weights stay fp32 even in fp8 training.

**Pitfalls / verify**
- MXFP8's block size (32) and power-of-two (E8M0) scales: verify in the TE primer/MX spec before quoting.
- DeepSeek-V3 tile sizes (1×128 activations, 128×128 weights): verify in §3.3.2.
- Do not claim specific speedups without a source.

**Figures**
- `outlier-scaling`: values of one tensor on a log axis, the E4M3 window placed by per-tensor scaling vs per-block windows.
- `delayed-scaling`: amax over steps, history window, chosen scale, and a spike that saturates.

---

## sys.stability-tricks — The numerics of large logits: z-loss, soft-capping and QK-norm

Level advanced · prereqs sys.stable-numerics, sys.mixed-precision · W3 · ~5 cards

**Scope (after review).** `llm.training-stability` (id from the current LLM plan draft; confirm) owns the training-recipe
view: what fails in LLM pretraining, Adam ε/β₂, warmup, data-side spikes, QK-Clip, μP. This lesson keeps only the
*numerics*: why large logits cost precision in bf16, what each bounding function does to values and gradients, and
representability. Adam ε is not covered here (it is in `sys.tuning-pipeline` and the `fund.adam` insertion row). If the
lesson ends up under 5 cards, merge it into `sys.stable-numerics` as 1–2 cards.

**Primary sources**
- Wortsman et al. 2023 §3.1.1 (attention-logit growth; qk-layernorm), §3.1.2 (output-logit divergence; z-loss $10^{-4}\log^2 Z$) —
  `sys_wortsman2023_small_scale_instabilities.txt`.
- PaLM §5 (z-loss) — shared `llm_chowdhery2022_palm.txt`; Gemma 2 §2 (soft-capping 50 attention / 30 final logits) —
  shared `llm_team2024_gemma2.txt`; ViT-22B (QK-norm) — shared `llm_dehghani2023_vit22b.txt`.
- `sys.fp-formats` spacing results (bf16 ulp at magnitude $x$ ≈ $x\cdot2^{-7}$).

**Subtopic map**
1. *Precision cost of large logits.* bf16 spacing at $|z| = 64$ is $64\cdot2^{-7} = 0.5$: logits that differ by less
   than that become equal, and $\exp$ of a rounded logit carries a large relative error. Softmax is shift-invariant, so
   $\log Z$ can drift without changing the loss, yet the drift pushes all logits to magnitudes where bf16 resolution is coarse.
2. *z-loss, mechanism stated correctly.* $\lambda(\log Z)^2$ has gradient $2\lambda\log Z\cdot\mathrm{softmax}(z)$ — proportional to
   the softmax, not to the all-ones vector. Decompose it: the component along $\mathbf 1/\sqrt n$ is a uniform downward
   shift that lowers $\log Z$ and leaves CE unchanged; the orthogonal remainder slightly shrinks the top logit's margin
   (a small entropy-raising regulariser that the CE gradient dominates).
3. *Logit soft-capping* $c\tanh(z/c)$ vs a hard clamp: value bounded by $c$; derivative $1 - \tanh^2(z/c)$ is smooth and
   never exactly zero, whereas a clamp has zero gradient beyond the cap. The argmax is unchanged (monotone map).
4. *QK-norm as bounding*: normalising $q$ and $k$ bounds $|q\cdot k| \le \|q\|\|k\| = $ (learned scale)², so attention
   logits cannot grow with weight norm (link `llm.training-stability` for the empirical story).
5. *Probes*: "why does a drifting log Z hurt in bf16 if the loss is unchanged", "what does the z-loss gradient do to the
   softmax", "tanh cap vs clamp".

**Worked computations (verified)**
- bf16 spacing at 64: 0.5; at 1: $2^{-7} \approx 0.0078$.
- z-loss for logits $[2, 0]$, $\lambda = 10^{-4}$: $\log Z \approx 2.127$, softmax ≈ $[0.881, 0.119]$, gradient ≈
  $[3.75\times10^{-4}, 5.07\times10^{-5}]$. Decomposition: shift component $[2.13, 2.13]\times10^{-4}$ (lowers $\log Z$ only),
  orthogonal component $[1.62, -1.62]\times10^{-4}$ (shrinks the margin). An exaggerated step $z - 100g$ moves the softmax
  from $[0.8808, 0.1192]$ to $[0.8774, 0.1226]$: slightly flatter, not unchanged.
- Soft-cap with $c = 30$: $z = 10 \to 9.65$ (derivative ≈ 0.897); $z = 100 \to 29.92$ (derivative ≈ 0.0051).

**Question ideas**
- Derivation step: decompose the z-loss gradient into shift and orthogonal parts; which part changes the CE loss?
- Compare: clamp vs tanh soft-cap gradients beyond the cap.
- Compute: bf16 spacing at a given logit magnitude; which logit differences are lost?
- Which is false: "soft-capping changes the argmax" (false) / "z-loss leaves the softmax exactly unchanged" (false) — use one per MCQ.

**Pitfalls / verify**
- Do not say z-loss "only shifts" logits (see item 2).

**Figures**
- `softcap`: $z$ vs $c\tanh(z/c)$ and its derivative for $c = 30$ next to a hard clamp.
- `zloss-decomposition`: the z-loss gradient for a 2-logit example drawn as a vector split into shift and margin components.

---

# Part B: Gradients

## sys.vanishing-exploding — Vanishing and exploding gradients in deep networks and transformers

Level core · prereqs fund.initialization, fund.rnn · W2 · ~8 cards

**Scope (re-scoped after review).** The basics are taught in `fund.*`: the RNN Jacobian product, Pascanu's conditions,
the eigenvalue picture, cliffs and truncated BPTT in `fund.rnn`; variance propagation and the 0.9⁵⁰ / 1.1⁵⁰ example in
`fund.initialization`; sigmoid saturation (0.25^L) in `fund.activations`; LSTM/GRU gating in `fund.lstm-gru`; norm layers
in `fund.normalization`. This lesson gives **one recap card** of the Jacobian-product argument and spends the rest on what
fund does not cover: why *deep residual networks and transformers* train at all, how gradient scale behaves with depth and
norm placement, and what explosion looks like in large-scale training. No figure from `fund.rnn` / `fund.initialization`
is redrawn here; link them (`fund.rnn/gradient-norm-vs-lag`, `fund.rnn/clipping-cliff`) instead.

**Problem the lesson answers.** "A 100-layer plain MLP doesn't train, but a 100-layer pre-LN transformer does. Why, and
what still goes wrong at scale?"

**Primary sources**
- He et al. 2016 (ResNet: identity shortcuts) — shared `papers/he2016_resnet.txt`.
- Xiong et al. 2020 (pre-LN vs post-LN: gradient norm of the last layers at initialisation, why post-LN needs warmup) —
  shared `papers/xiong2020_preln.txt`.
- GPT-2 §2.3 (residual-branch weights scaled by $1/\sqrt{N}$ at init) — shared `llm_radford2019_gpt2.txt`.
- Goodfellow DL §8.2.5 and §10.7 (recap of the Jacobian-product argument only) — shared `dlb_ch08_optimization.txt`, `dlb_ch10_rnn.txt`;
  Pascanu et al. 2013 §2.1 — shared `papers/pascanu2013_rnn_difficulty.txt`.
- Wortsman et al. 2023 §3.1 (attention-logit growth as an "explosion" mechanism) — `sys_wortsman2023_small_scale_instabilities.txt`.
- Refresher source for spectral norm and submultiplicativity: Goodfellow DL §2 (shared `dlb_ch02_linear_algebra.txt`).

**Refreshers to include inline.** Spectral norm $\|A\|_2 = \sigma_{\max}(A)$; $\|AB\| \le \|A\|\|B\|$; eigenvalues vs singular
values (they coincide only for normal matrices).

**Subtopic map**
1. *Recap (one card, link fund).* $\partial\mathcal L/\partial h_0 = \prod_\ell J_\ell^\top\,\partial\mathcal L/\partial h_L$ with
   $J_\ell = \mathrm{diag}(f')W_\ell$; $\|\prod J_\ell\| \le \prod\|J_\ell\|$, so norms shrink or grow geometrically.
2. *Residual Jacobians, derived for $L$ blocks.* $h_{\ell} = h_{\ell-1} + F_\ell(h_{\ell-1})$ gives
   $J_\ell = I + \partial F_\ell/\partial h$; the product $\prod_\ell (I + A_\ell)$ expands into a sum over all subsets of
   blocks and always contains the identity path, so the gradient cannot vanish just because each $A_\ell$ is small.
   Show the two-block expansion, then the $L$-block bound $\prod(1 + \|A_\ell\|) \le e^{\sum\|A_\ell\|}$ (growth is
   controlled if $\sum_\ell \|A_\ell\| = O(1)$).
3. *Why residual-branch scaling.* With $L$ residual additions of variance-$v$ branches the stream variance grows like
   $1 + Lv$; scaling branch outputs by $1/\sqrt{L}$ (GPT-2: $1/\sqrt{N}$ for $N$ residual layers) keeps forward activations
   and the Jacobian sum $O(1)$. Link `llm.training-stability` (id from the current LLM plan draft; confirm) for the recipe view.
4. *Pre-LN vs post-LN.* Post-LN puts the norm *after* the addition, so the identity path passes through every
   LayerNorm's Jacobian; Xiong 2020 shows last-layer gradient norms at init are large for post-LN and roughly
   depth-independent for pre-LN; hence post-LN needs LR warmup. Reader must state the two block equations and which one
   keeps a clean identity path.
5. *Normalisation as a gradient-scale stabiliser*: LayerNorm's Jacobian is scale-invariant ($\partial\,\mathrm{LN}(cx)/\partial x = \frac1c\,\partial\,\mathrm{LN}(x)/\partial x$), which
   damps growth of pre-activation scale (link `fund.normalization` for the full treatment).
6. *Diagnostics at scale*: per-layer (or per-block) gradient-norm logging, update-to-weight ratio, activation RMS per layer,
   attention entropy; what each pattern means (early layers ≈ 0 → vanishing; a spike in one block → local explosion).
7. *Explosion in practice*: loss spikes from attention-logit growth (link `sys.stability-tricks`, `llm.training-stability`),
   fp16 overflow masquerading as explosion (inf grads that GradScaler skips; link `sys.mixed-precision`), too-high LR or
   bad init, a few bad batches. What clipping does and does not fix (link `sys.gradient-clipping`).
8. *Probes*: "why do residual connections help — give the Jacobian argument", "pre-LN vs post-LN and warmup", "why scale
   residual branches by 1/√L", "your gradient-norm plot shows X; diagnose".

**Worked computations (verified)**
- Two-block residual product with $\|A_i\| = 0.1$: plain product of the $A$s has norm ≈ 0.01, residual product
  $I + A_1 + A_2 + A_2A_1$ keeps norm ≈ 1 (writers: give a concrete 2×2 example and compute its singular values).
- Residual-stream variance after $L = 48$ unit-variance branch additions: $1 + 48 = 49$ (std ×7) without scaling;
  with $1/\sqrt{L}$ scaling, $1 + 1 = 2$.
- Recap only (from fund): $0.9^{50} \approx 0.0052$, $1.1^{50} \approx 117$.

**Question ideas**
- Derive: expand $\prod_{\ell=1}^{3}(I + A_\ell)$ and identify the identity path; bound its norm.
- Predict: moving LayerNorm from pre- to post-position in a 48-layer transformer with no warmup.
- Debug: per-block gradient norms are flat except one attention block that spikes before every loss spike; likely causes.
- Which is false: "gradient clipping fixes vanishing gradients" (false) / "the residual Jacobian contains an identity term" / ...
- Compute: residual-stream variance with and without $1/\sqrt{L}$ branch scaling.

**Pitfalls / verify**
- Pascanu's text says "absolute value of the largest eigenvalue", but its proof bounds the spectral norm $\|W\|$; use
  $\sigma_{\max}$ consistently with `fund.rnn` and note the discrepancy.
- Verify Xiong 2020's exact statement (gradient norm scaling with $L$ for post-LN vs pre-LN) before quoting.

**Figures**
- `grad-norm-depth`: log per-layer gradient norm vs layer index for a 50-layer plain MLP vs the same with residual
  connections vs pre-LN residual, computed by actually backpropagating in numpy (the plain sigmoid/Xavier curves already
  live in `fund.initialization`, so show only the residual comparison).
- `pre-post-ln`: gradient norm per layer at initialisation for a toy pre-LN vs post-LN stack (computed).

---

## sys.gradient-clipping — Gradient clipping

Level core · prereqs sys.vanishing-exploding, sys.mixed-precision, fund.rnn · W2 · ~7 cards

**Problem.** "A single bad batch produces a gradient 100× larger than usual and one SGD step destroys the model. How do we
bound the step without changing its direction, and how does that interact with Adam, accumulation, mixed precision and
sharding?"

**Overlap with fund.** `fund.rnn` already teaches norm vs value clipping and the cliff picture (figure
`fund.rnn/clipping-cliff`). Here items 1–3 compress into **one recap card**; the lesson's weight is on the systems
content (items 4–8), which fund does not cover.

**Refreshers to include inline.** L-smoothness (Lipschitz gradient) and the step-size bound it implies, needed for Zhang
2020's $(L_0, L_1)$-smoothness argument.

**Primary sources**
- Pascanu et al. 2013 §3.2 "Scaling down the gradients" (Algorithm 1; threshold heuristic from average norms) — shared `papers/pascanu2013_rnn_difficulty.txt`.
- Goodfellow DL §10.11.1 "Clipping Gradients" (norm vs element-wise; clipping preserves direction for norm clipping only) — shared `dlb_ch10_rnn.txt`.
- d2l §9.5.3 "Gradient Clipping" (the $\min(1, \theta/\|g\|)$ form and its Lipschitz argument) — `sys_d2l_rnn_scratch_clipping.txt`.
- PyTorch `clip_grad_norm_` (total norm over all parameters viewed as one vector; returns the pre-clip norm;
  `error_if_nonfinite`) — `sys_pytorch_clip_grad_norm.txt`; AMP examples "Gradient clipping" (unscale first) — `sys_pytorch_amp_examples.txt`.
- Zhang et al. 2020 (why clipping accelerates: $(L_0, L_1)$-smoothness) — `sys_zhang2020_why_clipping.txt`; Brock et al. 2021 §4
  (Adaptive Gradient Clipping) — `sys_brock2021_nfnets_agc.txt`.

**Subtopic map**
1. *Definition and derivation.* $g \leftarrow g\cdot\min(1, c/\|g\|_2)$. Show: direction preserved, $\|g_{clipped}\| \le c$,
   identity when $\|g\| \le c$. The step $\eta g$ therefore has length at most $\eta c$ (a trust-region view).
2. *Global norm vs per-parameter norm vs value clipping.* Global: $\|g\| = \sqrt{\sum_p \|g_p\|^2}$ over all tensors
   (what `clip_grad_norm_` does). Per-tensor clipping changes the relative scale between layers. Value clipping
   (element-wise clamp) changes direction (show on $g = (3, 0.1)$ with $c = 1$).
3. *Why it helps.* Cliffs (Pascanu): near a cliff the gradient is huge and the linear model is valid only locally.
   Smoothness view (Zhang 2020): if the local smoothness grows with $\|\nabla f\|$, a step size inversely proportional
   to the gradient norm is the right choice, which clipping implements.
4. *Choosing the threshold.* From the running distribution of gradient norms (the tuning playbook's rule of thumb is
   taught in `sys.tuning-diagnostics`; link, don't duplicate); global-norm clipping at 1.0 is the setting in GPT-3 (App. B, shared `papers/brown2020_gpt3.txt`) and
   Llama 2 (§2.2, shared `llm_touvron2023_llama2.txt`); log the pre-clip norm and the fraction of clipped steps. If almost every step is clipped, the
   method behaves like normalised GD and the LR is effectively rescaled.
5. *Clipping and Adam.* Adam is approximately invariant to a constant gradient scale, but clipping still matters: it
   limits a single outlier step's effect on $m$ and $v$ (one spike inflates $v$ for ~$1/(1-\beta_2)$ steps and then $m/\sqrt v$ is small). Order: clip the raw gradient, then the optimizer.
6. *Ordering with other features.* Gradient accumulation: clip once after all micro-batches are summed. Mixed precision:
   `scaler.unscale_` before clipping; skip if inf/NaN. Weight decay is not clipped (decoupled). In DDP the gradients are
   already averaged, so every rank computes the same norm. In FSDP/ZeRO/TP each rank holds a shard, so the global norm
   needs a sum-of-squares all-reduce (FSDP's own `clip_grad_norm_`, DTensor-aware clipping). With pipeline parallelism
   the norm is reduced across stages.
7. *Adaptive Gradient Clipping* (NFNets): clip unit-wise when $\|G\|/\|W\|$ exceeds $\lambda$; why it is scale-aware. Brief.
8. *Non-finite gradients.* Clipping with a NaN norm produces NaNs everywhere; `error_if_nonfinite`, skipping steps.
9. *Probes*: "norm vs value clipping", "does clipping fix vanishing gradients", "where to clip with AMP/accumulation/FSDP",
   "why log the gradient norm".

**Worked computations (verified)**
- $g = (3, 4)$, $c = 1$: $\|g\| = 5$, clipped $g = (0.6, 0.8)$.
- Two tensors with grads $(3, 4)$ and $(12)$: global norm 13; with $c = 1.3$ both are scaled by 0.1 → $(0.3, 0.4)$ and $(1.2)$;
  per-tensor clipping at 1.3 would give $(0.78, 1.04)$ and $(1.3)$, a different direction.
- Value clipping of $(3, 0.1)$ at 1 gives $(1, 0.1)$: the direction rotates from 1.9° to 5.7° off the x-axis (recompute).

**Question ideas**
- Compute: global-norm clip of a 3-tensor gradient; which tensors change?
- Code order bug: `clip_grad_norm_` called before `scaler.unscale_`; what is the effect (threshold compared against $S\|g\|$,
  so effectively almost every step is clipped hard)?
- Predict: gradient accumulation over 8 micro-batches with clipping applied after each micro-batch; how does this differ
  from clipping the sum?
- Which is false: "norm clipping preserves direction" / "value clipping preserves direction" (false) / ...
- FSDP: why can't each rank clip its own shard independently?

**Pitfalls / verify**
- PyTorch's implementation detail of a small constant in the denominator is not in the cached doc; don't quote it.
- Clipping is not a remedy for vanishing gradients.

**Figures**
- (Do not redraw the cliff; link `fund.rnn/clipping-cliff`.)
- `clip-geometry`: 2D gradient vectors, the circle of radius $c$, norm-clipped vs value-clipped (box) results.
- `grad-norm-trace`: synthetic gradient-norm time series with spikes, the threshold line, and clipped steps marked.

---

# Part C: Memory

## sys.memory-anatomy — Where GPU memory goes when training

Level core · prereqs sys.mixed-precision · **W1** · ~9 cards

**Problem.** "My 7B model is 14 GB in bf16, yet training it OOMs on an 80 GB GPU. Where did the memory go?" The reader
must be able to produce a full memory budget for a transformer from its hyperparameters, including activations.

**Primary sources**
- Ultra-Scale Playbook "Memory usage in Transformers" (profile of a step; first step vs later steps; weights/grads/optimizer
  formulas incl. the 16 vs 20 bytes/param table; activation formula), lines ≈471–830 — `sys_ultrascale2025_playbook.txt`.
- Korthikanti et al. 2022 §4 (activation memory), §4.1 itemised per-layer derivation giving $sbh(34 + 5as/h)$ bytes,
  §4.3 embeddings/logits terms ($4sbv$ for fp32 logits), Table 2 — `sys_korthikanti2022_seqpar_recompute.txt`.
- ZeRO §3.1 (model states: $2N + 2N + KN$, $K = 12$ for mixed-precision Adam), §3.2 (residual states: activations,
  temporary buffers, fragmentation) — `sys_rajbhandari2020_zero.txt`.
- Overlap: `llm.params-flops` (id from the current LLM plan draft; confirm) also counts params and memory; this lesson owns the
  training-memory derivation (activations, optimizer state, peaks).
- Shared: `llm_eleuther2023_transformer_math.txt` (rules of thumb, cross-check).

**Subtopic map**
1. *Four consumers*: parameters, gradients, optimizer state ("model states"), activations; plus CUDA context and kernels
   (≈1–2 GB, Playbook "Memory usage in Transformers" note), temporary buffers (GEMM workspaces, communication buckets, all-gathered shards), fragmentation.
2. *Model states per parameter.* Derive bytes/param for: fp32 SGD (4+4), fp32 SGD+momentum (12), fp32 Adam (16), mixed
   bf16 Adam (2 + 2 + 4 master + 4 + 4 = 16), with fp32 grad accumulation (20), Adafactor/8-bit Adam (mention). Why
   optimizer state dominates.
3. *Parameter count of a transformer* from $(h, L, V)$: $\approx 12Lh^2 + Vh$ (4h MLP; derive $4h^2$ attention + $8h^2$ MLP);
   SwiGLU/GQA variants change the constant. Reader computes $N$ for a given config.
4. *Why activations must be stored.* Backprop of $Y = XW$ needs $X$ for $\partial L/\partial W = X^\top \partial L/\partial Y$;
   nonlinearities need their inputs; dropout needs masks; softmax needs its output. Activations scale with $b \cdot s$.
5. *Derive per-layer activation bytes* (Korthikanti §4.1, 16-bit activations, 1-byte dropout masks): attention block
   $11sbh + 5as^2b$ (QKV input 2sbh; Q and K 4sbh; softmax output $2as^2b$; softmax dropout mask $as^2b$; dropout output
   $2as^2b$ and V 2sbh; output-projection input 2sbh; output dropout mask sbh); MLP $19sbh$ (inputs 2sbh and 8sbh, GeLU input
   8sbh, dropout mask sbh); two LayerNorm inputs $4sbh$. Total $sbh(34 + 5as/h)$. Reader must reproduce at least the
   structure and say which term is quadratic in $s$.
6. *Effect of modern choices*: FlashAttention does not materialise the $s\times s$ scores, so the $5as/h$ term disappears
   (only $O(sb\,a)$ softmax statistics are kept); no dropout in LLM pretraining removes the mask terms; SwiGLU changes the
   MLP constant. Writers should state "≈ 34sbh per layer with FlashAttention, slightly less without dropout" only after
   recounting for their stated architecture.
7. *Other big buffers*: output logits $s\,b\,V$ in fp32 for the loss (and their gradient); embedding gradients; the
   optimizer step's temporaries (foreach kernels allocate per-group copies).
8. *Peak vs steady state.* Memory over a step (forward rises, backward frees activations while grads appear, optimizer
   step); the first step differs (optimizer state is allocated lazily at the first `step()` → "OOM on step 2"). Allocator internals (reserved vs
   allocated, fragmentation, `expandable_segments`, cross-stream frees) moved to `sys.profiling` item 4 to keep this lesson at 9 cards.
9. *Levers* (preview of later lessons, one line each): smaller micro-batch + accumulation, activation checkpointing,
   sequence/tensor/pipeline parallelism, ZeRO/FSDP sharding, offload, lower-precision optimizers, and for fine-tuning
   LoRA/QLoRA (frozen base weights need no grads or optimizer state; link `llm.lora`, `llm.qlora-peft`, ids from the current
   LLM plan draft; confirm).
10. *Probes*: "estimate memory to fine-tune a 7B model with Adam", "why does activation memory dominate at long context",
    "what is the 34 in 34sbh", "why OOM on the second step".

**Worked computations (verified)**
- 7B, mixed-precision Adam: $16 \times 7\times10^9 = 112$ GB of model states (140 GB with an fp32 grad buffer).
- Parameter count from the Playbook formula $N = hV + L(12h^2 + 13h) + 2h$ with $h = 4096, L = 32, V = 32000$: ≈ 6.58e9.
- GPT-3 (s = 2048, b = 1, h = 12288, a = 96, L = 96): $5as/h = 80$; per layer $sbh(34+80) \approx 2.87$ GB; all layers ≈ 275 GB.
- 7B-like (s = 4096, b = 1, h = 4096, a = 32, L = 32): without FlashAttention $5as/h = 160$ and activations ≈ 104 GB;
  with FlashAttention (≈ 34sbh per layer) ≈ 0.57 GB/layer, ≈ 18.3 GB total.
- Logits for s = 4096, b = 1, V = 128,000 in fp32: ≈ 2.1 GB (and the same again for their gradient).

**Question ideas**
- Compute: full memory budget (states + activations) for a given config; does it fit on 80 GB?
- Predict: doubling sequence length with and without FlashAttention — how does activation memory change (×2 vs up to ×4 for the quadratic term)?
- Debug: "training runs the first step and OOMs on the second" — why?
- Which is false: "mixed precision halves the memory of model states" (false) / "activation memory is proportional to micro-batch size" / ...
- Explain: why does inference of the same model need only ~2 bytes/param plus a KV cache (link `llm.kv-cache`, id to confirm)?
- Open estimate (chains lessons): 13B model, 4k context, micro-batch 1, bf16 + Adam, FlashAttention, no sharding — memory per
  GPU? Which lever do you pull first to fit on 80 GB?

**Pitfalls / verify**
- Korthikanti counts bytes assuming 16-bit activations and 1-byte masks; when writers say "elements" vs "bytes", be explicit.
- The 34sbh constant assumes a 4h GeLU MLP with dropout; recount for SwiGLU/no-dropout or say "for the GPT-style layer".
- GB vs GiB: an "80 GB" H100 actually has ≈80 GiB of HBM (nvidia-smi reports ≈81,559 MiB; torch ≈79 GiB). Usable memory is that minus the CUDA context and allocator reserve. Model sizes in GB (10^9) vs device memory in GiB is the usual trap.

**Figures**
- `memory-timeline`: stacked area of params / grads / optimizer / activations over a training step (forward, backward,
  optimizer), synthetic but derived from the formulas for a stated config; first-step vs later-step difference.
- `activation-breakdown`: stacked bar of per-layer activation terms (11sbh, 5as²b, 19sbh, 4sbh) for s = 1k, 4k, 16k.
- `memory-vs-seq`: model states (flat) vs activations (linear/quadratic) as sequence length grows, with an 80 GB line.

---

# Part D: The GPU and performance

## sys.gpu-basics — How a GPU executes a training step

Level core · prereqs — · **W1** · ~8 cards

**Problem.** "Roofline, fusion, FlashAttention, CUDA graphs and profiling all talk about SMs, HBM vs SRAM, tensor cores,
kernels and streams. What are these, and why does each matter for ML performance?" New lesson added after review; it is
the shared vocabulary for `sys.roofline`, `sys.fusion-codegen`, `sys.profiling`, `sys.mixed-precision` (tensor cores) and
`sys.fp8-training`. Keep it framework-neutral; the Scaling Book series has its own `sb.gpu-chip` (see also).

**Primary sources**
- Scaling Book Part 12 "What Is a GPU?" (SMs, 4 subpartitions each with a Tensor Core, warp scheduler, register file and
  CUDA cores; 990 TFLOP/s bf16 on Tensor Cores vs ≈66 TFLOP/s on CUDA cores for H100; up to 64 resident warps per SM;
  "Memory" — registers 256 KiB/SM, SMEM 256 kB/SM, L2 50 MB, HBM 80 GB; spec tables) — `sys_scalingbook_gpus.txt`.
- NVIDIA *GPU Performance Background* §2 "GPU Architecture Fundamentals" (SMs, L2, DRAM; Tensor Cores), §3 "GPU Execution
  Model" (threads → thread blocks → grid; latency hiding by switching threads; waves and the tail effect) —
  `sys_nvidia_gpu_perf_background.txt`; *Matrix Multiplication Background* §3 (tile and wave quantisation) — `sys_nvidia_matmul_perf.txt`.
- Ultra-Scale Playbook "A primer on GPU" and "How to improve performance with Kernels?" (memory hierarchy, coalescing,
  tiling, thread coarsening), lines ≈4111–4485 — `sys_ultrascale2025_playbook.txt`.
- PyTorch "CUDA semantics › Asynchronous execution" and "CUDA streams" — `sys_pytorch_cuda_semantics.txt`.
- Horace He 2022 "Compute" and "Overhead" — `sys_he2022_brrr.txt`.

**Subtopic map**
1. *Two halves of a GPU*: many Streaming Multiprocessors (H100: 132) that compute, attached to off-chip HBM (80 GB,
   3.35 TB/s) through an on-chip L2 (50 MB). The CPU (host) drives the GPU (device) over PCIe.
2. *Inside an SM*: 4 subpartitions, each with a Tensor Core (matrix-multiply unit), a warp scheduler, a register file and
   CUDA cores (vector ALUs). Tensor Cores supply almost all FLOPs (≈990 vs ≈66 TFLOP/s on H100), so only matmuls in
   supported dtypes/shapes reach peak; everything elementwise runs on the far slower vector units and is memory-bound anyway.
3. *Memory hierarchy*: registers (per thread) → shared memory/L1 (SMEM, per SM, ~256 kB, programmer-managed) → L2 →
   HBM. Bandwidth rises and capacity falls as you go up. Kernels are fast when they reuse data from SMEM/registers
   (tiling); this is the idea behind matmul kernels, fusion and FlashAttention.
4. *Execution model*: a **kernel** is a function launched over a grid of thread blocks; threads run in **warps** of 32 in
   lock-step (SIMT; divergent branches serialise); a block runs on one SM and its threads share SMEM; many resident warps
   let the SM hide memory latency by switching warps. **Waves**: if a kernel has fewer blocks than can run at once, or a
   partial last wave, SMs idle (tail effect, wave quantisation).
5. *Shapes matter*: Tensor Cores work on fixed tiles; dimensions that are multiples of 8 (fp16/bf16) or 64/128 avoid padding
   waste (NVIDIA §3 tile quantisation); why vocab sizes are padded.
6. *Host–device model*: the CPU enqueues kernels into **streams** and continues (asynchronous execution); operations in one
   stream run in order, different streams may overlap (compute vs communication vs copies); a synchronisation (e.g. `.item()`)
   makes the CPU wait. Each launch costs CPU time (microseconds), which is the "overhead" regime (link `sys.profiling`) and
   the motivation for CUDA graphs (link `sys.fusion-codegen`).
7. *Precision support*: which dtypes Tensor Cores accelerate (bf16/fp16, tf32, fp8 on Hopper), with fp32 accumulation.
8. *Probes*: "what is an SM / warp / kernel", "why is HBM bandwidth the bottleneck for elementwise ops", "what does
   shared memory buy", "why are GPU calls asynchronous and what forces a sync".

**Worked computations (verified from the Scaling Book tables)**
- H100 Tensor Core vs CUDA core throughput: ≈ 990 vs 66 TFLOP/s, a 15× gap → a matmul written as elementwise loops
  cannot exceed ≈ 7% of peak.
- Per-SM share of HBM bandwidth on H100: 3.35 TB/s / 132 SMs ≈ 25 GB/s per SM (writers: recompute; use only to show why
  data reuse inside SMEM matters).
- Wave quantisation: 12 blocks on an 8-SM GPU at one block per SM → 2 waves, second wave 50% utilised (NVIDIA Fig. 3).

**Question ideas**
- Predict: a GEMM with $N = 4097$ vs $N = 4096$ (tile/wave quantisation).
- Which is false: "CUDA cores provide most of an H100's bf16 FLOPs" (false) / "a warp is 32 threads" / ...
- Explain: why does `print(loss.item())` every step slow training?
- Compare: SMEM vs HBM (capacity, bandwidth, who manages it).

**Pitfalls / verify**
- Spec numbers vary by SKU (SXM vs PCIe); use the Scaling Book table values and say so.
- Keep thread-level CUDA programming out of scope beyond what later lessons need.

**Figures**
- `gpu-anatomy`: block diagram — HBM, L2, a row of SMs, one SM expanded into 4 subpartitions (Tensor Core, warp scheduler,
  registers, CUDA cores) and SMEM, with bandwidth/capacity labels.
- `memory-hierarchy`: log–log capacity vs bandwidth for registers, SMEM, L2, HBM, NVLink, PCIe (values from sources).
- `streams-async`: CPU timeline enqueuing kernels ahead of a GPU stream, with a `.item()` sync draining the queue.

---

## sys.roofline — The roofline model and arithmetic intensity

Level core · prereqs sys.gpu-basics · **W1** · ~8 cards

**Problem.** "Why does a matmul run at 70% of peak while a LayerNorm runs at 2%? Before optimising anything, how do we
know the best possible time for an operation?"

**Scope vs other plans.** `llm.arithmetic-intensity` and `llm.kv-cache` (ids from the current LLM plan draft; confirm) own
decode-time and attention-specific intensity; `sb.roofline-matmul` (Scaling Book series) covers the book's framing. This
lesson is the general model used by every training-systems lesson; numbers must agree with those lessons.

**Refreshers to include inline.** HBM vs SRAM and Tensor Cores in two sentences (full treatment in `sys.gpu-basics`).

**Primary sources**
- Scaling Book Part 1 "Where Does the Time Go?" ($T_{math}$, $T_{comms}$, lower bound max, upper bound sum), "Visualizing
  rooflines", "Matrix multiplication" (intensity of a bf16 matmul ≈ batch dimension when it is small), "Network
  communication rooflines", worked problems — `sys_scalingbook_roofline.txt`.
- Williams, Waterman & Patterson 2009 (operational intensity, the roofline plot, ceilings) — `sys_williams2009_roofline.txt`.
- NVIDIA *GPU Performance Background* §4 "Understanding Performance" (math- vs memory-limited, arithmetic intensity
  table for common ops) and §5 "DNN Operation Categories"; *Matrix Multiplication Background* §2 "Math and Memory Bounds",
  §3 tile/wave quantisation — `sys_nvidia_gpu_perf_background.txt`, `sys_nvidia_matmul_perf.txt`.
- Horace He 2022 "Bandwidth", "Reasoning about Memory-Bandwidth Costs" — `sys_he2022_brrr.txt`.
- Scaling Book Part 12 spec tables (H100) — `sys_scalingbook_gpus.txt`.

**Subtopic map**
1. *Three resources*: FLOP/s ($C$), memory bandwidth ($W_{HBM}$), interconnect bandwidth; capacity as a separate constraint.
2. *Time bounds*: $T_{math} = \text{FLOPs}/C$, $T_{mem} = \text{bytes}/W$; with perfect overlap $T \ge \max$, without $T \le$ sum.
3. *Arithmetic intensity* $I = \text{FLOPs}/\text{bytes moved}$ and the *ridge point* $I^* = C/W$: compute-bound iff $I > I^*$.
   Attainable FLOP/s $= \min(C, I \cdot W)$ — derive the roofline plot (log–log, slope-1 line meeting a flat roof).
4. *Intensity of common ops* (derive each): elementwise add in fp32 (1 FLOP per 12 bytes), ReLU in bf16 (1 per 4 bytes),
   reductions/softmax/LayerNorm (O(1) FLOPs per element) → always memory-bound; matmul $[B,D]\times[D,F]$ in bf16:
   $2BDF / 2(BD + DF + BF)$, which → $B$ when $B \ll D, F$ (so the token batch per weight load sets the regime); square
   $n\times n$ matmul: $n/3$.
5. *Consequences*: training matmuls with thousands of tokens per weight are compute-bound; small-batch decoding is
   memory-bound (link `llm.kv-cache` / `llm.arithmetic-intensity`, ids to confirm); fusion raises intensity of elementwise
   chains (`sys.fusion-codegen`); low precision moves
   the ridge (fp8 doubles $C$, the ridge goes to ≈590).
6. *Communication rooflines*: same logic with interconnect bandwidth (e.g. the DP condition $B/P > C/W$; TP condition).
7. *Beyond the simple model*: caches/SRAM reuse (tiling; vocabulary from `sys.gpu-basics`), tile and wave quantisation (dimensions that are not multiples of
   the tile size waste work; multiples of 64/128), achieved vs peak (80–85% on H100 for big matmuls), latency and launch overhead.
8. *Probes*: "is this op compute- or memory-bound on H100", "what batch size makes a matmul compute-bound", "why does
   fusion help", "draw a roofline".

**Worked computations (verified)**
- H100 ridge point (bf16): $989\text{e}12 / 3.35\text{e}12 \approx 295$ FLOPs/byte; A100: $312\text{e}12/2.0\text{e}12 = 156$.
- Square bf16 matmul intensity $n/3$: 85 ($n = 256$), 341 ($n = 1024$), 1365 ($n = 4096$).
- $[512, 8192]\times[8192, 8192]$ bf16: $I \approx 455$ → compute-bound on H100.
- Elementwise bf16 add of $10^9$ elements (6 GB moved): $T_{mem} \approx 1.79$ ms vs $T_{math} \approx 0.001$ ms.

**Question ideas**
- Compute: intensity of a given op and whether it is compute-bound on H100.
- Predict: halving precision (bf16 → fp8) for a memory-bound op vs a compute-bound op.
- Figure MCQ: points on a roofline plot; which is LayerNorm, which a large matmul, which a decode-time matvec?
- Which is false: "a $1\times4096$ by $4096\times4096$ matvec is compute-bound on H100" (false) / ...

**Pitfalls / verify**
- Count bytes for inputs *and* outputs; say whether weights are re-read.
- Peak FLOP/s for "dense" vs "with sparsity" differ by 2× in vendor sheets; use dense.

**Figures**
- `roofline-h100`: log–log roofline for H100 bf16 and fp8 with labelled points (elementwise add, LayerNorm, softmax,
  matmul at B = 64, 512, 4096) computed from the formulas.
- `matmul-intensity`: intensity vs batch dimension B for D = F = 8192 with the ridge line.

---

## sys.flops-mfu — Counting FLOPs, MFU/HFU and estimating training time

Level core · prereqs sys.roofline · **W1** · ~8 cards

**Problem.** "Your 7B run does 3,000 tokens/s/GPU on H100s. Is that good? How long will a 70B model on 15T tokens take?"

**Primary sources**
- Scaling Book Part 4 "Forward and reverse FLOPs" (2NPM forward, 4NPM backward → 6 × params × tokens), "MLPs",
  "Attention" (12BTSNH train FLOPs for dot-product attention), "General rule of thumb", "Fractional cost of attention with
  context length" (ratio $T/(8D)$ under its assumptions) — `sys_scalingbook_transformers.txt`.
- Korthikanti et al. 2022 §6.3 (MFU and HFU definitions following PaLM; A100 peak 312 TFLOP/s; MFU 56.3% for 1T), App. A
  ($72BLsh^2(1 + s/6h + v/12hL)$ model FLOPs; hardware/model ratio ≈ $1 + s/6h$ with selective recompute) —
  `sys_korthikanti2022_seqpar_recompute.txt`.
- Narayanan et al. 2021 §5.1 (FLOPs with full recompute $96BSlh^2(1 + s/6h + V/16lh)$; end-to-end time $\approx 8TP/(nX)$;
  GPT-3 175B on 1024 A100s at 140 TFLOP/s → 34 days) — `sys_narayanan2021_megatron_ptd.txt`.
- PaLM §4 / Table 3 (MFU definition; 45.7% / 46.2% without / with attention) — shared `llm_chowdhery2022_palm.txt`.
- Overlap: `llm.params-flops` / `llm.scaling-laws` (ids to confirm) use 6ND for compute budgets; this lesson owns MFU/HFU and
  training-time estimation. See also `sb.*` transformer-math lesson.

**Subtopic map**
1. *FLOPs of a matmul*: $[M,K]\times[K,N]$ costs $2MKN$ (multiply + add).
2. *Forward/backward*: for $Y = XW$ the backward needs $\partial L/\partial X$ and $\partial L/\partial W$, each $2MKN$ → train
   = 3× forward. Per token, each parameter in a matmul contributes 2 FLOPs forward and 4 backward → $6N$ per token.
3. *What 6N leaves out*: attention score/value matmuls ($12\,L\,h\,s$ per token for training without causal savings;
   halved by causal kernels), embeddings lookups (no FLOPs) vs LM head (counted), elementwise ops (negligible FLOPs, not
   negligible time). When attention matters ($s$ comparable to $8h$ under the Scaling Book's assumptions).
4. *MFU* = achieved model FLOP/s / peak: $\text{MFU} = \frac{\text{tokens/s} \times 6N}{n_{GPU} \cdot C}$ (plus attention term
   if stated). *HFU* counts FLOPs actually executed, including recomputation (full recompute adds a forward: 8N).
5. *Training-time estimate*: $T = 6ND/(n\,C\,\text{MFU})$; Narayanan's $8TP/(nX)$ with full recompute; sanity-check against
   published runs.
6. *Typical MFUs and why not 100%*: comm exposure, bubbles, memory-bound ops, kernel efficiency, stragglers; 35–55% is
   common for large dense runs (PaLM: 45.7% without attention FLOPs, 46.2% with — compare 45.7% against the 6N formula;
   Korthikanti 56.3%); MoE MFU uses active params.
7. *Inference contrast*: 2N per token forward; decode is memory-bound so MFU is the wrong metric (link `llm.kv-cache`, id to confirm).
8. *Probes*: "derive 6N", "compute MFU from tokens/s", "estimate days to train X", "MFU vs HFU".

**Worked computations (verified)**
- 7B at 3,000 tokens/s/GPU on H100 (989 TFLOP/s): MFU $= 6 \cdot 7\text{e}9 \cdot 3000 / 989\text{e}12 \approx 12.7\%$; 40% MFU needs ≈ 9,400 tokens/s/GPU.
- 70B on 2,048 H100s at 400 tokens/s/GPU: MFU ≈ 17%.
- 70B on 15T tokens: $6ND = 6.3\times10^{24}$ FLOPs; on 16,384 H100s at 40% MFU ≈ 11.2 days.
- GPT-3: $6 \cdot 175\text{e}9 \cdot 300\text{e}9 = 3.15\times10^{23}$; Narayanan's $8TP/(nX)$ with $n = 1024$, $X = 140$ TFLOP/s → ≈ 33.9 days (paper: 34).
- Attention share for a 7B-like model ($L = 32$, $h = 4096$, $s = 4096$): $12Lhs \approx 6.4\times10^9$ vs $6N = 4.2\times10^{10}$ FLOPs/token → 15.3% relative to 6N, i.e. 13.3% of total training FLOPs (say which).

**Question ideas**
- Compute: MFU from throughput; training days from MFU.
- Derive: why the backward is ≈ 2× the forward.
- Predict: turning on full activation recomputation — what happens to MFU and HFU at fixed step time? (HFU up, MFU same or down.)
- Which is false: "MFU counts recomputed FLOPs" (false) / ...
- Open estimate (chains lessons): 13B model, 4k context, 64 H100s, ZeRO-3 + full recompute, 1M tokens per step — step time at
  40% MFU? HFU? Is the ZeRO-3 communication hidden (per-GPU tokens vs ≈ 2475)?

**Pitfalls / verify**
- Different sources count attention differently (causal halving; Kaplan vs Megatron conventions); state the convention.
- Use dense peak (not sparse) and the right precision's peak.

**Figures**
- `flops-breakdown`: stacked bar of per-token training FLOPs (MLP, attention projections, attention scores, LM head) vs
  sequence length for a 7B-like config.
- `mfu-gauge`: step-time decomposition bar computed from a stated model (compute at peak from 6N·tokens/C, a stated kernel
  efficiency, exposed comm from the α–β/last-bucket model, bubble from $(p-1)/(m+p-1)$) for one concrete configuration.

---

## sys.profiling — Profiling training: compute-, memory- and overhead-bound

Level intermediate · prereqs sys.roofline, sys.gpu-basics · W2 · ~9 cards

**Problem.** "Training is slower than the FLOP estimate says it should be. Where does the time actually go, and how do
you find out?"

**Primary sources**
- Horace He 2022 (compute / bandwidth / overhead; diagnosing which regime by scaling the batch; operator fusion; Python
  and dispatcher overhead; async execution) — `sys_he2022_brrr.txt`.
- PyTorch Profiler recipe (record_function, key_averages, CUDA time, memory profiling, Chrome trace export, schedule
  wait/warmup/active) — `sys_pytorch_profiler_recipe.txt`; PyTorch Performance Tuning Guide (DataLoader workers and
  pinned memory, avoiding CPU–GPU syncs, `zero_grad(set_to_none)`, fusion, channels_last, cudnn.benchmark, DDP tips) —
  `sys_pytorch_tuning_guide.txt`; CUDA semantics "Asynchronous execution" — `sys_pytorch_cuda_semantics.txt`.
- Ultra-Scale Playbook A1 "Distributed Training Profiling" (reading traces, overlap of comm and compute) — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Part 9 (JAX profiler, Trace Viewer, "How to read an XLA op", Graph Viewer, memory profile) —
  `sys_scalingbook_profiling.txt`; JAX profiling docs (Perfetto, XProf, Nsight) — `sys_jax_profiling.txt`.

**Subtopic map**
1. *Three regimes* (He): compute-bound (good), memory-bandwidth-bound (fuse, lower precision), overhead-bound (Python,
   framework dispatch, kernel launches; fix with bigger work per kernel, compile, CUDA graphs).
2. *Asynchronous GPU execution*: the CPU enqueues kernels and runs ahead; anything that needs a GPU value on the host
   (`.item()`, `.cpu()`, printing a tensor, data-dependent shapes like `nonzero` or boolean masking, some `if tensor:`)
   forces a synchronisation and drains the queue.
3. *Measuring correctly*: wall-clock timers without `torch.cuda.synchronize()` measure launch time; use CUDA events; warm
   up (allocator, cudnn autotuning, compilation); report steady-state medians; profile a window via `schedule`.
4. *Tools and what each shows*: PyTorch profiler (op-level CPU/CUDA time, stacks, memory, traces), Nsight Systems
   (timeline across CPU threads, CUDA streams, NCCL, NVTX ranges), Nsight Compute (one kernel's achieved FLOP/s,
   bandwidth, occupancy), JAX/XProf (XLA ops, HLO), memory snapshots. Keep Nsight descriptions generic (no Nsight docs cached).
   *Memory profiling and the allocator* (moved from `sys.memory-anatomy`): allocated vs reserved memory, why `nvidia-smi`
   shows more than `memory_allocated()`, fragmentation and `expandable_segments`, cross-stream frees and FSDP's rate limiter
   (Zhao et al. 2023 §3.4.1–3.4.2, `sys_zhao2023_pytorch_fsdp.txt`), reading a memory snapshot.
5. *Reading a trace*: gaps between kernels (CPU-bound/overhead or data loader), long tail of tiny kernels, NCCL kernels not
   overlapping compute, host-to-device copies on the critical path, a single straggler rank, idle time at the optimizer step.
6. *Quick diagnostic experiments*: double the batch (time unchanged → overhead-bound); compare achieved FLOP/s with peak and
   achieved bytes/s with HBM bandwidth per kernel; remove the model and time the data loader alone; run on synthetic data.
7. *Common bottlenecks → fixes*: data loading (workers, pinned memory, non_blocking copies, prefetch); host syncs in the
   training loop (logging `.item()` every step); small kernels (fuse, compile); exposed communication (bucketing,
   prefetch, overlap); memory-bound ops (fusion, FlashAttention); recompilation; poor shapes (multiples of 8/64).
8. *Distributed profiling*: per-rank traces, stragglers, comm/compute overlap fraction; MFU as the top-level metric.
9. *Probes*: "your GPU utilisation is 30% — what do you check, in order", "why is timing without synchronize wrong",
   "how do you know you're overhead-bound".

**Worked computations**
- Elementwise chain example (He): `x.cos().cos()` reads and writes $x$ twice unfused (4 passes over memory) vs once fused
  (2 passes) → ≈ 2× for a memory-bound op (writers: compute bytes for a $10^8$-element fp32 tensor on H100).
- Overhead example: if one step launches 2,000 kernels at ≈ 5–10 µs CPU cost each, launch overhead alone is 10–20 ms
  (writers: present as an illustrative calculation and do not attribute exact per-launch costs without a source).

**Question ideas**
- Debug: a trace shows 40% of each step idle with the GPU waiting on `aten::item` — explanation and fix.
- Predict: doubling batch size makes step time go from 50 ms to 52 ms; which regime were you in?
- Which is false: "`time.time()` around a CUDA op measures its GPU runtime" (false) / ...
- Order: list the first five things you check when MFU is 15%.

**Pitfalls / verify**
- "GPU utilisation" in nvidia-smi is the fraction of time any kernel runs, not FLOP efficiency.
- The profiler adds overhead; profile a few steps with a schedule.

**Figures**
- `trace-sketch`: stylised timeline (CPU thread, compute stream, NCCL stream) for three cases: overhead-bound (gaps),
  sync stall, exposed comm.
- `regime-diagnosis`: step time vs batch size curve showing the flat overhead-bound region then the linear compute-bound region.

---

# Part E: Memory-saving techniques

## sys.activation-checkpointing — Activation checkpointing (rematerialisation)

Level intermediate · prereqs sys.memory-anatomy, sys.flops-mfu · W2 · ~7 cards

**Problem.** "Activations for a long-context run need 100+ GB. Can we trade compute for memory, how much compute, and
which activations should we drop?"

**Primary sources**
- Chen et al. 2016 §3–4 (segmenting an $n$-layer chain, storing only segment boundaries; $O(\sqrt n)$ memory for one
  extra forward; recursive $O(\log n)$ memory at $O(n \log n)$ compute) — `sys_chen2016_sublinear_memory.txt`.
- Korthikanti et al. 2022 §5 (selective activation recomputation: recompute only the attention-score part; GPT-3 saves
  ≈70% of activation memory for ≈2.7% extra FLOPs), §6.3 MFU vs HFU, Appendix A FLOPs — `sys_korthikanti2022_seqpar_recompute.txt`.
- Ultra-Scale Playbook "Activation recomputation" (full vs selective; 30–40% time overhead of full) — `sys_ultrascale2025_playbook.txt`.
- PyTorch `torch.utils.checkpoint` (use_reentrant=False recommended, RNG state preservation, `checkpoint_sequential`) —
  `sys_pytorch_checkpoint.txt`; JAX `jax.checkpoint` (residuals; policies such as `dots_with_no_batch_dims_saveable`;
  offload policies) — `sys_jax_checkpoint.txt`.
- Scaling Book Part 4 "Gradient checkpointing" — `sys_scalingbook_transformers.txt`.

**Subtopic map**
1. *The trade.* Forward normally stores everything the backward needs. Checkpointing stores a subset ("checkpoints")
   and recomputes the rest during backward by rerunning the forward from the nearest checkpoint.
2. *Chen's √n schedule* (recap from `fund.backprop` item 7 and its figure `fund.backprop/checkpointing`; one card, then move on). Chain of $n$ equal layers split into $k$ segments: memory $\approx n/k$ (one segment's internals
   during its backward) $+ k$ (boundaries); minimised at $k = \sqrt n$ giving $2\sqrt n$; cost is one extra forward.
   Derive. Recursive application gives $O(\log n)$ memory.
3. *Compute overhead.* With forward:backward ≈ 1:2, one extra forward adds ≈ 1/3 of the step's FLOPs (≈ 33%); hence
   HFU > MFU (hardware FLOPs include recomputation; with full recompute, 8N vs 6N FLOPs per token, so MFU = 0.75 HFU).
4. *Full (per-layer) checkpointing in transformers*: store only each layer's input ($2sbh$ bytes per layer) → total
   $2sbhL$ plus one layer's full activations during its backward.
5. *Selective recomputation* (Korthikanti §5): the $5as^2b$ attention terms are large but cheap to recompute (few FLOPs
   per byte); recompute only those. Memory becomes $34sbh$ per layer; extra FLOPs ≈ the attention-score matmuls. With
   FlashAttention this recomputation is already built into the kernel's backward.
6. *Policy-based checkpointing* (JAX policies; PyTorch selective AC / `context_fn`): save matmul outputs, recompute
   elementwise ops — "save what is expensive to recompute and cheap to store".
7. *Offloading* as the alternative trade (activations to CPU over PCIe; bandwidth arithmetic; when it pays off).
8. *Correctness details*: RNG state must be restored so dropout masks match; non-reentrant vs reentrant implementations;
   interaction with `torch.no_grad`, hooks and DDP (reentrant checkpoint + DDP unused-parameter issues).
9. *When to use*: as the last resort after micro-batch reduction, or when it unlocks a larger micro-batch or less model
   parallelism that more than pays back the ~30% cost. Interaction with pipeline parallelism (first stage holds $p$
   micro-batches) and with sequence parallelism (next part).
10. *Probes*: "derive the √n result", "what does checkpointing cost", "why selective recompute", "how do you make
    dropout consistent under recompute".

**Worked computations (verified)**
- $n = 100$ layers, $k = 10$ segments: memory ∝ $100/10 + 10 = 20$ units instead of 100.
- $n = 32$: $k \approx \sqrt{32} \approx 5.7$; with $k = 6$, $32/6 + 6 \approx 11.3$ units.
- GPT-3 config: full activations ≈ 275 GB; selective (34sbh per layer) ≈ 82 GB, i.e. 70% saved (= 80/114), matching
  Korthikanti's 70%; full per-layer checkpointing ($2sbhL$) ≈ 4.8 GB plus one layer's working set.
- 7B-like at s = 4096: $2sbhL \approx 1.07$ GB of checkpoints.
- MFU vs HFU with full recompute: $6N/8N = 0.75$.

**Question ideas**
- Derive: optimal number of checkpoints for an $n$-layer chain and the resulting memory.
- Compute: HFU is 52% with full recomputation; what is MFU?
- Predict: enabling checkpointing lets you double the micro-batch size; when does throughput go up despite recompute?
- Debug: loss differs between checkpointed and non-checkpointed runs with dropout on; cause?
- Which is false: "checkpointing changes the gradients" (false, up to floating-point/RNG handling) / ...

**Pitfalls / verify**
- "Gradient checkpointing" (common name) ≠ saving model checkpoints to disk.
- The 30–40% overhead figure for full recompute is from the Playbook/Korthikanti; the 33% FLOPs number is the idealised one.

**Figures**
- `checkpoint-schedule`: timeline of forward/backward over 8 layers showing stored vs recomputed activations for no
  checkpointing, every-layer, and √n segments.
- `memory-compute-tradeoff`: activation memory vs extra compute for none / selective / full / √n on the GPT-3 config (adds
  selective recompute and real numbers; does not duplicate `fund.backprop/checkpointing`).

---

## sys.gradient-accumulation — Gradient accumulation: equivalence and where it breaks

Level core · prereqs sys.memory-anatomy · W2 · ~7 cards

**Problem.** "I need a 512-sequence batch for optimisation reasons but only 8 fit in memory." Accumulating gradients over
micro-batches gives the large-batch gradient — exactly, in some cases, and not in others. (Batch-size *choice* is in the
tuning lessons `sys.tuning-*`; link there.)

**Overlap with fund.** `fund.training-loop` introduces accumulation and the HF normalisation bug; this lesson owns the
exactness conditions and the DDP/FSDP/PP/mixed-precision interactions.

**Primary sources**
- Ultra-Scale Playbook "Gradient accumulation" and "Revisit global batch size" (gbs = mbs × grad_acc × dp), lines ≈1061–1148, 1254 — `sys_ultrascale2025_playbook.txt`.
- Hugging Face blog "Fixing Gradient Accumulation" (token-count normalisation bug, Oct 2024) — `sys_hf2024_gradient_accumulation_fix.txt`.
- Li et al. 2020 §3.2.4 "Gradient Accumulation" (`no_sync` to skip all-reduce on intermediate micro-batches) — `sys_li2020_pytorch_ddp.txt`;
  Zhao et al. 2023 §3.3.4 (FSDP gradient accumulation with and without communication) — `sys_zhao2023_pytorch_fsdp.txt`.
- PyTorch AMP examples "Gradient accumulation" (scale, accumulate, unscale and step once) — `sys_pytorch_amp_examples.txt`.
- Goyal et al. 2017 §2.3 / §3 (BN statistics are per-worker batch; loss normalisation pitfalls) — shared `papers/goyal2017_large_minibatch.txt`.

**Subtopic map**
1. *Why it works*: the gradient of a sum is the sum of gradients; gradients accumulate in `.grad` by default in
   PyTorch. Derive: with $k$ micro-batches of equal size and a mean loss, divide each micro-batch loss by $k$.
2. *The loop*: zero grads once, $k$ forward/backward passes, optimizer step, zero; memory = one micro-batch of
   activations; compute is the same as the large batch; wall-clock is $k$ sequential passes, so time per example does
   not fall (accumulation buys memory, not throughput).
3. *Exact equivalence conditions*: per-sample losses, no batch-coupled layers, consistent normalisation, same RNG.
4. *Where it breaks*: (a) **BatchNorm** statistics are computed per micro-batch (and running stats updated $k$ times);
   (b) **token-level normalisation**: averaging per-micro-batch means when micro-batches contain different numbers of
   non-padding tokens is not the global per-token mean (derive; the HF 2024 bug); fix = sum losses and divide by the
   total token count of the whole accumulated batch (known in advance or all-reduced); (c) losses with in-batch
   negatives (contrastive) — a smaller micro-batch means fewer negatives; (d) anything stateful per forward (MoE load
   balancing statistics, dropout-RNG streams).
5. *With mixed precision*: scale every micro-batch loss, unscale and clip once before the single step.
6. *With DDP*: wrap the first $k-1$ backward passes in `no_sync()` so the all-reduce happens only once (k× less comm);
   with ZeRO-2/FSDP, accumulating unsharded gradients costs memory vs reduce-scattering every micro-batch (more comm) — trade-off.
7. *With pipeline parallelism*: micro-batches *are* gradient accumulation; global batch = micro-batch × $m$ × dp.
8. *Accumulation dtype and packing*: with bf16 gradients, accumulate in an fp32 buffer (the Playbook's 20-bytes/param
   variant; swamping from `sys.fp-error`); with sequence packing and document masks, normalise by the real (unmasked) token count.
9. *LR and schedules*: the optimizer sees one step per accumulated batch; schedules count optimizer steps, not micro-batches.
10. *Probes*: "is gradient accumulation exactly equivalent to a large batch", "what goes wrong with BatchNorm", "how do
   you normalise the loss for variable-length sequences", "why use no_sync".

**Worked computations (verified)**
- Two micro-batches with 10 and 30 tokens and summed losses 5 and 9: correct per-token mean $= 14/40 = 0.35$; the mean
  of per-micro-batch means $= (0.5 + 0.3)/2 = 0.40$ (a 14% error in the loss value). Per-token weights: correct $1/40 = 0.025$ for
  every token; buggy $1/20 = 0.05$ for tokens of the short micro-batch (2× overweight) and $1/60 \approx 0.017$ for the long one
  (0.67×). The `token-weighting` figure shows these weights.
- Global batch $= 4$ (micro) $\times 8$ (accumulation) $\times 64$ (DP) $= 2048$ sequences.

**Question ideas**
- Compute: the effective loss weighting under the buggy normalisation for given token counts.
- Predict: a ResNet with BatchNorm trained with micro-batch 4 × 64 accumulation vs true batch 256.
- Code: where does `loss / k` go and why; what if you forget it with Adam vs with SGD (Adam is approximately scale-invariant, SGD's LR is effectively ×k)?
- DDP: what does `no_sync` save, and what memory does it cost under FSDP?

**Pitfalls / verify**
- Forgetting to divide by $k$ is harmless for Adam only approximately (ε and clipping thresholds still see the scale).
- Don't conflate this with batch-size selection; link to the tuning lessons.

**Figures**
- `accumulation-loop`: timeline of k micro-batch forward/backward passes, one all-reduce (with no_sync) and one step, vs naive per-micro-batch all-reduce.
- `token-weighting`: bar chart of per-token weights under correct vs mean-of-means normalisation for micro-batches of unequal length.

---

# Part F: Communication

## sys.collectives — Communication primitives and the ring all-reduce

Level core · prereqs — · **W1** · ~9 cards

**Problem.** "Every parallelism scheme is just compute plus a few collectives. What exactly does each one do to the data,
how many bytes must each rank send, how does the ring all-reduce achieve that minimum, and which identities let us swap
one collective for another?" (The ring derivation moved here from `sys.collective-algorithms` because W1 lessons use it.)

**Primary sources**
- NCCL user guide "Collective Operations" (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather,
  Scatter; buffer sizes) — `sys_nvidia_nccl_collectives.txt`.
- nccl-tests `PERFORMANCE.md` (per-collective lower bounds: $2(n-1)/n \cdot S/B$ for AllReduce, $(n-1)/n \cdot S/B$ for
  AllGather/ReduceScatter, $S/B$ for Broadcast/Reduce; independent of algorithm) — `sys_nccl_tests_performance.txt`.
- Patarasuk & Yuan 2009 §2–3 (tight lower bound on all-reduce data per process; ring algorithm achieving it) —
  `sys_patarasuk2009_bandwidth_optimal_allreduce.txt`.
- Ultra-Scale Playbook Appendix A0 "Parallel Programming Crash Course" incl. "Ring AllReduce" (step-by-step phases),
  lines ≈5149–5503 — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Part 3 "Partitioning Notation and Collective Operations", "A Deeper Dive into TPU Communication
  Primitives" (AllToAll; ReduceScatter as the transpose of AllGather) — `sys_scalingbook_sharding.txt`.
- ZeRO §7.1 (all-reduce = reduce-scatter + all-gather, each moving $N$ elements per rank) — `sys_rajbhandari2020_zero.txt`.
- See also `sb.collective-costs` (Scaling Book series; numbers must agree).

**Refreshers to include inline.** The Jacobian of "concatenate shards" and of "sum then split" (for item 3).

**Subtopic map**
1. *Setting.* $P$ ranks, each with a buffer; point-to-point send/recv vs collectives; process groups/communicators.
2. *Each primitive defined by input/output shapes* (reader can draw before/after for 4 ranks): broadcast, reduce,
   all-reduce, gather / all-gather, scatter, reduce-scatter, all-to-all (a distributed transpose), barrier.
3. *Identities*: all-reduce = reduce-scatter + all-gather; all-gather is the backward (transpose) of reduce-scatter and vice
   versa (derive from the Jacobians); broadcast's backward is reduce; identity-forward ↔ all-reduce-backward (Megatron's
   $f$/$g$ pair, used in `sys.tensor-parallel`).
4. *Per-rank traffic lower bounds* (derive by counting what must leave/arrive at each rank): all-gather and reduce-scatter
   $(P-1)/P \cdot S$; all-reduce $2(P-1)/P \cdot S$; all-to-all $(P-1)/P \cdot S$ spread over many distinct pairs.
5. *Ring all-reduce, derived.* Naive root-based all-reduce makes the root carry $(P-1)S$. Ring: split the buffer into $P$
   chunks; reduce-scatter phase of $P-1$ steps, each rank sending one chunk ($S/P$) to its neighbour while receiving and
   adding another; all-gather phase of $P-1$ steps. Per-rank bytes $2(P-1)S/P$ = the lower bound → bandwidth-optimal
   (Patarasuk & Yuan). Time with link bandwidth $W$: $T \approx 2\frac{P-1}{P}\frac{S}{W}$, essentially independent of $P$;
   the latency term $2(P-1)\alpha$ grows with $P$ (developed in `sys.collective-algorithms`).
6. *Which bandwidth to plug in* (a rule every later lesson uses):
   - inside an H100 node: per-GPU NVLink egress, 450 GB/s (≈ 370 GB/s achieved);
   - across nodes, for all-reduce / reduce-scatter / all-gather: NCCL runs one ring (or tree) per NIC, or reduces inside
     the node first, so all 8 NICs carry traffic at once and the effective rate is the **node** egress, 400 GB/s
     (Scaling Book: $T_{AG/RS} \approx \text{bytes}/W_{\text{node egress}}$, all-reduce twice that without SHARP);
   - 50 GB/s per GPU is correct only when each GPU's whole buffer must leave through its own NIC: cross-node
     **all-to-all** (Scaling Book: ≈ 50 GB/s effective), or a TP group with one GPU per node. The "flat ring at 50 GB/s per
     GPU" model for cross-node all-reduce is the classic naive estimate and is ≈ 8× too pessimistic.
7. *Where each appears*: DDP grad sync (all-reduce), ZeRO/FSDP (AG params, RS grads), TP (AR or AG/RS of activations),
   SP (AG/RS), MoE expert parallel (all-to-all; owned by `llm.moe-systems`, id from the current LLM plan draft; confirm), PP
   (send/recv), init (broadcast), metrics (reduce), SyncBN (all-reduce of statistics).
8. *Synchronous semantics*: all ranks must call the same collectives in the same order with matching sizes; mismatches
   hang until the NCCL timeout; one slow rank stalls everyone (stragglers); async work handles and streams.
9. *Probes*: "derive ring all-reduce cost", "all-reduce vs reduce-scatter", "express all-reduce in other collectives",
   "which collective does FSDP use in forward/backward", "which bandwidth do you use across nodes and why".

**Worked computations (verified)**
- 4 ranks holding [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]: all-reduce → [28,32,36,40] everywhere; reduce-scatter
  gives rank $i$ the $i$-th entry; all-gather of those shards reconstructs the all-reduce result.
- Per-rank bytes for a 14 GB buffer, $P = 8$: reduce-scatter 12.25 GB, all-gather 12.25 GB, all-reduce 24.5 GB.
- 7B bf16 gradients (14 GB; assumes bf16 grads, e.g. bf16 params or a bf16 comm hook) all-reduced inside one node at
  450 GB/s: $2\cdot\tfrac78\cdot14\text{e}9/450\text{e}9 \approx 54$ ms (≈ 66 ms at 370 GB/s).
- Same 14 GB across 64 GPUs (8 nodes), node-egress model: $2\cdot\tfrac{63}{64}\cdot14\text{e}9/400\text{e}9 \approx 69$ ms.
  The naive flat-ring-per-GPU estimate at 50 GB/s gives 0.55 s — use it as a distractor and explain why it is wrong.
- 1 GB fp32 across 4 GPUs at 100 GB/s: 15 ms.

**Question ideas**
- Figure MCQ: given before/after buffers on 4 ranks, name the collective.
- Derive: per-rank bytes of ring all-reduce and why it matches the lower bound.
- Compute: cross-node all-reduce time for a given buffer; distractors include the 50 GB/s/GPU flat-ring answer.
- Derive: why is the backward of an all-gather a reduce-scatter?
- Debug: training hangs at step 1 on rank 3 only — likely cause (rank-dependent control flow → mismatched collectives).

**Pitfalls / verify**
- NCCL counts AllGather/ReduceScatter sizes per rank (`sendcount`/`recvcount`); nccl-tests' $S$ is the full array.
- "Reduce" means sum unless stated; averages are sum then divide (or `ReduceOp.AVG`).
- NVLink "900 GB/s" is bidirectional (450 per direction); IB "400G" is gigabits.

**Figures**
- `collectives-grid`: 4-rank before/after diagrams for broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all.
- `allreduce-decomposition`: all-reduce as reduce-scatter followed by all-gather on 4 ranks.
- `ring-allreduce`: 4-rank ring, chunk ownership after each of the 3 reduce-scatter and 3 all-gather steps.

---

## sys.collective-algorithms — Latency, trees, hierarchy and interconnects

Level intermediate · prereqs sys.collectives · W2 · ~7 cards

**Problem.** "The ring is bandwidth-optimal, so why does NCCL also use trees? Why does all-reduce cost about the same
across 8 or 1024 GPUs while all-to-all falls off a cliff at the node boundary? How do we measure collective performance?"

**Primary sources**
- NVIDIA blog "Massively Scale Your Deep Learning Training with NCCL 2.4" (2019): double binary trees — full bandwidth with
  logarithmic latency; hierarchical/2D rings — `sys_nvidia2019_nccl_double_binary_tree.txt`.
- Patarasuk & Yuan 2009 (ring vs butterfly/recursive-doubling algorithms; contention) — `sys_patarasuk2009_bandwidth_optimal_allreduce.txt`.
- nccl-tests `PERFORMANCE.md` (algbw vs busbw) — `sys_nccl_tests_performance.txt`.
- Scaling Book Part 12 "Networking" and "How Do Collectives Work on GPUs?" (intra-node ring; tree reduction with
  $\log N$ hops; empirical ≈370 GB/s; SHARP in theory halves all-reduce, ≈30% in practice; cross-node AG/RS ≈ bytes/400 GB/s
  via per-level ring analysis 514 / 413 / 17.1 TB/s; cross-node all-to-all ≈ 50 GB/s effective) — `sys_scalingbook_gpus.txt`.
- Ultra-Scale Playbook A3 "Math for Compute/Communication Overlap" — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Parts 2–3 (TPU ICI torus, bidirectional rings) — `sys_scalingbook_tpus.txt`, `sys_scalingbook_sharding.txt`.

**Subtopic map**
1. *α–β model*: $\alpha + n\beta$ per message; latency- vs bandwidth-bound; crossover size $\alpha/\beta$; ring's
   $2(P-1)\alpha$ latency term (recap the bandwidth term from `sys.collectives`).
2. *Trees*: reduce up a tree and broadcast down: $O(\log P)$ latency; a single tree wastes half the links (leaves only
   send), so NCCL uses two complementary binary trees each carrying half the data → full bandwidth with log latency
   (NVIDIA 2019). Recursive halving/doubling (butterfly) as the MPI-style alternative and its contention problems on
   real topologies (Patarasuk & Yuan). NCCL picks ring vs tree by message size and scale.
3. *Hierarchical collectives*: reduce-scatter inside the node over NVLink, all-reduce across nodes on $1/8$ of the data
   (each GPU's shard over its own NIC, all NICs in parallel), all-gather inside the node. Show it reproduces the
   node-egress rule from `sys.collectives`. Per-level ring analysis (Scaling Book): effective bandwidth
   $\min_i D_i W_i/(D_i - 1)$ = 413 GB/s for the reference SuperPod.
4. *Why all-to-all is different*: not reducible hierarchically; every GPU sends distinct data to every other GPU, so across
   nodes each GPU is limited by its own NIC (≈ 50 GB/s): $T \approx B/(M\cdot W_{\text{node}})$ for $M$ nodes. This is why
   expert parallelism is kept inside a node or limited to few nodes (link `llm.moe-systems`, id to confirm).
5. *In-network reduction (SHARP/NVLS)*: switches do the reduction; all-reduce cost $\approx \text{bytes}/W$ instead of
   $2\cdot\text{bytes}/W$ in theory; ≈30% gain in practice (Scaling Book).
6. *Interconnect hierarchy*: HBM (3.35 TB/s) ≫ NVLink (450 GB/s/GPU/direction) ≫ node egress over IB (400 GB/s per node,
   50 GB/s per NIC) ≫ PCIe/Ethernet; TPU ICI torus vs GPU switched fat tree. Achieved vs peak (large messages needed).
7. *Measuring*: algbw $= S/t$; busbw multiplies by the collective's factor ($2(n-1)/n$ for all-reduce) to compare with
   link bandwidth independent of $P$ (moved here from `sys.collectives`).
8. *Overlap*: communication runs on separate streams concurrently with compute, but NCCL kernels occupy SMs and contend for
   HBM bandwidth, so overlap is never entirely free.
9. *Probes*: "why trees if ring is optimal", "estimate a cross-node all-reduce", "why is EP all-to-all expensive across
   nodes", "what is busbw".

**Worked computations (verified)**
- Latency term: $P = 1024$, $\alpha = 5\,\mu$s: ring $2(P-1)\alpha \approx 10.2$ ms vs tree $2\log_2 P\,\alpha = 0.1$ ms.
- Hierarchical all-reduce of 2.6 GB (1.3B bf16 grads) over 8 nodes × 8 GPUs, phases run sequentially: intra-node RS at
  450 GB/s ≈ 5.1 ms + inter-node ring all-reduce of the 0.325 GB shard per GPU at 50 GB/s ≈ 11.4 ms + intra-node AG ≈ 5.1 ms
  → ≈ 21.5 ms; the pipelined node-egress model gives $2\cdot\tfrac{63}{64}\cdot2.6\text{e}9/400\text{e}9 \approx 12.8$ ms.
- Scaling Book per-level bandwidths: node $450\cdot8/7 = 514$ GB/s, leaf $400\cdot32/31 = 413$ GB/s, spine 17.1 TB/s.

**Question ideas**
- Compare: when does a tree beat a ring?
- Compute: hierarchical vs node-egress estimate for a given buffer.
- Predict: an all-to-all of $B$ bytes inside one H100 node vs spanning 2 nodes: Scaling Book gives $B/(8\cdot450\text{e}9)$ vs
  $B/(2\cdot400\text{e}9)$, a more than 4× degradation — explain why the NICs, not NVLink, become the limit.
- Which is false: "SHARP doubles achieved all-reduce bandwidth in practice" (false; ≈30%) / ...

**Pitfalls / verify**
- Hierarchical numbers depend on topology; label as idealised.
- Double-binary-tree details beyond the blog's level are not in the cache; keep claims at that level.

**Figures**
- `alpha-beta`: time vs message size (log–log) for ring and double-binary tree with α, β for NVLink and IB, showing the crossover.
- `hierarchical-allreduce`: 2 nodes × 4 GPUs: intra-node RS, inter-node AR on shards (one ring per NIC), intra-node AG.
- `interconnect-ladder`: bar chart (log scale) of HBM, NVLink, node egress, per-NIC, PCIe bandwidths, labelled.

---

# Part G: Parallelism and scale

## sys.ddp — Distributed Data Parallel (DDP)

Level core · prereqs sys.collectives, sys.memory-anatomy, sys.roofline · **W1** · ~9 cards

**Problem.** "Replicate the model on $P$ GPUs, give each a different slice of the batch, and keep the replicas identical.
How does DDP make the gradient sync almost free, and when does it stop being free?"

**Primary sources**
- Li et al. 2020 §2.2–2.3 (data parallelism, AllReduce), §3.2.1 gradient bucketing (small all-reduces are slow; Fig. 2
  bucket-size sweep), §3.2.2 overlap of computation and communication, §3.2.3 Algorithm 1 (autograd hooks; unused
  parameters; bucket order = reverse of `model.parameters()`), §3.2.4 gradient accumulation (`no_sync`), §5 evaluation
  (default bucket 25 MB) — `sys_li2020_pytorch_ddp.txt` (https://arxiv.org/abs/2006.15704).
- PyTorch note "Distributed Data Parallel" (construction-time broadcast of state_dict, buckets in roughly reverse order,
  autograd hooks, `find_unused_parameters`, buffer broadcast, DDPOptimizer with torch.compile) — `sys_pytorch_ddp_note.txt`.
- Ultra-Scale Playbook "Data Parallelism" (three optimisations: overlap, bucketing, interplay with accumulation;
  "Revisit global batch size"), lines ≈1149–1495 — `sys_ultrascale2025_playbook.txt`.
- See also `sb.*` lessons on data parallelism / FSDP (Scaling Book series); numbers must agree.
- Scaling Book Part 5 "Data Parallelism" and Part 12 "Rooflines for LLM Scaling on GPUs › Data Parallelism" (compute-bound
  iff per-GPU tokens $> C/W$: ≈2200 within a node, ≈2475 across nodes on H100) — `sys_scalingbook_training.txt`, `sys_scalingbook_gpus.txt`.

**Subtopic map**
1. *Setup.* Each rank holds a full replica, processes $B/P$ examples, computes local gradients; averaging the gradients
   makes the update identical to single-device training on the full batch (derive: mean of means with equal shards).
2. *Why all-reduce (vs parameter server)*: no central bottleneck; bandwidth-optimal ring (derived in `sys.collectives`).
3. *Initialisation*: broadcast parameters and buffers from rank 0 so replicas start identical; same seeds where needed;
   `DistributedSampler` gives each rank a disjoint shard (and `set_epoch` for shuffling).
4. *Naive version and its problem*: wait for the whole backward, then one all-reduce → communication fully exposed.
5. *Overlap via autograd hooks*: gradients become ready layer by layer in reverse order; launch communication for ready
   gradients while earlier layers are still computing backward.
6. *Bucketing*: many small all-reduces are latency-bound; fuse gradients into ~25 MB buckets (flattened, in reverse
   parameter order) and all-reduce each bucket when all its gradients are ready. Trade-off: big buckets = less overhead
   but less overlap (the first bucket waits longer); tiny buckets = latency-bound. `gradient_as_bucket_view` avoids a copy.
7. *Unused parameters*: a parameter that gets no gradient never fires its hook → the bucket never completes → hang;
   `find_unused_parameters=True` traverses the graph to mark them (cost); `static_graph`.
8. *Gradient accumulation with DDP*: `no_sync()` for the first $k-1$ micro-batches.
9. *When DDP stops scaling*: memory (every rank holds 16 bytes/param — motivates ZeRO); communication (comm time is
   fixed per step at ≈ $2N \cdot \text{bytes}/W$ while compute per GPU shrinks as $B/P$ shrinks): compute-bound only if
   per-GPU tokens exceed $C/W$; global batch-size limits (link tuning lessons).
10. *Details that bite*: BatchNorm uses per-rank statistics unless SyncBatchNorm; buffers re-broadcast each forward;
    loss/metric averaging for logging; uneven inputs at the end of an epoch (Join); rank-dependent control flow → hang;
    gradient averaging vs summing (DDP averages). One line of multi-host hygiene: normalise the loss by the *global* batch
    when per-rank counts differ; same init seed, different data seeds. Link `sys.tuning-pipeline` (card 5) for the rest of
    the checklist. (With DDP only, rank 0 can save the replicated state; sharded schemes differ — see
    `sys.reliability-checkpointing`.)
11. *Probes*: "walk me through DDP's backward", "why buckets", "why does DDP hang with unused params", "DDP vs
    DataParallel", "when is DDP communication-bound".

**Worked computations (verified)**
Assumption stated in every example: **bf16 gradients** (bf16 params or a bf16 comm hook). With default autocast + DDP the
gradients are fp32, which doubles comm time and the threshold (≈ 4400 / 4950 tokens).
- 1.3B params, bf16 grads (2.6 GB), one node: ring all-reduce at 450 GB/s ≈ 10 ms (fp32: ≈ 20 ms) vs backward compute for
  8192 tokens/GPU ≈ $4 \times 1.3\text{e}9 \times 8192 / 989\text{e}12 \approx 43$ ms at peak → hideable except the last bucket
  (≈ bucket size / bandwidth, which cannot overlap because the first layers' gradients arrive last).
- Same model across 64 GPUs (8 nodes): node-egress model $2\cdot\tfrac{63}{64}\cdot2.6\text{e}9/400\text{e}9 \approx 12.8$ ms;
  explicit sequential hierarchical schedule ≈ 21.5 ms (see `sys.collective-algorithms`). Both are well below the 43 ms
  backward → **hidden**. (The naive flat ring at 50 GB/s per GPU would give ≈ 102 ms; use it as a distractor.)
- An exposed case: the same 64 GPUs at 1024 tokens/GPU (below 2475): backward ≈ 5.4 ms < 12.8 ms of all-reduce → exposed.
  Or fp32 gradients without overlap: ≈ 25.6 ms.
- Buckets: `bucket_cap_mb` is MiB; 1.3B fp32 grads in 25 MiB buckets ≈ 198 buckets.
- Compute-bound threshold (Scaling Book): per-GPU tokens $> C/W$ = 989e12 / 450e9 ≈ 2200 (node), / 400e9 ≈ 2475 (across nodes),
  for bf16 gradients.

**Question ideas**
- Derive: the DP compute/comm ratio per layer and the $B/P > C/W$ condition.
- Debug: DDP hangs at the first backward when a branch of the model is skipped on some steps — why and two fixes.
- Predict: halving bucket size from 25 MiB to 1 MiB on a 1B model over InfiniBand.
- Compute: cross-node all-reduce time for 1.3B bf16 grads on 64 GPUs; distractors include the 50 GB/s/GPU flat-ring answer.
- Predict: the same run with default autocast (fp32 grads) — what happens to comm time and the compute-bound threshold?
- Which is false: "DDP all-reduces gradients after the entire backward finishes" (false) / "DDP broadcasts parameters at construction" / ...
- Compare: DDP vs `nn.DataParallel` (single process, GIL, scatter/gather through GPU 0).

**Pitfalls / verify**
- The 2200/2475-token thresholds assume perfect overlap and peak FLOPs; say so.
- DDP averages gradients by dividing by world size; with uneven per-rank token counts the average is not the global
  per-token mean (link `sys.gradient-accumulation`).

**Figures**
- `ddp-overlap`: timeline of backward compute per layer (reverse) with bucket all-reduces overlapping on a comm stream, vs the naive exposed version.
- `bucket-tradeoff`: step time vs bucket size generated from an α–β model plus last-bucket exposure (latency-bound left,
  poor overlap right), not hand-drawn.
- `dp-roofline`: comm time and backward time vs per-GPU tokens, crossing at $C/W$, with intra-node (450 GB/s) and cross-node
  (400 GB/s per node) lines, plus the fp32-gradient line.

---

## sys.zero — ZeRO: sharding optimizer state, gradients and parameters

Level core · prereqs sys.ddp, sys.memory-anatomy · **W1** · ~9 cards

**Problem.** "DDP stores 16 bytes/param on every GPU, so a 7B model needs 112 GB per GPU just for model states. Each
rank only needs the optimizer state for the parameters it updates — can we shard everything without extra communication?"
This lesson is the theory of ZeRO-1/2/3 (= FSDP's full sharding); `sys.fsdp` covers the implementation.

**Primary sources**
- Rajbhandari et al. 2020 §3.1 (model states $2\Psi + 2\Psi + K\Psi$, $K = 12$), §5 ZeRO-DP with §5.1 $P_{os}$, §5.2 $P_g$,
  §5.3 $P_p$ and Fig. 1 (7.5B on 64 GPUs: 120 → 31.4 → 16.6 → 1.9 GB), §7 communication analysis (DP $2\Psi$; $P_{os+g}$
  $2\Psi$; $P_{os+g+p}$ $3\Psi$ = 1.5×), §6 ZeRO-R (partitioned activation checkpointing, constant buffers,
  defragmentation) — `sys_rajbhandari2020_zero.txt` (https://arxiv.org/abs/1910.02054).
- Ultra-Scale Playbook "ZeRO (Zero Redundancy Optimizer)" with ZeRO-1/2/3 memory and comm diagrams, lines ≈1496–1853 — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Part 5 "Fully-Sharded Data Parallelism (FSDP)" (ZeRO stages have the same compute-bound condition as DP;
  when it becomes comm-bound) — `sys_scalingbook_training.txt`. See also `sb.fsdp`.
- Ren et al. 2021 (ZeRO-Offload: keep fp32 states and the optimizer step on CPU; why it is the optimal split) — `sys_ren2021_zero_offload.txt`.

**Subtopic map**
1. *Redundancy in DDP*: every rank stores identical fp32 master weights, Adam m, v, gradients and bf16 weights.
2. *ZeRO-1 ($P_{os}$)*: shard optimizer state (and fp32 master weights) $P$ ways. Step: all-reduce gradients (or
   reduce-scatter), each rank updates its $1/P$ shard, all-gather the updated bf16 params. Memory $4N + 12N/P$ (derive).
3. *ZeRO-2 ($P_{os+g}$)*: also shard gradients — reduce-scatter instead of all-reduce so each rank only receives the
   gradient shard it needs. Memory $2N + 14N/P$. Bucketed reduce-scatter during backward, freeing non-owned gradients.
4. *ZeRO-3 ($P_{os+g+p}$, = FSDP full shard)*: also shard bf16 params. Before each layer's forward, all-gather its
   params; discard after use; all-gather again in backward; reduce-scatter gradients. Memory $16N/P$ (+ the largest
   gathered unit + activations).
5. *Communication volume* (derive with the per-rank lower bounds): DDP all-reduce = RS + AG = $2N$ elements per rank;
   ZeRO-1/2: RS of grads ($N$) + AG of params ($N$) = $2N$ — same as DDP; ZeRO-3: AG in forward ($N$) + AG in backward ($N$)
   + RS of grads ($N$) = $3N$, i.e. 1.5× DDP.
5b. *Reconciling the two sources (one card + MCQ).* ZeRO §7 counts **volume**: 3N vs 2N per step, i.e. 1.5×. The Scaling
   Book says the stages "have the same communication cost" in the sense of the **compute/comm ratio**: in the forward,
   an all-gather of $2N$ bytes (bf16) faces $2NB$ FLOPs; in the backward, AG + RS of $2\cdot2N$ bytes face $4NB$ FLOPs. Both
   phases have bytes/FLOP $= 1/B$, the same as DDP's backward, so the threshold $B/P > C/W$ is unchanged: "1.5× more bytes,
   same compute-bound condition".
6. *Overlap and granularity*: ZeRO-3 must prefetch the next layer's all-gather during the current layer's compute;
   per-layer comm vs compute decides whether it is hidden (Scaling Book roofline: same $B/P > C/W$ condition as DP).
7. *What ZeRO does not shard*: activations (need TP/SP/CP or checkpointing), so per-GPU micro-batch still matters;
   ZeRO-R partitions activation checkpoints across TP ranks.
8. *Offload* (ZeRO-Offload/Infinity): fp32 states and Adam step on CPU, bf16 params/grads over PCIe; arithmetic of
   PCIe bandwidth vs step time; when it is worth it (small GPU count, fine-tuning).
9. *Choosing a stage*: ZeRO-1 composes well with pipeline parallelism (ZeRO-2/3 would need per-micro-batch
   reduce-scatter/all-gather); ZeRO-2 is often the sweet spot when params fit; ZeRO-3 when they don't.
10. *Probes*: "memory per GPU for model X on Y GPUs at each stage", "communication of ZeRO-3 vs DDP and why", "why is
    ZeRO-1 free", "ZeRO vs model parallelism".

**Worked computations (verified)**
- ZeRO paper check, $N = 7.5$B, $P = 64$: DDP 120 GB; ZeRO-1 $4N + 12N/64 = 31.4$ GB; ZeRO-2 $2N + 14N/64 = 16.6$ GB;
  ZeRO-3 $16N/64 = 1.875$ GB (paper's Fig. 1: 1.88).
- $N = 7$B, $P = 64$: 112 / 29.3 / 15.5 / 1.75 GB.
- $N = 70$B: ZeRO-3 on 512 GPUs → 2.19 GB/GPU of model states; ZeRO-1 with $P = 128$ → 286.6 GB (does not fit: ZeRO-1 alone cannot hold the replicated 4N = 280 GB of bf16 params+grads).
- Per-rank traffic per step for 7B in bf16: DDP $2 \times 14$ GB = 28 GB; ZeRO-3 42 GB.

**Question ideas**
- Compute: per-GPU model-state memory for a given $N$, $P$, stage.
- Derive: why ZeRO-2 has the same communication volume as DDP.
- Predict: switching from ZeRO-2 to ZeRO-3 at fixed per-GPU batch across nodes — what happens to step time and why?
- Which is false: "ZeRO-3 shards activations across data-parallel ranks" (false) / "ZeRO-1 shards the fp32 master weights" / ...
- Which is true: "ZeRO-3 moves 1.5× the bytes of DDP yet has the same per-GPU-batch threshold for being compute-bound" (true;
  distractors: "same bytes", "1.5× higher threshold").
- Open estimate: 13B model on 64 H100s with ZeRO-3, 4k context, 16k tokens per GPU — model-state memory per GPU (3.25 GB),
  activations with full recompute, comm per step, hidden or exposed?
- Compare: ZeRO-3 + large batch vs tensor parallelism for a 70B model (memory, comm pattern, batch-size needs).

**Pitfalls / verify**
- ZeRO paper writes $\Psi$ for parameters and $N_d$ for DP degree; translate explicitly.
- Some descriptions put fp32 master weights in "optimizer state" (ZeRO does: $K = 12$ includes them); be consistent.
- ZeRO-3's 1.5× is a volume statement; see item 5b for why the compute-bound threshold is unchanged.

**Figures**
- `zero-stages`: the classic per-rank memory bars (params / grads / optimizer) for DDP, ZeRO-1, ZeRO-2, ZeRO-3 with 4 ranks, coloured by shard ownership.
- `zero-memory-vs-gpus`: per-GPU model-state memory vs $P$ (log x) for each stage for a 7B model, with an 80 GB line.
- `zero3-dataflow`: one layer's forward/backward with all-gather, compute, discard, reduce-scatter on a timeline.

---

## sys.fsdp — FSDP in practice: units, prefetching, FSDP2, HSDP

Level intermediate · prereqs sys.zero · W2 · ~9 cards

**Problem.** "ZeRO-3 in theory is one line; in practice step time depends on how you wrap modules, when you prefetch,
what dtype you reduce in, and how big the all-gather peak is. How does PyTorch FSDP actually run, and what goes wrong?"

**Primary sources**
- Zhao et al. 2023 §3.1 model initialisation (deferred/meta init), §3.2.1 full sharding with FlatParameter, §3.2.2 hybrid
  sharding (shard within a group, replicate across; RS within + all-reduce across), §3.2.3 autograd, §3.3.1 overlap,
  §3.3.2 backward prefetching, §3.3.3 forward prefetching, §3.3.4 gradient accumulation (with/without communication),
  §3.4.1 caching allocator and cross-stream frees, §3.4.2 rate limiter, §4.4 native mixed precision, §5 evaluation —
  `sys_zhao2023_pytorch_fsdp.txt` (https://arxiv.org/abs/2304.11277).
- See also `sb.fsdp` (Scaling Book series).
- PyTorch `fully_shard` (FSDP2) docs: DTensor parameters sharded on dim 0, grouping per `fully_shard` call, all-gather and
  reduce-scatter streams, implicit and explicit prefetching (`set_modules_to_forward_prefetch`), `reshard_after_forward`,
  `MixedPrecisionPolicy` — `sys_pytorch_fsdp2_fully_shard.txt`; FSDP1 API (ShardingStrategy, BackwardPrefetch, CPUOffload,
  auto_wrap_policy, MixedPrecision) — `sys_pytorch_fsdp1.txt`.
- TorchTitan (FSDP2 per-parameter sharding vs FlatParameter; composability with TP/PP; Float8) — `sys_liang2024_torchtitan.txt`.
- DTensor placements (Shard, Replicate, Partial; DeviceMesh) — `sys_pytorch_dtensor.txt`.

**Subtopic map**
1. *FSDP unit*: the granularity at which params are gathered/freed (one transformer block is typical). Peak memory ≈
   sharded states + largest gathered unit(s) (current + prefetched) + activations. Root unit holds leftovers (embeddings, head).
2. *Wrapping granularity trade-off*: too coarse (whole model one unit) → no memory saving at peak; too fine (every
   Linear) → many small latency-bound collectives and poor overlap.
3. *Forward*: all-gather unit params → compute → free (unless `reshard_after_forward=False`, which is ZeRO-2-like:
   keep params for backward to save one all-gather at the cost of memory).
4. *Backward*: all-gather again (if resharded) → compute grads → reduce-scatter grads → free; **backward prefetch**
   issues the next unit's all-gather before the current unit's grad computation finishes; forward prefetch for
   static graphs.
5. *Streams and the rate limiter*: comm on separate streams; CPU can run ahead and issue too many all-gathers →
   memory spikes; FSDP limits in-flight all-gathers (FSDP1 `limit_all_gathers`); cross-stream frees and the caching allocator.
6. *Mixed precision in FSDP*: `param_dtype` (bf16 compute copy gathered), `reduce_dtype` (fp32 reductions for accuracy vs
   bf16 to halve comm), sharded fp32 master weights; this is the ZeRO memory model in practice.
7. *FSDP1 vs FSDP2* (one card): FSDP1 flattens and concatenates a unit's params into one FlatParameter and shards the flat buffer
   (awkward per-parameter dtype/requires_grad, opaque optimizer state); FSDP2 shards each parameter along dim 0 as a
   DTensor (clean state dicts, composes with TP via 2-D meshes, per-parameter features like Float8), at the cost of
   copy-in/copy-out for all-gather.
8. *HSDP (hybrid sharding)*: shard within a node (fast NVLink), replicate across nodes; gradient reduction = RS within
   group + all-reduce across groups. Trade: more memory per GPU, much less inter-node traffic.
9. *Other features*: deferred init on the meta device to build models larger than one GPU; gradient clipping needs a
   global norm across shards (link `sys.gradient-clipping`); gradient accumulation with or without reduce-scatter per
   micro-batch. CPU offload is taught in `sys.zero` item 8; checkpointing in `sys.reliability-checkpointing` (with FSDP,
   sharded state dicts are written by **all** ranks; gathering a full state dict to rank 0 is the slow, memory-hungry option).
10. *Failure modes*: OOM at all-gather peaks (the largest unit, e.g. the embedding or LM head, or too many prefetches);
    slow small units; exposed communication across nodes at small per-GPU batch; mismatched wrapping across ranks; mixing
    FSDP with pipeline parallelism (ZeRO-3 per micro-batch is expensive).
11. *Probes*: "what happens inside FSDP for one layer's forward and backward", "why prefetch", "FSDP vs DDP: when would
    you pick DDP", "what is HSDP", "what changed in FSDP2".

**Worked computations (verified)**
- 70B model, 80 layers, unit = one layer in bf16: gathered unit ≈ 70e9 / 80 × 2 bytes ≈ 1.75 GB; with one unit prefetched,
  ≈ 3.5 GB of transient parameter memory; during the unit's backward its full-size gradient also exists before the
  reduce-scatter (+1.75 GB in bf16, +3.5 GB in fp32) → ≈ 5.25–7 GB transient on top of the 2.19 GB of sharded states on 512 GPUs.
- HSDP inter-node traffic: with 8-GPU shard groups, the cross-node all-reduce moves gradient shards of size $N/8$ per GPU
  (writers: compute bytes for a stated model and compare with flat FSDP's reduce-scatter over all ranks).

**Question ideas**
- Order the events: AG(unit i+1) / forward(unit i) / free(unit i) / RS(unit i) / backward(unit i) in forward and backward.
- Debug: FSDP training OOMs only at the first backward of the LM head; why and what fixes it.
- Predict: wrapping every `nn.Linear` separately in a 7B model across 4 nodes.
- Predict: what all-gather peak does a 128k-vocab LM head (h = 8192, bf16) cause if it is its own unit vs merged into the root unit?
- Which is false: "`reshard_after_forward=False` reduces communication at the cost of memory" (true) / "resharding after forward
  saves memory at the cost of an extra all-gather in backward" (true) / ... (writers: build a which-is-false from behaviours, not API names).
- Compare: HSDP vs FSDP across 8 nodes.

**Pitfalls / verify**
- FSDP1 and FSDP2 option names differ (ShardingStrategy.SHARD_GRAD_OP vs `reshard_after_forward=False`); quote whichever
  version the lesson uses and mention the other.
- Exact prefetch defaults change across versions; describe behaviour, cite the version cached (docs 2.14).

**Figures**
- `fsdp-timeline`: two-stream timeline (compute / comm) for 4 units: forward with AG prefetch, backward with AG prefetch and RS.
- `fsdp-peak-memory`: memory over a step for coarse vs fine wrapping, showing all-gather peaks.
- `hsdp-mesh`: 2 nodes × 4 GPUs grid; shard groups inside nodes, replica groups across.

---

## sys.tensor-parallel — Tensor parallelism (Megatron-style)

Level core · prereqs sys.collectives, sys.memory-anatomy, sys.roofline · **W1** · ~9 cards

**Problem.** "One transformer layer's weights or activations are too big for one GPU, or data parallelism has run out of
batch. Can we split each matmul across GPUs, and where must the GPUs talk?"

**Refreshers to include inline.** Block-matrix multiplication (column/row partitions); log-sum-exp for item 5.

**Primary sources**
- Shoeybi et al. 2019 §3 "Model Parallel Transformers": MLP split (first GEMM column-parallel so GeLU stays local, second
  GEMM row-parallel, one all-reduce), self-attention split by heads, the conjugate operators $f$ (identity forward,
  all-reduce backward) and $g$ (all-reduce forward, identity backward), two all-reduces in forward and two in backward per
  layer, vocab-parallel embedding and fused cross-entropy to avoid gathering $b\times s\times v$ logits —
  `sys_shoeybi2019_megatron.txt` (https://arxiv.org/abs/1909.08053).
- Narayanan et al. 2021 §3 takeaway #1 (use TP up to the GPUs in a node, PP across nodes), §2.3 tensor model parallelism —
  `sys_narayanan2021_megatron_ptd.txt`.
- Korthikanti et al. 2022 §4.2.1 (TP activation memory $sbh(10 + 24/t + 5as/(ht))$; the replicated 10sbh) — `sys_korthikanti2022_seqpar_recompute.txt`.
- Ultra-Scale Playbook "Tensor Parallelism", "Tensor Parallelism in a Transformer Block" (column/row linear, comm
  exposed on the critical path), lines ≈1854–2200 — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Part 3 cases 1–4 (sharded matmuls) and Part 12 "Tensor Parallelism" (compute-bound iff $t < F\,W/C$ ≈ $F/2475$) —
  `sys_scalingbook_sharding.txt`, `sys_scalingbook_gpus.txt`; PyTorch TP tutorial (ColwiseParallel/RowwiseParallel, loss
  parallel, 2-D FSDP+TP) — `sys_pytorch_tp_tutorial.txt`. See also `sb.tensor-parallelism`.

**Subtopic map**
1. *Two ways to split $Y = XA$*: by columns of $A$ (each rank gets full $X$, produces a column slice of $Y$; no comm) or
   by rows of $A$ (each rank gets a slice of $X$'s columns, produces a partial sum; needs an all-reduce). Derive both with
   block matrices.
2. *MLP*: why column-then-row: GeLU is elementwise, so $\mathrm{GeLU}(XA_i)$ is local only if $A$ is split by columns
   (show that row-splitting would need an all-reduce *before* GeLU because $\mathrm{GeLU}(a+b) \ne \mathrm{GeLU}(a)+\mathrm{GeLU}(b)$);
   the second GEMM row-parallel consumes the column-sliced activations directly; one all-reduce at the end.
3. *Attention*: heads are independent → split Q, K, V projections by heads (column-parallel), attention per head locally,
   output projection row-parallel, one all-reduce.
4. *The $f$/$g$ operators and backward*: $f$ = identity forward / all-reduce backward (the input's gradient is a sum of
   partial gradients from all ranks); $g$ = all-reduce forward / identity backward. Total per layer: 2 all-reduces forward
   + 2 backward (derive).
5. *Embeddings and loss* (inline LSE refresher, since `sys.stable-numerics` is W2): vocab-parallel embedding (each rank holds $V/t$ rows; masked lookup + all-reduce); parallel
   cross-entropy computing max and sum-exp across shards with small all-reduces instead of gathering logits (link
   `sys.stable-numerics` LSE).
6. *Communication cost and placement*: each all-reduce moves $2\frac{t-1}{t} \cdot bsh \cdot 2$ bytes per rank, sits on the
   critical path (the next op depends on it), and happens $4L$ times per step. Compute per layer shrinks as $1/t$ while comm
   per layer does not → TP is only efficient on NVLink within a node (Scaling Book condition $t < F W / C$).
7. *Memory effect*: weights, grads and optimizer state divided by $t$; activations inside the blocks divided by $t$ but
   LayerNorm/dropout regions replicated (the $10sbh$ term) → motivates sequence parallelism (next lesson).
8. *Other details*: TP degree must divide heads (and KV heads for GQA) and hidden sizes; dropout RNG must differ across TP
   ranks inside the parallel region and match outside; overlapping TP comm with compute (async TP / collective matmul);
   TP in inference (latency).
9. *Probes*: "derive where the all-reduces go in a Megatron MLP", "why column first", "why TP within a node", "how many
   all-reduces per layer", "what does TP not shard".

**Worked computations (verified)**
- Layer with $b = 1$, $s = 4096$, $h = 8192$, $t = 8$: forward compute per layer ≈ $24bsh^2/t / C \approx 0.83$ ms on H100;
  forward comm (2 all-reduces of $bsh$ bf16 activations at 450 GB/s) ≈ 0.52 ms — the same order as compute even on NVLink,
  and ≈ 4.7 ms if the TP group is spread one GPU per node (each GPU's activations leave through its own 50 GB/s NIC — the
  case where the per-GPU NIC bandwidth is the right number), 5–6× the compute.
- Scaling-Book rule for an MLP with $F = 28672$: compute-bound up to $t \approx F/2475 \approx 11.6$ (across nodes) or
  $F/2200 \approx 13$ (within a node) → in practice 8-way TP = one node.

**Question ideas**
- Derive: shapes on each rank for a 2-way TP MLP with $X \in \mathbb{R}^{bs\times h}$, $A \in \mathbb{R}^{h\times 4h}$, $B \in \mathbb{R}^{4h\times h}$.
- Predict: what goes wrong if the first MLP GEMM is row-split.
- Compute: number and size of all-reduces per training step for $L$ layers.
- Which is false: "TP reduces activation memory of LayerNorm inputs" (false without SP) / "attention heads can be split across ranks without communication inside attention" / ...
- Compare: TP vs FSDP for scaling a 70B model across 64 GPUs (comm pattern: activations on critical path vs weights prefetchable).

**Pitfalls / verify**
- "2 all-reduces per layer" refers to the forward pass; per training step it is 4 (plus any for embeddings/loss).
- The Scaling-Book TP rule uses the MLP width $F$; check its assumptions (overlap, bf16) before quoting.

**Figures**
- `megatron-mlp`: block diagram of column-parallel $A$ / GeLU / row-parallel $B$ / all-reduce on 2 ranks with tensor shapes.
- `megatron-attention`: heads split across ranks with the row-parallel output projection.
- `tp-comm-vs-compute`: per-layer compute time vs comm time as $t$ grows for NVLink and IB (from the formulas).

---

## sys.sequence-context-parallel — Sequence parallelism and context parallelism

Level advanced · prereqs sys.tensor-parallel, sys.activation-checkpointing, sys.stable-numerics · W3 · ~8 cards

**Problem.** "TP leaves LayerNorm and dropout activations replicated, and at 128k context even one sequence's activations
don't fit. How do we shard along the sequence dimension, and what does attention need to communicate?"

**Refreshers to include inline.** Online softmax: merging two partial softmax results with running max and sum (or link
`sys.stable-numerics` item 8).

**Primary sources**
- Korthikanti et al. 2022 §4.2.2 sequence parallelism (operators $g$ = all-gather fwd / reduce-scatter bwd and $\bar g$ =
  reduce-scatter fwd / all-gather bwd; same bandwidth as TP's all-reduces), equation 4 ($sbh(34 + 5as/h)/t$), §4.2.3
  pipeline first-stage memory, Table 2 — `sys_korthikanti2022_seqpar_recompute.txt`.
- Ultra-Scale Playbook "Sequence Parallelism" (lines ≈2201–2532) and "Context Parallelism", "Discovering Ring Attention",
  "Zig-Zag Ring Attention" (lines ≈2533–2729) — `sys_ultrascale2025_playbook.txt`.
- Jacobs et al. 2023, DeepSpeed-Ulysses (all-to-all head sharding) — `sys_jacobs2023_ulysses.txt` (https://arxiv.org/abs/2309.14509).
- Liu et al. 2023, Ring Attention — shared `llm_liu2023_ring.txt` (overlap: ring attention also appears in `llm.long-context`,
  id to confirm; this lesson owns the parallelism/communication view); online softmax — `llm_milakov2018_online_softmax.txt`.
- PyTorch TP tutorial "Apply Sequence Parallel to LayerNorm/RMSNorm layers" — `sys_pytorch_tp_tutorial.txt`.

**Subtopic map**
1. *The replicated regions*: after TP, LayerNorm and dropout see the full $s\times b\times h$ tensor on every rank — the $10sbh$ term.
2. *Sequence parallelism (Megatron SP)*: these ops are independent per token → shard along $s$. Replace TP's all-reduce
   by reduce-scatter (into the SP region) and all-gather (into the TP region); derive why RS + AG = AR in volume, so SP
   costs no extra bandwidth. Memory becomes $sbh(34 + 5as/h)/t$ (everything divided by $t$).
3. *Context parallelism (CP)*: shard the sequence across ranks for *all* layers; MLP and norms are local; attention needs
   every query to see all keys/values.
4. *Ring attention*: pass K/V blocks around a ring; each rank computes partial attention for its queries against the
   current block and merges with online-softmax statistics (derive the merge of two partial softmaxes with running max
   and sum). Overlap condition: per-block attention compute ≥ K/V block transfer time.
5. *Causal load imbalance*: with contiguous chunks the last rank does most of the work; zig-zag assignment (chunks $i$ and
   $2P-1-i$) balances it.
6. *DeepSpeed-Ulysses* (Jacobs et al. 2023, `sys_jacobs2023_ulysses.txt`): sequence-sharded Q/K/V are re-partitioned by an
   all-to-all so each GPU holds the full sequence for a subset of heads, attention runs locally, and a second all-to-all
   restores sequence sharding; per-GPU volume stays constant as sequence length and GPU count grow together; limited by the
   number of heads; all-to-all is cheap inside a node and expensive across nodes (`sys.collective-algorithms` item 4).
   Compare with ring attention (K/V passed around a ring, no head limit).
7. *Composition*: CP is another mesh axis; combined with TP+SP inside nodes; CP communication is of K/V (GQA shrinks it).
8. *Probes*: "what does SP shard that TP doesn't", "why SP has no extra comm", "how does ring attention compute an exact
   softmax across shards", "why zig-zag".

**Worked computations**
- Per-layer activations for GPT-3 config (s = 2048, b = 1, h = 12288, a = 96) with $t = 8$: TP only
  $sbh(10 + 24/8 + 5\cdot96\cdot2048/(12288\cdot8)) = 23\,sbh \approx 0.58$ GB vs TP+SP $sbh(34 + 80)/8 = 14.25\,sbh \approx 0.36$ GB
  (no parallelism: $114\,sbh \approx 2.87$ GB).
- Ring attention overlap: block of $s/P$ tokens; compute ∝ $(s/P)^2 h$, transfer ∝ $(s/P)h$ → overlap holds once $s/P$
  exceeds a hardware-dependent threshold (derive with $C$ and $W$).

**Question ideas**
- Derive: the backward of SP's all-gather.
- Compute: TP vs TP+SP per-layer activation memory for a config.
- Explain: merging two partial attention results with different running maxima.
- Which is false: "context parallelism requires communication in the MLP" (false) / ...

**Pitfalls / verify**
- "Sequence parallelism" is used for different things in different papers (Megatron SP vs ring-style CP); define both.

**Figures**
- `sp-regions`: transformer layer strip with TP regions (sharded on h) and SP regions (sharded on s), with AG/RS at the boundaries.
- `ring-attention`: 4 ranks, Q blocks fixed, K/V blocks rotating over 4 steps; causal mask with zig-zag assignment.

---

## sys.pipeline-parallel — Pipeline parallelism: GPipe, the bubble, 1F1B

Level core · prereqs sys.memory-anatomy, sys.collectives, sys.ddp · **W1** · ~9 cards

**Problem.** "Split the layers into $p$ stages on $p$ GPUs (cheap communication: only activations at stage boundaries).
Naively only one GPU works at a time. How much idle time ('bubble') remains with micro-batching, and what does it cost in memory?"

**Primary sources**
- Huang et al. 2019 (GPipe) §2.2–2.3: partition into $K$ cells, split mini-batch into $M$ micro-batches, synchronous
  gradient accumulation and flush, re-materialisation, bubble $O((K-1)/(M+K-1))$, negligible for $M \ge 4K$ —
  `sys_huang2019_gpipe.txt` (https://arxiv.org/abs/1811.06965).
- Narayanan et al. 2021 §2.2.1 default schedule (bubble time $t_{pb} = (p-1)(t_f+t_b)$; bubble fraction $t_{pb}/t_{id} =
  (p-1)/m$), 1F1B memory (at most $p$ micro-batches in flight), §3 takeaways #2–3 — `sys_narayanan2021_megatron_ptd.txt`.
- Narayanan et al. 2019 (PipeDream) 1F1B steady state, weight stashing — `sys_narayanan2019_pipedream.txt`.
- Korthikanti et al. 2022 §4.2.3 (1F1B first stage stores $p$ micro-batches = $L$ layers' worth of activations) —
  `sys_korthikanti2022_seqpar_recompute.txt`.
- Ultra-Scale Playbook "Pipeline Parallelism", AFAB, "One-forward-one-backward" (lines ≈2730–3005) — `sys_ultrascale2025_playbook.txt`;
  Scaling Book Part 12 "Pipeline Parallelism" (comm cost tiny; why PP plays badly with ZeRO-3) — `sys_scalingbook_gpus.txt`.

**Subtopic map**
1. *Naive model parallelism*: stage $i$ waits for stage $i-1$; utilisation $1/p$.
2. *Micro-batching (GPipe / all-forward-all-backward)*: draw the schedule; derive total time $(m + p - 1)(t_f + t_b)$ vs
   ideal $m(t_f + t_b)$; bubble as a fraction of total time $(p-1)/(m+p-1)$; Narayanan's ratio to ideal time $(p-1)/m$
   (both conventions; know which one an interviewer means).
3. *Memory of GPipe*: each stage stores activations of all $m$ micro-batches until their backward → grows with $m$;
   GPipe uses re-materialisation to cut it.
4. *1F1B (PipeDream-Flush)*: warm-up forwards, then alternate one forward and one backward, then cool-down. Same bubble as
   GPipe for the same $m$ and $p$, but at most $p$ micro-batches in flight (first stage holds $p$ → $L$ layers' worth of
   activations in total, independent of $p$). Why this decouples memory from $m$, letting $m$ grow to shrink the bubble.
5. *Synchronous semantics* (inline refresher on gradient accumulation, since `sys.gradient-accumulation` is W2): flush at the
   end of each batch, gradients accumulated over micro-batches, one optimizer step —
   mathematically equal to the non-pipelined update.
6. *Communication*: send/recv of $b\cdot s\cdot h$ activations (and their gradients) between neighbours; tiny compared with
   TP/DP (Scaling Book estimate), so PP is used across nodes.
7. *Load balancing*: stages must take equal time; embedding and LM head make the first/last stage heavier; layer counts
   must divide; imbalance adds to the bubble.
8. *Interaction with DP/ZeRO*: global batch = micro-batch × $m$ × dp; large $m$ needs a large global batch; ZeRO-2/3 per
   micro-batch would repeat reduce-scatter/all-gather, so PP pairs with ZeRO-1; the DP all-reduce happens after the last
   micro-batch and is hard to overlap.
9. *Probes*: "derive the bubble fraction", "1F1B vs GPipe memory", "why must m ≫ p", "why PP across nodes and TP within".

**Worked computations (verified)**
- $p = 8$, $m = 32$: fraction of total time idle $= 7/39 \approx 17.9\%$; Narayanan ratio $(p-1)/m = 7/32 \approx 21.9\%$.
- $p = 4$, $m = 8$: 27.3% (ratio 37.5%); $p = 8$, $m = 8$: 46.7% (ratio 87.5%); $p = 16$, $m = 64$: 19.0%.
- GPipe's "negligible for $M \ge 4K$": $p = 8$, $m = 32$ still idles ≈ 18% of the time — "negligible" is a loose claim; say so.
- Memory: GPipe without remat stores $m = 32$ micro-batches per stage; 1F1B stores at most $p = 8$ on the first stage.

**Question ideas**
- Compute: bubble for given $p$, $m$ under both conventions.
- Draw/identify: given a schedule figure, is it GPipe or 1F1B and what is its peak in-flight count per stage? (figure MCQ)
- Predict: doubling $p$ at fixed global batch and micro-batch size — what happens to bubble and memory?
- Which is false: "1F1B has a smaller bubble than GPipe for the same m and p" (false; same bubble, less memory) / ...
- Explain: why is the DP gradient all-reduce hard to overlap when combined with PP?

**Pitfalls / verify**
- Bubble formulas assume equal stage times and $t_b \approx 2t_f$ only for drawing; the fraction itself does not depend on the ratio.
- GPipe paper uses $K$ partitions and $M$ micro-batches; Narayanan uses $p$ and $m$.

**Figures**
- `gpipe-schedule`: Gantt chart for $p = 4$, $m = 8$ (forward blue, backward double width), bubbles greyed.
- `1f1b-schedule`: same $p, m$ with 1F1B, annotated with in-flight micro-batch count per stage.
- `bubble-vs-m`: bubble fraction vs $m$ for $p = 4, 8, 16$ (both conventions as solid/dashed).

---

## sys.pipeline-schedules — Advanced pipeline schedules: interleaved, zero-bubble, DualPipe, async

Level advanced · prereqs sys.pipeline-parallel · W3 · ~8 cards

**Problem.** "With a fixed global batch we cannot make $m$ arbitrarily large. Can smarter schedules shrink the bubble
further, and what do they cost?"

**Primary sources**
- Narayanan et al. 2021 §2.2.2 interleaved schedule ($v$ model chunks per device; bubble $(p-1)/(vm)$; $v\times$ more
  communication), §4.1 scatter/gather optimisation — `sys_narayanan2021_megatron_ptd.txt`.
- Korthikanti et al. 2022 §4.2.3 (interleaving multiplies first-stage activation memory by $1 + (p-1)/(pm)$, with $m$ there
  meaning the number of interleaved stages — note the notation clash) — `sys_korthikanti2022_seqpar_recompute.txt`.
- Qi et al. 2023 (Zero Bubble): split backward into B (input gradient, on the critical path) and W (weight gradient,
  deferrable); ZB-H1 (1F1B memory, bubble about a third of 1F1B's), ZB-H2 (more memory, near-zero bubble),
  post-validation instead of optimizer synchronisation — `sys_qi2023_zero_bubble.txt`.
- DeepSeek-V3 §3.2.1 DualPipe (bidirectional scheduling, overlapping forward/backward compute with communication) — shared `llm_deepseek2024_v3.txt`.
- PipeDream (asynchronous 1F1B, weight stashing, staleness) — `sys_narayanan2019_pipedream.txt`.
- Ultra-Scale Playbook "Interleaving stages", "Zero Bubble and DualPipe" (lines ≈3006–3226) — `sys_ultrascale2025_playbook.txt`.

**Subtopic map**
1. *Interleaved 1F1B*: each device holds $v$ non-contiguous chunks; each chunk's $t_f, t_b$ are $1/v$ as long, so the
   bubble shrinks by $v$: $(p-1)/(vm)$ relative to ideal. Cost: $v\times$ more point-to-point messages; $m$ must be a
   multiple of $p$ (Megatron's constraint, verify); more activation memory.
2. *Zero-bubble*: backward of a linear layer has two parts: $\partial L/\partial X$ (needed by the previous stage now) and
   $\partial L/\partial W$ (needed only at the optimizer step). Scheduling W into bubbles fills idle time; derive why B
   and W have similar cost (each a matmul of the same size as forward).
3. *Optimizer-step synchronisation*: gradient-norm clipping and NaN checks need all stages; zero-bubble replaces the
   barrier by post-validation (roll back if needed).
4. *DualPipe*: feed micro-batches from both ends; overlap compute and the all-to-all communication of MoE; cost: two
   copies of some parameters (verify in DeepSeek-V3).
5. *Asynchronous pipelines* (PipeDream): no flush; weight stashing keeps the version used in forward for the matching
   backward; staleness changes optimisation semantics; why most LLM training uses synchronous schedules.
6. *Choosing*: interleaving when $m/p$ is small; zero-bubble variants when memory allows; all require more complex code
   (Scaling Book reason (1)).
7. *Probes*: "how does interleaving reduce the bubble and what does it cost", "what is the B/W split", "why can't you just
   drop the flush".

**Worked computations (verified)**
- $p = 8$, $m = 32$, $v = 4$: interleaved ratio $(p-1)/(vm) = 7/128 \approx 5.5\%$ (as a fraction of total ≈ 5.2%) vs 21.9% (17.9%) non-interleaved.

**Question ideas**
- Compute: bubble with and without interleaving; how many more sends per step.
- Explain: why weight gradients can be delayed but input gradients cannot.
- Which is false: "interleaving reduces communication volume" (false) / ...
- Predict: effect of zero-bubble scheduling on peak memory (ZB-H2 holds more activations: verify the $(2p-1)$ micro-batch figure).

**Pitfalls / verify**
- Notation clash: Korthikanti's $m$ in the interleaving memory factor is the number of interleaved stages (our $v$).
- Verify the ZB-H1 "one third" statement and ZB-H2 memory in Qi et al. §3 before quoting.

**Figures**
- `interleaved-schedule`: $p = 4$, $v = 2$ Gantt chart vs plain 1F1B, same $m$.
- `zero-bubble`: 1F1B vs ZB-H1 Gantt with F, B, W blocks in three colours.

---

## sys.parallelism-composition — Composing parallelism and choosing a configuration

Level advanced · prereqs sys.zero, sys.tensor-parallel, sys.pipeline-parallel, sys.sequence-context-parallel, sys.flops-mfu · W3 · ~9 cards

**Problem.** "You have 1024 H100s and a 70B dense model (or a large MoE). Which degrees of DP/FSDP, TP, PP, CP and EP do you
pick, in which order on the device mesh, and how do you check memory, communication and batch-size constraints?"

**Primary sources**
- Narayanan et al. 2021 §3 (takeaways #1–3: TP up to node size, PP across nodes, microbatch size; $t \cdot p \cdot d = n$) — `sys_narayanan2021_megatron_ptd.txt`.
- Ultra-Scale Playbook "5D parallelism in a nutshell" (comparison table: what each axis shards and communicates) and
  "Finding the Best Training Configuration" (step 1 fit memory, step 2 reach global batch, step 3 optimise throughput;
  benchmarking lessons), lines ≈3264–4110 — `sys_ultrascale2025_playbook.txt`.
- Scaling Book Part 12 "Rooflines for LLM Scaling on GPUs" (DP needs ≈2500 tokens/GPU; TP ≲ 8-way; PP cheap in comm; EP
  conditions; DeepSeek-V3 = 64 EP × 16 PP × 2 ZeRO-1 DP on 2048 H800; LLaMA-3 = 8 TP × 16 PP × 128 ZeRO-1 DP on 16k
  H100; "TLDR" recipe) and Part 5 "Combining FSDP and Tensor Parallelism" — `sys_scalingbook_gpus.txt`, `sys_scalingbook_training.txt`.
- TorchTitan (composable 4D parallelism with DeviceMesh) — `sys_liang2024_torchtitan.txt`; shared `llm_dubey2024_llama3.txt` (Llama 3 parallelism section).
- JAX `parallel.html` (Mesh with named axes) — `sys_jax_parallel.txt`.

**Subtopic map**
1. *What each axis shards and communicates* (table the reader can rebuild): DP (nothing sharded; grad all-reduce),
   ZeRO/FSDP (states; AG/RS of weights/grads), TP (weights + activations in blocks; per-layer all-reduce/AG/RS of activations),
   SP/CP (activations along sequence; AG/RS or K/V ring), PP (layers; p2p activations), EP (experts; all-to-all of tokens).
2. *Mesh and placement*: the device mesh is ordered so the most communication-hungry axis maps to the fastest links:
   TP (and EP) inside the NVLink node, PP and DP/FSDP across nodes; HSDP-style hybrid DP.
3. *Constraints*: $d \cdot t \cdot p \cdot c \,(\cdot e) = $ #GPUs; heads divisible by $t$; layers by $p$; global batch =
   micro-batch × $m$ × $d$ must match the optimisation target (link tuning lessons); per-GPU tokens ≥ ≈ $C/W$ for DP/FSDP to
   be compute-bound; $m \gg p$ for small bubbles.
4. *A procedure*: (1) memory: pick sharding/TP/PP/recompute so states + activations fit; (2) batch: reach the target
   global batch with DP × accumulation; (3) throughput: minimise exposed comm and bubbles, measure MFU, iterate.
5. *Worked design*: 70B dense on 512 or 1024 H100s: estimate model states under candidates (FSDP only; TP8 + FSDP; TP8 + PP
   + ZeRO-1), activations per GPU, comm per step, and the resulting per-GPU batch requirements.
6. *MoE twist*: $E/k$ multiplies the per-GPU batch needed for DP/FSDP (Scaling Book); EP replaces TP for experts; all-to-all
   is ≈ 8× more expensive across nodes (`sys.collective-algorithms` item 4). EP systems are owned by `llm.moe-systems` (id from
   the current LLM plan draft; confirm); link, don't duplicate.
6b. *Strong vs weak scaling* (one line): fixed global problem vs fixed per-device work as GPUs are added.
7. *Real configs as anchors*: LLaMA-3 (16M tokens, 16k GPUs ≈ 1k tokens/GPU; 8 TP × 16 PP × 128 DP; PP reduces DP comm by
   16×), DeepSeek-V3 (64 EP × 16 PP × 2 ZeRO-1 DP; ≈30k tokens/GPU).
8. *Probes*: open design question with numbers; "why not just FSDP everything" (batch-size floor, comm across nodes);
   "why ZeRO-1 with PP"; "what would you change at 4× the GPUs with the same batch".

**Worked computations**
- LLaMA-3: 16M tokens / 16,384 GPUs ≈ 977 tokens/GPU, below the ≈2475 DP threshold → why PP (16×) and TP (8×) are needed
  (the DP group then has 128 ranks, each "rank" being 128 GPUs that share a batch shard).
- DeepSeek-V3: the batch ramps from 3072 to 15360 sequences (DeepSeek-V3 §4.2, shared `llm_deepseek2024_v3.txt`; verify); at
  15360 × 4096 = 62,914,560 tokens on 2048 GPUs ≈ 30.7k tokens/GPU. The Scaling Book quotes both "4M" (l.271) and 62.9M (l.308);
  cite DeepSeek-V3 directly and note the inconsistency.
- 70B with TP = 8 and ZeRO-1 over $d = 64$: model states per GPU $= (4N + 12N/d)/t$ (writers: derive and compute;
  compare with FSDP-only over 512 GPUs = 2.19 GB plus activations).

**Question ideas**
- Design: given model, GPUs, interconnect and target batch, propose degrees and justify each with a number.
- Predict: moving TP from within-node to across-node.
- Which is false: "PP is placed inside the node because it communicates the most" (false) / ...
- Compute: per-GPU token count for a given global batch and DP degree; is DP compute-bound?

**Pitfalls / verify**
- Real configs vary over a training run (e.g. sequence-length phases); cite the Scaling Book description and Llama 3 paper.
- Keep EP/MoE detail minimal and link to the LLM plan.

**Figures**
- `device-mesh`: 2 nodes × 8 GPUs coloured by TP group (within node), PP stage and DP replica.
- (The axes table from item 1 is a markdown table in a card, not a figure.)
- `config-tradeoffs`: for a 70B example, bar chart of per-GPU memory and exposed comm for 3 candidate configs (from formulas).

---

## sys.reliability-checkpointing — Checkpointing and fault tolerance at scale

Level advanced · prereqs sys.fsdp · W3 · ~8 cards

**Problem.** "A 16k-GPU job loses a GPU every few hours. How big is a checkpoint, who writes it, how often should you
save, and how do you restart — possibly on a different number of GPUs?" New lesson added after review (it also owns the
tuning plan's `sys.checkpointing†` rows).

**Primary sources**
- Llama 3 §3.3.4 "Reliability and Operational Challenges" (466 job interruptions in a 54-day snapshot, 419 unexpected,
  ≈78% hardware-related; >90% effective training time; checkpoint writes of 1 MB–4 GB per GPU that saturate the storage
  fabric; reducing startup and checkpoint time) — shared `llm_dubey2024_llama3.txt`.
- PaLM §5.1 (loss spikes: rewind to a checkpoint ≈100 steps earlier and skip data batches; verify the numbers) — shared `llm_chowdhery2022_palm.txt`.
- Zhao et al. 2023 / PyTorch FSDP docs (sharded vs full state dicts) — `sys_zhao2023_pytorch_fsdp.txt`, `sys_pytorch_fsdp1.txt`;
  DTensor (resharding between placements and meshes) — `sys_pytorch_dtensor.txt`; TorchTitan (distributed checkpointing,
  async checkpointing) — `sys_liang2024_torchtitan.txt`.
- ZeRO §3.1 (16 bytes/param of model states) — `sys_rajbhandari2020_zero.txt`.
- Optimal checkpoint interval: give the one-line derivation in the lesson (first-order Young/Daly approximation); no cached source.

**Subtopic map**
1. *Failure rates at scale*: per-GPU failures are rare but a job's MTBF falls as $1/N_{GPU}$; Llama 3's 419 unexpected
   interruptions in 54 days ≈ one every 3.1 hours; causes (GPU, HBM, network, host, software).
2. *What a checkpoint contains*: model states (bf16 params if kept, fp32 master, Adam m and v: ≈ 16N bytes, or 12N
   without the bf16 copy), optimizer step count, LR scheduler and grad-scaler state, RNG states per rank, data-loader
   position (so the resumed run sees the same data order), and the parallelism layout.
3. *Who writes it*: with DDP every rank has the same states, so rank 0 can write one copy. With ZeRO/FSDP/TP/PP each rank
   holds a different shard: either **every rank writes its own shard** (sharded/distributed checkpoint; parallel I/O; the
   normal choice at scale) or the full state dict is gathered to rank 0 (costs memory and time; fine for small models and
   export). Checkpoint metadata maps shards to global tensors.
4. *Resharding on restore*: resuming on a different mesh (fewer GPUs, different TP/PP) requires loading shards into new
   placements (DTensor-style resharding); why layout-agnostic formats matter.
5. *Asynchronous checkpointing*: snapshot states to CPU memory quickly (GPU pause = device-to-host copy), then write to
   storage in a background thread while training continues; cost is CPU memory and storage bandwidth; Llama 3's bursty writes.
6. *How often*: with checkpoint cost $\delta$ and MTBF $M$, expected waste per unit time ≈ $\delta/\tau + \tau/(2M)$ (save
   cost plus half an interval lost per failure); minimising gives $\tau^* \approx \sqrt{2\delta M}$ (derive). Add restart time
   (re-initialising NCCL, loading checkpoints) to the lost work.
7. *Detecting failures*: NCCL timeouts and hangs (a dead rank stalls collectives), heartbeat/watchdogs, straggler detection
   (one slow GPU slows the whole synchronous job), silent data corruption (detect via loss/grad-norm anomalies or
   recomputation checks), loss-spike rollback (rewind and skip batches).
8. *Elastic and resilient training*: restarting with spare nodes; elastic world size and what changes (global batch,
   data sharding, LR); in-memory redundancy of checkpoints across nodes (mention).
9. *Multi-host hygiene* (from the tuning plan, one line each; link `sys.tuning-pipeline` for the checklist): log from one
   host; keep the N best/most recent checkpoints; evaluate at step (not wall-clock) intervals so curves survive preemption.
10. *Probes*: "how big is the checkpoint of a 70B model and how long does it take to write", "how often would you
    checkpoint", "how do you resume FSDP training on a different GPU count", "why is rank-0-only saving wrong for FSDP".

**Worked computations (verified)**
- 70B model states: $16 \times 70\text{e}9 = 1.12$ TB; at an assumed 100 GB/s aggregate write bandwidth ≈ 11.2 s
  (label the bandwidth as an assumption).
- Llama 3: 54 days × 24 h / 419 ≈ 3.1 h between unexpected interruptions.
- Young/Daly: $\delta = 60$ s, $M = 3$ h → $\tau^* = \sqrt{2 \cdot 60 \cdot 10800} \approx 1138$ s ≈ 19 min; expected lost work
  per failure ≈ $\tau^*/2 \approx 9.5$ min plus restart time.

**Question ideas**
- Compute: checkpoint size and optimal interval for given $N$, bandwidth and MTBF.
- Predict: halving the checkpoint write time — how does $\tau^*$ change (by $1/\sqrt2$)?
- Which is false: "with FSDP you must gather the full model to rank 0 to checkpoint" (false) / ...
- Debug: a 2048-GPU job hangs with all ranks in an all-reduce except one — what happened and how do you detect it automatically?
- Design: what state must be saved so a resumed run is bitwise-identical in data order?

**Pitfalls / verify**
- PaLM rewind/skip numbers and Llama 3 percentages: verify in the cited sections before quoting.
- Young/Daly is a first-order approximation (assumes $\delta \ll M$).

**Figures**
- `checkpoint-interval`: expected wasted fraction vs checkpoint interval for two MTBFs, minima marked at $\sqrt{2\delta M}$.
- `sharded-checkpoint`: ranks each writing their shard + metadata vs gather-to-rank-0, and reloading onto a different mesh.

---

# Part H: Compilation and frameworks

## sys.graph-capture — JIT compilation I: tracing, graph capture, retracing and graph breaks

Level intermediate · prereqs — · W2 · ~9 cards

**Problem.** "Eager Python is flexible but leaves performance on the table. How do JIT compilers turn a Python function into
a graph, what assumptions do they bake in, and why does my jitted function keep recompiling?"

**Refreshers to include inline.** What a CPython frame and bytecode are, and what PEP 523 lets a library intercept (item 6).

**Primary sources**
- JAX "Key concepts" (transformations, tracing, jaxprs), "Just-in-time compilation" (how tracing works with abstract
  values; why not jit everything — `TracerBoolConversionError` on Python control flow; `static_argnums`; caching) —
  `sys_jax_key_concepts.txt`, `sys_jax_jit.txt`; "The Sharp Bits" (pure functions; side effects run only at trace time;
  globals captured at first trace; control flow) — `sys_jax_sharp_bits.txt`.
- TensorFlow "Better performance with tf.function" (tracing, rules of tracing, controlling retracing: input_signature,
  unknown dimensions, reduce_retracing, Python scalars vs tensors; side effects) and "Introduction to graphs" (AutoGraph,
  polymorphism, ConcreteFunction) — `sys_tf_function.txt`, `sys_tf_intro_graphs.txt`.
- Ansel et al. 2024 §2 (prior PyTorch capture: jit.trace, jit.script, Lazy Tensors, torch.fx; comparison with JAX §2.6),
  §3 TorchDynamo (CPython frame evaluation hook, guards, symbolic evaluation, graph breaks, AOTAutograd §3.9) —
  `sys_ansel2024_pytorch2.txt`.
- PyTorch docs: Dynamo core concepts (bytecode tracing, graph breaks, guards, recompilations, dynamic shapes), common graph
  breaks, dealing with recompilations (`cache_size_limit`), dynamic shapes core concepts, `torch.compile` API
  (`fullgraph`, `dynamic`, `mode`) — `sys_pytorch_compile_*.txt`, `sys_pytorch_compile_api.txt`.
- Frostig et al. 2018 (tracing Python to XLA HLO) — `sys_frostig2018_jax.txt`.

**Subtopic map**
1. *Eager vs graph*: eager runs op by op (easy debugging, Python overhead per op, no cross-op optimisation); a graph
   enables fusion, memory planning, scheduling, and distribution. Define-then-run (TF1) vs define-by-run (PyTorch).
2. *Capture strategies*: tracing with example inputs (record executed ops: torch.jit.trace, tf.function, jax.jit);
   source/AST conversion (torch.jit.script, AutoGraph); lazy tensors (record until a value is needed); bytecode-level
   capture (TorchDynamo).
3. *What tracing bakes in*: only the branch taken; Python side effects (print, appending to lists, RNG from Python)
   happen once at trace time; global variables read at trace time; shapes and dtypes become part of the compiled program.
4. *JAX specifics*: tracers carry shape/dtype (abstract values), not data; `if x > 0` on a traced value raises; use
   `jnp.where`, `lax.cond`, `lax.while_loop`, `lax.fori_loop`, `lax.scan`; `static_argnums` makes an argument part of the
   cache key (recompile per distinct value); the cache key = function + shapes/dtypes + static values + pytree structure.
5. *Why does my jitted function retrace?* New shapes (variable sequence lengths → bucket/pad), new dtypes, passing Python
   scalars (TF: each new Python value retraces; pass tensors), changing static args, changing pytree structure, new
   closures/lambdas each call. Detecting retraces (logging in the Python body, JAX/TF tracing counters, `TORCH_LOGS=recompiles`).
6. *TorchDynamo*: hooks CPython frame evaluation (PEP 523), symbolically executes bytecode, extracts an FX graph of tensor
   ops, emits guards (checks on shapes, dtypes, Python values, globals) and rewritten bytecode. Graph breaks on untraceable
   code (data-dependent control flow on tensor values, `.item()`, printing, unsupported libraries): the graph is split and
   Python runs in between (lost fusion, extra overhead). `fullgraph=True` turns breaks into errors.
7. *Recompilation and dynamic shapes*: a failed guard triggers recompilation; after a shape-related recompile, automatic
   dynamic shapes generalise the dimension (symbolic shapes; 0/1 specialisation); `cache_size_limit` caps recompiles,
   after which the function runs eagerly; ints are specialised by default.
8. *AOTAutograd*: traces forward and backward together ahead of time so the backward graph is also compiled; partitions
   the joint graph (min-cut) to choose what to save vs recompute.
9. *Costs*: compile time, warm-up steps, memory for multiple cached graphs, debugging difficulty.
10. *Probes*: "why does my jitted function retrace", "trace vs script", "what is a graph break and why is it bad",
    "how does torch.compile handle Python control flow", "why are Python side effects dangerous under jit".

**Worked computations / examples (writers verify by running)**
- JAX: a jitted function with a `print` inside called with shapes (3,), (3,), (4,) prints twice; `jax.debug.print` prints every call.
- TF: `tf.function` called with Python ints 1, 2, 3 traces three times; with `tf.constant` values once.
- PyTorch: a function branching on `x.sum() > 0` under `torch.compile` produces a graph break (verify with `torch._dynamo.explain`).

**Question ideas**
- Predict output: code with a Python-side counter incremented inside a jitted function, called 5 times.
- Debug: training step time spikes every few hundred steps with variable-length batches — cause and two fixes.
- Which is false: "jax.jit traces with concrete values" (false) / "tf.function retraces for new Python scalar arguments" / ...
- Compare: graph break (PyTorch) vs tracing error (JAX) on data-dependent control flow — different design choices.

**Pitfalls / verify**
- API details change (e.g. Dynamo config names); cite the cached doc version and avoid exact default limits unless checked.
- torch.jit.script/trace (TorchScript) is in maintenance; describe as historical.

**Figures**
- `capture-pipeline`: diagram Python function → tracer/Dynamo → graph (jaxpr / FX) → compiler (XLA / Inductor) → kernels, with guards/cache.
- `retrace-timeline`: step times over a run with spikes at new sequence lengths, before and after padding to buckets.

---

## sys.fusion-codegen — JIT compilation II: operator fusion, XLA, Inductor/Triton, CUDA graphs

Level intermediate · prereqs sys.graph-capture, sys.roofline · W2 · ~8 cards

**Problem.** "Once we have a graph, what does the compiler actually do that makes it faster?" Mainly: fuse memory-bound ops,
plan memory, pick kernels, and remove launch overhead.

**Primary sources**
- Horace He 2022 "Bandwidth" (operator fusion; `x.cos().cos()` example; why fused activation functions are as cheap as a
  copy) — `sys_he2022_brrr.txt`.
- Ansel et al. 2024 §4 TorchInductor (decompositions §4.1, loop-level IR, scheduling and fusion §4.4, Triton and
  C++/OpenMP codegen, wrapper codegen §4.7, related compilers §4.8), §3.9 AOTAutograd min-cut — `sys_ansel2024_pytorch2.txt`.
- Tillet et al. 2019 (Triton block-level programming model) — `sys_tillet2019_triton.txt`.
- PyTorch blog "Accelerating PyTorch with CUDA Graphs" (CPU launch overhead, capture/replay, static inputs, graph-safe
  constraints, NCCL support) — `sys_pytorch_cuda_graphs_blog.txt`; `torch.compile` modes (`reduce-overhead` uses CUDA
  graphs; `max-autotune`) — `sys_pytorch_compile_api.txt`.
- Xu et al. 2021 GSPMD (XLA sharding propagation) — `sys_xu2021_gspmd.txt`; Scaling Book Part 9 "A Thousand-Foot View of the
  TPU Software Stack" (JAX → HLO → XLA) — `sys_scalingbook_profiling.txt`.
- FlashAttention as a hand-fused kernel — shared `llm_dao2022_flashattention.txt`.

**Subtopic map**
1. *Why fusion*: each unfused elementwise op reads inputs from and writes outputs to HBM; a chain of $k$ ops costs $\approx 2k$
   passes vs 2 fused (derive with the roofline). Intermediates stay in registers/SRAM.
2. *Fusion types*: pointwise–pointwise; pointwise into reductions (softmax, LayerNorm); epilogue fusion into matmuls
   (bias + activation); hand-written fusions for complex patterns (FlashAttention, fused optimizers, fused cross-entropy).
3. *What fusion cannot do*: change a compute-bound matmul's cost; fuse across communication or graph breaks.
4. *XLA*: HLO IR, fusion passes, layout assignment, buffer assignment (static memory planning), SPMD partitioning via
   sharding annotations (GSPMD); requires static shapes → padding/bucketing; compile time.
5. *TorchInductor*: decomposes ops to a smaller set, lowers to a loop-level IR, schedules fusion groups, generates Triton
   kernels on GPU (C++/OpenMP on CPU), uses vendor libraries or autotuned Triton templates for matmuls (`max-autotune`).
6. *Triton*: programs operate on blocks/tiles; the compiler handles shared-memory staging, coalescing and vectorisation;
   why it made generated kernels practical.
7. *CUDA graphs*: record a sequence of kernel launches once, replay with one launch → removes per-kernel CPU overhead
   (overhead-bound regimes: small batches, inference, many small kernels). Constraints: static shapes and control flow,
   fixed memory addresses (copy into static input buffers), no host syncs inside; private memory pool.
8. *Compiler-chosen recomputation*: the joint forward/backward partitioner recomputes cheap ops to save memory (link
   `sys.activation-checkpointing`).
9. *Probes*: "what does operator fusion buy you and for which ops", "why does torch.compile help a GELU-heavy model",
   "when do CUDA graphs help", "what is Triton".

**Worked computations**
- Unfused vs fused `y = gelu(x * a + b)` on $10^8$ bf16 elements: count HBM passes (unfused: 3 ops × read+write; fused: 1
  read + 1 write, plus parameters) and convert to time at 3.35 TB/s (writers: compute).
- CUDA-graph benefit estimate for a step with many tiny kernels (illustrative; label as such).

**Question ideas**
- Predict: speedup from fusing a chain of 5 elementwise ops vs fusing a matmul with its bias add.
- Which is false: "operator fusion reduces the FLOPs of the fused ops" (false) / ...
- Debug: enabling `mode="reduce-overhead"` crashes with variable batch sizes — why?
- Compare: XLA's static-shape compilation vs Inductor with dynamic shapes.

**Pitfalls / verify**
- Speedup numbers depend on hardware; derive bounds from the roofline instead of quoting benchmark numbers.

**Figures**
- `fusion-memory-traffic`: diagram of HBM ↔ SRAM traffic for three unfused ops vs one fused kernel.
- `cuda-graphs`: CPU launch timeline with gaps vs a single graph launch.

---

## sys.jax-model — The JAX programming model

Level intermediate · prereqs sys.graph-capture · W2 · ~9 cards

**Problem.** "Why does JAX make you pass random keys and parameters explicitly, return new arrays instead of mutating, and
write `lax.cond` instead of `if`? What does this buy for compilation and parallelism?"

**Primary sources**
- JAX "Key concepts" (transformations, tracing, jaxprs, pytrees, API layering NumPy/lax/XLA) — `sys_jax_key_concepts.txt`.
- "Pseudorandom numbers" (explicit keys, `split`, lack of sequential equivalence, why not a global state) — `sys_jax_random.txt`.
- "Pytrees" (leaves vs nodes, `jax.tree.map`, gotchas) — `sys_jax_pytrees.txt`.
- "The Sharp Bits" (purity, in-place updates via `.at[].set`, out-of-bounds indexing does not raise, control flow,
  dynamic shapes, NaN debugging, 64-bit off by default) — `sys_jax_sharp_bits.txt`.
- "Distributed arrays and automatic parallelization" (Mesh, NamedSharding/PartitionSpec, explicit vs auto sharding,
  `shard_map` manual mode) — `sys_jax_parallel.txt`; Scaling Book Part 10 — `sys_scalingbook_jax_stuff.txt`.
- `jax.checkpoint` — `sys_jax_checkpoint.txt`; Frostig et al. 2018 — `sys_frostig2018_jax.txt`.

**Subtopic map**
1. *Functional core*: functions of arrays to arrays; arrays immutable; updates return new arrays (`x.at[i].set(v)`),
   which XLA turns into in-place updates when safe (buffer donation).
2. *Transformations*: `grad` (returns a function; `value_and_grad`, `has_aux`; higher-order derivatives by composition),
   `jit`, `vmap` (batching by rule, `in_axes`), `jvp`/`vjp`, `checkpoint`; composition order matters only for semantics
   you can reason about (e.g. `jit(vmap(grad(f)))`). Jaxprs as the IR they transform.
3. *Why purity*: transformations re-trace the function; side effects and hidden state would execute at trace time or
   break batching/differentiation.
4. *Explicit PRNG*: counter-based generator with keys; `split` produces independent streams; same key → same numbers;
   reproducible and parallel-safe (no ordering dependence across devices); the "no sequential equivalence" note; dropout
   needs a key per step and per layer.
5. *Pytrees*: params, optimizer states and batches as nested containers; `tree.map` to apply updates; `None` handling;
   custom nodes.
6. *A training step* as a pure function `(params, opt_state, batch, key) -> (params, opt_state, metrics)`, jitted once.
7. *Parallelism*: a `Mesh` of devices with named axes; arrays carry a `NamedSharding`; `jit` with sharded inputs lets
   XLA/GSPMD propagate shardings and insert collectives (auto mode); explicit sharding mode makes shardings part of types;
   `shard_map` gives per-device code with explicit `psum`/`all_gather` (manual mode); `pmap` is legacy.
8. *Sharp bits*: out-of-bounds indexing clamps/drops instead of raising; no data-dependent shapes under jit; float64
   disabled by default; NaN debugging flag; asynchronous dispatch (`block_until_ready` for timing).
9. *Probes*: "why explicit keys", "what does vmap do", "how do you write data parallelism in JAX", "why can't I mutate
   arrays", "what happens to Python control flow under jit".

**Worked examples (writers run them if JAX is available; otherwise describe from the docs)**
- `split` a key into two and show that drawing from the same key twice repeats the numbers.
- `vmap` of a per-example loss vs a manual batch dimension: same result, different code.
- Data-parallel step with a 1-D mesh `('data',)`: batch sharded on axis 0, params replicated, gradient all-reduce inserted by the compiler.

**Question ideas**
- Predict: output of code that reuses a PRNG key for two dropout layers.
- Which is false: "jax.grad mutates parameters in place" (false) / "vmap adds a batch dimension without Python loops" / ...
- Debug: a jitted step silently returns wrong values when indexing past the end of an array — why no error?
- Predict: a data-parallel step written with `jit` + a batch-sharded input on a 1-D mesh — which collective does the compiler
  insert for the gradient, and where would you write it explicitly with `shard_map`?

**Pitfalls / verify**
- JAX APIs move (e.g. `jax.tree_util` → `jax.tree`, sharding modes); cite the cached docs.

**Figures**
- `transform-stack`: function → jaxpr → grad/vmap/jit transformations → XLA, as a layered diagram.
- `prng-split-tree`: key splitting tree for a training loop (per step, per layer).

---

## sys.frameworks — JAX vs PyTorch vs TensorFlow

Level core · prereqs sys.graph-capture · W2 · ~7 cards

**Problem.** "Interviewers ask which framework you'd pick and why, or what really differs under the hood. Give a precise
comparison of execution model, autodiff, state, randomness, compilation and distribution."

**Primary sources**
- Abadi et al. 2016 (TensorFlow: dataflow graphs, deferred execution, parameter servers, control flow in graphs) — `sys_abadi2016_tensorflow.txt`.
- TF guides (TF2 eager by default; `tf.function` tracing; AutoGraph) — `sys_tf_function.txt`, `sys_tf_intro_graphs.txt`.
- Ansel et al. 2024 §1–2 (eager PyTorch and the history of graph capture; §2.6 comparison to JAX) — `sys_ansel2024_pytorch2.txt`.
- Frostig et al. 2018 and JAX key concepts — `sys_frostig2018_jax.txt`, `sys_jax_key_concepts.txt`.
- Li et al. 2020 / PyTorch DTensor docs and JAX parallel docs for distributed APIs — `sys_li2020_pytorch_ddp.txt`,
  `sys_pytorch_dtensor.txt`, `sys_jax_parallel.txt`; Baydin et al. 2018 (reverse-mode AD background) — shared `papers/baydin2018_autodiff.txt`.

**Subtopic map**
1. *Execution models*: PyTorch define-by-run (eager) plus `torch.compile`; JAX NumPy-like API plus `jit` to XLA; TensorFlow
   in **one card**: TF1 define-then-run in a sentence of history, then TF2 eager + `tf.function` retracing as a contrast to
   `jax.jit` and `torch.compile` (D5: TF1 sessions/feed_dict/parameter servers are low interview value in 2026).
2. *Autodiff*: PyTorch's dynamic tape built during the forward, `.backward()` accumulates into `.grad` (hence
   `zero_grad`); TF2 `GradientTape`; JAX `grad` as a function transformation (no tape object; higher-order and per-example
   gradients via composition with `vmap`).
3. *State*: `nn.Module` objects with mutable parameters and buffers vs `tf.Variable` vs explicit parameter pytrees (Flax/
   Equinox wrap this); optimizer state handling (`torch.optim` vs optax pure functions).
4. *Randomness*: global stateful RNG (PyTorch, TF ops) vs explicit keys (JAX).
5. *Compilation*: XLA (JAX always under jit; TF with `jit_compile`), Inductor/Triton (PyTorch); static-shape vs dynamic-shape
   friendliness; TPU support (JAX/TF via XLA; PyTorch via PyTorch/XLA).
6. *Distribution*: PyTorch DDP/FSDP/DTensor/DeviceMesh (explicit wrappers), JAX sharding + GSPMD + `shard_map` (compiler-driven),
   TF `tf.distribute` strategies.
6b. *Per-example gradients and functional transforms across frameworks*: `jax.vmap(jax.grad(f))` vs `torch.func.vmap(torch.func.grad(f))`;
   what it computes and its memory cost.
6c. *torch.compile with DDP/FSDP*: DDPOptimizer splits graphs at bucket boundaries so communication can still overlap
   (`sys_pytorch_ddp_note.txt` "TorchDynamo DDPOptimizer").
7. *Ecosystem and trade-offs*: debugging ease, research flexibility, production/serving, hardware; avoid tribal claims.
8. *Typical questions*: "what does `loss.backward()` do vs `jax.grad`", "why does PyTorch need `zero_grad`", "tf.function vs
   torch.compile vs jax.jit", "how would you compute per-example gradients in each".

**Worked examples**
- Same tiny linear-regression step written in the three frameworks (short snippets, ≤ 8 lines each), highlighting where
  state, gradients and randomness live (writers: keep code correct for current APIs).

**Question ideas**
- Predict: calling `loss.backward()` twice without `zero_grad()` in PyTorch vs calling `jax.grad(loss)(params, batch)` twice —
  what does each produce?
- Predict: what does `vmap(grad(f))` compute for a batch of 64 examples, and how does its memory compare with one batched gradient?
- Compare: per-example gradients in JAX (`vmap(grad)`) vs PyTorch (`torch.func.vmap` + `grad`).
- Explain: why TF1 needed `tf.cond`/`tf.while_loop` while PyTorch could use Python `if`.

**Pitfalls / verify**
- Keep claims about current APIs to what the cached docs show; frameworks evolve quickly.

**Figures**
- None as figures: the comparison (rows: execution, autodiff, state, RNG, compiler, distribution; columns: PyTorch, JAX, TF2)
  is a markdown table in a card.

---

# Appendix: figure modules and shared helpers

Figures go in `tools/figures/<topic-id>.py` (see writing-brief "Figures"). Several figures share machinery; writers may
copy helpers between modules (each module must be self-contained):
- Gantt/timeline helper (pipeline schedules, DDP/FSDP overlap, CUDA graphs, trace sketches): rows = devices or streams,
  coloured blocks with labels, greyed idle time. Used by sys.ddp, sys.fsdp, sys.pipeline-parallel, sys.pipeline-schedules,
  sys.profiling, sys.fusion-codegen, sys.gradient-accumulation, sys.activation-checkpointing.
- Rank-grid helper (collectives before/after, ZeRO memory bars, device meshes): small boxes per rank with coloured shards.
  Used by sys.collectives, sys.collective-algorithms, sys.zero, sys.fsdp, sys.parallelism-composition, sys.reliability-checkpointing.
- Bit-layout helper (sys.fp-formats, sys.fp8-training).
- Roofline helper (sys.roofline, sys.fp8-training, sys.fusion-codegen, sys.ddp `dp-roofline`).
- Block-diagram helper (sys.gpu-basics anatomy and memory hierarchy).
All numbers in figures must come from the formulas in this plan (computed with numpy in the module), not hand-typed.

# Open items for writers

- Every worked number above was recomputed for this plan (PyTorch 2.1 / numpy); recompute again in the lesson.
- Items marked "verify" must be checked against the named section before they appear in content; drop them if unverifiable.
- Thakur et al. 2005 (MPICH collective algorithms) could not be cached (bot challenge); do not cite it. Patarasuk & Yuan 2009
  (ring optimality) and NVIDIA's 2019 NCCL double-binary-tree blog are cached instead; Chan et al. 2007 was not freely available.
- JAX is not installed locally (only PyTorch 2.1 and numpy); JAX examples must be checked against the docs text or run elsewhere.
