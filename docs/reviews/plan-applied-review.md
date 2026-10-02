# Review: content plan for Applied ML & Systems (`docs/content-plan-applied.md`)

Reviewer: adversarial plan review, 2026-10-02. I read the plan against `writing-brief.md`, `content-plan-tuning.md`,
`content-plan.md` (fund), `content-plan-scalingbook.md` and the source cache (`INDEX-sys.md` plus the cached texts).
I recomputed every worked number in the plan with Python (PyTorch 2.1.1 / numpy). The scripts are in the session scratchpad
(`fp.py`, `err.py`, `mem.py`, `comm.py`). I grepped 25+ cited sections in the cache (listed in §F).

**Overall.** This is a strong plan. The sources are mostly primary. The subtopic maps are deep, and almost all of the
~90 worked numbers are right. The problems are concentrated in a few places:
- one modelling error about cross-node bandwidth, which reverses a W1 conclusion;
- the wave/order structure: W1 parallelism lessons lean on W2 roofline/FLOP concepts;
- large unacknowledged overlap with `fund.*`;
- three coverage gaps a frontier-lab systems interviewer would hit: MoE/expert parallelism, reliability/checkpointing
  at scale, and GPU architecture basics;
- the LLM plan, which all `llm.*` cross-references depend on, does not exist yet.

Findings are listed in priority order. Tags: **blocking** = must fix before writers start the affected lesson; **major** =
fix before that lesson's wave; **minor** = writer-level fix or note.

---

## A. Blocking

### A1. Cross-node all-reduce is modelled as a flat ring at 50 GB/s per GPU. This is 8× too pessimistic and reverses the DDP conclusion. **blocking**
**Where:** `sys.ddp` worked computations ("Same model across 64 GPUs on IB (50 GB/s per GPU): ≈ 102 ms per all-reduce →
exceeds the backward → exposed"); `sys.collective-algorithms` worked computations ("Same 14 GB across 64 GPUs on IB at
50 GB/s/GPU (flat ring): ≈ 0.55 s"); the hardware-constants table row "50 GB/s per GPU" used as the collective bandwidth.

**Problem:**
- The plan's own source models cross-node AG/RS as bytes / W_node-egress = bytes / 400 GB/s
  (`sys_scalingbook_gpus.txt` l.199–200, takeaway l.217). An all-reduce costs twice that without SHARP.
- In practice NCCL runs several rings in parallel, one per NIC (or a hierarchical RS → inter-node AR on 1/8 of the data → AG),
  so every NIC on the node carries traffic at once.
- The same lesson quotes the 2475-tokens/GPU threshold, which is computed with W = 400 GB/s. So the plan contradicts itself:
  8192 tokens/GPU is above 2475, yet the plan calls the communication "exposed".

**Recomputed:**
- 1.3B params, bf16 grads (2.6 GB), 64 GPUs = 8 nodes, node-egress model: 2 × 2.6e9 / 400e9 ≈ **13 ms**.
- An explicit hierarchical schedule (intra-node RS at 450 GB/s + inter-node ring AR on S/8 at 50 GB/s/GPU + intra-node
  AG) gives ≈ **21 ms**.
- Both are well under the 43 ms backward, so the communication is **hidden**, not exposed.
- 14 GB over 64 GPUs: 2 × (63/64) × 14e9 / 400e9 ≈ **69 ms**, not 0.55 s.

**Fix:**
- Replace both examples with the node-egress (or hierarchical) model.
- State the rule: "cross-node collectives use all 8 NICs, so per node the bandwidth is 400 GB/s. 50 GB/s per GPU is
  the right number only when each GPU's whole buffer must leave through its own NIC, as in all-to-all or a TP group with one
  GPU per node."
- To keep an "exposed" example, pick a smaller per-GPU batch (e.g. 1024 tokens/GPU, below 2475) or fp32 gradients
  without overlap.
- Add the flat-ring-per-GPU figure only as the *naive* model, and explain why it is wrong. That makes a good MCQ
  distractor.
- Also add the all-to-all exception (Scaling Book l.208–209: cross-node all-to-all really does see ≈ 50 GB/s per GPU).
  It is the reason EP is kept inside the node.

---

## B. Major

### B1. W1 parallelism lessons depend on W2 performance concepts. Promote `sys.roofline` and `sys.flops-mfu` to W1 and move Part F before Part E. **major**
**Where:** topic table and wave-2 order.

**Problem:**
- `sys.ddp` (W1) uses the compute/comm roofline (per-GPU tokens > C/W) and backward FLOPs = 4N per token.
- `sys.tensor-parallel` (W1) uses forward FLOPs 24bsh² and the Scaling Book rule t < F·W/C.
- `sys.zero` item 6 uses the same B/P > C/W condition.
- `sys.pipeline-parallel` relies on step-time reasoning.
- These concepts are taught only in `sys.roofline` / `sys.flops-mfu` (W2, orders 220/230, *after* all parallelism lessons).
- Back-of-envelope FLOP/MFU/roofline questions are also the single most common systems question type at frontier labs
  ("is this memory-bound?", "how long will this run take?", "what MFU is that?").

**Fix:**
- Make `sys.roofline` and `sys.flops-mfu` W1.
- Demote `sys.vanishing-exploding` and `sys.gradient-clipping` to W2; most of their basic content is already in `fund.*`, see B3.
- Renumber Part F to orders 85–95 (after memory-anatomy) or 115–118 (before collectives).
- Add `sys.roofline` to the prereqs of `sys.ddp`, `sys.tensor-parallel` and `sys.fp8-training`, and `sys.flops-mfu` to the
  prereqs of `sys.activation-checkpointing` (HFU/MFU).

### B2. The ring all-reduce derivation (a top-5 interview item) sits in a W2 lesson that W1 lessons already use. **major**
**Where:** `sys.collectives` (W1) vs `sys.collective-algorithms` (W2).

**Problem:** `sys.ddp` and `sys.tensor-parallel` (W1) compute times with 2(P−1)/P·S/W. The ring derivation that justifies
it is taught only in W2. "Derive ring all-reduce cost" is a classic interview probe.

**Fix:**
- Move "ring all-reduce derivation + time formula" (collective-algorithms item 3) into `sys.collectives`.
- Move algbw/busbw (collectives item 7) the other way to make room.
- Keep collective-algorithms for α–β, trees, hierarchical/SHARP, interconnects and overlap.
- Move the `ring-allreduce` figure with it.

### B3. Large unacknowledged overlap with `fund.*`. `sys.vanishing-exploding` largely duplicates three fund lessons, including two identical figures. **major**
**Where:** Boundaries section ("fund: initialisation, normalisation layers and Adam are taught there"); `sys.vanishing-exploding`;
`sys.gradient-clipping`; `sys.activation-checkpointing` item 2; `sys.stable-numerics` items 1–3, 6; `sys.mixed-precision` / `sys.gradient-accumulation`.

**Problem.** `content-plan.md` already plans:
- `fund.rnn` (order 550): Jacobian product, Pascanu σ_max < 1/γ conditions, the eigenvalue picture, norm vs value
  clipping, cliffs, truncated BPTT, orthogonal init. Its figures `gradient-norm-vs-lag` (σ_max 0.9/1.0/1.1) and
  `clipping-cliff` are the **same figures** as sys `rnn-eigen` and `cliff`.
- `fund.initialization` (order 500): variance propagation, with the same "0.9⁵⁰, 1.1⁵⁰" calculation.
- `fund.activations`: the 0.25^L sigmoid saturation.
- `fund.backprop` item 7 and the figure `fund.backprop/checkpointing`: Chen's √L checkpointing.
- `fund.training-loop` items 7–9: AMP/GradScaler, the HF gradient-accumulation bug, reproducibility and checkpoint resume.
- `fund.logistic-regression` item 7 and the fund variance lesson: LSE and Welford.

**Fix:**
- Extend the Boundaries section with the exact ids (`fund.rnn`, `fund.lstm-gru`, `fund.initialization`, `fund.activations`,
  `fund.backprop`, `fund.training-loop`, `fund.normalization`, `fund.logistic-regression`).
- **Re-scope `sys.vanishing-exploding` to the deep-network/transformer view:**
  - one recap card of the Jacobian-product argument, linking `fund.rnn` / `fund.initialization`;
  - then spend the cards on what fund does not cover: the residual-stream Jacobian (I + ∂F/∂h) derived for L blocks,
    pre-LN vs post-LN gradient scale (Xiong 2020), depth-scaled residual init, per-layer gradient-norm diagnostics at scale;
  - and "explosion in practice": attention-logit growth, fp16 overflow masquerading as explosion, loss spikes, linking
    stability-tricks.
- Drop `rnn-eigen` and `cliff`, or reuse the fund figures by reference.
- Add `fund.rnn` and `fund.initialization` as prereqs.
- `sys.gradient-clipping`:
  - cut the definition and cliff material to a single recap card;
  - keep the systems content (AMP ordering, accumulation, sharded global norm under FSDP/TP/PP, Adam interaction, AGC,
    non-finite handling), which is genuinely new.
- `sys.activation-checkpointing` item 2: say the √n result is recapped from `fund.backprop` and go faster to selective/policy
  recompute.

### B4. Coverage gap: MoE / expert-parallel *systems*. The owning plan does not exist. **major**
**Where:** Boundaries ("`content-plan-llms.md` … MoE routing and expert parallelism details"); `sys.parallelism-composition` item 6.

**Problem:**
- `docs/content-plan-llms.md` does not exist (the docs directory has only content-plan.md, -applied, -tuning, -scalingbook),
  so nothing owns EP.
- EP systems questions are now standard at frontier labs:
  - all-to-all dispatch/combine cost and why it is 8× worse across nodes (Scaling Book l.208–209);
  - capacity factor and token dropping;
  - load imbalance as a straggler problem;
  - grouped GEMMs;
  - EP×DP meshes;
  - node-limited routing (DeepSeek-V3);
  - overlapping all-to-all with compute (DualPipe);
  - the E/k multiplier on the DP batch floor.
- One bullet in a W3 lesson is not enough.

**Fix:** Add `sys.expert-parallel` (W3, ~8 cards, prereqs `sys.collectives`, `sys.parallelism-composition`).
- Sources (all cached): Scaling Book gpus "Expert parallelism" (l.275–290), Playbook "Expert parallelism"
  (l.3227–3263), DeepSeek-V3 §3.2 (`llm_deepseek2024_v3.txt`), DeepSeekMoE (`llm_dai2024_deepseekmoe.txt`).
- Worked number: the all-to-all time for one MoE layer's dispatch across 1, 2 and 8 nodes.
- Alternatively, record in this plan that the LLM plan must own it, with the content above, and add a blocking TODO until
  that plan exists.

### B5. Coverage gap: reliability, distributed checkpointing and fault tolerance at scale. **major**
**Where:** Boundaries ("there is no separate checkpoint-to-disk lesson"); `sys.ddp` item 10; `sys.fsdp` item 9.

**Problem:**
- At 16k-GPU scale this is a first-class interview topic:
  - failure rates (Llama 3 §3.3.4 reports hundreds of interruptions over the run; `llm_dubey2024_llama3.txt` is cached);
  - checkpoint size and bandwidth (16N bytes for a 70B model = 1.12 TB);
  - sharded/distributed checkpoints written by all ranks;
  - asynchronous checkpointing;
  - resharding on restore when the mesh changes;
  - checkpoint frequency vs failure rate;
  - NCCL timeouts and hangs, straggler detection, silent data corruption, rollback after loss spikes;
  - deterministic, resumable data loaders.
- The tuning plan's `sys.checkpointing†` rows ("resilience to preemption, keep N best checkpoints") are mapped to
  `sys.ddp` item 10, which covers none of this.
- The guidance "save checkpoints from rank 0 only" is **wrong for FSDP/ZeRO-3**: either every rank writes its shard, or a full
  state dict is gathered to rank 0, which costs memory and time.

**Fix:** Add `sys.reliability-checkpointing` (W3, ~8 cards).
- Worked computations:
  - checkpoint bytes for 70B (1.12 TB of model states);
  - write time at an assumed aggregate bandwidth;
  - an optimal checkpoint interval (Young/Daly approximation √(2·C·MTBF), labelled as a standard approximation;
    cache a source or give a one-line derivation);
  - expected lost work per failure.
- Qualify DDP item 10 as "DDP only". In FSDP item 9, say that sharded state dicts are written by all ranks.

### B6. Coverage gap: GPU execution model and memory hierarchy are never taught, but many lessons assume them. **major**
**Where:** `sys.roofline` item 7 (SRAM/tiling), `sys.fusion-codegen` items 1, 6, 7 (registers/SRAM, Triton tiles, CUDA graphs),
`sys.mixed-precision` items 1 and 9 (tensor cores, multiple-of-8), `sys.profiling` items 2 and 4 (streams, occupancy,
Nsight Compute), `sys.collective-algorithms` item 7 ("comm kernels consume SMs").

**Problem:**
- A reader "who may never have thought about bits, collectives or sharding" also does not know these words:
  - SMs, warps/threads;
  - HBM vs L2 vs shared memory/registers;
  - tensor cores;
  - kernels and launches;
  - CUDA streams and asynchronous execution.
- The brief requires defining every term before use. No lesson owns this material.
- `sb.gpu-chip` exists, but it is in a different series and uses the book's framing.

**Fix:**
- Add `sys.gpu-basics` (W2, ~7 cards, order before `sys.roofline`). Sources (all cached): Playbook "GPU primer & kernels"
  (l.4111–4485), Scaling Book gpus "What is a GPU?" / spec tables, NVIDIA *GPU Performance Background* §2–3,
  PyTorch CUDA semantics "Asynchronous execution".
- Or at minimum add a two-card inline refresher at the start of `sys.roofline`, and list it as a prereq of fusion-codegen,
  profiling and fp8-training.

### B7. The ZeRO-3 communication claim clashes between two cited sources, and the plan never reconciles them. **major**
**Where:** `sys.zero` primary sources ("Scaling Book Part 5 FSDP (comm cost equals DP's …)") vs subtopic 5 ("ZeRO-3 … 3N, i.e. 1.5× DDP").

**Problem:**
- The Scaling Book says ZeRO-1/2/3 "all have the same communication cost" (`sys_scalingbook_training.txt` l.132–134).
  It argues that the extra forward all-gather is in the same proportion as the forward FLOPs, so the *comm/compute ratio*
  (and hence the roofline threshold) is unchanged.
- ZeRO §7 counts *volume*: 3Ψ vs 2Ψ.
- A writer citing both will produce contradictory cards.

**Fix:** Add an explicit reconciliation card.
- Per step, the volume is 1.5×.
- Per phase:
  - forward: AG of N·2 bytes against 2N·B FLOPs;
  - backward: AG + RS of 2·N·2 bytes against 4N·B FLOPs.
  - Both phases have ratio 1/B (bytes per FLOP, ×C/W), so the threshold B/P > C/W is the same as DDP's.
- Hence "1.5× more bytes, same compute-bound condition". Make this an MCQ.

### B8. `sys.stability-tricks` z-loss worked example states a wrong mechanism. **major** (W3, but conceptually wrong)
**Where:** `sys.stability-tricks` worked computations: "extra gradient 2λ log Z·softmax ≈ [3.75e-4, 5.07e-5] — it pushes
all logits down together, which lowers log Z without changing the softmax."

**Problem:**
- The gradient is proportional to softmax(z), not to the all-ones vector. A descent step subtracts different amounts from
  each logit (here 7.4× more from the top logit), so the softmax *does* change: it becomes slightly flatter.
- Numerically: one exaggerated step z − 100·g takes softmax from [0.8808, 0.1192] to [0.8774, 0.1226].
- Decomposed: the component along 1/√2·(1, 1) is [2.13e-4, 2.13e-4] (a pure shift, which lowers log Z), and the
  orthogonal part is [1.62e-4, −1.62e-4] (which reduces the logit gap).

**Fix:**
- Rewrite as "most of the gradient is a uniform downward shift that lowers log Z without affecting the CE loss. The rest
  slightly shrinks the top logit's margin, which is a small entropy-raising regulariser. The CE term dominates that part."
- Add the decomposition as a "derivation step" MCQ.

### B9. The `llm.*` cross-references are unresolvable, and the overlaps with LLM/Scaling Book lessons need explicit owners. **major**
**Where:** Boundaries; `sys.stability-tricks`; `sys.roofline` item 5; `sys.flops-mfu` item 7; `sys.memory-anatomy` (KV cache MCQ);
`sys.stable-numerics` item 8 (online softmax → FlashAttention).

**Problem:**
- `content-plan-llms.md` is referenced by `writing-brief.md` and by this plan, but it does not exist.
  `llm.training-stability` is also referenced by the tuning plan (Part A 340 card 6 and Part B).
- Likely overlaps once the LLM plan is written:
  - `llm.training-stability` (qk-norm, z-loss, soft-capping, small ε) vs `sys.stability-tricks`, which is almost entirely
    those topics.
  - `llm.inference` (KV cache, decode roofline) vs `sys.roofline` items 5–6 and `sys.flops-mfu` item 7.
  - `llm.pretraining` (MFU, 6ND, compute budgets) vs `sys.flops-mfu`.
- Quantization (int8/int4 weight-only, LLM.int8 outliers, QLoRA NF4) is not owned anywhere. Interviewers ask it right after
  fp formats. The cache has `llm_dettmers2022_int8.txt` and `llm_dettmers2023_qlora.txt`.
- Separately, the `sb.*` series repeats roofline, collectives, FSDP, TP, pipelining and profiling in the book's frame.

**Fix:**
- (a) Mark every `llm.*` id in this plan as provisional (†), as the tuning plan does, and add an "owner TBD" list:
  EP systems, quantization, KV-cache memory, training stability.
- (b) Decide now that `sys.stability-tricks` keeps only the numerics: why large log Z costs bf16 resolution (with a number:
  bf16 spacing at |z| = 64 is 0.5), clamp vs tanh gradient, and ε representability. Merge it into `sys.stable-numerics` as
  1–2 cards if the LLM plan takes the rest. Drop item 4 (Adam ε), which is already covered by `sys.tuning-pipeline` card 2
  and the `fund.adam` insertion row.
- (c) Either add `sys.quantization` (W3: absmax/zero-point int8, per-channel/group scales, outliers, weight-only vs W8A8,
  NF4) or assign it to the LLM plan explicitly.
- (d) Add "see also `sb.roofline-matmul` / `sb.collective-costs` / `sb.fsdp` / `sb.tensor-parallelism`" lines, and make
  sure the numbers agree with the book. A1 currently contradicts it.

### B10. Missing the fp32 vs bf16 gradient convention in DDP worked examples. Default PyTorch AMP all-reduces fp32 grads. **major**
**Where:** `sys.ddp` worked computations ("1.3B params, bf16 grads (2.6 GB)"), `sys.collective-algorithms` ("7B model, bf16 grads (14 GB)"),
and the DP threshold of 2200/2475 tokens.

**Problem:**
- With standard `torch.autocast` + DDP, parameters and therefore `.grad` are fp32, so DDP all-reduces fp32 gradients
  unless a bf16 compression comm hook is registered or the model is cast to bf16.
- That doubles comm time and doubles the compute-bound threshold (C/W counts 2 bytes per gradient element; fp32 gives
  ≈ 4400 / 4950 tokens).
- The plan uses bf16 throughout without saying so.

**Fix:**
- State the assumption explicitly in each example ("bf16 gradients, e.g. bf16 params or a bf16 comm hook").
- Add the fp32 variant as a "predict" question.
- In `sys.mixed-precision` item 7, add "DDP + autocast communicates fp32 grads by default".

### B11. Prerequisite chain gaps beyond B1. **major**
**Where:** topic table prereqs.
- `sys.ddp` → add `sys.memory-anatomy` (item 9 uses 16 bytes/param).
- `sys.tensor-parallel` → add `sys.memory-anatomy` (item 7 uses the 10sbh/24sbh activation terms). Add an inline LSE
  refresher for vocab-parallel CE (item 5), because `sys.stable-numerics` is W2.
- `sys.pipeline-parallel` → add `sys.collectives` (send/recv) and `sys.ddp` (item 8 DP/ZeRO-1 interplay). Add an inline
  refresher on gradient accumulation, since `sys.gradient-accumulation` is W2 and "micro-batches *are* accumulation".
- `sys.sequence-context-parallel` → add `sys.stable-numerics` (ring attention needs the online-softmax merge).
- `sys.gradient-clipping` → add `sys.mixed-precision` (item 6 ordering with GradScaler).
- `sys.vanishing-exploding` → `fund.initialization`, `fund.rnn` (B3).
- `sys.parallelism-composition` → add `sys.sequence-context-parallel` (it designs CP degrees).

### B12. `sys.memory-anatomy` and `sys.fsdp` will overflow 9 cards. **major**
**Where:** `sys.memory-anatomy` (10 subtopics, including a two-card Korthikanti derivation, parameter-count derivation,
allocator/fragmentation, and a worked budget, plus the mandatory probes and key-results cards); `sys.fsdp` (11 subtopics).

**Problem:** A realistic count is 11–12 cards for memory-anatomy and 11 for fsdp. The brief says split rather than compress.

**Fix:**
- memory-anatomy: move item 8's allocator internals (reserved vs allocated, fragmentation, `expandable_segments`,
  cross-stream frees) to `sys.profiling` item 4 (memory snapshots) or `sys.fsdp` item 5. Keep "peak vs steady state" and
  "OOM on step 2".
- fsdp: move CPU offload (already in `sys.zero` item 8) and state-dict/checkpointing (to B5's lesson) out. Keep FSDP1 vs
  FSDP2 to one card.

---

## C. Minor (correctness and conventions)

### C1. fp-formats: the general max-normal formula gives 240 for E4M3, while the table says 448. **minor**
**Where:** `sys.fp-formats` item 3 and the table.

**Problem:** E_max = 2^e − 2 − bias applies only to IEEE-style formats. For OCP E4M3 the all-ones exponent holds normals
(E_max = 8), and only mantissa 111 is NaN, so max = 1.75·2⁸ = 448 (Micikevicius 2022 l.148, l.176–177: "17 to 18 binades").

**Fix:** Add a row note "E4M3: formula does not apply; derive 448 from S.1111.110". This also makes a good compute MCQ:
"what would E4M3's max be under IEEE conventions?" Answer: 240.

### C2. fp8 ridge point arithmetic. **minor**
**Where:** `sys.fp8-training` worked computations, "2.0e15 / 3.35e12 ≈ 590".

**Problem:** 2.0e15 / 3.35e12 = **597**. The value 590 corresponds to 1.979e15 (dense spec), or to the Scaling Book's
2 × 295 (l.102).

**Fix:** Use 1.979e15 → 591, or say "≈ 2 × 295 ≈ 590 (Scaling Book)".

### C3. Gradient-accumulation "14% overweight" mislabels what is overweighted. **minor**
**Where:** `sys.gradient-accumulation` worked computations.

**Problem:** 0.40 vs 0.35 is a 14% error in the *loss value*. The per-token weights are 1/20 = 0.05 for the short
micro-batch vs the correct 1/40 = 0.025 (**2×** overweight), and 1/60 vs 1/40 (**0.67×**) for the long one.

**Fix:** State the per-token weights. They are what the `token-weighting` figure should show.

### C4. Micikevicius 2018 is paraphrased too strongly. **minor**
**Where:** `sys.mixed-precision` sources: "gradient histogram where most values fall below fp16's range".

**Problem:** The paper (l.160–166) says *many* activation-gradient values (SSD) fell below the range, and that a loss scale
of **8** was enough for that network.

**Fix:** Use the paper's wording. Contrast scale 8 with GradScaler's 65536 initial scale, as a "why does dynamic scaling
start high and back off" question.

### C5. Pascanu wording. **minor**
**Where:** `sys.vanishing-exploding` pitfalls.

**Problem:** The plan says to use "largest singular value (not spectral radius)". That is correct in substance: the proof
bounds ‖W‖. But the paper's text literally says "absolute value of the largest eigenvalue" (`pascanu2013` l.256–267), and
`fund.rnn` says σ_max.

**Fix:** Tell writers to note the discrepancy: the paper says eigenvalue, while the bound used in its proof is the
spectral norm. Use σ_max consistently with `fund.rnn`.

### C6. PaLM's 46.2% MFU includes attention FLOPs. **minor**
**Where:** `sys.flops-mfu` item 6.

**Problem:** The plan's MFU formula is 6N·tokens. PaLM reports 45.7% without attention and 46.2% with it
(`llm_chowdhery2022_palm.txt` l.5892).

**Fix:** Quote both, and use 45.7% when comparing against the 6N formula.

### C7. DeepSeek-V3 batch size: the Scaling Book contradicts itself. **minor**
**Where:** `sys.parallelism-composition` worked computations (62,914,560 tokens; 30.7k tokens/GPU).

**Problem:** `sys_scalingbook_gpus.txt` l.271 says DeepSeek "used 4M"; l.308 says 4096 × 15360 = 62.9M. DeepSeek-V3
(l.2058) ramps the batch from 3072 to 15360 sequences.

**Fix:** Cite DeepSeek-V3 directly, mention the ramp, and note the book's inconsistency.

### C8. Units in DDP bucket count. **minor**
**Where:** `sys.ddp` worked computations ("≈ 208 buckets").

**Problem:** 208 uses 25 × 10⁶ bytes. `bucket_cap_mb` is interpreted as MiB (25 × 2²⁰), which gives **≈ 198**.

**Fix:** Follow the plan's own GB/GiB rule: state MiB and use 198.

### C9. "Gradient underflow: values < 6e-8 flush to 0" is imprecise. **minor**
**Where:** `sys.mixed-precision` item 2b.

**Problem:** Under round-to-nearest, values below **2.98e-8** (half the smallest subnormal) round to 0. Values between that
and 6.1e-5 survive only as subnormals with reduced precision.

**Fix:** Say both.

### C10. Lost-update condition stated as an equality. **minor**
**Where:** `sys.mixed-precision` item 3.

**Problem:** ½ulp(w)/|w| ranges over [2^−(m+2), 2^−(m+1)] within a binade. The value 4.9e-4 for fp16 is the
worst case at w just above 1. Also, w − 1e-4 at w = 1.0 uses the *smaller* spacing below 1 (half-ulp 2.44e-4). It is still
lost, but it is a good subtlety.

**Fix:** Write "≈ |w|·2^−(m+1), up to a factor 2 within a binade".

### C11. fp16 Kahan example. **minor**
**Where:** `sys.fp-error` worked computations ("fp16 Kahan sum → 1000.0").

**Problem:** fp16(0.1) = 0.0999756, so the exact sum is 999.756. Kahan returns 1000.0 because that is the correctly rounded
fp16 value (spacing 0.5 at 1000), not because it recovers 0.1 × 10⁴.

**Fix:** Say so. Otherwise readers will think Kahan "fixes" the representation error of 0.1.

### C12. Attention-share example ratio vs fraction. **minor**
**Where:** `sys.flops-mfu` worked computations ("≈ 15%").

**Problem:** 12Lhs / 6N = 15.3% relative to 6N. As a fraction of total FLOPs it is **13.3%**.

**Fix:** Label which one.

### C13. FSDP transient memory omits the unsharded gradient of the current unit. **minor**
**Where:** `sys.fsdp` worked computations.

**Problem:** During a unit's backward, the full-size gradient (1.75 GB in bf16, 3.5 GB in fp32 for a 70B/80-layer unit)
exists before reduce-scatter.

**Fix:** Add it to the 3.5 GB transient figure.

### C14. Unsourced or "verify" numbers that the cache cannot verify. **minor**
**Problem:**
- `clip_grad_norm_`'s 1e-6 is not in the cached doc (`sys_pytorch_clip_grad_norm.txt`).
- "1.0 is a common LLM default" (clipping) has no source.
- "Half-precision matmuls are 2–16× faster" has no source.
- "CUDA context ≈ 1–2 GB" has no source.
- "Batch invariance" (`sys.fp-error` item 4) is a term from the 2025 Thinking Machines/He blog, which is not cached.

**Fix:** Cite a cached source (the Llama 3 / PaLM / GPT-3 training details for the 1.0 clip; NVIDIA perf background for
the tensor-core speedups) or drop the claim. Cache the nondeterminism blog if "batch invariance" stays.

### C15. DDP "fully hideable" overstates overlap. **minor**
**Where:** `sys.ddp` worked computations.

**Problem:** The last bucket (the first layers' gradients) cannot overlap with the backward.

**Fix:** Say "hideable except the last bucket ≈ bucket_size/bandwidth". Ties into the `bucket-tradeoff` figure.

---

## D. Pedagogy, questions and figures

### D1. Add an explicit "Refreshers" line per lesson. **minor**
The brief asks the plan to flag prerequisites a newcomer might lack. The plan rarely does. Suggested refreshers:
- spectral norm and submultiplicativity vs eigenvalues (vanishing-exploding);
- L-smoothness / Lipschitz gradient (gradient-clipping, for Zhang 2020's (L0, L1)-smoothness);
- block-matrix multiplication (tensor-parallel);
- the Jacobian of concatenate / sum-then-split (collectives item 3);
- CPython frames and bytecode, PEP 523 (graph-capture item 6);
- online softmax (sequence-context-parallel);
- HBM/SRAM/tensor cores (B6).

### D2. W1 lessons lack end-to-end estimation questions. **minor**
The question ideas are good on derivation and debugging. But the single most common frontier-lab systems question is an
open estimate that chains several lessons. Example: "13B model, 4k context, 64 H100s, ZeRO-3 + full recompute. Memory per
GPU? Step time at 40% MFU? Is DP comm hidden?" Add one such `open` card to each of `sys.zero`, `sys.memory-anatomy` and
`sys.flops-mfu`. Keep `sys.parallelism-composition` for the full design.

### D3. Recall-only question ideas to replace. **minor**
- `sys.frameworks`: "argnums defaults to 0".
- `sys.fsdp`: "FSDP2 shards a FlatParameter".
- `sys.jax-model`: the `pmap` vs `jit` vs `shard_map` compare, if it is answered as API trivia.

Replace them with predict-the-behaviour items, e.g. "what all-gather peak does wrapping the LM head separately cause"
or "what does `vmap(grad(f))` compute and what does it cost".

### D4. Figure fixes. **minor**
- `bucket-tradeoff` and `mfu-gauge` are "synthetic"/"illustrative". Generate `bucket-tradeoff` from an α–β + last-bucket
  exposure model so it encodes the actual trade-off.
- `framework-matrix` and `parallelism-table` are tables drawn as figures. Use markdown tables in a card instead; the brief
  says no decorative figures.
- **Add to `sys.ddp`:** `dp-roofline`, a plot of comm time and backward time against per-GPU tokens, crossing at C/W, with
  intra-node and cross-node lines. This is the key idea of the lesson and currently has no figure.
- Drop the duplicate `rnn-eigen` / `cliff` (B3).

### D5. Trim low-value TF history in `sys.frameworks`. **minor**
TF1 Sessions, `feed_dict` and parameter servers are of little interview value at frontier labs in 2026. Reduce TF to one
card (tf.function retracing as a contrast to jax.jit and torch.compile). Use the freed card for "per-example gradients and
functional transforms across frameworks" and for "torch.compile + DDP/FSDP interaction" (DDPOptimizer, graph breaks at
bucket boundaries; `sys_pytorch_ddp_note.txt` covers it).

### D6. Smaller coverage additions. **minor**
- `sys.gradient-accumulation`: accumulate in fp32 when gradients are bf16 (this is the Playbook's 20-bytes/param buffer;
  link `sys.fp-error` swamping). Also sequence packing with document masks, and its interaction with token-count
  normalisation.
- `sys.parallelism-composition`: strong vs weak scaling definitions (Scaling Book Part 0) as one line.
- `sys.memory-anatomy` item 9 levers: add LoRA/QLoRA ("fine-tune 70B on one node" is a common question).
- `sys.sequence-context-parallel` item 6: DeepSpeed-Ulysses (all-to-all head sharding) is asked often. Cache
  arXiv 2309.14509 rather than leave it "not covered".

### D7. `sys.ddp` item 10 duplicates the tuning plan's multi-host checklist. **minor**
The tuning plan lists the multi-host checklist as "do not duplicate, just link" (its Part B footer), but its own row for
`sys.data-parallel†` asks for parts of it. Resolve the conflict: keep one line in `sys.ddp` (loss normalised by the global
batch; same init seed, different data seeds) and link `sys.tuning-pipeline` card 5 for the rest.

---

## E. Source quality

### E1. Collective-algorithm theory has no primary source. **minor-major**
Thakur 2005 failed to download. The ring is covered by Playbook A0 and nccl-tests. But tree, double-binary-tree and
recursive-halving latency claims (collective-algorithms item 4) are unsourced.

**Fix:** Cache one of:
- Patarasuk & Yuan 2009, "Bandwidth optimal all-reduce algorithms for clusters of workstations" (JPDC; primary for ring
  optimality);
- Chan et al. 2007, "Collective communication: theory, practice, and experience" (α–β–γ costs of all algorithms);
- NVIDIA's 2019 NCCL 2.4 double-binary-tree blog.

Otherwise, reduce item 4 to what the Scaling Book states.

### E2. Secondary-only spots. **minor**
- Welford/Chan come only from Wikipedia. Cite Welford 1962 or Chan, Golub & LeVeque 1983, or treat Wikipedia as cross-check
  and derive in full.
- Nsight Systems/Compute descriptions (`sys.profiling` item 4) have no cached doc. Cache the NVIDIA Nsight docs or keep
  the claims generic.
- Hochreiter 1991 is mentioned but not cached. Cache Hochreiter et al. 2001, "Gradient flow in recurrent nets", or drop the
  attribution detail.
- Optimal checkpointing (Griewank & Walther, *revolve*) would strengthen `sys.activation-checkpointing`. Optional.

### E3. Otherwise the sources are excellent.
ZeRO, Megatron (both papers), Korthikanti, GPipe, PipeDream, Zero Bubble, Li 2020, Zhao 2023, TorchTitan, Micikevicius
2018/2022, Kalamkar, Goldberg, Blanchard–Higham, PyTorch 2 (Ansel), Frostig, Abadi and GSPMD are all primary. The Scaling
Book and the Playbook are the right textbook-grade anchors.

---

## F. Spot-check log (cited section → cache evidence)

| # | Plan claim | Cache evidence | Result |
|---|---|---|---|
| 1 | Korthikanti §5: selective recompute saves 70% for 2.7% FLOPs (GPT-3) | `sys_korthikanti…` l.540–542 | ✓ |
| 2 | Korthikanti §6.3 MFU 56.3% (HFU 57.0%) | l.812 | ✓ |
| 3 | Korthikanti §4.3: fp32 logits 4sbv/t | l.412 (§4.3), l.427 | ✓ |
| 4 | Korthikanti: HFU/MFU ≈ 1 + s/6h; interleaving factor 1 + (p−1)/(pm), with m = interleaving stages | l.806; l.406–409 | ✓ (notation clash correctly flagged) |
| 5 | Narayanan: 34 days at 140 TFLOP/s on 1024 A100s; m ≫ p; microbatches a multiple of p for interleaving | l.1585–1586; l.647; l.678 | ✓ |
| 6 | GPipe: bubble negligible for M ≥ 4K | l.283 | ✓ (plan rightly calls it loose: 17.9% at p=8, m=32) |
| 7 | FP8: E4M3 448, single NaN pattern, 17 → 18 binades | `sys_micikevicius2022_fp8` l.148, l.176–177 | ✓ |
| 8 | Zero Bubble: ZB-H1 bubble ≈ ⅓ of 1F1B; ZB-H2 (2p−1) micro-batches | `sys_qi2023` l.903–904; l.955 | ✓ |
| 9 | DDP default bucket 25 MB | `sys_li2020` l.994, l.1141 | ✓ (units: see C8) |
| 10 | GradScaler 65536 / 2 / 0.5 / 2000 | `sys_pytorch_amp` l.296 | ✓ |
| 11 | Gemma 2 soft-cap 50 (attention) / 30 (final) | `llm_team2024_gemma2` l.224 | ✓ |
| 12 | z-loss coefficient 1e-4 (Wortsman, PaLM) | `sys_wortsman2023` l.185; `llm_chowdhery2022_palm` l.788 | ✓ |
| 13 | DeepSeek-V3: 1×128 / 128×128 tiles; FP8 promotion with K = 4096; DualPipe two parameter copies | `llm_deepseek2024_v3` l.1771–1772; l.1795–1800; l.1548 | ✓ |
| 14 | ZeRO Fig. 1: 120 / 31.4 / 16.6 / 1.88 GB | `sys_rajbhandari2020_zero` l.459–515, l.584 | ✓ |
| 15 | Playbook 112 / 140 GB; full recompute costs 30–40% | `sys_ultrascale2025` l.687–689; l.949 | ✓ |
| 16 | TE: MXFP8 32-element blocks, E8M0 scales; amax_history_len=16, "max" | `sys_nvidia_te_fp8_primer` l.63, l.69, l.94 | ✓ |
| 17 | Scaling Book: attention/matmul FLOPs = T/8D | `sys_scalingbook_transformers` l.79 | ✓ |
| 18 | Scaling Book: 2200 / 2475 tokens; TP < F/2475; E/k multiplier | `sys_scalingbook_gpus` l.270–284 | ✓, but see A1 for the bandwidth model behind it |
| 19 | Scaling Book: FSDP comm "equals DP's" | `sys_scalingbook_training` l.132–134 | ✓ as cited, but clashes with ZeRO 1.5× (B7) |
| 20 | Micikevicius 2018 §3.2 histogram "most values below range" | l.155–166 | ✗ overstated (C4) |
| 21 | Pascanu: singular value | `pascanu2013` l.256–267 says "largest eigenvalue" | ~ (C5) |
| 22 | PaLM MFU 46.2% | `llm_chowdhery2022_palm` l.311, l.5892 | ✓, with attention (C6) |
| 23 | HF blog: divide by total non-padding tokens | `sys_hf2024…` l.49 | ✓ |
| 24 | Zhao §3.3.4 grad accumulation; §3.4.1 allocator; §3.4.2 rate limiter | `sys_zhao2023` l.976–978, l.996, l.1030 | ✓ |
| 25 | d2l §9.5.3 min(1, θ/‖g‖); Brock §4 AGC; Li §3.2.1–3.2.4 | d2l l.451, l.521; Brock l.316; Li l.502, l.435, l.623, l.837 | ✓ |
| 26 | `clip_grad_norm_` adds 1e-6 | not in cached doc | unverifiable (C14) |

## G. Recomputation summary

All of the following match the plan to the stated precision:
- **fp formats:** the fp-formats table (verified with `torch.finfo` and formulas; E4M3 448 needs the C1 note), the swamping
  examples, Adam ε → 0 in fp16, ln 65504 = 11.09.
- **fp error and stable numerics:** running sums (256 / 32), the 1e8 associativity example, variance 24.0 vs 22.5, the
  bf16 bound ≈ 16, softmax/LSE (1002.4076), log1p, the Welford trace.
- **mixed precision and memory:** loss scaling 1e-8·2¹⁶ = 6.55e-4; ZeRO 120/31.4/16.6/1.875 and 112/29.3/15.5/1.75;
  70B: 2.19 GB and 286.6 GB; DDP vs ZeRO-3 traffic 28 / 42 GB; N = 6.575e9; GPT-3 activations 2.87 GB/layer and 275 GB;
  7B activations 104 GB → 18.25 GB; logits 2.1 GB; selective 82 GB (70%); 2sbhL 4.8 / 1.07 GB.
- **communication and parallelism:** TP-only 23sbh = 0.58 GB vs TP+SP 14.25sbh = 0.36 GB; collective bytes
  12.25 / 24.5 GB; intra-node ring 54 / 66 ms; 1 GB at 100 GB/s → 15 ms; latency 10.2 vs 0.1 ms; DDP 10 / 20 ms vs
  43 ms; TP 0.83 ms compute vs 0.52 ms comm; F/2475 = 11.6.
- **pipelines:** bubbles 17.9 / 21.9 / 27.3 / 37.5 / 46.7 / 87.5 / 19.0%; interleaved 5.5 / 5.2%.
- **roofline:** ridges 295 / 156; n/3 = 85 / 341 / 1365; I = 455; elementwise 1.79 ms.
- **MFU and training time:** 12.7% and 9,419 tok/s; 17%; 6.3e24 FLOPs → 11.2 days; GPT-3 33.9 days; LLaMA-3 977 tok/GPU;
  DeepSeek 30.7k tok/GPU.
- **stability tricks and clipping:** soft-cap 9.65 / 0.897 / 29.92 / 0.0051; z-loss gradient values; clipping 1.9° → 5.7°;
  0.9⁵⁰, 1.1⁵⁰, 0.5⁵⁰, 0.25¹⁰.

Wrong or mislabelled:
- the cross-node all-reduce examples (A1);
- the fp8 ridge (C2);
- the "14% overweight" (C3);
- the bucket count (C8);
- the attention share (C12);
- the z-loss interpretation (B8).

---

**Verdict: approve with required changes.** Fix A1 (cross-node bandwidth model) before any W1 writing on `sys.ddp` /
`sys.collectives`. Re-wave to put roofline/FLOPs in W1 and move the ring derivation into `sys.collectives` (B1–B2).
De-duplicate against `fund.*` (B3). Add or assign EP systems, reliability/checkpointing, GPU basics and quantization
(B4–B6, B9) before W3 planning is final. The remaining items are writer-level corrections.

---

## Changes applied (curator, 2026-10-02)

All edits are in `docs/content-plan-applied.md`. The plan now has 30 topics (11 W1); see its topic table. New sources are
listed in `INDEX-sys.md`.

**A1 (blocking).** Added a "Bandwidth rule for collectives" to the conventions and as `sys.collectives` item 6. Cross-node
AG/RS/AR use node egress (400 GB/s); 50 GB/s per GPU is used only for cross-node all-to-all and for a TP group spread one GPU
per node. Updated the hardware-constants row. Downstream fixes:
- 14 GB / 64 GPUs: 0.55 s → ≈ 69 ms.
- DDP 1.3B on 64 GPUs: ≈ 12.8 ms (node egress) / ≈ 21.5 ms (sequential hierarchical) → **hidden**, not exposed.
- New exposed example: 1024 tokens/GPU (5.4 ms backward), plus the fp32-gradient variant (25.6 ms).
- The flat-ring numbers are kept only as labelled distractors.
- The all-to-all exception is in `sys.collective-algorithms` item 4, with the link to EP.
- The TP-across-nodes example is relabelled as the per-NIC case.

**B1.** `sys.roofline` and `sys.flops-mfu` are now W1. The new Part D (gpu-basics 100, roofline 110, flops-mfu 120, profiling
130) sits after memory-anatomy and before the parallelism block. `sys.roofline` was added to the prereqs of ddp,
tensor-parallel and fp8-training (fp8 is marked † as a forward prereq), and `sys.flops-mfu` to activation-checkpointing.
vanishing-exploding and gradient-clipping are demoted to W2.

**B2.** The ring all-reduce derivation and time formula, its figure, and the Patarasuk & Yuan source moved into
`sys.collectives` (now about 9 cards). algbw/busbw moved to `sys.collective-algorithms`, which is re-scoped to α–β, trees,
hierarchy, all-to-all, SHARP, interconnects and overlap.

**B3.**
- The Boundaries section now names the exact fund ids that own the basics, with their figures: fund.rnn, fund.lstm-gru,
  fund.initialization, fund.activations, fund.normalization, fund.backprop, fund.training-loop, fund.logistic-regression,
  fund.variance-covariance and fund.adam.
- `sys.vanishing-exploding` is rewritten around the deep-net and transformer view, with one recap card:
  - residual Jacobian derived for L blocks;
  - 1/√L branch scaling;
  - pre-LN vs post-LN (Xiong 2020);
  - LayerNorm scale invariance;
  - diagnostics at scale;
  - explosion in practice.
- Its prereqs are now fund.initialization and fund.rnn. The `rnn-eigen` and `cliff` figures are dropped.
- `sys.gradient-clipping` compresses items 1–3 into one recap card and links `fund.rnn/clipping-cliff`.
- activation-checkpointing item 2 recaps `fund.backprop`.
- Overlap notes added to stable-numerics, mixed-precision and gradient-accumulation.

**B4.** No `sys.expert-parallel` lesson, per the coordinator. EP systems are assigned to `llm.moe-systems` (id from the
current LLM plan draft, to confirm). `sys.collective-algorithms` item 4 and `sys.parallelism-composition` item 6 link to it.

**B5.** Added `sys.reliability-checkpointing` (W3, order 260). It covers:
- Llama 3 §3.3.4 failure statistics;
- checkpoint contents and size (70B: 1.12 TB);
- sharded checkpoints where every rank writes its shard, vs gather-to-rank-0;
- resharding on restore;
- async checkpointing;
- the Young/Daly interval derivation (≈ 19 min example);
- failure detection, stragglers, silent data corruption, rollback, elastic restarts.

The rank-0-only claim is removed. DDP item 10 now says "DDP only", and FSDP item 9 says sharded state dicts are written by
all ranks. The tuning plan's `sys.checkpointing†` now maps here.

**B6.** Added `sys.gpu-basics` (W1, order 100, before roofline). It covers SMs and subpartitions, Tensor Cores vs CUDA cores
(990 vs 66 TFLOP/s), the HBM/L2/SMEM/register hierarchy, kernels, warps, blocks and waves, tile quantisation, streams and
async execution, and launch overhead. It is a prereq of roofline and profiling; mixed-precision now carries a one-line
Tensor Core definition that points to it.

**B7.** Added `sys.zero` item 5b, reconciling ZeRO's 1.5× volume with the Scaling Book's "same cost": bytes/FLOP is 1/B in
both phases, so the threshold is unchanged. Added a matching MCQ and fixed the source wording.

**B8.** z-loss mechanism corrected, with the decomposition and a numeric step (softmax 0.8808 → 0.8774). Per B9(b),
`sys.stability-tricks` is cut to the numerics: bf16 spacing at |z| = 64 is 0.5, the z-loss decomposition, clamp vs tanh,
and QK-norm bounds. About 5 cards; merge it into stable-numerics if it ends up shorter. The Adam ε item is dropped.

**B9.** The LLM plan now exists, so the Boundaries section lists owners by its draft ids, marked "id to confirm":
- llm.moe-systems
- llm.quantization / llm.quantization-advanced (no sys quantization lesson, per the coordinator)
- llm.kv-cache / llm.arithmetic-intensity
- llm.training-stability
- llm.params-flops / llm.scaling-laws
- llm.flash-attention, llm.long-context, llm.lora / llm.qlora-peft

All generic `llm.*` references are replaced. "See also" lines for `sb.*` were added to collectives, ddp, zero, fsdp,
tensor-parallel and roofline.

**B10.** The gradient-dtype convention is now in the conventions section and in the DDP and collectives examples (bf16
assumed; fp32 doubles comm time and moves the threshold to 4400 / 4950 tokens). Added to mixed-precision item 7, with a
predict question.

**B11.** Prereqs updated as listed, plus inline refreshers:
- LSE in tensor-parallel item 5;
- gradient accumulation in pipeline-parallel item 5;
- online softmax for sequence-context-parallel.

**B12.**
- memory-anatomy: allocator internals moved to `sys.profiling` item 4. LoRA/QLoRA added to the levers, plus an open-estimate question.
- fsdp: CPU offload and checkpointing moved out (to zero item 8 and reliability-checkpointing). FSDP1 vs FSDP2 limited to one card.

**Minor fixes**
- C1: E4M3 note (448 vs 240 under IEEE conventions), plus MCQ.
- C2: fp8 ridge ≈ 591 (dense 1.979e15); the constants table now says 1.98e15.
- C3: per-token weights 2× / 0.67×.
- C4: Micikevicius wording (many values; scale 8), plus a question.
- C5: Pascanu eigenvalue vs σ_max note.
- C6: PaLM 45.7% / 46.2%.
- C7: DeepSeek-V3 batch ramp 3072 → 15360, citing the paper, with the book's inconsistency noted.
- C8: 198 buckets (MiB).
- C9: 2.98e-8 / 6.1e-5.
- C10: factor-2 range and the below-1 spacing subtlety.
- C11: Kahan caveat.
- C12: 15.3% vs 13.3%.
- C13: FSDP transient includes the unsharded gradient (5.25–7 GB).
- C14: 1e-6 claim dropped; clip 1.0 sourced to GPT-3 App. B and Llama 2 §2.2; tensor-core speedup sourced to the Scaling
  Book; CUDA context sourced to the Playbook; "batch invariance" name removed.
- C15: last-bucket exposure.

**Pedagogy, questions and figures**
- D1: refreshers added to 7 lessons.
- D2: open estimate cards added to memory-anatomy, zero and flops-mfu.
- D3: the argnums, FlatParameter and pmap questions are replaced with predict-the-behaviour items.
- D4: `dp-roofline` figure added; `bucket-tradeoff` and `mfu-gauge` are now model-generated; `framework-matrix` and
  `parallelism-table` become markdown tables.
- D5: TF trimmed to one card; added per-example gradients and torch.compile + DDP (DDPOptimizer).
- D6:
  - fp32 accumulation and sequence packing in gradient-accumulation;
  - strong vs weak scaling;
  - LoRA/QLoRA lever;
  - DeepSpeed-Ulysses cached (`sys_jacobs2023_ulysses.txt`) and taught in sequence-context-parallel item 6.
- D7: DDP hygiene trimmed to one line, linking `sys.tuning-pipeline`.

**Sources**
- E1: cached Patarasuk & Yuan 2009 (`sys_patarasuk2009_bandwidth_optimal_allreduce.txt`) and NVIDIA's 2019 NCCL 2.4
  double-binary-tree blog (`sys_nvidia2019_nccl_double_binary_tree.txt`). Chan et al. 2007 has no free copy (404).
- E2: Welford is now derived in full, with Wikipedia as cross-check only; Nsight claims are kept generic; the Hochreiter
  attribution is removed with the vanishing-exploding rewrite.
- Griewank/revolve: not added (optional).
