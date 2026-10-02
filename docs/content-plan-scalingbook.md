# Content plan: Special Edition, "How to Scale Your Model" (area `scalingbook`, prefix `sb.`)

A dedicated lesson series that follows the JAX Scaling Book (Austin et al., Google DeepMind, 2025),
https://jax-ml.github.io/scaling-book/, chapter by chapter, keeping its notation (B, D, F, T, S, L, N, K, H, G, V;
C, W, X/Y/Z, M_X; `A[I_X, J_Y]`, `{U_X}`), its hardware numbers (TPU v5e/v5p/v6e, H100/B200) and its worked
examples (LLaMA-2 13B, LLaMA-3 70B/405B). Files go in `PrepApp/Content/scalingbook/`; figures in
`tools/figures/sb.*.py`. Format and quality bar: [CONTENT_GUIDE.md](../CONTENT_GUIDE.md) and
[writing-brief.md](writing-brief.md).

**Sources.** The book's markdown sources are cached as `sb_*.md` in the shared source cache (see `INDEX-sb.md`).
Cite as `Scaling Book ch. N (Title)` or `Scaling Book ch. N §"Section"`, e.g. `Scaling Book ch. 3 §"Case 2"`.
Secondary sources already in the cache that can back up or extend a lesson: Williams et al. 2009 (roofline model,
`sys_williams2009_roofline.txt`), He 2022 "Making Deep Learning Go Brrrr" (`sys_he2022_brrr.txt`), Pope et al. 2022
"Efficiently Scaling Transformer Inference" (`llm_pope2022_inference_scaling.txt`), Rajbhandari et al. 2020 ZeRO,
Shoeybi et al. 2019 Megatron-LM, Huang et al. 2019 GPipe, Korthikanti et al. 2022, Dao et al. 2022 FlashAttention,
the JAX docs (`sys_jax_*.txt`), NCCL docs and the HF Ultra-Scale Playbook.

**Overlap with other plans.** `sys.*` (applied/systems) and `llm.*` topics cover some of the same ground (FSDP,
TP, KV cache, FlashAttention, speculative decoding) in a framework-neutral way. This series deliberately repeats it
in the book's own frame (rooflines first, TPU numbers, book notation), and links back with `prereqs` only inside
the series.

**Conventions for every lesson.** 6–10 explainer cards ending in "What interviewers probe" and "Key results";
10–14 MCQs; 8–12 flashcards with at least 3 `open`; `source:` on everything; 2–4 figures; every number recomputed
in Python. The book's end-of-chapter problems are rewritten as MCQs or `open` cards, with the computation in the
explanation. Where the book gives no answer, the writer derives one and states assumptions. Where the book rounds
in a way that changes a number noticeably, say so (example: ch. 2 Q4 gives B > 267; exact arithmetic gives ≈ 260).

## Lesson map

| order | id | title | chapter / sections | prereqs | batch |
|---:|---|---|---|---|:-:|
| 10 | `sb.roofline-basics` | Rooflines I: Compute, Bandwidth and Arithmetic Intensity | Part 0 (strong scaling); ch. 1 "Where does the time go?", "Visualizing rooflines" | — | **1** |
| 20 | `sb.roofline-matmul` | Rooflines II: Matmuls, Precision and Network Rooflines | ch. 1 "Matrix multiplication", "Network communication rooflines", Problems Q1–Q5 | roofline-basics | **1** |
| 30 | `sb.tpu-chip` | TPUs I: Inside the Chip | ch. 2 "What is a TPU?", App. A (VPU, scalar core), App. B (systolic array), Q1, Q4 | roofline-matmul | **1** |
| 40 | `sb.tpu-networking` | TPUs II: Hosts, ICI Tori, Pods and DCN | ch. 2 "TPU Networking", "Key takeaways", spec tables, Q2, Q3, Q5, Q6 | tpu-chip | **1** |
| 50 | `sb.sharding-notation` | Sharded Arrays: Meshes and Sharding Notation | ch. 3 "Partitioning notation", "How do we describe this in code?", Case 1, Q1 | tpu-networking | 2 |
| 60 | `sb.sharded-matmul` | Sharded Matmuls: The Four Cases | ch. 3 Cases 1–4, block-matrix view, Q4, Q9 | sharding-notation | 2 |
| 70 | `sb.collective-costs` | Collectives I: AllGather, ReduceScatter and AllReduce Costs | ch. 3 AllGather cost derivation, latency regime, multi-axis, RS/AR, Q2, Q3, pop quiz 2 | sharded-matmul | 2 |
| 80 | `sb.alltoall-overlap` | Collectives II: AllToAll, Transposes and Overlap | ch. 3 "AllToAll", "More about the ReduceScatter", "How to overlap", summary table, Q5–Q7, Q10 | collective-costs | 2 |
| 90 | `sb.flops-counting` | Counting FLOPs: Contractions and the 6ND Rule | ch. 4 "Counting dots", "Forward and reverse FLOPs", Q3 | alltoall-overlap | 3 |
| 100 | `sb.transformer-accounting` | Transformer Params and FLOPs, Layer by Layer | ch. 4 "Transformer accounting", MLP/attention/other, rule of thumb, T/8D, Q1, Q2, Q5, Q7 | flops-counting | 3 |
| 110 | `sb.transformer-memory` | MoE, Rematerialization and KV Caches | ch. 4 "Sparsity and MoE", "Gradient checkpointing", "KV caching", Q4, Q6, Q8 | transformer-accounting | 3 |
| 120 | `sb.flash-attention` | Flash Attention Through the Roofline Lens | ch. 4 App. A | transformer-memory | 3 |
| 130 | `sb.data-parallelism` | Training I: Setup and Data Parallelism | ch. 5 "What do we mean by scaling?", notation, four schemes, "Data parallelism" | flash-attention | 4 |
| 140 | `sb.fsdp` | Training II: FSDP / ZeRO Sharding | ch. 5 "FSDP", App. A (backward comms) | data-parallelism | 4 |
| 150 | `sb.tensor-parallelism` | Training III: Tensor (Megatron) Parallelism | ch. 5 "Tensor parallelism" | fsdp | 4 |
| 160 | `sb.fsdp-plus-tp` | Training IV: Mixing FSDP and Tensor Parallelism | ch. 5 "Combining FSDP and TP" (X_opt, B/N > α²/(M_X M_Y F)) | tensor-parallelism | 4 |
| 170 | `sb.pipeline-and-dcn` | Training V: Pipelining and Scaling Across Pods | ch. 5 "Pipelining", "Scaling across pods", "Takeaways" | fsdp-plus-tp | 5 |
| 180 | `sb.training-problems` | Training VI: Worked Problems on LLaMA-2 13B | ch. 5 Problems Q1–Q3 + takeaway tables | pipeline-and-dcn | 5 |
| 190 | `sb.llama3-training-cost` | Case Study: What LLaMA-3 70B Training Costs | ch. 6 "What does LLaMA 3 look like?", "Counting parameters and FLOPs" | training-problems | 5 |
| 200 | `sb.llama3-training-sharding` | Case Study: Sharding LLaMA-3 for Training | ch. 6 "How to shard LLaMA 3-70B", Worked problems Q1–Q2 (4 pods; 405B) | llama3-training-cost | 5 |
| 210 | `sb.inference-basics` | Inference I: Prefill, Generation and the KV Cache | ch. 7 "Basics", "What do we want to optimize?", "Linear operations" | llama3-training-sharding | 6 |
| 220 | `sb.inference-latency` | Inference II: Attention Rooflines and Step-Time Bounds | ch. 7 "What about attention?", "Theoretical estimates…", pop quiz, App. A | inference-basics | 6 |
| 230 | `sb.inference-memory` | Inference III: KV-Cache Memory and How to Shrink It | ch. 7 "What about memory?", LLaMA-2 13B tables, "Tricks…" | inference-latency | 6 |
| 240 | `sb.inference-sharding` | Inference IV: Sharding Prefill, Generation and the KV Cache | ch. 7 "Distributing inference…", App. B (2D weight-stationary), App. C (latency-bound comms), Q7 | inference-memory | 6 |
| 250 | `sb.inference-engines` | Inference V: Serving Engines and Speculative Sampling | ch. 7 "Designing an effective inference engine", JetStream, App. D | inference-sharding | 7 |
| 260 | `sb.inference-problems` | Inference VI: Worked Problems | ch. 7 Problems Q1–Q6 | inference-engines | 7 |
| 270 | `sb.llama3-serving` | Case Study: Serving LLaMA-3 70B, Memory and Throughput | ch. 8 "What's the LLaMA serving story?", "Thinking about throughput" | inference-problems | 7 |
| 280 | `sb.llama3-serving-tradeoffs` | Case Study: LLaMA-3 Sharding, Prefill and the Pareto Frontier | ch. 8 sharding Q, "What about prefill?", "Visualizing the latency-throughput tradeoff", Problems Q1–Q3 | llama3-serving | 7 |
| 290 | `sb.profiling` | Profiling TPU Programs | ch. 9 (all) | llama3-serving-tradeoffs | 8 |
| 300 | `sb.jax-sharding-modes` | JAX Parallelism: Auto, Explicit and shard_map | ch. 10 "How does parallelism work in JAX?" (3 modes) | profiling | 8 |
| 310 | `sb.jax-collective-matmul` | JAX in Practice: Collective Matmuls and MoE Routing | ch. 10 collective matmul example, Problems Q1–Q4 | jax-sharding-modes | 8 |
| 320 | `sb.capstone-review` | Conclusions: The Numbers to Carry Into an Interview | ch. 11 + cross-chapter recap | jax-collective-matmul | 8 |
| 330 | `sb.gpu-chip` | GPUs I: SMs, Tensor Cores and the Memory Hierarchy | ch. 12 "What is a GPU?", "Memory", spec tables, "GPUs vs TPUs at the chip level", Quiz 1 | tpu-networking | 9 |
| 340 | `sb.gpu-networking` | GPUs II: NVLink Nodes and Fat-Tree Networks | ch. 12 "Networking", "At the node level", "Beyond the node level", GB200, Quizzes 2–3, App. B | gpu-chip | 9 |
| 350 | `sb.gpu-collectives` | GPUs III: Collectives on GPUs | ch. 12 "How do collectives work on GPUs?", SHARP, cross-node, Quiz 4 | gpu-networking, alltoall-overlap | 9 |
| 360 | `sb.gpu-llm-rooflines` | GPUs IV: LLM Scaling Rooflines on GPUs | ch. 12 "Rooflines for LLM scaling on GPUs", Examples, TLDR, Quiz 5, App. A | gpu-collectives, pipeline-and-dcn | 9 |

Batch numbers after 1 are suggestions (about 4 lessons per batch). The GPU lessons follow the book's placement
(ch. 12, last), but their prereqs only need lessons 40 and 80, so they can be moved earlier if the user prefers.

---

## Per-lesson syllabus

### 10 · `sb.roofline-basics` (batch 1)
**Covers:** Part 0 "Why should you care?" (strong scaling); ch. 1 "Where does the time go?" and "Visualizing rooflines".
**Syllabus**
- The three limits: compute (FLOPs/s), bandwidth (bytes/s) and memory capacity (bytes). The book's FLOPs (count) vs FLOPs/s (rate) convention. bf16 as the default 2-byte dtype.
- Strong scaling: throughput proportional to chip count; why more chips shrink per-chip compute while communication does not shrink as fast, so we eventually become communication-bound.
- T_math = FLOPs / peak; worked numbers: 1e12 FLOPs on H100 (9.89e14 dense bf16) ≈ 1.01 ms, on v6e ≈ 1.09 ms; achievable fraction of peak (H100/B200 ≈ 80–85%, TPU ≈ 95%); the sparsity factor of 2 in NVIDIA spec sheets.
- T_comms = bytes / bandwidth, for HBM (H100 3.35 TB/s, v5e 8.2e11, v6e 1.6e12) and for inter-chip links (ICI, DCN, PCIe; detail in lesson 40).
- Overlap: lower bound max(T_math, T_comms), upper bound the sum; proof that sum ≤ 2·max; compute-bound vs comms-bound; fraction of peak used = T_math/T_comms when comms-bound.
- Arithmetic intensity (FLOPs per byte) and the derivation T_math > T_comms ⟺ I(alg) > C/W. Hardware critical intensity: v5e 240, H100 ≈ 295; also v5p ≈ 164, v6e ≈ 575 (from the spec table; the "240" rule is v5e-specific).
- Worked: dot product, (2N−1)/(4N+2) → ½ (bf16); runs on the VPU (v5p VPU ≈ 7e12 FLOPs/s per core → critical ≈ 3), still bandwidth-bound. Ops with size-independent intensity are always bandwidth-bound; fusion as the remedy (ch. 9 fusions).
- The roofline plot: attainable = min(C, I·W), derived; ridge point; two levers (raise intensity, raise bandwidth); log-log shape.
**Figures:** v5e roofline (HBM vs VMEM ramps, ridges 240 and ≈11, example algorithms); serial vs overlapped bars.
**Questions:** T_math compute (bandwidth-units trap); bounds from T_math/T_comms; "which is false" about the factor-2 bound; H100 critical intensity from the spec sheet (sparsity trap); fp32 dot product (¼) and fp32 elementwise add (1/12, from ch. 12 Quiz 1 Q7); attainable throughput at I = 60; effect of doubling bandwidth on two algorithms; which chip has the highest ridge; figure-based (dot product in VMEM); reading a measured time against T_math/T_comms; strong-scaling limit; which op is compute-bound. Open: derive the intensity condition; why elementwise ops are memory-bound and what to do; why a kernel misses its roofline.

### 20 · `sb.roofline-matmul` (batch 1)
**Covers:** ch. 1 "Matrix multiplication", "Network communication rooflines", Problems Q1–Q5.
**Syllabus**
- bf16 [B,D]·[D,F]: 2BDF FLOPs (BDF multiplies + BF(D−1) adds), 2BD + 2DF + 2BF bytes (output cast to bf16).
- Intensity BDF/(BD+DF+BF) → B when B ≪ D, F; interpretation (weight bytes dominate, each weight reused B times); B > 240 on v5e; ≈ 295 on H100 (Q5).
- Tokens, not sequences; per-replica (per-copy-of-weights) batch, because model sharding scales FLOPs and bandwidth together; 512 × 4096 tokens on 2048 chips → 1k tokens local.
- Exact vs approximate crossover (D = F = 4096 → B ≈ 272; D = F = 1024 → ≈ 450; ch. 1 Q3 plot).
- Precision: int8 × int8 (Q1: bytes BD + DF + BF, B > 240 unchanged); int8 weights + bf16 compute (Q2: B > 120); bf16 weights + int8 compute (B > 480); general B_crit = (peak at compute dtype / HBM BW) × (bytes per weight)/2 (ch. 7 β rule, stated correctly).
- Per-example weights int8[B,D,F] (Q4): intensity ≈ 2, always bandwidth-bound; pointer to decode attention.
- Tiling (footnote): intensity of a tiled matmul ≈ bm·bn/(bm+bn); square tile ≥ 480 for 240.
- Network roofline: D-split matmul over two chips; T_math = BDF/C, T_comms = 2BF/W; compute-bound iff D > 2C/W = 8755 (W = 4.5e10); why the threshold depends on D not B.
**Figures:** exact attainable FLOPs/s vs B (bf16 4096, bf16 1024, int8-weight 4096); two-chip contracting-dim split diagram.
**Questions:** FLOPs/bytes/time of a concrete matmul; why intensity ≈ B; int8 variants (120/240/480); per-example weights; local batch from global; D = 4096 two-chip ratio; exact crossover for D = F = 1024; minimum tile; figure-based; "which is false" on per-replica batch; H100 decode matmul utilization. Open: derive B_crit; why weight-only quantization speeds up decode; why per-example matmuls are memory-bound.

### 30 · `sb.tpu-chip` (batch 1)
**Covers:** ch. 2 "What is a TPU?", App. A (VPU, VREGs, scalar core), App. B (systolic arrays), Q1, Q4.
**Syllabus**
- TensorCore = MXU(s) + VPU + VMEM (+ scalar unit, SMEM) next to HBM; data path HBM → VMEM → VREGs → MXU/VPU → back.
- Systolic array: 128×128 MACs (256×256 on v6e), weight-stationary, activations stream from the left, partial sums flow down; bf16[8,128]@[128,128] per 8 cycles = 32768 FLOPs/cycle; ≈ 4.9e13 FLOPs/s per MXU at 1.5 GHz; 4 MXUs → 1.97e14 (v5e); pipeline fill bubble.
- Consequences: pad dims to 128 (256 on v6e), tile dims ≥ 128 × #MXUs; int8/int4 ≈ 2×/4× bf16 where supported; VPU math in fp32.
- Pipelining of copies and compute, why the max() model applies; starving the MXU.
- VMEM: 128 MiB on v5e, ≈ 22× HBM bandwidth → critical intensity ≈ 10–20; VMEM prefetch of FFW weights during attention; ch. 2 Q4 (int8 [B,4096]×[4096,16384]: B > ≈ 260 from HBM, > ≈ 11 from VMEM).
- VPU: (8 sublanes × 128 lanes) × 4 ALUs, VREGs (64 × 4 KiB = 256 KiB per v5p core), ≈ 7e12 FLOPs/s per v5p core (≈ 30× below MXU); sublane vs cross-lane (XLU) reductions; scalar core single-threaded (one DMA per cycle).
- Cores and chips: megacore (two cores sharing HBM on v4/v5p), single core on v5e, TPU7x has two cores linked rather than megacore.
- Latency lower bound for sampling (Q1): 200B bf16 on 32 v4p → ≈ 10 ms per step.
- Chip spec table (HBM, HBM BW, bf16/int8 FLOPs/s for v3, v4p, v5p, v5e, v6e, TPU7x) with computed ridge points.
**Figures:** chip block diagram; weight-stationary systolic array; HBM vs VMEM timing for the Q4 matmul.
**Questions:** MXU FLOPs/s; v6e 256×256 implications; padding utilization; VMEM ridge; Q4 crossover; latency bound variant (70B int8 on 8 v5e); VPU throughput; which ops run on VPU; cross-lane reductions; scalar-core consequence; megacore; why systolic arrays are efficient; figure-based (what flows down); when VMEM prefetch helps. Open: explain a systolic array; derive the sampling-latency lower bound; how a TPU differs from a CPU cache hierarchy.

### 40 · `sb.tpu-networking` (batch 1)
**Covers:** ch. 2 "TPU Networking", "Key takeaways", TPU spec tables, Q2, Q3, Q5, Q6.
**Syllabus**
- Trays of 4 chips, hosts (v5e: 8 chips per host as 4×2), PCIe to host (≈ 1.6e10 B/s per chip, ≈ 75–100× slower than HBM); host offload is slow; Q3 PCIe intensity (B > 57,500 on v6e).
- ICI: direct chip-to-chip links, 2D torus (v2, v3, v5e, v6e: 4 neighbours) or 3D torus (v4, v5p: 6); wraparound halves the worst-case distance per axis; twisted tori.
- Pods: v4 16×16×16, v5p 16×20×28 (8960 chips), built from 4×4×4 cubes joined by reconfigurable optical switches; wraparound only for multiples of a cube (v4/v5p); v5e/v6e 16×16 with wraparound only on size-16 axes; smaller slices lose wraparound (≈ 2× slower collectives).
- Bandwidth hierarchy and spec numbers: HBM ≫ ICI ≫ PCIe ≫ DCN (v5p: 2.8e12, 9e10 per link one way, 1.6e10, 6.25e9); one-way vs bidirectional bandwidth and when bidirectional is usable (needs a ring).
- Slices, multislice, DCN path (HBM → PCIe → host NIC → network → host → PCIe → HBM); "keep traffic on each channel proportional to its speed".
- TPU vs GPU networking: nearest-neighbour torus with constant links per chip vs switched hierarchy (8 or 72 GPUs per NVLink domain, O(log N) hops beyond).
- Latency + bandwidth model of a transfer (Q5: 4×4 v5e, 16.8 MB, two ports → ≈ 186 µs + 6 µs of hops); 45 kB latency threshold preview.
- Worked: pod totals (Q2: v5e 32 hosts / 256 cores / 5.0e16 FLOPs/s / 4 TB; v5p 2240 hosts / 17,920 cores / 4.1e18 / 860 TB); challenge Q6 (16 GiB host-resident int8 matrix → one chip; ICI ≈ 180 ms bottleneck, PCIe ≈ 67 ms, HBM ≈ 21 ms, FLOPs ≈ 1.4 ms).
**Figures:** mesh vs torus hop count; bandwidth ladder for one v5e chip (log scale); challenge-problem stage times.
**Questions:** PCIe crossover; hop counts; Q5 transfer time; which slice has wraparounds; pod totals; bandwidth ordering; meaning of bidi bandwidth; challenge bottleneck; TPU vs GPU networking; DCN path; latency threshold; scaling past a v5e pod. Open: design the data path for Q6; explain why a torus scales cheaply; when wraparound matters.

### 50 · `sb.sharding-notation`
**Covers:** ch. 3 "Partitioning notation and collective operations", "A unified notation for sharding", "How do we describe this in code?", "Computation with sharded arrays" (intro + Case 1), Q1.
**Syllabus**
- Why shard (memory) and why shard more (speed, latency); global/logical vs device-local shape.
- Device mesh with named axes; sharding `A[I_X, J_Y]`; replication along unused axes; multi-axis `I_XY` (order = traversal order); forbidden `A[I_X, J_X]`.
- Memory per device and total across devices (pop quizzes: fp32[1024,4096] with I_XY on {X:8,Y:2} → 1 MiB/device, ≈ 0.3 µs HBM load on H100; int8[128,2048] on {2,8,2} → 16 KiB/device, 512 KiB total); Q1 replication ratio = Y·Z = 16.
- JAX preview: `jax.make_mesh`, `PartitionSpec`, `NamedSharding`, `jit(out_shardings=…)`, `addressable_shards`.
- Elementwise ops have no overhead; block-matrix multiplication as the mental model; Case 1 (no sharded contracting dim → no comms; output inherits sharding).
**Figures:** colour-coded 2×2 mesh showing A[I,J], A[I_X,J], A[I_X,J_Y], A[I_XY,J]; block-matrix product diagram.
**Questions:** local shape from a spec; bytes per device and total; which sharding is illegal; replication factor; Case 1 output sharding; JAX P(...) to notation. Open: explain mesh/sharding to an interviewer.

### 60 · `sb.sharded-matmul`
**Covers:** ch. 3 Cases 2–4, Q4, Q9 (and ch. 4 Q2 on FLOPs with replication).
**Syllabus**
- Case 2: one input sharded on the contracting dim → AllGather it (semantics `AllGather_X: [I, J_X] → [I, J]`); alternative "local matmul then AllReduce".
- Case 3: both sharded identically on the contracting dim → local partial sums `{U_X}`; outer-product decomposition A·B = Σ_i A_{:,i} ⊗ B_{i,:}; AllReduce or ReduceScatter (choose which axis to shard, `ReduceScatter_{X,K}`).
- Case 4: same mesh axis on two non-contracting dims → only diagonal blocks computable; AllGather one input first, choice driven by downstream sharding.
- Q4/Q9: AllGather-then-matmul vs matmul-then-AllReduce/ReduceScatter; time formulas, when each wins (B vs C/W ≈ 2550 on v5p; D > 2B), why rare in practice (FSDP shards activations too); comm ratio M/K for RS variant.
- FLOPs with replication (ch. 4 Q2): total 2BDF·Z, per device 2BDF/(XY).
**Figures:** four-case decision diagram; partial-sum (outer-product) picture.
**Questions:** classify a sharded matmul into a case and name the collective; output sharding; invalid shardings; Q4 strategy comparison numbers; FLOPs per device. Open: walk through Case 3 with the outer-product argument.

### 70 · `sb.collective-costs`
**Covers:** ch. 3 "How is an AllGather actually performed?", "How long does this take?", ICI latency, multi-axis gathers, pop quiz 2, AllReduce = RS + AG, Q2, Q3, summary formulas.
**Syllabus**
- Ring AllGather: unidirectional N−1 hops of V/N; bidirectional ⌊N/2⌋ hops of 2V/N; T = V/W_bidi independent of N (derivation); empirical ≈ 95% of peak at ≈ 10 MB on v5e.
- Latency regime: T_hop = max(T_min, 2V/(N·W)); T_total = max(T_min·N/2, V/W); 45 kB threshold on v5e.
- Multi-axis: bandwidth × N_axes; T = max(T_min·Σ|X_i|/2, V/(W·N_axes)).
- ReduceScatter = same cost as AllGather; AllReduce = RS + AG = 2×; unidirectional formula with (N−1)/N.
- Pop quiz 2 (v5e {X:8,Y:4}: 34 MB → 377 µs ideal, ≈ 560 µs without wraparound, 680 µs measured; 256×256 case latency-bound ≈ 3 µs, 8 µs measured).
- Q2 (v4p 4×4×4: 23 µs, 46 µs, 11.6 µs; the "use the idle Z axis" trick → 11.5 µs / 31 µs); Q3 latency-bound ≈ 2 µs.
**Figures:** ring AllGather step diagram (uni vs bidi); T vs message size showing latency vs bandwidth regime (log-log); cost table as a figure.
**Questions:** compute AG/RS/AR times; does time depend on N; latency-bound check; multi-axis speedup; effect of missing wraparound; AR vs AG cost ratio. Open: derive the AllGather time from the ring picture.

### 80 · `sb.alltoall-overlap`
**Covers:** ch. 3 "AllToAll", "More about the ReduceScatter" (transposes), "How to overlap matmul communication with compute", "What have we learned?", Q5–Q7, Q10.
**Syllabus**
- AllToAll as moving a subscript (`[I_X, J] → [I, J_X]`), MoE motivation; cost V/(4W) on a bidirectional ring (derivations: N²/4 chunk sum; average distance N/4; bisection argument); ND formula V·max(A,B,C)/(4N·W); Q10 (unidirectional ½, bidirectional ¼).
- AllGather and ReduceScatter are transposes (broadcast = u ⊗ I, reduce = uᵀ ⊗ I) → derivative of one is the other; consequence for the backward pass.
- Deferring the AllGather (RS now, AG later); choosing the RS axis.
- Collective matmul (Wang et al.) overlapping chunks with the ring.
- Summary table of the four primitives; Q5 minimum-latency replicated output; Q6 three standard shardings on v5e 4×4 (writer derives); Q7 FFN on 2×2 with 300 MB/chip (FSDP vs TP choice; writer completes the math).
**Figures:** AllToAll chunk movement; AG/RS transpose diagram; collective-matmul timeline (overlapped vs not).
**Questions:** AllToAll cost vs AllGather; ND AllToAll; which collective is the gradient of which; Q7 memory check (536 MB per matrix). Open: derive the ¼ factor; explain collective matmul.

### 90 · `sb.flops-counting`
**Covers:** ch. 4 "Counting dots", "Forward and reverse FLOPs", Q3.
**Syllabus**
- Dot product 2P, mat-vec 2NP, matmul 2NPM; general contraction rule (2 × product of all dims, batch and contracting dims counted once; no factor 2 for pure elementwise); einsum definitions of batching vs contracting.
- Compute O(N³) vs data O(N²) → why matmul-heavy architectures scale.
- Backward: dL/dB = Aᵀ dL/dC and dL/dA = dL/dC Bᵀ, each 2NPM → 6NPM total per matmul in training; 6 · params · tokens.
- Q3 (A[I,J,K,L]·B[I,J,M,N,O] → 2·IJKLMNO).
**Figures:** einsum dimension classification diagram; forward vs backward FLOPs bars.
**Questions:** FLOPs of several einsums incl. batch dims; training vs inference FLOPs; which backward product contracts over which dim. Open: derive 6ND from a single matmul.

### 100 · `sb.transformer-accounting`
**Covers:** ch. 4 "Transformer accounting", "Global FLOPs and params", MLP, attention, other ops, rule of thumb, fractional cost of attention, Q1, Q2, Q5, Q7.
**Syllabus**
- Notation (B, T, S, D, F, N, K, H, G, L, V); gated MLP (3DF) vs plain (2DF); MHA/MQA/GQA (K = N, 1, divisor); pre-norm vs post-norm.
- MLP: 3DF params, 18BTDF training FLOPs; attention projections 2D(N+K)H params, 12BTD(N+K)H FLOPs; dot-product attention 12BT²NH (= 12BTSNH), causal halves useful FLOPs (needs a kernel); norms O(BTD); unembedding 6BTDV.
- Sum = 6 · tokens · params (ignoring attention scores); attention/matmul ratio T/(8D) with F = 4D, D = NH, N = K (derivation); 64K tokens for D = 8k, ≈ 37K for Gemma-27B; Q5 (scores = projections at T = 2D).
- Q1 (D=4096, F=4D, V=32k, L=64 → 16B params, ¼ attention, 512 KiB/token int8 KV); Q7 DeepSeek-V3 MFU ≈ 21.7% (FP8, 37B active, 14.8T tokens, 2.79M H800-hours).
**Figures:** per-layer parameter/FLOP breakdown bars; attention fraction vs T for several D.
**Questions:** param counts; FLOPs per token; T/8D crossover for given D; MFU computation; GQA effect on params. Open: derive T/8D; count LLaMA-style params from a config.

### 110 · `sb.transformer-memory`
**Covers:** ch. 4 "Sparsity and MoE", "Gradient checkpointing", "KV caching", "What should you take away", Q4, Q6, Q8.
**Syllabus**
- MoE: E experts, k active, sparsity E/k (DeepSeek-V3 k=8, E=256); params ×E, activated ×k; two AllToAlls; Q8 int8 MoE critical batch B > 120·E/k (= 3840 for DeepSeek).
- Rematerialization: O(L) memory instead of O(L²) FLOPs; ≈ 20 saved tensors per layer (why: f = exp(g) needs g and exp g); 84 TB example (BT = 4M, L = 64, D = 8192); block remat (1 checkpoint/layer, 4.2 TB, 6ND → 8ND); "big matmuls only" (≈ 7/layer); `jax.checkpoint`; Q6 extra 4BT²NH.
- KV cache shape [2, S, L, K, H]; size 2SLKH bytes in int8; 8 GiB example; why K ≪ N matters.
- Q4: GQA attention intensity: prefill TG/(G+1) = O(T); decode SG/(G+S) → G.
- Summary table (MLP 3DF / 18BTDF; attention 4DNH / 24BTDNH + 12BT²NH; vocab DV / 12BTDV).
**Figures:** remat memory vs FLOPs trade-off; KV cache size vs context for MHA vs GQA.
**Questions:** MoE critical batch; remat FLOPs multiplier; activation memory estimate; KV bytes; decode attention intensity. Open: explain remat policies and their costs.

### 120 · `sb.flash-attention`
**Covers:** ch. 4 App. A.
**Syllabus**
- The "quadratic attention" objection and its caveats (T > 8D; memory vs FLOPs).
- Online softmax: running max M, running denominator L, output O; combining chunks L = e^{M¹−max}L¹ + e^{M²−max}L²; the max-subtraction identity.
- Hardware view: Q chunk resident in VMEM/SRAM, stream KV chunks, raises intensity.
- Backward identity Σ_j S_ij dS_ij = Σ_d dO_id O_id (derivation) enabling block-local VJP and ring attention.
**Figures:** tiled attention schematic; running-statistics update.
**Questions:** combine two chunk statistics numerically; what is materialized; why the backward identity matters. Open: derive the online-softmax merge.

### 130 · `sb.data-parallelism`
**Covers:** ch. 5 "What do we mean by scaling?", notation tables, simplified MLP-only layer, the four schemes, "Data parallelism".
**Syllabus**
- Strong scaling at cluster level; notation (C, W bidirectional, X/Y/Z, M_X); MLP-only model In·W_in·W_out; forward (2 matmuls) and backward (4 matmuls) algorithm.
- Four schemes as shardings of In, W_in, W_out, Out (DP, FSDP, TP, PP).
- DP: algorithm, AllReduce in the backward only and off the critical path; memory limit (10 bytes/param with Adam → ≈ 9B params on v5p).
- Roofline: T_math = 8BDF/(XC), T_comms = 8DF/W → B/X > C/W = 2550 (v5p); 3 axes → 850 per chip, 7.6M tokens per pod.
- Context/sequence parallelism ("tokens are tokens"); critical batch size note (diminishing returns, fixed batch from scaling laws).
**Figures:** DP sharding diagram; T_math vs T_comms vs per-chip batch.
**Questions:** max model size under DP; per-chip batch threshold; effect of more mesh axes; why AllReduce is off the critical path. Open: derive B/X > C/W.

### 140 · `sb.fsdp`
**Covers:** ch. 5 "FSDP", App. A.
**Syllabus**
- ZeRO-1/2/3 (optimizer, gradients, weights); AllReduce → RS + AG at the same cost; just-in-time AllGather of weights in forward; algorithm.
- Same roofline (B/X > C/W); memory savings; why "basically always ZeRO-3".
- Examples: DeepSeek-V2 40M tokens → ≈ 47k chips; LLaMA-3 70B: 6.3e24 FLOPs, 16M batch over ≈ 18.8k chips at 50% MFU → ≈ 17 days.
- Critical-batch framing: fewer tokens per chip → comms-bound.
- App. A: deriving backward comms from the four gradient formulas and shardings.
**Figures:** FSDP sharding diagram; memory per chip DP vs ZeRO-1/2/3.
**Questions:** ZeRO stage memory; FSDP vs DP comm volume; training-time estimate; derive comms for a new sharding. Open: explain why FSDP is "free" relative to DP.

### 150 · `sb.tensor-parallelism`
**Covers:** ch. 5 "Tensor parallelism".
**Syllabus**
- Megatron sharding In[B, D_Y]·W_in[D, F_Y]·W_out[F_Y, D] → Out[B, D_Y]; AllGather before W_in, ReduceScatter after W_out (instead of two AllReduces); on the critical path.
- Roofline: T_math = 4BDF/(YC), T_comms = 4BD/W → Y < M_Y·F/2550; precision independence.
- Examples: LLaMA-3 70B (F ≈ 28.7k: 8-way fine, 16-way comms-bound); Gemma 7B (F ≈ 50k → ≈ 19-way).
**Figures:** TP sharding diagram; max TP degree vs F.
**Questions:** max TP degree; why precision cancels; critical path vs DP. Open: walk through the TP forward pass and its collectives.

### 160 · `sb.fsdp-plus-tp`
**Covers:** ch. 5 "Combining FSDP and tensor parallelism".
**Syllabus**
- Combined sharding In[B_X, D_Y]·W_in[D_X, F_Y]·W_out[F_Y, D_X]; algorithm; "FSDP moves weights, TP moves activations".
- T_FSDP = 4DF/(Y·W·M_X), T_TP = 4BD/(X·W·M_Y), T_math = 4BDF/(NC).
- X_opt = √(B·N·M_X/(F·M_Y)) by equating the two (derivation); example N = 64, B = 48k, F = 32768 → X ≈ 13.9 (16×4).
- Compute-bound iff B/N > α²/(M_X M_Y F) (α = C/W); ≈ 99 tokens/chip for F = 32768, M_X M_Y = 2 (≈ 8× better than FSDP's 850); √B scaling of compute/comms ratio; 16×16×16 example (≈ 4e5 tokens).
**Figures:** compute/comms ratio vs B for FSDP, TP, mixed (recreate the book's plot); comm time vs X at fixed N.
**Questions:** X_opt for given B, N, F; minimum per-chip batch; which regime needs mixing. Open: derive X_opt and the B/N bound.

### 170 · `sb.pipeline-and-dcn`
**Covers:** ch. 5 "Pipelining", "Scaling across pods", "Takeaways".
**Syllabus**
- Pipeline algorithm (layers over stages, pass activations forward and gradients back); pseudo-code idea; bubble; microbatching; overlapping forward, dX and dW matmuls (DeepSeek-V3 schedule); why less essential on TPUs.
- Across pods: DCN AllReduce; T_comms = 8DF/(M·W_dcn); B per ICI domain > C/W_dcn ≈ 73,440 (v5p); LLaMA-3 70B with 2M batch example (TP ≤ ≈ 11·M_Y; FSDP ≤ ≈ 2400 chips; mixed to ≈ 18k chips; 2 pods via DCN).
- Takeaway tables: compute and comms per layer for DP, FSDP, TP, FSDP+TP; thresholds (2550, 850, F/2550, 2550²/2F ≈ 100, 71–73k).
**Figures:** pipeline bubble schedule (GPipe-style vs 1F1B-style); hierarchy of thresholds.
**Questions:** bubble fraction with microbatches; DCN batch threshold; choose a strategy for a given batch/chips. Open: why pipelining is common on GPUs but not TPUs.

### 180 · `sb.training-problems`
**Covers:** ch. 5 Problems Q1–Q3 (LLaMA-2 13B: L=40, D=5120, F=13824, N=K=40, H=128, V=32k).
**Syllabus**
- Q1 params (FFW 8.5e9, attention 4.2e9, vocab 0.33e9 → 13.0e9).
- Q2 memory at 16M tokens: 130 GB params+Adam; activations 2·L·B·(D+2F) ≈ 42 TB.
- Q3 3M batch, 32k seq, v5p 16×16×16 (393 TB HBM): pure DP no (130 GB > 96 GB); pure FSDP: ≈ 8 TB fits but needs ≥ 3.48M tokens → comms-bound; mixed: threshold 2550²/(2F) ≈ 235/chip, X_opt ≈ 1333 → 1024 × 4; step ≈ 300 ms at 40% MFU.
**Figures:** memory breakdown bar; per-chip batch vs thresholds for the three options.
**Questions:** each sub-question as an MCQ; open card for the full Q3 reasoning chain.

### 190 · `sb.llama3-training-cost`
**Covers:** ch. 6 "What does LLaMA 3 look like?", "Counting parameters and FLOPs".
**Syllabus**
- Config (L 80, D 8192, F 28672, N 64, K 8, H 128, V 128,256); params: FFW 56.3e9, vocab 2.1e9, attention 12e9 → 70.4e9.
- 6 × 70e9 = 4.2e11 FLOPs/token (≈ 1 ms per token on one v5p); 15T tokens → 6.3e24 FLOPs (≈ 435 chip-years on v5p).
- Full v5p pod (8960) at 40% MFU → ≈ 44 days.
- Minimum chips by memory with 4M batch: 140 GB + 560 GB + ≈ 10.5 TB checkpoints ≈ 11.2 TB → ≈ 117 v5p (≈ 3369 days); ≈ 1.3 GB/chip on a full pod; ≈ 8ND with sparse checkpointing.
**Figures:** parameter breakdown; memory breakdown vs chip count.
**Questions:** each computation; why clusters are compute- not memory-motivated. Open: estimate a training run end to end.

### 200 · `sb.llama3-training-sharding`
**Covers:** ch. 6 "How to shard LLaMA 3-70B for training", Worked problems Q1–Q2.
**Syllabus**
- 4M tokens = 1024 × 4096: FSDP limited to 1024 chips without sequence sharding; with sequence sharding 468 tokens/chip < 850 → comms-bound; mixed threshold 2550²/(2F) ≈ 113 → OK; X_opt = √(2BN/F) ≈ 1618 → 2048 × 4 (or 1024 DP × 2 seq × 4 TP).
- Q1 (4 pods, same batch: per-chip ≈ 117 tokens, near the mixed limit; DCN per-pod batch 1M > 73k; writer derives time ≈ 11 days at 40% MFU and states assumptions).
- Q2 (LLaMA-3 405B: L 126, D 16384, F 53248, N 128, K 8, H 128, V 128,256; writer computes params ≈ 405B, FLOPs/token, total for 15T, sharding on 8 pods).
**Figures:** per-chip batch vs strategy thresholds for 1/2/4/8 pods.
**Questions:** strategy feasibility checks; X_opt; 405B param count and training time.

### 210 · `sb.inference-basics`
**Covers:** ch. 7 "The basics of Transformer inference", "What do we actually want to optimize?", "A more granular view", "Linear operations: what bottlenecks us?".
**Syllabus**
- Naive sampling cost (O(n²) FFW, O(n³) attention) → KV cache; prefill vs generation.
- Metrics: throughput per chip, TTFT, per-token latency; offline vs chat vs edge.
- Linear ops: B_crit = 240 (v5e bf16), 120 (int8 weights, bf16 math), 240 (int8 both), ≈ 280 on H100; prefill is compute-bound; generation must batch requests to approach B_crit.
**Figures:** naive vs cached sampling; prefill vs decode on the roofline.
**Questions:** B_crit by precision; is prefill of N tokens compute-bound; why decode needs batching. Open: explain prefill vs decode bottlenecks.

### 220 · `sb.inference-latency`
**Covers:** ch. 7 "What about attention?", "Theoretical estimates for LLM latency and throughput", pop quiz, App. A.
**Syllabus**
- Attention intensity ST/(S+T): T/2 at prefill, ≈ 1 at decode; why (each sequence has its own KV); diminishing returns when KV bytes ≈ param bytes (ratio ≈ 2DF/(SHK)).
- Min step time = (B·KV + params)/BW; max throughput; general formula with MLP max(FLOPs, params) term.
- Pop quiz: 30B int8 on v5e 4×4, 8k context, 100 kB/token → ≈ 2.5 ms at B=4, ≈ 21 ms at B=256.
- Latency/throughput Pareto (ESTI figure); nits (imperfect overlap); App. A empirical layer time flat to ≈ 240.
**Figures:** step time vs batch with parameter/KV/FLOPs components; throughput vs latency curve.
**Questions:** step-time computations; when attention dominates; throughput at large batch. Open: derive the step-time formula.

### 230 · `sb.inference-memory`
**Covers:** ch. 7 "What about memory?", "Modeling throughput and latency for LLaMA 2-13B", "Tricks for improving generation throughput and latency".
**Syllabus**
- Inference memory = params + KV cache (no optimizer, negligible activations; 80 MB for an 8k prefill activation).
- KV size 2·bytes·H·K·L·T; LLaMA-2 13B: 6.7 GB per 8k sequence; table of step time/throughput for B = 1…240 on 8 v5e (writer recomputes: 4.98 ms … 249 ms; OOM beyond 16) and with 5× smaller KV.
- Tricks: GQA/MQA, local attention layers, cross-layer KV sharing (doesn't always reduce step time), quantization, ragged reads and PagedAttention.
**Figures:** throughput vs batch for MHA vs GQA (recompute both tables); KV memory vs batch.
**Questions:** KV sizes; max batch that fits; which trick reduces step time vs only memory. Open: compare KV-reduction techniques.

### 240 · `sb.inference-sharding`
**Covers:** ch. 7 "Distributing inference over multiple accelerators" (prefill, generation, KV sharding), App. B, App. C, Q7.
**Syllabus**
- Prefill: model parallelism to the ICI bound (≈ F/2200 on one axis), then sequence parallelism (ring attention).
- Generation: no FSDP (move activations, not weights), no DP, no sequence sharding; model sharding beyond the FLOPs-ICI bound when memory-bound: Y > F/(B·β), β = W_hbm/W_ici ≈ 8 → 64-way for F = 16384, B = 32.
- Attention: shard W_Q/W_O over heads; replicate small KV weights; KV cache KV[2, B_Z, S, K_Y, H] with two AllToAlls per layer (algorithm); sequence-sharded KV.
- App. C latency-bound collectives (buffer < 45 kB per hop; 8-way → < 360 kB; B=16, D=8192 int8 = 131 kB is latency-bound; Y > BD/45,000).
- App. B / Q7 2D weight-stationary: comms 2BD/(X·W) + 4BF/(YZ·W); optimum X = √(N/8); beats 1D when N > 72 (F = 4D).
**Figures:** 1D vs 2D weight-stationary layouts; comm time vs N for both.
**Questions:** max useful model parallelism at given B; why FSDP is wrong for decode; latency-bound check; 2D vs 1D crossover. Open: design a decode sharding for a given model and slice.

### 250 · `sb.inference-engines`
**Covers:** ch. 7 "Designing an effective inference engine", "Continuous batching", "Prefix caching", "JetStream", App. D.
**Syllabus**
- Batched prefill+generate and its four drawbacks; interleaved (prefill at batch 1, batched decode) and its jitter; disaggregated serving (pros, KV transfer cost).
- Continuous batching (prefill function + generate function + orchestrator); prefix caching (chatbots, few-shot; HBM and host DRAM; affinity routing; trie + LRU).
- JetStream engine API (prefill / insert / generate; prefill, generate and transfer threads).
- Speculative sampling: greedy version, why it's a latency win (not compute-bound), throughput win at long context, Metropolis-Hastings style acceptance for sampling, drafter heads (Medusa, EAGLE, DeepSeek MTP); Chinchilla 70B + 4B drafter example.
**Figures:** three serving layouts; speculative decoding timeline.
**Questions:** which design fixes which drawback; prefix-cache savings; when speculation helps. Open: design a serving system for a chat product.

### 260 · `sb.inference-problems`
**Covers:** ch. 7 Worked problems Q1–Q6 (invented model: L 64, D 4096, F 16384, N 32, K 8, H 256, V 32,128).
**Syllabus**
- Q1 params 18.4e9, KV 262 kB/token int8; Q2 max batch at 128k on v5e 4×4 = 7 (56 with K=1); Q3 param load ≈ 1.4 ms.
- Q4 sharding for prefill/decode on 4×4 (writer derives: 2D ICI without wraparound, TP bound, KV sharding over heads then batch; per-step latency).
- Q5 MoE (E=16, k=2): 212e9 total, 31.2e9 active, B > 1920, same KV, 2·31.2e9·T FLOPs.
- Q6 expert sharding on v5e 8×16 (writer derives weight-load time and free HBM; smallest slice).
**Questions:** each part as an MCQ; open cards for Q4 and Q6.

### 270 · `sb.llama3-serving`
**Covers:** ch. 8 "What's the LLaMA serving story?", "Thinking about throughput".
**Syllabus**
- Hardware by FLOPs/$ (H100 3.3e17, v5p 3.9e17, v5e 5.8e17 per $; Feb 2025 prices).
- KV 160 kB/token int8; 5.3 GB per 32k sequence; BS 240 → 1.3 TB → ≈ 86 v5e.
- BS 32 @ 8k int8: 42 GB KV + 70 GB params = 112 GB → 4×2 tight, 4×4 safer; step ≈ 17 ms, ≈ 235 tok/s/chip; 4×4 halves latency, same per-chip throughput; ICI check Y < 2F/2200 ≈ 26.
- B_crit 240 / 120 / 240 by precision; min topologies bf16/int8/int4 = 16/8/4 chips with 43 KV caches at 8k; 19 ms per step when HBM is full; QPS/chip 0.27/0.55/1.11; doubling topology → 0.44/0.90/1.80 ("smallest topology is not the most efficient").
**Figures:** min-topology vs dtype; QPS/chip vs topology.
**Questions:** KV per token; memory totals; step time and throughput; QPS per chip. Open: pick a topology for a throughput target.

### 280 · `sb.llama3-serving-tradeoffs`
**Covers:** ch. 8 sharding question, "What about prefill?", "Visualizing the latency throughput tradeoff", Worked problems Q1–Q3.
**Syllabus**
- Model parallel limit Y ≈ F·M_Y/2200 ≈ 26 → 4×4 OK, 4×8 not for bf16 at large batch; small-batch exception Y = F/(8B) ≈ 56 at B = 64; T_ici ≈ 11 µs, T_hbm ≈ 18 µs, T_math ≈ 4 µs on 4×8; quantization enables more model parallelism.
- Prefill 8192 tokens on 16 v5e at 40% MFU ≈ 0.91 s; KV evictions per step B(P+G)/G = 96; disaggregated ratio P ≈ 3G.
- Latency-throughput Pareto for 16-way TP on v5e 4×4 (recreate with the book's code: param-load, KV-load and FLOPs components; KV loading dominates beyond 2k context; ≈ 100× cost range).
- Problems Q1–Q3 (405B FLOPs/token and bounds; 8B at BS 240 memory and topology; 405B under 15 ms/token; writer derives).
**Figures:** Pareto frontier at several context lengths; step-time breakdown vs batch.
**Questions:** sharding feasibility; prefill time; server ratio; reading the Pareto plot. Open: serve 405B under a latency cap.

### 290 · `sb.profiling`
**Covers:** ch. 9 (all).
**Syllabus**
- Stack: JAX → StableHLO → HLO (XLA passes: fusion, layout) → LLO → machine code; Pallas kernels as escape hatch; `jit(f).lower(...).compile().as_text()`.
- `jax.profiler.trace`, TensorBoard/XProf tabs: Trace Viewer (XLA Ops row, `named_scope`), Graph Viewer, Memory Profile/Viewer.
- Reading an XLA op: name, shape, layout `{2,1,0:T(8,128)(2,1)}`, memory space S(1) = VMEM; tiling and padding (f32[3,5] with T(2,2) → [4,6], 1.6×); retiling copies.
- Worked profile: 4-way DP × 2-way MP matmul predicted 95.6 ms vs 96 ms measured (TPU v2, 23e12 FLOPs/s per core); ReduceScatter ≈ 1.1 ms predicted vs 1.13 ms; Q1 recovering shapes and 8-way MP from replica groups; Q2 fixing sharding with `with_sharding_constraint`.
**Figures:** annotated anatomy of an HLO op string; tiling/padding diagram.
**Questions:** decode a layout string; memory space; predicted vs measured op time; padding overhead. Open: how you'd debug a slow step.

### 300 · `sb.jax-sharding-modes`
**Covers:** ch. 10 "How does parallelism work in JAX?" (Auto, Explicit, Manual).
**Syllabus**
- Three modes table (view, explicit sharding, explicit collectives); `jax.make_mesh` with axis types; `jax.set_mesh`; `NamedSharding`/`P`.
- Auto: jit + Shardy inserts collectives (AllReduce example and its HLO); `with_sharding_constraint`.
- Explicit: sharding in types (`jax.typeof` → `float32[8@X,2@Y]`), ambiguous einsum error and `out_sharding` choice (RS vs AR); `auto_axes`/`explicit_axes`.
- Manual: `shard_map` local view; `psum`, `pmean`, `all_gather`, `psum_scatter`, `all_to_all`, `ppermute`; the slice-and-average example semantics.
**Figures:** global vs per-device view; mapping lax collectives to the four primitives.
**Questions:** which collective a `jit` inserts; what `shard_map` returns; map lax op → primitive; explicit-mode error resolution. Open: when to drop to shard_map.

### 310 · `sb.jax-collective-matmul`
**Covers:** ch. 10 collective matmul example, Problems Q1–Q4.
**Syllabus**
- Collective matmul for A[B_X, D_Y]·W[D, F_Y]: per-step local matmul + `ppermute`; 311 µs (blocking AllGather) → 244 µs vs 224 µs unsharded baseline.
- Problems: per-shard averages (jit vs shard_map); roll within shards; MoE routing (mask-per-expert vs sort + `ragged_dot`; AllGather vs AllToAll inside a loop; top-k); AllReduce and ReduceScatter collective matmuls; bidirectional variants.
**Figures:** collective-matmul ring steps; blocked vs overlapped timeline.
**Questions:** what each ppermute step computes; why the MoE jit version gathers activations; complexity of masking vs sorting. Open: sketch a ReduceScatter collective matmul.

### 320 · `sb.capstone-review`
**Covers:** ch. 11 (conclusions, further reading) plus a cross-chapter recap.
**Syllabus**
- The numbers and rules worth memorizing: ridges (240/295), B_crit by precision, collective costs (V/W, 2V/W, V/4W), latency threshold, 6ND, T/8D, KV formula, DP/FSDP 2550 (850), TP F/2550, mixed 2550²/2F, DCN 73k, decode step-time formula, Y > F/(Bβ), GPU equivalents (2200/2475, F/2475).
- Further reading map (TPU deep dive, ESTI paper, Ultra-Scale Playbook, brrr, CS336, Pallas docs, collective-ops blog).
**Figures:** one-page "threshold ladder"; strategy decision tree (batch per chip vs model size).
**Questions:** mixed-chapter estimation problems (training time, serving latency, sharding choice). Open: a 5-minute "how would you train/serve model X on Y chips" answer.

### 330 · `sb.gpu-chip`
**Covers:** ch. 12 "What is a GPU?", "Memory", "Summary of GPU specs", "GPUs vs TPUs at the chip level", Quiz 1.
**Syllabus**
- SMs (132 on H100, 148 on B200), 4 subpartitions each with a Tensor Core, warp scheduler, 32 fp32 CUDA cores, 16k registers; TC ≈ 1024 bf16 FLOPs/cycle on H100; TMEM on Blackwell.
- SIMT vs SIMD, warp divergence, many resident warps; memory hierarchy (registers 256 kB/SM, SMEM 256 kB/SM, L2 ≈ 50 MB at ≈ 5.5 TB/s, HBM 3.35 TB/s H100, 8 TB/s B200).
- Spec tables (V100–B200); GPU↔TPU cheat sheet; modularity (132 SMs vs 2 TensorCores); TPUs have more fast on-chip memory.
- Quiz 1: 16,896 CUDA cores (H100) vs 8,192 VPU ALUs (v5p); vector FLOPs ≈ 26.9 TFLOP/s (no FMA); ridges 295 (H100), 281 (B200), ×2 for fp8; B200 matmul times ≈ 8.6 µs / 15 µs; SMEM total ≈ 33 MB; B200 clock ≈ 2.1 GHz; fp32 add intensity 1/12.
**Figures:** SM block diagram; GPU vs TPU unit-count comparison.
**Questions:** Quiz 1 items as MCQs. Open: compare GPU and TPU design philosophies.

### 340 · `sb.gpu-networking`
**Covers:** ch. 12 "Networking", "At the node level", "Beyond the node level", GB200 NVL72, Quizzes 2–3, App. B.
**Syllabus**
- Node = NVLink domain (8 GPUs; 72 in GB200 NVL72); H100: 18 NVLink4 links × 25 GB/s = 450 GB/s egress per GPU; 4 NVSwitches; 3.6 TB/s node bandwidth; bisection 3.6 TB/s.
- Scale-out InfiniBand fat tree: SU of 32 nodes under 8 leaf switches; 4 SUs + 16 spines = 1024 GPUs; 400 GB/s per node at every level (full bisection); 50 GB/s per GPU beyond the node.
- GB200 NVL72: 900 GB/s per GPU, 3.6 TB/s node egress; growing beyond 1024 GPUs (more spines, core switches).
- Quiz 2 (node AllGather of bf16[4096, 65536] ≈ 1.04 ms) and Quiz 3 (fat-tree bandwidth bookkeeping).
**Figures:** node and SuperPod topology; bandwidth per level table as a figure.
**Questions:** node bandwidth; bisection; per-GPU bandwidth beyond the node; GB200 change. Open: explain a fat tree vs a torus.

### 350 · `sb.gpu-collectives`
**Covers:** ch. 12 "How do collectives work on GPUs?" (intra-node, SHARP, cross-node, sharded reductions), Quiz 4.
**Syllabus**
- Intra-node AG/RS = bytes·(N−1)/(N·W_GPU) ≈ B/450e9 on H100; AllReduce 2×; tree vs ring for latency.
- AllToAll within node B(N−1)/(W N²) ≈ B/(8W) (2× better than TPU's B/4W); ragged top-k AllToAll; empirical ≈ 370 GB/s (vs 450) and large-message requirement.
- SHARP in-network reduction: theory B/W (2×), practice ≈ +30%.
- Cross-node: B/W_node (400 GB/s); per-level formula (514 / 413 / 17.1 TB/s → leaf-bound); AllToAll across M nodes ≈ B/(M·W_node) (> 4× slower than in-node); reductions with an inner sharded axis (benefit only when Y spans nodes).
- Quiz 4 (switch ingress/egress counts, SHARP AR, spine-level AR, 2-node AllGather bounded by the node level).
**Figures:** collective time vs message size (latency/BW regimes) H100 vs TPU; in-network reduction flow.
**Questions:** compute collective times in and across nodes; SHARP expectations; 2-node AG. Open: why GPUs need larger messages to reach peak bandwidth.

### 360 · `sb.gpu-llm-rooflines`
**Covers:** ch. 12 "Rooflines for LLM scaling on GPUs" (DP, TP, EP, PP), Examples (DeepSeek-V3, LLaMA-3), TLDR, Quiz 5, App. A.
**Syllabus**
- DP/FSDP: B/X > C/W: 2200 in-node, 2475 across nodes (H100); MoE × E/k (≈ 79,200 for k=4, E=128); 2-node DP halves the threshold (≈ 1237).
- TP: Y < F·W/C ≈ F/2475 → 8-way in a node, up to ≈ 16 with exactly 2 nodes.
- EP: AllToAll cost model and the two feasible regimes (F < 8C/W_node: ≈ 2-node EP; larger F: many-node EP).
- PP: tiny comms (≈ 1.5·2BD/(W·L) per layer) but code complexity, ZeRO-3 incompatibility, bubbles, critical-path transfers; zero-bubble schedules.
- Examples: DeepSeek-V3 (64-way EP over 8 nodes, 16-way PP, 2-way ZeRO-1; 62.9M-token batch, 30k/GPU); LLaMA-3 (8-way TP, 16-way PP, 128-way ZeRO-1; 16M batch).
- Quiz 5: B200 rooflines (TP ≈ F/2500; DP 5625 across nodes); LLaMA-3 70B on H100 (≥ 9 GPUs for weights+Adam; ≈ 40 days on 4096 at 45% MFU; 8-way TP + DP doesn't fit; ZeRO-3 comms-bound at 976 tokens/GPU; 8-way PP raises leaf bandwidth to 3.2 TB/s → threshold ≈ 309); Megatron-LM batch sizes (4096/2048/2048 tokens per GPU).
**Figures:** threshold comparison TPU v5p vs H100 vs B200; DeepSeek/LLaMA mesh layouts.
**Questions:** Quiz 5 items; choose a GPU sharding for a given model. Open: compare how you'd shard a dense 70B on TPU v5p vs H100.

---

## Items deliberately left out (whole series)
- Plotly interactive figures and GIF animations: re-drawn as static figures where useful.
- Exact JAX code listings: summarized, with only the API names and semantics needed to answer questions (the app shows short code blocks only).
- Hardware prices: used once (lesson 270) with the book's Feb 2025 date attached; not memorized.
