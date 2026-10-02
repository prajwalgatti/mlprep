# Source cache index: Applied ML & Systems (`sys.*`)

Plain-text extracts for the Applied ML & Systems area (plan: `docs/content-plan-applied.md`). Built 2026-10-02.
All files are in this directory with the prefix `sys_`. Original PDFs are in `sys_pdf/`, raw HTML in `sys_html/`.
The first line of every file is `SOURCE: <url>`. Search with `grep -n` (use `grep -a` for the PDF-derived papers;
a few contain odd bytes and grep otherwise reports "binary file matches").

## Extraction quality (read first)

- **PDF papers** (pdftotext, one marker per page like `=== [rajbhandari2020_zero p.14] ===`): prose is fine,
  **equations are mangled** (sub/superscripts on separate lines, fractions flattened). Section numbers usually sit on
  their own line with the title on the next line, so grep for the title text. Narayanan 2021 uses Unicode math-italic
  letters (e.g. `𝑝`, `𝑚`), so grep for words, not symbols. Check the PDF in `sys_pdf/` for any formula you quote.
- **Ultra-Scale Playbook** (`sys_ultrascale2025_playbook.txt`): the PDF is one very tall page, so there are no page
  markers. Side-notes are interleaved with the body text. Use the line ranges below.
- **Web pages** (Scaling Book, PyTorch/JAX/TF/NVIDIA docs, blogs): LaTeX is preserved where the page used MathJax/KaTeX
  (`\[...\]`, `$...$`). Headings are `#`-prefixed. These are the best files for copying exact formulas.
- **Bengio et al. 1994** is a scan; the text is tesseract OCR. Prose is readable, equations are not.
- **Goldberg 1991**: the PDF has broken font encoding (kept as `sys_pdf/..._GARBLED_FONTS.pdf`); use the Oracle HTML
  extract `sys_goldberg1991_floating_point.txt`, which is clean.

## Books, long-form guides and blogs

| File | Source | URL | Where things are |
|---|---|---|---|
| `sys_scalingbook_roofline.txt` | Austin et al., *How To Scale Your Model* (JAX Scaling Book), Part 1 "All About Rooflines" (2025) | https://jax-ml.github.io/scaling-book/roofline/ | "Where Does the Time Go?", "Visualizing rooflines", "Matrix multiplication" (intensity ≈ B), "Network communication rooflines", worked problems |
| `sys_scalingbook_tpus.txt` | Scaling Book Part 2 "How to Think About TPUs" | https://jax-ml.github.io/scaling-book/tpus/ | TPU networking (ICI torus), spec table; systolic array appendix |
| `sys_scalingbook_sharding.txt` | Scaling Book Part 3 "Sharded Matrices and How to Multiply Them" | https://jax-ml.github.io/scaling-book/sharding/ | sharding notation; the 4 matmul cases; AllGather/ReduceScatter/AllReduce/AllToAll costs; overlap; problems |
| `sys_scalingbook_transformers.txt` | Scaling Book Part 4 "All the Transformer Math You Need to Know" | https://jax-ml.github.io/scaling-book/transformers/ | "Forward and reverse FLOPs" (2/4 rule → 6ND), MLP/attention FLOPs and params, "Gradient checkpointing", attention fraction vs context |
| `sys_scalingbook_training.txt` | Scaling Book Part 5 "How to Parallelize a Transformer for Training" | https://jax-ml.github.io/scaling-book/training/ | Data Parallelism, FSDP, Tensor Parallelism, FSDP+TP, Pipelining, Scaling Across Pods; Appendix A backward-pass comms |
| `sys_scalingbook_gpus.txt` | Scaling Book Part 12 "How to Think About GPUs" (Aug 2025) | https://jax-ml.github.io/scaling-book/gpus/ | GPU spec tables (H100 9.9e14 bf16 FLOP/s, 3.4e12 B/s HBM); NVLink/NVSwitch node (450 GB/s/GPU), IB fabric (400 GB/s per node); "How Do Collectives Work on GPUs?"; "Rooflines for LLM Scaling on GPUs" (DP needs ≈2500 tokens/GPU, TP ≲ F/2475, PP, EP); DeepSeek-V3 and LLaMA-3 configs |
| `sys_scalingbook_profiling.txt` | Scaling Book Part 9 "How to Profile TPU Programs" | https://jax-ml.github.io/scaling-book/profiling/ | XLA/HLO stack, JAX profiler, Trace Viewer, reading an XLA op, memory profile |
| `sys_scalingbook_jax_stuff.txt` | Scaling Book Part 10 "Programming TPUs in JAX" | https://jax-ml.github.io/scaling-book/jax-stuff/ | auto / explicit sharding modes, `shard_map`, worked problems |
| `sys_ultrascale2025_playbook.txt` | Tazi et al., *The Ultra-Scale Playbook: Training LLMs on GPU Clusters*, Hugging Face (Feb 2025) | https://huggingface.co/spaces/nanotron/ultrascale-playbook | Lines (approx.): memory usage & formulas 471–830; activation recomputation 835–1060; gradient accumulation 1061–1148; data parallelism (bucketing, overlap) 1149–1495; ZeRO 1–3 1496–1853; tensor parallelism 1854–2200; sequence parallelism 2201–2532; context parallelism / ring attention 2533–2729; pipeline parallelism (AFAB, 1F1B, interleaving, zero bubble, DualPipe) 2730–3226; expert parallelism 3227–3263; 5D parallelism 3264–3903; finding a config 3904–4110; GPU primer & kernels 4111–4485; fused kernels & FlashAttention 4486–4584; mixed precision & FP8 4585–5147; A0 collectives crash course (incl. ring all-reduce) 5149–5503; A1 profiling 5504–5647; A2 typical scales 5648–5681; A3 overlap maths 5682–end |
| `sys_he2022_brrr.txt` | H. He, "Making Deep Learning Go Brrrr From First Principles" (2022) | https://horace.io/brrr_intro.html | Compute, Bandwidth (operator fusion, "Reasoning about Memory-Bandwidth Costs"), Overhead |
| `sys_goldberg1991_floating_point.txt` | D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM Computing Surveys 1991 (Oracle reprint) | https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html | Rounding Error › Floating-point Formats / Relative Error and Ulps / Guard Digits / Cancellation; The IEEE Standard › Formats and Operations, Special Quantities, NaNs, Infinity, Signed Zero, Denormalized Numbers, Rounding Modes; Systems Aspects › Optimizers › Theorem 8 (Kahan summation) |
| `sys_d2l_rnn_scratch_clipping.txt` | d2l.ai §9.5 RNN from scratch | https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html | §9.5.3 Gradient Clipping (LaTeX preserved) |
| `sys_hf2024_gradient_accumulation_fix.txt` | Hugging Face blog, "Fixing Gradient Accumulation" (Oct 2024) | https://huggingface.co/blog/gradient_accumulation | token-count normalisation bug with gradient accumulation |
| `sys_wiki_variance_algorithms.txt` | Wikipedia, "Algorithms for calculating variance" | https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance | naive algorithm, two-pass, Welford's online algorithm, parallel (Chan) combination |
| `sys_wiki_kahan.txt` | Wikipedia, "Kahan summation algorithm" | https://en.wikipedia.org/wiki/Kahan_summation_algorithm | algorithm, error bound, pairwise summation |
| `sys_wiki_bfloat16.txt` | Wikipedia, "bfloat16 floating-point format" | https://en.wikipedia.org/wiki/Bfloat16_floating-point_format | bit layout, exponent encoding, special values (cross-check only) |
| `sys_wiki_half_precision.txt` | Wikipedia, "Half-precision floating-point format" | https://en.wikipedia.org/wiki/Half-precision_floating-point_format | binary16 layout, exponent table, precision limits (cross-check only) |

## Framework and vendor documentation

| File | Source | URL |
|---|---|---|
| `sys_pytorch_ddp_note.txt` | PyTorch note: Distributed Data Parallel (design: broadcast, buckets in reverse order, autograd hooks, find_unused_parameters, DDPOptimizer) | https://docs.pytorch.org/docs/stable/notes/ddp.html (cached from /docs/2.14/) |
| `sys_pytorch_fsdp2_fully_shard.txt` | PyTorch `fully_shard` (FSDP2): DTensor dim-0 sharding, grouping, AG/RS streams, prefetching, reshard_after_forward | https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html |
| `sys_pytorch_fsdp1.txt` | PyTorch FSDP1 API (`FullyShardedDataParallel`): ShardingStrategy (FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD), BackwardPrefetch, CPUOffload, MixedPrecision, auto_wrap_policy | https://docs.pytorch.org/docs/stable/fsdp.html |
| `sys_pytorch_dtensor.txt` | PyTorch DTensor (Shard/Replicate/Partial placements, DeviceMesh) | https://docs.pytorch.org/docs/stable/distributed.tensor.html |
| `sys_pytorch_tp_tutorial.txt` | PyTorch tutorial: Tensor Parallel (Colwise/Rowwise, sequence parallel, loss parallel, 2-D FSDP+TP) | https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html |
| `sys_pytorch_amp.txt` | `torch.amp`: autocast, GradScaler (init_scale 65536, growth 2, backoff 0.5, interval 2000), autocast op lists | https://docs.pytorch.org/docs/stable/amp.html |
| `sys_pytorch_amp_examples.txt` | AMP examples: unscale before clipping, gradient accumulation, multiple optimizers, DDP | https://docs.pytorch.org/docs/stable/notes/amp_examples.html |
| `sys_pytorch_numerical_accuracy.txt` | PyTorch note: numerical accuracy (batched vs sliced results, TF32, reduced-precision reductions in fp16/bf16 GEMMs) | https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html |
| `sys_pytorch_randomness.txt` | PyTorch note: reproducibility (nondeterministic algorithms, `use_deterministic_algorithms`) | https://docs.pytorch.org/docs/stable/notes/randomness.html |
| `sys_pytorch_cuda_semantics.txt` | PyTorch note: CUDA semantics (asynchronous execution, streams, TF32 flags, caching allocator, CUDA graphs section) | https://docs.pytorch.org/docs/stable/notes/cuda.html |
| `sys_pytorch_checkpoint.txt` | `torch.utils.checkpoint` (use_reentrant, preserve_rng_state, checkpoint_sequential) | https://docs.pytorch.org/docs/stable/checkpoint.html |
| `sys_pytorch_clip_grad_norm.txt` | `clip_grad_norm_` API (total norm over all params as one vector) | https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html |
| `sys_pytorch_compiler_overview.txt` | torch.compiler overview | https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html |
| `sys_pytorch_compile_api.txt` | `torch.compile` API (fullgraph, dynamic, mode, backend) | https://docs.pytorch.org/docs/stable/generated/torch.compile.html |
| `sys_pytorch_compile_dynamo_core.txt` | Dynamo Core Concepts: bytecode tracing, graph breaks, guards, recompilations, dynamic shapes | https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html |
| `sys_pytorch_compile_graph_breaks.txt`, `sys_pytorch_compile_common_graph_breaks.txt` | Working with graph breaks; common graph breaks (data-dependent ops, printing) | .../programming_model.graph_breaks_index.html, .../programming_model.common_graph_breaks.html |
| `sys_pytorch_compile_recompilation.txt` | Dealing with recompilations (int specialisation, cache_size_limit) | .../programming_model.recompilation.html |
| `sys_pytorch_compile_dynamic_shapes.txt` | Dynamic shapes core concepts (automatic dynamic, 0/1 specialisation) | .../dynamic_shapes_core_concepts.html |
| `sys_pytorch_compile_tutorial.txt` | Intro to torch.compile tutorial (speedups, benefits over TorchScript, graph breaks) | https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html |
| `sys_pytorch_cuda_graphs_blog.txt` | PyTorch blog, "Accelerating PyTorch with CUDA Graphs" (2021) | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ |
| `sys_pytorch_profiler_recipe.txt` | PyTorch Profiler recipe (record_function, key_averages, memory, trace export, schedule) | https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html |
| `sys_pytorch_tuning_guide.txt` | PyTorch Performance Tuning Guide (data loading, avoiding syncs, fusion, channels_last, DDP tips) | https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html |
| `sys_jax_key_concepts.txt` | JAX key concepts: transformations, tracing, jaxprs, pytrees | https://docs.jax.dev/en/latest/key-concepts.html |
| `sys_jax_jit.txt` | JAX JIT compilation: tracing, why not jit everything, static args, caching | https://docs.jax.dev/en/latest/jit-compilation.html |
| `sys_jax_sharp_bits.txt` | JAX — The Sharp Bits: pure functions, in-place updates, OOB indexing, control flow, NaN debugging, float64 | https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html |
| `sys_jax_random.txt` | JAX pseudorandom numbers: explicit keys, split, no sequential equivalence | https://docs.jax.dev/en/latest/random-numbers.html |
| `sys_jax_pytrees.txt` | JAX pytrees | https://docs.jax.dev/en/latest/pytrees.html |
| `sys_jax_parallel.txt` | JAX distributed arrays and automatic parallelisation: Mesh, NamedSharding, explicit/auto/manual (shard_map) modes | https://docs.jax.dev/en/latest/parallel.html (old URL sharded-computation.html redirects here) |
| `sys_jax_checkpoint.txt` | `jax.checkpoint` / remat: residuals, policies (dots_with_no_batch_dims_saveable), offload, recursive checkpointing | https://docs.jax.dev/en/latest/gradient-checkpointing.html |
| `sys_jax_profiling.txt` | JAX profiling (Perfetto, XProf/TensorBoard, Nsight) | https://docs.jax.dev/en/latest/profiling.html |
| `sys_tf_function.txt` | TensorFlow guide: Better performance with tf.function (tracing, retracing rules, input_signature, reduce_retracing, side effects) | https://www.tensorflow.org/guide/function |
| `sys_tf_intro_graphs.txt` | TensorFlow guide: Introduction to graphs and tf.function (AutoGraph, polymorphism, ConcreteFunction) | https://www.tensorflow.org/guide/intro_to_graphs |
| `sys_nvidia_mixed_precision.txt` | NVIDIA, *Train With Mixed Precision* user guide (§2 loss scaling incl. choosing a scale factor, §3 AMP, §4 Tensor Core shape rules, FAQ) | https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html |
| `sys_nvidia_gpu_perf_background.txt` | NVIDIA, *GPU Performance Background* (§4 math vs memory bound, arithmetic intensity table, §5 operation categories) | https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html |
| `sys_nvidia_matmul_perf.txt` | NVIDIA, *Matrix Multiplication Background* (§2 math and memory bounds of GEMMs, §3 tile/wave quantisation) | https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html |
| `sys_nvidia_nccl_collectives.txt` | NCCL user guide: Collective Operations (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather, Scatter) | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html |
| `sys_nccl_tests_performance.txt` | nccl-tests `PERFORMANCE.md`: algbw vs busbw and the 2(n−1)/n, (n−1)/n factors per collective | https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md |
| `sys_nvidia2019_nccl_double_binary_tree.txt` | NVIDIA blog, "Massively Scale Your Deep Learning Training with NCCL 2.4" (2019): double binary trees (full bandwidth, log latency), hierarchical/2D rings | https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/ |
| `sys_nvidia_te_fp8_primer.txt` | NVIDIA Transformer Engine, "Using FP8 with Transformer Engine" (E4M3/E5M2, just-in-time vs delayed scaling, amax history, MXFP8 block scaling) | https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.0/user-guide/examples/fp8_primer.html (the unversioned URL now 404s) |

## Papers (`sys_<firstauthor><year>_<slug>.txt`)

| File | Paper | URL | Key sections |
|---|---|---|---|
| `sys_micikevicius2018_mixed_precision.txt` | Micikevicius et al., Mixed Precision Training (ICLR 2018) | https://arxiv.org/abs/1710.03740 | §3.1 FP32 master copy, §3.2 loss scaling (gradient histogram), §3.3 arithmetic precision |
| `sys_kalamkar2019_bf16.txt` | Kalamkar et al., A Study of BFLOAT16 for Deep Learning Training (2019) | https://arxiv.org/abs/1905.12322 | §2–3 format and why no loss scaling; §4 results |
| `sys_micikevicius2022_fp8.txt` | Micikevicius et al., FP8 Formats for Deep Learning (2022) | https://arxiv.org/abs/2209.05433 | §2 scaling factors, §3 E4M3 (max 448, single NaN pattern) / E5M2 (IEEE-like, max 57344), §4 results |
| `sys_gupta2015_limited_precision.txt` | Gupta et al., Deep Learning with Limited Numerical Precision (ICML 2015) | https://arxiv.org/abs/1502.02551 | stochastic rounding definition and its unbiasedness, fixed-point training |
| `sys_zamirai2020_revisiting_bf16.txt` | Zamirai et al., Revisiting BFloat16 Training (2020) | https://arxiv.org/abs/2010.06192 | why pure-bf16 weight updates stall (nearest rounding cancels small updates); Kahan summation and stochastic rounding fixes |
| `sys_blanchard2021_logsumexp.txt` | Blanchard, Higham & Higham, Accurately Computing the Log-Sum-Exp and Softmax Functions (IMA J. Numer. Anal. 2021; MIMS eprint) | https://doi.org/10.1093/imanum/draa038 | shifted LSE, error analysis, softmax variants |
| `sys_wortsman2023_small_scale_instabilities.txt` | Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities (2023) | https://arxiv.org/abs/2309.14322 | §3.1.1 attention logit growth & qk-layernorm, §3.1.2 output logit divergence & z-loss (1e-4·log²Z), §3.4 AdamW ε vs gradient RMS |
| `sys_bengio1994_long_term_deps.txt` | Bengio, Simard & Frasconi, Learning Long-Term Dependencies with Gradient Descent is Difficult (IEEE TNN 1994), OCR | https://doi.org/10.1109/72.279181 | the robust-latching vs vanishing-gradient trade-off |
| `sys_zhang2020_why_clipping.txt` | Zhang, He, Sra & Jadbabaie, Why Gradient Clipping Accelerates Training (ICLR 2020) | https://arxiv.org/abs/1905.11881 | (L0, L1)-smoothness; clipped GD vs fixed-step GD |
| `sys_brock2021_nfnets_agc.txt` | Brock et al., High-Performance Large-Scale Image Recognition Without Normalization (NFNets, 2021) | https://arxiv.org/abs/2102.06171 | §4 Adaptive Gradient Clipping (unit-wise ‖G‖/‖W‖ ratio) |
| `sys_chen2016_sublinear_memory.txt` | Chen, Xu, Zhang & Guestrin, Training Deep Nets with Sublinear Memory Cost (2016) | https://arxiv.org/abs/1604.06174 | √n segment schedule with one extra forward; recursive O(log n) variant |
| `sys_korthikanti2022_seqpar_recompute.txt` | Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (2022) | https://arxiv.org/abs/2205.05198 | §4.1 activation memory per layer sbh(34+5as/h) itemised, §4.2.1 with TP, §4.2.2 sequence parallelism (g/ḡ = AG/RS), §4.2.3 PP first-stage memory, §5 selective recomputation (GPT-3: 70% saved for 2.7% FLOPs), §6.3 MFU/HFU, App. A FLOP formulas |
| `sys_li2020_pytorch_ddp.txt` | Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training (VLDB 2020) | https://arxiv.org/abs/2006.15704 | §2.2–2.3 DP and AllReduce background, §3.2.1 bucketing, §3.2.2 overlap, §3.2.3 Algorithm 1 (unused params, bitmap), §3.2.4 no_sync gradient accumulation, §4 implementation, §5 evaluation (bucket size sweeps) |
| `sys_rajbhandari2020_zero.txt` | Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (SC 2020) | https://arxiv.org/abs/1910.02054 | §3.1 model states 16Ψ (K = 12), §3.2 residual states, §5 ZeRO-DP (Pos, Pg, Pp; Fig. 1 7.5B/64-GPU numbers), §6 ZeRO-R, §7 communication (2Ψ vs 3Ψ) |
| `sys_ren2021_zero_offload.txt` | Ren et al., ZeRO-Offload (USENIX ATC 2021) | https://arxiv.org/abs/2101.06840 | what to put on CPU (fp32 states + optimizer step) and why; delayed parameter update |
| `sys_zhao2023_pytorch_fsdp.txt` | Zhao et al., PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (VLDB 2023) | https://arxiv.org/abs/2304.11277 | §3.1 deferred init, §3.2.1 full sharding (FlatParameter), §3.2.2 hybrid sharding, §3.2.3 autograd, §3.3 overlap/backward & forward prefetch/grad accumulation, §3.4 caching allocator & rate limiter, §4.4 native mixed precision |
| `sys_liang2024_torchtitan.txt` | Liang et al., TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training (2024) | https://arxiv.org/abs/2410.06511 | FSDP2 per-parameter sharding vs FSDP1 FlatParameter, async TP, Float8, composable 4D parallelism, selective AC |
| `sys_shoeybi2019_megatron.txt` | Shoeybi et al., Megatron-LM (2019) | https://arxiv.org/abs/1909.08053 | §3 model-parallel transformers: column- then row-parallel MLP, head-parallel attention, f/g operators, 2 all-reduces fwd + 2 bwd per layer, vocab-parallel embedding & fused CE |
| `sys_narayanan2021_megatron_ptd.txt` | Narayanan et al., Efficient Large-Scale LM Training on GPU Clusters Using Megatron-LM (SC 2021) | https://arxiv.org/abs/2104.04473 | §2.2.1 GPipe/1F1B bubble (p−1)/m, §2.2.2 interleaved (÷v), §3 takeaways #1–3 (TP within node, PP across nodes, microbatch size), §4.1 scatter/gather, §5.1 FLOPs 96Bslh²(1+s/6h+V/16lh) and training time ≈ 8TP/(nX) |
| `sys_huang2019_gpipe.txt` | Huang et al., GPipe (NeurIPS 2019) | https://arxiv.org/abs/1811.06965 | §2.2–2.3 micro-batching, re-materialisation, bubble O((K−1)/(M+K−1)), negligible for M ≥ 4K |
| `sys_narayanan2019_pipedream.txt` | Narayanan et al., PipeDream (SOSP 2019; arXiv v1 2018) | https://arxiv.org/abs/1806.03377 | 1F1B steady state, weight stashing, vertical sync, partitioning |
| `sys_qi2023_zero_bubble.txt` | Qi et al., Zero Bubble Pipeline Parallelism (ICLR 2024) | https://arxiv.org/abs/2401.10241 | splitting backward into B (input grad) and W (weight grad); ZB-H1/ZB-H2; post-validation of optimizer step |
| `sys_williams2009_roofline.txt` | Williams, Waterman & Patterson, Roofline (CACM 2009) | https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf | operational intensity, the roofline plot, ceilings |
| `sys_ansel2024_pytorch2.txt` | Ansel et al., PyTorch 2: Faster ML Through Dynamic Python Bytecode Transformation and Graph Compilation (ASPLOS 2024) | https://pytorch.org/assets/pytorch2-2.pdf | §2 prior graph-capture attempts (jit.trace, jit.script, Lazy Tensors, FX), §2.6 vs JAX, §3 TorchDynamo (PEP 523, guards, graph breaks, AOTAutograd), §4 TorchInductor (loop-level IR, Triton codegen, fusion), §5 dynamic shapes |
| `sys_frostig2018_jax.txt` | Frostig, Johnson & Leary, Compiling machine learning programs via high-level tracing (SysML 2018) | https://mlsys.org/Conferences/doc/2018/146.pdf | tracing to XLA HLO, abstract values, transformations |
| `sys_abadi2016_tensorflow.txt` | Abadi et al., TensorFlow: A system for large-scale machine learning (OSDI 2016) | https://arxiv.org/abs/1605.08695 | dataflow graph, deferred execution, parameter servers, control flow |
| `sys_xu2021_gspmd.txt` | Xu et al., GSPMD: General and Scalable Parallelization for ML Computation Graphs (2021) | https://arxiv.org/abs/2105.04663 | sharding annotations and propagation in XLA (the basis of `jax.jit` sharding) |
| `sys_tillet2019_triton.txt` | Tillet, Kung & Cox, Triton: an intermediate language and compiler for tiled neural network computations (MAPL 2019) | https://doi.org/10.1145/3315508.3329973 | block/tile programming model |
| `sys_patarasuk2009_bandwidth_optimal_allreduce.txt` | Patarasuk & Yuan, Bandwidth optimal all-reduce algorithms for clusters of workstations (JPDC 2009; author PDF) | https://doi.org/10.1016/j.jpdc.2008.09.002 (PDF: https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) | tight lower bound on per-process all-reduce data; ring algorithm achieving it on tree topologies; contention of butterfly/recursive-doubling |
| `sys_jacobs2023_ulysses.txt` | Jacobs et al., DeepSpeed Ulysses (2023) | https://arxiv.org/abs/2309.14509 | all-to-all re-partitioning from sequence to head sharding; constant per-GPU comm volume as sequence and GPUs scale |

## Already in the shared cache (use these instead of re-fetching)

From `INDEX.md`: `dlb_ch04_numerical.txt` (Goodfellow DL §4.1 overflow/underflow, softmax stabilisation), `dlb_ch08_optimization.txt`
(§8.2.4 cliffs and exploding gradients, §8.2.5 long-term dependencies, §8.4 init, §8.7.1 BatchNorm), `dlb_ch10_rnn.txt`
(§10.7 challenge of long-term dependencies, §10.9 skip/leaky units, §10.10 LSTM/GRU, §10.11.1 clipping gradients,
§10.11.2 regularising information flow), `d2l/d2l_numerical_stability_and_init.txt` (d2l §5.4), `d2l/d2l_bptt.txt` (d2l §9.7),
`papers/pascanu2013_rnn_difficulty.txt` (§2.1 mechanics, sufficient condition λ₁ < 1/γ, §2.3 geometric picture, §3.2
Algorithm 1 norm clipping), `papers/glorot2010_init.txt`, `papers/he2015_prelu_init.txt`, `papers/saxe2013_orthogonal.txt`,
`papers/he2016_resnet.txt`, `papers/xiong2020_preln.txt`, `papers/goyal2017_large_minibatch.txt`, `papers/kingma2014_adam.txt`,
`papers/ioffe2015_batchnorm.txt`, `papers/baydin2018_autodiff.txt`.

From `INDEX-llm.md`: `llm_chowdhery2022_palm.txt` (PaLM: MFU definition §4 / Table 3, z-loss in §5),
`llm_team2024_gemma2.txt` (logit soft-capping), `llm_dehghani2023_vit22b.txt` (QK-layernorm), `llm_deepseek2024_v3.txt`
(§3.2.1 DualPipe, §3.3 FP8 training with fine-grained 1×128 / 128×128 scaling and higher-precision accumulation),
`llm_rouhani2023_microscaling.txt` (MX block formats), `llm_liu2023_ring.txt` (ring attention), `llm_dao2022_flashattention.txt`
(IO-aware fusion example), `llm_milakov2018_online_softmax.txt`, `llm_eleuther2023_transformer_math.txt` (memory/compute
rules of thumb), `llm_dubey2024_llama3.txt` (4D parallelism, 16K H100 setup; §3.3.4 reliability: 466 interruptions / 54 days), `llm_mccandlish2018_critical_batch.txt`,
`llm_touvron2023_llama2.txt` (§2.2 clipping 1.0), `llm_radford2019_gpt2.txt` (§2.3 residual init scaling). From `INDEX.md` also:
`papers/brown2020_gpt3.txt` (App. B: global-norm clipping at 1.0), `dlb_ch02_linear_algebra.txt` (norms refresher).

## Failures / not cached

- Chan et al. 2007, "Collective communication: theory, practice, and experience": no free copy found (utexas URLs 404).
  Patarasuk & Yuan 2009 and the NCCL 2.4 blog cover ring optimality and trees instead.
- Thakur, Rabenseifner & Gropp 2005, "Optimization of Collective Communication Operations in MPICH" (the classic α–β costs
  of ring / recursive doubling / Rabenseifner): anl.gov serves a Cloudflare bot challenge; not bypassed. The α–β ring and
  tree costs are covered by `sys_nccl_tests_performance.txt`, the Scaling Book (sharding, gpus) and Playbook A0.
- Bengio 1994: original is a scan; cached as OCR text (equations unreliable).
- Goldberg 1991 PDF (validlab.com): garbled font encoding; replaced by the Oracle HTML reprint.
- NVIDIA Transformer Engine FP8 primer: unversioned URL 404s; cached the release-2.0 copy.
