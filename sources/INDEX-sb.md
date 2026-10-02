# Source cache index: "How to Scale Your Model" (JAX Scaling Book) — `sb_*` files

Austin, Douglas, Frostig, Levskaya, Chen, Vikram, Lebron, Choy, Ramasesh, Webson, Pope (Google DeepMind, 2025).
Web: https://jax-ml.github.io/scaling-book/ · Source: https://github.com/jax-ml/scaling-book
Cached 2026-10-02 from the GitHub repo (commit 9e2bbee, 2026-09-22). These are the **original markdown sources**
(Jekyll/Distill): LaTeX is preserved verbatim (`$$...$$`), figures appear as `{% include figure.liquid path=... %}`
lines (images not cached), answers to worked problems sit inside `{% details %} ... {% enddetails %}` blocks, and
footnotes are inline `<d-footnote>` tags. Grep with `grep -n '^##' sb_*.md` for section headings.

Note: other agents cached some chapters as `llm_scalingbook_*.txt` and `sys_scalingbook_*.txt` (HTML-to-text).
The `sb_*.md` files here are the cleaner markdown versions of the same chapters.

| File | Chapter | URL | Main sections |
|---|---|---|---|
| `sb_00_index.md` | Part 0: Intro / outline | https://jax-ml.github.io/scaling-book/ | Why should you care? (strong scaling); High-level outline |
| `sb_01_roofline.md` | Ch. 1 All About Rooflines | https://jax-ml.github.io/scaling-book/roofline/ | Where does the time go?; Visualizing rooflines; Matrix multiplication; Network communication rooflines; Problems Q1–Q5 |
| `sb_02_tpus.md` | Ch. 2 How to Think About TPUs | https://jax-ml.github.io/scaling-book/tpus/ | What is a TPU?; TPU Networking; Key takeaways; TPU specs tables; Worked problems Q1–Q6; App. A (VPU, scalar core); App. B (systolic array) |
| `sb_03_sharding.md` | Ch. 3 Sharded Matrices and How to Multiply Them | https://jax-ml.github.io/scaling-book/sharding/ | Notation; Cases 1–4; AllGather/ReduceScatter/AllReduce costs; AllToAll; RS as derivative of AG; collective matmul; Problems Q1–Q10 |
| `sb_04_transformers.md` | Ch. 4 All the Transformer Math You Need to Know | https://jax-ml.github.io/scaling-book/transformers/ | Counting dots; forward/backward FLOPs; MLP/attention params & FLOPs; T/8D rule; MoE; gradient checkpointing; KV cache; Problems Q1–Q8; App. A Flash Attention |
| `sb_05_training.md` | Ch. 5 How to Parallelize a Transformer for Training | https://jax-ml.github.io/scaling-book/training/ | DP; FSDP; TP; FSDP+TP (X_opt); pipelining; DCN across pods; takeaways; Problems Q1–Q3 (LLaMA-2 13B); App. A backward comms |
| `sb_06_applied-training.md` | Ch. 6 Training LLaMA 3 on TPUs | https://jax-ml.github.io/scaling-book/applied-training/ | LLaMA-3 70B config; params/FLOPs; time & memory; how to shard; Problems (4 pods, 405B) |
| `sb_07_inference.md` | Ch. 7 All About Transformer Inference | https://jax-ml.github.io/scaling-book/inference/ | Prefill vs generation; linear-op and attention rooflines; step-time formulas; memory/KV cache; LLaMA-2 13B tables; KV tricks; sharding prefill/generation/KV; engines (interleaved, disaggregated, continuous batching, prefix caching, JetStream); Problems Q1–Q7; App. A–D |
| `sb_08_applied-inference.md` | Ch. 8 Serving LLaMA 3 on TPUs | https://jax-ml.github.io/scaling-book/applied-inference/ | Hardware choice; KV sizes; min topology; latency/throughput; sharding; prefill; Pareto plots + code; Problems Q1–Q4 |
| `sb_09_profiling.md` | Ch. 9 How to Profile TPU Code | https://jax-ml.github.io/scaling-book/profiling/ | JAX→StableHLO→HLO→LLO stack; profiler; Trace Viewer; reading an XLA op (layouts/tiling/memory spaces); Graph Viewer; example profile; Memory profile; Problems |
| `sb_10_jax-stuff.md` | Ch. 10 Programming TPUs in JAX | https://jax-ml.github.io/scaling-book/jax-stuff/ | Auto / Explicit / Manual (shard_map) modes; with_sharding_constraint; collective matmul via ppermute; Problems Q1–Q4 |
| `sb_11_conclusion.md` | Ch. 11 Conclusions and Further Reading | https://jax-ml.github.io/scaling-book/conclusion/ | Acknowledgments; further-reading list |
| `sb_12_gpus.md` | Ch. 12 How to Think About GPUs | https://jax-ml.github.io/scaling-book/gpus/ | SMs/Tensor Cores/CUDA cores; memory hierarchy; spec tables; GPU vs TPU; NVLink nodes; fat-tree SuperPod; GB200; collectives intra/cross node, SHARP; LLM rooflines on GPUs (DP/TP/EP/PP); DeepSeek & LLaMA-3 configs; Quizzes 1–5; App. A–B |
