# Content rebuild: status (paused 2026-10-02)

The rebuild was paused to save tokens. Everything in `PrepApp/Content/` validates (0 errors), and the app builds.
All unit tests and the end-to-end smoke test pass.

## How the pipeline works
1. **Plans:** curator agents wrote per-area syllabi with sources, and critic agents reviewed them. The plans:
   - [content-plan.md](content-plan.md): fundamentals (66 lessons) and generative (22)
   - [content-plan-llms.md](content-plan-llms.md): 61 lessons
   - [content-plan-applied.md](content-plan-applied.md): 30 lessons
   - [content-plan-scalingbook.md](content-plan-scalingbook.md): 36 lessons
   - [content-plan-tuning.md](content-plan-tuning.md): 6 lessons, plus insertions for other topics
2. **Writers** follow [writing-brief.md](writing-brief.md), using the prompt in [agents/writer_template.md](agents/writer_template.md).
3. **Critics** follow [review-rubric.md](review-rubric.md), using [agents/critic_template.md](agents/critic_template.md).
   Their reviews are in [reviews/](reviews/). The writer then fixes what the review found.
4. **Sources:** the text cache is in `sources/` at the repo root (git-ignored, about 52 MB). Indexes: `INDEX*.md`.
   The original PDFs aren't kept, but the URLs are in the indexes.

## Lessons by state

**Done (written, critiqued, revised): 26**
- Applied: sys.fp-formats, mixed-precision, memory-anatomy, gpu-basics, roofline, flops-mfu, collectives, ddp,
  zero, tensor-parallel, pipeline-parallel
- Tuning: sys.tuning-process, -search, -batch-size, -steps-schedules, -diagnostics, -pipeline
- Scaling Book: sb.roofline-basics, roofline-matmul, tpu-chip, tpu-networking, sharding-notation, sharded-matmul,
  collective-costs, alltoall-overlap, flops-counting, transformer-accounting, transformer-memory, flash-attention
- Generative: gen.gans

**Written and in the app, but review fixes not applied.** Each review lists the open findings.
- gen.latent-variables-elbo, gen.vae: reviews/content-gen-elbo-vae.md. The BCE claim is already fixed.
- gen.guidance, gen.flow-matching: reviews/content-gen-guidance-fm.md. The "never cross" claim is already fixed.
- fund.information-theory, fund.kl-divergence: reviews/content-fund-infotheory-kl.md.

**Written and in the app, not reviewed yet.** The writer's self-check passed.
- fund.bayes-theorem, mle, mle-map
- fund.bias-variance, regularization, classification-metrics
- fund.gradient-descent, sgd-momentum, adam
- fund.backprop, mlp-from-scratch, training-loop
- fund.linear-regression, logistic-regression, svm-margin-dual, decision-trees
- gen.ddpm-forward-reverse, ddpm-objective, diffusion-parameterizations
- llm.self-attention, llm.positional-encodings

**Not written yet.** Some of these already have figure modules in `tools/figures/`.
- `fund.kl-estimation` (order 102): Fisher information, f-divergences, and the k1/k2/k3 estimators.
  This material was cut from fund.kl-divergence when it was split, and the new lesson was never created.
  The split recipe is in reviews/content-fund-infotheory-kl.md.
- Rest of fundamentals wave 1: bagging-random-forests, pca, initialization, batchnorm
- LLM wave 1:
  - causal-attention, rope, transformer-block, params-flops, tokenization, pretraining-objective
  - scaling-laws, kv-cache, mqa-gqa-mla, flash-attention, sampling, speculative-decoding
  - moe, moe-load-balancing, sft, lora, reward-modeling, policy-gradients, rlhf-ppo, dpo, rlvr-grpo
  - linear-attention, ssm, mamba, modern-rnns-hybrids, architecture-comparison
- Scaling Book ch. 5–12 (orders 130–360)
- All wave 2 and wave 3 lessons in every plan
- Rewrite of `mlcoding/code.attention-bugs.yaml` (the correct choice is the longest option in 9 of 10 MCQs)

## Cheaper ways to resume
- Run 4–6 agents at a time instead of 20, so a usage limit doesn't stop everything partway through.
- Use Sonnet for critics. The validator now catches answer-length giveaways, long equations and missing figures.
- Fix outstanding reviews before writing new lessons. The review files already say exactly what to change.
