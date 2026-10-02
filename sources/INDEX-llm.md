# LLM source cache index (`llm_*`)

Built 2026-10-02 for the Large Language Models area (`llm.*`). The plan is in `docs/content-plan-llms.md` in the project.
This index covers only `llm_`-prefixed files. The other curator's files are indexed in `INDEX.md`.

## Layout and how to search
- `llm_<name>.txt`: extracted text. The first line is `SOURCE: <canonical URL>`.
- `llm_pdf/llm_<name>.pdf`: the original PDFs (arXiv, SLP3, GPT-2, DeepSeek V4 model card).
- `llm_html/`: raw HTML for the web pages (Scaling Book, d2l, blogs, transformer-circuits).
- `llm_toc/llm_<name>_toc.txt`: auto-extracted section headings with PDF page (`3.2 Title (p.5)`). They are noisy
  (table cells and footnotes sometimes slip in, and small-caps titles appear letter-spaced), but they are good for jumping to a section.
- PDF-derived text has page markers `=== [<name> p.N] ===`. **N is the PDF page**, not the printed page number.
  Cite "§x.y, pdf p.N". SLP3 files use `=== [slp3_chK pdf p.N] ===`.
- Search with `grep -n` or `rg`. NUL and form-feed bytes were stripped. A few files still contain other control characters,
  so if grep reports "binary file matches", use `grep -a`.
- **Extraction quality.** Prose is good. **Equations are mangled** (sub/superscripts split across lines, fractions flattened,
  some math-italic Unicode). Find the passage in the text, then rebuild the equation from the PDF. The web files keep LaTeX
  (`\(...\)`, `\[...\]`, `$...$`), which is the best source for exact equation forms (Scaling Book, d2l, Weng, EleutherAI).
- A grep helper used during planning is at `../llmwork/g.py` (`python3 g.py <name-without-llm_> 'regex' ...` prints matches with page numbers).

## Not cached (failures)
- bloc97's Reddit post introducing NTK-aware RoPE scaling: blocked by Reddit's network policy. The formula is in YaRN App. A.2 (`llm_peng2023_yarn.txt`).
- Hugging Face *Smol Training Playbook* (2025): a JavaScript Space, so no static text was retrievable.
- The arXiv export API was rate-limited, so the papers were fetched directly from arxiv.org/pdf and titles from arxiv.org/abs pages.
- Not fetched (cited only as mentions in the plan): Templeton et al. 2024 *Scaling Monosemanticity*, DeepSpeed-Ulysses, Orca, SGLang/RadixAttention, Medusa follow-ups, BLIP-2, Llama 4 blog, Gemini Diffusion/Mercury.

## Files

| kind | file | KB | source | canonical URL | used by lessons |
|---|---|---|---|---|---|
| book/course | `llm_d2l_attention_scoring.txt` | 30 | d2l.ai §11.3 Attention Scoring Functions | https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html | llm.self-attention |
| book/course | `llm_d2l_beam_search.txt` | 10 | d2l.ai §10.8 Beam Search | https://d2l.ai/chapter_recurrent-modern/beam-search.html | llm.decoding |
| book/course | `llm_d2l_multihead_attention.txt` | 17 | d2l.ai §11.5 Multi-Head Attention | https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html | llm.self-attention |
| book/course | `llm_d2l_self_attention_posenc.txt` | 18 | d2l.ai §11.6 Self-Attention and Positional Encoding | https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html | llm.architecture-comparison, llm.positional-encodings, llm.self-attention |
| book/course | `llm_d2l_transformer.txt` | 57 | d2l.ai §11.7 The Transformer Architecture | https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html | llm.transformer-block |
| book/course | `llm_scalingbook_applied-inference.txt` | 23 | JAX Scaling Book Part 8: Serving LLaMA 3-70B | https://jax-ml.github.io/scaling-book/applied-inference/ | llm.kv-cache |
| book/course | `llm_scalingbook_gpus.txt` | 79 | JAX Scaling Book: How to Think About GPUs | https://jax-ml.github.io/scaling-book/gpus/ | llm.arithmetic-intensity, llm.moe-systems |
| book/course | `llm_scalingbook_inference.txt` | 62 | JAX Scaling Book Part 7: All About Transformer Inference | https://jax-ml.github.io/scaling-book/inference/ | llm.causal-attention, llm.flash-attention-2-3, llm.kv-cache, llm.quantization, llm.serving-systems, llm.speculative-decoding |
| book/course | `llm_scalingbook_roofline.txt` | 20 | JAX Scaling Book Part 1: All About Rooflines | https://jax-ml.github.io/scaling-book/roofline/ | llm.arithmetic-intensity |
| book/course | `llm_scalingbook_sharding.txt` | 58 | JAX Scaling Book Part 3: Sharded Matrices (for the applied area; not used by LLM lessons) | https://jax-ml.github.io/scaling-book/sharding/ | — |
| book/course | `llm_scalingbook_training.txt` | 50 | JAX Scaling Book Part 5: How to Parallelize a Transformer for Training (for the applied area) | https://jax-ml.github.io/scaling-book/training/ | — |
| book/course | `llm_scalingbook_transformers.txt` | 27 | JAX Scaling Book Part 4: All the Transformer Math You Need to Know | https://jax-ml.github.io/scaling-book/transformers/ | llm.causal-attention, llm.flash-attention, llm.moe, llm.moe-systems, llm.mqa-gqa-mla, llm.params-flops, llm.scaling-laws, llm.self-attention |
| book/course | `llm_slp3_ch10.txt` | 33 | Jurafsky & Martin, SLP3 ch. 10 Interpretability (stub chapter) | https://web.stanford.edu/~jurafsky/slp3/10.pdf | llm.mech-interp |
| book/course | `llm_slp3_ch11.txt` | 71 | Jurafsky & Martin, SLP3 ch. 11 Information Retrieval and RAG | https://web.stanford.edu/~jurafsky/slp3/11.pdf | llm.rag |
| book/course | `llm_slp3_ch13.txt` | 94 | Jurafsky & Martin, SLP3 ch. 13 Machine Translation (encoder-decoder, beam search) | https://web.stanford.edu/~jurafsky/slp3/13.pdf | llm.decoding, llm.lm-objectives |
| book/course | `llm_slp3_ch2.txt` | 106 | Jurafsky & Martin, SLP3 ch. 2 Words and Tokens (BPE, Unicode, regex pre-tokenization) | https://web.stanford.edu/~jurafsky/slp3/2.pdf | llm.tokenization |
| book/course | `llm_slp3_ch3.txt` | 74 | Jurafsky & Martin, SLP3 ch. 3 N-gram LMs (perplexity, entropy) | https://web.stanford.edu/~jurafsky/slp3/3.pdf | llm.pretraining-objective |
| book/course | `llm_slp3_ch7.txt` | 96 | Jurafsky & Martin, SLP3 ch. 7 Transformers and Pretraining (attention, blocks, decoding/sampling, pretraining) | https://web.stanford.edu/~jurafsky/slp3/7.pdf | llm.causal-attention, llm.decoding, llm.evaluation-metrics, llm.pretraining-objective, llm.sampling, llm.self-attention, llm.transformer-block |
| book/course | `llm_slp3_ch8.txt` | 43 | Jurafsky & Martin, SLP3 ch. 8 Post-training (instruction tuning, PEFT, preference learning) | https://web.stanford.edu/~jurafsky/slp3/8.pdf | llm.dpo, llm.evaluation-metrics, llm.lora, llm.qlora-peft, llm.sft |
| book/course | `llm_slp3_ch9.txt` | 46 | Jurafsky & Martin, SLP3 ch. 9 Masked Language Models | https://web.stanford.edu/~jurafsky/slp3/9.pdf | llm.lm-objectives |
| paper | `llm_agarwal2023_gkd.txt` | 64 | Agarwal et al. 2023, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* | https://arxiv.org/abs/2306.13649 | llm.distillation |
| paper | `llm_aghajanyan2020_intrinsic_dim.txt` | 35 | Aghajanyan et al. 2020, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* | https://arxiv.org/abs/2012.13255 | llm.lora |
| paper | `llm_ahmadian2024_rloo.txt` | 75 | Ahmadian et al. 2024, *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* | https://arxiv.org/abs/2402.14740 | llm.rlhf-ppo |
| paper | `llm_ainslie2023_gqa.txt` | 23 | Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* | https://arxiv.org/abs/2305.13245 | llm.mqa-gqa-mla |
| paper | `llm_alayrac2022_flamingo.txt` | 190 | Alayrac et al. 2022, *Flamingo: a Visual Language Model for Few-Shot Learning* | https://arxiv.org/abs/2204.14198 | llm.cross-attention |
| paper | `llm_arora2023_zoology.txt` | 199 | Arora et al. 2023, *Zoology: Measuring and Improving Recall in Efficient Language Models* | https://arxiv.org/abs/2312.04927 | llm.architecture-comparison |
| paper | `llm_azar2023_ipo.txt` | 45 | Azar et al. 2023, *A General Theoretical Paradigm to Understand Learning from Human Preferences* | https://arxiv.org/abs/2310.12036 | llm.dpo, llm.preference-variants |
| paper | `llm_bai2022_constitutional.txt` | 118 | Bai et al. 2022, *Constitutional AI: Harmlessness from AI Feedback* | https://arxiv.org/abs/2212.08073 | llm.preference-variants |
| paper | `llm_belrose2023_tuned_lens.txt` | 93 | Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens* | https://arxiv.org/abs/2303.08112 | llm.mech-interp |
| paper | `llm_beltagy2020_longformer.txt` | 69 | Beltagy et al. 2020, *Longformer: The Long-Document Transformer* | https://arxiv.org/abs/2004.05150 | llm.sparse-attention |
| paper | `llm_bengio2015_scheduled_sampling.txt` | 33 | Bengio et al. 2015, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | https://arxiv.org/abs/1506.03099 | llm.causal-attention |
| paper | `llm_besiroglu2024_chinchilla_replication.txt` | 30 | Besiroglu et al. 2024, *Chinchilla Scaling: A replication attempt* | https://arxiv.org/abs/2404.10102 | llm.scaling-laws |
| paper | `llm_bi2024_deepseek_llm.txt` | 117 | DeepSeek-AI et al. 2024, *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* | https://arxiv.org/abs/2401.02954 | llm.pretraining-optimization |
| paper | `llm_biderman2024_lm_eval_lessons.txt` | 121 | Biderman et al. 2024, *Lessons from the Trenches on Reproducible Evaluation of Language Models* | https://arxiv.org/abs/2405.14782 | llm.evaluation-metrics |
| paper | `llm_biderman2024_lora_forgets_less.txt` | 98 | Biderman et al. 2024, *LoRA Learns Less and Forgets Less* | https://arxiv.org/abs/2405.09673 | llm.lora, llm.sft |
| paper | `llm_bostrom2020_bpe_suboptimal.txt` | 28 | Bostrom et al. 2020, *Byte Pair Encoding is Suboptimal for Language Model Pretraining* | https://arxiv.org/abs/2004.03720 | llm.tokenization |
| paper | `llm_brown2024_monkeys.txt` | 76 | Brown et al. 2024, *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* | https://arxiv.org/abs/2407.21787 | llm.test-time-compute |
| paper | `llm_cai2024_medusa.txt` | 86 | Cai et al. 2024, *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* | https://arxiv.org/abs/2401.10774 | llm.speculative-decoding |
| paper | `llm_chen2021_codex.txt` | 152 | Chen et al. 2021, *Evaluating Large Language Models Trained on Code* | https://arxiv.org/abs/2107.03374 | llm.evaluation-metrics, llm.test-time-compute |
| paper | `llm_chen2023_pi.txt` | 51 | Chen et al. 2023, *Extending Context Window of Large Language Models via Positional Interpolation* | https://arxiv.org/abs/2306.15595 | llm.context-extension |
| paper | `llm_chen2023_speculative_sampling.txt` | 31 | Chen et al. 2023, *Accelerating Large Language Model Decoding with Speculative Sampling* | https://arxiv.org/abs/2302.01318 | llm.speculative-decoding |
| paper | `llm_chiang2024_chatbot_arena.txt` | 88 | Chiang et al. 2024, *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* | https://arxiv.org/abs/2403.04132 | llm.evaluation-metrics |
| paper | `llm_child2019_sparse_transformer.txt` | 36 | Child et al. 2019, *Generating Long Sequences with Sparse Transformers* | https://arxiv.org/abs/1904.10509 | llm.sparse-attention |
| paper | `llm_choromanski2020_performer.txt` | 109 | Choromanski et al. 2020, *Rethinking Attention with Performers* | https://arxiv.org/abs/2009.14794 | llm.linear-attention |
| paper | `llm_chowdhery2022_palm.txt` | 288 | Chowdhery et al. 2022, *PaLM: Scaling Language Modeling with Pathways* | https://arxiv.org/abs/2204.02311 | llm.params-flops, llm.training-stability, llm.transformer-block |
| paper | `llm_christiano2017_rlhf.txt` | 57 | Christiano et al. 2017, *Deep reinforcement learning from human preferences* | https://arxiv.org/abs/1706.03741 | llm.reward-modeling |
| paper | `llm_dai2019_transformerxl.txt` | 86 | Dai et al. 2019, *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* | https://arxiv.org/abs/1901.02860 | llm.long-context, llm.positional-encodings |
| paper | `llm_dai2024_deepseekmoe.txt` | 94 | Dai et al. 2024, *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* | https://arxiv.org/abs/2401.06066 | llm.moe-systems |
| paper | `llm_dao2022_flashattention.txt` | 116 | Dao et al. 2022, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* | https://arxiv.org/abs/2205.14135 | llm.arithmetic-intensity, llm.flash-attention, llm.flash-attention-2-3 |
| paper | `llm_dao2023_flashattention2.txt` | 42 | Dao et al. 2023, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* | https://arxiv.org/abs/2307.08691 | llm.flash-attention-2-3 |
| paper | `llm_dao2024_mamba2.txt` | 188 | Dao et al. 2024, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* | https://arxiv.org/abs/2405.21060 | llm.mamba |
| paper | `llm_de2024_griffin.txt` | 81 | De et al. 2024, *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models* | https://arxiv.org/abs/2402.19427 | llm.architecture-comparison, llm.modern-rnns-hybrids |
| paper | `llm_deepseek2024_v2.txt` | 117 | DeepSeek-AI et al. 2024, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* | https://arxiv.org/abs/2405.04434 | llm.mqa-gqa-mla |
| paper | `llm_deepseek2024_v3.txt` | 152 | DeepSeek-AI et al. 2024, *DeepSeek-V3 Technical Report* | https://arxiv.org/abs/2412.19437 | llm.moe-load-balancing, llm.moe-systems, llm.mqa-gqa-mla, llm.pretraining-objective, llm.pretraining-optimization, llm.quantization-advanced, llm.tokenization-effects |
| paper | `llm_deepseek2025_r1.txt` | 234 | DeepSeek-AI et al. 2025, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | https://arxiv.org/abs/2501.12948 | llm.distillation, llm.rlvr-grpo |
| paper | `llm_deepseek2025_v32.txt` | 66 | DeepSeek-AI et al. 2025, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models* | https://arxiv.org/abs/2512.02556 | llm.rlvr-grpo, llm.sparse-attention |
| paper | `llm_deepseek2026_v4.txt` | 176 | DeepSeek-AI et al. 2026, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* | https://arxiv.org/abs/2606.19348 | llm.sparse-attention |
| paper | `llm_dehghani2018_universal.txt` | 60 | Dehghani et al. 2018, *Universal Transformers* | https://arxiv.org/abs/1807.03819 | llm.looped-transformers |
| paper | `llm_dehghani2023_vit22b.txt` | 118 | Dehghani et al. 2023, *Scaling Vision Transformers to 22 Billion Parameters* | https://arxiv.org/abs/2302.05442 | llm.training-stability |
| paper | `llm_deletang2023_compression.txt` | 61 | Delétang et al. 2023, *Language Modeling Is Compression* | https://arxiv.org/abs/2309.10668 | llm.pretraining-objective |
| paper | `llm_dettmers2022_int8.txt` | 71 | Dettmers et al. 2022, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* | https://arxiv.org/abs/2208.07339 | llm.quantization |
| paper | `llm_dettmers2023_qlora.txt` | 87 | Dettmers et al. 2023, *QLoRA: Efficient Finetuning of Quantized LLMs* | https://arxiv.org/abs/2305.14314 | llm.qlora-peft, llm.quantization-advanced |
| paper | `llm_devlin2018_bert.txt` | 63 | Devlin et al. 2018, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | https://arxiv.org/abs/1810.04805 | llm.lm-objectives |
| paper | `llm_ding2024_longrope.txt` | 59 | Ding et al. 2024, *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* | https://arxiv.org/abs/2402.13753 | llm.context-extension |
| paper | `llm_dubey2024_llama3.txt` | 355 | Grattafiori et al. 2024, *The Llama 3 Herd of Models* | https://arxiv.org/abs/2407.21783 | llm.causal-attention, llm.context-extension, llm.evaluation-pitfalls, llm.long-context, llm.params-flops, llm.pretraining-data, llm.rope, llm.scaling-laws |
| paper | `llm_dubois2024_length_controlled.txt` | 36 | Dubois et al. 2024, *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators* | https://arxiv.org/abs/2404.04475 | llm.evaluation-pitfalls |
| paper | `llm_elhage2022_superposition.txt` | 156 | Elhage et al. 2022, *Toy Models of Superposition* | https://arxiv.org/abs/2209.10652 | llm.mech-interp |
| paper | `llm_ethayarajh2024_kto.txt` | 83 | Ethayarajh et al. 2024, *KTO: Model Alignment as Prospect Theoretic Optimization* | https://arxiv.org/abs/2402.01306 | llm.preference-variants |
| paper | `llm_everett2024_scaling_exponents.txt` | 206 | Everett et al. 2024, *Scaling Exponents Across Parameterizations and Optimizers* | https://arxiv.org/abs/2407.05872 | llm.mup |
| paper | `llm_fedus2021_switch.txt` | 100 | Fedus et al. 2021, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* | https://arxiv.org/abs/2101.03961 | llm.moe, llm.moe-load-balancing |
| paper | `llm_frantar2022_gptq.txt` | 56 | Frantar et al. 2022, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* | https://arxiv.org/abs/2210.17323 | llm.quantization-advanced |
| paper | `llm_fu2024_data_128k.txt` | 45 | Fu et al. 2024, *Data Engineering for Scaling Language Models to 128K Context* | https://arxiv.org/abs/2402.10171 | llm.long-context-eval |
| paper | `llm_gale2022_megablocks.txt` | 61 | Gale et al. 2022, *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* | https://arxiv.org/abs/2211.15841 | llm.moe-load-balancing |
| paper | `llm_gao2022_rm_overoptimization.txt` | 54 | Gao et al. 2022, *Scaling Laws for Reward Model Overoptimization* | https://arxiv.org/abs/2210.10760 | llm.reward-modeling |
| paper | `llm_garg2022_icl_linear.txt` | 98 | Garg et al. 2022, *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* | https://arxiv.org/abs/2208.01066 | llm.in-context-learning |
| paper | `llm_geiping2025_recurrent_depth.txt` | 133 | Geiping et al. 2025, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* | https://arxiv.org/abs/2502.05171 | llm.looped-transformers |
| paper | `llm_gemma2025_gemma3.txt` | 68 | Gemma Team et al. 2025, *Gemma 3 Technical Report* | https://arxiv.org/abs/2503.19786 | llm.distillation, llm.sparse-attention, llm.transformer-block |
| paper | `llm_giannou2023_looped.txt` | 144 | Giannou et al. 2023, *Looped Transformers as Programmable Computers* | https://arxiv.org/abs/2301.13196 | llm.looped-transformers |
| paper | `llm_gloeckle2024_mtp.txt` | 88 | Gloeckle et al. 2024, *Better &amp; Faster Large Language Models via Multi-token Prediction* | https://arxiv.org/abs/2404.19737 | llm.pretraining-objective, llm.speculative-decoding |
| paper | `llm_graves2016_act.txt` | 44 | Graves et al. 2016, *Adaptive Computation Time for Recurrent Neural Networks* | https://arxiv.org/abs/1603.08983 | llm.looped-transformers, llm.mot-mod |
| paper | `llm_gu2020_hippo.txt` | 130 | Gu et al. 2020, *HiPPO: Recurrent Memory with Optimal Polynomial Projections* | https://arxiv.org/abs/2008.07669 | llm.ssm |
| paper | `llm_gu2021_s4.txt` | 95 | Gu et al. 2021, *Efficiently Modeling Long Sequences with Structured State Spaces* | https://arxiv.org/abs/2111.00396 | llm.ssm |
| paper | `llm_gu2023_mamba.txt` | 136 | Gu et al. 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* | https://arxiv.org/abs/2312.00752 | llm.architecture-comparison, llm.mamba, llm.ssm |
| paper | `llm_gu2023_minillm.txt` | 71 | Gu et al. 2023, *MiniLLM: On-Policy Distillation of Large Language Models* | https://arxiv.org/abs/2306.08543 | llm.distillation |
| paper | `llm_hagele2024_wsd.txt` | 79 | Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* | https://arxiv.org/abs/2405.18392 | llm.pretraining-optimization, llm.scaling-laws-practice |
| paper | `llm_hao2024_coconut.txt` | 57 | Hao et al. 2024, *Training Large Language Models to Reason in a Continuous Latent Space* | https://arxiv.org/abs/2412.06769 | llm.looped-transformers |
| paper | `llm_hinton2015_distillation.txt` | 33 | Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* | https://arxiv.org/abs/1503.02531 | llm.distillation |
| paper | `llm_hoffmann2022_chinchilla.txt` | 98 | Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models* | https://arxiv.org/abs/2203.15556 | llm.scaling-laws |
| paper | `llm_holtzman2019_nucleus.txt` | 51 | Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration* | https://arxiv.org/abs/1904.09751 | llm.decoding, llm.sampling |
| paper | `llm_hong2024_orpo.txt` | 66 | Hong et al. 2024, *ORPO: Monolithic Preference Optimization without Reference Model* | https://arxiv.org/abs/2403.07691 | llm.preference-variants |
| paper | `llm_houlsby2019_adapters.txt` | 52 | Houlsby et al. 2019, *Parameter-Efficient Transfer Learning for NLP* | https://arxiv.org/abs/1902.00751 | llm.qlora-peft |
| paper | `llm_hsieh2024_ruler.txt` | 84 | Hsieh et al. 2024, *RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?* | https://arxiv.org/abs/2404.06654 | llm.long-context-eval |
| paper | `llm_hu2021_lora.txt` | 83 | Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models* | https://arxiv.org/abs/2106.09685 | llm.lora |
| paper | `llm_hu2024_minicpm.txt` | 100 | Hu et al. 2024, *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* | https://arxiv.org/abs/2404.06395 | llm.pretraining-optimization |
| paper | `llm_jaegle2021_perceiver.txt` | 91 | Jaegle et al. 2021, *Perceiver: General Perception with Iterative Attention* | https://arxiv.org/abs/2103.03206 | llm.cross-attention |
| paper | `llm_jaegle2021_perceiver_io.txt` | 113 | Jaegle et al. 2021, *Perceiver IO: A General Architecture for Structured Inputs &amp; Outputs* | https://arxiv.org/abs/2107.14795 | llm.cross-attention |
| paper | `llm_jelassi2024_repeat_after_me.txt` | 73 | Jelassi et al. 2024, *Repeat After Me: Transformers are Better than State Space Models at Copying* | https://arxiv.org/abs/2402.01032 | llm.architecture-comparison |
| paper | `llm_jiang2023_mistral7b.txt` | 24 | Jiang et al. 2023, *Mistral 7B* | https://arxiv.org/abs/2310.06825 | llm.sparse-attention |
| paper | `llm_jiang2024_mixtral.txt` | 32 | Jiang et al. 2024, *Mixtral of Experts* | https://arxiv.org/abs/2401.04088 | llm.moe |
| paper | `llm_jolicoeur2025_trm.txt` | 47 | Jolicoeur-Martineau et al. 2025, *Less is More: Recursive Reasoning with Tiny Networks* | https://arxiv.org/abs/2510.04871 | llm.looped-transformers |
| paper | `llm_kadavath2022_know_what_they_know.txt` | 116 | Kadavath et al. 2022, *Language Models (Mostly) Know What They Know* | https://arxiv.org/abs/2207.05221 | llm.sampling |
| paper | `llm_kalajdzievski2023_rslora.txt` | 35 | Kalajdzievski et al. 2023, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* | https://arxiv.org/abs/2312.03732 | llm.lora |
| paper | `llm_kaplan2020_scaling.txt` | 89 | Kaplan et al. 2020, *Scaling Laws for Neural Language Models* | https://arxiv.org/abs/2001.08361 | llm.params-flops, llm.scaling-laws |
| paper | `llm_katharopoulos2020_linear.txt` | 53 | Katharopoulos et al. 2020, *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* | https://arxiv.org/abs/2006.16236 | llm.linear-attention |
| paper | `llm_kazemnejad2023_nope.txt` | 100 | Kazemnejad et al. 2023, *The Impact of Positional Encoding on Length Generalization in Transformers* | https://arxiv.org/abs/2305.19466 | llm.positional-encodings |
| paper | `llm_keskar2019_ctrl.txt` | 64 | Keskar et al. 2019, *CTRL: A Conditional Transformer Language Model for Controllable Generation* | https://arxiv.org/abs/1909.05858 | llm.sampling |
| paper | `llm_kim2016_seq_kd.txt` | 42 | Kim et al. 2016, *Sequence-Level Knowledge Distillation* | https://arxiv.org/abs/1606.07947 | llm.distillation |
| paper | `llm_kimi2025_k15.txt` | 95 | Kimi Team et al. 2025, *Kimi k1.5: Scaling Reinforcement Learning with LLMs* | https://arxiv.org/abs/2501.12599 | llm.rlvr-grpo, llm.rlvr-practice |
| paper | `llm_kimi2025_k2.txt` | 114 | Kimi Team et al. 2025, *Kimi K2: Open Agentic Intelligence* | https://arxiv.org/abs/2507.20534 | llm.moe-systems, llm.training-stability |
| paper | `llm_kimi2025_linear.txt` | 96 | Kimi Team et al. 2025, *Kimi Linear: An Expressive, Efficient Attention Architecture* | https://arxiv.org/abs/2510.26692 | llm.linear-attention, llm.modern-rnns-hybrids |
| paper | `llm_komatsuzaki2022_upcycling.txt` | 83 | Komatsuzaki et al. 2022, *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints* | https://arxiv.org/abs/2212.05055 | llm.moe |
| paper | `llm_krajewski2024_fine_grained_moe_scaling.txt` | 60 | Krajewski et al. 2024, *Scaling Laws for Fine-Grained Mixture of Experts* | https://arxiv.org/abs/2402.07871 | llm.moe-systems |
| paper | `llm_kudo2018_sentencepiece.txt` | 24 | Kudo et al. 2018, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing* | https://arxiv.org/abs/1808.06226 | llm.tokenization |
| paper | `llm_kudo2018_unigram.txt` | 36 | Kudo et al. 2018, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* | https://arxiv.org/abs/1804.10959 | llm.tokenization |
| paper | `llm_kwon2023_pagedattention.txt` | 82 | Kwon et al. 2023, *Efficient Memory Management for Large Language Model Serving with PagedAttention* | https://arxiv.org/abs/2309.06180 | llm.serving-systems |
| paper | `llm_lambert2025_rlhf_book.txt` | 605 | Lambert et al. 2025, *Reinforcement Learning from Human Feedback* | https://arxiv.org/abs/2504.12501 | llm.distillation, llm.dpo, llm.reward-modeling, llm.rlhf-ppo, llm.rlvr-grpo, llm.sft, llm.test-time-compute |
| paper | `llm_land2024_magikarp.txt` | 61 | Land et al. 2024, *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models* | https://arxiv.org/abs/2405.05417 | llm.tokenization-effects |
| paper | `llm_lee2021_dedup.txt` | 81 | Lee et al. 2021, *Deduplicating Training Data Makes Language Models Better* | https://arxiv.org/abs/2107.06499 | llm.pretraining-data |
| paper | `llm_lepikhin2020_gshard.txt` | 121 | Lepikhin et al. 2020, *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* | https://arxiv.org/abs/2006.16668 | llm.moe |
| paper | `llm_lester2021_prompt_tuning.txt` | 63 | Lester et al. 2021, *The Power of Scale for Parameter-Efficient Prompt Tuning* | https://arxiv.org/abs/2104.08691 | llm.qlora-peft |
| paper | `llm_leviathan2022_speculative.txt` | 47 | Leviathan et al. 2022, *Fast Inference from Transformers via Speculative Decoding* | https://arxiv.org/abs/2211.17192 | llm.speculative-decoding |
| paper | `llm_lewis2020_rag.txt` | 68 | Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | https://arxiv.org/abs/2005.11401 | llm.rag |
| paper | `llm_li2021_prefix_tuning.txt` | 58 | Li et al. 2021, *Prefix-Tuning: Optimizing Continuous Prompts for Generation* | https://arxiv.org/abs/2101.00190 | llm.qlora-peft |
| paper | `llm_li2024_dclm.txt` | 278 | Li et al. 2024, *DataComp-LM: In search of the next generation of training sets for language models* | https://arxiv.org/abs/2406.11794 | llm.pretraining-data |
| paper | `llm_li2024_eagle.txt` | 50 | Li et al. 2024, *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* | https://arxiv.org/abs/2401.15077 | llm.speculative-decoding |
| paper | `llm_liang2024_mot.txt` | 150 | Liang et al. 2024, *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models* | https://arxiv.org/abs/2411.04996 | llm.mot-mod |
| paper | `llm_lieber2024_jamba.txt` | 49 | Lieber et al. 2024, *Jamba: A Hybrid Transformer-Mamba Language Model* | https://arxiv.org/abs/2403.19887 | llm.modern-rnns-hybrids |
| paper | `llm_lightman2023_prm.txt` | 53 | Lightman et al. 2023, *Let&#39;s Verify Step by Step* | https://arxiv.org/abs/2305.20050 | llm.test-time-compute |
| paper | `llm_lin2023_awq.txt` | 67 | Lin et al. 2023, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* | https://arxiv.org/abs/2306.00978 | llm.quantization |
| paper | `llm_liu2023_lost_middle.txt` | 64 | Liu et al. 2023, *Lost in the Middle: How Language Models Use Long Contexts* | https://arxiv.org/abs/2307.03172 | llm.long-context-eval, llm.rag |
| paper | `llm_liu2023_ring.txt` | 52 | Liu et al. 2023, *Ring Attention with Blockwise Transformers for Near-Infinite Context* | https://arxiv.org/abs/2310.01889 | llm.long-context |
| paper | `llm_liu2024_dora.txt` | 73 | Liu et al. 2024, *DoRA: Weight-Decomposed Low-Rank Adaptation* | https://arxiv.org/abs/2402.09353 | llm.qlora-peft |
| paper | `llm_liu2024_kivi.txt` | 54 | Liu et al. 2024, *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* | https://arxiv.org/abs/2402.02750 | llm.quantization-advanced |
| paper | `llm_liu2025_drgrpo.txt` | 68 | Liu et al. 2025, *Understanding R1-Zero-Like Training: A Critical Perspective* | https://arxiv.org/abs/2503.20783 | llm.rlvr-practice |
| paper | `llm_liu2025_muon.txt` | 56 | Liu et al. 2025, *Muon is Scalable for LLM Training* | https://arxiv.org/abs/2502.16982 | llm.pretraining-optimization |
| paper | `llm_mccandlish2018_critical_batch.txt` | 93 | McCandlish et al. 2018, *An Empirical Model of Large-Batch Training* | https://arxiv.org/abs/1812.06162 | llm.pretraining-optimization |
| paper | `llm_meister2020_beam.txt` | 49 | Meister et al. 2020, *If beam search is the answer, what was the question?* | https://arxiv.org/abs/2010.02650 | llm.decoding |
| paper | `llm_meister2022_typical.txt` | 106 | Meister et al. 2022, *Locally Typical Sampling* | https://arxiv.org/abs/2202.00666 | llm.sampling |
| paper | `llm_meng2022_rome.txt` | 110 | Meng et al. 2022, *Locating and Editing Factual Associations in GPT* | https://arxiv.org/abs/2202.05262 | llm.mech-interp |
| paper | `llm_meng2024_simpo.txt` | 102 | Meng et al. 2024, *SimPO: Simple Preference Optimization with a Reference-Free Reward* | https://arxiv.org/abs/2405.14734 | llm.preference-variants |
| paper | `llm_merrill2023_cot_expressivity.txt` | 57 | Merrill et al. 2023, *The Expressive Power of Transformers with Chain of Thought* | https://arxiv.org/abs/2310.07923 | llm.architecture-comparison, llm.chain-of-thought |
| paper | `llm_merrill2024_illusion_state.txt` | 70 | Merrill et al. 2024, *The Illusion of State in State-Space Models* | https://arxiv.org/abs/2404.08819 | llm.architecture-comparison |
| paper | `llm_micikevicius2022_fp8.txt` | 32 | Micikevicius et al. 2022, *FP8 Formats for Deep Learning* | https://arxiv.org/abs/2209.05433 | llm.quantization-advanced |
| paper | `llm_milakov2018_online_softmax.txt` | 21 | Milakov et al. 2018, *Online normalizer calculation for softmax* | https://arxiv.org/abs/1805.02867 | llm.flash-attention |
| paper | `llm_miller2024_error_bars.txt` | 36 | Miller et al. 2024, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations* | https://arxiv.org/abs/2411.00640 | llm.evaluation-metrics |
| paper | `llm_muennighoff2023_data_constrained.txt` | 160 | Muennighoff et al. 2023, *Scaling Data-Constrained Language Models* | https://arxiv.org/abs/2305.16264 | llm.pretraining-data, llm.scaling-laws-practice |
| paper | `llm_muennighoff2025_s1.txt` | 158 | Muennighoff et al. 2025, *s1: Simple test-time scaling* | https://arxiv.org/abs/2501.19393 | llm.test-time-compute |
| paper | `llm_narayanan2021_megatron.txt` | 75 | Narayanan et al. 2021, *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* | https://arxiv.org/abs/2104.04473 | llm.params-flops |
| paper | `llm_nguyen2024_minp.txt` | 109 | Nguyen et al. 2024, *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs* | https://arxiv.org/abs/2407.01082 | llm.sampling |
| paper | `llm_nie2025_llada.txt` | 105 | Nie et al. 2025, *Large Language Diffusion Models* | https://arxiv.org/abs/2502.09992 | excluded-list note (diffusion LMs) |
| paper | `llm_olmo2024_olmo2.txt` | 186 | OLMo et al. 2024, *2 OLMo 2 Furious* | https://arxiv.org/abs/2501.00656 | llm.pretraining-data, llm.training-stability |
| paper | `llm_olsson2022_induction.txt` | 135 | Olsson et al. 2022, *In-context Learning and Induction Heads* | https://arxiv.org/abs/2209.11895 | llm.in-context-learning |
| paper | `llm_openai2023_gpt4.txt` | 283 | OpenAI et al. 2023, *GPT-4 Technical Report* | https://arxiv.org/abs/2303.08774 | llm.evaluation-pitfalls, llm.sampling |
| paper | `llm_openai2025_gptoss.txt` | 75 | OpenAI et al. 2025, *gpt-oss-120b &amp; gpt-oss-20b Model Card* | https://arxiv.org/abs/2508.10925 | llm.attention-sinks, llm.moe-systems, llm.quantization-advanced, llm.sparse-attention |
| paper | `llm_oren2023_contamination.txt` | 61 | Oren et al. 2023, *Proving Test Set Contamination in Black Box Language Models* | https://arxiv.org/abs/2310.17623 | llm.evaluation-pitfalls |
| paper | `llm_orvieto2023_lru.txt` | 121 | Orvieto et al. 2023, *Resurrecting Recurrent Neural Networks for Long Sequences* | https://arxiv.org/abs/2303.06349 | llm.ssm |
| paper | `llm_ouyang2022_instructgpt.txt` | 182 | Ouyang et al. 2022, *Training language models to follow instructions with human feedback* | https://arxiv.org/abs/2203.02155 | llm.reward-modeling, llm.rlhf-ppo, llm.sft |
| paper | `llm_pagnoni2024_blt.txt` | 88 | Pagnoni et al. 2024, *Byte Latent Transformer: Patches Scale Better Than Tokens* | https://arxiv.org/abs/2412.09871 | llm.tokenization-effects |
| paper | `llm_pearce2024_reconcile.txt` | 44 | Pearce et al. 2024, *Reconciling Kaplan and Chinchilla Scaling Laws* | https://arxiv.org/abs/2406.12907 | llm.scaling-laws-practice |
| paper | `llm_penedo2024_fineweb.txt` | 91 | Penedo et al. 2024, *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* | https://arxiv.org/abs/2406.17557 | llm.pretraining-data |
| paper | `llm_peng2023_rwkv.txt` | 90 | Peng et al. 2023, *RWKV: Reinventing RNNs for the Transformer Era* | https://arxiv.org/abs/2305.13048 | llm.modern-rnns-hybrids |
| paper | `llm_peng2023_yarn.txt` | 53 | Peng et al. 2023, *YaRN: Efficient Context Window Extension of Large Language Models* | https://arxiv.org/abs/2309.00071 | llm.context-extension, llm.rope |
| paper | `llm_petrov2023_tokenizer_unfairness.txt` | 109 | Petrov et al. 2023, *Language Model Tokenizers Introduce Unfairness Between Languages* | https://arxiv.org/abs/2305.15425 | llm.tokenization-effects |
| paper | `llm_pope2022_inference_scaling.txt` | 80 | Pope et al. 2022, *Efficiently Scaling Transformer Inference* | https://arxiv.org/abs/2211.05102 | llm.kv-cache, llm.serving-systems |
| paper | `llm_porian2024_discrepancies.txt` | 95 | Porian et al. 2024, *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* | https://arxiv.org/abs/2406.19146 | llm.scaling-laws-practice |
| paper | `llm_press2021_alibi.txt` | 74 | Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* | https://arxiv.org/abs/2108.12409 | llm.long-context-eval, llm.positional-encodings |
| paper | `llm_puigcerver2023_soft_moe.txt` | 92 | Puigcerver et al. 2023, *From Sparse to Soft Mixtures of Experts* | https://arxiv.org/abs/2308.00951 | llm.moe-load-balancing |
| paper | `llm_qiu2025_gated_attention.txt` | 63 | Qiu et al. 2025, *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free* | https://arxiv.org/abs/2505.06708 | llm.attention-sinks |
| paper | `llm_radford2019_gpt2.txt` | 93 | Radford et al. 2019, Language Models are Unsupervised Multitask Learners (GPT-2) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | llm.lm-objectives, llm.tokenization, llm.training-stability |
| paper | `llm_rafailov2023_dpo.txt` | 93 | Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* | https://arxiv.org/abs/2305.18290 | llm.dpo |
| paper | `llm_raffel2019_t5.txt` | 205 | Raffel et al. 2019, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | https://arxiv.org/abs/1910.10683 | llm.lm-objectives, llm.positional-encodings |
| paper | `llm_raposo2024_mod.txt` | 42 | Raposo et al. 2024, *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models* | https://arxiv.org/abs/2404.02258 | llm.mot-mod |
| paper | `llm_rouhani2023_microscaling.txt` | 28 | Rouhani et al. 2023, *Microscaling Data Formats for Deep Learning* | https://arxiv.org/abs/2310.10537 | llm.quantization-advanced |
| paper | `llm_sardana2023_beyond_chinchilla.txt` | 55 | Sardana et al. 2023, *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* | https://arxiv.org/abs/2401.00448 | llm.scaling-laws-practice |
| paper | `llm_saunshi2025_looped_reasoning.txt` | 97 | Saunshi et al. 2025, *Reasoning with Latent Thoughts: On the Power of Looped Transformers* | https://arxiv.org/abs/2502.17416 | llm.looped-transformers |
| paper | `llm_schaeffer2023_mirage.txt` | 42 | Schaeffer et al. 2023, *Are Emergent Abilities of Large Language Models a Mirage?* | https://arxiv.org/abs/2304.15004 | llm.evaluation-pitfalls, llm.scaling-laws-practice |
| paper | `llm_schulman2015_gae.txt` | 44 | Schulman et al. 2015, *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | https://arxiv.org/abs/1506.02438 | llm.rlhf-ppo |
| paper | `llm_schulman2017_ppo.txt` | 27 | Schulman et al. 2017, *Proximal Policy Optimization Algorithms* | https://arxiv.org/abs/1707.06347 | llm.rlhf-ppo |
| paper | `llm_sennrich2015_bpe.txt` | 45 | Sennrich et al. 2015, *Neural Machine Translation of Rare Words with Subword Units* | https://arxiv.org/abs/1508.07909 | llm.tokenization |
| paper | `llm_shah2024_flashattention3.txt` | 70 | Shah et al. 2024, *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* | https://arxiv.org/abs/2407.08608 | llm.flash-attention-2-3 |
| paper | `llm_shao2024_deepseekmath.txt` | 86 | Shao et al. 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | https://arxiv.org/abs/2402.03300 | llm.rlvr-grpo |
| paper | `llm_shaw2018_relative.txt` | 18 | Shaw et al. 2018, *Self-Attention with Relative Position Representations* | https://arxiv.org/abs/1803.02155 | llm.positional-encodings |
| paper | `llm_shazeer2017_moe.txt` | 63 | Shazeer et al. 2017, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | https://arxiv.org/abs/1701.06538 | llm.moe |
| paper | `llm_shazeer2019_mqa.txt` | 21 | Shazeer et al. 2019, *Fast Transformer Decoding: One Write-Head is All You Need* | https://arxiv.org/abs/1911.02150 | llm.mqa-gqa-mla |
| paper | `llm_singh2024_tokenization_arithmetic.txt` | 91 | Singh et al. 2024, *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs* | https://arxiv.org/abs/2402.14903 | llm.tokenization-effects |
| paper | `llm_singhal2023_length_rlhf.txt` | 71 | Singhal et al. 2023, *A Long Way to Go: Investigating Length Correlations in RLHF* | https://arxiv.org/abs/2310.03716 | llm.preference-variants |
| paper | `llm_snell2024_test_time.txt` | 92 | Snell et al. 2024, *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* | https://arxiv.org/abs/2408.03314 | llm.test-time-compute |
| paper | `llm_stiennon2020_summarize.txt` | 132 | Stiennon et al. 2020, *Learning to summarize from human feedback* | https://arxiv.org/abs/2009.01325 | llm.reward-modeling |
| paper | `llm_su2021_roformer.txt` | 45 | Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding* | https://arxiv.org/abs/2104.09864 | llm.rope |
| paper | `llm_sun2023_retnet.txt` | 41 | Sun et al. 2023, *Retentive Network: A Successor to Transformer for Large Language Models* | https://arxiv.org/abs/2307.08621 | llm.linear-attention, llm.modern-rnns-hybrids |
| paper | `llm_sun2024_massive_activations.txt` | 74 | Sun et al. 2024, *Massive Activations in Large Language Models* | https://arxiv.org/abs/2402.17762 | llm.attention-sinks |
| paper | `llm_team2024_gemma2.txt` | 63 | Gemma Team et al. 2024, *Gemma 2: Improving Open Language Models at a Practical Size* | https://arxiv.org/abs/2408.00118 | llm.distillation, llm.training-stability |
| paper | `llm_touvron2023_llama2.txt` | 263 | Touvron et al. 2023, *Llama 2: Open Foundation and Fine-Tuned Chat Models* | https://arxiv.org/abs/2307.09288 | llm.mqa-gqa-mla |
| paper | `llm_vaswani2017_attention.txt` | 39 | Vaswani et al. 2017, *Attention Is All You Need* | https://arxiv.org/abs/1706.03762 | llm.architecture-comparison, llm.causal-attention, llm.cross-attention, llm.positional-encodings, llm.self-attention, llm.transformer-block |
| paper | `llm_vonoswald2022_icl_gd.txt` | 100 | von Oswald et al. 2022, *Transformers learn in-context by gradient descent* | https://arxiv.org/abs/2212.07677 | llm.in-context-learning |
| paper | `llm_wang2022_lm_arch_objective.txt` | 87 | Wang et al. 2022, *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?* | https://arxiv.org/abs/2204.05832 | llm.lm-objectives |
| paper | `llm_wang2022_self_consistency.txt` | 90 | Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning in Language Models* | https://arxiv.org/abs/2203.11171 | llm.chain-of-thought |
| paper | `llm_wang2024_auxfree.txt` | 32 | Wang et al. 2024, *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* | https://arxiv.org/abs/2408.15664 | llm.moe-load-balancing |
| paper | `llm_wang2025_hrm.txt` | 74 | Wang et al. 2025, *Hierarchical Reasoning Model* | https://arxiv.org/abs/2506.21734 | llm.looped-transformers |
| paper | `llm_wei2021_flan.txt` | 144 | Wei et al. 2021, *Finetuned Language Models Are Zero-Shot Learners* | https://arxiv.org/abs/2109.01652 | llm.sft |
| paper | `llm_wei2022_cot.txt` | 135 | Wei et al. 2022, *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | https://arxiv.org/abs/2201.11903 | llm.chain-of-thought |
| paper | `llm_wei2022_emergent.txt` | 94 | Wei et al. 2022, *Emergent Abilities of Large Language Models* | https://arxiv.org/abs/2206.07682 | llm.scaling-laws-practice |
| paper | `llm_willard2023_structured_generation.txt` | 29 | Willard et al. 2023, *Efficient Guided Generation for Large Language Models* | https://arxiv.org/abs/2307.09702 | llm.decoding |
| paper | `llm_wortsman2023_instabilities.txt` | 69 | Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities* | https://arxiv.org/abs/2309.14322 | llm.mup, llm.training-stability |
| paper | `llm_xiao2022_smoothquant.txt` | 56 | Xiao et al. 2022, *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* | https://arxiv.org/abs/2211.10438 | llm.quantization |
| paper | `llm_xiao2023_attention_sinks.txt` | 71 | Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks* | https://arxiv.org/abs/2309.17453 | llm.attention-sinks, llm.long-context |
| paper | `llm_xie2021_icl_bayesian.txt` | 80 | Xie et al. 2021, *An Explanation of In-context Learning as Implicit Bayesian Inference* | https://arxiv.org/abs/2111.02080 | llm.in-context-learning |
| paper | `llm_xie2023_doremi.txt` | 70 | Xie et al. 2023, *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* | https://arxiv.org/abs/2305.10429 | llm.pretraining-data |
| paper | `llm_xie2025_mhc.txt` | 57 | Xie et al. 2025, *mHC: Manifold-Constrained Hyper-Connections* | https://arxiv.org/abs/2512.24880 | llm.training-stability |
| paper | `llm_xiong2023_long_context_scaling.txt` | 74 | Xiong et al. 2023, *Effective Long-Context Scaling of Foundation Models* | https://arxiv.org/abs/2309.16039 | llm.context-extension, llm.long-context-eval |
| paper | `llm_xu2024_dpo_vs_ppo.txt` | 66 | Xu et al. 2024, *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* | https://arxiv.org/abs/2404.10719 | llm.preference-variants |
| paper | `llm_xue2021_byt5.txt` | 65 | Xue et al. 2021, *ByT5: Towards a token-free future with pre-trained byte-to-byte models* | https://arxiv.org/abs/2105.13626 | llm.tokenization-effects |
| paper | `llm_yang2022_mup.txt` | 163 | Yang et al. 2022, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* | https://arxiv.org/abs/2203.03466 | llm.mup |
| paper | `llm_yang2024_gated_deltanet.txt` | 79 | Yang et al. 2024, *Gated Delta Networks: Improving Mamba2 with Delta Rule* | https://arxiv.org/abs/2412.06464 | llm.linear-attention, llm.modern-rnns-hybrids |
| paper | `llm_yang2025_qwen3.txt` | 115 | Yang et al. 2025, *Qwen3 Technical Report* | https://arxiv.org/abs/2505.09388 | llm.moe-systems |
| paper | `llm_ye2024_diff_transformer.txt` | 65 | Ye et al. 2024, *Differential Transformer* | https://arxiv.org/abs/2410.05258 | llm.attention-sinks |
| paper | `llm_yu2025_dapo.txt` | 40 | Yu et al. 2025, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* | https://arxiv.org/abs/2503.14476 | llm.rlvr-practice |
| paper | `llm_yuan2025_nsa.txt` | 65 | Yuan et al. 2025, *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* | https://arxiv.org/abs/2502.11089 | llm.sparse-attention |
| paper | `llm_yue2025_rl_beyond_base.txt` | 97 | Yue et al. 2025, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* | https://arxiv.org/abs/2504.13837 | llm.rlvr-practice |
| paper | `llm_zhang2023_h2o.txt` | 145 | Zhang et al. 2023, *H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* | https://arxiv.org/abs/2306.14048 | llm.long-context |
| paper | `llm_zheng2023_mtbench.txt` | 87 | Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | https://arxiv.org/abs/2306.05685 | llm.evaluation-pitfalls |
| paper | `llm_zheng2025_gspo.txt` | 22 | Zheng et al. 2025, *Group Sequence Policy Optimization* | https://arxiv.org/abs/2507.18071 | llm.rlvr-practice |
| paper | `llm_zhou2022_expert_choice.txt` | 46 | Zhou et al. 2022, *Mixture-of-Experts with Expert Choice Routing* | https://arxiv.org/abs/2202.09368 | llm.moe-load-balancing |
| paper | `llm_zhou2023_lima.txt` | 64 | Zhou et al. 2023, *LIMA: Less Is More for Alignment* | https://arxiv.org/abs/2305.11206 | llm.sft |
| paper | `llm_ziegler2019_finetuning_prefs.txt` | 91 | Ziegler et al. 2019, *Fine-Tuning Language Models from Human Preferences* | https://arxiv.org/abs/1909.08593 | llm.rlhf-ppo |
| paper | `llm_zoph2022_stmoe.txt` | 119 | Zoph et al. 2022, *ST-MoE: Designing Stable and Transferable Sparse Expert Models* | https://arxiv.org/abs/2202.08906 | llm.moe-load-balancing |
| blog/web | `llm_bricken2023_monosemanticity.txt` | 160 | Bricken et al. 2023, Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | https://transformer-circuits.pub/2023/monosemantic-features/index.html | llm.mech-interp |
| blog/web | `llm_deepseek2026_v4_modelcard.txt` | 5 | DeepSeek V4 Technical Documentation / model card (Apr 2026; used for the recent-developments note) | https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf | recent-developments note |
| blog/web | `llm_eleuther2021_rotary.txt` | 23 | EleutherAI, Rotary Embeddings: A Relative Revolution (2021) | https://blog.eleuther.ai/rotary-embeddings/ | llm.rope |
| blog/web | `llm_eleuther2023_transformer_math.txt` | 18 | EleutherAI, Transformer Math 101 (2023) | https://blog.eleuther.ai/transformer-math/ | llm.params-flops |
| blog/web | `llm_eleuther_mutransfer.txt` | 33 | EleutherAI & Cerebras, The Practitioner's Guide to the Maximal Update Parameterization | https://blog.eleuther.ai/mutransfer/ | llm.mup |
| blog/web | `llm_elhage2021_circuits_framework.txt` | 121 | Elhage et al. 2021, A Mathematical Framework for Transformer Circuits | https://transformer-circuits.pub/2021/framework/index.html | llm.in-context-learning, llm.mech-interp, llm.transformer-block |
| blog/web | `llm_he2022_brrr.txt` | 20 | Horace He, Making Deep Learning Go Brrrr From First Principles | https://horace.io/brrr_intro.html | llm.arithmetic-intensity |
| blog/web | `llm_hf_moe_explained.txt` | 29 | Hugging Face blog, Mixture of Experts Explained | https://huggingface.co/blog/moe | llm.moe |
| blog/web | `llm_jordan2024_muon.txt` | 23 | Keller Jordan, Muon: An optimizer for hidden layers in neural networks (2024) | https://kellerjordan.github.io/posts/muon/ | llm.pretraining-optimization |
| blog/web | `llm_kipply2022_inference_arithmetic.txt` | 32 | kipply, Transformer Inference Arithmetic (2022) | https://kipp.ly/transformer-inference-arithmetic/ | llm.arithmetic-intensity, llm.kv-cache |
| blog/web | `llm_raschka_2026_dream_of_spring.txt` | 33 | Sebastian Raschka, A Dream of Spring for Open-Weight LLMs: 10 Architectures from Jan–Feb 2026 (used for the recent-developments note) | https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight | recent-developments note |
| blog/web | `llm_raschka_beyond_standard_llms.txt` | 47 | Sebastian Raschka, Beyond Standard LLMs (2025) | https://magazine.sebastianraschka.com/p/beyond-standard-llms | llm.mamba |
| blog/web | `llm_raschka_big_arch_comparison.txt` | 88 | Sebastian Raschka, The Big LLM Architecture Comparison (updated through Apr 2026) | https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison | llm.linear-attention, llm.moe-systems, llm.mqa-gqa-mla, llm.transformer-block |
| blog/web | `llm_raschka_deepseek_v3_to_v32.txt` | 36 | Sebastian Raschka, A Technical Tour of the DeepSeek Models from V3 to V3.2 | https://magazine.sebastianraschka.com/p/technical-deepseek | llm.rlvr-grpo |
| blog/web | `llm_schulman2020_kl_approx.txt` | 6 | John Schulman, Approximating KL Divergence (blog) | http://joschu.net/blog/kl-approx.html | llm.rlhf-ppo |
| blog/web | `llm_weng2023_inference_optimization.txt` | 43 | Lilian Weng, Large Transformer Model Inference Optimization (2023) | https://lilianweng.github.io/posts/2023-01-10-inference-optimization/ | llm.kv-cache, llm.quantization |
| blog/web | `llm_weng2023_transformer_family_v2.txt` | 68 | Lilian Weng, The Transformer Family Version 2.0 (2023) | https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/ | llm.long-context |
| blog/web | `llm_weng2024_reward_hacking.txt` | 50 | Lilian Weng, Reward Hacking in Reinforcement Learning (2024) | https://lilianweng.github.io/posts/2024-11-28-reward-hacking/ | llm.reward-modeling, llm.rlvr-practice |
| blog/web | `llm_weng2025_why_we_think.txt` | 55 | Lilian Weng, Why We Think (2025) | https://lilianweng.github.io/posts/2025-05-01-thinking/ | llm.chain-of-thought |

## Shared-cache files (other curator) cited by the LLM plan
`papers/xiong2020_preln.txt`, `papers/shazeer2020_glu.txt`, `papers/zhang2019_rmsnorm.txt`, `papers/ba2016_layernorm.txt`,
`papers/rombach2022_ldm.txt`, `papers/brown2020_gpt3.txt`, `papers/loshchilov2017_adamw.txt`, `papers/kingma2014_adam.txt`,
`d2l/d2l_lr_scheduler.txt`, `d2l/d2l_information_theory.txt`, `pml1.txt`.
