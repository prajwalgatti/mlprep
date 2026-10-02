# Content review: sys.gpu-basics, sys.roofline, sys.flops-mfu

Reviewer: critic agent, 2026-10-02. Rubric: `docs/review-rubric.md`. Plan: `docs/content-plan-applied.md` Part D, plus
`docs/reviews/plan-applied-review.md`.

Checks run:
- Recomputed every worked number, MCQ answer and distractor in Python.
- Viewed all 9 figures (workbench theme, plus terminal or editorial).
- Grepped the cached sources: Scaling Book Parts 1, 4 and 12; NVIDIA GPU-perf and matmul guides; PyTorch CUDA semantics;
  He 2022; Dao 2022; PaLM; Korthikanti; Narayanan; Llama 3; Shoeybi; Kaplan.
- `validate.py`: 0 errors, 0 warnings for these three files.

## Writer's flagged points (verified independently)

| Point | Verdict |
|---|---|
| Step-time figure is illustrative with assumed costs | **Labelled OK.** The caption opens with "Illustrative" and lists the assumptions; I recomputed 2.15 → MFU 46%, HFU 62%. Two fixes. (a) Say in the caption that 0.25 and 0.15 are in units of ideal compute time. (b) The 46% lands right next to PaLM's 46.2% in the reference list, which can read as PaLM's breakdown. Change one assumed cost (e.g. memory-bound 0.30 → MFU ≈ 45%) or add "(not any published run's breakdown)". |
| Wave quantisation, one tile per SM | **Fine.** It is exactly NVIDIA's own A100 model (256×128 tiles, one block per SM, wave = 108 tiles; Matmul Background §3.2). Cite that sentence so the model looks sourced, not invented. |
| LayerNorm ≈ 8 FLOPs/element | **Fine.** mean (1) + centre/square/accumulate (3) + normalise (2) + scale/shift (2) ≈ 8. NVIDIA's table says "< 10". The text already says it is convention-dependent. |
| 50,257 → 50,304 | **Arithmetic correct** (786·64 = 50,304; +0.094%), but no cached source pads to 50,304 (that is nanoGPT). Shoeybi et al. 2019 (cached, l.450–455) pads GPT-2's 50,257 to **51,200** (a multiple of 128·8 for TP). Add that as the sourced real-world example, or cite the 50,304 as your own arithmetic. |
| Cards of 300–370 words | **Real problem in gpu-basics** (7 of 9 cards > 300; cards [4]–[6] are 354–371). Roofline [2] is 350. See G findings. |
| Forward refs to sys.collectives, sys.zero | **Collectives OK.** The ring factor 2 comes with a one-line reason (RS + AG). **ZeRO is slightly mis-framed** (see flops-mfu finding 6). |
| Narayanan 140 TFLOP/s → HFU 44.9%, MFU ≈ 33.7% | **Correct.** Narayanan l.1561/1585 say X includes recomputation (96Bslh² = 72·4/3). The exact ratio with the un-recomputed logit term is 0.7507 vs 0.75, so 33.7% holds. |
| Comm roofline 2200 / 2475 | **Correct, and matches the Scaling Book** (Part 12 "Data Parallelism"). Nit: the book divides by 990e12. With the lesson's 989e12, 400 GB/s gives 2472. Write "≈ 2470" or say the book rounds C to 990. |

## Overlap with the Scaling Book lessons (sb.roofline-basics, sb.roofline-matmul)

**The writer says no examples are shared. That is not accurate.**

1. **Identical example (major).** `sys.roofline` q3 uses $[512, 8192]\times[8192, 8192]$ → I ≈ 455. This is the same matmul
   and the same 455 computation as `sb.roofline-basics` q13. Replace it with a case where the $I \approx B$ shortcut flips
   the verdict. For example, the Llama-3-8B MLP $[320, 4096]\times[4096, 14336]$ has exact I ≈ 291 < 295, so it is
   memory-bound, while $I \approx B$ says 320 and compute-bound. The exact crossover is ≈ 325 tokens. That teaches more
   than the old item.
2. **Near-duplicate (major).** `sys.roofline` q1 (bf16 add → 1/6, distractors 1/4 and 1/12) mirrors `sb.roofline-basics`
   q6 (fp32 add → 1/12) and q5 (fp32 dot → 1/4). sb q6's answer is a distractor here. Swap in an op the sb lessons don't
   use: a bias add with a broadcast `[h]` vector (I = 1/4; tempting distractor 1/6 from treating the bias as a full tensor),
   or a fused add + ReLU.
3. **Same template (minor).** `sys.roofline` q2 (A100 "with sparsity" → 156, inverted-ratio distractor) has the same
   structure as `sb.roofline-basics` q4 (H100 sparsity → 295, inverted-ratio distractor). Reframe, e.g. H100 → H200
   (same 989 TFLOP/s, 4.8 TB/s, ridge ≈ 206): which of a memory-bound op and a compute-bound op speeds up?
4. **Same scenario, different number (major for consistency).** `sys.roofline` card [5] and f10 cover fp8 math with bf16
   storage on H100 and give **B\* ≈ 736** (exact). `sb.roofline-matmul` q5 asks the same question and answers **≈ 590**
   ($I \approx B$). A reader who does both will think one is wrong. Add one sentence: "≈ 591 with $I \approx B$; ≈ 736
   once activation bytes are counted for $4096\times11008$". Or pick a scenario the sb lesson doesn't cover.
5. **Similar open card (minor).** `sys.roofline` f7 ("Draw an H100 roofline … place ops") vs `sb.roofline-basics` f11
   ("Sketch a roofline plot …"). f7 is differentiated by the op placement; keep it, but drop its "what does not help"
   overlap.
6. **Conceptual overlap is acceptable.** The max/sum ≤ 2× bound, $I \approx B$, "tokens not sequences" and the
   $DF/(D+F)$ ceiling are core to both, and the sys lesson uses GPU-specific examples (7B MLP sweep, LayerNorm bandwidth,
   comm roofline).
7. **Outside the named scope (minor, FYI).** `sys.flops-mfu` q3 (7B on 2T tokens; 6ND = 8.4e22) shares its setup with
   `sb.flops-counting` q7. The 70B/15T worked example (6.3e24) is also `sb.flops-counting` f7. The 70B/15T case is the
   plan's own problem statement, so keep it, but change q3's model or token count.

---

## sys.gpu-basics

**Verdict: revise**

### Findings

1. **blocking**: `explainer[2]` "How kernels get there: reuse", and `f9`. The naive-kernel intensity is off by 2×. A
   thread computing $C_{ij}$ loads $K$ elements of $A$ **and** $K$ of $B$ (4K bytes) for 2K FLOPs: **0.5 FLOP/byte**,
   not 1. "Each loaded number feeds one multiply-add, so 2 bytes buys 2 FLOPs" double-counts, because every multiply-add
   consumes two loaded numbers. Likewise, $T\times T$ tiling gives **T/2** FLOPs per byte in bf16 (per K-step: $2T^2k$
   FLOPs over $4Tk$ bytes), not "T uses per load" read as T FLOPs/byte. Fix both numbers ("≈ 600× too little" in f9).
   Then add the payoff, which the card currently implies is easy: a 128×256 tile gives $bm\,bn/(bm+bn) \approx 85$
   FLOPs/byte from its own loads, below the 295 needed. The rest must come from L2 hits between neighbouring blocks (and
   register-level reuse). Today "cutting HBM traffic further" hides that this is essential, not a bonus.
2. **major**: `explainer[1]`. It computes **67** TFLOP/s for the CUDA cores, then immediately uses **66** ("990/66 ≈ 15×",
   "66/990 ≈ 7%"). q2's explanation also uses 66, while q1, f1 and key results use 67. Pick one: 67 (the derivation)
   with 990/67 ≈ 15×, ≈ 6.8% ≈ 7%. Mention that the Scaling Book prints 66.
3. **major (G)**: overfull cards. Prose word counts (display math excluded): [0] 306, [1] 322, [2] 347, [3] 315,
   [4] 354, [5] 371, [6] 358. Fixes:
   - Split [5] "Host and device" into "Asynchrony and streams" and "Sync points, timing traps and launch overhead".
   - Trim [2]: move coalescing to f7 only, or cut it to one sentence.
   - Trim [4]: the alignment paragraph can lose the vocab example once it moves to f4.
   - That makes 10 cards, inside the brief's 6–10.
4. **major (E)**: three pure-recall MCQs (limit 2): q3 (which-is-false on blocks/SMs), q12 (TF32 definition) and q13
   (which-is-false on the hierarchy). Turn q13 into a predict item. For example: "A kernel's tile working set grows from
   200 kB to 300 kB per block. What happens on H100?" Answer: it no longer fits in the 256 kB SMEM; the tile must shrink
   or spill, cutting reuse.
5. **minor**: `explainer[4]` worked example. Add NVIDIA's own wording that this model (one tile per SM) is what their
   A100 measurements use. Also note that 64-element alignment is NVIDIA's **A100** advice. The interviewer and key-results
   cards generalise it as "ideally 64" without the scope.
6. **minor**: `explainer[4]` vocab padding. Cite Shoeybi et al. 2019 (51,200, cached), or label 50,304 as your own
   rounding (see table above).
7. **minor (F)**: the `wave-quantisation` caption says "Steps come from tiles (every 256 columns) and waves (every 132
   tiles)". In the plot, time only jumps when an added tile column pushes the count past a multiple of 132. Most
   256-column boundaries produce no step. Reword: "time jumps when a new tile column (every 256 columns) pushes the tile
   count past a multiple of 132".
8. **minor (B)**: `explainer[1]` uses "warp scheduler" and "warp" before warps are defined in [3]. Add "(a warp is a
   group of 32 threads; next cards)".
9. **minor**: the reading note references `sb.gpu-chip`. It is planned (scalingbook plan, order 330) but not yet written.
   Fine if it lands; otherwise drop the id.

### Strengths
- The layer walkthrough ([6]) and the streams figure tie the vocabulary to real costs (0.56 ms vs 80 µs vs launch
  overhead).
- q6 (N = 8200 exactly fills 8 waves) is an excellent predict question.
- Async/sync coverage (`.item()`, `time.time()`, `CUDA_LAUNCH_BLOCKING`) is practical and correct.

---

## sys.roofline

**Verdict: revise** (mainly the Scaling Book duplicates; the core teaching is strong)

### Findings

1. **major**: q3 duplicates `sb.roofline-basics` q13. Replace it (see Overlap 1; the suggested $[320,4096]\times[4096,14336]$
   gives I ≈ 291).
2. **major**: q1 near-duplicates `sb.roofline-basics` q5/q6 (Overlap 2).
3. **major**: card [5] and f10 give fp8 crossover 736 vs `sb.roofline-matmul` q5's 590 for the same scenario. Reconcile
   in one sentence (Overlap 4).
4. **major (A/E)**: the q9 explanation says the "slower, since fp8 arithmetic is not supported on CUDA cores" option
   "invents a hardware restriction". As far as I know, Hopper's CUDA cores have **no** native fp8 arithmetic: fp8 is a
   Tensor Core input format plus conversion instructions. The distractor's premise is true; its conclusion ("slower") is
   what's wrong. Rewrite the explanation: "Even if you emulate fp8 math (CUDA cores convert to fp16/fp32), a memory-bound
   op's time is bytes/W, so it is neither faster nor meaningfully slower." Better still, rephrase the stem as "computes in
   fp16 instead of fp32 internally" to avoid the unphysical premise. Card [5]'s "Corollary for elementwise ops" has the
   same premise; phrase it as "lower-precision arithmetic".
5. **minor**: q2 shares the sb sparsity template (Overlap 3).
6. **minor**: `explainer[6]`. The 2475 threshold is computed with C = 990e12 while the lesson uses 989e12 everywhere else
   (989/400 = 2472). Say "≈ 2470 (the Scaling Book rounds C to 990e12 and gets 2475)", or just "≈ 2500".
7. **minor (B)**: `explainer[6]` justifies the backward's 4N FLOPs per token as "`sys.flops-mfu` derives the 2 + 4 split".
   That is a forward reference, but [3] of this lesson already shows the two backward matmuls each cost 2BDF. Point there
   instead.
8. **minor**: `explainer[3]` and f-cards link `llm.arithmetic-intensity`. The LLM plan was revised today and that id was
   **merged into `llm.kv-cache` and `llm.flash-attention`** (content-plan-llms.md l.12–13, 52). The same applies to
   flops-mfu [7]. Drop the dead id.
9. **minor (C)**: "below the ridge extra FLOP/s does nothing; above it extra bandwidth does nothing" appears in [1], [5]
   ("What does not help"), f6 and key results. Cut the first two sentences of [5]'s "What does not help".
10. **minor (G)**: `explainer[2]` is 350 words. Moving the fusion paragraph's last sentence (GELU vs ReLU) into f4 brings
    it near 300.
11. **minor (F)**: in `roofline-h100` and `roofline-points`, the 4096-token marker (point C) sits on top of the "bf16 roof"
    label. Nudge the label right or below.
12. **minor (D)**: the plan's subtopic 6 names a TP condition. It is only deferred ("in their lessons"). Acceptable, but
    one line would help: TP all-reduce bytes ∝ $Bh$ vs FLOPs ∝ $Bh^2/t$, so the condition is on $h/t$, not on the batch.

### Strengths
- The intensity derivations ($1/I = 1/B + 1/D + 1/F$, backward has the same I, the $DF/(D+F)$ cap) are clean and shown
  step by step.
- The 7B MLP sweep plus batch-sweep figure ties the roofline to the Tuning Playbook's throughput sweep exactly as the
  insertions table asks.
- q11 and q13 (effective intensity from re-reads) test understanding, not recall.

---

## sys.flops-mfu

**Verdict: revise** (small fixes; the structure and numbers are good)

### Findings

1. **major (A)**: `explainer[4]`, f4 and key results say selective recomputation gives hardware/model ≈ $1 + s/(6h)$.
   That is Korthikanti's eq. 9 as printed, but **the lesson's own numbers contradict it**.
   - Selective recompute adds one *forward* of the score and value matmuls: $4sh$ per token per layer (Korthikanti's own
     text, l.946–950: "2Bs²h + 2Bs²h").
   - Model FLOPs per layer per token are $72h^2 + 12sh$.
   - So the ratio is $\approx 1 + 4sh/72h^2 = 1 + s/(18h)$.
   - The paper's eq. 8 writes $s/3h$ where the arithmetic gives $2s/9h$; its Table 5 numbers follow its formula.

   Fix: derive $1 + s/(18h)$ in one line and add "Korthikanti et al. print ≈ 1 + s/6h (their eq. 9); their own term-by-term
   count gives s/18h. Either way it's ~1%." Keep 56.3%/57.0% as reported.
2. **major (B)**: symbol clash. Card [0] defines **$n$ = number of GPUs**. Card [1] and the figure then use $n$ as a matmul
   dimension ($[k, n]$, "2bkn"), and the MFU formula uses $n\cdot C$ again. Rename the matmul dims to $[b,k]\times[k,m]$
   (and update the figure), or write $n_{gpu}$. Also note that the previous lesson (`sys.roofline`) used $D$ for a hidden
   width, while here $D$ is tokens. One sentence in [0] ("here $D$ is tokens, not a width") prevents the confusion.
3. **major (E)**: q3 shares its setup with `sb.flops-counting` q7 (7B, 2T tokens). Change it to, e.g., 8B on 15T tokens
   on 2,048 H100s at 40% MFU ($7.2\times10^{23}$ FLOPs, ≈ 10.3 days). Recompute the distractors.
4. **minor (A/E)**: q2's 63% distractor is explained as "a peak half as large, e.g. dividing by a different GPU's peak".
   That is not a real misconception. Replace it with "About 16%, dividing by the spec sheet's 1,979 TFLOP/s sparse peak"
   (6·13e9·4000/1979e12 = 15.8%).
5. **minor (A)**: q10's "About 37%, averaging total and active parameters" computes to **36.4%** (N = 30B). Write 36%.
6. **minor (A)**: interviewer [8] and f9 say "ZeRO-3 moves about 1.5× DDP's bytes … still clears it". The conclusion
   holds, but the framing implies the threshold rises 1.5× (to ≈ 3,700). Plan-review B7 reconciled this: the forward
   all-gather overlaps the forward pass, so bytes per FLOP are 1/B in both phases and the threshold stays ≈ C/W. Say that
   (with the `sys.zero` pointer), or say "even if all 1.5× were exposed to the backward, ≈ 3,700 ≪ 15,600".
7. **minor (B)**: `explainer[0]` says "$N$ … strictly, those used in matmuls; see card 3". That definition is in card 2
   (the next card). Fix the pointer.
8. **minor (B)**: `explainer[6]` gives the pipeline bubble $(p-1)/(m+p-1)$ with no definition of stages or micro-batches.
   Add a half-sentence: "the model is split into $p$ sequential stages and each batch into $m$ micro-batches; the pipeline
   is idle while it fills and drains".
9. **minor**: `explainer[2]` "Conventions to state". Kaplan's context term ($2\,n_{layer} n_{ctx} d$ forward) is itself
   **half** of the uncausal $4Lsh$, i.e. it already applies the causal halving. Say so; it's a common source of
   "my count disagrees with Kaplan's".
10. **minor (F)**: `step-time` (see the table above): add units to the caption and avoid the 46% coincidence with PaLM.
    In the **terminal** theme, "model FLOPs at peak" (`p.good`) and "kernels at 85%" (`p.c(0)`) render as the same lime
    green. Use a hatched or faint fill for the kernel-inefficiency segment so it reads as waste in every theme.
11. **minor**: q3's 2.8-day explanation ("uses 8ND while keeping 45% as if it were HFU-free") is garbled. Write "counts
    the recomputation FLOPs (8ND) as model FLOPs at the same 45%".
12. **minor (D)**: the plan wanted `mfu-gauge` exposed comm "from the α–β/last-bucket model". The writer uses an assumed
    0.15. That is acceptable given the forward dependency and the illustrative label; just note it in the report.

### Strengths
- The 6N derivation is the best-taught part of the three lessons, with an honest "what it leaves out" list.
- The PaLM check (45.7% → 46.2%) and the Megatron 8TP/(nX) ↔ 6ND reconciliation are exactly what an interviewer
  probes.
- q4 and q7 (recompute ×4/3, HFU vs MFU) force reasoning, and all numbers recompute correctly.

---

## Numbers verified (all correct unless listed above)
- gpu-basics: 16,896 lanes, 66.9 TFLOP/s; 295 FLOPs/byte; 8 and 16 resident warps; 512/544 tiles, 4 → 5 waves, 97%,
  82%, +25%; 300 blocks → 3 waves, 75.8%; N = 8200 → 1056 = 8·132; 5.5e11 FLOPs, 0.556 ms; 134 MB, 80 µs; 302 MB,
  90 µs; 50,304 (+0.094%).
- roofline: 120 µs vs 1 µs; ridges 295/591/156; n/3 = 85/341/1365; 7B MLP: I = 1.0/15.9/62.7/122.7/1727, T_mem
  26.9/27.1/27.5/28.1/63.9 µs, T_math 11.7/373 µs, 42%; B\* = 327.6 (exact), 736 (fp8 math); cap 2985; q3 455; q4 n ≈ 886;
  q8 +2.1%; q11 167.5 (95.5%); LayerNorm 2.95 TB/s (88%); comm 2198/2473 (book 2475).
- flops-mfu: 12.7%, 23,548, 9,419, 17.0%; PaLM 45.70%; attention 6.44e9 → 15.3%/13.3%; 70B: 6.3e24, 6.48e18 FLOP/s,
  11.25 days; GPT-3 33.9 days; 38.4% required MFU; decode 4.18 ms, 0.34%; HFU 44.9% → MFU 33.65%; chained 3.08 s, 53%,
  15,625; q2 31.5/10.5/42.1; q3 2.13/0.71/2.84 days; q6 26.7%/160%/4.4%; q10 15.8/57.0/36.4/5.3; q12 3.15e23. Breakdown
  figure: 8/14/25/57/84% at 2k–128k, half at 25.2k.
