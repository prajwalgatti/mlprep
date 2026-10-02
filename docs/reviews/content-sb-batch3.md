# Content review: Scaling Book batch 3 (lessons 90–120)

Reviewed: `sb.flops-counting`, `sb.transformer-accounting`, `sb.transformer-memory`, `sb.flash-attention`.
Checked against `sb_04_transformers.md` (all of ch. 4 including App. A), `sb_06_applied-training.md`, `sb_07_inference.md`
§"What about attention?" and Q5, `sb_05_training.md` (context-parallelism note, 10 bytes/param), and `sb_12_gpus.md` (MoE E/k).
I recomputed every number in Python, including all MCQ answers and distractors, and the backward identity (numerically, against
finite differences). I looked at all 10 figures: workbench for every one, plus terminal or editorial. `validate.py` reports
0 errors and no warnings for these files. Card counts are 9 explainer cards, 12 MCQs and 9–10 flashcards (3 open) per lesson. No card
is over 300 words. The correct choice is the longest at most once per lesson.

## The coordinator's claims, checked independently

| Claim | Verdict | Notes |
|---|---|---|
| Ch. 4 Q1 is 17.4B exactly; the book says 16B | **Correct.** | $L(3DF+4D^2+2D)+2DV$ = 17.44B at $D = 4096$. The book's own arithmetic, with $D$ rounded to 4e3 and $F$ to 16e3, gives 16.6B, which it quotes as "16B". The lesson says so explicitly. The plan's "16B" should be read as the book's rounding. |
| LLaMA-3 70B's real crossover is near 52k, not 65k | **Correct.** | $12TNH = 18DF + 12D(N+K)H$ gives $T$ = 52,224, non-causal, with $K = 8$ and $F = 28672$. The q12 MLP-only crossover (43,008) and the q7 share (13.6%) also recompute. |
| MoE $B > 120E/k$, which is 3840 for DeepSeek | **Correct.** | Matches ch. 4 Q8 word for word. The bf16 form $240E/k$ matches ch. 7 Q5, which uses 240·16/2 = 1920. |
| 84 TB of activations | **Correct.** | $2\cdot20\cdot4\text{e}6\cdot8192\cdot64$ = 8.39e13 B. Block remat gives 4.19e12, i.e. 4.2 TB. Both match the book. |
| Prefill compute-bound above 480 tokens (MHA), ≈ 270 (G = 8) | **Correct, but the book contradicts itself.** | $TG/(G+1) > 240$ gives 480 and 270. Ch. 7 says "roughly > 480". **Ch. 4 Q4's answer says "S = 240"**, and the lesson doesn't mention this (see transformer-memory finding 1). |
| Kernel intensity ≈ query-block size | **Correct as a back-of-envelope, and labelled as the writer's own.** | Per KV block, $4b_qb_kH$ FLOPs over $4b_kH$ bytes gives $b_q$. This is the standard FA-2 IO count. It does conflict with lesson 110 unless the two are reconciled (see flash-attention finding 2). |
| Causal skipping computes 36 of 64 blocks at n = 8 | **Correct.** | $n(n+1)/2$ = 36, which is 56%. |
| FlashAttention backward identity | **Correct.** | It matches App. A. I verified numerically both $\sum_j S_{ij}dS_{ij} = \sum_d dO_{id}O_{id}$ and $ds = S\odot(dS - \text{rowsum}(S\odot dS))$ against finite differences. |
| 44 days for LLaMA-3 70B on a v5p pod | **Correct.** | 6.3e24 / (8960 · 4.59e14 · 0.4) = 3.8e6 s = 44.3 days. Matches ch. 6. |
| DeepSeek-V3 MFU ≈ 22% | **Correct.** | Exact arithmetic gives 3.29e24 / 1.52e25 = 21.6%. The book's 21.7% comes from rounding to 3.3e24. "≈ 22%" is fine. |

---

## sb.flops-counting

**Verdict: pass** (minor polish only)

### Findings
1. **minor: the explanation doesn't address q12's distractor "About 9%".** It is the forward-only slip: $2ND = 0.4\times10^{20}$ over $4.5\times10^{20}$ gives 8.8%. Add one clause saying so.
2. **minor: the fwd-bwd figure bottom labels read "inference: 2" and "backward: 4"** with no unit. Make them "forward: 2NPM" and "backward: 4NPM".
3. **minor: explainer[6], "the 80–95% achievable peak".** This is unsourced. Cite it or cut it. The MFU sentence works without it.
4. **minor: explainer[1], "Hardware peaks are quoted the same way, so the two cancel consistently."** "Cancel" is the wrong word. Use "so FLOP counts and peak FLOP/s are directly comparable".
5. **minor: coverage note.** Ch. 4 Q2 (FLOPs duplicated over an unused mesh axis Z) is listed under lesson 100 in the plan, but neither lesson has it. It is already covered by `sb.sharded-matmul` q10 ({4, 8, 4}, $4\times$), so that's acceptable. The writer's report should say so.

### Strengths
A clean build: dot product, then mat-vec, then matmul, then the einsum rule, then the backward pass, then 6ND, then wall-clock time. Every number recomputes (8.4e22, 6.6 h and its three slips, 683, 3.3e24, 26%). The 6ND scope list (attention, remat, MoE, unembedding) is exactly what interviewers probe.

---

## sb.transformer-accounting

**Verdict: pass** (minor polish only)

### Findings
1. **minor: q1's distractor 34.9B is unexplained.** It is exactly 2× the answer, so the explanation should say what slip it represents, e.g. "counting each $D\times F$ matrix twice (forward and transposed)". Or replace it with 16.6B, the book's rounded-input arithmetic, and explain that the gap comes from rounding $D$ to 4000.
2. **minor: explainer[0] uses $\sigma$ without defining it.** Add "($\sigma$ = an activation such as SiLU/Swish)".
3. **minor: flops-breakdown caption, "attention scores take over by 128k".** That is true, but the lesson's own crossover for these shapes is ≈ 52k. Say "overtake everything else near 52k (non-causal)" so the figure and explainer[5] tell one story. The percentages in the figure recompute (76/51/23%, 7/39/72%).
4. **minor: softmax probabilities are called $P$ here but $S$ in lesson 120.** In lesson 120, $S$ is also the KV length. Pick one name across the series and say so in lesson 120.

### Strengths
Handles the book's slip honestly: the 16B vs 17.4B gap and the $12BTDV$ vocab entry are both flagged. Goes past the book with LLaMA-3 70B's real shapes: the GQA saving of 9.4B, the 52k and 43k crossovers, and the unembedding at ≈ 1.2 layers. All of these recompute. The T/8D derivation is shown step by step, with its scope stated.

---

## sb.transformer-memory

**Verdict: revise** (light)

### Findings
1. **major: the 480-token prefill threshold contradicts the book's ch. 4 Q4 answer ("S = 240"), and the lesson doesn't say so.** q9 even uses 240 as a distractor. A reader who checks the cited ch. 4 Q4 will find the book's answer marked wrong with no explanation. 480 is right by the book's own formula ($T/2 > 240$), and it is the figure ch. 7 gives. Fix: in explainer[5] add "(ch. 4 Q4's answer says 240; that is the large-$G$ limit $TG/(G+1)\to T$. Ch. 7 gives 480, which follows from $T/2 > 240$ for MHA.)". In q9's explanation, say that 240 is the figure the book quotes, and why it applies only as $G\to\infty$. Per the plan's convention, discrepancies like this should be flagged.
2. **minor: explainer[2], "the 70B-scale model's parameters plus Adam state are well under 1 TB".** The example model ($L = 64$, $D = 8192$) has ≈ 69B params. At 10 bytes/param that is ≈ 0.69 TB, and with fp32 master weights (16 B/param) it is ≈ 1.1 TB. Replace with "≈ 0.7 TB at the book's 10 bytes/param, two orders of magnitude below 84 TB".
3. **minor: the remat-tradeoff figure.** In the right panel the "+ small recompute" label sits where a row label would be for the **block remat** row, so it reads as "block remat: + small recompute". It belongs to the "big matmuls only" bar (6 + $4BT^2NH$). Move it next to that bar's value, e.g. "6 (+ attn recompute)".
4. **minor: explainer[6] (Summary) repeats lesson 100's "Adding it up" table almost line for line.** The plan asks for the summary table, so keep it, but cut the parameter and FLOP rows down to one line ("as in lesson 100, plus MoE: ×E params, ×k FLOPs"). The space is better used on the memory rows, which are new.
5. **minor: q7's correct choice, "About 5.4 GB (5 GiB)", is the only choice with two units**, which makes it stand out. Put both units on every choice, or drop the parenthetical.
6. **minor: q3's distractor "About 480 tokens" is unexplained.** Explain it, e.g. "240·k: multiplies by k instead of dividing", or swap it for 120 (= 240/2, int8 thinking without $E/k$).

### Strengths
The MoE roofline is derived, not quoted, with its scope stated (all experts touched, balanced routing). The $f = e^g$ motivation for saved activations is made concrete. The decode-intensity card explains *why* batch cancels (each sequence has its own cache), which is the interview point. KV numbers (160 KiB/token, 5 GiB at 32k, the 13k/105k crossings of 16 GiB) all recompute.

---

## sb.flash-attention

**Verdict: revise**

### Findings
1. **blocking: q1's explanation of the distractor "About 3.77" is arithmetically wrong.** "Rescales chunk 2 by $e^{+1}$ instead of chunk 1 by $e^{-1}$" gives $L^1 + e\cdot L^2$ = 1.368 + 2.854 = **4.22**, not 3.77. No natural slip gives 3.77 ($e\cdot L^1 + L^2$ = 4.77). Fix: change the distractor to "About 4.22" and keep the explanation.
2. **major: lessons 110 and 120 give contradictory regime verdicts for prefill attention, and neither reconciles them.** Lesson 110 (q9, explainer[5]) says MHA prefill attention is compute-bound above ≈ 480 tokens. Lesson 120 (explainer[4], q6) says a kernel with $b_q = 128$ is bandwidth-bound on K/V reads *at any $T$*. Both are right under different assumptions. The book's $TG/(G+1)$ (ch. 4 Q4, ch. 7) assumes K and V are read from HBM **once in total**, which is equivalent to $b_q = T$. The blocked kernel re-reads K/V once per query block, so its intensity is capped at $b_q$ (or $b_q G$ when the $G$ heads sharing a KV head go in one block). Add two sentences to explainer[4]: "Lesson 110's $T/2$ assumes K/V are read once, as if the whole query sequence fitted on-chip. A real kernel re-reads them $T/b_q$ times, so its intensity is $\min(T/2, b_q)$ for MHA. Grouping a KV head's $G$ query heads into one block multiplies this by $G$." Without this, a careful reader will think one of the two lessons is wrong.
3. **major: explainer[5] asserts the softmax backward step.** The line "Through the softmax, the gradient of score $s_{ij}$ is $ds_{ij} = S_{ij}(dS_{ij} - \sum_l S_{il}dS_{il})$" is the only non-obvious step in the card, and it isn't derived. The plan marks this identity "(derivation)". Add the softmax Jacobian in one line, $\partial S_{il}/\partial s_{ij} = S_{il}(\delta_{lj} - S_{ij})$, so that $ds_{ij} = \sum_l dS_{il}\,S_{il}(\delta_{lj} - S_{ij})$, which gives the stated form.
4. **minor: notation clash.** $S$ is the KV length in explainer[0] ($[B, T, S, N]$) and the softmax weight matrix $S_{ij}$ in explainer[5]. The book has the same clash. Say once, "here $S_{ij}$ is the softmax matrix, not the sequence length", or rename it $P_{ij}$ to match lesson 100.
5. **minor: the tiling figure.** (a) The side list says "output acc. O", but the text calls the accumulator $A$ and reserves $O$ for the final output. Change it to "unnormalized output A". (b) At phone size the skipped (upper-triangle) cells are barely darker than the computed ones. Give the skipped cells a hatch or noticeably more contrast.
6. **minor: duplicates.** q4 reuses explainer[0]'s exact 137 GB example. q1 is a strict sub-step of q2, on the same numbers as the explainer worked example. Change q4 to another shape (e.g. $T = 64$k, $N = 32$ gives 275 GB), and q1 to different scores (e.g. chunk maxima 0 and 2).
7. **minor: explainer[0], "Q- and output-sized buffers are only $2TNH \approx 0.5$ GB".** $2TNH$ is *one* bf16 buffer. Say "each ≈ 0.5 GB".
8. **minor: memory-vs-T caption, "passes a chip's HBM near 32k".** The curve crosses 89 GiB at ≈ 28k. Say "just below 32k".

### Strengths
The online-softmax card has a worked numerical merge that recomputes exactly (26.21), followed by the associativity point. The overflow motivation is concrete ($e^{100}$ vs fp32's $e^{88.7}$). Ring attention, sequence-sharded KV and split-K are tied back to the same merge, which is the right "connections" card. The intensity estimate is clearly labelled as the writer's own.

---

## Batch 2 fix check

I checked the blocking items from `content-sb-batch2.md`, and the majors I could verify quickly, against the current files.

| Lesson | Item | Status |
|---|---|---|
| collective-costs | **blocking** q6 had two defensible answers | **Resolved.** The stem now pins the baseline ("using the lesson's ring cost $T = V/W = V/(2W_1)$"), and the "2×" distractor became "3×". Only 1.5× is correct. Residual minor: the explanation still has a parenthetical about the tighter-bound 2×. It is accurate and now framed as an aside, so it can stay. explainer[2] now carries the small-$N$ caveat ("sits slightly above the bound… the book's $V/W$ is this algorithm's time"). |
| collective-costs | **major** q12 figure leak | **Resolved.** The figure-prompt q12 was removed (q13/q14 replace it). The figure caption now says "Modelled (not measured)". |
| alltoall-overlap | **blocking** Q7 labelled compute-bound | **Resolved.** explainer[7] now says "**HBM-bound** (≈ 330 µs), not compute-bound … intensity ≈ 128, below the v5e ridge of 240". q12's explanation and f9 say the same. No "compute-bound" or "comfortably" label remains. |
| alltoall-overlap | **major** Q5/Q6 coverage, 17.5 vs 11.6 µs | **Resolved.** There is a new worked card for Q5 and Q6, plus q5. The one-link-per-length-2-axis explanation for 17.5 µs is in explainer[7] and the Key results. |
| alltoall-overlap | minor: the "3.2" large-N ratio | **Resolved.** The routing assumption is stated (antipodal piece split, giving exactly ¼). |

No regressions spotted in the lines I read.
