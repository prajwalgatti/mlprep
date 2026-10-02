# Review: Scaling Book special edition, batch 1

Files: `sb.roofline-basics`, `sb.roofline-matmul`, `sb.tpu-chip`, `sb.tpu-networking`, their figure modules and the
previews (workbench for every figure, plus terminal or editorial). I checked the files against `sb_00`, `sb_01`, `sb_02`
(and `sb_03`, `sb_07`, `sb_12` where a lesson cites them), and recomputed every number in Python
(scratchpad `check_sb1.py`). `validate.py` reports 0 errors and no warnings for these four files.

## The writer's deliberate deviations, checked one by one

| Deviation | Writer right? | Explained to the reader? |
|---|---|---|
| **Mixed-precision $B_\text{crit} = (C_\text{compute}/W)\cdot b_w/2$** (matmul, explainer[4], f4, key results) | **Yes.** From $I \approx 2BDF/(b_w DF) = 2B/b_w$, you get $B > (C/W)\,b_w/2$, which reproduces 120, 240 and 480. Ch. 7 writes $B_\text{crit} = \beta\,\alpha_\text{hbm}$ with $\beta$ = bits per param / bits per activation. That form is only right if $\alpha$ is the **bf16** ridge and the math runs at the activation precision with peak ∝ 1/bits. If you plug in the int8 ridge (480) for int8-everything, you get 480, not 240. | **No.** The card states the rule as the lesson's own, with no derivation line and no mention that ch. 7 writes it differently. The reader meets $\beta\alpha_\text{hbm}$ again in lesson 210 and won't know the two agree. See matmul finding 2. |
| **Ch. 2 Q4: B ≈ 260, not 267** (tpu-chip explainer[4], q5, figure) | **Yes.** The exact answer is 259.3. Rounding $2DF$ to $1.3\times10^8$ alone gives 268.4, and also rounding $20480 \to 2\times10^4$ gives the book's 267.4. The VMEM case is 11.0 either way. | **Yes**, clearly, in explainer[4] ("after rounding $2DF$ down to $1.3\times10^8$"). The q5 explanation's "the book rounds to 267" is looser; say "the book's rounded arithmetic gives 267". |
| **Ch. 2 Q6: 179 ms, not 167 ms (GiB vs GB)** (networking explainer[6], q8, f9, figure) | **Yes.** $(2^{17})^2 = 2^{34} = 1.718\times10^{10}$ B = 16 GiB. ICI: $\tfrac{15}{16}\cdot1.718\times10^{10}/9\times10^{10} = 179$ ms. The book used 15e9/9e10 = 167 ms. PCIe 67 ms (book 63), HBM 21 ms (book 20) and FLOPs 1.40 ms all check. | **Mostly.** The Size line gives 16 GiB ≈ 1.72e10, and the closing note says "the book uses 16×10^9 bytes and gets ≈ 167 ms". Be precise: the book treats 16 GiB as 16e9 B, so 15e9 B is received. |
| **Ch. 2 Q5: 192 µs, not 194 µs** | **Yes.** 16,777,216 B / 9e10 = 186.4 µs, plus 6 µs. The book rounded the bytes to 1.7e7. | **Yes**, explicitly. |
| **"B > 240" scoped to v5e** | **Yes.** The book's takeaway says "most TPUs", but its own spec table gives v5p 164, v6e 575, v4p 229 and TPU7x 311. | **Adequately.** basics explainer[4] and q9 ("the familiar 240 is specific to the v5e"). Adding one clause in matmul explainer[1], "the book says 'most TPUs', but…", would make the deviation explicit. |

I found one more silent deviation, and it is harmless. Basics explainer[1] uses the spec-table v6e peak (9.2e14 → 1.09 ms), while ch. 1's prose says 9.1e14 → 1.1 ms. No change needed.

---

## sb.roofline-basics

**Verdict: revise** (one blocking MCQ; otherwise strong)

### Findings
1. **blocking — q10 (and the roofline figure's dot-product marker).** The "correct" answer says a dot product from VMEM "rises about 22×, to roughly 9 TFLOP/s". But explainer[5] teaches that a dot product runs on the **VPU**, whose peak is ≈ 7e12 FLOP/s (v5p core, 1.75 GHz). With the same 8×128×4 VPU at the v5e's 1.5 GHz, the peak is ≈ 6.1e12. At $0.5 \times 1.8\times10^{13} = 9\times10^{12}$ the op would exceed the VPU roof. So it becomes **compute-bound on the VPU** at ≈ 6–7 TFLOP/s, which makes the distractor "rises about 22× and becomes compute-bound" arguably correct. The question contradicts the lesson's own dot-product card. Fix options:
   - (a) Make it a generic MXU-bound algorithm with $I = 0.5$, not a dot product.
   - (b) Make the correct answer "it becomes limited by the VPU's ≈ 6–7 TFLOP/s peak: compute-bound on the VPU, yet ≈ 3% of MXU peak", and rewrite the distractors.

   Also add a line to the figure caption: the dot-product marker is placed on the MXU roofline only for illustration.
2. **major — q12 giveaway.** The correct choice is 117 characters against 48–59 for the distractors. Shorten it, e.g. "Per-chip compute shrinks faster than per-chip communication". Or lengthen the distractors, e.g. "Each chip's achievable FLOPs/s falls as more chips share the same host".
3. **minor — q11 wording.** "Comms-bound, with communication mostly overlapped" has it backwards. In a comms-bound op it is the *math* that hides under the communication. Use "Comms-bound; the math is mostly hidden under communication, ≈ 1 ms exposed". The correct choice is also the longest (77 vs 34–43 characters), so trim it.
4. **minor — teach-from-scratch.** "2:4 structured sparsity" (explainer[1]) is never explained: add "two of every four weights zero". VMEM first appears in the roofline figure caption and explainer[6] ("lesson 3") without a one-line definition: add "VMEM, a small on-chip scratchpad" at first use.
5. **minor — roofline figure.** In the terminal theme the "I = 1000" label collides with the "ridge ≈ 240" leader line. Move the label above the roof or shift the leader's xytext. The "HBM bandwidth" legend text ends close to the VMEM dashed line; it is legible, but the margin is tight.

### Strengths
The derivations are clean: the factor-2 bound, the intensity rearrangement, and $P(I) = \min(C, IW)$ from the two cases. The plan's question list is covered item for item, with good trap distractors (bandwidth units in q1, sparsity in q4, the write-back in q6).

---

## sb.roofline-matmul

**Verdict: revise** (light; no wrong numbers)

### Findings
1. **major (fidelity) — explainer[3], q9, q11, f5 and the figure, all cited as "Question 3".** The book's Q3 is explicitly "the setup from Question 2", i.e. **int8 weights + bf16 activations**: its code uses `D*F`, not `2*D*F`. The book's answers would be **≈ 136** (D = F = 4096) and **≈ 226** (D = F = 1024), and it says "almost doubles". The lesson's 272 and 453 are the bf16-weight versions. They are correct, but they are attributed to a problem that doesn't pose them. Fix: add one sentence to explainer[3]: "The book poses this for int8 weights (Q2's setup): ≈ 136 and ≈ 226. With bf16 weights every crossover exactly doubles, to 272 and 453." Then cite q9/q11 as "adapted from Question 3". The figure already has the int8-W 4096 curve (136); consider adding int8-W 1024 (226) so the book's plot is reproduced.
2. **major — explainer[4] "General rule".** The rule is right (see the deviation table), but:
   - (a) Derive it in one line: "bytes ≈ $b_w DF$, so $I \approx 2B/b_w$; compute-bound iff $2B/b_w > C/W$".
   - (b) Tell the reader that ch. 7 writes it as $\beta\,\alpha_\text{hbm}$ with $\beta$ = bits per param / bits per activation. That form assumes $\alpha$ is the bf16 ridge and that peak doubles when activation bits halve. Writing the compute-precision peak explicitly avoids that hidden assumption, and for H100 fp8 (q5) the two forms agree.
3. **minor — explainer[3] last line.** "Within about 10–15%" should be "about 5–15%": 255 vs 240 is 6% at D = F = 8192, and 272 vs 240 is 13% at 4096.
4. **minor — explainer[6] (tiles).** A reader can confuse the 128×128 MXU block from the next lesson with the tile size here: "128 tiles → only 64 FLOPs/byte" sounds like TPUs can't exceed 27% of peak. Add: "these are VMEM tiles, usually much larger than the 128×128 MXU block".
5. **minor — two-chip figure.** Both arrowheads land inside chip 1's half: one under Y1/Z1, the other under X1. Neither points clearly back to Z0, so the "swap" is visually ambiguous. Put the heads under Z1 (≈ x 8.75) and under Z0 (≈ x 4.15).
6. **minor — q2 giveaway.** The correct choice is 102 characters against 61–73. Trim it to "Weight bytes dominate and each weight is reused for all B tokens".
7. **minor — batch-roofline figure.** The "saturates at B ≈ 453" label nearly touches the right edge. Shift it left about 20 px.

### Strengths
The worked example at B = 512 vs 256 shows the batch "doing the deciding". The ceiling $DF/(D+F)$ is a nice addition beyond the book. The per-example-weights → decode-attention link and the "depends on D, not B" explanation are exactly what an interviewer probes.

---

## sb.tpu-chip

**Verdict: revise**

### Findings
1. **major (correctness) — explainer[3], last paragraph.** "a systolic array that does about 200 trillion multiply-adds per second". explainer[1] has just taught that 1 MAC = 2 FLOPs and that a v5e does 1.97e14 **FLOPs**/s, which is ≈ 1e14 multiply-adds/s. The paragraph imports the book's slip and contradicts the lesson's own convention. It is also a near-verbatim copy of the book's takeaway box, against the brief's "own words" rule, and it repeats the card's content. **Cut the paragraph**, or replace it with one sentence in your own words using "≈ 2e14 FLOPs/s".
2. **major — systolic figure.** It shows only boundary arrows: x in at the left, y out at the bottom. It doesn't draw what makes the array *systolic*: no right-pointing arrows between cells for activations, no down-pointing arrows between cells for partial sums, and no skew (the label says "skewed", but all inputs are aligned). Add small →/↓ arrows between neighbouring cells, and stagger the $x_{\cdot r}$ inputs by one cell per row (or draw a "t = 0, 1, 2…" wavefront).
3. **major — q13 is a giveaway.** The figure in the prompt carries the annotation "pass x right and sum down", which answers "what moves down". Use a version of the figure without the side annotation for the quiz (e.g. a `systolic-bare` variant), or ask something the figure doesn't state, e.g. "after how many cycles does $y_{\cdot 3}$ for the first activation row emerge?" (answer: skew plus depth ≈ 2n − 1).
4. **major — too many recall questions.** q10 (scalar-core consequence), q11 (what megacore is), q12 (why systolic arrays are efficient) and q8 are definition or recall. The brief allows at most 2 pure-definition questions. Several also have giveaway lengths: q10 99 vs ≤ 79 characters, q11 90 vs ≤ 72, q12 119 vs ≤ 77, q14 90 vs ≤ 63. Convert two of them:
   - q11 → "A v5p chip runs a matmul as one megacore device. What per-core peak and HBM share does each TensorCore effectively get, and what is the chip's ridge?" (≈ 2.3e14 per core, shared 2.8e12 B/s, ridge 164)
   - q12 → predict-the-behaviour, e.g. "A $[B, 4096]\times[4096, 96]$ matmul on 128×128 MXUs: what fraction of MXU columns does it use?" (96/128 = 75%)
   - Then even out the choice lengths in q10 and q14.
5. **minor — explainer[3] and f11, unsourced claims.** "No register-file reads per operation" (explainer[1]) and "without cache misses or eviction surprises" (f11) are reasonable, but neither is in the book. Either cite Jouppi 2017, which is already in the reading list, or soften the wording.
6. **minor — teach-from-scratch.** Expand DMA (direct memory access) and SIMD at first use. Define XLU as "cross-lane unit" (explainer[7]).
7. **minor — vmem-vs-hbm figure.** The "(int8 MXU)" label crosses the compute line near B = 1–2. Move it right or down.
8. **minor — redundancy across lessons.** The VMEM ridge ≈ 11 is tested in basics q10/f11 and again in chip q4/f4. That's acceptable for spaced repetition, but chip q4 could instead ask the VMEM crossover for a *different* matmul to add something new.

### Strengths
The chip internals are accurate: 32768 FLOPs/cycle, 4 MXUs → 1.97e14, the VPU at 7.2e12 (≈ 32× below the MXUs), and all six spec-table ridges are correct. The writer's correction that megacore applies to v4/v5p only (the book's "v4, v5, v6" is wrong for v5e/v6e) is right. The 70B-int8-on-8-v5e variant, including the bf16 doesn't-fit check, is a good original problem.

---

## sb.tpu-networking

**Verdict: revise** (one blocking MCQ)

### Findings
1. **blocking — q7, ambiguous.** The book's footnote defines bidirectional bandwidth first as "the total bytes that can be sent along a single link in both directions". That is exactly the distractor "the bandwidth available when two chips send to each other at the same moment, on any slice": a link carries both directions on any slice. Replace the distractor with something unambiguously wrong, e.g. "The one-way bandwidth of two parallel links between the same pair of chips", or "The rate one link reaches when both chips' PCIe links also feed it".
2. **major — f9 (open) model answer.** It plans only the gather onto (0,0). An excellent candidate would also point out two things:
   - The 16 GiB matrix **cannot be resident** on chip (0,0): a v5e has exactly 16 GiB of HBM. So the pipelined "slowest stage" model isn't optional; chunks must be consumed as they arrive.
   - If the task allows it, **move the compute to the data**. Each chip multiplies its own shard after its PCIe load (67 ms, plus 1.3 ms of HBM reads), then only small partial outputs ($[8, 2^{17}]$, ≈ 2–4 MB per chip) cross ICI. The total is ≈ 70 ms, PCIe-bound, instead of ≈ 180 ms.

   Add both points to f9, and the HBM-capacity caveat to explainer[6].
3. **major — q4 and q12 overlap, and the set is recall-heavy.** Both ask which v5e/v5p slice axes wrap: q12 is the 8×16 case that q4's explanation already gives away. q6, q9, q10 and q13 are also recall. Replace q12 with a compute question, e.g. "Q5 on a full 16×16 v5e torus from (0,0) to (8,8): hops and total time". The distance is 8 + 8 = 16 hops either way round, so the latency is 16 µs. But the source and destination now have 4 usable links each, so the bandwidth term halves to ≈ 93 µs. Or try a DCN compute: "AllReduce-sized 1 GB per chip between two v5e pods at 3.125e9 B/s/chip". q10's correct choice is also the longest (94 vs ≤ 69 characters).
4. **minor — Key results, PCIe line.** "Weights streamed over PCIe (v6e)… B > 57,500" uses 1.6e10, but the same card lists v6e PCIe as 3.2e10. Add "(at the book's assumed 1.6e10 B/s; ≈ 28,750 at 3.2e10)", as explainer[0] already does.
5. **minor — explainer[6], option 1.** Say why only one PCIe link is used: the whole matrix must enter chip (0,0) from host 0. The book hints at the alternative of "the 8 on host 0"; one clause would cover it.
6. **minor — redundancy.** "Slice" is defined in explainer[2] ("A job gets a slice") and again in explainer[4] ("A set of ICI-connected chips is a slice"). Keep one.
7. **minor — torus-hops figure.** With "hop 1"/"hop 2" printed at both ends of each wrap stub, the top-right corner is crowded at phone size. Label each wrap once, at the midpoint convention ("hop 1 ↔", "hop 2 ↕"), or draw the wrap as a curved arc on the torus side.

### Strengths
Full coverage of the plan: trays and hosts, PCIe Q3, tori and wraparound rules, pods, one-way vs bidirectional, DCN path, TPU vs GPU, and Q2/Q5/Q6. Every number recomputes. The bandwidth-ladder and challenge-stage figures are clear and encode the lesson's main point. The deviations from the book are flagged in the text.
