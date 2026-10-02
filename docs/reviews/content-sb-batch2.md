# Review: Scaling Book special edition, batch 2

Files reviewed: `sb.sharding-notation`, `sb.sharded-matmul`, `sb.collective-costs` and `sb.alltoall-overlap`, plus their
figure modules and previews. I looked at the workbench theme for every figure, and at terminal or editorial for one
figure per topic. Everything was checked against `sb_03_sharding.md`, with `sb_02`, `sb_04`, `sb_05` and `sb_10`
where a lesson cites them. I recomputed every number in Python. `validate.py` reports 0 errors and no warnings for
these files. Card lengths are all under 300 words, and no correct MCQ choice is the longest more than 3 times in a
topic.

## The writer's three flags, checked independently

| Flag | Verdict | Notes |
|---|---|---|
| **Q7 (alltoall-overlap) is the writer's own derivation** | **Arithmetic right, but one wrong conclusion.** | All of these recompute: each weight is $2DF$ = 536.9 MB, the TP weights come to 268.4 MB per chip, the activations are 2.10 MB, and FLOPs per chip are $4BDF/4 = 3.44\times10^{10}$, which is 174 µs. The FSDP weight gather is $0.75\cdot1.07$ GB over $2\times4.5\times10^{10}$, about 8.9 ms. The 17.5 µs per collective (3/4 of 2.1 MB received over 2 inbound links of $W_1$) is the right lower bound for a 2×2 slice without wraparound. It is *more* accurate than the lesson's own multi-axis formula $V/(W N_\text{axes})$, which gives 11.6 µs here. That formula assumes 2 links per chip per axis, but an axis of length 2 has only one neighbour. The lesson never says why it departs from its own formula (see alltoall finding 2). **Wrong:** the lesson calls the layer "compute-bound". At $B = 128$ the matmul intensity is 128, below the v5e ridge of 240 (lesson 20). Reading 268 MB of weights from HBM takes ≈ 327 µs, more than the 174 µs of FLOPs. So the layer is HBM-bound. It is *not comms-bound*, but it is not compute-bound either (alltoall finding 1). |
| **The 87/224 µs split in the collective-matmul figure is inferred** | **Plausible, and labelled in the text.** | Ch. 10 gives 311 µs for jit with a blocking AllGather, 244 µs for the collective matmul, and 224 µs for the "matmul with no sharding on the contracting dimension", which is the same local matmul. So 311 − 224 = 87 µs assumes jit does no overlap and no other work, and the profile description ("big blocking AllGather at the beginning") supports that. Independent check: the book's arrays come from `jnp.arange`, so they are int32. $A_X$ = [512, 2048] int32 = 4.19 MB. It is gathered over Y = 4 on a v5e-8 (2×4, so no wraparound), and the line bound is $0.75\cdot4.19\text{ MB}/4.5\times10^{10}$ = 70 µs. 87 µs is ≈ 80% of that, the same efficiency as the book's pop quiz 2 (560 → 680 µs). The int32 dtype also explains why the matmul takes 224 µs rather than the ≈ 22 µs of the bf16 roofline. The explainer says the split is inferred, but the figure caption and q8 do not (minor). |
| **The torus AllToAll formula rests on the book's rough bisection argument** | **Correct.** | The bisection argument is written correctly (it matches the book): $V/4$ crosses $2N/\max$ links at $W_1$ each, giving $V\max/(4NW)$. In 1D it reproduces $V/(4W)$. I also derived it a second way, by counting link loads. On an $A\times B$ torus, a byte travels on average $A/4$ hops along $A$, so the $A$-axis links carry $V\cdot A/4$ byte-hops spread over $2N$ link-directions. That gives $VA/(8NW_1) = VA/(4NW)$, and the $B$ links carry less. So the bisection bound is achieved, and the max-axis formula and the 16× example for 16×16 (vs a 256-ring) both hold. One small addition: say that bisection gives a *lower bound* and that link counting shows it is achieved, and that it needs wraparound on every axis (v5e slices under 16 have none). |

---

## sb.sharding-notation

**Verdict: revise** (light; no wrong numbers)

### Findings
1. **major: duplicated questions (q2, q11, q12).** q2 and q11 test the same fact on the same configuration ($A[I_X, J]$ on $\{X{:}4, Y{:}8\}$ stores $|Y| = 8$ copies). q5 tests the replication factor a third time. q12 repeats explainer worked example 1 word for word (fp32[1024, 4096], $I_{XY}$, {8, 2}, H100, 0.3 µs). Fix:
   - Rewrite q11 as a which-is-false question about something else, e.g. "$A[I_{XY}, J]$ and $A[I_{YX}, J]$ have different local shapes" or a JAX `P(('X','Y'))` statement.
   - Turn q12 into a new compute, e.g. the int8[128, 2048] pop quiz with the shard read time on a v5e (16 KiB / 8.2e11 ≈ 20 ns: "latency dominates"), or a 3D mesh with $A[I_{XZ}, J_Y]$.
2. **minor: explainer[4] JAX snippet.** It drops `dtype=jnp.bfloat16`, so `jnp.zeros` would create fp32 arrays. The byte counts the lesson implies elsewhere are bf16. Add the dtype back, or say "fp32 by default".
3. **minor: f9 and q8 overlap with explainer[6]'s "hidden cost" paragraph.** This is fine for spaced repetition, but q8 and q7 could use different mesh sizes so they don't both read straight off the $\{4, 2\}$ example.
4. **minor: sharding-patterns figure.** The $A[I, J]$ panel uses 16 thin stripes per row. It is legible, but at phone size the stripes read as noise. Two wider stripes per cell (or a hatched cell with the label "all 4") would read better. Otherwise it is correct; the device colours match the legend.

### Strengths
Clear progression from the two kinds of shape, to the notation, to memory accounting, to block matmul, to Case 1. The "divide by the axes you use, multiply by the axes you don't" rule is a good mnemonic. The book's pop quizzes and Q1 all recompute (1 MiB, 0.31 µs, 16 KiB, 512 KiB, 16×). The duplicated-FLOPs point in Case 1 goes beyond the book and is useful.

---

## sb.sharded-matmul

**Verdict: revise** (light)

### Findings
1. **major: notation clash in explainer[5], q6, q7, f6 and f9.** The book's Q4 writes $X[B, D]\cdot Y[D_X, F]$, in which $X$ and $Y$ are **matrices**, and $Y$ is sharded over the mesh axis $X$. Everywhere else in this series, $X$ and $Y$ are mesh axes. A newcomer reads "$Y[D_X, F]$" as "something on mesh axis Y". The lesson also writes $|X|$ for the axis size in the same equations. Rename to $\text{In}[B, D]\cdot W[D_X, F]$ (as ch. 5 does), and say once that the book's Q4 uses X/Y as matrix names.
2. **minor: explainer[6], "Case 3's other variant (book Q9)".** Q9 is Case 2's input sharding ($A[I, J_X]$ with $B$ replicated) solved the Case 3 way, by slicing $B$. The label is muddled, and it repeats the S1/S2 comparison from the previous card. Move it into explainer[5] as "the ReduceScatter version of S2: compare $NM$ with $NK$". That also keeps the FLOPs card focused.
3. **minor: q8 and q11 use the same numbers (4096, 8192, 1024 → 8×) as the explainer[1] worked example.** Three items test the 8× ratio. Change q11 to, e.g., $N = 2048$, $M = 4096$, $K = 16384$, where the AllGather wins by 4×. The case where gathering the input is cheaper is not tested anywhere.
4. **minor: q10 repeats the explainer[6] worked example exactly** ({4, 8, 4}, $4\times 2BDF$). Use a different mesh, e.g. {X:2, Y:4, Z:8} with $A[B_X, D_Y]$ → $8\times$.
5. **minor: four-cases figure.** The "No" branch asks whether two non-contracted dims share a mesh axis only when $J$ is unsharded. Case 4 can also occur together with a sharded $J$. Change the footnote to "Case 4 can co-occur with 2 or 3: resolve it first".

### Strengths
The outer-product worked example (device 0: [[5, 6], [15, 18]], device 1: [[14, 16], [28, 32]]) is exactly the concrete-first teaching the brief asks for. q5 tests it well. The S1/S2 numbers recompute: 1.20, 2.98, 0.30 and 0.75 ms, and 2.40, 0.75, 0.60 and 2.98 ms. The q7 counter-case, where gathering wins at $B > 2550$ and $D < 5100$, is a good addition. The case4-diagonal and partial-sums figures encode the mechanism directly.

---

## sb.collective-costs

**Verdict: revise**

### Findings
1. **blocking: q6 has an arguably correct distractor ("About 2× slower"), and explainer[1], explainer[2] and explainer[6] mix two ring models.** The lesson's "accounting" is the bound that each chip must receive $(N-1)/N$ of $V$ through its inbound links. For a 4-ring that bound is $0.375\,V/W_1$, and it is achievable by splitting the antipodal shard. The book's simple bidirectional algorithm takes $V/(2W_1) = 0.5\,V/W_1$, because it sends the antipodal shard both ways. explainer[2] says "the accounting agrees", but that is only true for large $N$. The line case uses the tight bound ($0.75\,V/W_1$), while the ring case uses the algorithm's time, so "1.5× for $N = 4$" compares a bound against an algorithm. Apply the lesson's own accounting to both and the line is 2× slower. Ch. 2 also says a missing wraparound "typically doubles" the time. q6's explanation then quotes both ring numbers ("$0.75V/(2W_1)$; ideally … $V/(2W_1)$"), which is confusing. Fix:
   - In q6's stem, say "using the lesson's ring cost $T = V/W$", or replace the "2×" distractor with "About 3× slower".
   - Rewrite q6's explanation with a single ring time.
   - In explainer[2], add: "for small $N$ the simple bidirectional ring sends the antipodal shard both ways, so it sits slightly above the accounting bound; the book's $V/W$ is this algorithm's time".
2. **major: q12 figure leak.** The latency-vs-bandwidth figure in the prompt labels "N = 16: 8 µs floor" and marks the 720 kB crossover, so the answer (100 kB over 16 chips ≈ 8 µs) can be read straight off the annotation. Use an unannotated variant (`latency-vs-bandwidth-bare`) in the prompt. Or ask about something the figure doesn't label: $N = 64$ (32 µs floor; crossover 2.9 MB), or "at what array size does the $N = 8$ curve leave its floor?" (360 kB).
3. **minor: explainer[5], "The additions run on the VPU and are usually hidden."** The book says only that the reduction time is excluded. Cite a source or soften to "the additions are cheap next to the transfers".
4. **minor: q1 distractor "About 0.35 ms".** It divides by 16, but the stem gives no axis length, so nothing makes it tempting. Replace it with "About 8.3 ms (assuming a line, (N−1)/N·V/W₁ with N = 4, W₁ = 9e10)", which tests the wraparound assumption.
5. **minor: Q2's part-2 fun fact.** The book also cuts $\text{AllGather}_{XY}$ from 46 µs to 31 µs by gathering over XYZ ($V/(3W)$). One line in explainer[4] would complete it.
6. **minor: latency-vs-bandwidth caption.** Say it is the model curve ("modelled, not measured"), so it isn't mistaken for the book's empirical v5e plot.

### Strengths
A real derivation of why $N$ cancels, checked two ways (hop counting and per-chip inbound bytes). The latency formula and the 45 kB crossover are derived, not stated. The pop quiz and Q2/Q3 all recompute: 373, 559 and 680 µs (82%), 23.3, 46.6 and 11.65 µs, and 2 µs. The decode example (bf16[16, 4096] over 8 chips ≈ 4 µs, against 1.46 µs from the bandwidth term) ties the latency regime to inference concretely. The ring-allgather figure shows who holds what at each step.

---

## sb.alltoall-overlap

**Verdict: revise**

### Findings
1. **blocking: Q7 is labelled "compute-bound" in explainer[7], q12's explanation and f9.** At $B = 128$ tokens the per-chip matmuls have intensity ≈ 128 FLOPs/byte, below the v5e ridge of 240 that lesson 20 teaches. Reading the 268 MB of weight shards from HBM takes $2.68\times10^8/8.2\times10^{11}$ ≈ 327 µs, against 174 µs of FLOPs. The correct reading: "comms (≈ 35 µs) are well hidden; the layer is bound by streaming the weights from HBM (≈ 330 µs), with FLOPs at ≈ 174 µs". q12's correct choice can stay as it is (it compares comms with FLOPs, which is what the book asks). Change "Comfortably compute-bound" to "not comms-bound (and in fact HBM-bound at B = 128 < 240)".
2. **major: plan coverage, Q5 missing and Q6 timing missing.** The plan lists Q5 (minimum-latency matmul with a replicated output on v4p 4×4×4) and asks the writer to derive Q6's comms-vs-compute times on a v5e 4×4. Q5 appears nowhere. Q6 is classified but never timed. Add one worked card or an `open` card:
   - **Q5.** Shard the output dimension $I_{XYZ}$ or $K_{XYZ}$, then AllGather the output: $2IK/(3W)$, against an AllReduce of $2IK$ bytes for the $J_{XYZ}$ option, which costs ≈ 2× more. FLOPs per chip are $2IJK/64$ in options 1–3. The book's answer is "(1) and (2) are equally good".
   - **Q6**, symbolically, on a v5e 4×4 with no wraparound. Case 3 AllReduces $C[I_X, K]$, which is $2|I||K|/|X|$ bytes. The FSDP layout gathers $B$, which is $2|J||K|/|Y|$ bytes. Each time is up to 1.5× the ring formula because the axes are lines of 4.
   - Also explain in Q7 why 17.5 µs, not the 11.6 µs that $V/(WN_\text{axes})$ gives: each axis of length 2 has one neighbour, hence one link.
3. **minor: explainer[1] and f3, "The ¼ is a large-N limit: for N = 8 the ratio is 3.2."** This holds only if the antipodal piece is sent one way. If it is split across both directions, the busiest direction carries exactly $N^2/8$ piece-hops, i.e. $V/8$ for every even $N$. That is the book's footnote count ($N^2/4$ chunks per device over both directions). The ratio against the book's AllGather $V/W$ is then exactly 4 (and 3.5 against the exact $(N-1)/N$ AllGather at $N = 8$). Ch. 3 Q10 uses the unsplit large-$D$ form, so the book is not self-consistent either. Either state the routing assumption, or drop "3.2". Apply the same fix to the chunk-hops figure, or label its curves "antipode sent one way".
4. **minor: q8 and the collective-matmul figure present the inferred 87/224 split as fact.** Add "(split inferred as 311 − 224 µs)" to the stem and to the figure caption, along with "schematic".
5. **minor: q9's explanation contradicts itself.** "Gathering $A$ would also work … and still leaves $B$ split on $J$." If $B$ is still split on $J$, gathering $A$ alone does *not* work. Replace it with: "Gathering $A$ over $Y$ alone would leave $B[J_Y, K]$ sharded on the contraction (Case 2), so you'd still need a second collective."
6. **minor: terms of art.** "Expert parallelism" (explainer[2]) is used without a definition. Add "placing different experts on different devices".
7. **minor: missing figure from the plan.** The plan lists an AG/RS transpose diagram. explainer[3]'s $u\otimes I$ argument would benefit from a small $p = 2$, $n = 2$ matrix picture showing that the AllGather matrix and the ReduceScatter matrix are transposes.
8. **minor: Key results, display line.** The ring/torus AllToAll equation runs past ~45 rendered characters. Split it into two `$$` lines.
9. **minor: q7 duplicates the explainer[4] worked example** (2 GB, 22 vs 11 ms). Change the size, or ask about the choice of axis instead.

### Strengths
The AllToAll is taught three ways (piece-hop sums, average distance, bisection), and all of them check out. The 16× torus example is correct and memorable. The transpose argument includes the $(8, 9) \to 17$ check and the FSDP consequence. The collective-matmul description matches the ch. 10 code. Apart from the regime label, the Q7 arithmetic is careful and correct.

---

## Batch 1 fix check

I checked the items marked blocking or major in `content-sb-batch1.md` against the current files. All are resolved, and I found no regressions. `validate.py` is clean, and no correct choice is the longest more than twice per topic.

| Topic | Item | Status |
|---|---|---|
| roofline-basics | **blocking** q10 (dot product on the MXU roofline) | **Resolved.** q10 removed. The new q14 is a generic MXU kernel at $I = 5$ from VMEM: 90 TFLOP/s, still bandwidth-bound since 5 < 11 (recomputed). The figure caption now says the I = 0.5 marker is illustrative and that real dot products run on the VPU. |
| roofline-basics | major q12 giveaway | **Resolved.** 59 characters vs 48–60. |
| roofline-matmul | major Q3 fidelity (int8 weights) | **Resolved.** explainer[3] gives the book's 136/226 and explains the bf16 doubling (recomputed: 135.9/225.9 → 271.8/452.6). q9/q11 are cited as "adapted". |
| roofline-matmul | major general-rule derivation and the ch. 7 βα form | **Resolved.** Derived in one line ($I \approx 2B/b_w$), with the ch. 7 form and its hidden assumption explained. |
| tpu-chip | major "200 trillion multiply-adds" paragraph | **Resolved.** Replaced with "about 2×10¹⁴ FLOPs/s" in the writer's own words. |
| tpu-chip | major systolic figure | **Resolved.** Inter-cell → and ↓ arrows are drawn, and the inputs are skewed by one cell per row (workbench preview). |
| tpu-chip | major q13 figure giveaway | **Resolved.** q13 removed. The new q18 asks the cycle at which cell (3, 3) fires: 6, correct. |
| tpu-chip | major recall-heavy set and length giveaways | **Resolved.** q11 and q12 were replaced by compute items: q16 megacore (2.3e14, ridge 164) and q17 VPU time share (78%), both correct. q14 is still the longest (58 vs ≤ 52 characters), which is minor. |
| tpu-networking | **blocking** q7 ambiguous distractor | **Resolved.** The distractor is now "one-way rate of two parallel links". The correct choice is 3 characters longer than the longest distractor, which is minor. |
| tpu-networking | major f9 (HBM residency, compute at the data) | **Resolved**, in both f9 and explainer[6] (new "can't be resident" paragraph). The ≈ 70 ms PCIe-bound alternative is included. |
| tpu-networking | major q4/q12 overlap and recall | **Resolved.** q12 was replaced by q14 (16×16 torus, (0,0)→(8,8): 16 µs of latency plus 93 µs of bandwidth ≈ 110 µs, correct) and q15 (DCN 320 ms vs ICI 22 ms, correct). q10 is no longer the longest. The set still leans on recall (q4, q7, q9, q10, q13), but it now has 7 compute items, which is acceptable. |
