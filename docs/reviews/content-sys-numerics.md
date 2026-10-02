# Content review: numerics and memory (sys.fp-formats, sys.mixed-precision, sys.memory-anatomy)

Reviewed: `PrepApp/Content/applied/sys.{fp-formats,mixed-precision,memory-anatomy}.yaml`, their figure modules, all 9 previews
(workbench, plus terminal or editorial for 4 of them), the plan sections in `docs/content-plan-applied.md`, and `docs/reviews/plan-applied-review.md`
(B10, B12, C1, C4, C9, C10).

**Sources checked:** Goldberg 1991 (Formats, Denormalized Numbers, machine-epsilon convention); Micikevicius 2022 l.80–190 (scaling, saturation,
E4M3/E5M2 recommendation, 448, 17→18 and 32 binades, why not 480, rounding mode left to implementations); Micikevicius 2018 l.125–200
(2048 ratio, 5% below $2^{-24}$, 80% loss, "roughly halved", $[2^{-27},2^{-24})$, scales 8–32K, unscale before clipping); Kalamkar 2019
(RNE emulation, GNMT/GAN/recsys, no hyperparameter changes); the PyTorch numerical-accuracy and CUDA-semantics TF32 sections (1.12 default,
`fp32_precision`, the A100 benchmark: 0.11/0.016 s ≈ 7×, 0.0022 vs 3.8e-5), MI200 flushing; torch.amp (op lists, BCE error, GradScaler
defaults, the bf16-pretrained warning, backward dtype); AMP examples (unscale_ twice raises); NVIDIA guide (N=2000, ×2, ×0.5, multiples of 8,
vocab padding); Scaling Book GPUs l.32 (990 vs 66 TFLOP/s); Playbook (parameter formula, 1–2 GB kernels, nanotron fp32 grads, first step);
Korthikanti l.206–216, 445–446 (11sbh+5as²b, 19sbh, <0.01% for 22B); ZeRO l.281–349 (24 GB, 6 GB buffer, 30% fragmentation, K=12, 60 GB
footnote); EleutherAI (1.2×, 8-bit 6 bytes, SGD momentum 8).

**Recomputed in Python/PyTorch 2.1:** all of the following check out. The finfo table, q1 ($4.29\times10^9$, $2^{-9}$), q2 (13/5/6.5/26),
bf16/fp16 swamping, 256+1, 2048+1, token ids 50176/50272, 4097→4096, $10^{-6}$→$1.0133\times10^{-6}$, $10^{-8}$→0, 65519/65520, 257/259,
chop vs RNE for 2/3, 1000.3→1000/1000.5, $\ln 65504=11.09$, the q11 FTZ difference ($3.0\times10^{-5}$, 33288), and the binade counts (18 and 32).
Mixed precision: lost-update thresholds, ±1e-4 and ±2e-4 at $w=1$, bf16 $0.05+3\times10^{-5}$ lost, the $2^{-13}$ and $1.9\times10^{-9}$ half-ulps,
$10^{-8}\cdot2^{16}=6.55\times10^{-4}$, $S\ge6100\to2^{13}$, the 0.1×10⁴ sums (bf16 32, fp16 256), the GradScaler trace, 18/16/20 bytes, 50264.
Memory: $N=6.575\times10^9$, 105/13/79/132 GB, GPT-3 2.87 GB and 275 GB, 6.6B at 104 / 18.25 GB, the ratio 3.65, 2.1 / 4.19 GB logits,
3.7 layers, 13B ($12.75\times10^9$, 204, 28.5, 1.05, 235, 87%, 57 GB), q9 distractors 8.55 / 25.3, and the figure numbers (76/86 GB peaks,
54/82/95%, 0.31/3.3/45 GB, 20 GB states).

`validate.py`: 0 errors, and no warnings for these three files.

---

## Writer's uncertain points (verified independently)

1. **fp8 cast gives NaN in PyTorch 2.1: true.** `torch.tensor(500.).to(torch.float8_e4m3fn)` → NaN. There is a nuance worth one clause. The cast
   rounds first, so 449–464 still give 448 (464 is the tie, and it goes to even). Only values that would round to 480, i.e. above 464, become NaN.
   E5M2 overflows to inf (61440, the tie point, → inf). Rephrase "In our test (PyTorch 2.1)" as "`.to(torch.float8_e4m3fn)` is non-saturating:
   values that round past 448 become NaN (checked in PyTorch 2.1)".
2. **"DDP all-reduces fp32 grads under autocast": true, and I confirmed it empirically.** I ran single-rank gloo DDP with a comm hook that logs
   `bucket.buffer().dtype`, under `torch.autocast("cpu", torch.bfloat16)`: the bucket is `torch.float32`, and `.grad` is fp32. The reasoning in the
   lesson matches the docs ("backward ops run in the type autocast used for the forward"; the cast's backward returns the parameter dtype).
   Add "verified in PyTorch" to the `source:` of mixed-precision explainer[6] and q12.
3. **105 GB next to "7B → 112 GB": acceptable, but add one sentence.** In memory-anatomy explainer[2], after "16N ≈ 105 GB", add: "'7B' figures
   elsewhere use a round $7\times10^9$. Llama-2-7B, with SwiGLU ($d_{ff}=11008$) and untied embeddings, has $6.74\times10^9$ parameters (108 GB)."
   (Recomputed.)
4. **SwiGLU ≈ 18sbh: holds only for fused kernels. Eager PyTorch saves 23.3sbh.** I counted the saved tensors with `saved_tensors_hooks`.
   In eager mode, `w2(silu(w1(x)) * w3(x))` saves $x$, the gate pre-activation, **silu(gate)** (saved by `mul`), the up projection and the product:
   $2+4\cdot\tfrac{16}{3}\approx23.3\,sbh$. GeLU gives exactly 18. See the memory-anatomy finding below.
5. **Omitting ZeRO's "60 GB for GPT-2 1.5B": agree.** Footnote 3 uses a coarse $12\,h\,b\,s\,L$-element rule. That disagrees with Korthikanti:
   for $h=1600, L=48, a=25, s=1024, b=32$ his formula gives ≈ 287 GB with stored scores, or ≈ 86 GB at 34sbh. Quoting it would contradict the
   lesson's own formula. It could appear as a one-line pitfall ("rules of thumb differ by 3×; recount"), but that is optional.
6. **Cards over 300 words: confirmed.** fp-formats ex[3] 328, ex[5] 322, ex[7] 323, ex[6] 308; mixed-precision ex[3] 333 (prose words, display math
   excluded). Trims are listed below.
7. **Overlaps q11/f8 and q13/f10 (mixed-precision): confirmed, and there are more.** q9 replays the explainer[3] trace with the same numbers. Two pairs
   overlap *across* lessons: fp-formats f8 / mixed-precision f8 (bf16 still needs master weights) and fp-formats f10 / mixed-precision f11
   (fp16 NaN checklist). Replacements are given below.

---

## sys.fp-formats

**Verdict: pass** (minor polish only)

### Findings
1. **minor: q6 explanation is garbled.** "480 would need the NaN pattern too, which the format gives up to keep ±NaN symmetric" has it backwards.
   The format *keeps* S.1111.111 as NaN for both signs and gives up 480. Fix: "480 would need S.1111.111, which E4M3 keeps as NaN (for both
   signs) so that integer compare and sort still work (Micikevicius §3.1)."
2. **minor: explainer[6], fp8 cast wording.** See point 1 above. Also, "returns NaN (E4M3)" should say "values that round past $x_{max}$".
3. **minor: TF32 described inconsistently.** explainer[3] says tensor cores "read only the top 10 mantissa bits" (which suggests truncation).
   explainer[8], q10 and f7 say inputs are "rounded to 10 mantissa bits". The docs use both phrasings. Pick "rounded" and use it everywhere,
   or say once that the docs phrase it both ways.
4. **minor: fp16 underflow threshold stated loosely.** explainer[3] says "underflows below $6\times10^{-8}$", and explainer[7] says "Small gradients
   underflow below $6\times10^{-8}$". Key results and the mixed-precision lesson use $2.98\times10^{-8}$ (to 0) and $6.1\times10^{-5}$ (start of the
   subnormals). Use "lose precision below $6.1\times10^{-5}$ and round to 0 below $3\times10^{-8}$" in both places.
5. **minor: q4 distractor wording.** "In fp16, $10^{-6}$ is representable only as a subnormal." $10^{-6}$ is not exactly representable at all, so a
   pedant could call this false too. Fix: "In fp16, $10^{-6}$ is stored as a subnormal."
6. **minor: undefined terms.** "OCP" (q5, explainer[3]) is never expanded. Add "(Open Compute Project spec, which NVIDIA, Arm and Intel adopted)"
   once, or drop it. "GEMM" (explainer[6]) is used before it's defined.
7. **minor: padding.** Cut explainer[0]'s last paragraph ("By the end you should be able to…"), which restates the summary, and explainer[3]'s
   "This is the whole fp16/bf16 debate in one line."
8. **minor: overfull cards.** explainer[3] (328): move the TF32-in-PyTorch paragraph's benchmark sentence into f7 or the reading note.
   explainer[5] (322): cut the LayerNorm NaN paragraph, which q13 and f10 already cover. explainer[7] (323): the fp16 sub-bullets repeat explainer[3]
   and explainer[5]; keep the $\epsilon$ and inference bullets. explainer[6] (308): the trims above suffice.
9. **minor: cross-lesson overlap.** f8 ("why bf16 replaced fp16, and why it still keeps fp32 master weights") has the same second half as
   mixed-precision f8. Here, keep f8 about range and conversion only: drop point 3–4, or replace it with "compare bf16 and fp16 for a gradient
   of $10^{-9}$ and an activation of $10^5$". f10 (fp16 NaN checklist) overlaps mixed-precision f11. Refocus f10 on format causes only
   (overflow, $\infty-\infty$, $\log 0$, $\epsilon\to0$) and drop the loss-scale step.
10. **minor: q9 explanation.** "A gap of 0.25 would mean 11 mantissa bits (counting the hidden bit as stored)" is hard to parse. Use "0.25 is the
    gap you get by using $p=11$ in place of $m=10$."
11. **minor: q13 overlaps** explainer[5]'s last paragraph almost word for word. Make it a predict question: "…before a softmax instead of a LayerNorm.
    Which ops turn the inf into NaN?" Alternatively, leave it and accept it as recall.

### Strengths
Everything derives cleanly from $(e,m)$, and every number verified. The E4M3 448 vs 240 treatment and the RNE overflow case (65520 → inf) are
excellent interview material. The three figures are correct and legible, and `number-line` labels its band "illustrative" as required.

---

## sys.mixed-precision

**Verdict: revise** (light)

### Findings
1. **major: "clip before unscale shrinks updates by S" is true only for SGD.** This appears in explainer[4], q5 (correct choice and explanation)
   and f11 item 2. Adam's update $m/(\sqrt v+\epsilon)$ is invariant to a constant gradient scale as long as $|g|\gg\epsilon$. So the claim that updates
   shrink ~S× is wrong for the optimizer most readers use, and an interviewer will catch it. What actually happens:
   - every step is clipped to the same norm, so the clip threshold is meaningless and the step-to-step magnitude information is lost;
   - with SGD, updates shrink by ~$S$;
   - with Adam, the per-element gradients become tiny. A 1B model at total norm $1/65536$ has an RMS element of ≈ $5\times10^{-10}\ll\epsilon=10^{-8}$,
     so $\epsilon$ dominates and updates shrink by an amount that depends on the model, not by $S$.

   Fixes:
   - q5: change the correct choice to "Almost every step is clipped; gradients reaching the optimizer are ~S× too small", and add the SGD/Adam
     sentence to the explanation.
   - explainer[4]: change "updates shrink by up to a factor S" to "the gradients handed to the optimizer are ~$S$× too small. With SGD the updates
     shrink by that factor. Adam rescales much of it away until $\epsilon$ dominates, but the clip now fires every step."
2. **major: three MCQs replay explainer numbers verbatim, and so test recall.**
   - q9 uses the exact explainer[3] trace (steps 1–3 overflow, step 2004 → $2^{14}$). Replace it with: "steps 1–5 overflow, then all clean. Scale
     at step 4006, and the number of skipped steps?" Answer $2^{13}$, 5 skipped. (Halving to $2^{11}$, then doublings after steps 2005 and 4005.)
     Distractors: $2^{11}$ (no growth), $2^{12}$ (one doubling), $2^{13}$ with 0 skipped.
   - q11 duplicates explainer[1] and f8 ($w=0.05$, $3\times10^{-5}$). Use $w=3.0$, step $5\times10^{-3}$ in bf16: the gap in $[2,4)$ is $2^{-6}$ and the
     half-ulp is $7.8\times10^{-3}$, so the update is lost (verified).
   - q13 duplicates explainer[5] and f10 (0.1 × 10⁴ → 32). Use fp16 adding 0.01 ten thousand times: it ends at **32**, because the gap at 32 is
     $2^{-5}$ and $0.01<2^{-6}$ (verified). Distractors: 100, 64, 16.
3. **minor: the "15× gap" omits TF32.** In explainer[0] and f9 ("about 15× the CUDA cores"), the fair fp32 baseline on Ampere and later is TF32 on
   tensor cores, at half the bf16 rate. NVIDIA's GPU Performance Background gives A100 dense figures of 156 TF32 vs 312 FP16 TFLOP/s
   (`sys_nvidia_gpu_perf_background.txt` l.34). Add: "vs fp32 matmuls with TF32 enabled the gap is ~2× (A100: 156 vs 312 TFLOP/s); vs strict fp32
   on CUDA cores ~15×."
4. **minor: "Tensor Cores accumulate in fp32" needs a caveat.** This is in explainer[5] and f10. PyTorch allows reduced-precision intermediate
   reductions for fp16 and bf16 GEMMs by default (`allow_fp16/bf16_reduced_precision_reduction`; PyTorch "Numerical accuracy › Reduced Precision
   Reduction"). Add one clause, because it explains rare GEMM infs.
5. **minor: explainer[7], "master copy is free" reasoning.** "Relative to the full 16 bytes it is free, because pure fp32 already stores 4 bytes per
   weight" gives the wrong reason. Mixed precision stores 6 bytes of weights (2 + 4) against 4. The extra 2 bytes are paid for by bf16 gradients
   (2 against 4). Say that.
6. **minor: undefined terms.** "buckets" and "compression communication hook" (explainer[6], q12): add one clause each ("DDP groups gradients into
   ~25 MiB buckets and all-reduces each as soon as it fills; a comm hook is a user function that can cast a bucket to bf16 before sending").
   `custom_fwd`/`custom_bwd` (f11) also needs a gloss ("decorators that make a custom autograd Function respect autocast").
7. **minor: explainer[3], "N caps how often a step can be skipped".** NVIDIA's wording covers only growth-induced overflows. Gradient drift can still
   cause skips at any time. Say "growth attempts cause at most one skip per $N$ steps".
8. **minor: overfull card.** explainer[3] is 333 words. Cut the trace paragraph down to its first two sentences: the full trace is in the figure,
   and the new q9 tests it.
9. **minor: plan pitfall not addressed.** The plan asks for a neutral mention that `torch.cuda.amp.*` is deprecated in favour of `torch.amp.*`.
   One clause under the code block.
10. **minor: padding.** Cut explainer[0]'s last line ("This lesson derives each fix, then answers…").

### Strengths
The structure (three failure modes → one fix each) teaches well, and the lost-update derivation includes the binade factor and the "decrease
below a power of two" subtlety from the plan review. The DDP-fp32 point is correct (verified). The dynamic-scale figure's trace is internally
consistent: every doubling falls 2000 steps after the last reset.

---

## sys.memory-anatomy

**Verdict: revise**

### Findings
1. **blocking: an "80 GB" H100 is not 74.5 GiB, and PyTorch does not report 74.5 GiB.** This is in explainer[0] (Units paragraph) and q12.
   HBM is sized in binary units: an H100 80GB has 80 GiB (≈ 85.9×10⁹ bytes). nvidia-smi shows 81,559 MiB total, and PyTorch reports a total capacity
   of about 79 GiB. (I'm confident of this from routine use, but it isn't in the source cache, so the writer should verify it with
   `torch.cuda.get_device_properties(0).total_memory` on any H100 log or doc, or drop the specific figure.) The plan's pitfall line
   ("80 GB H100 ≈ 74.5 GiB usable") has the same error.

   With the real numbers, q12's answer flips. A 78 GB (72.6 GiB) estimate against ~79 GiB leaves ≈ 6.5 GiB, so "it fits with ~6–8% to spare"
   becomes the defensible answer. That is one of the current distractors.

   Fixes:
   - explainer[0]: "GPU 'GB' are really GiB: an 80 GB H100 has 80 GiB ≈ 85.9×10⁹ bytes, of which PyTorch sees ≈ 79 GiB. Our estimates are in
     10⁹-byte GB, so convert before comparing."
   - q12: give the capacity in the stem to avoid hardware trivia. "PyTorch reports 79.1 GiB total. Your estimate is 76 GB (10⁹ bytes). Prediction?"
     76 GB = 70.8 GiB, so ≈ 8.3 GiB of headroom covers a 1–2 GB context plus buffers: it fits. Distractors:
     - "Likely OOM: 76 is within 4% of 79.1" (treats GB as GiB);
     - "OOM: 76 GB exceeds the 74.5 GiB an 80 GB card really has" (the current misconception);
     - "Fits only if the caching allocator compresses idle tensors".
2. **major: SwiGLU recount (explainer[5], f4).** "That totals about 18sbh, the same as GeLU without dropout. More if the implementation also saves
   the SiLU output." This presents the fused-kernel case as the default. In eager PyTorch, `mul` saves both operands, so silu(gate) is stored:
   $2+4\cdot\tfrac{16}{3}\approx23.3\,sbh$ (verified with `saved_tensors_hooks`). Fused SwiGLU kernels that recompute SiLU in the backward get 18.
   A Llama-style layer (FlashAttention, no dropout, RMSNorm, SwiGLU) is therefore ≈ 37sbh in eager mode and ≈ 32sbh fused, not "≈ 32" in general.

   Fixes:
   - explainer[5]: "Eager PyTorch saves five tensors, ≈ 23sbh. A fused SwiGLU kernel that recomputes SiLU saves four, ≈ 18sbh (= GeLU without dropout)."
   - f4: give both numbers.
3. **major: q11 is weak.** It is recall of explainer[6]'s inference line and interviewer probe 6, and two distractors are implausible ("smaller
   vocabulary removes the embeddings", "four fp32 copies of every activation"). Replace it with a compute question: "Serving a 13B model in bf16
   with EleutherAI's 1.2× rule, no large KV cache: memory?" Answer ≈ 31 GB (2 × 12.75 × 1.2 = 30.6). Distractors: 26 GB (no overhead),
   61 GB (fp32 weights × 1.2), 204 GB (training states).
4. **minor: the step-2 OOM story omits gradients.** In explainer[6], q3 and f9, if `zero_grad(set_to_none=False)` is used (the default before
   PyTorch 2.0), gradients also persist into step 2's forward. Add one clause to explainer[6] so the reader doesn't treat Adam state as the only
   cause. Don't add it as a q3 distractor: that would make two choices correct.
5. **minor: q3 distractors are weak.** "LR warmup makes step-2 gradients larger in memory" is implausible. A better one: "The caching allocator
   frees step 1's activations only at the end of step 2". That is false (they're freed during backward) but sounds plausible.
6. **minor: q7 explanation.** Say what 1.0 GB and 16.8 GB represent, or replace them with errors someone would actually make: 8.4 GB (logits plus
   gradient) and 0.52 GB ($V=32$k).
7. **minor: parameter formula scope (explainer[2]).** Say that it excludes learned positional embeddings (the Playbook notes this). GPT-2/3-style
   models add $s_{max}h$.
8. **minor: the plan's item 7 mentions embedding gradients**, which aren't mentioned here. One clause in explainer[6]: "a dense embedding gradient
   is another $Vh$ tensor".
9. **minor: terminal-theme legend collision** in `memory-timeline`: "weights (bf16 + fp32 master)" runs into the "Adam m, v" entry. Shorten it to
   "weights (bf16 + fp32)" or move the legend to one column.

### Strengths
The Korthikanti derivation is itemised with the reason each tensor is kept, and every worked number recomputes. The 13B budget with "diagnose
before you pull" is exactly the interview move. The figures are formula-driven and correct: the timeline peaks (76/86 GB) and the breakdown
percentages all check out.
