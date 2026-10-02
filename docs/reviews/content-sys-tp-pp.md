# Content review: sys.tensor-parallel, sys.pipeline-parallel

Reviewer checks: read both YAMLs, plan sections (`docs/content-plan-applied.md` §sys.tensor-parallel and §sys.pipeline-parallel),
bandwidth conventions (`docs/reviews/plan-applied-review.md` A1), both figure modules, all 8 figures (workbench, plus editorial or terminal).
I recomputed every number in Python (`scratchpad/tppp_check.py`) and simulated GPipe/1F1B with unequal stage times (`scratchpad/sim.py`).
`validate.py` reports no warnings for either file. Claims checked against the cache: Shoeybi §3/§4.2/App B.2; Narayanan 2021 §3.2, §3.3.1, §3.5,
Takeaway #1; Korthikanti §4.2.1–4.2.3, §6.3 Table 4; Huang §2.2–2.3; PipeDream staleness eqs.; Ultra-Scale "TP in a transformer block"
(GQA, LayerNorm); Scaling Book Part 12 TP/PP/Examples; Llama 3 §3.3.2 + Table 4; kipply §model parallelism/§latency; PyTorch TP tutorial (loss parallel); TorchTitan (AsyncTP).

## Writer's flagged uncertainties (checked first)

1. **"Full recomputation repeats the forward all-reduces."** This is supported, but the lesson does not cite a source for it. Narayanan 2021 §3.5 defines recomputation as
   "running the forward pass a second time just before the backward pass", and Shoeybi §3 puts the two $g$ all-reduces in that forward pass.
   Korthikanti §6.3 Table 4 is indirect evidence: recomputing the full layer costs 7.6 ms against a 7.7 ms forward, and that forward includes communication. Two fixes. Add
   `Narayanan et al. 2021 §3.5; Korthikanti et al. 2022 §6.3` to the `source:` of explainer[3], f3 and q3. Also scope the claim: "Megatron's full recomputation re-runs the whole layer
   forward, including its 2 all-reduces". Strictly, the layer's own backward does not need the second $g$'s output, because the next layer has
   already checkpointed it. So "+2L" describes the implementation, not a law. q3's key is unaffected, because its stem says "no recomputation".
2. **Inference TP claim (explainer[7], last bullet).** It is correct and supported. kipply §model parallelism says that both the weight streaming and the FLOPs are divided by
   the TP degree. kipply §latency calculations treats the per-token communications as latency-dominated (about 8 µs per message) at small batch. Optional:
   add Pope et al. 2022 (`llm_pope2022_inference_scaling.txt`) as a second source. Also add one sentence on why: with roughly $2L$ latency-bound all-reduces
   per token, communication latency puts a floor under per-token latency, so adding ranks gives diminishing returns.
3. **Llama 3's DP flavour.** The sources disagree. The Llama 3 paper (§3.3.2, cache l.1036–1040) says that its FSDP "shards optimizer states
   and gradients", and that parameters are not resharded after the forward pass. That is ZeRO-2-like, used *with* 16-way PP. The Scaling Book (Part 12
   "Examples", l.313) calls it "128-way ZeRO-1". The writer was right not to label it. But the lesson's bullet that "ZeRO-2 … $m$ times the traffic" is then
   too absolute (see PP finding M2).
4. **"Slowest stage sets the pace."** The principle is right, but the card's formula and its worked example disagree. See PP finding B1, which includes the exact formula.
5. **PP uses TP results that are not in its prereqs.** Confirmed. explainer[0], explainer[3] (the 23 sbh formula "from `sys.tensor-parallel`"), explainer[5] and q8 all depend on TP. TP has
   order 210 and PP 230, and the plan's reading order (l.138) already puts TP first. So add `sys.tensor-parallel` to the prereqs. There is no cycle.
   Also add `sys.zero` (order 190), which explainer[7], q10 and f6 rely on. See PP finding M3.

---

## sys.tensor-parallel

**Verdict: revise** (light: one major finding and polish. The core is correct and well taught.)

### Findings

**Major**
- **M1. explainer[8] "Why within a node?" uses the 50 GB/s case beyond its scope.** The bullet says "Across nodes, at 50 GB/s per GPU,
  it takes several times the compute." Under the A1 convention, and in `sys.collectives` l.134, 50 GB/s per GPU applies only to a TP group
  *spread one GPU per node*. A TP group spanning whole nodes uses node egress (400 GB/s). The lesson itself says this in f5 ("$F/2475$ … at
  the 400 GB/s node rate"), so the probe and f5 contradict each other. The Scaling Book (Part 12, l.283) even notes that 16-way TP over exactly 2
  nodes can stay compute-bound. Fix: "If the TP group is placed one GPU per node, each GPU's activations leave through its own 50 GB/s NIC, which takes about 5× the
  compute. Even a sensible cross-node layout only gets $t<F/2475$, and the all-reduce is on the critical path, so TP stays within 1 node (at most 2)."
  Make the same change to f8's "Numbers" bullet ("if each GPU must use its own 50 GB/s NIC": add "i.e. the group is spread one GPU per node").
  In explainer[5], add one clause after the 4.7 ms bullet: "(a TP group spanning two *full* nodes fares far better: $t<F/2475$; the Scaling Book allows ≈16-way)".

**Minor**
- **m1. explainer[3], f3, q3: cite recomputation** (see flagged item 1).
- **m2. f5 wording.** Change "if the group spans nodes" to "if the group spans whole nodes (e.g. 16 GPUs on 2 nodes)". That separates it from the one-GPU-per-node case.
- **m3. explainer[5] roofline scope.** The rule $t<FW/C$ assumes a 2-matrix MLP, as the Scaling Book's $2\cdot2\cdot BDF$ does. The example $F=28{,}672$ is Llama 3 70B's
  SwiGLU width, and SwiGLU has 3 matrices, so compute is $6nhF/t$ and the bound loosens to $t<1.5\,FW/C$. Add "(2-matrix MLP; a gated MLP has 1.5× the compute)".
- **m4. q6 distractor 0.24 GB** equals $sbh(34+5as/h)/t$, which is exactly the TP+SP value. It is a fine distractor given the stem, but in the explanation write "…wrongly shards the
  replicated $10sbh$ (sharding it is what sequence parallelism does, next lesson)". That way it doesn't later look as if it contradicts `sys.sequence-context-parallel`.
- **m5. q13 style.** The correct choice ("It rises by about 2.3×", 22 chars) is much shorter than the distractors (46–58 chars), and only the distractors carry
  a "since …" reason. Either give it a reason, "It rises by about 2.3×, since compute halves and traffic grows 7/6", or strip the reasons from the distractors.
- **m6. Figures.** In `megatron-mlp` and `megatron-attention`, the right-hand output box touches or clips the right edge. In all three themes, `f` is labelled "fwd: copy"
  where the text says "identity". Use "identity" (or "no-op", as in `layer-comm`). The title of `megatron-mlp` says "the sum is the only communication", but the backward $f$ also
  communicates, so add "in forward". `layer-comm` shows a residual "+" after attention but none after the MLP; add the second "+".
- **m7. q12 LaTeX.** `$[a/t$ heads$]$` splits the brackets across math spans. Write "its $a/t$ heads".
- **m8. Mild redundancy.** The within-node argument and its numbers appear in explainer[5], explainer[8], f8 and q8. This is acceptable because they serve different roles, but explainer[8]'s bullet
  could shrink to one line that points back to the numbers.

**Verified correct** (sample): $XA$ tiny example. GeLU(1)+GeLU(−1)=0.68. Per-layer forward FLOPs 7.15e12 → 0.903 ms/rank. $S=67.1$ MB, 117 MB
traffic, 0.52 ms (58%) on NVLink and 4.70 ms (5.2×) at 50 GB/s. $C/W=2198$ / $2472$, giving $t<13.0$ / $11.6$. Logits 1.05 GB and 49 KB. Vocab pad 51,200 (Shoeybi:
"divisible by 128×8=1024"). Korthikanti 34 / 10 / 24 itemisation, 5as/h=80, 2.87 GB → 0.58 GB, 43%. q4 33.6 → 50.3 MB. q6 0.386 GB and all
three distractors. q7 6.5 → 4. q13 2.33. LLaMA-3 8B has 8 KV heads and TP=16 needs duplication (Ultra-Scale l.1984–1987). Dropout RNG rules (Shoeybi App B.2).
Narayanan's $8bsh\frac{t-1}{t}$ per layer per device. Exactly one MCQ choice is correct in every item. The correct choice is the longest in 4 of 13 items, never by much.

### Strengths
The block-matrix tiny example is concrete and comes before the general derivation. The "why column first" argument uses a numeric GeLU counterexample, and the $f/g$ backward derivation
comes from the chain rule. The memory card itemises Korthikanti's terms instead of just quoting the formula. Keep all of these.

---

## sys.pipeline-parallel

**Verdict: revise**

### Findings

**Blocking**
- **B1. explainer[6] and q11: the load-imbalance formula disagrees with its own example, and q11's stem is ambiguous.** The card says the step "stretches by roughly
  $\max_i t_i/\bar t$". With $\bar t$ the mean stage time, the example gives $10.6/8.65=1.23$, not the stated "about a third" ($10.6/8=1.33$). The 1.33
  is relative to the same pipeline *without* the LM head. q11 asks "how much slower … than a perfectly balanced pipeline?". The natural
  reading (same total work, perfectly rebalanced) gives about 22%, not 33%. I simulated GPipe and 1F1B with stage costs (8, 8, 8, 10.6) and $t_b=2t_f$:

  - against the head-free pipeline: 1.24× at $m=8$, 1.27× at $m=16$, 1.30× at $m=32$, and 1.32× as $m\to\infty$;
  - against a rebalanced pipeline with the same total work: 1.14×, 1.18×, 1.20×, 1.23×.

  For identical micro-batches, with stage $i$ taking $\tau_i=t^f_i+t^b_i$, the step is exactly
  $$T=\textstyle\sum_i\tau_i+(m-1)\max_i\tau_i.$$
  The simulation matches this in every case, and it reduces to $(m+p-1)(t_f+t_b)$ when the stages are balanced.

  Fix for the card: state the exact formula above (one aligned display), then say "for $m\gg p$ the step is $\approx m\max_i t_i$, so relative to the
  head-free pipeline it stretches by $t_{\max}/t_0$". Fix for q11: change the stem to "…than the same pipeline without the LM head (large $m$)?". Keep 33% as the key.
  The "8%" distractor then becomes "the cost if the head's work could be spread perfectly" ($34.6/32$), which is a good real misconception. Replace the "2.6%" distractor, which is
  implausible, with "About 23% slower, the overload measured against the mean stage time". That is the right answer to a different baseline, so name the baseline in the explanation.
  Also add "(for large $m$; at $m=16$ it is ≈27%)" to the explanation.

**Major**
- **M1. The figure `bubble-vs-m` is rendered but never referenced.** grep finds no `sys.pipeline-parallel/bubble-vs-m` in the YAML. The plan asks for it, and it
  shows both conventions. Put it in explainer[1] after "Both shrink only when $m\gg p$", with a caption such as "Solid: idle fraction $(p-1)/(m+p-1)$; dashed:
  Megatron's $(p-1)/m$. Both need $m\gg p$; at $m=4p$ the idle fraction is still ≈18%." Polish: the "p=4" label sits on the x-axis line.
- **M2. explainer[7] and f6 are too absolute about ZeRO-2.** "ZeRO-2 shards gradients, so the accumulated gradient would be reduce-scattered after every micro-batch's backward"
  presents one implementation as forced. Llama 3, the lesson's own running example, used FSDP that "shards optimizer states and gradients" together with 16-way PP, and kept
  parameters unsharded between forward and backward (Llama 3 §3.3.2). The Scaling Book labels the same setup ZeRO-1. Fix: give ZeRO-2 the same "or …" alternative that ZeRO-3 already has:
  "…or keep a full gradient buffer and reduce-scatter once, which forfeits the gradient saving". Change "usually paired with ZeRO-1, as in DeepSeek-V3" to add "(Llama 3's report
  describes FSDP sharding gradients and optimizer state with PP; the Scaling Book calls it ZeRO-1, so labels vary)". The same over-claim appears in `sys.zero` explainer
  l.186. Fix it there too, or leave a note for that lesson's owner.
- **M3. The prereqs omit `sys.tensor-parallel` and `sys.zero`.** See flagged item 5. explainer[3]'s example uses the TP activation formula, explainer[5] uses TP traffic
  and timings, and explainer[7], q10 and f6 use ZeRO stages. Set the prereqs to `[sys.memory-anatomy, sys.collectives, sys.ddp, sys.zero, sys.tensor-parallel]`. explainer[5] already restates the
  TP traffic formula inline, so q8 can be answered from the lesson. Keep it that way.

**Minor**
- **m1. explainer[5]:** "PP communication is about 1–9%". The recomputed values are 8.7% when every TP rank sends the full tensor, and $(0.126+0.098)/11.57=1.9\%$ with scatter/gather. Write "about 2–9%".
  Also, the 4.7 ms figure for "the stage's own TP all-reduces … even on NVLink" happens to equal the TP lesson's 4.7 ms for the one-GPU-per-node case, which is a different configuration. Add
  "(12 layers × 2 all-reduces)" so readers don't conflate the two.
- **m2. explainer[3] example:** "1F1B stage 1: 55.6 GB … large but bounded". Each GPU also holds GPT-3's model states: $175\text{B}/64$ GPUs × 16 B ≈ 44 GB with mixed-precision
  Adam. So the total is ≈ 99 GB > 80 GB. Add one clause: "plus ≈44 GB of model states, so in practice recomputation, SP or ZeRO-1 is still needed (Narayanan §3.5:
  recomputation is required for large models with PP)". Otherwise readers infer that it fits.
- **m3. explainer[4] and f10: PipeDream staleness** "up to $p$ steps". PipeDream's update $w_1^{(t-n+1)},\dots,w_n^{(t)}$ means stage 1 lags by $p-1$. Write "up to $p-1$".
- **m4. explainer[1] forward phase:** "Micro-batch $k$ reaches stage $i$ after $i-1$ forward slots" is loose. State the start time: micro-batch $k$ starts on stage $i$ at
  $(k+i-2)t_f$, so the last one finishes on stage $p$ at $(m+p-1)t_f$. This takes one line and makes the derivation reproducible at a whiteboard.
- **m5. explainer[0]** last paragraph ("The rest of the lesson asks three questions…") is a roadmap that the card titles already provide. It could be cut.
- **m6. q8 length:** the correct choice (53 chars) is the shortest by 11+ chars. That is not a giveaway, but tighten the distractors.

**Verified correct** (sample): bubble values 27.3/37.5, 17.9/21.9, 46.7/87.5, 15.8→30.4 (q5), and Llama 3 $m=p=16$ → 48.4%. The figures' total 33 $t_f$ = $(8+3)\cdot3$.
1F1B warm-up $p-i$ (1-indexed) and peak $p-i+1$, which the simulation and the quiz figure confirm (stage 2 peak = 3, and the figure does not annotate the answer). The lower-bound argument for any
flush schedule. GPipe "negligible for $M\ge4K$" and the early-recompute remark (Huang l.283–285). BatchNorm statistics per micro-batch, and moving averages over the mini-batch
(Huang l.165–167). 0.58 GB/layer → 222.3 GB GPipe vs 55.6 GB 1F1B. Stage inputs 50.3 MB → 1.61 GB. Stage forward 11.57 ms. p2p 1.01 ms, scatter/gather 0.126 ms,
NVLink AG 0.098 ms. q8 $42\,bsh$. Head $V/(12h)=2.61$. Narayanan's $(n-d)/b'$ (§3.3.1). Llama 3 TP8/PP16/DP128, 16M tokens, 16 seq per DP group, 1 layer removed from
the first and last stages, async p2p (§3.3.2, Table 4). DeepSeek-V3 2-way ZeRO-1 and the "16× cheaper DP all-reduce" remark (Scaling Book l.304–313). The bandwidth conventions are respected (the p2p
send from each GPU through its own NIC is the legitimate 50 GB/s case). Exactly one correct choice per MCQ, apart from q11's baseline (B1).

### Strengths
The bubble is derived twice: once from the schedule, once as a lower bound that explains *why* 1F1B cannot beat GPipe. The schedules are simulated from dependencies, not hand-drawn.
Both bubble conventions are carried consistently through the cards and questions. The batch-size constraint and the Llama 3 example tie the lesson to real configurations. Keep all of these.
