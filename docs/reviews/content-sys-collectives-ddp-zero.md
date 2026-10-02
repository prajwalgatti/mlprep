# Content review: sys.collectives, sys.ddp, sys.zero

What I checked:
- All three YAMLs, read in full.
- The plan sections in `docs/content-plan-applied.md` (conventions, §sys.collectives, §sys.ddp, §sys.zero) and the bandwidth rules in `docs/reviews/plan-applied-review.md` (A1, B7, B10, C8, C15).
- All three figure modules, and all 10 figures in workbench plus terminal or editorial.
- Every number, recomputed in Python. Every worked number, MCQ key and distractor matches except where noted below.
- `validate.py`: 0 errors and no warnings for these files.
- Claims checked against the cache:
  - Patarasuk & Yuan Lemmas 1–5 (induction, l.236–382);
  - nccl-tests PERFORMANCE.md correction factors (l.61–99);
  - NCCL guide "undefined behavior… hangs, crashes, or data corruption" (l.15);
  - Scaling Book Part 12: 370 GB/s, SHARP ≈30%, cross-node AG/RS = bytes/400e9, all-to-all ≈50 GB/s, 2200/2475, (X−1)/X at l.277;
  - Scaling Book Part 5 FSDP "same communication cost" (l.132–134);
  - Li et al. 2020: Fig. 2 60M params, Fig. 3, bitmap all-reduce, Fig. 7 bucket sweeps (l.1060–1084), parameter averaging (l.174–196);
  - PyTorch DDP note: Reducer, DDPOptimizer, wrap before compile;
  - PyTorch tuning guide: DDP over DataParallel, no_sync, bucket rebuild with `find_unused_parameters=False`;
  - Ultra-Scale Playbook: 512+ GPU note (l.1339), ZeRO-1 steps and hooks note (l.1600–1650), "ZeRO-2 is usually the best option" (l.1700–1704), DP ≤ 512 for ZeRO-3 (l.1743), ZeRO-1/2 with PP and DeepSeek-V3 (l.3354–3358);
  - ZeRO: 33→2 GB (l.631), 3B fused buffer 12 GB (l.647), 1.5× (l.742, 798), 1T on 1024 GPUs;
  - Ren et al. 2021: 4M edge weights, "minimum communication volume", 13B on one V100.

## Writer's flagged points (checked first)

1. **The general-knowledge claims are all correct.** None of them is in the cache.
   - `DistributedSampler` and `set_epoch` (ddp explainer[1]): correct, but the card's `source:` doesn't cover it. Add "PyTorch `torch.utils.data.distributed.DistributedSampler` docs".
   - `SyncBatchNorm` (ddp explainer[7], q10): correct.
   - DataParallel GIL contention (ddp explainer[1], q8, f10): correct. The cached tuning guide only says DDP "offers much better performance and scaling". The GIL point, plus the per-iteration model *replication* that the lesson omits, comes from the PyTorch "Getting Started with DDP" tutorial ("Comparison between DataParallel and DistributedDataParallel"). Cite that, and add "re-replicates the model to every GPU each forward" as a third reason. Graders often look for it.
   - "Autocast leaves `.grad` fp32": correct. `sys_pytorch_amp_examples.txt` l.27 says "Creates model and optimizer in default precision". Autocast casts op inputs, not parameters, so `.grad` matches the fp32 parameter dtype. q2's source is fine.
   - Goyal et al. 2017 (ddp explainer[0], [7], q9, q10) is **not in the cache**. The two claims ($1/(kn)$ loss normalisation; fixed per-worker BN batch) match the paper as I know it, so keep them. Just note that they weren't verified against the cache.
2. **"Illustrative 25 GB/s PCIe" (zero explainer[7]).** That is acceptable. To make it concrete, say "≈ PCIe Gen4 x16 achieved; Gen5 (H100 hosts) is roughly double". See also Z-m3 (CPU Adam time).
3. **Node-egress factor, (P−1)/P with P = 64 GPUs vs the Scaling Book's (X−1)/X with X = 8 nodes. Decision: mention it in one sentence. No number needs to change.** Which factor is right depends on the algorithm.
   - *8 parallel flat rings*, each carrying $S/8$ through a different NIC: $2\cdot\frac{63}{64}\cdot\frac{S/8}{50\text{e}9}=2\cdot\frac{63}{64}\cdot\frac{S}{400\text{e}9}$, i.e. 69 ms / 12.8 ms. So the writer's number is exact for that algorithm.
   - *Hierarchical* (the scheme the card actually describes first): the cross-node phase is an all-reduce of $S/8$ per GPU over X = 8 nodes, $2\cdot\frac78\cdot\frac{S}{400\text{e}9}$, i.e. **61 ms / 11.4 ms**, plus the NVLink phases. That 61 ms is also the node-level lower bound: treat each node as one rank with 400 GB/s egress.
   - Why it matters: ddp explainer[5] computes the hierarchical cross-node phase as 11.4 ms, *below* the 12.8 ms "node-egress model" a line earlier. An attentive reader will think one of them is wrong. collectives explainer[5] also calls the two schemes "equivalent", and q3 calls 69 ms the "bandwidth-optimal" estimate, which is 12% above the node-level bound.
   - The ZeRO lesson uses a third convention: explainer[5] divides 28/42 GB by 400 GB/s with factor 1, giving 70/105 ms.
   - Fix: see C-m1. Leave all keys as they are. The distractors are far enough apart that nothing becomes ambiguous.
4. **Two cards at about 335 words: confirmed, and a few more are over 300.** Measured body word counts:

   | Lesson | Card | Words |
   |---|---|---|
   | collectives | explainer[5] | 340 |
   | collectives | explainer[4] | 308 |
   | ddp | explainer[6] | 339 |
   | ddp | explainer[0] | 317 |
   | ddp | explainer[5] | 307 |
   | zero | explainer[7] | 311 |

   For trims, see C-m4 and D-m5. The ~310-word cards are borderline and can stay.
5. **Dropping `static_graph` and `gradient_as_bucket_view` is acceptable.** Interviewers rarely probe them. Still, add one clause each in ddp explainer[3] / [6]; see D-m3. Plan item 10's `Join` is also unnamed (D-m3).

---

## sys.collectives

**Verdict: revise** (light: one blocking caption error that comes from a figure weakness, plus polish. The core derivations are correct and well ordered.)

### Findings

**Blocking**
- **C-B1. explainer[1] caption and the `collectives-grid` figure.** The caption says "only all-reduce and all-gather leave every rank with the same buffer". Broadcast also does, and the figure's own broadcast panel shows four identical rows. The figure makes it worse: chunks are coloured only by *source rank*, so the all-gather output and the all-to-all output look **identical** (both have every row = r0|r1|r2|r3 colours). A reader comparing the panels will conclude that all-to-all also leaves every rank with the same buffer. That is exactly the misconception q1 tests.
  - Fix the caption: "…only broadcast, all-reduce and all-gather leave every rank with the same buffer."
  - Fix the figure: print the chunk index $k$ in each cell of the all-gather and all-to-all panels. With that, all-gather row $j$ reads $0,1,2,3$, while all-to-all row $j$ reads $j,j,j,j$, coloured r0…r3. Alternatively, shade by chunk index.

**Minor**
- **C-m1. explainer[5] node-egress factor** (see flagged point 3). Replace "(Running 8 rings at once … is equivalent.)" with: "Running 8 flat rings at once, each carrying 1/8 through a different NIC, gives $2\cdot\frac{63}{64}\cdot S/400\text{e}9\approx69$ ms. The hierarchical version's cross-node phase is an all-reduce over the $X=8$ nodes, $2\cdot\frac{X-1}{X}\cdot S/400\text{e}9\approx61$ ms (the Scaling Book's form, and the node-level lower bound), plus the NVLink phases. Either way, budget about $2S/400\text{e}9$."
  - In q3's explanation, change "bandwidth-optimal" to "multi-ring/hierarchical". This fixes the apparent clash with ddp explainer[5] in one place.
- **C-m2. explainer[5] Scope, SHARP.** The Scaling Book's "about 30% in practice" is an **intra-node** (NVLink SHARP) measurement (Part 12 l.192–195). The card puts it under cross-node scope. Say "(measured within a node)".
- **C-m3. explainer[7] probe "Is a broadcast cheaper than an all-gather of the same total size?"** "No" is right, but the answer implies a big gap. The bounds are $S/W$ vs $\frac{P-1}{P}S/W$, so they are about equal, and a pipelined ring broadcast reaches $S/W$. Rephrase as "No, about the same. Broadcast is bounded by the root sending all $S$, all-gather by each rank receiving $\frac{P-1}{P}S$."
- **C-m4. explainer[5] is 340 words.** Move the Scope paragraph (fat-tree / SHARP) into explainer[8] Key results or f9's caveats, or cut the flat-ring arithmetic, since q3 and f9 already show it.
- **C-m5. f8 "broadcast ⊗ identity"** is vague notation. Write "$A=\mathbf 1_P\otimes I$ stacks $P$ copies of the concatenated shards; $A^\top$ sums the copies and hands block $r$ to rank $r$, which is a reduce-scatter."
- **C-m6. q6 explanation, last two sentences, are garbled.** "Scaling by $1/8$ assumes the smaller chunks come without the extra steps; 30 ms wrongly doubles the time for more ranks." Split it so that each distractor gets its own clean sentence.

### Strengths
The lower-bound card followed by the ring derivation is the right arc. The all-gather/reduce-scatter transpose derivation is done properly, via the chain rule plus the linear-map view. The ring figure is simulated and correct: I checked every outlined send against the "(i−j) mod P" rule. The bandwidth rule is taught with the flat-ring trap as a contrast, exactly as A1 asked.

---

## sys.ddp

**Verdict: revise** (one major finding that misreports a source; the rest is polish. The B/P > C/W derivation and the worked example are excellent.)

### Findings

**Major**
- **D-M1. explainer[3] misreports Li et al.'s bucket sweep.** The card says "Li et al. found 10–25 MB best for ResNet-50 and BERT with NCCL on 16 GPUs." The paper (l.1068, l.1078–1081) found 10–25 MB best for **ResNet-50 only**. For **BERT**, with 15× more parameters, **50 MB** was best with NCCL. With Gloo, 5 MB won for both.
  - The paper's real lesson is that bigger models want bigger buckets, which is a better teaching point than the one the card makes.
  - Fix: "Li et al. found 10–25 MB best for ResNet-50 with NCCL, but 50 MB for the 15× larger BERT; with Gloo, 5 MB won for both, because Gloo's all-reduce stops speeding up beyond about 512 KB."

**Minor**
- **D-m1. Node-egress factor clash in explainer[5].** See flagged point 3 and C-m1. After the hierarchical bullet, add one clause: "(its cross-node phase, 11.4 ms, is below 12.8 ms because the 12.8 ms figure models 8 flat rings; both round to $2S/400\text{e}9\approx13$ ms)."
- **D-m2. explainer[3] reuses $\alpha$ with a different meaning.** Here, $T\approx k\alpha+\dots$ makes $\alpha$ a *per-collective* fixed cost. In `sys.collectives`, $\alpha$ is *per message*, and a ring all-reduce pays $2(P-1)\alpha$. Write $\alpha_{\text{coll}}$, or add "(for a ring, $\alpha_{\text{coll}}\approx2(P-1)\alpha$)". This also explains why the latency term grows with $P$, which matters for the 512+ GPU note in explainer[7].
- **D-m3. Small coverage gaps** (plan items 6, 7, 10). One clause each:
  - explainer[3]: "`gradient_as_bucket_view=True` makes `.grad` a view into the bucket, saving the copy and one gradient-sized buffer."
  - explainer[6]: "`static_graph=True` lets DDP record the used-parameter set once, instead of walking the graph every step."
  - explainer[6], uneven data: "PyTorch's `Join` context manager shadows the missing collectives for ranks that finish early; `DistributedSampler` pads by default so ranks see equal counts."
- **D-m4. Duplicate items.** q8 and f10's DataParallel bullet test the same fact. Either turn f10 into a parameter-server vs ring comparison only, or make q8 a predict question ("GPU 0 runs out of memory first under DataParallel; why?", answer: outputs are gathered and the loss is computed on GPU 0).
- **D-m5. explainer[6] is 339 words.** Move the `torch.compile` paragraph into explainer[7] (which then needs a trim of its own) or into a flashcard. Alternatively, cut "Stragglers", which `sys.collectives` explainer[6] already covers.
- **D-m6. `bucket-tradeoff` figure has visible spikes** at about 3, 5–6, 25 and 45 MiB. These come from `ceil(S/bucket)` discretisation in `_exposed`. Fix: evaluate only at bucket sizes that divide S evenly, or smooth with a running minimum. The near-vertical cliff at about 1.6 / 4 MiB is genuine (queue backlog), but a reader will think it's a bug. Add to the caption: "the cliff is where a bucket's all-reduce becomes faster than the backward produces the next bucket".
- **D-m7. `dp-roofline` figure: the "4,945" label is struck through by the backward line** in every theme. Move it up and left.
- **D-m8. explainer[1] `source:`** is missing the DistributedSampler and DataParallel sources (flagged point 1).

### Strengths
The derivation of $B/P>C/W$ is clean: a FLOP refresher, cancellation of $N$, scope, and the fp32 doubling, all consistent with B10. The 1 node / 8 nodes / 1,024 tokens worked example teaches the hidden-vs-exposed judgement interviewers want. The flat-ring distractor is placed and explained correctly. MCQs are mostly predict, compute and debug, with lengths well balanced.

---

## sys.zero

**Verdict: revise** (one major finding where the lesson contradicts its cited source; otherwise correct and well derived).

### Findings

**Major**
- **Z-M1. ZeRO-2 + pipeline parallelism contradicts the cited Playbook.** The claim appears in explainer[8], f10 item 2 and Key results ("ZeRO-1 with PP"). explainer[8] says "With ZeRO-2/3, each pipeline micro-batch would trigger its own gradient reduce-scatter or weight re-gather, multiplying the traffic", and cites the Playbook. The Playbook says the opposite (l.3354–3358): "ZeRO-1 and ZeRO-2 … can be easily combined with Pipeline Parallelism and are complementary to it". Its warning (l.3340–3350) is about **ZeRO-3** only. Llama 3 also ran a ZeRO-2-like FSDP with 16-way PP (see `content-sys-tp-pp.md` flagged point 3).
  - There is a real point underneath: if gradients are sharded, accumulating across micro-batches means either reduce-scattering every micro-batch or keeping a full gradient buffer. That is true of any gradient accumulation, and it is the reason DeepSpeed's pipeline engine supports only ZeRO-1. But it is a trade-off, not a rule.
  - Fix explainer[8]: "ZeRO-1 composes most simply with PP. ZeRO-2 also works (the Playbook calls both easy to combine), but sharded gradients must be reduce-scattered per micro-batch or held in full until the last one, a traffic-vs-memory trade-off; DeepSpeed's pipeline engine allows only ZeRO-1. ZeRO-3 is the problematic one: it would re-gather weights per micro-batch unless it keeps them resident."
  - Make matching edits to f10 item 2 ("ZeRO-1, or ZeRO-2 with per-micro-batch RS") and to Key results ("ZeRO-1/2 with PP; avoid ZeRO-3 with PP"). q10 only concerns ZeRO-3, so it is fine as written.

**Minor**
- **Z-m1. explainer[3] says ZeRO-3 "issues $2L-1$ more all-gathers per step than ZeRO-2".** The count is implementation-dependent. ZeRO-2's parameter all-gather is itself bucketed, and whether the last layer is re-gathered varies. Write "about $2L$ per-layer all-gathers per step, instead of ZeRO-2's single (bucketed) post-update all-gather". Also, the Playbook's "DP shouldn't exceed 512" (l.1743) is framed as an overlap limit, not specifically latency. "One reason" is acceptable, but say "the Playbook's rule of thumb".
- **Z-m2. explainer[6] "Below the threshold … ZeRO-3's exposed time is about 1.5× that of ZeRO-2."** Exposed time is comm minus overlappable compute, so the ratio isn't 1.5× in general. Write "in the strongly comm-bound limit, step time approaches 1.5× ZeRO-2's".
  - Same card: "The Scaling Book points out they aren't on the critical path". The book (Part 5 l.125–131) marks the **reduce-scatters** "not on critical path" and the all-gathers "can be done ahead of time". Reword to match.
- **Z-m3. explainer[7] offload omits CPU Adam time.** An Adam step on 7B fp32 parameters on a CPU takes on the order of a second or more, and that often dominates the PCIe time. Ren et al. address it with an optimised CPU-Adam and one-step delayed parameter update (DPU). Add one sentence, and add one line naming **ZeRO-Infinity** (NVMe and parameter offload), which plan item 8 lists. q8's first distractor describes it, so the lesson should say it exists.
- **Z-m4. Figure placement.** `zero-stages` sits at the end of the ZeRO-2 card but shows all four stages, including ZeRO-3, which hasn't been taught yet. Move it to explainer[0] (after the three-stage list) or to explainer[4] (worked memory).
- **Z-m5. `zero-memory-vs-gpus` figure: the "80 GB H100" label is crossed** by the ZeRO-1/2 curves in every theme. In editorial, the DDP line and the 80 GB dashed line are both red-orange. Move the label right, to $P\approx64$ just above the dashed line, and give DDP a non-red colour.
- **Z-m6. Communication-time convention.** explainer[5] gives "roughly 70 ms and 105 ms" (factor 1), while ddp/collectives use 63/64 and f9 here uses 63/64 (77 GB, 0.19 s). All are "≈", but add "(large-$P$ limit)" after "70 ms and 105 ms" so the reader sees why the numbers differ from the 69 ms in `sys.collectives`.

### Strengths
Each stage's step sequence and memory formula is derived from what it shards. All numbers match the paper: 120/31.4/16.6/1.875 and 112/29.3/15.5/1.75, plus the 70B 286.6 GB that cannot fit at any P. The B7 reconciliation card (1.5× volume, same $B/P>C/W$) is exactly right and backed by the Scaling Book footnote. The f9 chained estimate is correct line by line: 3.25 GB states, 6.7 + 2.9 GB activations, 3.2 s step, 77 GB of traffic, hidden. The `zero3-dataflow` figure shows the prefetch offsets and the exposed first gather and last reduce-scatter correctly.
