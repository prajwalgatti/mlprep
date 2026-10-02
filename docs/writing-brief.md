# Writing brief: study content (v2)

This brief is shared by writer and critic agents. The format is in [CONTENT_GUIDE.md](../CONTENT_GUIDE.md).
The per-topic plans, with their sources and the technical core each topic must cover, are in
[content-plan.md](content-plan.md) (general ML and generative models), [content-plan-llms.md](content-plan-llms.md) and
[content-plan-applied.md](content-plan-applied.md). The Scaling Book special edition has its own plan in
[content-plan-scalingbook.md](content-plan-scalingbook.md). Also check the insertions table in
[content-plan-tuning.md](content-plan-tuning.md): it lists Tuning Playbook points that belong in your topic.

**The reader** is a third-year ML PhD student preparing for research scientist interviews at top labs.
They have solid general maths and ML maturity: probability, linear algebra, calculus, training neural nets.
But **assume they have not met this specific topic before.** Their own words: "If it's a topic I don't know
(like ELBO) I don't want to just skim through high level notes that assume familiarity."

The previous version disappointed them for two reasons. It **stated results instead of deriving them**,
and its questions tested recall instead of understanding.

## Teach it, don't summarise it
The explainer is a self-contained lesson that someone new to the topic can follow with no outside help,
and that leaves them able to reproduce the key derivations at a whiteboard.
- **Start from the problem.** Open with what we are trying to do and why the obvious approach fails.
  For example, "we want $\log p(x)$ but the integral over $z$ is intractable".
  The concept should arrive as the answer to a question the reader already has.
- **Build in order.** Introduce each idea before it's used. Define every symbol the first time it appears.
  Say in words what each term of an important equation means. If a topic depends on a prerequisite the reader might be
  shaky on (Jensen's inequality, change of variables, the Bellman equation…), give a short refresher inline rather
  than assuming it.
- **Descriptive, not padded.** Each sentence should add a definition, a step, a reason, an example or a caveat.
  Cut throat-clearing ("It is important to note that…"), restatements of the previous paragraph, and generic praise
  ("a powerful technique"). Being thorough is good; being repetitive is not.
- **Explain the why at each step.** Say why we take this step, why the term vanishes, why this approximation is acceptable.
  Then say what the result means: an interpretation of each term and the intuition behind it.
- **Concrete before abstract, where it helps.** A tiny worked example or a picture often belongs *before* the general
  derivation, not only after it.
- **Cover the whole topic, not a few highlights.** Work through the plan's full subtopic map for your topic: every
  important derivation, variant, design choice and failure mode an interviewer could reasonably probe. If you drop
  a subtopic, say why in your report.
- **Keep each lesson to about 6–10 cards, roughly a 5–10 minute read**, because the user studies in short breaks.
  Big areas are split into a sequence of topic files linked by `prereqs`, so depth comes from the sequence, not from
  one bloated lesson. Never compress a derivation to fit. If a topic overflows, tell the coordinator so it can be split.
- **Self-test:** could a smart reader who has never seen this topic follow every line without opening another source?
  If they'd have to look something up, add it.

## Principles
1. **Derive, don't assert.** Every non-obvious result in an explainer is either derived step by step, or, if a derivation
   doesn't fit, given with a one-line reason why it holds and a citation. "One can show that…" is not allowed.
   No step may be skipped that an interviewer would ask about.
2. **Grounded in sources.** Use the cached texts in the source cache (see `INDEX.md`, `INDEX-llm.md` and `INDEX-sys.md` there) and check claims against them.
   Every explainer card, MCQ and flashcard gets a `source:` naming the section you actually checked, for example
   `Goodfellow DL §8.3.2`, `Bishop PRML §3.2`, `ESL §7.3`, `Murphy PML1 §4.5.2`, `Boyd CVX §9.5`, `SSBD §5.1`,
   `MacKay ITILA §2.6`, `d2l §11.10`, `CS229 notes §1.2`, or `Kingma & Ba 2015 §2.1`. Separate several with "; ".
   Where sources disagree on conventions, say so in the text.
3. **Write in your own words.** Do not copy sentences or passages from the sources. Notation can follow the main source.
4. **Correctness over coverage.** Recompute every number in Python (via Bash) before writing it down. If you're unsure of
   a claim and can't verify it, leave it out.

## Explainer (about 6–10 cards per topic)
A suggested arc, to adapt per topic:
1. Motivation and setup: the problem, why naive approaches fail, the objects and notation
2. Core derivation(s), possibly split across 2–3 cards
3. A worked numerical example with small numbers the reader can follow
4. Geometric or intuitive picture, and why it matters
5. Variants, connections, and how it is used in modern deep learning
6. Pitfalls, edge cases, and common wrong answers
7. "What interviewers probe": the follow-up chain, with short answers
8. "Key results": a compact recap card of the 4–8 equations and facts to remember, for quick revision later.
   This is the only place where results appear without their derivation.

Each card should take one or two phone screens: about 150–300 words plus equations. Split cards rather than overfilling them.

**Phone math rules (important):**
- Display equations go on their own line as `$$…$$`. Keep each line under ~45 characters of rendered width.
  Split derivations with `\begin{aligned} a &= b \\ &= c \end{aligned}` inside `$$`, one step per line.
  `tools/validate.py` warns about long unaligned equations; aim for zero warnings.
- Put a short justification in prose between derivation chunks ("using independence, the cross term vanishes:").
- Use only standard MathJax: no custom macros, no `align` environment (use `aligned` inside `$$`), and no `\tag`.
- Inline math `$…$` is fine in prompts, choices and explanations.
- There are no markdown tables. Use bullet lists, or a figure for a real table.
- YAML: quote any one-line value that contains `: ` (e.g. a `source:` with a section title). Use `|` blocks for
  anything with math.

## Figures (where they genuinely help; typically 2–4 per topic)
A figure belongs wherever a picture carries the idea better than words. Examples:
- Curves: loss vs capacity, activation functions and their derivatives, ROC/PR curves, learning-rate schedules,
  scaling-law fits.
- Geometry: SVM margins, PCA axes, gradient-descent paths on contours, KL asymmetry on two densities.
- Process or architecture diagrams: LSTM cell, transformer block, FlashAttention tiling, pipeline schedules,
  sharding layouts, the forward diffusion process.

Don't add decorative figures.

- **Write** a module `tools/figures/<topic-id>.py` with functions decorated `@register("<topic-id>", "<name>")`.
  Each function takes a palette `p` and returns a figure. Copy the pattern in `tools/figures/fund.bias-variance.py`.
  Diagram helpers (`box`, `arrow`, `blank`) are in `tools/figures/style.py`.
- **Never hard-code colours.** Use `p.fg`, `p.muted`, `p.faint`, `p.surface`, `p.accent`, `p.label`, `p.c(i)`,
  `p.good` and `p.bad`. Every figure is rendered once per app theme.
- **Size for a phone.** Use `figure(height)` (3.6 in wide), keep text at 7.5–9 pt, prefer direct labels over legends,
  and avoid dense tick labels. Use synthetic data computed with numpy, generated from the real formulas where
  possible (e.g. actually run k-means, or actually compute the Gaussian KL).
- Keep any single line of figure text under about 50 characters. In the terminal theme's monospace font, a wide label
  makes matplotlib shrink the whole plot.
- **Render** with `python3 tools/make_figures.py <topic-id>`. Then **look at the PNG previews** in
  `tools/figures/_previews/` with the Read tool, at least one theme per figure, and fix overlaps, clipping and
  illegible text before moving on.
- **Reference** it from any body, prompt or explanation on a line of its own: `![Short caption with optional $math$](<topic-id>/<name>)`.
  Captions say what to notice, not just what is shown.
- **Figure-based MCQs are encouraged.** Example: "Which curve is the derivative of GELU?" with a figure in the prompt.

## MCQs (10–14 per topic)
Use a mix of these types, with at most 2 pure-definition questions:
- **Compute**: a small numeric problem, such as a gradient, variance, or the number of parameters.
- **Derivation step**: which step is invalid, or what is the next line.
- **Predict the behaviour**: what happens to X if we change Y?
- **Which is false**: plausible statements with one subtle error.
- **Compare**: when does A beat B, and why?
- **Edge case or failure mode**: what breaks, and when?

Choices:
- Exactly one choice is unambiguously correct. The other three should be distractors drawn from **real misconceptions**.
- All four choices should have similar length and style.
- Don't make the correct choice the longest or the most hedged one. Avoid "always"/"never" giveaways.
- Choices are shuffled at display time, so never use "all of the above" or "both A and B".

The `explanation` says why the right answer is right **and** why the most tempting distractor is wrong, in 2–5 sentences.

## Flashcards (8–12 per topic)
- `type: flash` cards are for things worth memorising: key equations, definitions with their conditions, and canonical numbers.
  The front is a precise question; the back is short and complete.
- `type: open` cards (at least 3 per topic) are interview-style prompts: "Derive…", "Explain to an interviewer…", "Compare…".
  The back is a structured model answer an excellent candidate would give, including the derivation skeleton.

## Reading (3–5 per topic)
Point to exact sections with working public URLs, and add a note on what each one is good for.

## IDs and revisions (existing topics)
- When rewriting an existing topic file, keep an item's `id` if it still asks essentially the same thing.
  If you change its meaning substantially, give it a new id instead.
- Don't reuse an id for different content. New ids continue the sequence (`q15`, `f13`, …).
- Don't add `rev` unless the item keeps its id and its answer changed.

## Self-check before you finish (the most common critic findings)
- **Length giveaway:** for each MCQ, compare rendered lengths (`validate.py` warns if the correct choice is the longest in more than 40% of MCQs). The correct choice must not be the longest more
  than about a third of the time across the topic, and should never be much longer than the distractors.
- **Arguably-correct distractors:** for each distractor, ask "could a careful expert defend this?" If so, fix it.
  Check distractors against the lesson's own text: they must not contradict something you taught.
- **Recall vs reasoning:** at most 2 pure-recall MCQs per topic. Turn spec or definition questions into
  compute, predict or which-is-false questions.
- **Duplicates:** no two items should test the same fact.
- **Figure leaks:** a figure used in a question stem must not annotate the answer.
- **Internal consistency:** numbers must agree across cards (e.g. FLOPs vs multiply-adds, GB vs GiB, one-way vs
  bidirectional bandwidth). State units and conventions explicitly.
- **Originality:** reread anything that tracks a source's wording closely and rephrase it in your own words.
- **Synthetic figures say so:** if a figure is illustrative rather than real data, its caption says "illustrative"
  or "synthetic", and it must not borrow real numbers from a paper in a way that suggests it is that paper's data.
- **One definition per term:** when sources define a term differently (e.g. "critical batch size"), pick one,
  name the other explicitly, and never switch between them silently.
- **Scope every result:** state the assumptions it holds under (model, optimizer, regime) and where it is known to
  fail, e.g. "η<2/λ_max is the quadratic/GD threshold; with momentum it's (2+2β)/η; for Adam it's unclear".
  An empirical finding is scoped to the setting it was measured in.
- **Define every term of art** used in a question or explanation, even in passing, e.g. "low-discrepancy".
- **Figures that show mechanisms must actually show them,** e.g. data-flow arrows, ordering and skew, not just boxes.

## Done means
- `python3 tools/validate.py` reports 0 errors and no warnings for your files.
- Every card and item has a `source`.
- Figures are rendered, inspected and referenced, and `validate.py` confirms the files exist.
- The plan's "technical core" bullets for the topic are covered, or you explain why one isn't.
