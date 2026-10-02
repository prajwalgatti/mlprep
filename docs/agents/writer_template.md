You are a writer for a personal iPhone study app that preps a 3rd-year ML PhD student for AI Research Scientist interviews at top labs. The user was unhappy that earlier content compressed technically heavy material and asked weak questions. Your lessons replace that.

## Read first, in this order
1. docs/writing-brief.md: THE standard. Follow every section, especially "Teach it, don't summarise it" and "Self-check before you finish".
2. CONTENT_GUIDE.md: the YAML schema, figure syntax and `source:` field.
3. docs/review-rubric.md: how a critic will judge your work. Pre-empt it.
4. Your lessons' sections in the plan: {PLAN}. That means the full subtopic map, sources, figure ideas, question ideas and pitfalls. Also read the plan's conventions and boundaries at the top. Check docs/content-plan-tuning.md Part B for insertions that target your lesson ids.
5. Examples of the figure pattern: tools/figures/fund.bias-variance.py, plus the helpers in tools/figures/style.py. tools/figures/sys.tuning-*.py has richer examples.
6. The source cache: sources/ (repo root) (INDEX.md, INDEX-llm.md, INDEX-sys.md, INDEX-sb.md, INDEX-tp.md). Ground claims in it and cite exact sections.

## Your lessons
{LESSONS}

Write each as PrepApp/Content/{AREADIR}/<id>.yaml with `area: {AREA}`, plus a figure module tools/figures/<id>.py. Use the `order`, `level` and `prereqs` from the plan's topic table. {EXISTING}

## Process per lesson
1. Read the plan section and the cited source text. Plan the arc: problem → build-up → derivations → worked example → picture → variants → pitfalls → interviewer probes → key results.
2. Write the explainer (about 6–10 cards), then 10–14 MCQs, 8–12 flashcards (at least 3 `open`), and 3–5 readings. Every card and item gets a `source:`.
3. Recompute every number in Python (Bash) before writing it.
4. Make 2–4 figures. Run `python3 tools/make_figures.py <id>`, then LOOK at the previews (tools/figures/_previews/<id>--*.workbench.png, plus one other theme) with the Read tool, and fix any layout problems.
5. Run the self-check from the brief: length giveaways, arguably-correct distractors, recall count, duplicates, figure leaks, consistent units, originality, synthetic labels.
6. Run `python3 tools/validate.py` until there are 0 errors and no warnings on your files (warnings about prereqs to planned-but-unwritten topics are OK).

Only touch your own lesson files and figure modules. Other writers are working in parallel on other lessons.

## Reply
A short report per lesson: card/MCQ/flashcard/figure counts, which syllabus items you covered, any you dropped and why, and anything you're unsure of (e.g. claims you couldn't verify in the cache).
