# Content Guide

All study content lives in `PrepApp/Content/`. Each topic is one YAML file, and the app finds every file on its own.
**To add a topic:** drop a new `.yaml` file in, run `python3 tools/validate.py`, and rebuild. No Swift changes needed.

```
PrepApp/Content/
  areas.yaml                       # list of areas (order, icon, color)
  fundamentals/fund.bias-variance.yaml
  llms/llm.self-attention.yaml
  ...
```

## Rules
- **Filename = topic `id`** plus `.yaml`. IDs are globally unique, lowercase, and shaped like `<areaPrefix>.<slug>`.
- **Item `id`s (q1, f1, …) must never change once published.** Your review history is keyed by `topicID/itemID`.
  You can freely edit an item's text, add new items, or delete items. Deleted items' history is just ignored.
- To **force re-learning** of an item after a big rewrite, bump its `rev` (default 1 → 2). That resets its schedule.
- Prefer YAML block scalars (`|`) for any text containing math, colons, or quotes. Inside them, write LaTeX with single backslashes.

## Topic file schema
```yaml
id: fund.bias-variance          # required, matches filename
area: fundamentals              # required, must exist in areas.yaml
title: Bias–Variance Tradeoff   # required
summary: Why error splits into bias, variance and noise — and how model capacity trades them off.
order: 10                       # sort order within the area (gaps are fine)
level: core                     # core | intermediate | advanced
tags: [generalization, theory]
prereqs: []                     # other topic ids (shown as hints, not enforced)

explainer:                      # 3–6 swipeable cards, read in order
  - title: Intuition
    body: |
      Markdown body. Inline math $\hat f(x)$, display math on its own lines:

      $$\mathbb{E}\big[(y-\hat f(x))^2\big] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$$

      - bullet lists work
      - **bold**, *italic*, `inline code`
  - title: What interviewers ask
    body: |
      ...

quiz:                           # MCQs: shown in the lesson, then they join spaced repetition
  - id: q1
    prompt: |
      A model has low training error but high validation error. Most likely:
    correct: High variance (overfitting)
    wrong:
      - High bias (underfitting)
      - Irreducible noise dominates
      - The learning rate is too low
    explanation: |
      Optional but strongly encouraged. Shown after answering.
    rev: 1                      # optional

cards:                          # flashcards: self-graded Again / Hard / Good / Easy
  - id: f1
    front: Write the bias–variance decomposition of expected squared error.
    back: |
      $$\mathbb{E}[(y-\hat f)^2] = (\mathbb{E}\hat f - f)^2 + \mathrm{Var}(\hat f) + \sigma^2$$
  - id: f2
    type: open                  # "say it out loud" prompt: answer verbally, then reveal
    front: Explain to an interviewer why ensembling reduces variance but not bias.
    back: |
      Model answer ...

reading:                        # optional
  - title: "The Elements of Statistical Learning, §7.3"
    url: https://hastie.su.domains/ElemStatLearn/
    kind: book                  # paper | blog | book | video | course | docs
    note: Optional one-liner on why it's worth reading.
```

### Choices
- MCQ choices are **shuffled at display time**, so never write "All of the above" or "Both A and B".
- Use 3 wrong answers when you can. All options should be plausible and similar in length.

### Supported markdown in bodies
Paragraphs, `-` bullet lists, `1.` numbered lists, fenced code blocks (```` ```python ````), display math `$$…$$`
on its own line(s), inline math `$…$`, **bold**, *italic*, `code`, and [links](https://…).
Headings aren't supported inside bodies; use the card `title`.

**Figures:** put a line on its own, `![Caption, can contain $math$](fund.svm/margin)`, in any body, prompt, explanation or card.
Figures are drawn in Python (`tools/figures/<topic-id>.py`, one `@register(topic, name)` function per figure), rendered for
every theme by `python3 tools/make_figures.py`, and bundled as `PrepApp/Content/figures/<topic>--<name>.<theme>.pdf`.
In the app, tap a figure to zoom.

**Sources:** explainer cards, MCQs and flashcards take an optional `source:` (e.g. `Goodfellow DL §8.3.2; Bishop PRML §3.2`),
shown as a small citation line.
Display equations wider than the screen scroll sideways. For long derivations, split them across lines with
`\begin{aligned} ... \\ ... \end{aligned}` so they stay readable on a phone.

## areas.yaml schema
```yaml
- id: fundamentals
  title: ML Fundamentals
  icon: function                # SF Symbol name
  color: blue                   # blue|purple|pink|red|orange|yellow|green|teal|indigo|gray
  blurb: Learning theory, classical models, probability, optimization, DL basics.
  order: 1
```

## Fixing content from the app
In the app, tap **Flag** on any card you think is wrong. Then go to **Saved → Flagged → Export**, which produces a Markdown list
with `topicID/itemID` references. Paste it into Claude to get fixes, then mark the flags resolved.
