# Agent guide

This is an offline SwiftUI iPhone app (iOS 17+) for ML research-scientist interview prep.
**Almost all contributions are content**: YAML files under `PrepApp/Content/`. You shouldn't need to touch Swift.

## Adding or editing a topic
1. Read [CONTENT_GUIDE.md](CONTENT_GUIDE.md) for the schema, math/figure syntax and the `source:` field.
   Then read [docs/writing-brief.md](docs/writing-brief.md) for the quality bar.
2. Look at `PrepApp/Content/areas.yaml` for the valid areas, and at 1–2 finished topics in the same area as style references
   (e.g. `PrepApp/Content/applied/sys.ddp.yaml`, `PrepApp/Content/genmodels/gen.gans.yaml`).
3. Check that the topic doesn't already exist: `ls PrepApp/Content/*/`. The per-area syllabi in `docs/content-plan*.md`
   list planned topics, with their ids, order, prereqs and sources. Use them if your topic appears there.
4. Write `PrepApp/Content/<area-dir>/<id>.yaml`. The filename must equal `id`.
5. Run `python3 tools/validate.py` (needs `pip3 install pyyaml`). You need **0 errors** before you're done.
6. Optional figures: add `tools/figures/<id>.py` (copy the pattern in `tools/figures/fund.bias-variance.py`),
   run `python3 tools/make_figures.py <id>`, and look at the previews in `tools/figures/_previews/`.

## Hard rules
- **Never change or reuse an existing item `id`** (`q1`, `f3`, …). User progress is keyed by `topicID/itemID`.
  Add new items with new ids. To force re-learning after a rewrite, bump `rev`.
- **Derive, don't state.** Lessons teach from the problem, through the derivation, to the result, for a reader new to the topic.
  Questions should test understanding, not recall. Recompute every number you write (run the code).
- **Ground claims in real sources** (papers, textbooks, official docs) and cite them in `source:`.
  `sources/INDEX*.md` list the sources used so far, with URLs. The text cache itself isn't in git, so fetch what you need.
- Keep lessons to about 6–10 explainer cards. Split big topics into several files linked by `prereqs`.

## Reviewing (optional but recommended)
For a second pass, have a separate agent critique the topic against [docs/review-rubric.md](docs/review-rubric.md).
Prompt templates for a writer→critic loop are in `docs/agents/`. The output of that process is in `docs/reviews/`.

## Code changes
```
PrepApp/  App · Models (content + YAML loader, SwiftData) · SRS (FSRS-5, queues) · Components (Theme.swift, RichText)
          Features (Today, Topics, Session, Exam, Stats, Saved, Settings) · Fonts (OFL .ttf) · Content
tools/    validate.py, make_figures.py (+ figures/<id>.py), make_icons.py
docs/     content plans, writing brief, review rubric, reviews
Config/   Signing.xcconfig + git-ignored Local.xcconfig (never commit it: it holds a personal Team ID)
```
- Files added to existing folders are picked up automatically. Run `xcodegen generate` only after editing `project.yml`.
- Themes are presets in `Components/Theme.swift`. To add a font, put the `.ttf` in `Fonts/` and add a case to `FontFamily`.
- Tests: `xcodebuild -project PrepApp.xcodeproj -scheme PrepApp -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test`
  (prefix `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` if xcodebuild complains).
