# ML Interview Prep — iPhone App Plan

Goal: replace break-time scrolling with 2–5 minute interview-prep sessions for AI Research Scientist roles.
Starting point: Silvia Sapora's topic list (https://silviasapora.github.io/blog/ml-interviews.html).

## Status (2026-10-01)
- **Built:** the full app, running against a small fixed content set (9 topics: 6 Fundamentals, 2 LLMs, 1 ML Coding).
  It covers lessons, MCQs, flashcards, "say it out loud" cards, FSRS reviews, mock exams, stats, bookmarks, notes,
  flag & export, and a Settings screen with content diagnostics.
- **Tests:** FSRS + content + parser unit tests, and a UI smoke test (lesson → review → exam → all tabs).
- **Next:** use it for a few days and flag anything clunky. Then grow content area by area. Later: widget,
  reminders, remote content updates (the loader already reads `Documents/ContentOverride/`).

## Open decision: external knowledge (deferred)
The current content was drafted from model knowledge. The goal is to bring in external reputable sources
(textbooks, course notes, original papers), for two reasons:
1. **Verifiability:** a per-card `sources` field with citations, shown in the app.
2. **Diversity:** model-generated content may reflect one "mode" of explanation. Synthesizing from several distinct
   sources per topic brings different framings, examples, intuitions and question styles.

Options to weigh later:
- agents that read 2–3 different references per topic before drafting
- requiring each card to cite its source
- a re-grounding pass over the existing topics

## Decisions
- **Platform:** Native SwiftUI + SwiftData, iOS 17+, fully offline, no backend.
- **Signing:** Free Apple ID for now (re-install from Xcode every 7 days). Can upgrade to paid later for TestFlight.
- **MVP content area:** ML Fundamentals.
- **Areas:** Silvia's 6 (RL, LLMs, Generative Modeling, Applied ML, ML Fundamentals, Linear Algebra) + Probability & Stats (inside Fundamentals for now) + **Vision & Video** + **ML Coding drills**.
- One-time setup: `sudo xcode-select -s /Applications/Xcode.app`

## Core loop
Home → **Start session**: a mixed queue of due reviews plus one new mini-lesson. You can quit anytime without losing progress.

Topic flow: **Explainer** (3–6 swipe cards: intuition → math → "what interviewers ask") → **Quiz** (6–10 items) → flashcards join the spaced-repetition deck.

## Question types
| Type | v | Notes |
|---|---|---|
| MCQ | 1 | explanation shown after answering |
| Flashcard | 1 | self-graded Again/Hard/Good/Easy |
| Cloze | 2 | fill in a blank in an equation/statement |
| Code: spot-the-bug / fill-the-line / tensor shape | 3 | for the ML Coding area |
| "Say it out loud" | 3 | answer verbally, compare with a model answer, grade yourself |

## Features by phase
**Phase 1 (MVP)**
- Area → topic browser with a mastery ring per topic (new / learning / reviewing / mastered)
- Lesson flow (explainer → quiz → cards unlocked)
- Spaced repetition with FSRS; missed quiz questions go into the review queue
- Daily session queue, streak, daily goal
- Bookmarks, per-topic notes, "flag as wrong" button (exportable list for fixes)
- LaTeX rendering for math, monospaced code blocks
- ML Fundamentals content complete

**Phase 2**
- Remaining areas' content
- Mock exam: timed, drawn from many topics, presets (general / RL-heavy / applied-systems-heavy); results list weakest topics
- Stats: activity heatmap, accuracy by area, 5 weakest topics
- Reading list per topic, search

**Phase 3**
- Cloze, code and "say it out loud" question types; ML Coding area
- Widget ("N cards due"), daily reminder notification
- (optional) application tracker, "ask Claude about this card"

## Architecture
```
PrepApp/
  App/            PrepApp.swift, RootTabView (Today · Topics · Exam · Stats · Saved)
  Models/         Content (Codable, read-only from YAML) + UserState (SwiftData)
  SRS/            FSRS scheduler (pure Swift, unit-tested)
  Features/       Today, TopicBrowser, Lesson, Quiz, Review, Exam, Stats, Bookmarks
  Components/     MathText, CodeBlock, MasteryRing, CardStack
  Content/        areas.yaml, <area>/<topic-id>.yaml
tools/validate.py   schema, LaTeX-delimiter and duplicate-ID checks
```
Content is read-only YAML (see CONTENT_GUIDE.md) bundled with the app; user progress lives in SwiftData, keyed by stable item IDs, so content updates never wipe progress.

### Content format
One YAML file per topic: explainer cards, MCQs (`correct` + `wrong`, shuffled at display), flashcards
(`flash` / `open`), and reading links. The full schema and rules are in [CONTENT_GUIDE.md](CONTENT_GUIDE.md).

### User state (SwiftData)
- `CardState(itemID, stability, difficulty, due, lastReview, reps, lapses)`
- `TopicProgress(topicID, explainerDone, quizBestScore, bookmarked, note)`
- `ReviewLog(itemID, date, rating, durationMs)`, which drives stats and the streak
- `Flag(itemID, reason, date)`

## MVP content: ML Fundamentals (~45 topics)
- **Learning theory:** supervised vs unsupervised, bias–variance, over/underfitting, curse of dimensionality, no free lunch, cross-validation, regularization (L1/L2/dropout), early stopping
- **Classical models:** linear regression, logistic regression, kNN, SVMs, decision trees, bagging, boosting, ensembles, k-means, dimensionality reduction (PCA)
- **Probability & stats:** Bayes' theorem, MLE vs MAP, expectation/variance/covariance, PDF/PMF, entropy & cross-entropy, KL divergence, JS divergence, confidence intervals
- **Optimization:** GD/SGD, momentum, Adam/AdamW/Adagrad, Newton's method & second-order methods, convexity
- **Deep learning basics:** backprop, activation functions, loss functions, weight initialization, exploding/vanishing gradients, BatchNorm/LayerNorm/RMSNorm, CNNs, RNNs/LSTMs, autoencoders, Gumbel-Softmax, data whitening
- **Evaluation:** precision/recall/F1, ROC-AUC/PR-AUC
- **Generalization beyond i.i.d.:** transfer learning, domain adaptation, few/zero-shot

Content workflow: Claude drafts each topic as YAML, then `tools/validate.py` checks it; you review while studying and flag errors, and the flags get fixed in batches.

## Later areas (topic lists from Silvia's post, plus additions)
- **Vision & Video:** ViT, CLIP/contrastive learning, DINO/MAE, DiT, latent diffusion, video diffusion, VQ-VAE/tokenizers, FID/FVD/CLIP-score, optical flow basics
- **ML Coding:** attention (self/causal/cross), MHA shapes, flash attention idea, attention backward, MLP forward/backward, training loop bugs, PyTorch/JAX gotchas
