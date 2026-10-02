#!/usr/bin/env python3
"""Validate all study content in PrepApp/Content. Usage: python3 tools/validate.py"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent / "PrepApp" / "Content"
ID_RE = re.compile(r"^[a-z0-9]+\.[a-z0-9-]+$")
ITEM_ID_RE = re.compile(r"^[a-z0-9-]+$")
LEVELS = {"core", "intermediate", "advanced"}
KINDS = {"paper", "blog", "book", "video", "course", "docs"}
COLORS = {"blue", "purple", "pink", "red", "orange", "yellow", "green", "teal", "indigo", "gray"}
BANNED_CHOICES = re.compile(r"\b(all|none|both) of the above\b|\bboth [a-d] and [a-d]\b", re.I)

FIGURES = ROOT / "figures"
THEMES = ("workbench", "terminal", "editorial")
FIG_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$", re.M)

errors: list[str] = []
warnings: list[str] = []


def err(where, msg):
    errors.append(f"{where}: {msg}")


def warn(where, msg):
    warnings.append(f"{where}: {msg}")


def text_field(where, obj, key, required=True):
    v = obj.get(key)
    if v is None:
        if required:
            err(where, f"missing '{key}'")
        return ""
    if not isinstance(v, (str, int, float)):
        err(where, f"'{key}' must be text, got {type(v).__name__}")
        return ""
    s = str(v)
    if required and not s.strip():
        err(where, f"'{key}' is empty")
    check_math(where + f".{key}", s)
    check_figures(where + f".{key}", s)
    return s


def rendered_len(s):
    """Approximate on-screen length: drop LaTeX commands, braces and delimiters."""
    s = re.sub(r"\\[a-zA-Z]+", "x", str(s))
    s = re.sub(r"[{}$^_\\]", "", s)
    return len(s.strip())


def check_figures(where, s):
    for ref in FIG_RE.findall(s):
        base = ref.strip().replace("/", "--")
        missing = [t for t in THEMES if not (FIGURES / f"{base}.{t}.pdf").exists()]
        if missing and not (FIGURES / f"{base}.pdf").exists():
            err(where, f"figure '{ref}' missing for {', '.join(missing)} (run tools/make_figures.py)")


def check_math(where, s):
    # Strip code before counting dollar signs.
    stripped = re.sub(r"```.*?```", "", s, flags=re.S)
    stripped = re.sub(r"`[^`]*`", "", stripped)
    stripped = stripped.replace("\\$", "")
    if stripped.count("$") % 2:
        err(where, "unbalanced $ delimiters")
    if stripped.count("{") != stripped.count("}"):
        warn(where, "unbalanced braces { } (check LaTeX)")
    for block in re.findall(r"\$\$(.*?)\$\$", stripped, flags=re.S):
        if len(block.strip()) > 110 and "aligned" not in block:
            warn(where, f"long display equation ({len(block.strip())} chars) without aligned; hard to read on a phone")


def main():
    areas_file = ROOT / "areas.yaml"
    if not areas_file.exists():
        print(f"missing {areas_file}")
        sys.exit(1)
    areas = yaml.safe_load(areas_file.read_text()) or []
    area_ids = set()
    for i, a in enumerate(areas):
        w = f"areas.yaml[{i}]"
        for k in ("id", "title", "icon", "color", "blurb", "order"):
            if k not in a:
                err(w, f"missing '{k}'")
        if a.get("color") not in COLORS:
            err(w, f"color must be one of {sorted(COLORS)}")
        if a.get("id") in area_ids:
            err(w, f"duplicate area id {a.get('id')}")
        area_ids.add(a.get("id"))

    topic_ids = {}
    prereq_refs = []
    counts = {"topics": 0, "explainer": 0, "quiz": 0, "cards": 0}
    for path in sorted(ROOT.rglob("*.yaml")):
        if path.name == "areas.yaml":
            continue
        rel = path.relative_to(ROOT)
        try:
            t = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            err(str(rel), f"YAML parse error: {e}")
            continue
        if not isinstance(t, dict):
            err(str(rel), "top level must be a mapping")
            continue
        tid = t.get("id", "")
        w = str(rel)
        if not ID_RE.match(str(tid)):
            err(w, f"bad id '{tid}' (want prefix.slug)")
        if path.stem != tid:
            err(w, f"filename must equal id ('{tid}.yaml')")
        if tid in topic_ids:
            err(w, f"duplicate topic id, also in {topic_ids[tid]}")
        topic_ids[tid] = w
        if t.get("area") not in area_ids:
            err(w, f"unknown area '{t.get('area')}'")
        text_field(w, t, "title")
        text_field(w, t, "summary")
        if t.get("level", "core") not in LEVELS:
            err(w, f"level must be one of {sorted(LEVELS)}")
        if not isinstance(t.get("order", 0), int):
            err(w, "order must be an integer")
        for p in t.get("prereqs") or []:
            prereq_refs.append((w, p))

        ex = t.get("explainer") or []
        if not ex:
            err(w, "needs at least one explainer card")
        for i, c in enumerate(ex):
            text_field(f"{w} explainer[{i}]", c, "title")
            text_field(f"{w} explainer[{i}]", c, "body")
            text_field(f"{w} explainer[{i}]", c, "source", required=False)

        quiz = t.get("quiz") or []
        longest = sum(
            1 for q in quiz
            if q.get("wrong") and rendered_len(q.get("correct", "")) > max(rendered_len(o) for o in q["wrong"])
        )
        if len(quiz) >= 6 and longest / len(quiz) > 0.4:
            warn(w, f"correct choice is the longest in {longest}/{len(quiz)} MCQs (rendered length); vary it")

        seen = set()
        for i, q in enumerate(quiz):
            qw = f"{w} quiz[{i}]"
            qid = str(q.get("id", ""))
            if not ITEM_ID_RE.match(qid):
                err(qw, f"bad item id '{qid}'")
            if qid in seen:
                err(qw, f"duplicate item id '{qid}'")
            seen.add(qid)
            text_field(qw, q, "prompt")
            correct = text_field(qw, q, "correct")
            wrong = q.get("wrong") or []
            if not isinstance(wrong, list) or len(wrong) < 1:
                err(qw, "'wrong' must be a list with ≥1 option")
                wrong = []
            elif len(wrong) < 3:
                warn(qw, f"only {len(wrong)} wrong option(s); 3 recommended")
            for j, opt in enumerate(wrong):
                if not isinstance(opt, (str, int, float)):
                    err(qw, f"wrong[{j}] must be text")
                    continue
                check_math(f"{qw}.wrong[{j}]", str(opt))
                if str(opt).strip() == correct.strip():
                    err(qw, f"wrong[{j}] duplicates the correct answer")
            for opt in [correct, *map(str, wrong)]:
                if BANNED_CHOICES.search(opt):
                    err(qw, f"choice '{opt}' doesn't survive shuffling")
            text_field(qw, q, "explanation", required=False)
            text_field(qw, q, "source", required=False)
            if not q.get("explanation"):
                warn(qw, "no explanation")

        for i, c in enumerate(t.get("cards") or []):
            cw = f"{w} cards[{i}]"
            cid = str(c.get("id", ""))
            if not ITEM_ID_RE.match(cid):
                err(cw, f"bad item id '{cid}'")
            if cid in seen:
                err(cw, f"duplicate item id '{cid}' (ids are shared across quiz and cards)")
            seen.add(cid)
            if c.get("type", "flash") not in ("flash", "open"):
                err(cw, "type must be 'flash' or 'open'")
            text_field(cw, c, "front")
            text_field(cw, c, "back")
            text_field(cw, c, "source", required=False)

        for i, r in enumerate(t.get("reading") or []):
            rw = f"{w} reading[{i}]"
            text_field(rw, r, "title")
            url = text_field(rw, r, "url")
            if url and not url.startswith("http"):
                err(rw, "url must start with http")
            if r.get("kind", "paper") not in KINDS:
                err(rw, f"kind must be one of {sorted(KINDS)}")

        counts["topics"] += 1
        counts["explainer"] += len(ex)
        counts["quiz"] += len(t.get("quiz") or [])
        counts["cards"] += len(t.get("cards") or [])

    for w, p in prereq_refs:
        if p not in topic_ids:
            warn(w, f"prereq '{p}' does not exist (yet)")

    for m in warnings:
        print(f"⚠️  {m}")
    for m in errors:
        print(f"❌ {m}")
    print(
        f"\n{counts['topics']} topics · {counts['explainer']} explainer cards · "
        f"{counts['quiz']} MCQs · {counts['cards']} flashcards · "
        f"{len(errors)} errors · {len(warnings)} warnings"
    )
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
