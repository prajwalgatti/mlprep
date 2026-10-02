#!/usr/bin/env python3
"""Render content figures for every app theme.

Each module tools/figures/<topic-id>.py registers draw functions with @register(topic, name).
Output: PrepApp/Content/figures/<topic>--<name>.<theme>.pdf (vector, transparent background),
plus a PNG contact sheet per topic in tools/figures/_previews/ for checking the result by eye.

Usage:
  python3 tools/make_figures.py                 # all topics
  python3 tools/make_figures.py fund.svm ...    # only these topics
Reference a figure from content as a line of its own:  ![Caption text](fund.svm/margin)
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from figures import style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "PrepApp" / "Content" / "figures"
PREVIEWS = Path(__file__).resolve().parent / "figures" / "_previews"
BG = {"workbench": "#151821", "terminal": "#0F110E", "editorial": "#211C16"}   # theme surface colors


def load_modules(only):
    for path in sorted((Path(__file__).resolve().parent / "figures").glob("*.py")):
        if path.stem in ("style", "__init__"):
            continue
        if only and path.stem not in only:
            continue
        spec = importlib.util.spec_from_file_location(f"figures.{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def main():
    only = set(sys.argv[1:])
    load_modules(only)
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    by_topic = {}
    failures = 0
    for spec in style.REGISTRY:
        if only and spec.topic not in only:
            continue
        for theme, p in style.PALETTES.items():
            style.apply(p)
            try:
                fig = spec.draw(p)
            except Exception as e:  # keep going so one broken figure doesn't block the rest
                print(f"❌ {spec.topic}/{spec.name} [{theme}]: {e}")
                failures += 1
                break
            fig.savefig(OUT / f"{spec.topic}--{spec.name}.{theme}.pdf")
            png = PREVIEWS / f"{spec.topic}--{spec.name}.{theme}.png"
            fig.savefig(png, dpi=200, transparent=False, facecolor=BG[theme])
            plt.close(fig)
            by_topic.setdefault(spec.topic, []).append(png)
        print(f"wrote {spec.topic}/{spec.name}")
    print(f"\npreviews in {PREVIEWS.relative_to(ROOT)} · {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
