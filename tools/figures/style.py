"""Shared figure style: one palette per app theme, matching PrepApp/Components/Theme.swift.

Each figure is a function `draw(p)` that receives a Palette and returns a matplotlib Figure.
It is rendered once per theme, so never hard-code colors; use the palette fields.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence font-subsetting timestamp noise
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "PrepApp" / "Fonts"

# Phone layout: figures are shown ~360 pt wide. Width 3.6 in at 100 pt/in ⇒ 9 pt text renders ≈ 12.5 pt.
WIDTH = 3.6


@dataclass(frozen=True)
class Palette:
    theme: str
    fg: str           # primary text and lines
    muted: str        # secondary text, axes, tick labels
    faint: str        # gridlines, background shapes
    surface: str      # filled boxes (diagram nodes)
    accent: str       # the main highlighted thing
    label: str        # annotations / callouts
    series: tuple     # categorical colors for curves, in order
    good: str
    bad: str
    font: str         # font family name for text
    mono: str         # font family for code-ish labels
    rounded: bool = True   # diagram boxes: rounded corners or square

    def c(self, i):
        """i-th categorical color."""
        return self.series[i % len(self.series)]


PALETTES = {
    "workbench": Palette(
        theme="workbench", fg="#E6E8EE", muted="#9AA0AE", faint="#262B36", surface="#1B1F2A",
        accent="#8C9BFF", label="#E8A23A",
        series=("#8C9BFF", "#E8A23A", "#4FC1B0", "#F07AAE", "#5CC98A", "#F0616D"),
        good="#3FB97A", bad="#F0616D", font="Inter", mono="JetBrains Mono"),
    "terminal": Palette(
        theme="terminal", fg="#D5DECB", muted="#8C9683", faint="#2E3529", surface="#151913",
        accent="#B8F15A", label="#F5C451",
        series=("#B8F15A", "#F5C451", "#6BC7FF", "#FF6B5B", "#9FE3C4", "#D5DECB"),
        good="#B8F15A", bad="#FF6B5B", font="IBM Plex Mono", mono="IBM Plex Mono", rounded=False),
    "editorial": Palette(
        theme="editorial", fg="#EFE6D8", muted="#A3968A", faint="#3A3229", surface="#2A241D",
        accent="#E0714F", label="#E8B04F",
        series=("#E0714F", "#8DB3D6", "#8FC48A", "#E8B04F", "#BE9CC7", "#7DBFB2"),
        good="#8FC48A", bad="#E86A5A", font="Space Grotesk", mono="JetBrains Mono"),
}

_fonts_loaded = False


def _load_fonts():
    global _fonts_loaded
    if _fonts_loaded:
        return
    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    _fonts_loaded = True


def apply(p: Palette):
    """Set matplotlib rcParams for a palette. Called before each draw."""
    _load_fonts()
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": p.font, "font.size": 9,
        "mathtext.fontset": "cm",
        "text.color": p.fg, "axes.labelcolor": p.muted, "axes.edgecolor": p.muted,
        "axes.titlecolor": p.fg, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.color": p.muted, "ytick.color": p.muted, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.grid": False, "grid.color": p.faint, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 7.5,
        "axes.prop_cycle": matplotlib.cycler(color=list(p.series)),
        "pdf.fonttype": 42,
    })


def figure(height=2.4, width=WIDTH, **kw):
    """New figure sized for the phone. Returns (fig, ax) like plt.subplots."""
    return plt.subplots(figsize=(width, height), constrained_layout=True, **kw)


def box(ax, xy, w, h, text, p: Palette, color=None, fill=None, fontsize=8, mono=False, lw=1.0, **text_kw):
    """Diagram node: a (rounded) rectangle with centered text. xy is the lower-left corner in data coords."""
    from matplotlib.patches import FancyBboxPatch
    style = "round,pad=0.02,rounding_size=0.06" if p.rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=style, linewidth=lw,
                                edgecolor=color or p.muted, facecolor=fill or p.surface,
                                linestyle="--" if (p.theme == "terminal" and color is None) else "-"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=text_kw.pop("text_color", p.fg), family=p.mono if mono else p.font, **text_kw)


def arrow(ax, start, end, p: Palette, color=None, lw=1.0, style="-|>", **kw):
    """Diagram edge from start to end (data coords)."""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9, linewidth=lw,
                                 color=color or p.muted, shrinkA=2, shrinkB=2, **kw))


def blank(ax, xlim, ylim):
    """Turn an axes into a diagram canvas with the given data limits."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


@dataclass
class FigureSpec:
    topic: str
    name: str
    draw: object
    caption: str = ""


REGISTRY: list = []


def register(topic: str, name: str):
    """Decorator: @register("fund.bias-variance", "u-curve") on a draw(p) function."""
    def deco(fn):
        REGISTRY.append(FigureSpec(topic, name, fn))
        return fn
    return deco
