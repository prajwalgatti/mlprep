#!/usr/bin/env python3
"""Render the app icons: a ∇ (gradient) that doubles as a loss valley, with optimizer steps
zig-zagging down to the minimum, over loss-landscape contours. One variant per theme.

Usage: python3 tools/make_icons.py   (writes into PrepApp/Assets.xcassets)
"""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "PrepApp" / "Assets.xcassets"
SS = 4                      # supersampling factor
N = 1024 * SS               # canvas size while drawing


def hexc(h, a=255):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def P(x, y):
    """Design coordinates are on a 1024 grid."""
    return (x * SS, y * SS)


# The ∇: top-left, top-right, bottom (clockwise in screen coordinates).
A, B, C = (214, 262), (810, 262), (512, 784)


def offset_triangle(tri, t_top, t_right, t_left):
    """Inset each edge by its own thickness; return the inner triangle (gives stroke contrast)."""
    (ax, ay), (bx, by), (cx, cy) = tri
    edges = [((ax, ay), (bx, by), t_top), ((bx, by), (cx, cy), t_right), ((cx, cy), (ax, ay), t_left)]
    lines = []
    for (x1, y1), (x2, y2), t in edges:
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L          # inward normal for a clockwise triangle (y down)
        lines.append(((x1 + nx * t, y1 + ny * t), (dx, dy)))

    def intersect(l1, l2):
        (p, r), (q, s) = l1, l2
        cross = r[0] * s[1] - r[1] * s[0]
        u = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
        return (p[0] + u * r[0], p[1] + u * r[1])

    return [intersect(lines[2], lines[0]), intersect(lines[0], lines[1]), intersect(lines[1], lines[2])]


def contours(draw, center, color, width, dashed=False):
    cx, cy = center
    for i, r in enumerate(range(250, 900, 78)):
        rx, ry = r * 1.25, r * 0.78
        if not dashed:
            draw.ellipse([P(cx - rx, cy - ry), P(cx + rx, cy + ry)], outline=color, width=width * SS)
            continue
        steps = 90
        for k in range(0, steps, 2):
            a0, a1 = 2 * math.pi * k / steps, 2 * math.pi * (k + 1) / steps
            draw.line([P(cx + rx * math.cos(a0), cy + ry * math.sin(a0)),
                       P(cx + rx * math.cos(a1), cy + ry * math.sin(a1))], fill=color, width=width * SS)


def valley_path(region, n=7, t0=0.10, t1=0.86, swing=0.33):
    """Optimizer trajectory inside a triangular valley: bounces side to side, damping as it descends."""
    (lx, ly), (rx, ry), (tx, ty) = region
    pts = []
    for i in range(n):
        t = t0 + (t1 - t0) * i / (n - 1)
        xl, xr = lx + (tx - lx) * t, rx + (tx - rx) * t
        y = ly + (ty - ly) * t
        frac = 0.5 + (swing if i % 2 else -swing) * (1 - i / (n - 1)) ** 0.8
        pts.append((xl + (xr - xl) * frac, y))
    return pts


def ring_layer(tri_outer, tri_inner, color):
    mask = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(mask)
    d.polygon([P(*p) for p in tri_outer], fill=255)
    d.polygon([P(*p) for p in tri_inner], fill=0)
    layer = Image.new("RGBA", (N, N), color)
    layer.putalpha(mask)
    return layer


def trajectory(draw, path, color, line_color, shape="circle", final_color=None):
    draw.line([P(*p) for p in path], fill=line_color, width=7 * SS, joint="curve")
    n = len(path)
    for i, (x, y) in enumerate(path):
        r = 21 if i == n - 1 else 19 - 6 * i / n
        fill = final_color if (final_color and i == n - 1) else color
        box = [P(x - r, y - r), P(x + r, y + r)]
        if shape == "square":
            draw.rectangle(box, fill=fill)
        else:
            draw.ellipse(box, fill=fill)


def workbench():
    img = Image.new("RGBA", (N, N), hexc("0E1014"))
    d = ImageDraw.Draw(img)
    contours(d, (512, 600), hexc("1C2130"), 5)
    inner = offset_triangle([A, B, C], 50, 26, 80)       # serif-like contrast: heavy left, hairline right
    img.alpha_composite(ring_layer([A, B, C], inner, hexc("8C9BFF")))
    d = ImageDraw.Draw(img)
    trajectory(d, valley_path(offset_triangle(inner, 14, 14, 14)), hexc("E8A23A"), hexc("E8A23A", 120),
               final_color=hexc("FFD08A"))
    return img


def terminal():
    img = Image.new("RGBA", (N, N), hexc("0A0B0A"))
    d = ImageDraw.Draw(img)
    contours(d, (512, 600), hexc("1E241B"), 5, dashed=True)
    inner = offset_triangle([A, B, C], 52, 52, 52)       # uniform, monospace stroke
    img.alpha_composite(ring_layer([A, B, C], inner, hexc("B8F15A")))
    d = ImageDraw.Draw(img)
    trajectory(d, valley_path(offset_triangle(inner, 14, 14, 14)), hexc("D5DECB"), hexc("D5DECB", 90),
               final_color=hexc("B8F15A"))
    d.rectangle([P(600, 752), P(736, 784)], fill=hexc("B8F15A"))   # cursor: reads as a "∇_" prompt
    return img


def editorial():
    img = Image.new("RGBA", (N, N), hexc("16130F"))
    d = ImageDraw.Draw(img)
    contours(d, (512, 600), hexc("2A231B"), 6)
    # Soft, filled ∇: an inset triangle grown by a round brush (rounded corners).
    r = 46
    inset = offset_triangle([A, B, C], r, r, r)
    d.polygon([P(*p) for p in inset], fill=hexc("E0714F"))
    d.line([P(*p) for p in inset + [inset[0]]], fill=hexc("E0714F"), width=2 * r * SS, joint="curve")
    for x, y in inset:
        d.ellipse([P(x - r, y - r), P(x + r, y + r)], fill=hexc("E0714F"))
    trajectory(d, valley_path(offset_triangle([A, B, C], 58, 58, 58)), hexc("F6EDDF"), hexc("F6EDDF", 150),
               final_color=hexc("FFFFFF"))
    return img


def write_iconset(name, img):
    folder = ASSETS / f"{name}.appiconset"
    folder.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS).save(folder / "icon-1024.png")
    (folder / "Contents.json").write_text(json.dumps({
        "images": [{"filename": "icon-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
        "info": {"author": "xcode", "version": 1},
    }, indent=2))
    print("wrote", folder.relative_to(ROOT))
    # Small copy the app can display in Settings (app icon sets aren't loadable as images).
    preview = ASSETS / f"Preview-{name}.imageset"
    preview.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((240, 240), Image.Resampling.LANCZOS).save(preview / "preview.png")
    (preview / "Contents.json").write_text(json.dumps({
        "images": [{"filename": "preview.png", "idiom": "universal"}],
        "info": {"author": "xcode", "version": 1},
    }, indent=2))


if __name__ == "__main__":
    write_iconset("AppIcon", workbench())
    write_iconset("AppIcon-Terminal", terminal())
    write_iconset("AppIcon-Editorial", editorial())
