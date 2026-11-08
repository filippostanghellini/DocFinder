#!/usr/bin/env python
"""Generate the macOS Squircle app icon from the approved flat design.

Renders the centered white "document + magnifier" symbol (soft geometry,
3 text lines, detached capsule handle) onto the dark Apple-squircle
background (824x824 box on a 1024x1024 canvas, ~10% transparent margin)
with brand cyan/violet glows and a specular rim highlight.

Usage:
    .venv/bin/python scripts/make_icon.py [output.png ...]

Defaults to Logo.png in the project root. Requires Pillow + PyMuPDF + numpy
(dev tools only; not runtime dependencies of DocFinder).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Approved "2A" symbol geometry (viewBox 0 0 100 100, white strokes).
SYMBOL_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100" fill="none">
<g transform="translate(0.5,1)"><path d="M62 27V23c0-4.5-3.5-8-8-8H33c-3 0-5.5 1.2-7.5 3.2l-7.3 7.3c-2 2-3.2 4.5-3.2 7.5v42c0 4.5 3.5 8 8 8h31c4.5 0 8-3.5 8-8v-3" stroke="white" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M31 15.5V25c0 3.5-2.5 6-6 6h-9.5" stroke="white" stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/><line x1="25" y1="44" x2="36" y2="44" stroke="white" stroke-width="4.4" stroke-linecap="round"/><line x1="25" y1="54" x2="36" y2="54" stroke="white" stroke-width="4.4" stroke-linecap="round"/><line x1="25" y1="64" x2="33" y2="64" stroke="white" stroke-width="4.4" stroke-linecap="round"/><circle cx="60" cy="50" r="14.5" stroke="white" stroke-width="4.6"/><line x1="72.5" y1="62.5" x2="83" y2="73" stroke="white" stroke-width="6.5" stroke-linecap="round"/></g></svg>"""

SIZE = 1024
BOX = 824  # Apple HIG: 824x824 squircle on a 1024x1024 canvas (80% grid)
RADIUS = 185
PAD = (SIZE - BOX) // 2  # 100px transparent margin on each side
# Symbol artwork occupies ~64% of the squircle plate (approved proportions:
# airy margin around the artwork, thin strokes visible at dock sizes).
SYMBOL_BOX_PX = 700


def render_symbol(box_px: int) -> Image.Image:
    """Rasterize the SVG symbol scaled to ``box_px`` (4x supersampled)."""
    scale = box_px / 100.0 * 4
    doc = fitz.open(stream=SYMBOL_SVG.encode(), filetype="svg")
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    if pix.n < 4:
        raise RuntimeError("Symbol SVG rendered without alpha")
    img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    if img.size != (int(box_px * 4), int(box_px * 4)):
        raise RuntimeError(f"unexpected symbol raster size: {img.size}")
    return img.resize((box_px, box_px), Image.Resampling.LANCZOS)


def squircle_mask() -> np.ndarray:
    """Antialiased squircle alpha mask (824px box, continuous-corner feel)."""
    ss = 2
    mask = Image.new("L", (SIZE * ss, SIZE * ss), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [PAD * ss, PAD * ss, (PAD + BOX) * ss, (PAD + BOX) * ss],
        radius=RADIUS * ss,
        fill=255,
    )
    return np.array(mask.resize((SIZE, SIZE), Image.Resampling.LANCZOS))


def build_bg() -> np.ndarray:
    """Dark navy gradient with cyan/violet brand glows (approved palette)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    t = (xx + yy) / (2.0 * SIZE)
    r = 22 * (1 - t) + 7 * t
    g = 27 * (1 - t) + 9 * t
    b = 46 * (1 - t) + 18 * t
    for cx, cy, rad, cr, cg, cb in (
        (0.25 * SIZE, 0.20 * SIZE, 0.55 * SIZE, 67, 173, 255),   # cyan glow
        (0.80 * SIZE, 0.80 * SIZE, 0.55 * SIZE, 234, 122, 242),  # violet glow
    ):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        glow = np.int8(0)
        glow = np.clip(35.0 * (1.0 - np.sqrt(d2) / rad), 0, 90)
        r += glow * cr / 255.0
        g += glow * cg / 255.0
        b += glow * cb / 255.0
    return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)


def make_icon() -> Image.Image:
    mask = squircle_mask()
    shadow = np.zeros((SIZE, SIZE), dtype=np.uint8)
    shadow[14:, :] = mask[:-14, :]
    shadow = Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(24))
    sh_arr = np.array(shadow, dtype=np.float32) * 0.55
    shadow_layer = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    shadow_layer[:, :, 3] = sh_arr.clip(0, 255).astype(np.uint8)

    bg = build_bg()
    bg_layer = np.dstack([bg, mask[..., None]])

    symbol_layer = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    symbol = np.array(render_symbol(SYMBOL_BOX_PX), dtype=np.uint8)
    # The SVG content barycenter already sits at the box center (verified by
    # the white-symbol bbox check below), so a straight centered paste is exact.
    off = (SIZE - SYMBOL_BOX_PX) // 2
    symbol_layer[off : off + SYMBOL_BOX_PX, off : off + SYMBOL_BOX_PX] = symbol

    rim = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    rim_draw = ImageDraw.Draw(rim)
    rim_draw.rounded_rectangle(
        [PAD + 1, PAD + 1, PAD + BOX - 1, PAD + BOX - 1],
        radius=RADIUS,
        outline=(255, 255, 255, 48),
        width=3,
    )

    out = Image.alpha_composite(Image.fromarray(shadow_layer, "RGBA"), Image.fromarray(bg_layer, "RGBA"))
    out = Image.alpha_composite(out, Image.fromarray(symbol_layer, "RGBA"))
    out = Image.alpha_composite(out, rim)
    return out.convert("RGBA")


def self_check(icon: Image.Image) -> None:
    """Fail loudly if the icon breaks the Apple HIG squircle contract."""
    arr = np.array(icon)
    assert icon.size == (SIZE, SIZE), f"wrong size: {icon.size}"
    corners = [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)]
    assert all(arr[y, x, 3] == 0 for x, y in corners), "safety margin not transparent"
    inset = 2  # sample just inside the edge: the AA ramp keeps the boundary pixel semi-transparent
    for x, y in [(PAD + inset, SIZE // 2), (SIZE // 2, PAD + inset), (SIZE - PAD - 1 - inset, SIZE // 2)]:
        assert arr[y, x, 3] >= 250, f"squircle not opaque at edge ({x},{y})"
    rgb = arr[:, :, :3]
    white = ((rgb > 245).all(axis=2) & (arr[:, :, 3] == 255))
    assert int(white.sum()) > 2000, f"white symbol missing: {int(white.sum())} px"
    wys, wxs = np.where(white)
    span_w, span_h = wxs.max() - wxs.min(), wys.max() - wys.min()
    assert 0.58 * BOX < span_w < 0.68 * BOX and 0.58 * BOX < span_h < 0.68 * BOX, (
        f"symbol proportion off HIG artwork grid: {span_w}x{span_h}"
    )
    cx, cy = (wxs.min() + wxs.max()) / 2, (wys.min() + wys.max()) / 2
    assert abs(cx - SIZE / 2) < 6 and abs(cy - SIZE / 2) < 6, f"symbol not centered: ({cx},{cy})"
    ys, xs = np.where(arr[:, :, 3] >= 250)
    # The opaque squircle plate must respect the Apple 10% margin grid
    # (the faint baked soft shadow is allowed to spill a few px).
    assert xs.min() >= PAD - 10 and xs.max() <= SIZE - PAD + 10, "icon plate too large for HIG grid"


def main() -> None:
    icon = make_icon()
    self_check(icon)
    targets = sys.argv[1:] or ["Logo.png"]
    for target in targets:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        icon.save(path)
        print(f"written {path}")


if __name__ == "__main__":
    main()
