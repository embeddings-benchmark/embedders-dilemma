#!/usr/bin/env python3
"""Standardize logos in visualizations/logos/ into uniform 128x128 PNGs.

Input formats handled:   .svg, .webp, .png
Output:  One <key>.png per logo at 128x128 transparent-background PNG.

Dependencies:
  pip install cairosvg Pillow
"""

from pathlib import Path
from PIL import Image
import io

LOGOS_DIR = Path(r"C:\Users\ADNAN\LLM\LLMvsEmbeddings\human-eval-results\visualizations\logos")

# Map: current filename stem (lower) → canonical key we want to use in the plot
RENAME_MAP = {
    "gemini-color":    "google",
    "gemma-color":     "embgemma",
    "jina":            "jina",
    "nvidia-color":    "nvidia",
    "qwen-color":      "qwen",
    "salesforce":      "salesforce",
    "tencent-color":   "tencent",
    "baai":            "baai",
    "codefuse":        "codefuse",
    "linq":            "linq",
    "octen":           "octen",
    "microsoft-color": "microsoft",
}

SIZE = (128, 128)


def svg_to_pil(svg_path: Path) -> Image.Image:
    """Convert SVG file to a PIL Image via svglib + reportlab."""
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(svg_path))
        if drawing is None:
            raise ValueError("svglib returned None")
        # Scale to our target size
        sx = SIZE[0] / drawing.width
        sy = SIZE[1] / drawing.height
        drawing.width = SIZE[0]
        drawing.height = SIZE[1]
        drawing.transform = (sx, 0, 0, sy, 0, 0)
        png_bytes = renderPM.drawToString(drawing, fmt="PNG", dpi=96)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:
        print(f"  [warn] svglib failed for {svg_path.name}: {e}")
        return Image.new("RGBA", SIZE, (200, 200, 200, 255))


def standardize_logo(src_path: Path, canonical_key: str) -> Path:
    """Load any format, resize to SIZE, save as <canonical_key>.png."""
    suffix = src_path.suffix.lower()
    if suffix == ".svg":
        img = svg_to_pil(src_path)
    else:
        img = Image.open(src_path).convert("RGBA")

    # Paste onto a white background so transparencies look good
    bg = Image.new("RGBA", SIZE, (255, 255, 255, 255))
    # Resize while preserving aspect ratio, then center-paste
    img.thumbnail(SIZE, Image.LANCZOS)
    offset = ((SIZE[0] - img.width) // 2, (SIZE[1] - img.height) // 2)
    bg.paste(img, offset, img)

    # Save to same directory
    out_path = LOGOS_DIR / f"{canonical_key}.png"
    bg.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path


def main():
    print(f"Standardizing logos in: {LOGOS_DIR}\n")
    for src_path in sorted(LOGOS_DIR.iterdir()):
        stem = src_path.stem.lower()
        if src_path.suffix.lower() not in (".svg", ".webp", ".png"):
            continue
        canonical_key = RENAME_MAP.get(stem)
        if canonical_key is None:
            print(f"  [skip] {src_path.name} — no entry in RENAME_MAP")
            continue
        try:
            out = standardize_logo(src_path, canonical_key)
            print(f"  ✓  {src_path.name:30s}  →  {out.name}")
        except Exception as e:
            print(f"  ✗  {src_path.name}: {e}")


if __name__ == "__main__":
    main()
