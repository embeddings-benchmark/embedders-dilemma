#!/usr/bin/env python3
"""Generate clean monogram logo badges for LLM families we have no brand asset for.

Produces rounded-square badges (brand color + white monogram) so the category
leaderboards show an icon for every model. Real brand logos, if added later to
visualizations/logos/<key>.png, take precedence automatically.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "visualizations" / "logos"

# key -> (monogram, background color)
BADGES = {
    "deepseek": ("DS",  "#4D6BFE"),
    "glm":      ("GLM", "#2E5CE6"),
    "kimi":     ("Ki",  "#16A34A"),   # Moonshot/Kimi green
    "minimax":  ("MM",  "#E1483B"),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_badge(text, color, size=512):
    scale = 4
    S = size * scale
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(S * 0.22)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=color)
    # fit font to text width
    fs = int(S * (0.5 if len(text) <= 2 else 0.34))
    font = _font(fs)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((S - tw) / 2 - bbox[0], (S - th) / 2 - bbox[1]), text,
           font=font, fill="white")
    return img.resize((size, size), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (text, color) in BADGES.items():
        make_badge(text, color).save(OUT / f"{key}.png")
        print(f"  wrote logos/{key}.png  ({text} on {color})")


if __name__ == "__main__":
    main()
