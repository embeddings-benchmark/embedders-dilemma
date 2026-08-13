#!/usr/bin/env python3
"""MTEB Ranking Bar Chart V2: LLMs vs Embedding Models.
Outputs to visualizations/ranking_mteb.{png,pdf}

Style reference: Artificial Analysis Intelligence Index bar chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib.colors import to_rgba

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'Helvetica']

ROOT = Path(__file__).resolve().parent.parent.parent
LOGOS_DIR = ROOT / "visualizations" / "logos"

# ── Curated model list: best per company, no duplicates ──────────────────────
# Order matters: LLMs first, then embeddings (sorted by score descending)
CURATED_MODELS = [
    # LLMs
    "google__gemini-3.1-pro-preview",
    "google__gemini-3-flash-preview",
    # Embeddings — recognizable names, one per company
    "tencent__KaLM-Embedding-Gemma3-12B-2511",
    "Qwen__Qwen3-Embedding-8B",
    "nvidia__llama-embed-nemotron-8b",
    "jinaai__jina-embeddings-v5-text-small",
    "Salesforce__SFR-Embedding-2_R",
    "google__embeddinggemma-300m",
    "intfloat__multilingual-e5-large-instruct",
]

MODEL_INFO = {
    # LLMs — Gemini blue
    "google__gemini-3.1-pro-preview":        ("Gemini 3.1\nPro",              "#3186FF", "gemini"),
    "google__gemini-3-flash-preview":        ("Gemini 3\nFlash",              "#60A5FA", "gemini"),
    "google__gemini-3.1-flash-lite-preview": ("Gemini 3.1\nFlash Lite",      "#93C5FD", "gemini"),
    # Embeddings — colors from logo pixel analysis
    "tencent__KaLM-Embedding-Gemma3-12B-2511": ("KaLM\n12B",                  "#0052D9", "tencent"),
    "bflhc__Octen-Embedding-8B":               ("Octen\n8B",                   "#56BF60", "octen"),
    "Qwen__Qwen3-Embedding-8B":                ("Qwen3\nEmbed 8B",             "#694FEF", "qwen"),
    "jinaai__jina-embeddings-v5-text-small":   ("Jina v5\nSmall",             "#14B8A6", "jina"),
    "nvidia__llama-embed-nemotron-8b":         ("Nemotron\n8B",                "#74B71B", "nvidia"),
    "Salesforce__SFR-Embedding-2_R":           ("SFR\nEmbed-2",               "#0AA4E1", "salesforce"),
    "codefuse-ai__F2LLM-v2-14B":              ("F2LLM\n14B",                 "#727CD1", "codefuse"),
    "Linq-AI-Research__Linq-Embed-Mistral":    ("Linq\nMistral",              "#3E6ABD", "linq"),
    "google__embeddinggemma-300m":             ("Gemma\n300M",                "#63A2FF", "gemma"),
    "intfloat__multilingual-e5-large-instruct":("mE5 Large\nInstruct",        "#0078D4", "microsoft"),
}
DEFAULT_COLOR = "#94A3B8"
FLOOR = 50
GROUP_GAP = 1.2


# ── Logo loading ─────────────────────────────────────────────────────────────
_LOGO_CACHE: dict = {}

def load_logo(logo_key: str, target_h: int = 32):
    if not logo_key:
        return None
    if logo_key in _LOGO_CACHE:
        arr = _LOGO_CACHE[logo_key]
        if arr is None:
            return None
        zoom = target_h / arr.shape[0]
        return OffsetImage(arr, zoom=zoom, interpolation="lanczos")
    required_px = int(target_h * 4.5)
    for ext in ['.png', '.webp']:
        path = LOGOS_DIR / f"{logo_key}{ext}"
        if path.exists() and path.stat().st_size > 300:
            try:
                img = Image.open(path).convert("RGBA")
                if img.height < required_px:
                    s = required_px / img.height
                    img = img.resize((int(img.width * s), required_px), Image.LANCZOS)
                arr = np.array(img)
                _LOGO_CACHE[logo_key] = arr
                zoom = target_h / arr.shape[0]
                return OffsetImage(arr, zoom=zoom, interpolation="lanczos")
            except Exception:
                pass
    _LOGO_CACHE[logo_key] = None
    return None


def desaturate(hex_color, factor=0.15):
    r, g, b, a = to_rgba(hex_color)
    gray = 0.5
    return (r + (gray - r) * factor, g + (gray - g) * factor,
            b + (gray - b) * factor, a)


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(ROOT / "data" / "scores.csv")
    avg = df.groupby(["model", "model_type"])["score"].mean().reset_index()

    # Filter to curated list, preserve curated order
    avg = avg[avg["model"].isin(CURATED_MODELS)].copy()
    avg["_order"] = avg["model"].map({m: i for i, m in enumerate(CURATED_MODELS)})
    avg = avg.sort_values("_order").reset_index(drop=True)

    n = len(avg)
    scores = (avg["score"] * 100).values
    max_score = scores.max()
    is_llm = [row["model_type"] == "llm" for _, row in avg.iterrows()]
    n_llm = sum(is_llm)

    # Best embedding score for reference line
    best_emb = max(scores[i] for i in range(n) if not is_llm[i])

    # ── X-positions with gap ─────────────────────────────────────────────────
    xs = np.array([float(i) if i < n_llm else float(i) + GROUP_GAP for i in range(n)])

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 10.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.set_xticks([])

    Y_TOP = max_score + 8
    Y_BOT = FLOOR - 28
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_xlim(xs[0] - 0.9, xs[-1] + 1.6)

    # ── Section headers: bracket lines + labels ──────────────────────────────
    bracket_y = max_score + 2.0       # line just above tallest bar
    label_y   = max_score + 3.5       # text above bracket

    lx0, lx1 = xs[0] - 0.42, xs[n_llm - 1] + 0.42
    ex0, ex1 = xs[n_llm] - 0.42, xs[-1] + 0.42

    # LLMs bracket — thick blue line with down-ticks at ends
    ax.plot([lx0, lx1], [bracket_y, bracket_y],
            color="#2563EB", lw=3, solid_capstyle="round", zorder=10, clip_on=False)
    for bx in [lx0, lx1]:
        ax.plot([bx, bx], [bracket_y, bracket_y - 1.2],
                color="#2563EB", lw=2, solid_capstyle="round", zorder=10, clip_on=False)
    ax.text((lx0 + lx1) / 2, label_y, "LLMs",
            ha="center", va="bottom",
            fontsize=34, fontweight="bold", color="#1D4ED8",
            clip_on=False, zorder=11)

    # Embedding bracket — thinner slate line with down-ticks
    ax.plot([ex0, ex1], [bracket_y, bracket_y],
            color="#64748B", lw=2.5, solid_capstyle="round", zorder=10, clip_on=False)
    for bx in [ex0, ex1]:
        ax.plot([bx, bx], [bracket_y, bracket_y - 1.2],
                color="#64748B", lw=1.5, solid_capstyle="round", zorder=10, clip_on=False)
    ax.text((ex0 + ex1) / 2, label_y, "Embedding Models",
            ha="center", va="bottom",
            fontsize=34, fontweight="bold", color="#475569",
            clip_on=False, zorder=11)

    # ── LLM background glow zone ────────────────────────────────────────────
    bg = mpatches.FancyBboxPatch(
        (lx0, FLOOR - 0.3), lx1 - lx0, max_score - FLOOR + 1.5,
        boxstyle="round,pad=0.2",
        facecolor="#EFF6FF", edgecolor="#BFDBFE",
        linewidth=0.8, zorder=0, alpha=0.85,
    )
    ax.add_patch(bg)

    # ── Grid lines ───────────────────────────────────────────────────────────
    for g in range(FLOOR, int(max_score) + 5, 5):
        ax.axhline(g, color="#F1F5F9", linewidth=0.6, zorder=1)
    ax.axhline(FLOOR, color="#CBD5E1", linewidth=1.0, zorder=2)

    # ── Draw bars ────────────────────────────────────────────────────────────
    bar_w = 0.72
    Y_LOGO = FLOOR - 3.5
    Y_TEXT = FLOOR - 7.5

    for i, row in avg.iterrows():
        x = xs[i]
        score = scores[i]
        llm = is_llm[i]

        info = MODEL_INFO.get(row["model"])
        name     = (info[0] if info else row["model"].split("__")[-1][:14]).replace("\n", " ")
        color    = info[1] if info else DEFAULT_COLOR
        logo_key = info[2] if info else ""

        bar_h = score - FLOOR

        if llm:
            # Glow behind LLM bars
            ax.bar(x, bar_h, bottom=FLOOR, width=bar_w + 0.24,
                   color=color, alpha=0.18, zorder=2, edgecolor="none")
            ax.bar(x, bar_h, bottom=FLOOR, width=bar_w,
                   color=color, zorder=3, edgecolor="none", linewidth=0)
        else:
            c = desaturate(color, factor=0.12)
            ax.bar(x, bar_h, bottom=FLOOR, width=bar_w,
                   color=c, zorder=3, edgecolor="none", linewidth=0)

        # Score — centered, large
        cy = (FLOOR + score) / 2
        ax.text(x, cy, f"{score:.0f}",
                ha="center", va="center",
                fontsize=27, fontweight="bold",
                color="white", zorder=5)

        # Logo
        img_obj = load_logo(logo_key, target_h=46)
        if img_obj:
            ab = AnnotationBbox(
                img_obj, (x, Y_LOGO),
                frameon=False, pad=0.0,
                clip_on=False, annotation_clip=False, zorder=5,
            )
            ax.add_artist(ab)

        # Model name — rotated for legibility
        ax.text(x, Y_TEXT, name,
                ha="right", va="top",
                fontsize=26, fontweight="bold",
                color="#1F2937", rotation=38, rotation_mode="anchor",
                clip_on=False, zorder=5)

    # ── Best embedding reference line (no label — the bracket tells the story)
    ax.plot([xs[0] - 0.5, xs[-1] + 0.5], [best_emb, best_emb],
            color="#94A3B8", linewidth=0.9, linestyle=(0, (6, 4)),
            zorder=4, alpha=0.5)

    # ── Title ────────────────────────────────────────────────────────────────
    fig.suptitle("MTEB(LLM) Leaderboard Scores", fontsize=40, fontweight="bold",
                 y=1.03, color="#0F172A")

    # ── Layout ───────────────────────────────────────────────────────────────
    plt.subplots_adjust(bottom=0.02, top=0.94, left=0.02, right=0.96)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = ROOT / "visualizations" / "ranking_mteb.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
