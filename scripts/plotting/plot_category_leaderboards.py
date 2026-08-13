#!/usr/bin/env python3
"""Category-Level MTEB Ranking Bar Charts: The Curated Prestige Style.

Strictly follows the identical visual grammar and layout of the main paper's 
ranking bar chart (bracket groups, curated models, unrounded numbers) and 
dynamically applies it to all 5 subset categories + the Overall dataset.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib.colors import to_rgba

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

ROOT = Path(__file__).resolve().parent.parent.parent
LOGOS_DIR = ROOT / "visualizations" / "logos"
OUT_DIR = ROOT / "visualizations"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # single source of truth: complete models, macro scores, LLM meta

# ── Elite Visual Grammar ─────────────────────────────────────────────────────
# LLMs are pulled dynamically from the registry (all complete LLMs); embeddings
# stay a curated, representative subset so the chart doesn't overcrowd.
CURATED_EMBEDDINGS = [
    "tencent__KaLM-Embedding-Gemma3-12B-2511",
    "Qwen__Qwen3-Embedding-8B",
    "nvidia__llama-embed-nemotron-8b",
    "jinaai__jina-embeddings-v5-text-small",
    "Salesforce__SFR-Embedding-2_R",
    "google__embeddinggemma-300m",
    "intfloat__multilingual-e5-large-instruct",
    "Alibaba-NLP__gte-Qwen2-7B-instruct",
    "BAAI__bge-m3",
    "bflhc__Octen-Embedding-8B",
    "codefuse-ai__F2LLM-v2-14B",
    "Linq-AI-Research__Linq-Embed-Mistral",
    "intfloat__e5-mistral-7b-instruct",
    "jinaai__jina-embeddings-v5-text-nano",
    "intfloat__multilingual-e5-base"
]


def _llm_multiline(model: str) -> str:
    """Two-line display label for an LLM bar (name split for the tall bars)."""
    name = registry.display_name(model)
    if " " in name:      # e.g. "Gemini 3.1 Pro" -> "Gemini 3.1\nPro"
        head, tail = name.rsplit(" ", 1)
        return f"{head}\n{tail}"
    if "-" in name:      # e.g. "DeepSeek-R1" -> "DeepSeek\nR1"
        head, tail = name.split("-", 1)
        return f"{head}\n{tail}"
    return name


MODEL_INFO = {
    "tencent__KaLM-Embedding-Gemma3-12B-2511": ("KaLM\n12B",                  "#0052D9", "tencent"),
    "Alibaba-NLP__gte-Qwen2-7B-instruct":    ("GTE Qwen2\n7B",              "#FF6A00", "qwen"),
    "BAAI__bge-m3":                          ("BGE-M3",                     "#E4242B", "baai"),
    "bflhc__Octen-Embedding-8B":               ("Octen\n8B",                   "#56BF60", "octen"),
    "codefuse-ai__F2LLM-v2-14B":             ("F2LLM\n14B",                 "#727CD1", "codefuse"),
    "Linq-AI-Research__Linq-Embed-Mistral":  ("Linq\nMistral",              "#3E6ABD", "linq"),
    "Qwen__Qwen3-Embedding-8B":                ("Qwen3\nEmbed 8B",             "#694FEF", "qwen"),
    "jinaai__jina-embeddings-v5-text-small":   ("Jina v5\nSmall",             "#14B8A6", "jina"),
    "nvidia__llama-embed-nemotron-8b":         ("Nemotron\n8B",                "#74B71B", "nvidia"),
    "Salesforce__SFR-Embedding-2_R":           ("SFR\nEmbed-2",               "#0AA4E1", "salesforce"),
    "google__embeddinggemma-300m":             ("Gemma\n300M",                "#63A2FF", "gemma"),
    "intfloat__multilingual-e5-large-instruct":("mE5 Large\nInstruct",        "#0078D4", "microsoft"),
    "intfloat__multilingual-e5-base":          ("mE5\nBase",                  "#59A5F0", "microsoft"),
    "intfloat__e5-mistral-7b-instruct":        ("E5 Mistral\n7B",             "#1F66B8", "microsoft"),
    "jinaai__jina-embeddings-v5-text-nano":    ("Jina v5\nNano",              "#5EEAD4", "jina"),
}
# Inject an entry for every complete LLM from the registry (name, color, logo).
for _m, _meta in registry.LLM_META.items():
    MODEL_INFO[_m] = (_llm_multiline(_m), _meta["color"], _meta["logo"])

# Ordered by descending overall (macro) score so the strongest LLM leads.
_CS = registry.category_scores()
_LLMS_BY_SCORE = [m for m in _CS.sort_values("Overall", ascending=False).index
                  if _CS.loc[m, "model_type"] == "llm"]
CURATED_MODELS = _LLMS_BY_SCORE + [m for m in CURATED_EMBEDDINGS if m in _CS.index]

DEFAULT_COLOR = "#94A3B8"
GROUP_GAP = 1.0

# ── Logo loading ─────────────────────────────────────────────────────────────
_LOGO_CACHE: dict = {}

def load_logo(logo_key: str, target_h: int = 32):
    if not logo_key: return None
    if logo_key in _LOGO_CACHE:
        arr = _LOGO_CACHE[logo_key]
        if arr is None: return None
        zoom = target_h / arr.shape[0]
        return OffsetImage(arr, zoom=zoom, interpolation="lanczos")
        
    required_px = int(target_h * 4.5)
    for ext in ['.png', '.webp', '.jpg']:
        path = LOGOS_DIR / f"{logo_key}{ext}"
        if path.exists():
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


def draw_curated_leaderboard(cat, cs):
    """Draws the hyper-curated, bracket-separated ranking.

    `cs` is the registry's per-model category table (macro Overall + per-category
    means, complete models only). `cat` selects the column ("Overall" or a category).
    """
    col = "Overall" if cat == "Overall" else cat
    avg = cs.reset_index()[["model", "model_type", col]].rename(columns={col: "score"})

    # Filter strictly to curated models
    avg = avg[avg["model"].isin(CURATED_MODELS)].copy()
    
    # Map explicit order index so we don't accidentally sort Gemini alphabetically!
    avg["_explicit_order"] = avg["model"].map({m: i for i, m in enumerate(CURATED_MODELS)})
    
    # Separate and structure visually
    llms = avg[avg["model_type"] == "llm"].sort_values("_explicit_order", ascending=True)
    embs = avg[avg["model_type"] == "embedding"].sort_values("score", ascending=False)
    
    avg = pd.concat([llms, embs]).reset_index(drop=True)

    n = len(avg)
    scores = (avg["score"] * 100).values
    max_score = scores.max()
    min_score = scores.min()
    
    # Calculate bounding geometry
    FLOOR = max(0, min_score - (max_score - min_score) * 0.7)
    FLOOR = int(FLOOR / 5) * 5 # snap to nearest 5

    is_llm = [row["model_type"] == "llm" for _, row in avg.iterrows()]
    n_llm = sum(is_llm)

    best_emb = max(scores[i] for i in range(n) if not is_llm[i])

    # X-coordinates with Group Gap explicitly modeled
    xs = np.array([float(i) if i < n_llm else float(i) + GROUP_GAP for i in range(n)])
    
    # Stretchy Figure dimension mimicking Main Text to accommodate 18 columns instead of 15
    fig, ax = plt.subplots(figsize=(32, 14), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    for sp in ax.spines.values():
        sp.set_visible(False)
        
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Top Y limit needs extra massive headroom to clear the bracket labels 
    Y_TOP = max_score + ((max_score - FLOOR) * 0.38)
    Y_BOT = FLOOR - ((max_score - FLOOR) * 0.55)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_xlim(xs[0] - 1.2, xs[-1] + 1.2)

    # ── Section brackets (Restored to prestigious logic) ──
    bracket_y = max_score + ((max_score - FLOOR) * 0.08)
    label_y   = bracket_y + ((max_score - FLOOR) * 0.07)
    
    lx0, lx1 = xs[0] - 0.42, xs[n_llm - 1] + 0.42
    ex0, ex1 = xs[n_llm] - 0.42, xs[-1] + 0.42

    # LLM Brackets
    ax.plot([lx0, lx1], [bracket_y, bracket_y], color="#DC2626", lw=3.5, solid_capstyle="round", zorder=10, clip_on=False)
    for bx in [lx0, lx1]:
        ax.plot([bx, bx], [bracket_y, bracket_y - ((max_score - FLOOR) * 0.05)], color="#DC2626", lw=2, solid_capstyle="round", zorder=10, clip_on=False)
    ax.text((lx0 + lx1) / 2, label_y, "LLMs", ha="center", va="bottom",
            fontsize=48, fontweight="bold", color="#B91C1C", clip_on=False, zorder=11)

    # Embedded Brackets
    ax.plot([ex0, ex1], [bracket_y, bracket_y], color="#2563EB", lw=3.0, solid_capstyle="round", zorder=10, clip_on=False)
    for bx in [ex0, ex1]:
        ax.plot([bx, bx], [bracket_y, bracket_y - ((max_score - FLOOR) * 0.05)], color="#2563EB", lw=2, solid_capstyle="round", zorder=10, clip_on=False)
    ax.text((ex0 + ex1) / 2, label_y, "Embedding Models", ha="center", va="bottom",
            fontsize=48, fontweight="bold", color="#1D4ED8", clip_on=False, zorder=11)

    # ── Grid & Glow ──
    bg = mpatches.FancyBboxPatch(
        (lx0, FLOOR - 0.3), lx1 - lx0, max_score - FLOOR + 1.5,
        boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#FECACA",
        linewidth=0.8, zorder=0, alpha=0.85,
    )
    ax.add_patch(bg)

    step = 5 if (max_score - FLOOR) > 15 else 2
    for g in range(int(FLOOR), int(max_score) + 5, step):
        ax.axhline(g, color="#F1F5F9", linewidth=1.0, zorder=1)
    ax.axhline(FLOOR, color="#CBD5E1", linewidth=1.5, zorder=2)

    # ── Bars ──
    bar_w = 0.75
    Y_LOGO = FLOOR - ((max_score - FLOOR) * 0.08)
    Y_TEXT = FLOOR - ((max_score - FLOOR) * 0.18)

    for i, row in avg.iterrows():
        x = xs[i]
        score = scores[i]
        llm = is_llm[i]

        info = MODEL_INFO.get(row["model"])
        name     = (info[0] if info else row["model"].split("__")[-1][:12]).replace("\n", " ")
        logo_key = info[2] if info else ""

        # Using Deep Semantic Red/Blue explicitly, matching the main format
        fill_color = "#DC2626" if llm else "#2563EB"
        bar_h = score - FLOOR

        # Ghost behind the LLM bars
        if llm:
            ax.bar(x, bar_h, bottom=FLOOR, width=bar_w + 0.24, color=fill_color, alpha=0.10, zorder=2, edgecolor="none")
        
        ax.bar(x, bar_h, bottom=FLOOR, width=bar_w,
               color=fill_color, zorder=3, edgecolor="none", linewidth=0)

        # Main Score - centered inside bar (smaller than bar width for side padding)
        cy = (FLOOR + score) / 2
        ax.text(x, cy, f"{score:.0f}", ha="center", va="center", fontsize=37, fontweight="bold", color="white", zorder=5)

        # Logo Drop
        img_obj = load_logo(logo_key, target_h=46)
        if img_obj:
            ab = AnnotationBbox(img_obj, (x, Y_LOGO), frameon=False, pad=0.0, clip_on=False, annotation_clip=False, zorder=5)
            ax.add_artist(ab)

        # Explicit name drop — rotated for legibility at 18 bars
        ax.text(x, Y_TEXT, name, ha="right", va="top", fontsize=36, fontweight="bold",
                color="#1F2937", rotation=38, rotation_mode="anchor", clip_on=False, zorder=5)


    # Best embedding reference line
    ax.plot([xs[0] - 0.5, xs[-1] + 0.5], [best_emb, best_emb],
            color="#94A3B8", linewidth=1.5, linestyle="--", zorder=4, alpha=0.6)

    # Clean Title without Exhaustive label
    fig.suptitle(f"MTEB(LLM) Leaderboard: {cat}", fontsize=52, fontweight="heavy", y=1.02)

    plt.tight_layout()
    out_path = OUT_DIR / f"ranking_{cat}.pdf" # Appendix plots best in PDF
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / f"ranking_{cat}.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" - Generated Prestige Category Leaderboard: {out_path.name}")

def main():
    print("Generating Curated Categories mimicking Main Text layout...")
    cs = registry.category_scores()   # complete models, macro Overall + per-cat means

    categories = ["Overall"] + registry.CATEGORIES

    for cat in categories:
        col = "Overall" if cat == "Overall" else cat
        if col in cs.columns:
            draw_curated_leaderboard(cat, cs)

    print("\nAll standalone curated leaderboards successfully formatted.")

if __name__ == "__main__":
    main()
