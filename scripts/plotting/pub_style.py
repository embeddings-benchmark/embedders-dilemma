"""Shared publication style for all plotting scripts.

Single source of truth for colors, fonts, sizes, and helpers.
Every plotting script does: from pub_style import *
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# Two-class palette (muted, print-safe, WCAG AA on white)
C_EMB = "#2D6A9F"  # steel blue
C_LLM = "#C0392B"  # brick red

# Category colors
CAT_COLORS = {
    "Classification":     "#2D6A9F",
    "Clustering":         "#27AE60",
    "STS":                "#E67E22",
    "PairClassification": "#8E44AD",
    "Retrieval":          "#C0392B",
}
CAT_ORDER = ["Classification", "Clustering", "STS", "PairClassification", "Retrieval"]

# Cost-breakdown segments
C_INPUT    = "#2D6A9F"
C_CACHED   = "#85C1E9"
C_OUTPUT   = "#E67E22"
C_THINKING = "#C0392B"

# Embedding model family colors (for scaling plot)
FAMILY_COLORS = {
    "Qwen3":     "#C0392B",
    "F2LLM":     "#2980B9",
    "Jina":      "#27AE60",
    "mE5":       "#F39C12",
    "GTE-Qwen2": "#8E44AD",
    "Other":     "#7F8C8D",
}

# Figure sizes (inches) for CoLM two-column format
FIG_COL  = (3.25, 2.4)
FIG_FULL = (6.75, 3.5)
FIG_TALL = (6.75, 5.5)

# White stroke for text legibility over data
STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]
STROKE_THIN = [pe.withStroke(linewidth=1.8, foreground="white")]


def apply_style():
    """Set matplotlib rcParams for publication quality."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size":         9,
        "axes.linewidth":    0.6,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.frameon":    True,
        "legend.edgecolor":  "#cccccc",
        "legend.fancybox":   False,
        "legend.framealpha":  0.95,
        "grid.alpha":        0.12,
        "grid.linewidth":    0.5,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# Short display names (consolidated from all scripts)
_SHORT = {
    "jina-embeddings-v5-text-small":      "Jina-v5-S",
    "jina-embeddings-v5-text-nano":       "Jina-v5-Nano",
    "multilingual-e5-large-instruct":     "mE5-L-Inst",
    "multilingual-e5-large":              "mE5-L",
    "multilingual-e5-base":               "mE5-B",
    "multilingual-e5-small":              "mE5-S",
    "Qwen3-Embedding-8B":                 "Qwen3-E-8B",
    "Qwen3-Embedding-4B":                 "Qwen3-E-4B",
    "Qwen3-Embedding-0.6B":              "Qwen3-E-0.6B",
    "KaLM-Embedding-Gemma3-12B-2511":     "KaLM-12B",
    "F2LLM-v2-14B":                       "F2LLM-14B",
    "F2LLM-v2-8B":                        "F2LLM-8B",
    "F2LLM-v2-4B":                        "F2LLM-4B",
    "F2LLM-v2-1.7B":                      "F2LLM-1.7B",
    "F2LLM-v2-0.6B":                      "F2LLM-0.6B",
    "llama-embed-nemotron-8b":             "Nemotron-8B",
    "snowflake-arctic-embed-l-v2.0":       "Arctic-L-v2",
    "gte-Qwen2-7B-instruct":              "GTE-Qwen2-7B",
    "gte-Qwen2-1.5B-instruct":            "GTE-Qwen2-1.5B",
    "SFR-Embedding-2_R":                   "SFR-2",
    "Linq-Embed-Mistral":                  "Linq-Mistral",
    "embeddinggemma-300m":                 "EmbGemma-300M",
    "e5-mistral-7b-instruct":             "E5-Mistral-7B",
    "Octen-Embedding-8B":                  "Octen-8B",
    "bge-m3":                              "BGE-M3",
    "GritLM-7B":                           "GritLM-7B",
    "gemini-3.1-pro-preview":              "Gemini 3.1 Pro",
    "gemini-3-flash-preview":              "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview":       "Gemini 3.1 Flash Lite",
    # Cross-family LLMs added in rebuttal (the 10 complete, cost-comparable LLMs)
    "deepseek-r1":                         "DeepSeek-R1",
    "deepseek-v4-flash":                   "DeepSeek-V4-Flash",
    "glm-4.7":                             "GLM-4.7",
    "kimi-k2.6":                           "Kimi-K2.6",
    "minimax-m2.7":                        "MiniMax-M2.7",
    "qwen3.6-27b":                         "Qwen3.6-27B",
    "qwen3.6-35b-a3b":                     "Qwen3.6-35B-A3B",
}


def short_name(model: str) -> str:
    suffix = model.split("__")[-1]
    return _SHORT.get(suffix, suffix)


def save_fig(fig, name, out_dir=None):
    """Save figure as PNG + PDF to visualizations/."""
    if out_dir is None:
        out_dir = ROOT / "visualizations"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}.png + .pdf")


def get_family(model_short: str) -> str:
    """Map a short model name to its family for coloring."""
    for fam in ["Qwen3", "F2LLM", "Jina", "mE5", "GTE-Qwen2"]:
        if fam.lower().replace("-", "") in model_short.lower().replace("-", ""):
            return fam
    return "Other"
