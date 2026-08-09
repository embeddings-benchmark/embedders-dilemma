"""Capability radar profile — 3 models across 5 task categories.

Shows where LLMs and embeddings have complementary strengths:
LLMs dominate retrieval, embeddings dominate classification.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pub_style import save_fig   # only save_fig — no apply_style
import registry                   # single source of truth

_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
}

FONTSIZE       = 18
FONTSIZE_SMALL = 13


def main():
    plt.rcParams.update(_STYLE)
    plt.rcParams.update({
        "font.size":       FONTSIZE_SMALL,
        "axes.labelsize":  FONTSIZE_SMALL,
        "axes.titlesize":  FONTSIZE,
    })

    # 3 representative models, chosen data-driven from the registry:
    #   best LLM, best NON-Gemini LLM (cross-family robustness), best embedding.
    cs = registry.category_scores()
    llm = cs[cs.model_type == "llm"].sort_values("Overall", ascending=False)
    emb = cs[cs.model_type == "embedding"].sort_values("Overall", ascending=False)

    best_llm = llm.index[0]
    best_nongemini = next(
        (m for m in llm.index if registry.LLM_META.get(m, {}).get("family") != "Gemini"),
        llm.index[1],
    )
    best_emb = emb.index[0]

    C_PRO   = "#DC2626"   # vivid red  (best LLM)
    C_FLASH = "#F59E0B"   # amber      (best non-Gemini LLM)
    C_KALM  = "#2563EB"   # vivid blue (best embedding)

    models = {
        best_llm:       (f"{registry.display_name(best_llm)} (LLM)",       C_PRO,   "-",  "D", 2.5),
        best_nongemini: (f"{registry.display_name(best_nongemini)} (LLM)", C_FLASH, "--", "s", 2.5),
        best_emb:       (f"{registry.display_name(best_emb)} (Embedding)", C_KALM,  "-",  "o", 2.5),
    }

    categories  = registry.CATEGORIES
    cat_labels  = ["Classification", "Clustering", "STS", "Pair Cls.", "Retrieval"]

    # Per-category score for each model (from the registry's macro table)
    data = {
        model_id: [float(cs.loc[model_id, cat]) for cat in categories]
        for model_id in models
    }

    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white",
                           subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)    # Classification at top
    ax.set_theta_direction(-1)         # clockwise

    for model_id, (name, color, ls, marker, lw) in models.items():
        values = data[model_id] + data[model_id][:1]
        ax.plot(
            angles, values,
            color=color, linewidth=lw, linestyle=ls,
            marker=marker, markersize=12,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.4,
            label=name, zorder=3,
        )
        ax.fill(angles, values, color=color, alpha=0.10, zorder=2)

    # ── Category labels ───────────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", pad=18)

    # ── Radial ticks ──────────────────────────────────────────────────────────
    ax.set_ylim(0.40, 1.0)
    ax.set_yticks([0.50, 0.65, 0.80])
    ax.set_yticklabels(["0.50", "0.65", "0.80"],
                       fontsize=FONTSIZE_SMALL - 1, color="#888888")

    # ── Grid & spines ─────────────────────────────────────────────────────────
    ax.spines["polar"].set_alpha(0.18)
    ax.grid(True, alpha=0.20, linewidth=0.8)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Capability Profile Across Task Categories",
        fontsize=FONTSIZE + 2, fontweight="bold", pad=28,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=FONTSIZE_SMALL,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.4,
        columnspacing=1.2,
    )

    plt.tight_layout()
    save_fig(fig, "radar_capability_profile")


if __name__ == "__main__":
    main()
