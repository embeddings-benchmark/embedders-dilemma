#!/usr/bin/env python3
"""Retrieve-then-rerank: where reranking helps (BRIGHT) vs where it doesn't (BEIR).

Two-panel grouped bar chart of avg nDCG@10 for four first-stage retrievers, each
shown as {pure, best cross-encoder, LLM listwise reranker}. Reads only the cached
pipeline results via aggregate_rerank_matrix.py — no GPU/endpoints.

Story: an LLM listwise reranker sharply lifts every first stage on reasoning-heavy
BRIGHT, but on semantic BEIR a strong embedding first stage already beats every
reranker — so reranking pays off exactly where single-vector embeddings are weakest.

Output: visualizations/reranker_beir_bright.{png,pdf}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from pub_style import save_fig
import aggregate_rerank_matrix as rr

# Three pipelines per first stage: pure, best cross-encoder, LLM listwise.
SERIES = [
    ("Pure first stage",        None,               "#2D6A9F"),   # steel blue
    ("+ Cross-encoder (Qwen3-RR-4B)", "Qwen3-RR-4B", "#E67E22"),  # orange
    ("+ LLM listwise (Qwen3.6-27B)",  "llm-qwen3.6-27b", "#C0392B"),  # brick red
]
FS_LABELS = {"BM25": "BM25", "BGE-large": "BGE-large",
             "GTE-MC-v1": "GTE-MC", "Qwen3-E-8B": "Qwen3-E-8B"}


def _panel(ax, tasks, title):
    fs_names = list(rr.FIRST_STAGES)
    x = np.arange(len(fs_names))
    w = 0.26
    for i, (label, rr_key, color) in enumerate(SERIES):
        suffix = None if rr_key is None else rr.RERANKERS[rr_key]
        vals = [rr._avg(rr.FIRST_STAGES[fs], suffix, tasks) or np.nan for fs in fs_names]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=18, fontweight="bold",
                        color="#1F2937")
    ax.set_xticks(x)
    ax.set_xticklabels([FS_LABELS[f] for f in fs_names], fontsize=18)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=8)
    ax.set_ylabel("avg nDCG@10", fontsize=19)
    ax.set_ylim(0, 72)
    ax.tick_params(axis="y", labelsize=17)
    ax.grid(True, axis="y", alpha=0.15, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial"]})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="white")
    _panel(a1, rr.BRIGHT, "BRIGHT (reasoning)")
    _panel(a2, rr.BEIR, "BEIR (semantic)")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=17,
               frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, "reranker_beir_bright")


if __name__ == "__main__":
    main()
