#!/usr/bin/env python3
"""Same-hardware throughput: open-weight LLMs vs embedding models on one H100.

Answers reviewers 6ux1 / rYBc: the throughput gap is architectural, not an API
artifact. Both paradigms are served on a single NVIDIA H100 80GB and measured in
tokens/second.

  * Embeddings: measured median tok/s from data/embedding_throughput.csv
    (dedicated throughput benchmark, seq len 512, max batch).
  * Open-weight LLMs: vLLM serving, BF16, TP=1, 256 concurrent, 200/100 in/out
    tokens. Values are the measured numbers reported in the COLM rebuttal
    (general comment, pt. 4) — the raw sweep is not in this repo, so they are
    entered here as published constants (see LLM_TPUT below).

This replaces the old two-panel throughput_comparison figure, whose API-concurrency
panel depended on throughput_results/ (no longer in the repo). The same-hardware
result supersedes that panel: it shows the gap holds without any API rate limit.

Output: visualizations/throughput_comparison.{png,pdf}  (drop-in for the paper)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pub_style import save_fig, C_EMB, C_LLM

# Open-weight LLM throughput on 1x H100 (vLLM, BF16, TP=1, 256 concurrent,
# 200/100 in/out) is read from data/llm_throughput_h100.csv (provenance in that
# file's `source` column) so the figure is fully reproducible/auditable.
def _load_llm_tput():
    df = pd.read_csv(ROOT / "data" / "llm_throughput_h100.csv")
    return {r["label"]: {"tok_s": float(r["tok_per_sec"]), "req_s": float(r["req_per_sec"])}
            for _, r in df.iterrows()}

# Representative embeddings spanning the size range (steady, readable spread).
EMB_SHOW = [
    ("intfloat/multilingual-e5-small",           "mE5-Small (118M)"),
    ("jinaai/jina-embeddings-v5-text-nano",      "Jina-v5-Nano (212M)"),
    ("google/embeddinggemma-300m",               "EmbGemma (300M)"),
    ("jinaai/jina-embeddings-v5-text-small",     "Jina-v5-S (596M)"),
    ("Qwen/Qwen3-Embedding-8B",                  "Qwen3-Emb (8B)"),
    ("tencent/KaLM-Embedding-Gemma3-12B-2511",   "KaLM (12B)"),
    ("codefuse-ai/F2LLM-v2-14B",                 "F2LLM (14B)"),
]


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial"]})
    thr = pd.read_csv(ROOT / "data" / "embedding_throughput.csv").set_index("model_id")
    LLM_TPUT = _load_llm_tput()

    rows = []
    for mid, label in EMB_SHOW:
        if mid in thr.index:
            rows.append((label, float(thr.loc[mid, "median_tok_per_sec"]), C_EMB, "embedding"))
    for label, v in LLM_TPUT.items():
        rows.append((label, float(v["tok_s"]), C_LLM, "llm"))

    rows.sort(key=lambda r: r[1])          # slowest (LLMs) at bottom
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=1.4, zorder=3, height=0.74)

    for yi, v in zip(y, vals):
        txt = f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
        ax.text(v * 1.18, yi, txt + " tok/s", va="center", ha="left",
                fontsize=13, fontweight="bold", color="#1F2937")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xscale("log")
    ax.set_xlim(3.5e3, 1.1e7)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else (f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}")))
    ax.xaxis.set_minor_locator(ticker.NullLocator())          # no minor-tick clutter
    ax.set_xlabel("Throughput on 1$\\times$H100 (tokens/second, log scale)",
                  fontsize=13.5, fontweight="bold")
    ax.tick_params(axis="both", length=0, labelsize=12.5)
    ax.grid(False)                                             # no gridlines; every bar is labelled
    for sp in ax.spines.values():
        sp.set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_EMB, label="Embedding model"),
                       Patch(color=C_LLM, label="Open-weight LLM (vLLM)")],
              loc="lower right", fontsize=12.5, frameon=False)

    fig.tight_layout()
    save_fig(fig, "throughput_comparison")


if __name__ == "__main__":
    main()
