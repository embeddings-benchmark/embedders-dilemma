#!/usr/bin/env python3
"""Task-Level Correlation Heatmap

Produces a lower-triangle heatmap showing the Pearson correlation of models
based on their behavior across the canonical MTEB(LLM) tasks.
Demonstrates orthogonality of LLM vs Embedding capabilities.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plotting.pub_style import save_fig
from plotting import registry

_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}

def main():
    plt.rcParams.update(_STYLE)
    
    # Load scores; keep only complete models over the canonical task set.
    df_all = registry.load_scores()
    canon = set(registry.canonical_tasks(df_all))
    models_to_use = set(registry.complete_models(df_all))
    llm_names = {registry.display_name(m)
                 for m in registry.complete_models(df_all, "llm")}
    df = df_all[df_all.model.isin(models_to_use) & df_all.task.isin(canon)].copy()

    df["name"] = df["model"].map(registry.display_name)
    pivot = df.pivot_table(index="name", columns="task", values="score")
    
    model_order_df = df.groupby(["name", "model_type"])["score"].mean().reset_index()
    model_order_df["sort_pri"] = model_order_df["model_type"].map({"llm": 0, "embedding": 1})
    model_order = model_order_df.sort_values(["sort_pri", "score"], ascending=[True, False])["name"].tolist()
    
    pivot = pivot.reindex(model_order)
    corr_matrix = pivot.T.corr()
    
    fig, ax = plt.subplots(figsize=(20, 18), facecolor="white")
    n = len(model_order)

    # Lower-triangle mask (hide the redundant upper half incl. diagonal).
    corr = corr_matrix.values
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = np.ma.masked_where(mask, corr)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(corr_masked, cmap=cmap, vmin=0.3, vmax=1.0, aspect="equal")

    # Thin white cell borders (replicates seaborn's linewidths look).
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Pearson Correlation", labelpad=15, fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    n_llms = len([m for m in model_order if m in llm_names])
    # Dashed divider around the LLM block (cells span [i-0.5, i+0.5]).
    b = n_llms - 0.5
    ax.plot([-0.5, b], [b, b], color="#1E293B", lw=2, linestyle="--")
    ax.plot([b, b], [b, n - 0.5], color="#1E293B", lw=2, linestyle="--")

    ax.set_title("Model Capability Correlation Across Tasks", fontsize=26, fontweight="bold", pad=20)
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, fontsize=17, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index, fontsize=17, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out_dir = ROOT / "visualizations"
    out_dir.mkdir(exist_ok=True)
    save_fig(fig, "task_correlation_heatmap", out_dir)
    print("Saved task_correlation_heatmap")

if __name__ == "__main__":
    main()
