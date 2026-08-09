#!/usr/bin/env python3
"""Per-category cost-vs-performance Pareto panels (appendix).

Six small panels (Overall + five task categories) show how the cost--quality
frontier shifts by category: LLMs climb toward the frontier on reasoning-heavy
retrieval but are strictly dominated by embeddings elsewhere. LLM cost is constant
across panels (same model); only the category score changes. Scores on the 0--100
scale, cost on a log axis.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))
import registry

C_EMB = "#2563EB"
C_LLM = "#DC2626"
PANELS = ["Overall", "Retrieval", "Classification", "Clustering", "STS", "PairClassification"]
TITLES = {"Overall": "Overall", "Retrieval": "Retrieval", "Classification": "Classification",
          "Clustering": "Clustering", "STS": "STS", "PairClassification": "Pair Classification"}


def frontier(xs, ys):
    """Upper-left Pareto frontier: cheapest-first, keep running-max score."""
    order = np.argsort(xs)
    fx, fy, best = [], [], -np.inf
    for i in order:
        if ys[i] >= best - 1e-9:
            fx.append(xs[i])
            fy.append(ys[i])
            best = max(best, ys[i])
    return np.array(fx), np.array(fy)


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
    df = registry.load_scores()
    cs = registry.category_scores(df, complete_only=True)
    cost = pd.read_csv(ROOT / "data" / "cost_summary.csv").set_index("model")

    cs = cs.join(cost["total_cost"], how="inner")
    emb = cs[cs.model_type == "embedding"]
    llm = cs[cs.model_type == "llm"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), facecolor="white")
    for ax, cat in zip(axes.flat, PANELS):
        ex, ey = emb["total_cost"].values, emb[cat].values * 100
        lx, ly = llm["total_cost"].values, llm[cat].values * 100
        # Pareto frontier over ALL models (embeddings + LLMs)
        allx = np.concatenate([ex, lx])
        ally = np.concatenate([ey, ly])
        fx, fy = frontier(allx, ally)
        ax.plot(fx, fy, color="#1E3A8A", lw=2.2, zorder=2, alpha=0.8)
        ax.scatter(ex, ey, s=48, color=C_EMB, edgecolor="white", lw=0.7, zorder=3, label="Embedding")
        ax.scatter(lx, ly, s=64, marker="D", color=C_LLM, edgecolor="white", lw=0.7, zorder=4, label="LLM")
        # highlight best LLM in this category
        bi = int(np.argmax(ly))
        ax.scatter([lx[bi]], [ly[bi]], s=230, marker="*", color="#F59E0B",
                   edgecolor="#B45309", lw=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(4e-4, 5e2)
        ax.set_xticks([1e-2, 1e0, 1e2])   # sparse ticks so the larger labels don't collide
        ax.set_title(TITLES[cat], fontsize=22, fontweight="bold", pad=6)
        ax.tick_params(labelsize=18, length=0)
        ax.grid(True, which="major", color="#EEF2F6", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#CBD5E1")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}" if x >= 1 else f"${x:.2f}"))

    axes[-1, 1].set_xlabel("Cost per benchmark pass (log)", fontsize=21, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score", fontsize=21, fontweight="bold")

    h = [plt.Line2D([], [], marker="o", ls="", color=C_EMB, ms=8, label="Embedding model"),
         plt.Line2D([], [], marker="D", ls="", color=C_LLM, ms=8, label="LLM"),
         plt.Line2D([], [], marker="*", ls="", color="#F59E0B", mec="#B45309", ms=14, label="Best LLM (category)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, -0.062))
    # No suptitle: the LaTeX caption already titles the figure (avoids redundancy).
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = ROOT / "visualizations"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pareto_per_category.{ext}", dpi=180, bbox_inches="tight")
    print("Saved: pareto_per_category.png + .pdf")


if __name__ == "__main__":
    main()
