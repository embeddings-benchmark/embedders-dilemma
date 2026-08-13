#!/usr/bin/env python3
"""Additional publication-quality figures for the LLM-vs-Embeddings paper.

Figure 1 – Embedding Scaling Curve  (score_vs_params.png/pdf)
    Scatter of avg_score vs parameter count (log scale) for 26 embedding models,
    with LLM reference lines and a log-linear trend line.

Figure 2 – Cost Efficiency  (score_per_dollar.png/pdf)
    Horizontal bar chart of score-per-dollar for the top-15 embedding models
    plus the 3 LLMs, coloured by model type.

Usage:
    python scripts/plot_additional.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# SHARED STYLE HELPERS

C_EMB = "#3A86FF"   # vivid blue  (same palette as plot_pareto)
C_LLM = "#FF006E"   # vivid pink-red

# Model-family colours for the scaling-curve plot
FAMILY_COLOURS = {
    "Qwen3":       "#E63946",
    "F2LLM":       "#457B9D",
    "Jina":        "#2A9D8F",
    "mE5":         "#E9C46A",
    "GTE-Qwen2":   "#F4A261",
    "Other":       "#6C757D",
}


def _family(short_name: str) -> str:
    """Map a short display name to a model-family key."""
    sn = short_name.lower()
    if "qwen3-e" in sn:
        return "Qwen3"
    if "f2llm" in sn:
        return "F2LLM"
    if "jina" in sn:
        return "Jina"
    if sn.startswith("me5") or sn.startswith("e5"):
        return "mE5"
    if "gte-qwen2" in sn:
        return "GTE-Qwen2"
    return "Other"


def _pub_style():
    """Set publication-quality rcParams (serif, 11 pt, thin spines)."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         11,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    })


# DATA LOADING

def load_data():
    """Return (cost_df, throughput_df) from the CSV files in data/."""
    cost_df = pd.read_csv(ROOT / "data" / "cost_summary.csv")
    tp_df   = pd.read_csv(ROOT / "data" / "embedding_throughput.csv")
    # Normalise model key in throughput file to match cost_summary
    tp_df["model"] = tp_df["model_id"].str.replace("/", "__", regex=False)
    return cost_df, tp_df


# FIGURE 1 — EMBEDDING SCALING CURVE

def plot_score_vs_params(cost_df, tp_df, out_dir):
    """Scatter: avg_score vs parameter count for embedding models."""
    _pub_style()

    # Merge to get params alongside avg_score
    emb = cost_df[cost_df["type"] == "Embedding"].copy()
    emb = emb.merge(tp_df[["model", "params"]], on="model", how="left")
    emb = emb.dropna(subset=["params"])
    emb["params_b"] = emb["params"] / 1e9          # billions

    # LLM reference scores
    gemini_pro_score   = float(cost_df.loc[cost_df["short_name"] == "Gemini 3.1 Pro",  "avg_score"].iloc[0])
    gemini_flash_score = float(cost_df.loc[cost_df["short_name"] == "Gemini 3 Flash", "avg_score"].iloc[0])

    # Family assignments
    emb["family"] = emb["short_name"].apply(_family)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- scatter by family ---
    for fam, colour in FAMILY_COLOURS.items():
        sub = emb[emb["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub["params_b"], sub["avg_score"],
                   c=colour, marker="o", s=60, alpha=0.85,
                   edgecolors="white", linewidth=0.5,
                   label=fam, zorder=4)

    # --- label notable points ---
    notable = {
        "KaLM-12B", "Qwen3-E-8B", "Qwen3-E-4B", "Qwen3-E-0.6B",
        "Jina-v5-S", "Jina-v5-Nano", "EmbGemma-300M",
        "Octen-8B", "F2LLM-14B", "F2LLM-0.6B",
        "mE5-S", "mE5-L-Inst",
        "GritLM-7B", "E5-Mistral-7B",
        "Nemotron-8B",
    }
    for _, r in emb.iterrows():
        if r.short_name in notable:
            ax.annotate(
                r.short_name,
                (r.params_b, r.avg_score),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color="#333",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=6,
            )

    # --- LLM reference lines ---
    ax.axhline(gemini_pro_score, linestyle="--", linewidth=1.2,
               color=C_LLM, alpha=0.7, zorder=2)
    ax.text(emb["params_b"].max() * 1.08, gemini_pro_score,
            f"Gemini 3.1 Pro ({gemini_pro_score:.3f})",
            fontsize=8.5, color=C_LLM, va="center", ha="left",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=6)

    ax.axhline(gemini_flash_score, linestyle="--", linewidth=1.2,
               color="#E76F51", alpha=0.7, zorder=2)
    ax.text(emb["params_b"].max() * 1.08, gemini_flash_score,
            f"Gemini 3 Flash ({gemini_flash_score:.3f})",
            fontsize=8.5, color="#E76F51", va="center", ha="left",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=6)

    # --- log-linear trend line ---
    log_params = np.log10(emb["params_b"].values)
    scores     = emb["avg_score"].values
    coeffs     = np.polyfit(log_params, scores, 1)
    xs = np.linspace(log_params.min() - 0.15, log_params.max() + 0.15, 200)
    ax.plot(10**xs, np.polyval(coeffs, xs),
            color="#888", linewidth=1.8, linestyle="-.", alpha=0.6,
            label=f"Log-linear fit (slope={coeffs[0]:.3f})", zorder=3)

    # --- axes ---
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.1f}B" if x >= 1 else f"{x*1000:.0f}M"))
    ax.set_xlabel("Parameter Count (log scale)", fontsize=12, labelpad=8)
    ax.set_ylabel("Average Score (38 MTEB Tasks)", fontsize=12, labelpad=8)
    ax.set_title("Embedding Model Scaling: Score vs. Parameter Count",
                 fontsize=14, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.07, linewidth=0.4)

    # y-axis padding
    ylo = min(emb["avg_score"].min(), 0.47) - 0.02
    yhi = gemini_pro_score + 0.03
    ax.set_ylim(ylo, yhi)
    # x-axis padding for labels on the right
    ax.set_xlim(emb["params_b"].min() * 0.6, emb["params_b"].max() * 3.0)

    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92,
              edgecolor="#ccc", fancybox=True, ncol=2)

    plt.tight_layout()

    fig.savefig(out_dir / "score_vs_params.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "score_vs_params.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'score_vs_params.png'}")
    print(f"Saved: {out_dir / 'score_vs_params.pdf'}")


# FIGURE 2 — COST EFFICIENCY (SCORE PER DOLLAR)

def plot_score_per_dollar(cost_df, out_dir):
    """Horizontal bar chart of score-per-dollar."""
    _pub_style()

    df = cost_df.copy()
    df["spd"] = df["avg_score"] / df["total_cost"]

    emb = df[df["type"] == "Embedding"].sort_values("spd", ascending=False)
    llm = df[df["type"] == "LLM"].sort_values("spd", ascending=False)

    # Top 15 embeddings + all 3 LLMs
    top_emb = emb.head(15)
    shown = pd.concat([top_emb, llm], ignore_index=True)
    # Sort ascending so highest bar is at the top of the horizontal chart
    shown = shown.sort_values("spd", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colours = [C_LLM if t == "LLM" else C_EMB for t in shown["type"]]

    bars = ax.barh(shown["short_name"], shown["spd"],
                   color=colours, edgecolor="white", linewidth=0.5,
                   height=0.7, zorder=3)

    # Annotate values
    for bar, spd_val in zip(bars, shown["spd"]):
        # Place text just to the right of the bar
        # Use .3f for very small values so they don't display as "0.0"
        label = f"{spd_val:,.3f}" if spd_val < 1 else f"{spd_val:,.1f}"
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="left", fontsize=8, color="#333",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=5)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:.2f}"))
    ax.set_xlabel("Score per Dollar  (avg_score / total_cost, log scale)",
                  fontsize=12, labelpad=8)
    ax.set_title("Cost Efficiency: Score per Dollar",
                 fontsize=14, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="x", alpha=0.18, linewidth=0.7)
    ax.grid(True, which="minor", axis="x", alpha=0.07, linewidth=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_EMB, edgecolor="white", label="Embedding"),
        Patch(facecolor=C_LLM, edgecolor="white", label="LLM"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right",
              framealpha=0.92, edgecolor="#ccc", fancybox=True)

    # x-axis padding for annotations
    xmax = shown["spd"].max()
    ax.set_xlim(right=xmax * 3.0)

    plt.tight_layout()

    fig.savefig(out_dir / "score_per_dollar.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "score_per_dollar.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'score_per_dollar.png'}")
    print(f"Saved: {out_dir / 'score_per_dollar.pdf'}")


# MAIN

def main():
    cost_df, tp_df = load_data()
    out_dir = ROOT / "visualizations"
    out_dir.mkdir(exist_ok=True)

    print("Figure 1: Embedding Scaling Curve ...")
    plot_score_vs_params(cost_df, tp_df, out_dir)

    print("\nFigure 2: Cost Efficiency ...")
    plot_score_per_dollar(cost_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
