#!/usr/bin/env python3
"""LLM API cost breakdown — vertical figure in the MTEB(LLM) leaderboard style.

Vertical stacked bars at honest total-cost heights, with a soft background panel
and a subtle per-bar halo, brand logos and rotated model names beneath (matching
the appendix leaderboards). Total cost is labelled on top of each bar and the
reasoning ("thinking") cost is shown inside the red segment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry                                   # canonical display names / metadata
from plot_category_leaderboards import load_logo  # reuse the leaderboard logo loader

# ── Palette (reasoning red matches the appendix leaderboards) ─────────────────
C_INPUT    = "#94A3B8"   # Slate 400
C_CACHED   = "#CBD5E1"   # Slate 300
C_OUTPUT   = "#3B82F6"   # Blue
C_THINKING = "#DC2626"   # Leaderboard red
C_TEXT     = "#1F2937"   # Slate 800

_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": False,
    "axes.labelcolor":   C_TEXT,
    "text.color":        C_TEXT,
    "xtick.color":       C_TEXT,
    "ytick.color":       C_TEXT,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}


def load_data():
    df = pd.read_csv(ROOT / "data" / "llm_token_usage.csv")
    df["short_name"] = df["model"].map(registry.display_name)
    df["cost_nc"]  = (df["input_tokens"] - df["cached_tokens"]) * df["price_input_per_mtok"] / 1e6
    df["cost_ca"]  = df["cached_tokens"] * df["price_cached_per_mtok"] / 1e6
    df["cost_out"] = df["output_tokens"] * df["price_output_per_mtok"] / 1e6
    df["cost_th"]  = df["thinking_tokens"] * df["price_output_per_mtok"] / 1e6
    df["cost_total"]  = df["cost_nc"] + df["cost_ca"] + df["cost_out"] + df["cost_th"]
    df["thinking_pct"] = (df["cost_th"] / df["cost_total"] * 100).fillna(0.0)
    return df.sort_values("cost_total", ascending=False).reset_index(drop=True)


def plot_cost_breakdown(df, out_dir):
    plt.rcParams.update(_STYLE)

    df = df.sort_values("cost_total", ascending=False).reset_index(drop=True)
    n = len(df)
    xs = np.arange(n, dtype=float)
    tot = df["cost_total"]
    mx = tot.max()

    segs = [
        (df["cost_nc"],  C_INPUT,    "Input (non-cached)"),
        (df["cost_ca"],  C_CACHED,   "Cached input"),
        (df["cost_out"], C_OUTPUT,   "Standard output"),
        (df["cost_th"],  C_THINKING, "Reasoning (“thinking”)"),
    ]

    # Leaderboard-style padding ratios
    FLOOR = 0.0
    RANGE = mx - FLOOR
    Y_TOP = mx + RANGE * 0.30
    Y_LOGO = FLOOR - RANGE * 0.085
    Y_TEXT = FLOOR - RANGE * 0.165
    Y_BOT = FLOOR - RANGE * 0.42

    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_xlim(xs[0] - 1.0, xs[-1] + 1.0)

    # Soft background panel behind the bar field
    panel = mpatches.FancyBboxPatch(
        (xs[0] - 0.65, FLOOR), (xs[-1] - xs[0]) + 1.3, mx * 1.06,
        boxstyle="round,pad=0.02,rounding_size=6", mutation_aspect=0.02,
        facecolor="#FBFCFE", edgecolor="#EEF2F7", lw=1.2, zorder=0)
    ax.add_patch(panel)

    for g in range(0, int(mx) + 30, 50):
        ax.axhline(g, color="#F1F5F9", lw=1.2, zorder=1)
    ax.axhline(FLOOR, color="#CBD5E1", lw=1.6, zorder=2)

    bar_w = 0.66
    # Subtle halo behind each bar
    ax.bar(xs, tot.values, width=bar_w + 0.22, bottom=FLOOR, color="#94A3B8",
           alpha=0.10, zorder=2, edgecolor="none")
    # Stacked segments
    bottoms = np.zeros(n)
    for v, c, lab in segs:
        ax.bar(xs, v.values, width=bar_w, bottom=bottoms, color=c,
               edgecolor="white", lw=1.6, label=lab, zorder=3)
        bottoms = bottoms + v.values

    for i in range(n):
        h = tot.iloc[i]
        # Total cost on top
        ax.text(xs[i], h + RANGE * 0.055, f"${h:,.0f}", ha="center", va="bottom",
                fontsize=31, fontweight="bold", color=C_TEXT, zorder=6)
        # Reasoning cost inside the red block
        th = df["cost_th"].iloc[i]
        if th / mx > 0.05:
            base_red = h - th
            ax.text(xs[i], base_red + th / 2, f"${th:,.0f}", ha="center", va="center",
                    fontsize=25, fontweight="bold", color="white", zorder=7)
        # Logo + rotated model name
        m = df["model"].iloc[i]
        key = registry.LLM_META.get(m, {}).get("logo", "")
        img = load_logo(key, target_h=46) if key else None
        if img is not None:
            ax.add_artist(AnnotationBbox(img, (xs[i], Y_LOGO), frameon=False, pad=0.0,
                                         clip_on=False, annotation_clip=False, zorder=6))
        ax.text(xs[i], Y_TEXT, registry.display_name(m), ha="right", va="top", rotation=38,
                rotation_mode="anchor", fontsize=27, fontweight="bold", color=C_TEXT,
                clip_on=False, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([0, 50, 100, 150])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.tick_params(axis="y", labelsize=24, length=0)
    ax.tick_params(axis="x", length=0)

    ax.legend(fontsize=24, loc="upper right", frameon=False, handletextpad=0.7,
              labelspacing=0.55, borderaxespad=1.4)
    fig.suptitle("API cost breakdown: reasoning tokens dominate LLM inference cost",
                 fontsize=40, fontweight="bold", x=0.5, y=1.0)
    fig.tight_layout()

    out_dir.mkdir(exist_ok=True, parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"cost_breakdown.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: {out_dir / 'cost_breakdown.png'}")


def main():
    print("Generating LLM cost breakdown (leaderboard style)...")
    df = load_data()
    out_dir = ROOT / "visualizations"
    plot_cost_breakdown(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
