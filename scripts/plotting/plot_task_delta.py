#!/usr/bin/env python3
"""Task-level performance — range bars (Full-Page Appendix Overhaul).

For each of 37 tasks, shows the embedding score range [min, max] as a
horizontal bar with the best LLM score overlaid as a diamond marker.
Tasks are grouped by category and formatted into a highly comfortable
vertical layout that cleanly fills a single standard Appendix page.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pub_style import CAT_COLORS, CAT_ORDER
import registry   # single source of truth

C_LLM = "#DC2626"
C_EMB = "#2563EB"

_STYLE = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth":    2.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.size":         101,
    "axes.labelsize":    119,
    "xtick.labelsize":   97,
    "ytick.labelsize":   97,
    "legend.fontsize":   101,
}

_NAME_CLEAN = {
    "AILAStatutes":                           "AILA Statutes",
    "AmazonCounterfactualClassification":     "Amazon CF",
    "ArxivClusteringP2P":                     "Arxiv P2P",
    "ArxivClusteringS2S":                     "Arxiv S2S",
    "BIOSSES":                                "BIOSSES",
    "Banking77Classification":                "Banking77",
    "BigPatentClustering":                    "BigPatent",
    "BiorxivClusteringP2PV2":                 "Biorxiv P2P",
    "ImdbClassification":                     "IMDB",
    "LegalBenchCorporateLobbying":            "LegalBench Corp.",
    "LegalBenchPC":                           "LegalBench PC",
    "MTOPDomainClassification":               "MTOP Domain",
    "MassiveIntentClassification":            "Massive Intent",
    "MassiveScenarioClassification":          "Massive Scenario",
    "MedrxivClusteringP2PV2":                "Medrxiv P2P",
    "MedrxivClusteringS2SV2":                "Medrxiv S2S",
    "RTE3PC":                                 "RTE3",
    "RedditClusteringP2P":                    "Reddit P2P",
    "SICKR":                                  "SICK-R",
    "STS12":                                  "STS12",
    "STS13":                                  "STS13",
    "STS14":                                  "STS14",
    "STS15":                                  "STS15",
    "STS16":                                  "STS16",
    "STS17":                                  "STS17",
    "STS22v2":                                "STS22",
    "STSBenchmark":                           "STS-B",
    "SpartQA":                                "SpartQA",
    "SprintDuplicateQuestionsPC":             "Sprint DupQ",
    "StackExchangeClusteringP2PV2":          "StackEx P2P",
    "StackExchangeClusteringV2":             "StackExchange",
    "TempReasonL1":                           "TempReason L1",
    "ToxicConversationsClassification":       "ToxicConv",
    "TweetSentimentExtractionClassification": "TweetSentiment",
    "TwentyNewsgroupsClusteringV2":           "20Newsgroups",
    "TwitterHjerneRetrieval":                 "Twitter Hjerne",
    "TwitterURLCorpusPC":                     "Twitter URL",
    "WinoGrande":                             "WinoGrande",
    # Retrieval tasks added in the rebuttal task set
    "FQuADRetrieval":                         "FQuAD",
    "HC3FinanceRetrieval":                    "HC3 Finance",
    "LegalBenchConsumerContractsQA":          "Consumer Contracts",
    "PublicHealthQA":                         "PublicHealth QA",
}

def strip_prefix(name):
    raw = name[3:] if name.startswith("LLM") else name
    return _NAME_CLEAN.get(raw, raw)

def main():
    plt.rcParams.update(_STYLE)

    # Complete models over the canonical task set only, so ranges/best-LLM
    # markers are computed from fully-comparable models.
    df = registry.load_scores()
    canon = set(registry.canonical_tasks(df))
    keep = set(registry.complete_models(df))
    df = df[df.model.isin(keep) & df.task.isin(canon)]
    emb = df[df.model_type == "embedding"]
    llm = df[df.model_type == "llm"]

    tasks = sorted(emb["task"].unique())
    rows = []
    for t in tasks:
        e   = emb[emb.task == t]["score"]
        l   = llm[llm.task == t]["score"]
        cat = df[df.task == t]["task_category"].iloc[0]
        rows.append({
            "task":       t,
            "task_short": strip_prefix(t),
            "category":   cat,
            "emb_min":    e.min(),
            "emb_max":    e.max(),
            "llm_best":   float(l.max()) if len(l) > 0 else np.nan,
            "delta":      (float(l.max()) - float(e.max())) if len(l) > 0 else np.nan,
        })
    td = pd.DataFrame(rows)

    td["cat_order"] = td["category"].map({c: i for i, c in enumerate(CAT_ORDER)})
    td = td.sort_values(["cat_order", "delta"], ascending=[True, False]).reset_index(drop=True)

    n = len(td)
    
    # ── Ultra-Wide, Balanced Portrait Layout ──
    # Giving it enormous horizontal width while keeping vertical height proportionate.
    fig, ax = plt.subplots(figsize=(75.0, max(102, n * 2.72)), facecolor="white")
    ax.set_facecolor("white")

    # Range bars bounds purely for drawing the background alternating colors
    cat_yranges = {}
    for i, (_, r) in enumerate(td.iterrows()):
        y = n - 1 - i
        cat = r["category"]
        if cat not in cat_yranges:
            cat_yranges[cat] = [y, y]
        else:
            cat_yranges[cat][0] = min(cat_yranges[cat][0], y)
            cat_yranges[cat][1] = max(cat_yranges[cat][1], y)

    # Alternate background bands
    for cat in CAT_ORDER:
        if cat not in cat_yranges:
            continue
        y_lo, y_hi = cat_yranges[cat]
        ci = CAT_ORDER.index(cat)
        if ci % 2 == 0:
            ax.axhspan(y_lo - 0.5, y_hi + 0.5, color="#F1F5F9", zorder=0, lw=0)

    # Data
    for i, (_, r) in enumerate(td.iterrows()):
        y     = n - 1 - i
        cat   = r["category"]
        color = CAT_COLORS.get(cat, "#888")

        # Clean, elegant bars resting comfortably inside massive vertical whitespace
        ax.plot([r["emb_min"], r["emb_max"]], [y, y],
                color=color, linewidth=60, alpha=0.3,
                solid_capstyle="round", zorder=1)

        ax.scatter(r["emb_max"], y, c=color, s=3000, marker="o",
                   edgecolors="white", linewidth=5.0, zorder=3, alpha=0.95)

        if not np.isnan(r["llm_best"]):
            ax.scatter(r["llm_best"], y, c=C_LLM, s=3800, marker="D",
                       edgecolors="white", linewidth=6.0, zorder=4)

    # Y-labels
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(
        [td.iloc[n - 1 - i]["task_short"] for i in range(n)],
        fontsize=100,
    )

    # Category block separator lines
    prev_cat = None
    for i, (_, r) in enumerate(td.iterrows()):
        y   = n - 1 - i
        cat = r["category"]
        if cat != prev_cat and prev_cat is not None:
            ax.axhline(y + 0.5, color="#CBD5E1", linewidth=1.0, zorder=5)
        prev_cat = cat

    # Right margin categorical group labels
    cat_positions = {}
    for i, (_, r) in enumerate(td.iterrows()):
        y   = n - 1 - i
        cat = r["category"]
        cat_positions.setdefault(cat, []).append(y)

    for cat, positions in cat_positions.items():
        mid_y = float(np.mean(positions))
        color = CAT_COLORS.get(cat, "#888")
        label = cat.replace("PairClassification", "Pair\nCls.")
        ax.text(1.02, mid_y, label,
                transform=ax.get_yaxis_transform(),
                fontsize=106, fontweight="bold", color=color,
                va="center", ha="left", linespacing=1.4)

    ax.set_xlabel("Task Performance Score", fontsize=119, labelpad=38, fontweight="medium")
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-1, n)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=20)

    # Global legend positioned cleanly at the bottom center with breathing room below X-axis
    handles = [
        mpatches.Patch(color=C_EMB, alpha=0.32, label="Embedding rating [Min -> Max]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_EMB,
               markersize=110, label="Top Embedding"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_LLM,
               markersize=110, label="Top LLM"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=101, framealpha=0.95, edgecolor="#e2e8f0")

    ax.set_title("Task Performance Ranges: Visualizing LLM Capability Overlaps",
                 fontsize=139, fontweight="bold", y=1.01)

    out_dir = ROOT / "visualizations"
    out_dir.mkdir(exist_ok=True)
    
    # Manually adjust tight layout to ensure legend fits without cropping
    plt.tight_layout(rect=[0, 0.06, 0.93, 1.0])
    
    fig.savefig(out_dir / "task_performance_range.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "task_performance_range.png", dpi=100, bbox_inches="tight")
    print(f"Locked in Full-Page Appendix Range Plot: {out_dir}/task_performance_range.png")
    
if __name__ == "__main__":
    main()
