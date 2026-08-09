#!/usr/bin/env python3
"""Single source of truth for the MTEB(LLM) paper.

Every figure and table script derives its model list, task set, display names,
scores, and LLM pricing from here, so the whole paper regenerates consistently
from data/scores.csv.

Key definitions
---------------
* Canonical task set: the union of tasks covered by embedding models in
  scores.csv (currently 37 tasks: Cls 8, Clust 9, PairCls 4, Retr 6, STS 10).
* Complete model: a model that covers every canonical task. Partial-coverage
  models (some cross-family LLMs still backfilling; a couple of embeddings) are
  excluded here and will be picked up automatically once their runs finish.
* Overall score: MACRO average = mean of the 5 per-category means. This matches
  the paper's per-category main-results table and avoids category-size bias
  (STS has 10 tasks, PairClassification only 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pub_style import short_name  # noqa: E402  (canonical display-name map)

SCORES_CSV = ROOT / "data" / "scores.csv"

CATEGORIES = ["Classification", "Clustering", "STS", "PairClassification", "Retrieval"]

# ─────────────────────────────────────────────────────────────────────────────
# LLM pricing (USD per 1 M tokens) for the complete, cost-comparable LLMs.
# Gemini: https://ai.google.dev/gemini-api/docs/pricing (March 2026).
# Non-Gemini: OpenRouter prompt/completion $/MTok (June 2026); cached = input/10.
# The keys of this dict ARE the headline LLM set — a model only enters the paper's
# cost figures/tables once it has both a price here and full task coverage.
# ─────────────────────────────────────────────────────────────────────────────
LLM_PRICING = {
    "google__gemini-3.1-flash-lite-preview": {"input": 0.25,   "output": 1.50,   "cached": 0.025},
    "google__gemini-3-flash-preview":        {"input": 0.50,   "output": 3.00,   "cached": 0.05},
    "google__gemini-3.1-pro-preview":        {"input": 2.00,   "output": 12.00,  "cached": 0.20},
    "deepseek-r1":                           {"input": 0.70,   "output": 2.50,   "cached": 0.070},
    "deepseek__deepseek-v4-flash":           {"input": 0.0983, "output": 0.1966, "cached": 0.00983},
    "glm-4.7":                               {"input": 0.40,   "output": 1.75,   "cached": 0.040},
    "kimi-k2.6":                             {"input": 0.684,  "output": 3.42,   "cached": 0.0684},
    "minimax-m2.7":                          {"input": 0.279,  "output": 1.20,   "cached": 0.0279},
    "qwen3.6-27b":                           {"input": 0.29,   "output": 3.20,   "cached": 0.029},
    "qwen3.6-35b-a3b":                       {"input": 0.14,   "output": 1.00,   "cached": 0.014},
}

# LLM display metadata: bar color + logo key (logo file in visualizations/logos/;
# None when we have no logo asset yet — the plot falls back to a colored bar).
LLM_META = {
    "google__gemini-3.1-pro-preview":        {"family": "Gemini",   "color": "#3186FF", "logo": "gemini"},
    "google__gemini-3-flash-preview":        {"family": "Gemini",   "color": "#60A5FA", "logo": "gemini"},
    "google__gemini-3.1-flash-lite-preview": {"family": "Gemini",   "color": "#93C5FD", "logo": "gemini"},
    "deepseek-r1":                           {"family": "DeepSeek",  "color": "#4D6BFE", "logo": "deepseek"},
    "deepseek__deepseek-v4-flash":           {"family": "DeepSeek",  "color": "#8AA0FF", "logo": "deepseek"},
    "glm-4.7":                               {"family": "GLM",       "color": "#2E5CE6", "logo": "glm"},
    "kimi-k2.6":                             {"family": "Kimi",      "color": "#111827", "logo": "kimi"},
    "minimax-m2.7":                          {"family": "MiniMax",   "color": "#E1483B", "logo": "minimax"},
    "qwen3.6-27b":                           {"family": "Qwen",      "color": "#6E2CF4", "logo": "qwen"},
    "qwen3.6-35b-a3b":                       {"family": "Qwen",      "color": "#9A6BF7", "logo": "qwen"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + canonical sets
# ─────────────────────────────────────────────────────────────────────────────
def load_scores(path: Path | None = None) -> pd.DataFrame:
    """Load the long-format per-task score table."""
    return pd.read_csv(path or SCORES_CSV)


def _df(df: pd.DataFrame | None) -> pd.DataFrame:
    return load_scores() if df is None else df


def task_category_map(df: pd.DataFrame | None = None) -> dict[str, str]:
    df = _df(df)
    return dict(zip(df["task"], df["task_category"]))


def canonical_tasks(df: pd.DataFrame | None = None) -> list[str]:
    """The canonical task set = union of tasks covered by embedding models."""
    df = _df(df)
    return sorted(df.loc[df.model_type == "embedding", "task"].unique())


def canonical_tasks_by_category(df: pd.DataFrame | None = None) -> dict[str, list[str]]:
    df = _df(df)
    tcmap = task_category_map(df)
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for t in canonical_tasks(df):
        out.setdefault(tcmap[t], []).append(t)
    return {c: sorted(ts) for c, ts in out.items()}


def complete_models(df: pd.DataFrame | None = None, model_type: str | None = None) -> list[str]:
    """Models covering every canonical task (optionally filtered by model_type)."""
    df = _df(df)
    canon = set(canonical_tasks(df))
    keep = []
    for m, g in df.groupby("model"):
        if model_type is not None and g["model_type"].iloc[0] != model_type:
            continue
        if canon.issubset(set(g["task"])):
            keep.append(m)
    return sorted(keep)


def coverage_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-model task count vs the canonical set — for auditing what's dropped."""
    df = _df(df)
    n_canon = len(canonical_tasks(df))
    rows = []
    for m, g in df.groupby("model"):
        n = g["task"].nunique()
        rows.append({
            "model": m,
            "model_type": g["model_type"].iloc[0],
            "n_tasks": n,
            "complete": n >= n_canon,
        })
    return pd.DataFrame(rows).sort_values(["model_type", "complete", "n_tasks"],
                                          ascending=[True, False, False])


# ─────────────────────────────────────────────────────────────────────────────
# Canonical scores (MACRO overall = mean of category means)
# ─────────────────────────────────────────────────────────────────────────────
def category_scores(df: pd.DataFrame | None = None, complete_only: bool = True) -> pd.DataFrame:
    """Wide per-model table: one column per category + Overall (macro) + model_type.

    Rows are indexed by model. Category cells are the mean over that category's
    canonical tasks; Overall is the mean of the (present) category means.
    """
    df = _df(df)
    if complete_only:
        keep = set(complete_models(df))
        df = df[df.model.isin(keep)]
    canon = set(canonical_tasks(df))
    df = df[df.task.isin(canon)]

    per_cat = (
        df.groupby(["model", "model_type", "task_category"])["score"]
        .mean()
        .unstack("task_category")
    )
    # Order columns; keep only categories that exist
    cols = [c for c in CATEGORIES if c in per_cat.columns]
    per_cat = per_cat[cols]
    per_cat["Overall"] = per_cat[cols].mean(axis=1)
    per_cat = per_cat.reset_index().set_index("model")
    return per_cat


def overall_scores(df: pd.DataFrame | None = None, complete_only: bool = True) -> pd.Series:
    """model -> macro Overall score."""
    cs = category_scores(df, complete_only=complete_only)
    return cs["Overall"]


def display_name(model: str) -> str:
    return short_name(model)


if __name__ == "__main__":
    # Quick audit when run directly.
    df = load_scores()
    print("Canonical tasks by category:")
    for c, ts in canonical_tasks_by_category(df).items():
        print(f"  {c:20s} {len(ts):2d}")
    print(f"  {'TOTAL':20s} {len(canonical_tasks(df)):2d}")
    print("\nComplete LLMs:", len(complete_models(df, 'llm')))
    for m in complete_models(df, "llm"):
        print("   ", display_name(m))
    print("\nComplete embeddings:", len(complete_models(df, "embedding")))
    cs = category_scores(df)
    print("\nTop by Overall (macro):")
    for m, r in cs.sort_values("Overall", ascending=False).head(8).iterrows():
        print(f"   {display_name(m):22s} {r['model_type']:10s} {r['Overall']:.4f}")
    print("\nDropped (incomplete):")
    rep = coverage_report(df)
    for _, r in rep[~rep.complete].iterrows():
        print(f"   {display_name(r.model):24s} {r.model_type:10s} {r.n_tasks}/{len(canonical_tasks(df))}")
