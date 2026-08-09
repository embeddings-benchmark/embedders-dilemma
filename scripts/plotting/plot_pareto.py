#!/usr/bin/env python3
"""Cost-Performance Pareto Frontier: LLMs vs Embedding Models.

Produces a publication-quality scatter plot with Pareto frontier.
  X-axis: Total benchmark cost (USD, log scale)
  Y-axis: MACRO average score across the canonical MTEB(LLM) tasks (mean of the
          5 per-category means), complete models only, via registry.py

Inputs:
  - scores.csv                  per-task scores for all models
  - embedding_throughput.csv    cost_usd_per_mtok for 26 embedding models
  - llm_results/                JSON result files with token usage stats

Outputs:
  - visualizations/pareto_cost_performance.{png,pdf}
  - cost_summary.csv            all models with scores + costs
  - llm_token_usage.csv         detailed token + pricing breakdown for LLMs

Usage:
    python plot_pareto.py
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# LLM pricing + canonical macro scoring live in the shared registry (single
# source of truth). LLM_PRICING keys ARE the complete, cost-comparable LLM set;
# partial-coverage models are excluded until their runs finish.
from registry import LLM_PRICING, category_scores as _category_scores  # noqa: E402

# Per-model token counts computed by scripts/count_tokens_per_model.py using
# each model's OWN tokenizer on the 38 mteb/llm-eval-* datasets.  Zero mismatch.
# Saved in embedding_costs_per_model.csv.  Falls back to 7.25M (median) if missing.
EMB_COSTS_CSV = "data/embedding_costs_per_model.csv"
EMB_FALLBACK_MTOKENS = 7.25  # median across all tokenizers


# DATA LOADING

def load_average_scores() -> pd.DataFrame:
    """Canonical MACRO overall score (mean of the 5 per-category means) for every
    complete model. Matches the paper's per-category main-results table."""
    cs = _category_scores(complete_only=True)
    avg = (
        cs.reset_index()[["model", "model_type", "Overall"]]
        .rename(columns={"Overall": "avg_score"})
    )
    return avg


def load_embedding_costs() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "embedding_throughput.csv")
    df["model"] = df["model_id"].str.replace("/", "__", regex=False)
    return df[["model", "cost_usd_per_mtok", "params"]].copy()


def compute_llm_costs(canonical_tasks: set[str] | None = None) -> dict[str, dict]:
    """Scan result JSONs for every priced LLM (registry.LLM_PRICING); return
    token counts + cost.

    Gemini API fields (mapped by the MTEB evaluation framework):
      input_tokens   = prompt_token_count          (includes cached portion)
      output_tokens  = candidates_token_count       (excludes thinking)
      cached_tokens  = cached_content_token_count   (subset of input_tokens)
      total_tokens   = total_token_count            (input + output + thinking)

    Cost formula:
      cost = (input - cached) * input_rate          # non-cached input
           + cached           * cached_rate          # 10% of input_rate
           + (output + thinking) * output_rate       # thinking billed at output rate

    If canonical_tasks is given, only those task files are counted.
    """
    llm_dir = ROOT / "llm_results"
    results = {}

    for model_name, pricing in LLM_PRICING.items():
        model_path = llm_dir / model_name
        if not model_path.exists():
            print(f"  [warn] directory not found: {model_name}")
            continue

        total_input = total_output = total_cached = total_total = total_reason = 0
        n_tasks = 0
        seen_tasks: set[str] = set()   # dedup: some model dirs contain two
                                       # identical copies of every task file
                                       # (no_model_name__available vs _available)

        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from aggregate_scores import canonicalize

        for jf in sorted(model_path.rglob("*.json")):
            if jf.stem == "model_meta" or jf.stem.endswith("_samples"):
                continue
            canon = canonicalize(jf.stem)
            # Skip duplicate copies of the same task (count each task once)
            if canon in seen_tasks:
                continue
            # Filter to canonical tasks only (skip e.g. LLMHagridRetrieval)
            if canonical_tasks is not None and canon not in canonical_tasks:
                continue
            seen_tasks.add(canon)
            try:
                d = json.loads(jf.read_text())
                for _split, sd in d.get("scores", {}).items():
                    if not sd:
                        continue
                    # Multi-entry splits: classification tasks duplicate
                    # usage_stats across entries (same for all); STS/PairCls
                    # multilingual tasks have per-language entries with DIFFERENT
                    # usage_stats. Detect and handle both cases.
                    entries = sd if isinstance(sd, list) else [sd]
                    u0 = entries[0].get("usage_stats", {})
                    if len(entries) > 1:
                        u1 = entries[1].get("usage_stats", {})
                        same = (u0.get("input_tokens") == u1.get("input_tokens"))
                    else:
                        same = True

                    if same:
                        # Duplicated: use entries[0] only
                        total_input  += u0.get("input_tokens", 0)
                        total_output += u0.get("output_tokens", 0)
                        total_cached += u0.get("cached_tokens", 0)
                        total_total  += u0.get("total_tokens", 0)
                        total_reason += u0.get("thinking_tokens", 0) or 0
                    else:
                        # Per-language: sum all entries
                        for entry in entries:
                            u = entry.get("usage_stats", {})
                            total_input  += u.get("input_tokens", 0)
                            total_output += u.get("output_tokens", 0)
                            total_cached += u.get("cached_tokens", 0)
                            total_total  += u.get("total_tokens", 0)
                            total_reason += u.get("thinking_tokens", 0) or 0
                n_tasks += 1
            except Exception as exc:
                print(f"  [warn] {jf}: {exc}")

        # Generation billed at the output rate = every non-input token. Providers
        # report reasoning two ways, both handled here:
        #   Gemini: output EXCLUDES reasoning; total = input+output+reasoning
        #           -> reasoning = total-input-output.
        #   OpenRouter (open models): output ALREADY includes reasoning; total =
        #           input+output; reasoning is a separately reported subset
        #           (usage_stats.thinking_tokens).
        generated  = max(0, total_total - total_input)
        reasoning  = max(total_total - total_input - total_output, total_reason)
        reasoning  = max(0, min(reasoning, generated))
        visible    = generated - reasoning         # standard (non-reasoning) output
        non_cached = total_input - total_cached

        cost = (
            (non_cached / 1e6) * pricing["input"]
            + (total_cached / 1e6) * pricing["cached"]
            + (generated / 1e6) * pricing["output"]     # visible + reasoning, both at output rate
        )

        results[model_name] = {
            "input_tokens":    total_input,
            "output_tokens":   visible,                  # standard output (reasoning split out)
            "cached_tokens":   total_cached,
            "thinking_tokens": reasoning,                # reasoning, however the provider reports it
            "total_tokens":    total_input + generated,
            "n_tasks":         n_tasks,
            "total_cost":      cost,
            "cost_input":      (non_cached / 1e6) * pricing["input"],
            "cost_cached":     (total_cached / 1e6) * pricing["cached"],
            "cost_output":     (generated / 1e6) * pricing["output"],
            "price_input":     pricing["input"],
            "price_output":    pricing["output"],
            "price_cached":    pricing["cached"],
        }
        print(f"  {model_name}:")
        print(f"    tasks={n_tasks}  in={total_input/1e6:.1f}M  out={visible/1e6:.1f}M  "
              f"cached={total_cached/1e6:.1f}M  reasoning={reasoning/1e6:.1f}M")
        print(f"    cost=${cost:.2f}")

    return results


# PARETO FRONTIER

def pareto_frontier_indices(costs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Non-dominated points (minimise cost, maximise score)."""
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if costs[j] <= costs[i] and scores[j] >= scores[i]:
                if costs[j] < costs[i] or scores[j] > scores[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


# DISPLAY NAMES — single source of truth in pub_style (incl. all 10 LLMs)
from pub_style import short_name  # noqa: E402


def short_name_flat(model: str) -> str:
    """Single-line version for CSVs / tables."""
    return short_name(model).replace("\n", " ")


def _auto_label(ax, points, fs, bbox_props, arrow_color, zorder=12, avoid=None, hints=None):
    """Greedy non-overlapping label placement with leader lines (display coords).

    points: list of (x_data, y_data, text, color). Places each label in the
    nearest candidate slot that doesn't collide with markers, already-placed
    labels, or the `avoid` keep-out regions (data-coord (x0,y0,x1,y1) boxes,
    e.g. the cost-gap arrow). Scales to any number of points.

    hints: optional {text: (ux_sign, uy_sign)} preferred direction (e.g. (0, 1)
    to prefer placing the label ABOVE its point). 0 means "no preference" on that axis.
    """
    hints = hints or {}
    fig = ax.figure
    fig.canvas.draw()                      # realize transforms + renderer
    ppd = fig.dpi / 72.0                    # points -> pixels
    disp = [ax.transData.transform((x, y)) for x, y, *_ in points]
    axbox = ax.get_window_extent()

    dirs = [(1, 0.3), (1, -0.3), (-1, 0.3), (-1, -0.3),
            (0.4, 1), (0.4, -1), (-0.4, 1), (-0.4, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1)]
    dists = [16, 28, 42, 60, 84, 116]      # hug the marker first; last reach avoids overlaps

    placed = [(px - 36, py - 36, px + 36, py + 36) for px, py in disp]  # seed: markers
    for (ax0, ay0, ax1, ay1) in (avoid or []):                          # keep-out regions
        (dx0, dy0) = ax.transData.transform((ax0, ay0))
        (dx1, dy1) = ax.transData.transform((ax1, ay1))
        placed.append((min(dx0, dx1), min(dy0, dy1), max(dx0, dx1), max(dy0, dy1)))

    def overlap(a, b):
        ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ox * oy

    for i in sorted(range(len(points)), key=lambda k: -points[k][1]):  # top-down
        x, y, text, color = points[i]
        pref = hints.get(text)
        px, py = disp[i]
        pad = 13 * ppd                     # breathing room between labels
        w = len(text) * fs * 0.60 * ppd + pad
        h = fs * 1.3 * ppd + pad
        best, best_cost = None, None
        for dist in dists:
            for ux, uy in dirs:
                cx, cy = px + ux * dist * ppd, py + uy * dist * ppd
                x0 = cx - w / 2 if ux == 0 else (cx if ux > 0 else cx - w)
                y0 = cy if uy >= 0 else cy - h
                box = (x0, y0, x0 + w, y0 + h)
                cost = 9.0 * sum(overlap(box, p) for p in placed)   # never overlap another label
                cost += 60 * (max(0, axbox.x0 - x0) + max(0, x0 + w - axbox.x1)
                              + max(0, axbox.y0 - y0) + max(0, y0 + h - axbox.y1))
                cost += dist * ppd * 18        # ...but still bias toward the closest slot
                if pref:                       # honor a preferred direction (e.g. above)
                    if pref[1] and uy * pref[1] <= 0:
                        cost += 6000
                    if pref[0] and ux * pref[0] <= 0:
                        cost += 6000
                    if pref[1] and not pref[0]:      # vertical hint -> prefer DIRECTLY above
                        cost += abs(ux) * 3500
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, (box, ux, uy, dist)
            if best_cost == 0:
                break
        box, ux, uy, dist = best
        placed.append(box)
        # Only draw a leader line when the label had to be pushed out; adjacent
        # labels read cleanly without one (avoids the "connect-the-dots" look).
        lead = dict(arrowstyle="-", color=arrow_color, lw=0.8, shrinkB=4) if dist > 60 else None
        ax.annotate(text, xy=(x, y), xytext=(ux * dist, uy * dist),
                    textcoords="offset points", fontsize=fs, fontweight="bold",
                    color=color, ha="center" if ux == 0 else ("left" if ux > 0 else "right"),
                    va="bottom" if uy >= 0 else "top", bbox=bbox_props,
                    arrowprops=lead, zorder=zorder)


# PLOTTING

def plot_pareto(df, pareto, out_dir):
    """Elegant, publication-quality cost-performance Pareto figure.

    Embeddings form the frontier (blue line, bottom-left); every LLM sits far to
    the right (red diamonds). Clean typography, subtle zones, auto-placed labels.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pub_style import save_fig, short_name as sn
    from matplotlib.lines import Line2D

    emb = df[df.type == "Embedding"].copy()
    llm = df[df.type == "LLM"].copy()
    if emb.empty or llm.empty:
        return
    best_emb = emb.loc[emb["avg_score"].idxmax()]
    C_EMB = "#2563EB"
    C_FRONT = "#475569"   # neutral frontier line: it spans both paradigms (embeddings + the one Pareto-optimal LLM)
    C_LLM, C_STAR = "#DC2626", "#F59E0B"

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig, ax = plt.subplots(figsize=(16.5, 9.9), facecolor="white", dpi=150)
    ax.set_facecolor("white")

    # ── Subtle zones + divider ───────────────────────────────────────────────
    ax.axvspan(4e-4, 1.0, color="#F4F8FE", zorder=0)
    ax.axvspan(1.0, 900, color="#FDF5F4", zorder=0)
    ax.axvline(1.0, color="#CBD5E1", lw=1.1, ls=(0, (5, 4)), zorder=1)
    ax.text(6.5e-4, 0.800, "EMBEDDINGS", fontsize=27, fontweight="bold",
            color=C_EMB, alpha=0.42, ha="left", va="top", zorder=1)
    ax.text(1.5, 0.800, "LLMs", fontsize=27, fontweight="bold",
            color=C_LLM, alpha=0.42, ha="left", va="top", zorder=1)

    # ── Points ───────────────────────────────────────────────────────────────
    pf_all = pareto.sort_values("total_cost")          # global frontier (incl. the Pareto-optimal LLM)
    pf_emb = pareto[pareto["type"] == "Embedding"].sort_values("total_cost")
    non_pf = emb[~emb["model"].isin(pareto["model"].values)]
    ax.scatter(non_pf.total_cost, non_pf.avg_score, s=95, color=C_EMB, alpha=0.45,
               edgecolor="white", linewidth=1.0, zorder=3)
    # frontier line spans the embedding frontier and steps up to the one Pareto-optimal LLM
    ax.plot(pf_all.total_cost, pf_all.avg_score, color=C_FRONT, lw=3.4, zorder=4,
            solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(pf_emb.total_cost, pf_emb.avg_score, s=215, color=C_EMB,
               edgecolor="white", linewidth=2.0, zorder=5)
    ax.scatter([best_emb.total_cost], [best_emb.avg_score], s=760, marker="*",
               color=C_STAR, edgecolor="#B45309", linewidth=1.6, zorder=7)
    ax.scatter(llm.total_cost, llm.avg_score, s=250, marker="D", color=C_LLM,
               edgecolor="white", linewidth=1.3, zorder=6)

    # ── Mark the sole Pareto-optimal LLM (Gemini 3.1 Pro) with a gold ring ────
    pro = llm.loc[llm.avg_score.idxmax()]
    ax.scatter([pro.total_cost], [pro.avg_score], s=250, marker="D", color=C_LLM,
               edgecolor="#B45309", linewidth=2.6, zorder=8)   # gold ring = Pareto-optimal

    # ── Axis scale + limits are set BEFORE labelling so the label placer works
    #    in the FINAL coordinate system (otherwise placement is computed on the
    #    autoscaled axes and then breaks when the limits change).
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 900)
    ax.set_ylim(0.615, 0.806)

    # ── Labels: every frontier embedding + every LLM, placed adjacent ────────
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.72)
    short_lbl = {"Qwen3.6-35B-A3B": "Qwen3.6-35B"}   # compact for the crowded cluster
    pts = [(r.total_cost, r.avg_score, sn(r.model), "#1E3A8A") for _, r in pf_emb.iterrows()]
    pts += [(r.total_cost, r.avg_score, short_lbl.get(sn(r.model), sn(r.model)), C_LLM)
            for _, r in llm.iterrows()]
    hints = {"Gemini 3.1 Flash Lite": (0, 1),   # prefer placing above the point
             "DeepSeek-V4-Flash": (0, 1),
             "MiniMax-M2.7": (0, 1),
             "Octen-8B": (0, 1)}               # above the star, clear of the frontier line
    _auto_label(ax, pts, 18, bbox_props, "#9CA3AF", avoid=[], hints=hints)

    # ── Axis cosmetics ───────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}" if x >= 1 else (f"${x:.2f}" if x >= 0.01 else f"${x:.3f}")))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}"))
    ax.set_xlabel("Total benchmark cost (USD, log scale)", fontsize=21, labelpad=10, color="#374151")
    ax.set_ylabel("Mean MTEB(LLM) score", fontsize=21, labelpad=10, color="#374151")
    ax.set_title("Cost vs. Performance:  LLMs vs. Embedding Models",
                 fontsize=27, fontweight="bold", color="#111827", pad=16)
    ax.tick_params(labelsize=17, colors="#4B5563")
    ax.grid(True, which="major", color="#EBEEF3", lw=1.0, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#9CA3AF")

    # ── Legend (horizontal, below the axis — frees the canvas for labels) ────
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_LLM,
               markeredgecolor="white", markersize=13, label="LLM"),
        Line2D([0], [0], color=C_FRONT, lw=3, label="Pareto frontier"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_EMB,
               markersize=12, label="Embedding model"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_STAR,
               markeredgecolor="#B45309", markersize=20, label=f"Best embedding ({sn(best_emb.model)})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=4, fontsize=15, frameon=False, handletextpad=0.6, columnspacing=2.0)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "pareto_cost_performance", out_dir)



# DATA EXPORT

def save_llm_token_usage(llm_cost_data: dict, out_path: Path):
    """Save detailed token + pricing breakdown for LLMs."""
    rows = []
    for model_name, cd in llm_cost_data.items():
        p = LLM_PRICING[model_name]
        rows.append({
            "model": model_name,
            "short_name": short_name_flat(model_name),
            "n_tasks": cd["n_tasks"],
            "input_tokens": cd["input_tokens"],
            "output_tokens": cd["output_tokens"],
            "cached_tokens": cd["cached_tokens"],
            "thinking_tokens": cd["thinking_tokens"],
            "total_tokens": cd["total_tokens"],
            "price_input_per_mtok": p["input"],
            "price_output_per_mtok": p["output"],
            "price_cached_per_mtok": p["cached"],
            "cost_input_usd": cd["cost_input"],
            "cost_cached_usd": cd["cost_cached"],
            "cost_output_usd": cd["cost_output"],
            "total_cost_usd": cd["total_cost"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


# MAIN

def main():
    print("Loading scores ...")
    avg_scores = load_average_scores()

    print("Loading embedding throughput ...")
    emb_costs = load_embedding_costs()

    print("Computing LLM costs ...")
    # Canonical 37 tasks (driven by embedding results in scores.csv)
    scores_raw = pd.read_csv(ROOT / "data" / "scores.csv")
    canonical_tasks = set(scores_raw[scores_raw.model_type == "embedding"]["task"].unique())
    print(f"  Filtering to {len(canonical_tasks)} canonical tasks")
    llm_cost_data = compute_llm_costs(canonical_tasks)

    # Build unified table
    rows = []

    # Embeddings — use per-model token counts from embedding_costs_per_model.csv
    emb_costs_pm = None
    pm_path = ROOT / EMB_COSTS_CSV
    if pm_path.exists():
        emb_costs_pm = pd.read_csv(pm_path)
        emb_costs_pm["model"] = emb_costs_pm["model_id"].str.replace("/", "__", regex=False)
        print(f"  Loaded per-model token counts from {EMB_COSTS_CSV}")

    emb_df = avg_scores[avg_scores.model_type == "embedding"]
    for _, row in emb_df.iterrows():
        match = emb_costs[emb_costs.model == row.model]
        if match.empty:
            continue
        cpm = match.iloc[0]["cost_usd_per_mtok"]

        # Try per-model exact cost first, fall back to median token count
        if emb_costs_pm is not None:
            pm_match = emb_costs_pm[emb_costs_pm.model == row.model]
            if not pm_match.empty and pm_match.iloc[0]["total_cost"] > 0:
                total_cost = pm_match.iloc[0]["total_cost"]
            else:
                total_cost = (EMB_FALLBACK_MTOKENS) * cpm
        else:
            total_cost = (EMB_FALLBACK_MTOKENS) * cpm

        rows.append({
            "model": row.model,
            "type": "Embedding",
            "avg_score": row.avg_score,
            "total_cost": total_cost,
            "cost_per_mtok": cpm,
        })

    # LLMs
    llm_df = avg_scores[avg_scores.model_type == "llm"]
    for model_name in LLM_PRICING:
        match = llm_df[llm_df.model == model_name]
        if match.empty:
            print(f"  [warn] No scores for {model_name}")
            continue
        cd = llm_cost_data.get(model_name)
        if cd is None:
            continue
        rows.append({
            "model": model_name,
            "type": "LLM",
            "avg_score": match.iloc[0]["avg_score"],
            "total_cost": cd["total_cost"],
            "cost_per_mtok": None,
        })

    df = pd.DataFrame(rows)
    n_emb = (df.type == "Embedding").sum()
    n_llm = (df.type == "LLM").sum()
    print(f"\n{len(df)} models  ({n_emb} embedding, {n_llm} LLM)")

    # Pareto frontier
    costs  = df["total_cost"].values
    scores = df["avg_score"].values
    pidx   = pareto_frontier_indices(costs, scores)
    pareto = df.iloc[pidx].sort_values("total_cost")

    print(f"\nPareto-optimal ({len(pidx)}):")
    for _, r in pareto.iterrows():
        print(f"  {short_name_flat(r.model):30s}  score={r.avg_score:.4f}  cost=${r.total_cost:.4f}")

    # Plot
    out_dir = ROOT / "visualizations"
    out_dir.mkdir(exist_ok=True)
    plot_pareto(df, pareto, out_dir)

    # Save cost summary CSV
    tbl = df[["model", "type", "avg_score", "total_cost"]].copy()
    tbl["short_name"] = tbl["model"].apply(short_name_flat)
    tbl = tbl.sort_values("total_cost")
    tbl.to_csv(ROOT / "data" / "cost_summary.csv", index=False)
    print(f"Saved: {ROOT / 'cost_summary.csv'}")

    # Save LLM token usage reference CSV
    token_df = save_llm_token_usage(llm_cost_data, ROOT / "data" / "llm_token_usage.csv")

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Model':<30s} {'Type':<11s} {'Avg Score':>10s} {'Cost (USD)':>12s}")
    print("-" * 90)
    for _, r in tbl.iterrows():
        print(f"{r.short_name:<30s} {r.type:<11s} {r.avg_score:>10.4f} ${r.total_cost:>11.4f}")
    print("=" * 90)

    print("\nLLM Token Usage & Cost Breakdown:")
    print("-" * 100)
    for _, r in token_df.iterrows():
        non_cached = r.input_tokens - r.cached_tokens
        print(f"  {r.short_name}:")
        print(f"    Input (non-cached): ${r.cost_input_usd:>8.2f}  ({non_cached/1e6:.1f}M tok x ${r.price_input_per_mtok}/MTok)")
        print(f"    Cached input:       ${r.cost_cached_usd:>8.2f}  ({r.cached_tokens/1e6:.1f}M tok x ${r.price_cached_per_mtok}/MTok)")
        print(f"    Output + thinking:  ${r.cost_output_usd:>8.2f}  ({r.output_tokens/1e6:.1f}M + {r.thinking_tokens/1e6:.1f}M tok x ${r.price_output_per_mtok}/MTok)")
        print(f"    TOTAL:              ${r.total_cost_usd:>8.2f}")
        print()


if __name__ == "__main__":
    main()
