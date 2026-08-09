#!/usr/bin/env python3
"""Throughput & Parallelizability: LLMs vs Embedding Models.

Produces a two-panel figure comparing LLM API throughput at different
concurrency levels with embedding model throughput on H100.

  Panel (a): LLM samples/min vs API concurrency  (line chart)
  Panel (b): LLM peak vs embedding models          (bar chart, log scale)

Inputs:
  - throughput_results/<model_slug>/concurrency_*/ LLM IMDB results per model
  - throughput_results/mteb_results/results/       Embedding IMDB results (n_experiments=1)

Outputs:
  - visualizations/throughput_comparison.{png,pdf}
  - throughput_results/llm_concurrency_sweep.csv

Running the concurrency sweep:
  The sweep evaluates IMDB Classification at each concurrency level.
  Results are stored per-model, so you can run multiple LLMs:

    MODEL="google/gemini-3-flash-preview" \\
    BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \\
    TOKEN="your-key" USE_RLM=False USE_STRICT_JSON=false \\
    python scripts/throughput_experiment.py --run

    MODEL="google/gemini-3.1-flash-lite-preview" \\
    ... python scripts/throughput_experiment.py --run

  Without --run, the script reads all existing results and plots.

Usage:
    python scripts/throughput_experiment.py            # plot only
    python scripts/throughput_experiment.py --run      # sweep + plot
    python scripts/throughput_experiment.py --run --concurrency 1 5 10
    python scripts/throughput_experiment.py --run --force
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "throughput_results"
LLM_CSV = RESULTS_DIR / "llm_concurrency_sweep.csv"

CONCURRENCY_LEVELS = [1, 2, 5, 10, 25, 50]

# Representative embedding models (spread of sizes for the bar chart)
EMBEDDING_MODELS = [
    ("tencent/KaLM-Embedding-Gemma3-12B-2511", "KaLM-12B"),
    ("Qwen/Qwen3-Embedding-8B",                 "Qwen3-Emb (8B)"),
    ("jinaai/jina-embeddings-v5-text-small",     "Jina-v5-S (596M)"),
    ("Qwen/Qwen3-Embedding-0.6B",               "Qwen3-Emb (0.6B)"),
    ("jinaai/jina-embeddings-v5-text-nano",      "Jina-v5-Nano (212M)"),
    ("intfloat/multilingual-e5-small",           "mE5-Small (118M)"),
]

# Display names for LLM models. NOTE: this figure needs concurrency-sweep data in
# throughput_results/ (currently absent) — the cross-family + open-weight entries
# below are staged so labels resolve once those sweeps are run on the H100.
_SHORT = {
    "google/gemini-3-flash-preview":        "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.1-pro-preview":        "Gemini 3.1 Pro",
    "deepseek/deepseek-r1":                 "DeepSeek-R1",
    "deepseek/deepseek-v4-flash":           "DeepSeek-V4-Flash",
    "z-ai/glm-4.7":                         "GLM-4.7",
    "moonshotai/kimi-k2.6":                 "Kimi-K2.6",
    "minimax/minimax-m2.7":                 "MiniMax-M2.7",
    "qwen/qwen3.6-27b":                     "Qwen3.6-27B",
    "qwen/qwen3.6-35b-a3b":                 "Qwen3.6-35B-A3B",
}


def short_name(model: str) -> str:
    return _SHORT.get(model, model.split("/")[-1])


# DATA LOADING

def _read_imdb_result(out_dir: Path) -> dict | None:
    """Extract timing, accuracy, and sample count from an IMDB result JSON."""
    for jf in out_dir.rglob("ImdbClassification.json"):
        data = json.loads(jf.read_text())
        eval_time = data.get("evaluation_time")
        if not eval_time:
            continue
        accuracy = None
        n_samples = None
        try:
            block = data["scores"]["test"][0]
            accuracy = block["accuracy"]
            # scores_per_experiment[0] has per-sample predictions;
            # usage_stats.input_tokens / ~150 tokens per review ≈ n_samples
            # But the most reliable count: MTEB logs n_samples in some versions
            n_samples = block.get("n_samples")
        except (KeyError, IndexError):
            pass
        return {
            "evaluation_time": round(eval_time, 2),
            "accuracy": accuracy,
            "n_samples": n_samples,
        }
    return None


def load_llm_sweep() -> list[pd.DataFrame]:
    """Load concurrency sweep results for all LLM models.

    Scans throughput_results/<model_slug>/concurrency_*/ for per-model results.
    Also checks for legacy flat layout (concurrency_*/ at top level).

    Returns a list of DataFrames, one per model, each with attrs["model"] set.
    """
    N_SAMPLES = 1000  # mteb/llm-eval-imdb test split

    if not RESULTS_DIR.exists():
        return []

    # Collect (model_name, concurrency_dir) pairs
    model_conc_dirs: dict[str, list[tuple[int, Path]]] = {}

    for entry in sorted(RESULTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        # Skip embedding results
        if entry.name in ("mteb_results",):
            continue

        # New layout: <model_slug>/concurrency_N/
        if not entry.name.startswith("concurrency_"):
            model_id = entry.name.replace("__", "/", 1)
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and sub.name.startswith("concurrency_"):
                    try:
                        conc = int(sub.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    model_conc_dirs.setdefault(model_id, []).append((conc, sub))

        # Legacy layout: concurrency_N/ at top level
        elif entry.name.startswith("concurrency_"):
            try:
                conc = int(entry.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            # Read model name from sweep_meta.json
            meta_path = RESULTS_DIR / "sweep_meta.json"
            model_id = "LLM"
            if meta_path.exists():
                model_id = json.loads(meta_path.read_text()).get("model", "LLM")
            model_conc_dirs.setdefault(model_id, []).append((conc, entry))

    # Build one DataFrame per model
    dfs = []
    for model_id, conc_dirs in model_conc_dirs.items():
        rows = []
        for conc, d in sorted(conc_dirs):
            result = _read_imdb_result(d)
            if result is None:
                continue
            eval_time = result["evaluation_time"]
            spm = N_SAMPLES / eval_time * 60
            rows.append({
                "concurrency": conc,
                "n_samples": N_SAMPLES,
                "evaluation_time_s": eval_time,
                "samples_per_min": round(spm, 2),
                "accuracy": result["accuracy"],
            })
        if rows:
            df = pd.DataFrame(rows).sort_values("concurrency")
            df.attrs["model"] = model_id
            df.attrs["n_samples"] = N_SAMPLES
            dfs.append(df)

    return dfs


def load_embedding_throughput() -> pd.DataFrame:
    """Load measured embedding throughput from MTEB IMDB results (n_experiments=1).

    Results are read from throughput_results/mteb_results/results/<model>/.
    Both LLM and embedding throughput are measured end-to-end through MTEB
    with n_experiments=1 for a fair comparison.
    """
    N_SAMPLES = 1000  # mteb/llm-eval-imdb test split

    emb_dir = RESULTS_DIR / "mteb_results" / "results"
    if not emb_dir.exists():
        print(f"  Warning: no embedding results at {emb_dir}")
        return pd.DataFrame(columns=["model_id", "short_name", "evaluation_time_s",
                                      "samples_per_min"])

    # Build lookup from model_id to display name
    display_names = dict(EMBEDDING_MODELS)

    rows = []
    for model_dir in sorted(emb_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Directory name is model_id with / replaced by __
        model_id = model_dir.name.replace("__", "/", 1)
        display_name = display_names.get(model_id)
        if display_name is None:
            continue  # not one of our representative models

        # Find the result JSON (nested under revision hash)
        for jf in model_dir.rglob("LLMImdbClassification.json"):
            data = json.loads(jf.read_text())
            eval_time = data.get("evaluation_time")
            if not eval_time:
                continue
            accuracy = None
            try:
                accuracy = data["scores"]["test"][0]["accuracy"]
            except (KeyError, IndexError):
                pass
            spm = N_SAMPLES / eval_time * 60
            rows.append({
                "model_id": model_id,
                "short_name": display_name,
                "evaluation_time_s": round(eval_time, 2),
                "samples_per_min": round(spm, 1),
                "accuracy": accuracy,
            })
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("samples_per_min")


# CONCURRENCY SWEEP  (optional — only with --run)

def run_sweep(concurrencies: list[int], force: bool = False):
    """Run IMDB Classification at each concurrency level.

    Monkey-patches settings.max_concurrency between runs and uses a fresh
    MTEB cache directory per level so results are never skipped.
    """
    import mteb
    from llm_judge.main import _DummyEncoder
    from llm_judge.tasks.classification import LLMImdbClassification
    from llm_judge.llm_client import settings as llm_settings
    import llm_judge.evaluators.llm_classification_evaluator as eval_mod

    RESULTS_DIR.mkdir(exist_ok=True)

    # Per-model output directory
    model_slug = llm_settings.model.replace("/", "__")
    model_dir = RESULTS_DIR / model_slug
    model_dir.mkdir(exist_ok=True)

    print(f"\nModel:    {llm_settings.model}")
    print(f"Base URL: {llm_settings.base_url}")
    print(f"Output:   {model_dir}")
    print("Task:     IMDB Classification")
    print(f"Sweep:    {sorted(concurrencies)}\n")

    for conc in sorted(concurrencies):
        out_dir = model_dir / f"concurrency_{conc}"

        # Skip if cached
        if not force and _read_imdb_result(out_dir) is not None:
            r = _read_imdb_result(out_dir)
            spm = 1000 / r["evaluation_time"] * 60
            print(f"  [cached] concurrency={conc:>3}  "
                  f"{r['evaluation_time']:>7.1f}s  "
                  f"{spm:>7.1f} samp/min  acc={r['accuracy']}")
            continue

        # Clean old results
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # Monkey-patch concurrency
        llm_settings.max_concurrency = conc
        eval_mod.LLMClassificationEvaluator.GLOBAL_USAGE = None

        print(f"\n  Running: concurrency={conc} ...")

        task = LLMImdbClassification()
        model = _DummyEncoder()
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = mteb.cache.ResultCache(out_dir)

        start = time.perf_counter()
        mteb.evaluate(model=model, tasks=[task], cache=cache)
        elapsed = time.perf_counter() - start

        r = _read_imdb_result(out_dir)
        eval_time = r["evaluation_time"] if r else elapsed
        spm = 1000 / eval_time * 60
        print(f"  Done: {eval_time:.1f}s  ({spm:.1f} samples/min)  "
              f"acc={r['accuracy'] if r else '?'}")


# PLOTTING

def plot_throughput(llm_dfs: list[pd.DataFrame], emb_df: pd.DataFrame, out_dir: Path):
    """Publication-quality throughput comparison — clean serif, two-panel."""
    import matplotlib.patheffects as pe
    import matplotlib.patches as mpatches

    # ── Style (pub_style.py colors, serif fonts) ─────────────────────────────
    C_LLM   = "#C0392B"   # brick red
    C_EMB   = "#2D6A9F"   # steel blue
    LLM_COLORS = [C_LLM, "#E74C3C"]   # two distinct reds: solid / dashed
    LINESTYLES = ["-",    "--"]
    MARKERS    = ["o",    "D"]
    STROKE = [pe.withStroke(linewidth=3.5, foreground="white")]

    # Font sizes calibrated for 42" canvas
    FS       = 50
    FS_TITLE = 65
    FS_LABEL = 58
    FS_TICK  = 48
    FS_ANNOT = 50

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Georgia"],
        "axes.linewidth":    1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })

    # ── Data prep ────────────────────────────────────────────────────────────
    llm_bars = []
    for df in llm_dfs:
        name = short_name(df.attrs.get("model") or "LLM")
        llm_bars.append({"name": name, "value": df["samples_per_min"].max(),
                         "color": C_LLM, "type": "llm"})
    llm_best = max(b["value"] for b in llm_bars) if llm_bars else 0

    all_bars = llm_bars + [
        {"name": r["short_name"], "value": r["samples_per_min"], "color": C_EMB, "type": "emb"}
        for _, r in emb_df.iterrows()
    ]
    all_bars.sort(key=lambda b: b["value"])

    names  = [b["name"]         for b in all_bars]
    values = [float(b["value"]) for b in all_bars]
    colors = [b["color"]        for b in all_bars]
    types  = [b["type"]         for b in all_bars]

    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(42, 16), 
        gridspec_kw={"width_ratios": [1, 1.3]}, 
    )
    # FURTHER DECREASED top to 0.78 to give suptitle massive clearance
    fig.subplots_adjust(wspace=0.55, top=0.78, bottom=0.15, left=0.06, right=0.96)

    # ════════════════════════════════════════════════════════════════════════
    # Panel A — LLM Concurrency Scaling
    # ════════════════════════════════════════════════════════════════════════
    all_spm, all_conc = [], []
    max_conc = max(CONCURRENCY_LEVELS) if CONCURRENCY_LEVELS else 50

    for i, df in enumerate(llm_dfs):
        label = short_name(df.attrs.get("model") or "LLM")
        c  = LLM_COLORS[i % len(LLM_COLORS)]
        m  = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]

        ax1.plot(df["concurrency"], df["samples_per_min"],
                 color=c, ls=ls, lw=3.5, marker=m, ms=16,
                 mec="white", mew=2.0, zorder=5, label=label)

        # Peak annotation
        max_idx  = df["samples_per_min"].idxmax()
        peak_row = df.loc[max_idx]
        if peak_row["concurrency"] >= max_conc:
            ax1.annotate(
                f'{peak_row["samples_per_min"]:.0f}',
                (peak_row["concurrency"], peak_row["samples_per_min"]),
                textcoords="offset points", xytext=(-10, 14),
                ha="right", fontsize=FS_ANNOT, color=c, fontweight="bold",
                path_effects=STROKE, zorder=6)
        else:
            ax1.annotate(
                f'{peak_row["samples_per_min"]:.0f}',
                (peak_row["concurrency"], peak_row["samples_per_min"]),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=FS_ANNOT, color=c, fontweight="bold",
                path_effects=STROKE, zorder=6)

        all_spm.extend(df["samples_per_min"].tolist())
        all_conc.extend(df["concurrency"].tolist())

    peak = max(all_spm) if all_spm else 1

    ax1.set_xlabel("API Concurrency", fontsize=FS_LABEL, labelpad=14)
    ax1.set_ylabel("Throughput  (samples / min)", fontsize=FS_LABEL, labelpad=14)
    # ADDED explicit padding to title to pull it up slightly
    ax1.set_title("(a)  LLM Concurrency Scaling", fontsize=FS_TITLE, pad=25)
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sorted(set(all_conc)))
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.tick_params(labelsize=FS_TICK, length=7, width=1.2)
    
    # Legend in upper left
    ax1.legend(loc="upper left", fontsize=FS, framealpha=0.92,
               edgecolor="#cccccc", fancybox=False, borderpad=0.8,
               labelspacing=0.5, handlelength=2.4)
    ax1.grid(True, which="major", alpha=0.15, lw=1.0, color="#94A3B8")
    ax1.set_xlim(0.8, max(all_conc) * 1.3 if all_conc else 60)
    
    ax1.set_ylim(min(all_spm) * 0.4 if all_spm else 0.1, peak * 15.0) 
    
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color("#333333")
        ax1.spines[spine].set_linewidth(1.2)

    # ════════════════════════════════════════════════════════════════════════
    # Panel B — Peak Throughput Lollipop
    # ════════════════════════════════════════════════════════════════════════
    y_pos = list(range(len(names)))
    x_min_display = min(values) * 0.3 if values else 1
    x_right = max(values) * 15 if values else 100

    ax2.set_ylim(-0.8, len(names) + 0.5)

    # Baseline vertical anchor
    ax2.axvline(x_min_display, color="#333333", linewidth=2.5, zorder=2)

    # Track lines
    ax2.hlines(y=y_pos, xmin=x_min_display, xmax=x_right,
               color="#E5E7EB", linewidth=1.5, linestyle=":", zorder=1)

    # Drop shadows for the markers
    ax2.scatter(values, [y - 0.08 for y in y_pos], color="black", s=1800, alpha=0.1, zorder=2)

    # Stems
    ax2.hlines(y=y_pos, xmin=x_min_display, xmax=values,
               color=colors, alpha=0.85, linewidth=12, zorder=3)

    # Dots
    ax2.scatter(values, y_pos, color=colors, s=1800,
                edgecolor="white", linewidth=4, zorder=4)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=FS, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=FS_TICK, length=7, width=1.2)
    ax2.tick_params(axis="y", length=0, pad=14)
    ax2.set_xlabel("Throughput  (samples / min, log scale)", fontsize=FS_LABEL, labelpad=14)
    # ADDED explicit padding to title
    ax2.set_title("(b)  Peak Throughput  (speedup vs. best LLM)",
                  fontsize=FS_TITLE, pad=25)
    ax2.set_xscale("log")
    ax2.set_xlim(left=x_min_display, right=x_right)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#333333")
        ax2.spines[spine].set_linewidth(1.2)

    # Value labels and inline Speedup Badges
    for y, val, t, c in zip(y_pos, values, types, colors):
        # Absolute value label
        ax2.text(val * 1.4, y, f"{val:,.0f}",
                 va="center", ha="left", fontsize=FS_ANNOT, fontweight="bold",
                 color=c, path_effects=STROKE, zorder=5)

        # Inline speedup multiplier
        if t == "emb" and llm_best > 0:
            speedup = val / llm_best
            badge_text = f" {speedup:.0f}× faster "
            ax2.text(val * 0.55, y + 0.35, badge_text,
                     va="center", ha="right", fontsize=FS_ANNOT-12, fontweight="bold",
                     color="white", bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none", alpha=0.95),
                     zorder=6)

    # ── Legends & Titles ────────────────────────────────────────────────────
    
    # MUCH SMALLER legend for panel B, with tighter padding
    leg_handles = [
        mpatches.Patch(color=C_LLM, label="LLM (Gemini API)"),
        mpatches.Patch(color=C_EMB, label="Embedding model (H100 GPU)"),
    ]
    ax2.legend(handles=leg_handles, fontsize=FS - 14,
               framealpha=0.92, edgecolor="#cccccc", fancybox=False,
               borderpad=0.6, labelspacing=0.4, handlelength=1.5,
               loc="lower right")

    # RAISED suptitle y-coordinate to fully clear the sub-panel titles
    fig.suptitle(
        "Inference Throughput:  LLMs  vs.  Embedding Models",
        fontsize=FS_TITLE + 10, fontweight="bold", y=0.98, color="#111827",
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        path = out_dir / f"throughput_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir / 'throughput_comparison.png'}")
    print(f"Saved: {out_dir / 'throughput_comparison.pdf'}")


# MAIN

def main():
    # ---------------------------------------------------------------------------
    # DEPRECATED.  This script produced the OLD two-panel throughput figure whose
    # panel (a) plotted LLM *API throughput vs concurrency* — the API-vs-throughput
    # conflation reviewers 6ux1/rYBc flagged.  It has been superseded by
    # plot_throughput_h100.py, which serves both paradigms on the SAME H100.
    # Its inputs (throughput_results/) are also no longer in the repo.  It is
    # neutralised here so it can never overwrite the current
    # visualizations/throughput_comparison.{png,pdf}.  Remove the guard only if
    # you deliberately want to resurrect the old API-concurrency analysis.
    # ---------------------------------------------------------------------------
    print("[DEPRECATED] plot_throughput.py is the old API-concurrency throughput figure; "
          "use plot_throughput_h100.py (same-hardware). Not regenerating throughput_comparison.")
    return

    parser = argparse.ArgumentParser(
        description="Throughput comparison: LLM concurrency vs embedding throughput",
    )
    parser.add_argument("--run", action="store_true",
                        help="Run the concurrency sweep before plotting")
    parser.add_argument("--concurrency", type=int, nargs="+",
                        default=CONCURRENCY_LEVELS,
                        help="Concurrency levels to sweep (default: 1 2 5 10 25 50)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist")
    args = parser.parse_args()

    # Optional: run the sweep
    if args.run:
        run_sweep(args.concurrency, args.force)

    # Load data
    print("Loading LLM sweep results ...")
    llm_dfs = load_llm_sweep()
    if not llm_dfs:
        print(f"  No results found in {RESULTS_DIR}/")
        print("  Run with --run first.")
        sys.exit(1)

    print("Loading embedding throughput ...")
    emb_df = load_embedding_throughput()

    # Save CSV (all models combined)
    all_rows = []
    for df in llm_dfs:
        model = df.attrs.get("model", "unknown")
        for _, r in df.iterrows():
            row = r.to_dict()
            row["model"] = model
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(LLM_CSV, index=False)
    print(f"\nSaved: {LLM_CSV}")

    # Summary
    for df in llm_dfs:
        model = short_name(df.attrs.get("model", "unknown"))
        print(f"\n  {model}:")
        print(f"  {'Concurrency':>12} {'Time (s)':>10} {'Samp/min':>10} {'Accuracy':>10}")
        print(f"  {'-'*46}")
        for _, r in df.iterrows():
            acc = f"{r['accuracy']:.4f}" if pd.notna(r.get("accuracy")) else "n/a"
            print(f"  {int(r['concurrency']):>12} {r['evaluation_time_s']:>10.1f} "
                  f"{r['samples_per_min']:>10.1f} {acc:>10}")
        print(f"  Peak: {df['samples_per_min'].max():.0f} samples/min")

    llm_best_peak = max(df["samples_per_min"].max() for df in llm_dfs)
    print(f"\nBest LLM peak: {llm_best_peak:.0f} samples/min")

    if not emb_df.empty:
        print("\nEmbedding throughput (H100, MTEB end-to-end, n_experiments=1):")
        print(f"  {'Model':<25s} {'Time (s)':>10} {'Samp/min':>12} {'Speedup':>10} {'Acc':>8}")
        print(f"  {'-'*70}")
        for _, r in emb_df.iterrows():
            speedup = r["samples_per_min"] / llm_best_peak if llm_best_peak > 0 else 0
            acc = f"{r['accuracy']:.3f}" if pd.notna(r.get("accuracy")) else "n/a"
            eval_t = f"{r['evaluation_time_s']:.1f}" if "evaluation_time_s" in r.index else "n/a"
            print(f"  {r['short_name']:<25s} {eval_t:>8}s {r['samples_per_min']:>10,.0f}  "
                  f"{speedup:>9,.0f}x  {acc:>8}")

    # Plot
    plot_throughput(llm_dfs, emb_df, ROOT / "visualizations")


if __name__ == "__main__":
    main()