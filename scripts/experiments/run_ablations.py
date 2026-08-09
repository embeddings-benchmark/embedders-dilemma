#!/usr/bin/env python3
"""Ablation experiments to address reviewer feedback.

This script handles THREE types of experiments:

1. FREE analyses (no API cost):
   --analyze          Run bootstrap significance, TempReason exclusion,
                      cost sensitivity. Uses existing results only.

2. No-CoT ablation (API cost ~$2-5 for Flash Lite-level pricing):
   --run-no-cot       Re-run tasks with thinking disabled.
                      Uses max_tokens=8192 and no reasoning_effort.
                      Estimated cost: ~$2-5 (input same, output much less)

3. Few-shot classification (API cost ~$5-15):
   --run-fewshot N    Re-run classification tasks with N in-context examples.
                      Estimated cost: ~$5-15 (more input tokens from examples)

Environment variables (for API experiments):
    MODEL, BASE_URL, TOKEN, USE_STRICT_JSON, USE_VERTEX_AI

Usage:
    python scripts/run_ablations.py --analyze
    python scripts/run_ablations.py --run-no-cot
    python scripts/run_ablations.py --run-fewshot 5
    python scripts/run_ablations.py --run-no-cot --tasks classification retrieval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# 1. FREE ANALYSES (no API calls)
# ==============================================================================

def run_free_analyses():
    """Bootstrap significance, TempReason exclusion, cost sensitivity."""
    print("=" * 70)
    print("FREE ANALYSES (no API cost)")
    print("=" * 70)

    df = pd.read_csv(ROOT / "data" / "scores.csv")
    cost = pd.read_csv(ROOT / "data" / "cost_summary.csv")
    valid = set(cost["model"])
    df = df[df["model"].isin(valid)]

    # --- A) Paired bootstrap significance ---
    print("\n--- A) Paired Bootstrap Significance Test ---\n")
    pro = "google__gemini-3.1-pro-preview"
    kalm = "tencent__KaLM-Embedding-Gemma3-12B-2511"

    # Get per-task scores for both models
    pro_scores = df[df.model == pro].set_index("task")["score"]
    kalm_scores = df[df.model == kalm].set_index("task")["score"]
    common = sorted(set(pro_scores.index) & set(kalm_scores.index))
    pro_arr = np.array([pro_scores[t] for t in common])
    kalm_arr = np.array([kalm_scores[t] for t in common])
    observed_delta = pro_arr.mean() - kalm_arr.mean()

    n_bootstrap = 10000
    rng = np.random.RandomState(42)
    deltas = np.zeros(n_bootstrap)
    n_tasks = len(common)
    for i in range(n_bootstrap):
        idx = rng.choice(n_tasks, n_tasks, replace=True)
        deltas[i] = pro_arr[idx].mean() - kalm_arr[idx].mean()

    p_value = (deltas <= 0).mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    print("  Models: Pro vs KaLM-12B")
    print(f"  Tasks: {n_tasks}")
    print(f"  Observed delta: {observed_delta:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  p-value (Pro > KaLM): {1 - p_value:.4f}")
    print(f"  p-value (two-sided): {2 * min(p_value, 1 - p_value):.4f}")
    print(f"  Significant at a=0.05: {'Yes' if 2 * min(p_value, 1 - p_value) < 0.05 else 'No'}")

    # Per-category significance
    print("\n  Per-category bootstrap (Pro vs best embedding per category):")
    for cat in ["Classification", "Clustering", "STS", "PairClassification", "Retrieval"]:
        cat_df = df[df.task_category == cat]
        emb_cat = cat_df[cat_df.model_type == "embedding"]
        best_emb_model = emb_cat.groupby("model")["score"].mean().idxmax()

        pro_cat = cat_df[cat_df.model == pro].set_index("task")["score"]
        best_cat = cat_df[cat_df.model == best_emb_model].set_index("task")["score"]
        ct = sorted(set(pro_cat.index) & set(best_cat.index))
        if not ct:
            continue
        pa = np.array([pro_cat[t] for t in ct])
        ba = np.array([best_cat[t] for t in ct])
        obs = pa.mean() - ba.mean()
        boot = np.zeros(5000)
        for i in range(5000):
            idx = rng.choice(len(ct), len(ct), replace=True)
            boot[i] = pa[idx].mean() - ba[idx].mean()
        ci = np.percentile(boot, [2.5, 97.5])
        print(f"    {cat:20s}: D={obs:+.4f}  95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
              f"  best_emb={best_emb_model.split('__')[-1][:20]}")

    # --- B) Aggregates excluding TempReasonL1 ---
    print("\n\n--- B) Aggregates Excluding TempReasonL1 ---\n")
    df_no_temp = df[df.task != "LLMTempReasonL1"]
    for mid, name in [(pro, "Pro"), ("google__gemini-3-flash-preview", "Flash"),
                      (kalm, "KaLM-12B")]:
        full = df[df.model == mid]["score"].mean()
        excl = df_no_temp[df_no_temp.model == mid]["score"].mean()
        print(f"  {name:12s}: full={full:.4f}  excl_TempReason={excl:.4f}  diff={full-excl:+.4f}")

    pro_excl = df_no_temp[df_no_temp.model == pro]["score"].mean()
    kalm_excl = df_no_temp[df_no_temp.model == kalm]["score"].mean()
    print(f"\n  Gap (Pro-KaLM): full={pro_arr.mean()-kalm_arr.mean():.4f}"
          f"  excl_TempReason={pro_excl-kalm_excl:.4f}")

    # --- C) Cost sensitivity analysis ---
    print("\n\n--- C) Cost Sensitivity Analysis ---\n")
    emb_cost = cost[cost.type == "Embedding"]
    best_emb_cost = emb_cost.loc[emb_cost.avg_score.idxmax(), "total_cost"]
    pro_cost = cost[cost.model == pro]["total_cost"].values[0]

    scenarios = [
        ("H100 spot $2.49/hr (our setup)", 1.0),
        ("H100 on-demand $3.99/hr", 3.99 / 2.49),
        ("A100 spot $1.49/hr (slower throughput est.)", 1.49 / 2.49 * 1.5),
        ("L4 spot $0.49/hr (est. 3x slower)", 0.49 / 2.49 * 3.0),
        ("Cloud embedding API $0.10/MTok", None),
    ]

    print(f"  {'Scenario':<45s} {'Emb Cost':>10s} {'Ratio':>8s}")
    print(f"  {'-'*45} {'-'*10} {'-'*8}")
    for name, factor in scenarios:
        if factor is not None:
            adj_cost = best_emb_cost * factor
        else:
            # Cloud API: rough estimate at $0.10/MTok for ~7.5M tokens
            adj_cost = 0.10 * 7.5
        ratio = pro_cost / adj_cost
        print(f"  {name:<45s} ${adj_cost:>9.3f} {ratio:>7.0f}x")

    print(f"\n  LLM cost (Pro): ${pro_cost:.2f}")
    print(f"  Conclusion: cost ratio ranges from ~{int(pro_cost/(best_emb_cost*3.0))}x to"
          f" ~{int(pro_cost/best_emb_cost)}x across hardware scenarios")

    # --- D) Pair classification threshold details ---
    print("\n\n--- D) Pair Classification Threshold Details ---\n")
    print("  MTEB pair classification for embeddings:")
    print("  - Computes cosine similarity for all pairs")
    print("  - Sweeps thresholds to find optimal AP, accuracy, F1")
    print("  - Reports max_ap (AP at optimal threshold)")
    print("  - No per-task dev tuning; threshold is optimized on the eval set itself")
    print("  LLM pair classification:")
    print("  - Direct binary prediction (1/0) per pair, no threshold needed")
    print("  - Both evaluated on identical test pairs")


# ==============================================================================
# 2. NO-COT ABLATION (API cost)
# ==============================================================================

def run_no_cot_ablation(task_categories: list[str] | None = None):
    """Re-run with thinking disabled to test if retrieval advantage holds."""
    print("=" * 70)
    print("NO-COT ABLATION")
    print("=" * 70)
    print("\nThis will run LLM evaluation with reasoning_effort=low")
    print("to minimize (but not eliminate) chain-of-thought tokens.\n")

    # Override env vars BEFORE importing modules that read Settings()
    # NOTE: Gemini 3 models cannot fully disable thinking (only 2.5 can).
    # Setting reasoning_effort="low" minimizes thinking tokens.
    # Flash Lite (no thinking at all) serves as the zero-thinking baseline.
    import os
    os.environ["max_tokens"] = "65536"
    os.environ["reasoning_effort"] = "low"

    sys.path.insert(0, str(ROOT))
    from llm_judge.settings import Settings
    from llm_judge.main import _DummyEncoder

    import mteb

    settings = Settings()
    print(f"Model: {settings.model}")
    print(f"Max concurrency: {settings.max_concurrency}")
    print(f"Max tokens: {settings.max_tokens}")
    print(f"Reasoning effort: {settings.reasoning_effort}")

    # Select tasks based on category
    all_tasks = {
        "classification": [
            "LLMImdbClassification", "LLMBanking77Classification",
            "LLMToxicConversationsClassification",
        ],
        "retrieval": [
            "LLMTempReasonL1", "LLMWinoGrande", "LLMAILAStatutes",
            "LLMSpartQA", "LLMLegalBenchCorporateLobbying",
            "LLMTwitterHjerneRetrieval",
        ],
        "sts": [
            "LLMSTSBenchmark", "LLMSICKR", "LLMSTS12",
        ],
        "clustering": [
            "LLMRedditClusteringP2P", "LLMBigPatentClustering",
        ],
    }

    if task_categories is None:
        task_categories = ["retrieval", "classification"]

    # Import task classes
    from llm_judge.tasks.classification import (
        LLMImdbClassification, LLMBanking77Classification,
        LLMToxicConversationsClassification,
    )
    from llm_judge.tasks.retrieval import (
        LLMTempReasonL1, LLMWinoGrande, LLMAILAStatutes,
        LLMSpartQA, LLMLegalBenchCorporateLobbying,
        LLMTwitterHjerneRetrieval,
    )
    from llm_judge.tasks.sts import LLMSTSBenchmark, LLMSICKR, LLMSTS12
    from llm_judge.tasks.clustering import (
        LLMRedditClusteringP2P, LLMBigPatentClustering,
    )

    task_map = {
        "LLMImdbClassification": LLMImdbClassification,
        "LLMBanking77Classification": LLMBanking77Classification,
        "LLMToxicConversationsClassification": LLMToxicConversationsClassification,
        "LLMTempReasonL1": LLMTempReasonL1,
        "LLMWinoGrande": LLMWinoGrande,
        "LLMAILAStatutes": LLMAILAStatutes,
        "LLMSpartQA": LLMSpartQA,
        "LLMLegalBenchCorporateLobbying": LLMLegalBenchCorporateLobbying,
        "LLMTwitterHjerneRetrieval": LLMTwitterHjerneRetrieval,
        "LLMSTSBenchmark": LLMSTSBenchmark,
        "LLMSICKR": LLMSICKR,
        "LLMSTS12": LLMSTS12,
        "LLMRedditClusteringP2P": LLMRedditClusteringP2P,
        "LLMBigPatentClustering": LLMBigPatentClustering,
    }

    tasks = []
    for cat in task_categories:
        for tname in all_tasks.get(cat, []):
            if tname in task_map:
                tasks.append(task_map[tname]())

    if not tasks:
        print("No tasks selected!")
        return

    print(f"\nRunning {len(tasks)} tasks: {[t.__class__.__name__ for t in tasks]}")

    # Output to a separate directory
    model_slug = settings.model.replace("/", "__")
    out_dir = ROOT / "ablation_results" / "no_cot" / model_slug

    print(f"Output: {out_dir}")
    print("\nEstimated cost: ~$2-5 (no thinking tokens)")
    print("Press Ctrl+C to cancel.\n")

    dummy = _DummyEncoder()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = mteb.cache.ResultCache(out_dir)
    mteb.evaluate(
        model=dummy,
        tasks=tasks,
        cache=cache,
    )

    print(f"\nResults saved to: {out_dir}")
    print("Compare with full-thinking results in llm_results/")


# ==============================================================================
# 3. FEW-SHOT CLASSIFICATION (API cost)
# ==============================================================================

def run_fewshot_classification(n_shots: int = 5):
    """Few-shot classification lives in its own runner; delegate there."""
    print("Few-shot classification is implemented in scripts/experiments/run_fewshot_cls.py.")
    print(f"Run: python scripts/experiments/run_fewshot_cls.py --n-shots {n_shots}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation experiments for reviewer response",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablations.py --analyze              # Free analyses (no API cost)
  python scripts/run_ablations.py --run-no-cot           # No-CoT ablation (~$2-5)
  python scripts/run_ablations.py --run-no-cot --tasks retrieval
  python scripts/run_ablations.py --run-fewshot 5        # Few-shot classification (~$5-15)
        """
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Run free analyses (bootstrap, TempReason exclusion, cost sensitivity)")
    parser.add_argument("--run-no-cot", action="store_true",
                        help="Run no-CoT ablation (needs API, ~$2-5)")
    parser.add_argument("--run-fewshot", type=int, metavar="N",
                        help="Run few-shot classification with N examples (needs API, ~$5-15)")
    parser.add_argument("--tasks", nargs="+",
                        choices=["classification", "retrieval", "sts", "clustering"],
                        help="Task categories for ablations (default: retrieval + classification)")

    args = parser.parse_args()

    if not any([args.analyze, args.run_no_cot, args.run_fewshot]):
        parser.print_help()
        return

    if args.analyze:
        run_free_analyses()

    if args.run_no_cot:
        run_no_cot_ablation(args.tasks)

    if args.run_fewshot:
        run_fewshot_classification(args.run_fewshot)


if __name__ == "__main__":
    main()
