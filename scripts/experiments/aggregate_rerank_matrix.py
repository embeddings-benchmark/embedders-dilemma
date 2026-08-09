"""Aggregate the BRIGHT + BEIR first-stage × reranker matrix from cached results.

Reads only committed data — pipeline_results/cache (nDCG@10 scores) and
pipeline_results/usage (LLM token + rank-coverage sidecars) — so it runs
anywhere (no GPU, no endpoints). Prints, per benchmark:
  * avg nDCG@10 for each (first-stage, reranker) cell
  * per-LLM-reranker token totals + rank-coverage audit (silent-fallback check)

Usage: python scripts/experiments/aggregate_rerank_matrix.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "pipeline_results" / "cache"
USAGE = ROOT / "pipeline_results" / "usage"

# Display name -> cache first-stage slug.
FIRST_STAGES = {
    "BM25": "bm25s",
    "BGE-large": "BAAI__bge-large-en-v1.5",
    "GTE-MC-v1": "lightonai__GTE-ModernColBERT-v1",
    "Qwen3-E-8B": "Qwen__Qwen3-Embedding-8B",
}
# Display name -> reranker cache-dir suffix (None = pure first stage).
RERANKERS = {
    "pure": None,
    "bge-rerank-v2-m3": "BAAI__bge-reranker-v2-m3",
    "mxbai-rerank-v2": "mixedbread-ai__mxbai-rerank-large-v2",
    "Qwen3-RR-0.6B": "Qwen__Qwen3-Reranker-0.6B",
    "Qwen3-RR-4B": "Qwen__Qwen3-Reranker-4B",
    "Qwen3-RR-8B": "Qwen__Qwen3-Reranker-8B",
    "bge-rerank-gemma": "BAAI__bge-reranker-v2-gemma",
    "llm-qwen3.6-27b": "llm-qwen3.6-27b",
    "llm-qwen3.6-35b-a3b": "llm-qwen3.6-35b-a3b",
    "llm-kimi-k2.6": "llm-kimi-k2.6",
    "llm-minimax-m2.7": "llm-minimax-m2.7",
}
BRIGHT = ["BRIGHTBiology", "BRIGHTEarthScience", "BRIGHTEconomics", "BRIGHTPsychology",
          "BRIGHTRobotics", "BRIGHTStackoverflow", "BRIGHTSustainableLiving"]
BEIR = ["FiQA2018", "NFCorpus", "SciFact", "SCIDOCS", "TRECCOVID"]


def _cell_ndcg(fs_slug: str, rr_suffix: str | None, task: str) -> float | None:
    """nDCG@10 for one (first-stage, reranker, task) cell, or None if not run."""
    cache_dir = fs_slug if rr_suffix is None else f"{fs_slug}__{rr_suffix}"
    files = glob.glob(str(CACHE / cache_dir / "results" / "**" / f"{task}.json"), recursive=True)
    # For a pure first stage, exclude any reranker dirs that share the prefix.
    if rr_suffix is None:
        files = [f for f in files if f"/{fs_slug}/results/" in f]
    if not files:
        return None
    scores = json.loads(Path(files[0]).read_text())["scores"]["test"][0]
    return scores["ndcg_at_10"] * 100


def _avg(fs_slug, rr_suffix, tasks):
    vals = [v for t in tasks if (v := _cell_ndcg(fs_slug, rr_suffix, t)) is not None]
    return sum(vals) / len(vals) if vals else None


def print_matrix(bench_name, tasks, rows):
    print(f"\n=== {bench_name}: avg nDCG@10 ({len(tasks)} tasks) ===")
    hdr = f'{"first-stage":16s}' + "".join(f"{r:>18s}" for r in rows)
    print(hdr)
    print("-" * len(hdr))
    for fs_name, fs_slug in FIRST_STAGES.items():
        line = f"{fs_name:16s}"
        for r in rows:
            v = _avg(fs_slug, RERANKERS[r], tasks)
            line += f"{v:18.1f}" if v is not None else f'{"--":>18s}'
        print(line)


def print_usage_audit():
    print("\n=== LLM listwise reranker: token totals + rank-coverage audit (summed over first-stages) ===")
    print(f'{"model":20s} {"cells":>5s} {"in(M)":>7s} {"out(M)":>7s} {"think(M)":>8s} '
          f'{"full":>6s} {"partial":>7s} {"empty":>6s} {"backfill":>8s} {"audited":>8s}')
    print("-" * 92)
    by_model: dict[str, dict] = {}
    for d in sorted(USAGE.glob("*__llm-*")):
        model = d.name.split("__llm-")[-1]
        m = by_model.setdefault(model, {k: 0 for k in (
            "cells", "audited_cells", "input_tokens", "output_tokens", "thinking_tokens",
            "n_fully_ranked", "n_partial_ranked", "n_empty_ranked", "docs_backfilled")})
        for f in d.glob("*.json"):
            u = json.loads(f.read_text())
            m["cells"] += 1
            # coverage fields only exist on cells run after the audit was added;
            # count how many cells we can actually vouch for.
            if "n_fully_ranked" in u:
                m["audited_cells"] += 1
            for k in ("input_tokens", "output_tokens", "thinking_tokens",
                      "n_fully_ranked", "n_partial_ranked", "n_empty_ranked", "docs_backfilled"):
                m[k] += u.get(k, 0) or 0
    for model, m in sorted(by_model.items()):
        print(f'{model:20s} {m["cells"]:5d} {m["input_tokens"]/1e6:7.1f} '
              f'{m["output_tokens"]/1e6:7.1f} {m["thinking_tokens"]/1e6:8.1f} '
              f'{m["n_fully_ranked"]:6d} {m["n_partial_ranked"]:7d} '
              f'{m["n_empty_ranked"]:6d} {m["docs_backfilled"]:8d} {m["audited_cells"]}/{m["cells"]:>3d}')
    print("\npartial/empty/backfill > 0 ⇒ queries fell back to stage-1 order (truncated/malformed "
          "LLM output). 'audited' = cells carrying coverage fields; cells run before the audit was "
          "added show 0s as 'no data', not verified-clean (those were separately confirmed via "
          "finish_reason!=length checks + output << max_tokens).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="also write the full long-form matrix to this CSV path")
    args = ap.parse_args()

    rows = list(RERANKERS)
    print_matrix("BRIGHT (reasoning)", BRIGHT, rows)
    print_matrix("BEIR (semantic)", BEIR, rows)
    print_usage_audit()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["benchmark", "first_stage", "reranker", "task", "ndcg_at_10"])
            for bench, tasks in [("BRIGHT", BRIGHT), ("BEIR", BEIR)]:
                for fs_name, fs_slug in FIRST_STAGES.items():
                    for rr_name, rr_suffix in RERANKERS.items():
                        for t in tasks:
                            v = _cell_ndcg(fs_slug, rr_suffix, t)
                            if v is not None:
                                w.writerow([bench, fs_name, rr_name, t, round(v, 2)])
        print(f"\nWrote long-form matrix → {args.csv}")


if __name__ == "__main__":
    main()
