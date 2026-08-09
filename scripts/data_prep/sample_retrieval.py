"""Sample and re-upload retrieval datasets for the LLM vs Embeddings benchmark.

Uses MTEB's own task loading (mteb.get_tasks) to handle the varying dataset
structures, then reformats and pushes to HuggingFace as mteb/llm-eval-* with
a standardised LOFT-style format:

    corpus  config, split='corpus'       → _id, title, text
    queries config, split='queries'      → _id, text
    default config, split='test'         → query-id, corpus-id, score

Usage:
    # Dry run — show stats, don't push
    python scripts/sample_retrieval.py --dry-run

    # Push all to HF
    python scripts/sample_retrieval.py --hf-org mteb

    # Push single dataset
    python scripts/sample_retrieval.py --hf-org mteb --dataset hagrid
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

SEED = 42


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTask:
    name: str               # short key → becomes llm-eval-{name}
    hf_path: str            # source HF path
    revision: str           # source revision
    qrels_split: str        # actual split name for qrels (varies: test/dev/train)
    max_queries: int | None # cap; None = keep all
    # Optional overrides for sources with non-standard config naming
    # (e.g. PublicHealthQA uses "english-corpus"/"english-queries"/"english-qrels").
    # If None, defaults to "corpus"/"queries" + auto-detect of "default"/"qrels".
    corpus_config: str = "corpus"
    queries_config: str = "queries"
    qrels_config: str | None = None  # None = auto-detect


TASKS = [
    RetrievalTask(
        name="tempreason-l1",
        hf_path="mteb/TempReasonL1",
        revision="2065ce0bde70750ed98bb91abd96d0d68ab94324",
        qrels_split="test",
        max_queries=100,    # cap from 4000
    ),
    RetrievalTask(
        name="legalbench-corporate-lobbying",
        hf_path="mteb/legalbench_corporate_lobbying",
        revision="20d5533535aea9886054cf7765f7e427b6a87163",
        qrels_split="test",
        max_queries=100,   # keep all 340
    ),
    RetrievalTask(
        name="aila-statutes",
        hf_path="mteb/AILA_statutes",
        revision="ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
        qrels_split="test",
        max_queries=100,   # keep all 50
    ),
    RetrievalTask(
        name="spartqa",
        hf_path="mteb/SpartQA",
        revision="2b3a888557b53698508a4a2b54c45c698a968d90",
        qrels_split="test",
        max_queries=100,    # cap from 3594
    ),
    RetrievalTask(
        name="winogrande",
        hf_path="mteb/WinoGrande",
        revision="e1bf37bb30f7aca43fda9541e72237bc5c220f8e",
        qrels_split="test",
        max_queries=100,    # cap from 1267
    ),
    RetrievalTask(
        name="twitter-hjerne",
        hf_path="mteb/TwitterHjerneRetrieval",
        revision="97ad55673cf9746f8e4b3aaa92b1bb92d82e52db",
        qrels_split="train",
        max_queries=100,   # keep all 78
    ),
    # ---- New tasks added for revised retrieval suite (post-audit) ----
    RetrievalTask(
        name="humaneval",
        hf_path="mteb/HumanEvalRetrieval",
        revision="db44dbc45b006cfb9588cde47e4408ad8022e836",
        qrels_split="test",
        max_queries=100,   # cap from 158
    ),
    RetrievalTask(
        name="fquad",
        hf_path="mteb/FQuADRetrieval",
        revision="cba2a9a9bd2183a387baa99555811d65d2abbf8e",
        qrels_split="test",
        max_queries=100,   # cap from 400
    ),
    RetrievalTask(
        name="legalbench-consumer-contracts",
        hf_path="mteb/legalbench_consumer_contracts_qa",
        revision="f9eafd458f9c61e531d4a2510d8a11dfd2282b21",
        qrels_split="test",
        max_queries=100,   # cap from 396
    ),
    RetrievalTask(
        name="builtbench",
        hf_path="mteb/BuiltBenchRetrieval",
        revision="d11fc55bc772b54346225389becc6cee876c2597",
        qrels_split="test",
        max_queries=100,   # cap from 334
    ),
    RetrievalTask(
        name="finance-bench",
        hf_path="mteb/FinanceBenchRetrieval",
        revision="c5215673492044833ff5e0dfd9b5fa848d5df0f2",
        qrels_split="test",
        max_queries=100,   # cap from 150
    ),
    RetrievalTask(
        name="hc3-finance",
        hf_path="mteb/HC3FinanceRetrieval",
        revision="f7674417d3552fd20e071762bff049137f17b075",
        qrels_split="test",
        max_queries=100,   # cap from 415
    ),
    # PublicHealthQA uses per-language configs; we use the English subset.
    RetrievalTask(
        name="public-health-qa",
        hf_path="mteb/PublicHealthQA",
        revision="234bcb2aec8dac7d55a4767114482ae26e215ea2",
        qrels_split="test",
        max_queries=100,    # cap from 172
        corpus_config="english-corpus",
        queries_config="english-queries",
        qrels_config="english-qrels",
    ),
]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _first_split(dataset_dict) -> object:
    """Return the first split from a DatasetDict (usually the only one)."""
    return dataset_dict[list(dataset_dict.keys())[0]]


def process_task(task: RetrievalTask, hf_org: str, dry_run: bool) -> None:
    from datasets import DatasetDict, load_dataset

    print(f"\n{'='*60}")
    print(f"  {task.name}  ({task.hf_path})")
    print(f"{'='*60}")

    rev = task.revision

    # ── Corpus ────────────────────────────────────────────────────────────────
    # Load full DatasetDict and take first split (handles 'corpus', 'dev', etc.)
    print(f"  Loading corpus (config='{task.corpus_config}') ...")
    corpus_dd = load_dataset(task.hf_path, task.corpus_config, revision=rev, trust_remote_code=False)
    corpus_ds = _first_split(corpus_dd)
    # Normalise 'id' → '_id' if needed
    if "_id" not in corpus_ds.column_names and "id" in corpus_ds.column_names:
        corpus_ds = corpus_ds.rename_column("id", "_id")
    # Ensure standard columns
    if "title" not in corpus_ds.column_names:
        corpus_ds = corpus_ds.add_column("title", [""] * len(corpus_ds))
    corpus_ds = corpus_ds.select_columns(
        [c for c in ["_id", "title", "text"] if c in corpus_ds.column_names]
    )
    print(f"  Corpus: {len(corpus_ds):,} docs  (split='{list(corpus_dd.keys())[0]}')")

    # ── Queries ───────────────────────────────────────────────────────────────
    print(f"  Loading queries (config='{task.queries_config}') ...")
    queries_dd = load_dataset(task.hf_path, task.queries_config, revision=rev, trust_remote_code=False)
    queries_ds = _first_split(queries_dd)
    # Normalise 'id' → '_id' if needed
    if "_id" not in queries_ds.column_names and "id" in queries_ds.column_names:
        queries_ds = queries_ds.rename_column("id", "_id")
    queries_ds = queries_ds.select_columns(
        [c for c in ["_id", "text"] if c in queries_ds.column_names]
    )
    print(f"  Queries: {len(queries_ds):,}  (split='{list(queries_dd.keys())[0]}')")

    # ── Qrels ─────────────────────────────────────────────────────────────────
    # Config name: explicit override or auto-detected ('default' or 'qrels')
    from datasets import get_dataset_config_names
    if task.qrels_config is not None:
        qrels_config = task.qrels_config
    else:
        available_configs = get_dataset_config_names(task.hf_path, revision=rev)
        qrels_config = "default" if "default" in available_configs else "qrels"
    print(f"  Loading qrels (config='{qrels_config}', split='{task.qrels_split}') ...")
    qrels_ds = load_dataset(
        task.hf_path, qrels_config,
        split=task.qrels_split,
        revision=rev,
        trust_remote_code=False,
    )
    # Column names also vary: 'score' or 'relevance_score'
    score_col = "score" if "score" in qrels_ds.column_names else "relevance_score"
    rename_map = {}
    if "query-id" not in qrels_ds.column_names and "query_id" in qrels_ds.column_names:
        rename_map["query_id"] = "query-id"
    if "corpus-id" not in qrels_ds.column_names and "corpus_id" in qrels_ds.column_names:
        rename_map["corpus_id"] = "corpus-id"
    if score_col != "score":
        rename_map[score_col] = "score"
    if rename_map:
        qrels_ds = qrels_ds.rename_columns(rename_map)
    qrels_ds = qrels_ds.select_columns(
        [c for c in ["query-id", "corpus-id", "score"] if c in qrels_ds.column_names]
    )
    print(f"  Qrels: {len(qrels_ds):,} rows")

    # ── Query subsampling ─────────────────────────────────────────────────────
    all_query_ids = set(queries_ds["_id"])
    qrel_query_ids = set(qrels_ds["query-id"])
    valid_query_ids = all_query_ids & qrel_query_ids

    if task.max_queries is not None and len(valid_query_ids) > task.max_queries:
        rng = random.Random(SEED)
        valid_query_ids = set(rng.sample(sorted(valid_query_ids), task.max_queries))
        print(f"  Subsampled: {len(all_query_ids & qrel_query_ids):,} → {len(valid_query_ids):,} queries (seed={SEED})")
    else:
        print(f"  No subsampling (using all {len(valid_query_ids):,} valid queries)")

    queries_ds = queries_ds.filter(lambda row: row["_id"] in valid_query_ids)
    qrels_ds   = qrels_ds.filter(lambda row: row["query-id"] in valid_query_ids)
    print(f"  Final: {len(corpus_ds):,} docs | {len(queries_ds):,} queries | {len(qrels_ds):,} qrel rows")


    corpus_dict  = DatasetDict({"corpus":  corpus_ds})
    queries_dict = DatasetDict({"queries": queries_ds})
    qrels_dict   = DatasetDict({"test":    qrels_ds})   # always 'test' → base class expects this

    repo_id = f"{hf_org}/llm-eval-{task.name}"

    if dry_run:
        print(f"\n  🔍 Dry run — would push to {repo_id}:")
        print(f"     corpus  config → split 'corpus'  ({len(corpus_ds):,} rows)")
        print(f"     queries config → split 'queries' ({len(queries_ds):,} rows)")
        print(f"     default config → split 'test'    ({len(qrels_ds):,} rows)")
        print(f"     Corpus[0]:  {corpus_ds[0]}")
        print(f"     Query[0]:   {queries_ds[0]}")
        print(f"     Qrel[0]:    {qrels_ds[0]}")
    else:
        print(f"\n  ↑ Pushing corpus to {repo_id} ...")
        corpus_dict.push_to_hub(repo_id, config_name="corpus")
        print(f"  ↑ Pushing queries to {repo_id} ...")
        queries_dict.push_to_hub(repo_id, config_name="queries")
        print(f"  ↑ Pushing qrels to {repo_id} ...")
        qrels_dict.push_to_hub(repo_id, config_name="default")
        print(f"  ✓ Done → https://huggingface.co/datasets/{repo_id}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample & upload retrieval datasets for LLM vs Embeddings benchmark"
    )
    parser.add_argument("--hf-org", type=str, default="mteb",
                        help="HuggingFace org to push to")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process only this dataset (by name). Default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats but don't push to HuggingFace")
    args = parser.parse_args()

    tasks_to_run = TASKS
    if args.dataset:
        tasks_to_run = [t for t in TASKS if t.name == args.dataset]
        if not tasks_to_run:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {[t.name for t in TASKS]}")
            return

    print(f"Processing {len(tasks_to_run)} retrieval datasets")
    print(f"HF org: {args.hf_org} | Seed: {SEED} | dry_run={args.dry_run}")

    for task in tasks_to_run:
        process_task(task, args.hf_org, args.dry_run)

    print("\n\nDone! 🎉")
    if not args.dry_run:
        print("\nNext steps:")
        print("  1. Grab the commit SHA for each repo from HuggingFace")
        print("  2. Update the revision='main' strings in llm_judge/tasks/retrieval.py")


if __name__ == "__main__":
    main()
