"""Convert LOFT RAG datasets to HuggingFace datasets with gold answers.

This script is similar to loft_to_mteb.py, but specifically processes the 6
LOFT datasets that evaluate Retrieval-Augmented Generation (RAG).

Unlike standard MTEB retrieval (which only needs query-to-document qrels),
RAG evaluation requires the actual text string answers to compute Exact Match (EM)
and F1 scores against the LLM's generated text.

The output HF dataset contains:
  - config 'corpus'  : _id, title, text
  - config 'queries' : _id, text, gold_answers (list of acceptable answer strings)
  - config 'default' : query-id, corpus-id, score (qrels for retrieval evaluation)

Datasets processed: nq, topiocqa, hotpotqa, musique, qampari, quest
Scales: 32k, 128k, 1m

Usage:
    python scripts/loft_to_rag.py --base-dir ./loft_data --hf-org mteb
    python scripts/loft_to_rag.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import HfApi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LOFT RAG datasets
# ---------------------------------------------------------------------------
# Only these 6 datasets support RAG (they have gold text answers)
RAG_DATASETS = [
    "nq",
    "topiocqa",
    "hotpotqa",
    "musique",
    "qampari",
    "quest",
]

LOFT_URLS = {
    d: f"https://storage.googleapis.com/loft-bench/rag/{d}.zip"
    for d in RAG_DATASETS
}

SCALES = ["32k", "128k", "1m"]
QUERY_FILES = ["dev_queries.jsonl", "few_shot_queries.jsonl", "test_queries.jsonl"]


# ---------------------------------------------------------------------------
# Progress-bar download helper
# ---------------------------------------------------------------------------
class _DownloadProgress(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {url}")
    with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urlretrieve(url, dest, reporthook=t.update_to)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists():
        print(f"  ✓ Already extracted: {extract_to.name}")
        return
    print(f"  ↗ Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to.parent)


def download_loft_dataset(dataset: str, base_dir: Path) -> Path:
    zip_path = base_dir / "zips" / f"{dataset}_rag.zip"
    _download(LOFT_URLS[dataset], zip_path)
    dataset_dir = base_dir / "rag" / dataset
    _extract_zip(zip_path, dataset_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_rag_corpus(corpus_path: Path) -> list[dict]:
    """Load LOFT corpus.jsonl and convert to MTEB-style corpus."""
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs.append({
                "_id": str(entry["pid"]),
                "title": entry.get("title_text", ""),
                "text": entry.get("passage_text", entry.get("text", "")),
            })
    return docs


def load_rag_queries(query_path: Path) -> tuple[list[dict], list[dict]]:
    """Load a LOFT query file, extracting both qrels AND gold text answers.

    RAG specific addition: we parse `metadata["answers"]` (or direct `"answers"`
    if formatted differently) to capture the acceptable generation strings.
    """
    queries = []
    qrels = []

    with open(query_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = str(entry["qid"])
            text = entry.get("query_text", "")
            
            # Standard LOFT retrieval answers (doc_id, score) pairs
            retrieval_answers = entry.get("metadata", {}).get("qrels", [])
            
            # Extract gold text answers for RAG generation evaluation
            gold_text_answers = entry.get("answers", [])
            if not isinstance(gold_text_answers, list) and gold_text_answers:
                gold_text_answers = [gold_text_answers]

            # Multi-turn conversation
            if isinstance(text, list):
                for i, turn_r_ans in enumerate(retrieval_answers):
                    turn_qid = f"{qid}_t{i}"
                    turn_text = str(text[i]) if i < len(text) else str(text[-1])
                    
                    turn_gold = []
                    if gold_text_answers and i < len(gold_text_answers):
                        ans_item = gold_text_answers[i]
                        if isinstance(ans_item, list):
                            turn_gold = [str(a) for a in ans_item]
                        else:
                            turn_gold = [str(ans_item)]
                    elif gold_text_answers:
                        # Fallback if un-nested
                        turn_gold = [str(a) for a in gold_text_answers if not isinstance(a, list)]
                        
                    queries.append({
                        "_id": turn_qid,
                        "text": turn_text,
                        "gold_answers": turn_gold
                    })
                    
                    if not isinstance(turn_r_ans, list):
                        turn_r_ans = [turn_r_ans]
                        
                    for individual_ans in turn_r_ans:
                        if isinstance(individual_ans, list) and len(individual_ans) >= 2:
                            qrels.append({
                                "query-id": turn_qid,
                                "corpus-id": str(individual_ans[0]),
                                "score": int(individual_ans[1]) if individual_ans[1] else 1,
                            })
                        elif isinstance(individual_ans, str):
                            qrels.append({"query-id": turn_qid, "corpus-id": individual_ans, "score": 1})
            
            # Single query
            else:
                flat_gold = []
                for a in gold_text_answers:
                    if isinstance(a, list):
                        flat_gold.extend([str(x) for x in a])
                    else:
                        flat_gold.append(str(a))
                        
                queries.append({
                    "_id": qid,
                    "text": str(text),
                    "gold_answers": flat_gold
                })
                for r_ans in retrieval_answers:
                    if isinstance(r_ans, list) and len(r_ans) >= 2:
                        qrels.append({
                            "query-id": qid,
                            "corpus-id": str(r_ans[0]),
                            "score": int(r_ans[1]) if r_ans[1] else 1,
                        })
                    elif isinstance(r_ans, str):
                        qrels.append({"query-id": qid, "corpus-id": r_ans, "score": 1})

    return queries, qrels


def convert_scale(
    loft_dir: Path,
    scale: str,
) -> tuple[list[dict], list[dict], list[dict]] | None:
    scale_dir = loft_dir / scale
    if not scale_dir.exists():
        print(f"    ⚠ Scale {scale} not found")
        return None

    corpus_path = scale_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"    ⚠ No corpus.jsonl in {scale}")
        return None
    corpus = load_rag_corpus(corpus_path)

    scale_to_qfile = {
        "32k":  "dev_queries.jsonl",
        "128k": "test_queries.jsonl",
        "1m":   "test_queries.jsonl",
    }
    qfile_name = scale_to_qfile.get(scale)
    if not qfile_name:
        return None
    
    qfile = scale_dir / qfile_name
    if not qfile.exists():
        print(f"    ⚠ {qfile_name} not found in {scale}")
        return None

    queries, qrels = load_rag_queries(qfile)
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Validation & Push
# ---------------------------------------------------------------------------
def validate(corpus: list[dict], queries: list[dict], qrels: list[dict], dataset: str, scale: str) -> bool:
    corpus_ids = {d["_id"] for d in corpus}
    query_ids = {q["_id"] for q in queries}
    errors = []

    for qr in qrels:
        if qr["query-id"] not in query_ids:
            errors.append(f"qrel query-id {qr['query-id']} missing")
        if qr["corpus-id"] not in corpus_ids:
            errors.append(f"qrel corpus-id {qr['corpus-id']} missing")

    # RAG specific validation (ensure we got at least some text answers)
    queries_with_answers = sum(1 for q in queries if q.get("gold_answers"))
    
    print(f"    📊 {dataset}/{scale}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    print(f"    🎯 Queries with gold text answers: {queries_with_answers}/{len(queries)}")
    
    if queries_with_answers == 0:
        print("    ⚠ WARNING: No gold text answers found. RAG evaluation requires these.")

    if errors:
        for e in errors[:5]:
            print(f"    ❌ {e}")
        return False

    return True


def push_to_hf(corpus: list[dict], queries: list[dict], qrels: list[dict], repo_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"    🔍 Dry run — would push to {repo_id}")
        q_samp = queries[0] if queries else None
        print(f"       query sample: {q_samp}")
        return

    print(f"    ↑ Pushing to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Corpus
    corpus_ds = Dataset.from_list(corpus, features=Features({
        "_id": Value("string"),
        "title": Value("string"),
        "text": Value("string"),
    }))
    corpus_ds.push_to_hub(repo_id, config_name="corpus", split="corpus")

    # Queries (RAG modified: includes gold_answers)
    queries_ds = Dataset.from_list(queries, features=Features({
        "_id": Value("string"),
        "text": Value("string"),
        "gold_answers": Sequence(Value("string")),
    }))
    queries_ds.push_to_hub(repo_id, config_name="queries", split="queries")

    # Qrels
    qrels_ds = Dataset.from_list(qrels, features=Features({
        "query-id": Value("string"),
        "corpus-id": Value("string"),
        "score": Value("int32"),
    }))
    qrels_ds.push_to_hub(repo_id, config_name="default", split="test")

    print(f"    ✓ Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert LOFT RAG datasets to HF format")
    parser.add_argument("--base-dir", type=str, default="./loft_data")
    parser.add_argument("--hf-org", type=str, default="mteb")
    parser.add_argument("--dataset", type=str, choices=RAG_DATASETS)
    parser.add_argument("--scale", type=str, choices=SCALES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    datasets = [args.dataset] if args.dataset else RAG_DATASETS
    scales = [args.scale] if args.scale else SCALES

    print(f"Processing {len(datasets)} RAG datasets × {len(scales)} scales")
    
    for dataset in datasets:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        loft_dir = download_loft_dataset(dataset, base_dir)

        for scale in scales:
            print(f"\n  Scale: {scale}")
            result = convert_scale(loft_dir, scale)
            if not result:
                continue

            corpus, queries, qrels = result
            if validate(corpus, queries, qrels, dataset, scale):
                repo_id = f"{args.hf_org}/rag-{dataset}-{scale}"
                push_to_hf(corpus, queries, qrels, repo_id, args.dry_run)


if __name__ == "__main__":
    main()
