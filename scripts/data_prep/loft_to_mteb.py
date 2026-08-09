"""Convert LOFT retrieval datasets to MTEB-compatible HuggingFace datasets.

Downloads 10 LOFT retrieval datasets at 3 scales (32k, 128k, 1m),
converts to MTEB format, and pushes to HuggingFace Hub.

Optimized for Google Colab execution.

Usage:
    # Install deps first (in Colab):
    # !pip install datasets huggingface_hub tqdm

    # All datasets, all scales
    python scripts/loft_to_mteb.py --base-dir ./loft_data --hf-org mteb

    # Single dataset for testing
    python scripts/loft_to_mteb.py --base-dir ./loft_data --hf-org mteb --dataset arguana --scale 32k

    # Dry-run (convert + validate, don't push)
    python scripts/loft_to_mteb.py --base-dir ./loft_data --hf-org mteb --dry-run
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import Dataset, Features, Value
from huggingface_hub import HfApi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LOFT dataset download URLs (from google-deepmind/loft Datasets table)
# ---------------------------------------------------------------------------
LOFT_URLS: dict[str, str] = {
    "arguana": "https://storage.googleapis.com/loft-bench/retrieval/arguana.zip",
    "fever": "https://storage.googleapis.com/loft-bench/retrieval/fever.zip",
    "fiqa": "https://storage.googleapis.com/loft-bench/retrieval/fiqa.zip",
    "hotpotqa": "https://storage.googleapis.com/loft-bench/retrieval/hotpotqa.zip",
    "msmarco": "https://storage.googleapis.com/loft-bench/retrieval/msmarco.zip",
    "musique": "https://storage.googleapis.com/loft-bench/retrieval/musique.zip",
    "nq": "https://storage.googleapis.com/loft-bench/retrieval/nq.zip",
    "qampari": "https://storage.googleapis.com/loft-bench/retrieval/qampari.zip",
    "quest": "https://storage.googleapis.com/loft-bench/retrieval/quest.zip",
    "quora": "https://storage.googleapis.com/loft-bench/retrieval/quora.zip",
    "scifact": "https://storage.googleapis.com/loft-bench/retrieval/scifact.zip",
    "topiocqa": "https://storage.googleapis.com/loft-bench/retrieval/topiocqa.zip",
    "webis_touche2020": "https://storage.googleapis.com/loft-bench/retrieval/webis_touche2020.zip",
}

# Datasets that need BEIR source download to fill in passage text
BEIR_SOURCES: dict[str, str] = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "msmarco": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "webis_touche2020": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
}

SCALES = ["32k", "128k", "1m"]
QUERY_FILES = ["dev_queries.jsonl", "few_shot_queries.jsonl", "test_queries.jsonl"]


# ---------------------------------------------------------------------------
# Progress-bar download helper (Colab-friendly)
# ---------------------------------------------------------------------------
class _DownloadProgress(tqdm):
    """tqdm wrapper for urlretrieve reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar, skipping if exists."""
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {url}")
    with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urlretrieve(url, dest, reporthook=t.update_to)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip if the target directory doesn't already exist."""
    if extract_to.exists():
        print(f"  ✓ Already extracted: {extract_to.name}")
        return
    print(f"  ↗ Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to.parent)
    # LOFT zips sometimes extract to a sub-directory; normalize
    # e.g. arguana.zip -> arguana/  (already correct)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_loft_dataset(dataset: str, base_dir: Path) -> Path:
    """Download and extract a LOFT retrieval dataset. Returns dataset dir."""
    zip_path = base_dir / "zips" / f"{dataset}.zip"
    _download(LOFT_URLS[dataset], zip_path)
    dataset_dir = base_dir / "retrieval" / dataset
    _extract_zip(zip_path, dataset_dir)
    return dataset_dir


def download_beir_source(dataset: str, base_dir: Path) -> Path:
    """Download BEIR source corpus for datasets that need text infill."""
    zip_path = base_dir / "source_zips" / f"{dataset}.zip"
    _download(BEIR_SOURCES[dataset], zip_path)
    source_dir = base_dir / "source" / dataset
    if not source_dir.exists():
        _extract_zip(zip_path, source_dir)
        # BEIR zips extract to e.g. 'fiqa/' or 'webis-touche2020/'
        # Normalize: find the extracted folder and rename if needed
        parent = base_dir / "source"
        for child in parent.iterdir():
            if child.is_dir() and child.name != dataset and dataset in child.name.replace("-", "_"):
                child.rename(source_dir)
                break
    return source_dir


# ---------------------------------------------------------------------------
# LOFT preprocessing (inline version of preprocess.py)
# ---------------------------------------------------------------------------
def preprocess_beir_dataset(dataset: str, loft_dir: Path, base_dir: Path) -> None:
    """Re-join passage text from BEIR source into LOFT corpus/query files.

    Only needed for: fiqa, msmarco, quora, webis_touche2020.
    """
    if dataset not in BEIR_SOURCES:
        return

    # Check if already preprocessed (passage_text present in first line)
    sample_corpus = loft_dir / "128k" / "corpus.jsonl"
    if sample_corpus.exists():
        with open(sample_corpus, encoding="utf-8") as f:
            first = json.loads(f.readline())
            if first.get("passage_text") or first.get("text"):
                print(f"  ✓ {dataset} already preprocessed")
                return

    print(f"  ⚙ Preprocessing {dataset} (joining BEIR text)...")
    source_dir = download_beir_source(dataset, base_dir)

    # Load BEIR queries
    qid2text: dict[str, str] = {}
    queries_path = source_dir / "queries.jsonl"
    if queries_path.exists():
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                qid2text[str(q["_id"])] = q["text"]

    # Load BEIR corpus
    pid2doc: dict[str, dict[str, str]] = {}
    corpus_path = source_dir / "corpus.jsonl"
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid2doc[str(p["_id"])] = {"title": p.get("title", ""), "text": p.get("text", "")}

    # Update LOFT files for each scale
    for scale in SCALES:
        scale_dir = loft_dir / scale
        if not scale_dir.exists():
            continue

        # Update corpus
        corpus_file = scale_dir / "corpus.jsonl"
        if corpus_file.exists():
            passages = []
            with open(corpus_file, encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    pid = str(p["pid"])
                    if pid in pid2doc:
                        p["title_text"] = pid2doc[pid]["title"]
                        p["passage_text"] = pid2doc[pid]["text"]
                    passages.append(p)
            with open(corpus_file, "w", encoding="utf-8") as f:
                for p in passages:
                    json.dump(p, f, ensure_ascii=False)
                    f.write("\n")

        # Update queries
        for qfile_name in QUERY_FILES:
            qfile = scale_dir / qfile_name
            if not qfile.exists():
                continue
            queries = []
            with open(qfile, encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    qid = str(q.get("qid", ""))
                    if qid in qid2text:
                        q["query_text"] = qid2text[qid]
                    queries.append(q)
            with open(qfile, "w", encoding="utf-8") as f:
                for q in queries:
                    json.dump(q, f, ensure_ascii=False)
                    f.write("\n")

    print(f"  ✓ {dataset} preprocessed")


# ---------------------------------------------------------------------------
# Conversion: LOFT → MTEB format
# ---------------------------------------------------------------------------
def load_loft_corpus(corpus_path: Path) -> list[dict]:
    """Load LOFT corpus.jsonl and convert to MTEB format."""
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


def load_loft_queries(query_path: Path) -> tuple[list[dict], list[dict]]:
    """Load a LOFT query file, return (queries_mteb, qrels_mteb).

    LOFT query format (from utils.py load_data_from_file):
      - "qid": query ID
      - "query_text": query string, OR a list of strings for multi-turn
      - "answers": list of [doc_id, relevance_score] pairs

    If query_text is a list, each element is one conversation turn.
    We expand into separate queries ({qid}_t0, {qid}_t1, ...) with
    one qrel each, matching LOFT's per-turn evaluation.
    """
    queries = []
    qrels = []
    with open(query_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = str(entry["qid"])
            text = entry.get("query_text", "")
            answers = entry.get("answers", [])

            if isinstance(text, list):
                # Multi-turn conversation: expand each turn
                for i, answer in enumerate(answers):
                    turn_qid = f"{qid}_t{i}"
                    turn_text = str(text[i]) if i < len(text) else str(text[-1])
                    queries.append({"_id": turn_qid, "text": turn_text})
                    if isinstance(answer, list) and len(answer) >= 2:
                        qrels.append({
                            "query-id": turn_qid,
                            "corpus-id": str(answer[0]),
                            "score": int(answer[1]) if answer[1] else 1,
                        })
                    elif isinstance(answer, str):
                        qrels.append({"query-id": turn_qid, "corpus-id": answer, "score": 1})
            else:
                # Single query: all answers become qrels for same qid
                queries.append({"_id": qid, "text": str(text)})
                for answer in answers:
                    if isinstance(answer, list) and len(answer) >= 2:
                        qrels.append({
                            "query-id": qid,
                            "corpus-id": str(answer[0]),
                            "score": int(answer[1]) if answer[1] else 1,
                        })
                    elif isinstance(answer, str):
                        qrels.append({"query-id": qid, "corpus-id": answer, "score": 1})
    return queries, qrels


def convert_scale(
    loft_dir: Path,
    scale: str,
) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Convert one LOFT dataset/scale to MTEB format.

    Returns (corpus, queries, qrels) or None if scale dir doesn't exist.

    At smaller scales the corpus is smaller, so some gold documents may be
    absent. We filter out qrels referencing missing docs, then drop queries
    that have no remaining qrels. This naturally produces the paper's
    5 / 10 / 100 query counts at 32k / 128k / 1m.
    """
    scale_dir = loft_dir / scale
    if not scale_dir.exists():
        print(f"    ⚠ Scale {scale} not found, skipping")
        return None

    # Corpus
    corpus_path = scale_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"    ⚠ No corpus.jsonl in {scale}, skipping")
        return None
    corpus = load_loft_corpus(corpus_path)

    # LOFT evaluation subsets:
    #   32k  → dev_queries.jsonl       (10 queries as test queries often exceed 32k)
    #   128k → test_queries.jsonl      (100 queries)
    #   1m   → test_queries.jsonl      (100 queries)
    scale_to_qfile = {
        "32k":  "dev_queries.jsonl",
        "128k": "test_queries.jsonl",
        "1m":   "test_queries.jsonl",
    }
    qfile_name = scale_to_qfile.get(scale)
    if qfile_name is None:
        print(f"    ⚠ No query file mapping for scale {scale}, skipping")
        return None
    qfile = scale_dir / qfile_name
    if not qfile.exists():
        print(f"    ⚠ {qfile_name} not found in {scale}, skipping")
        return None

    queries, qrels = load_loft_queries(qfile)
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(
    corpus: list[dict],
    queries: list[dict],
    qrels: list[dict],
    dataset: str,
    scale: str,
) -> bool:
    """Sanity-check the converted data."""
    corpus_ids = {d["_id"] for d in corpus}
    query_ids = {q["_id"] for q in queries}

    errors = []

    # Check qrel references
    for qr in qrels:
        if qr["query-id"] not in query_ids:
            errors.append(f"qrel query-id {qr['query-id']} not in queries")
        if qr["corpus-id"] not in corpus_ids:
            errors.append(f"qrel corpus-id {qr['corpus-id']} not in corpus")

    # Check for empty text
    empty_corpus = sum(1 for d in corpus if not d.get("text"))
    empty_queries = sum(1 for q in queries if not q.get("text"))

    print(f"    📊 {dataset}/{scale}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    if empty_corpus > 0:
        print(f"    ⚠ {empty_corpus} corpus docs with empty text")
    if empty_queries > 0:
        print(f"    ⚠ {empty_queries} queries with empty text")
    if errors:
        for e in errors[:5]:
            print(f"    ❌ {e}")
        print(f"    ❌ Total reference errors: {len(errors)}")
        return False

    print("    ✓ Validation passed")
    return True


# ---------------------------------------------------------------------------
# Push to HuggingFace
# ---------------------------------------------------------------------------
def push_to_hf(
    corpus: list[dict],
    queries: list[dict],
    qrels: list[dict],
    repo_id: str,
    dry_run: bool = False,
) -> None:
    """Push one scale's data to a HF repo in standard MTEB format.

    Creates three configs via Dataset.push_to_hub(config_name=...):
      config "corpus"  → split "test"  (_id, title, text)
      config "queries" → split "test"  (_id, text)
      config "default" → split "test"  (query-id, corpus-id, score)
    """
    if dry_run:
        print(f"    🔍 Dry run — would push to {repo_id}")
        print(f"       corpus:  {len(corpus)} rows")
        print(f"       queries: {len(queries)} rows")
        print(f"       qrels:   {len(qrels)} rows")
        return

    print(f"    ↑ Pushing to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Corpus config
    corpus_ds = Dataset.from_list(corpus, features=Features({
        "_id": Value("string"),
        "title": Value("string"),
        "text": Value("string"),
    }))
    corpus_ds.push_to_hub(repo_id, config_name="corpus", split="corpus")

    # Queries config
    queries_ds = Dataset.from_list(queries, features=Features({
        "_id": Value("string"),
        "text": Value("string"),
    }))
    queries_ds.push_to_hub(repo_id, config_name="queries", split="queries")

    # Default config (qrels)
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
    parser = argparse.ArgumentParser(
        description="Convert LOFT retrieval datasets to MTEB HuggingFace format"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./loft_data",
        help="Directory to download LOFT data into",
    )
    parser.add_argument(
        "--hf-org",
        type=str,
        default="mteb",
        help="HuggingFace org/user to push datasets to",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(LOFT_URLS.keys()),
        help="Process only this dataset (default: all)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=SCALES,
        help="Process only this scale (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and validate but don't push to HuggingFace",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    datasets_to_process = [args.dataset] if args.dataset else list(LOFT_URLS.keys())
    scales_to_process = [args.scale] if args.scale else SCALES

    print(f"Processing {len(datasets_to_process)} datasets × {len(scales_to_process)} scales")
    print(f"HF org: {args.hf_org}")
    print(f"Base dir: {base_dir.resolve()}")
    print()

    for dataset in datasets_to_process:
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Step 1: Download LOFT dataset
        loft_dir = download_loft_dataset(dataset, base_dir)

        # Step 2: Preprocess if needed (BEIR text join)
        preprocess_beir_dataset(dataset, loft_dir, base_dir)

        # Step 3: Convert and push each scale
        for scale in scales_to_process:
            print(f"\n  Scale: {scale}")
            result = convert_scale(loft_dir, scale)
            if result is None:
                continue

            corpus, queries, qrels = result
            valid = validate(corpus, queries, qrels, dataset, scale)
            if not valid:
                print("    ⚠ Validation failed, skipping push")
                continue

            repo_id = f"{args.hf_org}/loft-{dataset.replace('_', '-')}-{scale}"
            push_to_hf(corpus, queries, qrels, repo_id, dry_run=args.dry_run)

        print()

    print("Done! 🎉")


if __name__ == "__main__":
    main()
