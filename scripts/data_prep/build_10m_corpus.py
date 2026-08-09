"""Build 10M token context corpus for LOFT retrieval datasets.

This script expands the 1M scale `loft-{dataset}-1m` dataset by injecting 
documents from the original base corpus (e.g., BEIR zip from UKP) until the 
overall corpus reaches approximately 10M tokens (budgeted at ~36M characters).

It preserves the identical queries and gold QRELs from the 1M scale, ensuring
a strict subset relationship. Using the official UKP BEIR zip files guarantees
that the corpus PIDs exactly match the original LOFT formats without any 
HuggingFace schema casting issues.

Usage:
    python scripts/build_10m_corpus.py --hf-org mteb --dataset msmarco
    python scripts/build_10m_corpus.py --dry-run
"""

import argparse
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import Dataset, Features, Value, load_dataset
from tqdm import tqdm


# Official UKP BEIR Zip sources matching the LOFT preprocessing script
BEIR_ZIPS = {
    "msmarco": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
    "nq": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
    "fever": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip",
    "hotpotqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
    "topiocqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/topiocqa.zip",
}

SEED = 42
TARGET_TOKENS = 9_000_000  # 0.9 * 10M token budget (leave room for prompt)
CHARS_PER_TOKEN = 4.0      # Rough estimation
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)


class _DownloadProgress(tqdm):
    """tqdm wrapper for urlretrieve reporthook."""
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download_and_extract(url: str, dest_dir: Path, dataset_name: str) -> Path:
    """Download BEIR zip and extract it returning the corpus path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{dataset_name}.zip"
    
    if not zip_path.exists():
        print(f"  ↓ Downloading {url}")
        with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=zip_path.name) as t:
            urlretrieve(url, zip_path, reporthook=t.update_to)
    else:
        print(f"  ✓ Already downloaded: {zip_path.name}")
        
    extracted_dir = dest_dir / dataset_name
    if not extracted_dir.exists():
        print(f"  ↗ Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
            
        # Rename if the root folder in zip is not precisely dataset_name
        for child in dest_dir.iterdir():
            if child.is_dir() and child.name != dataset_name and dataset_name in child.name.replace("-", "_"):
                child.rename(extracted_dir)
                break
    else:
        print(f"  ✓ Already extracted: {extracted_dir.name}")
        
    # The BEIR format always has corpus.jsonl at root
    return extracted_dir / "corpus.jsonl"


def build_10m_scale(dataset_name: str, hf_org: str, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"Building 10M scale for {dataset_name}")
    print(f"{'='*60}")

    if dataset_name not in BEIR_ZIPS:
        print(f"  ⚠ No base corpus mapping for {dataset_name}, skipping.")
        return

    # 1. Load the 1M scale (so we ensure subset relationship)
    loft_1m_repo = f"mteb/loft-{dataset_name.replace('_', '-')}-1m"
    try:
        print(f"  📥 Loading 1M scale from {loft_1m_repo}")
        corpus_1m = load_dataset(loft_1m_repo, "corpus", split="corpus")
        queries_1m = load_dataset(loft_1m_repo, "queries", split="queries")
        qrels_1m = load_dataset(loft_1m_repo, "default", split="test")
    except Exception as e:
        print(f"  ❌ Failed to load 1M scale: {e}")
        return

    print(f"  ✓ Loaded 1M scale: {len(corpus_1m)} docs, {len(queries_1m)} queries, {len(qrels_1m)} qrels")

    # Track existing IDs to avoid inserting duplicates
    existing_ids = set(map(str, corpus_1m["_id"]))
    
    # Count current characters/tokens
    current_chars = 0
    for doc in corpus_1m:
        current_chars += len(doc.get("title", "") or "") + len(doc.get("text", "") or "")
    
    print(f"  ℹ Current 1M corpus holds ~{int(current_chars / CHARS_PER_TOKEN):,} tokens")

    # 2. Download Base Corpus from UKP (BEIR)
    beir_cache_dir = Path("./beir_data")
    corpus_jsonl_path = _download_and_extract(BEIR_ZIPS[dataset_name], beir_cache_dir, dataset_name)
    
    if not corpus_jsonl_path.exists():
        print(f"  ❌ Could not find {corpus_jsonl_path} after extraction.")
        return

    # 3. Stream and extract distractors (documents not already in 1m split)
    new_candidates = []
    print("  Streaming original BEIR corpus to find distractors...")
    
    with open(corpus_jsonl_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("_id"))
            if doc_id not in existing_ids:
                new_candidates.append({
                    "_id": doc_id,
                    "title": doc.get("title", "") or "",
                    "text": doc.get("text", "") or ""
                })
    
    print(f"  ✓ Found {len(new_candidates):,} available distractor documents")
    
    # 4. Shuffle distractors reproducibly
    random.seed(SEED)
    random.shuffle(new_candidates)

    # 5. Append until 10M budget is hit
    appended_docs = []
    print(f"  Injecting distractors until budgeted {TARGET_TOKENS:,} tokens ({TARGET_CHARS:,} chars)...")
    
    for doc in new_candidates:
        if current_chars >= TARGET_CHARS:
            break
        appended_docs.append(doc)
        current_chars += len(doc["title"]) + len(doc["text"])

    print(f"  ✓ Added {len(appended_docs):,} distractors")
    print(f"  ✓ Final token count est: ~{int(current_chars / CHARS_PER_TOKEN):,}")

    # Create the new final list
    final_corpus = list(corpus_1m) + appended_docs
    
    # 6. Push 10M dataset to HF
    repo_10m = f"{hf_org}/loft-{dataset_name.replace('_', '-')}-10m"
    if dry_run:
        print(f"\n  🔍 DRY RUN - Would push to {repo_10m}")
        print(f"     Corpus: {len(final_corpus)} docs")
        print(f"     Queries: {len(queries_1m)} queries")
        print(f"     Qrels: {len(qrels_1m)} qrels")
    else:
        print(f"\n  ↑ Pushing dataset to {repo_10m}...")
        
        # Format explicitly as huggingface Dataset
        corpus_ds = Dataset.from_list(final_corpus, features=Features({
            "_id": Value("string"),
            "title": Value("string"),
            "text": Value("string"),
        }))
        
        corpus_ds.push_to_hub(repo_10m, config_name="corpus", split="corpus")
        queries_1m.push_to_hub(repo_10m, config_name="queries", split="queries")
        qrels_1m.push_to_hub(repo_10m, config_name="default", split="test")
        
        print(f"  ✓ Successfully pushed to https://huggingface.co/datasets/{repo_10m}")


def main():
    parser = argparse.ArgumentParser(description="Build 10M corpus scales for LOFT retrieval using original BEIR zips")
    parser.add_argument("--hf-org", type=str, default="mteb", help="HF org to push to")
    parser.add_argument("--dataset", type=str, choices=list(BEIR_ZIPS.keys()), help="Specify one dataset")
    parser.add_argument("--dry-run", action="store_true", help="Don't push, just print stats")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(BEIR_ZIPS.keys())

    for ds in datasets:
        build_10m_scale(ds, args.hf_org, args.dry_run)
        
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
