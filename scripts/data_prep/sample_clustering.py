"""Sample and re-upload clustering datasets for the LLM vs Embeddings benchmark.

All datasets use the MTEB nested format:
    sentences: list[str],  labels: list[str]
Each row is one independent clustering problem.

Strategy:
    1. Keep at most `max_rows` rows per dataset (random pick, seed-controlled).
    2. Drop rows with ≤ 1 unique label (not meaningful clustering).
    3. If a row has more than `max_labels` unique labels, keep only the top-N
       most frequent labels and their documents.
    4. Stratified-sample each row to ≤ `sample_size` docs, preserving every
       remaining label (force-include 1 doc per label, fill rest randomly).
    5. Upload in the SAME nested format.

Usage:
    # Dry run
    python scripts/sample_clustering.py --dry-run

    # Push to HF
    python scripts/sample_clustering.py --hf-org mteb
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass

from datasets import Dataset, load_dataset


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
@dataclass
class ClusteringTask:
    name: str               # short identifier
    hf_path: str            # HuggingFace dataset path
    hf_name: str | None     # HF config/subset name
    split: str              # split to process
    sample_size: int        # Max documents per clustering row
    max_rows: int           # Max rows (clustering problems) to keep
    max_labels: int         # Max unique labels per row


# twenty_newsgroups labels are ints (0-19) — these are the category names
TWENTY_NG_LABELS = {
    0: "alt.atheism", 1: "comp.graphics", 2: "comp.os.ms-windows.misc",
    3: "comp.sys.ibm.pc.hardware", 4: "comp.sys.mac.hardware", 5: "comp.windows.x",
    6: "misc.forsale", 7: "rec.autos", 8: "rec.motorcycles",
    9: "rec.sport.baseball", 10: "rec.sport.hockey", 11: "sci.crypt",
    12: "sci.electronics", 13: "sci.med", 14: "sci.space",
    15: "soc.religion.christian", 16: "talk.politics.guns",
    17: "talk.politics.mideast", 18: "talk.politics.misc", 19: "talk.religion.misc",
}

# big_patent labels are single letters — these are the CPC section names
BIG_PATENT_LABELS = {
    "a": "Human Necessities",
    "b": "Performing Operations; Transporting",
    "c": "Chemistry; Metallurgy",
    "d": "Textiles; Paper",
    "e": "Fixed Constructions",
    "f": "Mechanical Engineering; Lighting; Heating; Weapons",
    "g": "Physics",
    "h": "Electricity",
    "y": "General Tagging of Cross-Sectional Technologies",
}


TASKS = [
    ClusteringTask(
        name="reddit_clustering_p2p",
        hf_path="mteb/reddit-clustering-p2p",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="big_patent_clustering",
        hf_path="jinaai/big-patent-clustering",
        hf_name=None,
        split="test",
        sample_size=130,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="twenty_newsgroups_v2",
        hf_path="mteb/twentynewsgroups-clustering",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="stackexchange_clustering_p2p_v2",
        hf_path="mteb/stackexchange-clustering-p2p",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="stackexchange_clustering_v2",
        hf_path="mteb/stackexchange-clustering",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="arxiv_clustering_p2p",
        hf_path="mteb/arxiv-clustering-p2p",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="arxiv_clustering_s2s",
        hf_path="mteb/arxiv-clustering-s2s",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="biorxiv_clustering_p2p_v2",
        hf_path="mteb/biorxiv-clustering-p2p",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="medrxiv_clustering_p2p_v2",
        hf_path="mteb/medrxiv-clustering-p2p",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
    ClusteringTask(
        name="medrxiv_clustering_s2s_v2",
        hf_path="mteb/medrxiv-clustering-s2s",
        hf_name=None,
        split="test",
        sample_size=200,
        max_rows=5,
        max_labels=30,
    ),
]

SEED = 42


# ---------------------------------------------------------------------------
# Label decoding helpers
# ---------------------------------------------------------------------------
def _decode_labels(task_name: str, labels: list) -> list[str]:
    """Convert integer / single-char labels to human-readable strings."""
    if task_name == "twenty_newsgroups":
        return [TWENTY_NG_LABELS.get(l, str(l)) for l in labels]
    if task_name == "big_patent_clustering":
        return [BIG_PATENT_LABELS.get(str(l).lower(), str(l)) for l in labels]
    # Default: ensure strings
    return [str(l) for l in labels]


# ---------------------------------------------------------------------------
# Label capping: keep only top-N most frequent labels
# ---------------------------------------------------------------------------
def _cap_labels(sentences: list[str], labels: list[str], max_k: int) -> tuple[list[str], list[str]]:
    """If more than max_k unique labels, keep docs for top max_k labels only."""
    unique = set(labels)
    if len(unique) <= max_k:
        return sentences, labels

    counts = Counter(labels)
    top_labels = {lbl for lbl, _ in counts.most_common(max_k)}

    filtered = [(s, l) for s, l in zip(sentences, labels) if l in top_labels]
    return [s for s, _ in filtered], [l for _, l in filtered]


# ---------------------------------------------------------------------------
# Stratified sub-sampling within a single row
# ---------------------------------------------------------------------------
def _stratified_sample_lists(
    sentences: list[str], labels: list[str], n: int, seed: int
) -> tuple[list[str], list[str]]:
    """Sub-sample to at most *n* items, guaranteeing every label survives."""
    if len(sentences) <= n:
        sample_indices = list(range(len(sentences)))
    else:
        from sklearn.model_selection import train_test_split

        label_counts = Counter(labels)
        min_count = min(label_counts.values())
        indices = list(range(len(sentences)))

        if min_count < 2:
            # Can't sklearn-stratify with singletons — force-include one doc
            # per label then fill randomly.
            rng = random.Random(seed)
            label_to_indices: dict[str, list[int]] = {}
            for i, lbl in enumerate(labels):
                label_to_indices.setdefault(lbl, []).append(i)
            forced = [rng.choice(idxs) for idxs in label_to_indices.values()]
            forced_set = set(forced)
            remaining = [i for i in indices if i not in forced_set]
            n_extra = n - len(forced)
            if n_extra > 0:
                sample_indices = forced + rng.sample(remaining, min(n_extra, len(remaining)))
            else:
                sample_indices = forced[:n]
        else:
            _, sample_indices = train_test_split(
                indices, test_size=n, stratify=labels, random_state=seed,
            )

    # Shuffle the final chosen indices to completely randomize document order
    rng = random.Random(seed + 999)
    rng.shuffle(sample_indices)

    return (
        [sentences[i] for i in sample_indices],
        [labels[i] for i in sample_indices],
    )


# ---------------------------------------------------------------------------
# Process one task
# ---------------------------------------------------------------------------
def process_task(task: ClusteringTask, hf_org: str, dry_run: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  {task.name}")
    print(f"{'='*60}")

    print(f"  Loading {task.hf_path}..." + (f" [{task.hf_name}]" if task.hf_name else ""))
    try:
        ds = load_dataset(task.hf_path, task.hf_name, split=task.split)
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return

    n_rows_orig = len(ds)
    print(f"  Rows (cluster sets): {n_rows_orig}")

    # --- Step 0: Detect Format & Normalize to Nested ---
    is_flat = isinstance(ds[0]["sentences"], str)
    
    if is_flat:
        print(f"  Detected FLAT format. Synthesizing {task.max_rows} nested cluster sets...")
        from collections import defaultdict
        
        # Group all sentences by label
        lbl_to_sents = defaultdict(list)
        for row in ds:
            lbl_to_sents[row["labels"]].append(row["sentences"])
            
        all_labels = list(lbl_to_sents.keys())
        rng = random.Random(SEED + 100)
        
        synthetic_rows = []
        for _ in range(task.max_rows):
            # Select up to max_labels for this synthesized problem
            chosen_labels = rng.sample(all_labels, min(task.max_labels, len(all_labels)))
            
            row_sents = []
            row_lbls = []
            for lbl in chosen_labels:
                # Give each label a fair chunk, but allow _stratified_sample to cap it down later
                lbl_docs = lbl_to_sents[lbl]
                row_sents.extend(lbl_docs)
                row_lbls.extend([lbl] * len(lbl_docs))
                
            synthetic_rows.append({
                "sentences": row_sents,
                "labels": _decode_labels(task.name, row_lbls) # decode labels during synthesis
            })
            
        ds = Dataset.from_list(synthetic_rows)
        n_rows_orig = len(ds)
        print(f"  Successfully built {n_rows_orig} nested sets.")
    else:
        # Standard nested processing
        def decode_row(row):
            return {
                "sentences": row["sentences"],
                "labels": _decode_labels(task.name, row["labels"]),
            }
        ds = ds.map(decode_row, desc="Decoding labels")

    # --- Step 1: Drop rows with ≤ 1 unique label ---
    valid_indices = [i for i in range(len(ds)) if len(set(ds[i]["labels"])) > 1]
    if len(valid_indices) < n_rows_orig:
        dropped = n_rows_orig - len(valid_indices)
        print(f"  Dropped {dropped} row(s) with ≤ 1 unique label")
        ds = ds.select(valid_indices)

    # --- Step 2: Cap rows ---
    if len(ds) > task.max_rows:
        rng = random.Random(SEED)
        keep = sorted(rng.sample(range(len(ds)), task.max_rows))
        print(f"  Keeping {task.max_rows} of {len(ds)} rows: {keep}")
        ds = ds.select(keep)
    else:
        print(f"  Keeping all {len(ds)} rows")

    n_rows = len(ds)
    total_docs_before = sum(len(row["sentences"]) for row in ds)
    print(f"  Total docs (selected rows): {total_docs_before}")

    # --- Original label stats ---
    orig_unique_per_row = [len(set(row["labels"])) for row in ds]
    orig_unique_total = len(set(lbl for row in ds for lbl in row["labels"]))
    print(f"  Original unique labels (overall): {orig_unique_total}")
    print(f"  Original unique labels per row:   "
          f"min={min(orig_unique_per_row)}  "
          f"max={max(orig_unique_per_row)}  "
          f"avg={sum(orig_unique_per_row)/len(orig_unique_per_row):.1f}")

    # --- Step 3: Cap labels + Step 4: Subsample docs ---
    new_rows = []
    for i in range(n_rows):
        sents = ds[i]["sentences"]
        lbls = ds[i]["labels"]

        # Ensure sents and lbls match in length (some raw MTEB rows are malformed)
        min_len = min(len(sents), len(lbls))
        if min_len < len(sents) or min_len < len(lbls):
            sents = sents[:min_len]
            lbls = lbls[:min_len]

        # Cap labels
        sents, lbls = _cap_labels(sents, lbls, task.max_labels)

        # Subsample docs
        sents, lbls = _stratified_sample_lists(sents, lbls, task.sample_size, SEED)

        new_rows.append({"sentences": sents, "labels": lbls})

    sampled = Dataset.from_list(new_rows)

    total_docs_after = sum(len(row["sentences"]) for row in sampled)
    pct = (total_docs_after / total_docs_before * 100) if total_docs_before else 0
    print(f"\n  Total docs after:  {total_docs_after} (kept {pct:.1f}%)")

    # --- Final label stats ---
    samp_unique_per_row = [len(set(row["labels"])) for row in sampled]
    samp_unique_total = len(set(lbl for row in sampled for lbl in row["labels"]))
    print(f"  Final unique labels (overall):    {samp_unique_total}")
    print(f"  Final unique labels per row:      "
          f"min={min(samp_unique_per_row)}  "
          f"max={max(samp_unique_per_row)}  "
          f"avg={sum(samp_unique_per_row)/len(samp_unique_per_row):.1f}")
    print(f"  Avg docs per cluster per row:     "
          f"{sum(len(r['sentences'])/max(len(set(r['labels'])),1) for r in sampled)/n_rows:.1f}")

    # Per-row detail
    for i in range(n_rows):
        orig_n = len(ds[i]["sentences"])
        samp_n = len(sampled[i]["sentences"])
        orig_k = orig_unique_per_row[i]
        samp_k = samp_unique_per_row[i]
        docs_per_cluster = samp_n / samp_k if samp_k else 0
        capped = f" (capped from {orig_k})" if orig_k > task.max_labels else ""
        print(f"    Row {i}: {orig_n} → {samp_n} docs, "
              f"{samp_k} labels{capped}, "
              f"~{docs_per_cluster:.1f} docs/cluster")

    # --- Push (nested format) ---
    repo_id = f"{hf_org}/llm-eval-{task.name}"
    push_kwargs = {"split": "test"}
    if task.hf_name:
        push_kwargs["config_name"] = task.hf_name

    if dry_run:
        print(f"\n  🔍 Dry run — would push to {repo_id}")
        r0 = sampled[0]
        print(f"     Row 0: {len(r0['sentences'])} docs, "
              f"{len(set(r0['labels']))} unique labels")
        print(f"     First sentence: {r0['sentences'][0][:80]!r}...")
        print(f"     First label:    {r0['labels'][0]!r}")
    else:
        print(f"\n  ↑ Pushing to {repo_id}...")
        sampled.push_to_hub(repo_id, **push_kwargs)
        print(f"  ✓ Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sample clustering datasets")
    parser.add_argument("--hf-org", type=str, default="mteb")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks_to_run = TASKS
    if args.dataset:
        tasks_to_run = [t for t in TASKS if t.name == args.dataset]
        if not tasks_to_run:
            print(f"Unknown dataset: {args.dataset}")
            return

    for task in tasks_to_run:
        process_task(task, args.hf_org, args.dry_run)

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
