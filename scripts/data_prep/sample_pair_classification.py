"""Sample and re-upload pair classification datasets for the LLM vs Embeddings benchmark.

All datasets use balanced class quotas (equal positives/negatives) + final shuffle.

Source formats:
  - NESTED (Sprint, Twitter): .json.gz where sent1 may be a plain string (one anchor,
    many candidates) OR a parallel list (one sent1 per pair). Both handled by the ijson
    parser. Streamed token-by-token — never loads the full JSON into RAM.
  - FLAT (LegalBench, PubChem): one row per pair, streamed via HF datasets.

Output: shuffled flat dataset with sentence1 | sentence2 | label (ClassLabel).

Usage:
    python scripts/sample_pair_classification.py --dry-run
    python scripts/sample_pair_classification.py --hf-org embeddings-benchmark
    python scripts/sample_pair_classification.py --dataset sprint_duplicate_questions --dry-run
"""

from __future__ import annotations

import argparse
import gc
import gzip
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Iterator

from datasets import Dataset, ClassLabel, load_dataset


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
@dataclass
class PairClassificationTask:
    name: str
    hf_path: str
    hf_name: str | None
    split: str
    sample_size: int
    format: Literal["nested", "flat"]
    sent1_col: str = "sentence1"
    sent2_col: str = "sentence2"
    label_col: str = "labels"
    label_names: list[str] = field(default_factory=lambda: ["negative", "positive"])


TASKS = [
    PairClassificationTask(
        name="sprint_duplicate_questions",
        hf_path="mteb/sprintduplicatequestions-pairclassification",
        hf_name=None,
        split="test",
        sample_size=500,
        format="nested",
        sent1_col="sent1",
        sent2_col="sent2",
        label_col="labels",
        label_names=["not_duplicate", "duplicate"],
    ),
    PairClassificationTask(
        name="twitter_url_corpus",
        hf_path="mteb/twitterurlcorpus-pairclassification",
        hf_name=None,
        split="test",
        sample_size=500,
        format="nested",
        sent1_col="sent1",
        sent2_col="sent2",
        label_col="labels",
        label_names=["not_paraphrase", "paraphrase"],
    ),
    PairClassificationTask(
        name="legal_bench_pc",
        hf_path="mteb/LegalBenchPC",
        hf_name=None,
        split="test",
        sample_size=500,
        format="flat",
        sent1_col="sentence1",
        sent2_col="sentence2",
        label_col="labels",
        # Mixed sub-tasks: yes/no, correct/incorrect, relevant/irrelevant
        label_names=["no/incorrect/irrelevant", "yes/correct/relevant"],
    ),
    # RTE3: Multilingual (de, en, fr, it)
    *[PairClassificationTask(
        name=f"rte3_{lang}",
        hf_path="mteb/RTE3",
        hf_name=lang,
        split="test",
        sample_size=500,
        format="flat",
        sent1_col="sentence1",
        sent2_col="sentence2",
        label_col="labels",
        label_names=["not_entailment", "entailment"],
    ) for lang in ["de", "en", "fr", "it"]],
]

SEED = 42


# ---------------------------------------------------------------------------
# ijson streaming for nested .json.gz files
# ---------------------------------------------------------------------------
def _stream_nested_gz(
    local_path: str, sent1_col: str, sent2_col: str, label_col: str
) -> Iterator[tuple[str, str, int]]:
    """Stream (s1, s2, label) pairs from a nested .json.gz using ijson.

    Handles two sent1 shapes:
      - Plain string: one anchor question paired with each sent2/label element.
      - Parallel list: zip(sent1, sent2, labels) to get one pair per index.

    Never loads the full JSON into RAM — token-by-token streaming.
    """
    import ijson

    with gzip.open(local_path, "rb") as f:
        sent1_val: str | list[str] | None = None  # str or list depending on JSON
        sent2_buf: list[str] = []
        labels_buf: list[int] = []
        in_sent1 = False
        in_sent2 = False
        in_labels = False

        for _prefix, event, value in ijson.parse(f):
            if event == "map_key":
                in_sent1 = (value == sent1_col)
                in_sent2 = (value == sent2_col)
                in_labels = (value == label_col)

            elif event == "start_array":
                if in_sent1:
                    # sent1 is an array — allocate list
                    sent1_val = []

            elif event in ("string", "number", "integer"):
                if in_sent1:
                    if isinstance(sent1_val, list):
                        sent1_val.append(str(value))
                    elif sent1_val is None:
                        # sent1 is a plain string (scalar)
                        sent1_val = str(value)
                        in_sent1 = False
                elif in_sent2 and isinstance(value, str):
                    sent2_buf.append(value)
                elif in_labels:
                    labels_buf.append(int(value))

            elif event == "end_array":
                if in_sent1:
                    in_sent1 = False
                elif in_sent2:
                    in_sent2 = False
                elif in_labels and labels_buf:
                    # Complete row — emit all pairs
                    if isinstance(sent1_val, list):
                        # Parallel lists: zip all three
                        for s1, s2, lbl in zip(sent1_val, sent2_buf, labels_buf):
                            yield s1, s2, lbl
                    else:
                        # Single anchor string with many candidates
                        s1 = sent1_val or ""
                        for s2, lbl in zip(sent2_buf, labels_buf):
                            yield s1, s2, lbl
                    # Reset for next row (list-of-dicts case)
                    sent1_val = None
                    sent2_buf = []
                    labels_buf = []
                    in_labels = False


# ---------------------------------------------------------------------------
# Balanced quota helper
# ---------------------------------------------------------------------------
def _balanced_quotas(class_totals: dict[int, int], n: int) -> dict[int, int]:
    """Equal per-class allocation, capped by availability."""
    sorted_classes = sorted(class_totals.keys())
    if sum(class_totals.values()) <= n:
        return dict(class_totals)
    k = len(sorted_classes)
    per_class = n // k
    quotas = {cls: min(per_class, class_totals[cls]) for cls in sorted_classes}
    leftover = n - sum(quotas.values())
    for cls in sorted_classes:
        if leftover <= 0:
            break
        give = min(leftover, class_totals[cls] - quotas[cls])
        quotas[cls] += give
        leftover -= give
    return quotas


# ---------------------------------------------------------------------------
# Reservoir update helper
# ---------------------------------------------------------------------------
def _reservoir_update(
    reservoirs: dict, seen: dict, s1: str, s2: str, li: int,
    quotas: dict, rng: random.Random
) -> None:
    seen[li] += 1
    q = quotas[li]
    if len(reservoirs[li]) < q:
        reservoirs[li].append((s1, s2, li))
    else:
        j = rng.randint(0, seen[li] - 1)
        if j < q:
            reservoirs[li][j] = (s1, s2, li)


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------
def sample_pairs(task: PairClassificationTask, seed: int) -> Dataset:
    """Two-pass balanced reservoir sampler with final shuffle.

    Nested: streams .json.gz with ijson — O(row_size) peak memory.
    Flat: streams via HF datasets — O(1) per row.
    """
    rng = random.Random(seed)

    if task.format == "nested":
        from huggingface_hub import hf_hub_download
        fname = f"{task.split}.json.gz"
        print(f"  Downloading raw {fname}...")
        local_path = hf_hub_download(
            repo_id=task.hf_path, filename=fname, repo_type="dataset",
        )

        # Pass 1: count classes
        print("  Pass 1: counting classes (ijson)...")
        class_totals: dict[int, int] = {}
        for _, _, li in _stream_nested_gz(
            local_path, task.sent1_col, task.sent2_col, task.label_col
        ):
            class_totals[li] = class_totals.get(li, 0) + 1
        print(f"  Total pairs: {sum(class_totals.values())}, classes: {class_totals}")

        quotas = _balanced_quotas(class_totals, task.sample_size)
        print(f"  Per-class quotas: {quotas}")
        sorted_classes = sorted(class_totals.keys())

        # Pass 2: reservoir sample
        print("  Pass 2: reservoir sampling (ijson)...")
        reservoirs: dict[int, list] = {cls: [] for cls in sorted_classes}
        seen: dict[int, int] = {cls: 0 for cls in sorted_classes}
        for s1, s2, li in _stream_nested_gz(
            local_path, task.sent1_col, task.sent2_col, task.label_col
        ):
            _reservoir_update(reservoirs, seen, s1, s2, li, quotas, rng)

    else:
        # Flat: stream via HF datasets
        print("  Pass 1: counting classes (streaming)...")
        class_totals = {}
        for row in load_dataset(
            task.hf_path, task.hf_name, split=task.split, streaming=True
        ):
            li = int(row[task.label_col])
            class_totals[li] = class_totals.get(li, 0) + 1
        print(f"  Total pairs: {sum(class_totals.values())}, classes: {class_totals}")

        quotas = _balanced_quotas(class_totals, task.sample_size)
        print(f"  Per-class quotas: {quotas}")
        sorted_classes = sorted(class_totals.keys())

        print("  Pass 2: reservoir sampling (streaming)...")
        reservoirs = {cls: [] for cls in sorted_classes}
        seen = {cls: 0 for cls in sorted_classes}
        for row in load_dataset(
            task.hf_path, task.hf_name, split=task.split, streaming=True
        ):
            li = int(row[task.label_col])
            _reservoir_update(
                reservoirs, seen,
                row[task.sent1_col], row[task.sent2_col], li,
                quotas, rng,
            )

    # Combine reservoirs and shuffle
    all_pairs: list[tuple[str, str, int]] = []
    for cls in sorted_classes:
        all_pairs.extend(reservoirs[cls])
    del reservoirs
    gc.collect()

    rng.shuffle(all_pairs)  # shuffle so classes are interleaved

    out: dict[str, list] = {
        "sentence1": [p[0] for p in all_pairs],
        "sentence2": [p[1] for p in all_pairs],
        "label": [p[2] for p in all_pairs],
    }
    del all_pairs
    gc.collect()

    dist = dict(sorted(Counter(out["label"]).items()))
    print(f"  Sampled {len(out['label'])} pairs, distribution: {dist}")
    result = Dataset.from_dict(out)
    del out
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Process one task
# ---------------------------------------------------------------------------
def process_task(task: PairClassificationTask, hf_org: str, dry_run: bool) -> None:
    print(f"\n{'='*50}")
    print(f"  {task.name}  [{task.format}]")
    print(f"{'='*50}")

    sampled = sample_pairs(task, SEED)

    features = sampled.features.copy()
    features["label"] = ClassLabel(names=task.label_names)
    sampled = sampled.cast(features)
    print(f"  ✓ Cast label → ClassLabel{task.label_names}")

    # Determine repo_id and config_name
    if task.name.startswith("rte3_"):
        repo_id = f"{hf_org}/llm-eval-rte3"
        config_name = task.hf_name
    else:
        repo_id = f"{hf_org}/llm-eval-{task.name}"
        config_name = "default"

    if dry_run:
        print(f"\n  🔍 Dry run — would push to {repo_id} (config: {config_name})")
        print(f"     Columns: {sampled.column_names}")
        print(f"     Rows: {len(sampled)}")
        print("     First 3 rows:")
        for i in range(min(3, len(sampled))):
            r = sampled[i]
            print(f"       [{i}] label={r['label']}  s1={r['sentence1'][:60]!r}  s2={r['sentence2'][:60]!r}")
    else:
        print(f"\n  ↑ Pushing to {repo_id} (config: {config_name})...")
        sampled.push_to_hub(repo_id, config_name=config_name, split="test")
        print(f"  ✓ Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample pair classification datasets for LLM vs Embeddings benchmark"
    )
    parser.add_argument("--hf-org", type=str, default="mteb")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks_to_run = TASKS
    if args.dataset:
        tasks_to_run = [t for t in TASKS if t.name == args.dataset]
        if not tasks_to_run:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {[t.name for t in TASKS]}")
            return

    print(f"Processing {len(tasks_to_run)} pair classification datasets")
    print(f"HF org: {args.hf_org} | Seed: {SEED}")

    for task in tasks_to_run:
        process_task(task, args.hf_org, args.dry_run)

    print("\n\nDone! 🎉")


if __name__ == "__main__":
    main()
