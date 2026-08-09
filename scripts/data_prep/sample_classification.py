"""Sample and re-upload classification datasets for the LLM vs Embeddings benchmark.

Downloads 7 MTEB classification datasets, stratified-samples to 500 examples
(except Banking77 which is kept full), and pushes to HuggingFace.

Usage:
    # Dry run — sample and print stats, don't push
    python scripts/sample_classification.py --dry-run

    # Push sampled datasets to HF
    python scripts/sample_classification.py --hf-org mteb

    # Single dataset
    python scripts/sample_classification.py --hf-org mteb --dataset imdb
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from datasets import Dataset, load_dataset


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
@dataclass
class ClassificationTask:
    """Definition of a classification dataset to sample."""
    name: str                   # short identifier
    hf_path: str                # HuggingFace dataset path
    hf_name: str | None         # HF config/subset name (None = default)
    split: str                  # which split to sample from
    sample_size: int | None     # None = keep full (no subsampling)
    text_col: str               # column containing the text
    label_col: str              # column containing the label
    label_text_col: str | None  # column with human-readable label name (if exists)
    label_map: dict[int, str] | None = None  # fallback mapping for plain-int labels


TASKS = [
    ClassificationTask(
        name="imdb",
        hf_path="mteb/imdb",
        hf_name=None,
        split="test",
        sample_size=500,
        text_col="text",
        label_col="label",
        label_text_col="label_text",
        label_map={0: "negative", 1: "positive"},
    ),
    ClassificationTask(
        name="banking77",
        hf_path="mteb/banking77",
        hf_name=None,
        split="test",
        sample_size=None,
        text_col="text",
        label_col="label",
        label_text_col="label_text",
    ),
    # MTOP Domain: Multilingual Task-Oriented Semantic Parsing
    *[ClassificationTask(
        name=f"mtop_domain_{lang}",
        hf_path="mteb/MTOPDomainClassification",
        hf_name=lang,
        split="test",
        sample_size=800,
        text_col="text",
        label_col="label",
        label_text_col=None,
        label_map={
            0: "messaging",
            1: "calling",
            2: "event",
            3: "timer",
            4: "music",
            5: "weather",
            6: "alarm",
            7: "people",
            8: "reminder",
            9: "recipes",
            10: "news"
        }
    ) for lang in ["en", "de", "fr"]],

    # Massive Intent: 51 languages. Sampling representative subset.
    *[ClassificationTask(
        name=f"massive_intent_{lang}",
        hf_path="mteb/amazon_massive_intent",
        hf_name=lang,
        split="test",
        sample_size=1000, # Increased for 60 labels
        text_col="text",
        label_col="label",
        label_text_col=None,
    ) for lang in ["en", "de", "fr", "ja"]],

    # Massive Scenario
    *[ClassificationTask(
        name=f"massive_scenario_{lang}",
        hf_path="mteb/amazon_massive_scenario",
        hf_name=lang,
        split="test",
        sample_size=800, # Increased for 18 labels
        text_col="text",
        label_col="label",
        label_text_col=None,
    ) for lang in ["en", "de", "fr", "ja"]],

    ClassificationTask(
        name="toxic_conversations",
        hf_path="mteb/toxic_conversations_50k",
        hf_name=None,
        split="test",
        sample_size=500,
        text_col="text",
        label_col="label",
        label_text_col=None,
        label_map={0: "not toxic", 1: "toxic"},
    ),
    ClassificationTask(
        name="tweet_sentiment",
        hf_path="mteb/tweet_sentiment_extraction",
        hf_name=None,
        split="test",
        sample_size=500,
        text_col="text",
        label_col="label",
        label_text_col="label_text",
    ),
    # Amazon Counterfactual: 3 languages
    *[ClassificationTask(
        name=f"amazon_counterfactual_{lang}",
        hf_path="mteb/amazon_counterfactual",
        hf_name=lang,
        split="test",
        sample_size=500,
        text_col="text",
        label_col="label",
        label_text_col=None,
        label_map={0: "not-counterfactual", 1: "counterfactual"},
    ) for lang in ["en", "de", "ja"]],
]

SEED = 42


# ---------------------------------------------------------------------------
# Sampling logic
# ---------------------------------------------------------------------------
def stratified_sample(ds: Dataset, n: int, label_col: str, seed: int) -> Dataset:
    """Stratified sample of n examples, preserving label distribution.

    If the dataset has fewer than n examples, returns the full dataset.
    """
    if len(ds) <= n:
        return ds

    # Use sklearn for proper stratified splitting
    from sklearn.model_selection import train_test_split

    labels = ds[label_col]
    indices = list(range(len(ds)))

    # Check if any label has fewer than 2 examples (can't stratify)
    from collections import Counter
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count < 2:
        # Fall back to random sampling if stratification isn't possible
        print("    ⚠ Some labels have <2 examples, using random sampling")
        return ds.shuffle(seed=seed).select(range(n))

    _, sample_indices = train_test_split(
        indices,
        test_size=n,
        stratify=labels,
        random_state=seed,
    )
    return ds.select(sorted(sample_indices))


def process_task(task: ClassificationTask, hf_org: str, dry_run: bool) -> None:
    """Load, sample, and push one classification dataset."""
    print(f"\n{'='*50}")
    print(f"  {task.name}" + (f" [{task.hf_name}]" if task.hf_name else ""))
    print(f"{'='*50}")

    # Load
    print(f"  Loading {task.hf_path}" + (f" [{task.hf_name}]" if task.hf_name else ""))
    full_ds = load_dataset(task.hf_path, task.hf_name)
    ds = full_ds[task.split]
    print(f"  Loaded: {len(ds)} examples from {task.split} split")

    # Show label distribution
    label_counts = Counter(ds[task.label_col])
    n_labels = len(label_counts)
    print(f"  Labels: {n_labels} unique")
    if n_labels <= 10:
        for label, count in sorted(label_counts.items()):
            pct = count / len(ds) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")

    # Sample
    if task.sample_size is not None:
        print(f"\n  Sampling {task.sample_size} (stratified, seed={SEED})...")
        sampled = stratified_sample(ds, task.sample_size, task.label_col, SEED)
    else:
        print(f"\n  No subsampling (keeping all {len(ds)} examples)")
        sampled = ds

    # Find label names before removing columns
    label_names = None
    from datasets import ClassLabel
    
    # 1. Try to get names from ClassLabel feature in the dataset itself
    label_feature = ds.features.get(task.label_col)
    if isinstance(label_feature, ClassLabel):
        label_names = label_feature.names
        print(f"  ✓ Found existing ClassLabel with {len(label_names)} classes")
    
    # 2. Try manual label_map if provided
    elif task.label_map:
        max_label = max(task.label_map.keys())
        label_names = [task.label_map.get(i, str(i)) for i in range(max_label + 1)]
        print(f"  ✓ Using manual label_map with {len(label_names)} classes")
    
    # 3. Try to extract from label_text_col if provided
    elif task.label_text_col and task.label_text_col in ds.column_names:
        # Extract mapping from ALL splits to ensure all classes are captured
        mapping = {}
        for split_name in full_ds.keys():
            # Get unique pairs from this split
            split_ds = full_ds[split_name].select_columns([task.label_col, task.label_text_col])
            for row in split_ds:
                l_id, l_text = row[task.label_col], row[task.label_text_col]
                if l_id not in mapping:
                    mapping[l_id] = l_text
        
        if mapping:
            max_label = max(mapping.keys())
            label_names = [mapping.get(i, str(i)) for i in range(max_label + 1)]
            print(f"  ✓ Extracted {len(mapping)} classes from {task.label_text_col} column across all splits")

    # 4. Fallback: Aggregate unique labels across all splits
    if label_names is None:
        unique_labels = set()
        for split_name in full_ds.keys():
            unique_labels.update(full_ds[split_name][task.label_col])
        unique_labels = sorted(list(unique_labels))
        
        try:
            # If they look like integers, use a continuous range
            max_label = int(max(unique_labels))
            label_names = [str(i) for i in range(max_label + 1)]
            print(f"  ⚠ No label names found — using numeric range up to {max_label}")
        except (ValueError, TypeError):
            # If they are actual strings, use them directly
            label_names = [str(l) for l in unique_labels]
            print(f"  ⚠ No label names found — using {len(label_names)} unique values from all splits")

    def format_dataset(dataset):
        # Rename to standard names if needed
        if task.text_col != "text":
            dataset = dataset.rename_column(task.text_col, "text")
        if task.label_col != "label":
            dataset = dataset.rename_column(task.label_col, "label")

        # Drop all columns except text and label
        drop_cols = [c for c in dataset.column_names if c not in ["text", "label"]]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

        # Before casting, if the labels are strings but we have a ClassLabel with those names,
        # we need to map the string values to their corresponding integer indices.
        # Check first element to see if mapping is needed
        if len(dataset) > 0 and isinstance(dataset[0]["label"], str) and dataset[0]["label"] in label_names:
            print("  ⚠ Labels are currently strings, mapping to integer indices...")
            # Create a mapping from label name to index
            str_to_id = {name: i for i, name in enumerate(label_names)}
            
            # Map the dataset
            dataset = dataset.map(
                lambda x: {"label": str_to_id.get(x["label"], -1)},
                desc="Mapping string labels to integers"
            )

            # Filter out any unmapped labels (-1)
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: x["label"] != -1, desc="Filtering invalid labels")
            if len(dataset) < original_len:
                print(f"  ⚠ Dropped {original_len - len(dataset)} examples with unknown string labels")

        # Cast label column to ClassLabel
        from datasets import ClassLabel
        features = dataset.features.copy()
        if not isinstance(features["label"], ClassLabel) or features["label"].names != label_names:
            features["label"] = ClassLabel(names=label_names)
            dataset = dataset.cast(features)
        return dataset

    sampled = format_dataset(sampled)
    print("  ✓ Formatted test split")

    from datasets import DatasetDict
    ds_dict = DatasetDict()
    ds_dict["test"] = sampled

    print("\n  Loading train split...")
    try:
        ds_train = load_dataset(task.hf_path, task.hf_name, split="train")
        ds_train = format_dataset(ds_train)
        ds_dict["train"] = ds_train
        print(f"  ✓ Added train split with {len(ds_train)} examples")
    except Exception as e:
        print(f"  ⚠ Could not load train split: {e}")

    # Show sampled distribution
    sampled_counts = Counter(sampled["label"])
    print(f"\n  Test Split Sampled: {len(sampled)} examples, {len(sampled_counts)} labels")
    if n_labels <= 10:
        for label, count in sorted(sampled_counts.items()):
            pct = count / len(sampled) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")

    # Push
    # If it's a multilingual dataset (indicated by task.hf_name),
    # strip the language suffix from the task name to get the base repo name.
    base_name = task.name
    if task.hf_name and base_name.endswith(f"_{task.hf_name}"):
        base_name = base_name[:-(len(task.hf_name) + 1)]
    
    repo_id = f"{hf_org}/llm-eval-{base_name}"
    push_kwargs = {}
    if task.hf_name:
        push_kwargs["config_name"] = task.hf_name

    if dry_run:
        print(f"\n  🔍 Dry run — would push to {repo_id} {push_kwargs}")
        print(f"     Splits: {list(ds_dict.keys())}")
        print(f"     Test cols: {sampled.column_names}")
        print(f"     Test rows: {len(sampled)}")
        if "train" in ds_dict:
            print(f"     Train rows: {len(ds_dict['train'])}")
        print(f"     Test Sample: {sampled[0]}")
    else:
        print(f"\n  ↑ Pushing to {repo_id} {push_kwargs}...")
        ds_dict.push_to_hub(repo_id, **push_kwargs)
        print(f"  ✓ Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample classification datasets for LLM vs Embeddings benchmark"
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
        help="Process only this dataset (by name). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and print stats but don't push to HuggingFace",
    )
    args = parser.parse_args()

    tasks_to_run = TASKS
    if args.dataset:
        tasks_to_run = [t for t in TASKS if t.name == args.dataset]
        if not tasks_to_run:
            available = [t.name for t in TASKS]
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {available}")
            return

    print(f"Processing {len(tasks_to_run)} classification datasets")
    print(f"HF org: {args.hf_org}")
    print("Sample size: 500 (except Banking77 = full)")
    print(f"Seed: {SEED}")

    for task in tasks_to_run:
        process_task(task, args.hf_org, args.dry_run)

    print("\n\nDone! 🎉")


if __name__ == "__main__":
    main()
