"""Sample and re-upload STS datasets for the LLM vs Embeddings benchmark.

Usage:
    python scripts/sample_sts.py --dry-run
    python scripts/sample_sts.py --hf-org embeddings-benchmark
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class STSTask:
    name: str
    hf_path: str
    hf_name: str | None
    split: str
    sample_size: int = 500

TASKS = [
    STSTask(name="biosses", hf_path="mteb/biosses-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="sickr", hf_path="mteb/sickr-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="sts12", hf_path="mteb/sts12-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="sts13", hf_path="mteb/sts13-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="sts14", hf_path="mteb/sts14-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="sts15", hf_path="mteb/sts15-sts", hf_name=None, split="test", sample_size=500),
    STSTask(name="stsbenchmark", hf_path="mteb/stsbenchmark-sts", hf_name=None, split="test", sample_size=500),
    
    # STS17: representative languages (300 pairs to save cost)
    *[STSTask(name=f"sts17_{lang}", hf_path="mteb/sts17-crosslingual-sts", hf_name=lang, split="test", sample_size=300) 
      for lang in ["en-en", "en-de", "es-es", "fr-en"]],
      
    # STS22.v2: representative languages (300 pairs to save cost)
    *[STSTask(name=f"sts22_v2_{lang}", hf_path="mteb/sts22-crosslingual-sts", hf_name=lang, split="test", sample_size=300) 
      for lang in ["en", "de", "es", "fr", "ru", "zh"]],
]

def process_task(task: STSTask, hf_org: str, dry_run: bool):
    print(f"\nProcessing {task.name}...")
    
    try:
        ds = load_dataset(task.hf_path, task.hf_name, split=task.split)
    except Exception as e:
        print(f"  ✗ Failed to load {task.hf_path} [{task.hf_name}]: {e}")
        return

    print(f"  Loaded {len(ds)} pairs.")
    
    # Analysis of gold scores
    if "score" in ds.column_names:
        scores = ds["score"]
        min_s = min(scores)
        max_s = max(scores)
        
        # Check if any score has a decimal part
        is_continuous = any(float(s) != int(float(s)) for s in scores)
        score_type = "Continuous/Decimal" if is_continuous else "Discrete/Integer"
        
        print("  [DATASET ANALYSIS]")
        print(f"  Score Type: {score_type}")
        print(f"  Actual Range: {min_s} to {max_s}")
        
    if len(ds) > task.sample_size:
        ds = ds.shuffle(seed=42).select(range(task.sample_size))
        print(f"  Sampled down to {task.sample_size} pairs.")

    # Determine repo_id and config_name
    base_name = task.name
    if "_" in task.name:
        if task.name.startswith("sts17_") or task.name.startswith("sts22_v2_"):
            base_name = task.name.rsplit("_", 1)[0]

    if task.hf_name and (task.name.startswith("sts17_") or task.name.startswith("sts22_v2_")):
        repo_id = f"{hf_org}/llm-eval-{base_name}"
        config_name = task.hf_name
    else:
        repo_id = f"{hf_org}/llm-eval-{task.name}"
        config_name = "default"

    if dry_run:
        print(f"  🔍 Dry run: would push to {repo_id} (config: {config_name})")
    else:
        print(f"  ↑ Pushing to {repo_id} (config: {config_name})...")
        ds.push_to_hub(repo_id, config_name=config_name, split="test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-org", type=str, default="mteb")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for task in TASKS:
        process_task(task, args.hf_org, args.dry_run)

if __name__ == "__main__":
    main()
