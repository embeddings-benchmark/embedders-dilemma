#!/usr/bin/env python3
"""Few-shot LLM classification ablation (standalone, does NOT touch existing code).

Runs classification tasks with N in-context examples per query to match
the supervision level of the embedding kNN pipeline.

Examples are sampled from the same test set (leave-one-out: the query
sample is excluded from the example pool). This mirrors how kNN operates
on the same data.

Usage:
    python scripts/run_fewshot_cls.py --n-shots 5
    python scripts/run_fewshot_cls.py --n-shots 5 --tasks imdb banking77
    python scripts/run_fewshot_cls.py --n-shots 5 --dry-run   # estimate cost only

Environment variables (from .env):
    MODEL, BASE_URL, TOKEN, USE_STRICT_JSON

Estimated cost: ~$5-10 for 5-shot on 3 tasks with Flash
"""

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Task definitions: dataset name, label map, instruction template
TASKS = {
    "imdb": {
        "dataset": "mteb/llm-eval-imdb",
        "labels": {"0": "negative", "1": "positive"},
        "instruction": "Classify the sentiment of the movie review as 'positive' or 'negative'.",
    },
    "banking77": {
        "dataset": "mteb/llm-eval-banking77",
        "labels": None,  # too many labels, use dataset labels directly
        "instruction": "Classify the customer query into one of the banking intent categories.",
    },
    "toxic": {
        "dataset": "mteb/llm-eval-toxic_conversations",
        "labels": {"0": "not toxic", "1": "toxic"},
        "instruction": "Classify whether the conversation is 'toxic' or 'not toxic'.",
    },
    "tweet_sentiment": {
        "dataset": "mteb/llm-eval-tweet_sentiment",
        "labels": {"0": "negative", "1": "neutral", "2": "positive"},
        "instruction": "Classify the sentiment of the tweet as 'negative', 'neutral', or 'positive'.",
    },
}


def build_fewshot_prompt(instruction: str, examples: list[dict], query_text: str,
                         label_map: dict | None) -> str:
    """Build a few-shot prompt with labeled examples."""
    parts = [instruction, "", "Here are some labeled examples:", ""]

    for i, ex in enumerate(examples, 1):
        label = ex["label"]
        if label_map:
            label = label_map.get(str(label), str(label))
        parts.append(f"Example {i}:")
        parts.append(f"  Text: \"{ex['text'][:500]}\"")
        parts.append(f"  Label: {label}")
        parts.append("")

    parts.append("Now classify this text:")
    parts.append(f"  Text: \"{query_text}\"")
    parts.append("")
    parts.append("Respond with JSON: {\"reasoning\": \"...\", \"output\": \"<label>\"}")

    return "\n".join(parts)


def sample_examples(dataset, query_idx: int, n_shots: int, label_col: str = "label") -> list[dict]:
    """Sample N examples, stratified by label, excluding the query."""
    labels = dataset[label_col]
    texts = dataset["text"]

    # Group indices by label
    by_label = defaultdict(list)
    for i, lab in enumerate(labels):
        if i != query_idx:
            by_label[str(lab)].append(i)

    # Sample evenly across labels
    unique_labels = sorted(by_label.keys())
    per_label = max(1, n_shots // len(unique_labels))
    remainder = n_shots - per_label * len(unique_labels)

    examples = []
    for lab in unique_labels:
        pool = by_label[lab]
        k = min(per_label, len(pool))
        examples.extend(random.sample(pool, k))

    # Fill remainder from random labels
    all_remaining = [i for lab in unique_labels for i in by_label[lab] if i not in examples]
    if remainder > 0 and all_remaining:
        examples.extend(random.sample(all_remaining, min(remainder, len(all_remaining))))

    random.shuffle(examples)
    return [{"text": texts[i], "label": labels[i]} for i in examples[:n_shots]]


async def run_fewshot_task(task_name: str, task_config: dict, n_shots: int,
                           dry_run: bool = False):
    """Run few-shot classification for one task."""
    from datasets import load_dataset
    from llm_judge.llm_client import send_request

    print(f"\n{'='*60}")
    print(f"Task: {task_name} (n_shots={n_shots})")
    print(f"{'='*60}")

    ds = load_dataset(task_config["dataset"], split="test")
    label_map = task_config["labels"]
    instruction = task_config["instruction"]

    # Get unique labels from dataset
    unique_labels = sorted(set(str(l) for l in ds["label"]))
    print(f"  Dataset size: {len(ds)}")
    print(f"  Labels: {len(unique_labels)}")
    print(f"  Examples per query: {n_shots}")

    if dry_run:
        # Estimate tokens
        avg_text_len = sum(len(t) for t in ds["text"][:50]) / 50
        example_tokens = n_shots * (avg_text_len / 4 + 20)  # rough char-to-token
        query_tokens = avg_text_len / 4 + 50  # instruction overhead
        total_input = (example_tokens + query_tokens) * len(ds)
        print(f"  Est. input tokens: {total_input/1e6:.2f}M")
        print(f"  Est. cost (Flash $0.50/MTok in): ${total_input/1e6 * 0.5:.2f}")
        return None

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    correct = 0
    total = 0
    samples = []

    from llm_judge.settings import Settings
    _settings = Settings()
    sem = asyncio.Semaphore(_settings.max_concurrency)

    async def classify_one(idx: int):
        nonlocal correct, total
        async with sem:
            text = ds["text"][idx]
            gold = str(ds["label"][idx])
            gold_name = label_map.get(gold, gold) if label_map else gold

            examples = sample_examples(ds, idx, n_shots)
            prompt = build_fewshot_prompt(instruction, examples, text, label_map)

            response, usage = await send_request(
                instructions="You are a text classifier. Respond only with valid JSON.",
                input=prompt,
                response_format=None,
            )

            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            # Parse response
            predicted = "PARSE_FAILURE"
            try:
                import re
                # Try JSON parse
                raw = response or ""
                # Remove thinking tags if present
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
                match = re.search(r'"output"\s*:\s*"([^"]+)"', raw)
                if match:
                    predicted = match.group(1)
            except Exception:
                pass

            is_correct = (predicted == gold_name or predicted == gold)
            if is_correct:
                correct += 1
            total += 1

            if len(samples) < 5:
                samples.append({
                    "input_text": text[:200],
                    "n_examples": n_shots,
                    "predicted": predicted,
                    "gold": gold_name,
                    "correct": is_correct,
                })

    # Run all classifications
    tasks = [classify_one(i) for i in range(len(ds))]
    from tqdm.asyncio import tqdm_asyncio
    await tqdm_asyncio.gather(*tasks)

    accuracy = correct / total if total > 0 else 0
    print("\n  Results:")
    print(f"    Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"    Tokens - in: {total_usage['input_tokens']/1e6:.2f}M"
          f"  out: {total_usage['output_tokens']/1e6:.2f}M"
          f"  total: {total_usage['total_tokens']/1e6:.2f}M")

    # Save results
    out_dir = ROOT / "ablation_results" / "fewshot" / f"{n_shots}shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task": task_name,
        "n_shots": n_shots,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "usage": total_usage,
        "samples": samples,
    }
    (out_dir / f"{task_name}.json").write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out_dir / f'{task_name}.json'}")

    return result


async def main_async(args):
    task_names = args.tasks or ["imdb", "banking77", "toxic"]
    results = []

    for tname in task_names:
        if tname not in TASKS:
            print(f"Unknown task: {tname}. Available: {list(TASKS.keys())}")
            continue
        r = await run_fewshot_task(tname, TASKS[tname], args.n_shots, args.dry_run)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY (n_shots={args.n_shots})")
        print(f"{'='*60}")
        print(f"{'Task':<25s} {'Accuracy':>10s} {'Zero-shot ref':>15s}")
        print(f"{'-'*25} {'-'*10} {'-'*15}")

        # Reference zero-shot scores (from our existing results)
        zeroshot_ref = {
            "imdb": 0.846,  # approximate from existing Pro results
            "banking77": 0.800,
            "toxic": 0.810,
            "tweet_sentiment": 0.700,
            "amazon_counterfactual": 0.816,
        }
        for r in results:
            ref = zeroshot_ref.get(r["task"], None)
            ref_str = f"{ref:.3f}" if ref else "N/A"
            print(f"{r['task']:<25s} {r['accuracy']:>10.4f} {ref_str:>15s}")


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot LLM classification ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-shots", type=int, default=5,
                        help="Number of in-context examples per query (default: 5)")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()),
                        help="Tasks to run (default: imdb banking77 toxic)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only estimate cost, don't run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for example sampling")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.dry_run:
        print("DRY RUN - estimating costs only\n")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
