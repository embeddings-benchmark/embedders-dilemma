#!/usr/bin/env python3
"""Count raw-text tokens per task, broken down by split (test/train, query/corpus).

Uses tiktoken (GPT-4o tokenizer) as a model-independent proxy.
This gives approximate token counts for cost estimation. Actual counts
vary by ±15-40% depending on the model's vocabulary size (see the
per-model script count_tokens_per_model.py which uses each embedding
model's own tokenizer for exact cost computation).

Difference from count_tokens_per_model.py:
  - count_tokens_per_model.py: extracts ALL text into one flat list,
    tokenizes with each of the 26 embedding model tokenizers, and
    computes per-model cost. Used for the cost analysis in the paper.
  - THIS script: counts tokens per task per split with a single proxy
    tokenizer. Used for the datasets table so readers can estimate
    costs for their own models/APIs.

Note: these are RAW TEXT tokens. LLM corpus-in-context (LOFT) formatting
adds ~20-25 tokens per document in ID/TITLE/CONTENT markers. For tasks
with many short documents (e.g., TempReasonL1: 12,504 docs), this can
inflate the actual LLM prompt by 3-7x over raw text.

Usage:
    python scripts/count_tokens_per_task.py
    python scripts/count_tokens_per_task.py --output data/token_counts_per_task.csv
"""
import argparse
import csv
import sys
from pathlib import Path

from datasets import load_dataset
import tiktoken

ROOT = Path(__file__).parent.parent
enc = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(texts):
    return sum(len(enc.encode(str(t))) for t in texts)


# fmt: off
TASKS = [
    # Classification: text column, test + train splits
    {"name": "ImdbClassification",            "path": "mteb/llm-eval-imdb",               "type": "cls", "configs": [None]},
    {"name": "Banking77Classification",       "path": "mteb/llm-eval-banking77",           "type": "cls", "configs": [None]},
    {"name": "AmazonCounterfactualCls",       "path": "mteb/llm-eval-amazon_counterfactual", "type": "cls", "configs": ["en", "de", "ja"]},
    {"name": "MTOPDomainCls",                 "path": "mteb/llm-eval-mtop_domain",         "type": "cls", "configs": ["en", "de", "fr"]},
    {"name": "MassiveIntentCls",              "path": "mteb/llm-eval-massive_intent",      "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "MassiveScenarioCls",            "path": "mteb/llm-eval-massive_scenario",    "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "ToxicConversationsCls",         "path": "mteb/llm-eval-toxic_conversations", "type": "cls", "configs": [None]},
    {"name": "TweetSentimentCls",             "path": "mteb/llm-eval-tweet_sentiment",     "type": "cls", "configs": [None]},
    # STS: sentence1 + sentence2 from test split
    {"name": "STSBenchmark",    "path": "mteb/llm-eval-stsbenchmark", "type": "sts", "configs": [None]},
    {"name": "SICK-R",          "path": "mteb/llm-eval-sickr",        "type": "sts", "configs": [None]},
    {"name": "STS12",           "path": "mteb/llm-eval-sts12",        "type": "sts", "configs": [None]},
    {"name": "STS13",           "path": "mteb/llm-eval-sts13",        "type": "sts", "configs": [None]},
    {"name": "STS14",           "path": "mteb/llm-eval-sts14",        "type": "sts", "configs": [None]},
    {"name": "STS15",           "path": "mteb/llm-eval-sts15",        "type": "sts", "configs": [None]},
    {"name": "STS16",           "path": "mteb/llm-eval-sts16",        "type": "sts", "configs": [None]},
    {"name": "BIOSSES",         "path": "mteb/llm-eval-biosses",      "type": "sts", "configs": [None]},
    {"name": "STS17",           "path": "mteb/llm-eval-sts17",        "type": "sts", "configs": ["en-en", "en-de", "es-es", "fr-en"]},
    {"name": "STS22v2",         "path": "mteb/llm-eval-sts22_v2",     "type": "sts", "configs": ["en", "de", "es", "fr", "ru", "zh"]},
    # Clustering: sentences from test split
    {"name": "RedditClusteringP2P",         "path": "mteb/llm-eval-reddit_clustering_p2p",          "type": "clust", "configs": [None]},
    {"name": "TwentyNewsgroupsV2",          "path": "mteb/llm-eval-twenty_newsgroups_v2",           "type": "clust", "configs": [None]},
    {"name": "StackExchangeClustP2PV2",     "path": "mteb/llm-eval-stackexchange_clustering_p2p_v2","type": "clust", "configs": [None]},
    {"name": "StackExchangeClustV2",        "path": "mteb/llm-eval-stackexchange_clustering_v2",    "type": "clust", "configs": [None]},
    {"name": "ArxivClusteringP2P",          "path": "mteb/llm-eval-arxiv_clustering_p2p",           "type": "clust", "configs": [None]},
    {"name": "ArxivClusteringS2S",          "path": "mteb/llm-eval-arxiv_clustering_s2s",           "type": "clust", "configs": [None]},
    {"name": "BiorxivClusteringP2PV2",      "path": "mteb/llm-eval-biorxiv_clustering_p2p_v2",      "type": "clust", "configs": [None]},
    {"name": "MedrxivClusteringP2PV2",      "path": "mteb/llm-eval-medrxiv_clustering_p2p_v2",      "type": "clust", "configs": [None]},
    {"name": "MedrxivClusteringS2SV2",      "path": "mteb/llm-eval-medrxiv_clustering_s2s_v2",      "type": "clust", "configs": [None]},
    # Pair Classification: sentence1 + sentence2 from test split
    {"name": "SprintDuplicateQuestionsPC",  "path": "mteb/llm-eval-sprint_duplicate_questions", "type": "pair", "configs": [None]},
    {"name": "TwitterURLCorpusPC",          "path": "mteb/llm-eval-twitter_url_corpus",         "type": "pair", "configs": [None]},
    {"name": "LegalBenchPC",                "path": "mteb/llm-eval-legal_bench_pc",              "type": "pair", "configs": [None]},
    {"name": "RTE3PC",                      "path": "mteb/llm-eval-rte3",                        "type": "pair", "configs": ["de", "en", "fr", "it"]},
    # Retrieval: queries + corpus splits (current 6-task suite; lang in comment)
    {"name": "AILAStatutes",                "path": "mteb/llm-eval-aila-statutes",               "type": "ret"},  # eng
    {"name": "FQuADRetrieval",              "path": "mteb/llm-eval-fquad",                       "type": "ret"},  # fra
    {"name": "HC3FinanceRetrieval",         "path": "mteb/llm-eval-hc3-finance",                 "type": "ret"},  # eng
    {"name": "LegalBenchConsumerContractsQA","path": "mteb/llm-eval-legalbench-consumer-contracts","type": "ret"}, # eng
    {"name": "PublicHealthQA",              "path": "mteb/llm-eval-public-health-qa",            "type": "ret"},  # eng
    {"name": "TwitterHjerneRetrieval",      "path": "mteb/llm-eval-twitter-hjerne",              "type": "ret"},  # dan
]
# fmt: on


def load_ds(path, config=None, split="test"):
    try:
        if config:
            return load_dataset(path, config, split=split)
        return load_dataset(path, split=split)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()

    rows = []

    for task in TASKS:
        name = task["name"]
        path = task["path"]
        ttype = task["type"]
        configs = task.get("configs", [None])

        sys.stdout.write(f"  {name:40s} ")
        sys.stdout.flush()

        if ttype == "cls":
            test_tok = 0
            train_tok = 0
            test_n = 0
            train_n = 0
            for cfg in configs:
                ds = load_ds(path, cfg, "test")
                if ds:
                    test_tok += count_tokens([r.get("text", "") for r in ds])
                    test_n += len(ds)
                ds = load_ds(path, cfg, "train")
                if ds:
                    train_tok += count_tokens([r.get("text", "") for r in ds])
                    train_n += len(ds)
            rows.append({
                "task": name, "category": "Classification",
                "test_tokens": test_tok, "train_tokens": train_tok,
                "query_tokens": 0, "corpus_tokens": 0,
                "total_tokens": test_tok + train_tok,
                "test_n": test_n, "train_n": train_n,
            })
            print(f"test={test_tok:>8,}  train={train_tok:>8,}")

        elif ttype in ("sts", "pair"):
            cat = "STS" if ttype == "sts" else "PairClassification"
            tok = 0
            n = 0
            for cfg in configs:
                ds = load_ds(path, cfg, "test")
                if ds:
                    for r in ds:
                        tok += count_tokens([str(r.get("sentence1", "")),
                                             str(r.get("sentence2", ""))])
                    n += len(ds)
            rows.append({
                "task": name, "category": cat,
                "test_tokens": tok, "train_tokens": 0,
                "query_tokens": 0, "corpus_tokens": 0,
                "total_tokens": tok, "test_n": n, "train_n": 0,
            })
            print(f"test={tok:>8,}")

        elif ttype == "clust":
            tok = 0
            n = 0
            ds = load_ds(path, None, "test")
            if ds:
                for r in ds:
                    sents = r.get("sentences", [])
                    if isinstance(sents, list):
                        tok += count_tokens(sents)
                        n += len(sents)
            rows.append({
                "task": name, "category": "Clustering",
                "test_tokens": tok, "train_tokens": 0,
                "query_tokens": 0, "corpus_tokens": 0,
                "total_tokens": tok, "test_n": n, "train_n": 0,
            })
            print(f"test={tok:>8,}")

        elif ttype == "ret":
            q_tok = 0
            c_tok = 0
            q_n = 0
            c_n = 0
            qds = load_ds(path, "queries", "queries")
            if qds:
                q_tok = count_tokens([r.get("text", "") for r in qds])
                q_n = len(qds)
            cds = load_ds(path, "corpus", "corpus")
            if cds:
                texts = []
                for r in cds:
                    title = str(r.get("title", "") or "")
                    text = str(r.get("text", "") or "")
                    texts.append((title + " " + text).strip())
                c_tok = count_tokens(texts)
                c_n = len(cds)
            rows.append({
                "task": name, "category": "Retrieval",
                "test_tokens": 0, "train_tokens": 0,
                "query_tokens": q_tok, "corpus_tokens": c_tok,
                "total_tokens": q_tok + c_tok,
                "test_n": q_n, "train_n": c_n,
            })
            print(f"queries={q_tok:>8,}  corpus={c_tok:>8,}")

    # Summary
    total = sum(r["total_tokens"] for r in rows)
    print(f"\n  TOTAL: {total:,} tokens ({total/1e6:.1f}M)")

    # Write CSV
    if args.output:
        outpath = Path(args.output)
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "task", "category", "test_tokens", "train_tokens",
                "query_tokens", "corpus_tokens", "total_tokens",
                "test_n", "train_n",
            ])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
