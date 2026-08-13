#!/usr/bin/env python3
"""Recount benchmark tokens per embedding model on the CANONICAL 37-task set.

Same extraction/costing methodology as scripts/count_tokens_per_model.py, but the
TASK list is the current 37-task MTEB(LLM) suite (no BigPatent / TempReason /
SpartQA / WinoGrande / LegalBenchCorporateLobbying; adds the 6-task retrieval
suite). Writes data/embedding_costs_per_model.csv.

Extraction (matches the original methodology exactly):
  cls  -> test-split "text"
  sts  -> test-split sentence1 + sentence2
  pair -> test-split sentence1 + sentence2
  clust-> "sentences" list
  ret  -> queries-config text + corpus-config (title + text)

Cost = tokens/1e6 * cost_usd_per_mtok (from data/embedding_throughput.csv).
"""
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

# Canonical 37-task list (paths from scripts/count_tokens_per_task.py; latest revs).
TASKS = [
    # Classification (8): test "text"
    {"name": "ImdbClassification",                "path": "mteb/llm-eval-imdb",                "type": "cls", "configs": [None]},
    {"name": "Banking77Classification",           "path": "mteb/llm-eval-banking77",           "type": "cls", "configs": [None]},
    {"name": "AmazonCounterfactualClassification","path": "mteb/llm-eval-amazon_counterfactual","type": "cls", "configs": ["en", "de", "ja"]},
    {"name": "MTOPDomainClassification",          "path": "mteb/llm-eval-mtop_domain",         "type": "cls", "configs": ["en", "de", "fr"]},
    {"name": "MassiveIntentClassification",       "path": "mteb/llm-eval-massive_intent",      "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "MassiveScenarioClassification",     "path": "mteb/llm-eval-massive_scenario",    "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "ToxicConversationsClassification",  "path": "mteb/llm-eval-toxic_conversations", "type": "cls", "configs": [None]},
    {"name": "TweetSentimentExtractionClassification", "path": "mteb/llm-eval-tweet_sentiment","type": "cls", "configs": [None]},
    # STS (10): sentence1 + sentence2
    {"name": "STSBenchmark", "path": "mteb/llm-eval-stsbenchmark", "type": "sts", "configs": [None]},
    {"name": "SICKR",        "path": "mteb/llm-eval-sickr",        "type": "sts", "configs": [None]},
    {"name": "STS12",        "path": "mteb/llm-eval-sts12",        "type": "sts", "configs": [None]},
    {"name": "STS13",        "path": "mteb/llm-eval-sts13",        "type": "sts", "configs": [None]},
    {"name": "STS14",        "path": "mteb/llm-eval-sts14",        "type": "sts", "configs": [None]},
    {"name": "STS15",        "path": "mteb/llm-eval-sts15",        "type": "sts", "configs": [None]},
    {"name": "STS16",        "path": "mteb/llm-eval-sts16",        "type": "sts", "configs": [None]},
    {"name": "BIOSSES",      "path": "mteb/llm-eval-biosses",      "type": "sts", "configs": [None]},
    {"name": "STS17",        "path": "mteb/llm-eval-sts17",        "type": "sts", "configs": ["en-en", "en-de", "es-es", "fr-en"]},
    {"name": "STS22v2",      "path": "mteb/llm-eval-sts22_v2",     "type": "sts", "configs": ["en", "de", "es", "fr", "ru", "zh"]},
    # Clustering (9): "sentences" list
    {"name": "RedditClusteringP2P",          "path": "mteb/llm-eval-reddit_clustering_p2p",          "type": "clust", "configs": [None]},
    {"name": "TwentyNewsgroupsClusteringV2", "path": "mteb/llm-eval-twenty_newsgroups_v2",           "type": "clust", "configs": [None]},
    {"name": "StackExchangeClusteringP2PV2", "path": "mteb/llm-eval-stackexchange_clustering_p2p_v2","type": "clust", "configs": [None]},
    {"name": "StackExchangeClusteringV2",    "path": "mteb/llm-eval-stackexchange_clustering_v2",    "type": "clust", "configs": [None]},
    {"name": "ArxivClusteringP2P",           "path": "mteb/llm-eval-arxiv_clustering_p2p",           "type": "clust", "configs": [None]},
    {"name": "ArxivClusteringS2S",           "path": "mteb/llm-eval-arxiv_clustering_s2s",           "type": "clust", "configs": [None]},
    {"name": "BiorxivClusteringP2PV2",       "path": "mteb/llm-eval-biorxiv_clustering_p2p_v2",      "type": "clust", "configs": [None]},
    {"name": "MedrxivClusteringP2PV2",       "path": "mteb/llm-eval-medrxiv_clustering_p2p_v2",      "type": "clust", "configs": [None]},
    {"name": "MedrxivClusteringS2SV2",       "path": "mteb/llm-eval-medrxiv_clustering_s2s_v2",      "type": "clust", "configs": [None]},
    # Pair Classification (4): sentence1 + sentence2
    {"name": "SprintDuplicateQuestionsPC", "path": "mteb/llm-eval-sprint_duplicate_questions", "type": "pair", "configs": [None]},
    {"name": "TwitterURLCorpusPC",         "path": "mteb/llm-eval-twitter_url_corpus",         "type": "pair", "configs": [None]},
    {"name": "LegalBenchPC",               "path": "mteb/llm-eval-legal_bench_pc",              "type": "pair", "configs": [None]},
    {"name": "RTE3PC",                     "path": "mteb/llm-eval-rte3",                        "type": "pair", "configs": ["de", "en", "fr", "it"]},
    # Retrieval (6): queries + corpus
    {"name": "AILAStatutes",                 "path": "mteb/llm-eval-aila-statutes",                 "type": "ret"},
    {"name": "FQuADRetrieval",               "path": "mteb/llm-eval-fquad",                         "type": "ret"},
    {"name": "HC3FinanceRetrieval",          "path": "mteb/llm-eval-hc3-finance",                   "type": "ret"},
    {"name": "LegalBenchConsumerContractsQA","path": "mteb/llm-eval-legalbench-consumer-contracts", "type": "ret"},
    {"name": "PublicHealthQA",               "path": "mteb/llm-eval-public-health-qa",              "type": "ret"},
    {"name": "TwitterHjerneRetrieval",       "path": "mteb/llm-eval-twitter-hjerne",                "type": "ret"},
]


def load_ds(path, config=None, split="test"):
    try:
        if config:
            return load_dataset(path, config, split=split)
        return load_dataset(path, split=split)
    except Exception as e:
        print(f"    [load fail {path}/{config}/{split}: {str(e)[:60]}]")
        return None


def extract_all_texts():
    texts = []
    per_task = {}
    for task in TASKS:
        path, ttype = task["path"], task["type"]
        configs = task.get("configs", [None])
        before = len(texts)
        if ttype == "cls":
            for cfg in configs:
                ds = load_ds(path, cfg)
                if ds:
                    texts.extend(str(r.get("text", "")) for r in ds)
        elif ttype in ("sts", "pair"):
            for cfg in configs:
                ds = load_ds(path, cfg)
                if ds:
                    for r in ds:
                        texts.append(str(r.get("sentence1", "")))
                        texts.append(str(r.get("sentence2", "")))
        elif ttype == "clust":
            ds = load_ds(path, None)
            if ds:
                for r in ds:
                    sents = r.get("sentences", [])
                    if isinstance(sents, list):
                        texts.extend(str(s) for s in sents)
        elif ttype == "ret":
            qds = load_ds(path, "queries", split="queries")
            if qds:
                texts.extend(str(r.get("text", "")) for r in qds)
            cds = load_ds(path, "corpus", split="corpus")
            if cds:
                for r in cds:
                    title = str(r.get("title", "") or "")
                    text = str(r.get("text", "") or "")
                    texts.append((title + " " + text).strip())
        per_task[task["name"]] = len(texts) - before
    return texts, per_task


def count_tokens_batch(tokenizer, texts, batch_size=1000):
    total = 0
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=False,
                        truncation=False, padding=False, return_attention_mask=False)
        total += sum(len(ids) for ids in enc["input_ids"])
    return total


def main():
    print("Stage 1: extracting text from the 37 MTEB(LLM) datasets ...")
    texts, per_task = extract_all_texts()
    print(f"  {len(texts):,} texts, {sum(len(t) for t in texts):,} chars")
    for k, v in per_task.items():
        print(f"    {k:42s} {v:>8,} texts")

    print("\nStage 2: tokenizing with each model's own tokenizer ...")
    tp = pd.read_csv(ROOT / "data" / "embedding_throughput.csv")
    results = []
    for _, r in tp.iterrows():
        model_id = r["model_id"]
        cpm = r["cost_usd_per_mtok"]
        short = model_id.split("/")[-1][:35]
        sys.stdout.write(f"  {short:37s} ... "); sys.stdout.flush()
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            n = count_tokens_batch(tok, texts)
            cost = (n / 1e6) * cpm
            print(f"{n:>12,d} tok  ${cost:.4f}")
            results.append({"model_id": model_id, "model_short": short,
                            "tokens": n, "cost_per_mtok": cpm, "total_cost": cost})
            del tok
        except Exception as e:
            print(f"FAILED: {str(e).splitlines()[0][:70]}")
            results.append({"model_id": model_id, "model_short": short,
                            "tokens": 0, "cost_per_mtok": cpm, "total_cost": 0})

    df = pd.DataFrame(results)
    ok = df[df.tokens > 0]
    print("\n" + "=" * 72)
    print(f"tokenized {len(ok)}/{len(df)} models")
    print(f"token range {ok.tokens.min():,} - {ok.tokens.max():,}  "
          f"(median {ok.tokens.median():,.0f})")
    print(f"cost range ${ok.total_cost.min():.4f} - ${ok.total_cost.max():.4f}")
    print("=" * 72)
    out = ROOT / "data" / "embedding_costs_per_model.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
