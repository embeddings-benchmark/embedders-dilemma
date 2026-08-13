#!/usr/bin/env python3
"""Count benchmark tokens using each embedding model's OWN tokenizer.

Stage 1: Extract all text from the 38 llm-eval datasets
Stage 2: Tokenize with each of the 26 models' tokenizers
Stage 3: Compute cost = model_tokens / 1M * model_cost_per_mtok

Usage:
    conda activate mteb
    python scripts/count_tokens_per_model.py
"""
import csv, json, sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

ROOT = Path(__file__).parent.parent

# Stage 1: Extract all benchmark text (tokenizer-independent)

TASKS = [
    # Classification: text column
    {"name": "LLMImdbClassification",            "path": "mteb/llm-eval-imdb",               "rev": "c7cd15a51954e6862a5d29508c8d8db61cb8f1e8", "type": "cls", "configs": [None]},
    {"name": "LLMBanking77Classification",       "path": "mteb/llm-eval-banking77",           "rev": "2c82d499a4aab26ba0d98a4d37a8c838871d1bb1", "type": "cls", "configs": [None]},
    {"name": "LLMAmazonCounterfactualClassification", "path": "mteb/llm-eval-amazon_counterfactual", "rev": "8df4b672d55146368ad9d82a498ac0f16b8f177f", "type": "cls", "configs": ["en", "de", "ja"]},
    {"name": "LLMMTOPDomainClassification",      "path": "mteb/llm-eval-mtop_domain",         "rev": "14315a9fd4305bf99cf0e400e06fb54a9b815f9c", "type": "cls", "configs": ["en", "de", "fr"]},
    {"name": "LLMMassiveIntentClassification",   "path": "mteb/llm-eval-massive_intent",      "rev": "69fe18bc73414a7fc85905fda463a60da8ac10ce", "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "LLMMassiveScenarioClassification", "path": "mteb/llm-eval-massive_scenario",    "rev": "ba1521289b080d967f820b289dd285f22fb968a8", "type": "cls", "configs": ["en", "de", "fr", "ja"]},
    {"name": "LLMToxicConversationsClassification", "path": "mteb/llm-eval-toxic_conversations", "rev": "34eeb6105ca217433c04b207ea66810a2ff42625", "type": "cls", "configs": [None]},
    {"name": "LLMTweetSentimentExtractionClassification", "path": "mteb/llm-eval-tweet_sentiment", "rev": "5e847f1f41ec9089cfdb7ea7b2852a23bd5ea7c8", "type": "cls", "configs": [None]},
    # STS: sentence1 + sentence2
    {"name": "LLMSTSBenchmark", "path": "mteb/llm-eval-stsbenchmark", "rev": "86bbaf4470f501ee381411836b3a22f112bfe42a", "type": "sts", "configs": [None]},
    {"name": "LLMSICKR",        "path": "mteb/llm-eval-sickr",        "rev": "82eb9939fa177ddd94d5ddf4668e011d7446c1da7", "type": "sts", "configs": [None]},
    {"name": "LLMSTS12",        "path": "mteb/llm-eval-sts12",        "rev": "1559a6e5259e61f06ab1f4ef1c71a921e4877dd1", "type": "sts", "configs": [None]},
    {"name": "LLMSTS13",        "path": "mteb/llm-eval-sts13",        "rev": "e0251ba4e151f033f74bd5543c5d32706e810eab", "type": "sts", "configs": [None]},
    {"name": "LLMSTS14",        "path": "mteb/llm-eval-sts14",        "rev": "6237db5b2c4be5e7d4aacad8e850ba8d550093d6", "type": "sts", "configs": [None]},
    {"name": "LLMSTS15",        "path": "mteb/llm-eval-sts15",        "rev": "6283e03ad9bd28cb1ed3d3a472d08a0e9df58fc1", "type": "sts", "configs": [None]},
    {"name": "LLMSTS16",        "path": "mteb/llm-eval-sts16",        "rev": "69862fce58ef5e5e9fdfa7c8b321227cc63961ea", "type": "sts", "configs": [None]},
    {"name": "LLMBIOSSES",      "path": "mteb/llm-eval-biosses",      "rev": "cf968edb41fa17a96392f6b373819efac1c2d6d6", "type": "sts", "configs": [None]},
    {"name": "LLMSTS17",        "path": "mteb/llm-eval-sts17",        "rev": "fe4f4e1b9fdafeae22df69e66bdc3b634be30e9d", "type": "sts", "configs": ["en-en", "en-de", "es-es", "fr-en"]},
    {"name": "LLMSTS22v2",      "path": "mteb/llm-eval-sts22_v2",     "rev": "7a79fd41b024091522d04e84d8d9dc93d223cf8c", "type": "sts", "configs": ["en", "de", "es", "fr", "ru", "zh"]},
    # Clustering: all sentences
    {"name": "LLMRedditClusteringP2P",         "path": "mteb/llm-eval-reddit_clustering_p2p",         "rev": "298518f04f9f2a5849282dd20e43672172428790", "type": "clust", "configs": [None]},
    {"name": "LLMBigPatentClustering",         "path": "mteb/llm-eval-big_patent_clustering",         "rev": "fbb01d02cb8308089d99223cbcc35c584a67ba77", "type": "clust", "configs": [None]},
    {"name": "LLMTwentyNewsgroupsClusteringV2","path": "mteb/llm-eval-twenty_newsgroups_v2",          "rev": "0187d654bb254527a7050c608ced9967dc04db91", "type": "clust", "configs": [None]},
    {"name": "LLMStackExchangeClusteringP2PV2","path": "mteb/llm-eval-stackexchange_clustering_p2p_v2","rev": "5de039af34921939fb4a1c8b2e4f9f0a6ed6cd50", "type": "clust", "configs": [None]},
    {"name": "LLMStackExchangeClusteringV2",   "path": "mteb/llm-eval-stackexchange_clustering_v2",   "rev": "2cbee8fd7c715e92a99639d46ee940e4bf64e5d2", "type": "clust", "configs": [None]},
    {"name": "LLMArxivClusteringP2P",          "path": "mteb/llm-eval-arxiv_clustering_p2p",          "rev": "3ce50711e47553e7c9f47b53819ccd64cd0f5b9b", "type": "clust", "configs": [None]},
    {"name": "LLMArxivClusteringS2S",          "path": "mteb/llm-eval-arxiv_clustering_s2s",          "rev": "28c899f0bb82d9001804d0f408d66932effc0c1c", "type": "clust", "configs": [None]},
    {"name": "LLMBiorxivClusteringP2PV2",      "path": "mteb/llm-eval-biorxiv_clustering_p2p_v2",     "rev": "9e11c95384ef78952ba754f9d8942084ddbb61a7", "type": "clust", "configs": [None]},
    {"name": "LLMMedrxivClusteringP2PV2",      "path": "mteb/llm-eval-medrxiv_clustering_p2p_v2",     "rev": "63c8e6cfbcab3f986291141799fe646c60bb441c", "type": "clust", "configs": [None]},
    {"name": "LLMMedrxivClusteringS2SV2",      "path": "mteb/llm-eval-medrxiv_clustering_s2s_v2",     "rev": "c565eea82f4a8728b8fe5181388b407d305f2647", "type": "clust", "configs": [None]},
    # Pair Classification: sentence1 + sentence2
    {"name": "LLMSprintDuplicateQuestionsPC", "path": "mteb/llm-eval-sprint_duplicate_questions", "rev": "d1c6be04a5f84b606b758024ed9d3b3fb7e4029a", "type": "pair", "configs": [None]},
    {"name": "LLMTwitterURLCorpusPC",         "path": "mteb/llm-eval-twitter_url_corpus",         "rev": "741fbe5cf43a594d8c37073d66baf8ab879281ce", "type": "pair", "configs": [None]},
    {"name": "LLMLegalBenchPC",               "path": "mteb/llm-eval-legal_bench_pc",              "rev": "a0217dc60ec5a45077e213a9538e845533523ed0", "type": "pair", "configs": [None]},
    {"name": "LLMRTE3PC",                     "path": "mteb/llm-eval-rte3",                        "rev": "5d745bd9e435cc190599f916f54c0ab197ddeb73", "type": "pair", "configs": ["de", "en", "fr", "it"]},
    # Retrieval: queries + corpus
    {"name": "LLMTempReasonL1",                "path": "mteb/llm-eval-tempreason-l1",              "rev": "1c61e95df3d70835785db4d70f17d3a16a7bd4ce", "type": "ret"},
    {"name": "LLMLegalBenchCorporateLobbying", "path": "mteb/llm-eval-legalbench-corporate-lobbying","rev": "300cb175d1b608cfc3fdb94fd06b3ceb937f9b6a", "type": "ret"},
    {"name": "LLMAILAStatutes",                "path": "mteb/llm-eval-aila-statutes",              "rev": "a2acf12d293ea934823ca26752a05c5bfab24dff", "type": "ret"},
    {"name": "LLMSpartQA",                     "path": "mteb/llm-eval-spartqa",                    "rev": "d6628df131c3279007b88fe3e057e809dac056af", "type": "ret"},
    {"name": "LLMWinoGrande",                  "path": "mteb/llm-eval-winogrande",                 "rev": "bc2a1e4c24771a6f3432578016cfb737732398fd", "type": "ret"},
    {"name": "LLMTwitterHjerneRetrieval",      "path": "mteb/llm-eval-twitter-hjerne",             "rev": "31f9b918c30ef94e15a168cb95ca4ebf291396eb", "type": "ret"},
]


def load_ds(path, config, rev, split="test"):
    try:
        if config:
            return load_dataset(path, config, split=split, revision=rev)
        return load_dataset(path, split=split, revision=rev)
    except:
        try:
            if config:
                return load_dataset(path, config, split=split)
            return load_dataset(path, split=split)
        except:
            return None


def extract_all_texts():
    """Extract all benchmark texts as a flat list of strings."""
    texts = []
    for task in TASKS:
        path, rev, ttype = task["path"], task["rev"], task["type"]
        configs = task.get("configs", [None])

        if ttype in ("cls",):
            for cfg in configs:
                ds = load_ds(path, cfg, rev)
                if ds:
                    for row in ds:
                        texts.append(str(row.get("text", "")))

        elif ttype in ("sts", "pair"):
            for cfg in configs:
                ds = load_ds(path, cfg, rev)
                if ds:
                    for row in ds:
                        texts.append(str(row.get("sentence1", "")))
                        texts.append(str(row.get("sentence2", "")))

        elif ttype == "clust":
            ds = load_ds(path, None, rev)
            if ds:
                for row in ds:
                    sents = row.get("sentences", [])
                    if isinstance(sents, list):
                        for s in sents:
                            texts.append(str(s))

        elif ttype == "ret":
            qds = load_ds(path, "queries", rev, split="queries")
            if qds:
                for row in qds:
                    texts.append(str(row.get("text", "")))
            cds = load_ds(path, "corpus", rev, split="corpus")
            if cds:
                for row in cds:
                    title = str(row.get("title", "") or "")
                    text = str(row.get("text", "") or "")
                    texts.append((title + " " + text).strip())

    return texts


def count_tokens_batch(tokenizer, texts, batch_size=1000):
    """Count total tokens efficiently in batches."""
    total = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False,
                            padding=False, return_attention_mask=False)
        total += sum(len(ids) for ids in encoded["input_ids"])
    return total


def main():
    # Stage 1: Extract all text
    print("Stage 1: Extracting all benchmark text from 38 llm-eval datasets...")
    texts = extract_all_texts()
    total_chars = sum(len(t) for t in texts)
    print(f"  {len(texts):,} texts, {total_chars:,} chars ({total_chars/1e6:.1f}M)")

    # Stage 2: Tokenize with each model's tokenizer
    print("\nStage 2: Counting tokens with each model's own tokenizer...")
    tp = pd.read_csv(ROOT / "embedding_throughput.csv")

    results = []
    for _, r in tp.iterrows():
        model_id = r["model_id"]
        cpm = r["cost_usd_per_mtok"]
        short = model_id.split("/")[-1][:35]

        sys.stdout.write(f"  {short:37s} ... ")
        sys.stdout.flush()

        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            n_tokens = count_tokens_batch(tok, texts)
            cost = (n_tokens / 1e6) * cpm
            print(f"{n_tokens:>12,d} tok  ${cost:.4f}")
            results.append({
                "model_id": model_id,
                "model_short": short,
                "tokens": n_tokens,
                "cost_per_mtok": cpm,
                "total_cost": cost,
            })
        except Exception as e:
            err = str(e).split('\n')[0][:80]
            print(f"FAILED: {err}")
            results.append({
                "model_id": model_id,
                "model_short": short,
                "tokens": 0,
                "cost_per_mtok": cpm,
                "total_cost": 0,
            })
            tok = None

        # Free memory
        if tok is not None:
            del tok
        import gc; gc.collect()

    # Stage 3: Summary
    print("\n" + "=" * 80)
    df = pd.DataFrame(results)
    ok = df[df.tokens > 0]
    print(f"Successfully tokenized: {len(ok)}/{len(df)} models")
    print(f"Token range: {ok.tokens.min():,} - {ok.tokens.max():,}")
    print(f"  Min tokens: {ok.loc[ok.tokens.idxmin(), 'model_short']}")
    print(f"  Max tokens: {ok.loc[ok.tokens.idxmax(), 'model_short']}")
    print(f"  Mean: {ok.tokens.mean():,.0f}, Std: {ok.tokens.std():,.0f}")
    print(f"  Variation: {ok.tokens.std()/ok.tokens.mean()*100:.1f}%")
    print(f"\nCost range: ${ok.total_cost.min():.4f} - ${ok.total_cost.max():.4f}")
    print("=" * 80)

    # Save
    out = ROOT / "embedding_costs_per_model.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
