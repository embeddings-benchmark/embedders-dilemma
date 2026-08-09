"""
Empirical throughput benchmark for embedding models on H100 80GB.

Uses mteb.get_model() (same as eval pipeline) to load each model,
finds max batch size, measures actual tokens/sec, converts to $/MTok.

H100 reference: $2.49/hr (Lambda Labs H100 80GB SXM5 spot, March 2026)
"""

import time
import gc
import csv
import torch  # type: ignore[import]
import mteb   # type: ignore[import]

H100_COST_HR = 2.49   # USD/hr

# ~256 tokens of English text across most tokenizers (~1200 chars)
SAMPLE_TEXT = (
    "The field of natural language processing has seen remarkable advances "
    "in recent years, driven by the development of large-scale transformer "
    "models trained on vast corpora of text data. These models have demonstrated "
    "impressive capabilities across a wide range of tasks including text "
    "classification, question answering, summarization, and machine translation. "
    "The ability to represent text as dense vector embeddings has proven "
    "particularly valuable for semantic search, retrieval-augmented generation, "
    "and clustering applications. Modern embedding models capture nuanced "
    "semantic relationships between words and sentences, enabling more accurate "
    "similarity computations than traditional sparse representations like TF-IDF."
)

# (name, params_B, score, api_$/MTok or None)
MODELS = [
    ("tencent/KaLM-Embedding-Gemma3-12B-2511",  11.77, 0.7328, None),
    ("bflhc/Octen-Embedding-8B",                 7.57,  0.7259, None),
    ("Qwen/Qwen3-Embedding-8B",                  7.57,  0.7256, 0.010),  # DeepInfra
    ("Qwen/Qwen3-Embedding-4B",                  4.02,  0.7194, 0.020),  # DeepInfra
    ("jinaai/jina-embeddings-v5-text-small",     0.596, 0.7029, 0.050),  # Jina AI
    ("nvidia/llama-embed-nemotron-8b",           7.50,  0.7023, None),
    ("jinaai/jina-embeddings-v5-text-nano",      0.212, 0.6978, 0.050),  # Jina AI
    ("Salesforce/SFR-Embedding-2_R",             7.11,  0.6922, None),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct",        7.07,  0.6917, None),
    ("codefuse-ai/F2LLM-v2-14B",                13.99,  0.6905, None),
    ("Qwen/Qwen3-Embedding-0.6B",                0.596, 0.6888, 0.010),  # OpenRouter
    ("codefuse-ai/F2LLM-v2-8B",                  7.57,  0.6813, None),
    ("Linq-AI-Research/Linq-Embed-Mistral",       7.11,  0.6780, None),
    ("google/embeddinggemma-300m",               0.308, 0.6766, None),
    ("codefuse-ai/F2LLM-v2-4B",                  4.02,  0.6751, None),
    ("codefuse-ai/F2LLM-v2-1.7B",                1.72,  0.6735, None),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",      1.54,  0.6662, None),
    ("codefuse-ai/F2LLM-v2-0.6B",                0.596, 0.6576, None),
    ("intfloat/multilingual-e5-large-instruct",  0.560, 0.6576, 0.010),  # DeepInfra
    ("BAAI/bge-m3",                              0.568, 0.6125, 0.010),  # DeepInfra
    ("intfloat/multilingual-e5-large",           0.560, 0.6110, 0.010),  # DeepInfra
    ("Snowflake/snowflake-arctic-embed-l-v2.0",  0.568, 0.6063, None),
    ("intfloat/multilingual-e5-base",            0.278, 0.6037, None),
    ("intfloat/multilingual-e5-small",           0.118, 0.5922, None),
    ("GritLM/GritLM-7B",                         7.24,  0.5080, None),
    ("intfloat/e5-mistral-7b-instruct",          7.11,  0.4872, None),
]


def get_token_count(model, text: str) -> int:
    """Count tokens using the model's own tokenizer; fall back to char estimate."""
    try:
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "tokenizer"):
            tokenizer = inner.tokenizer
        else:
            mod = inner._first_module() if hasattr(inner, "_first_module") else None
            tokenizer = mod.tokenizer if (mod and hasattr(mod, "tokenizer")) else None
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=True))
    except Exception:
        pass
    return len(text) // 4  # ~4 chars/token fallback


def find_max_batch(model, texts_pool: list, step: int = 32) -> int:
    """Step up batch size until OOM; return last successful batch size."""
    last_ok = step
    batch = step
    while batch <= len(texts_pool):
        try:
            torch.cuda.empty_cache()
            model.encode(texts_pool[:batch], batch_size=batch, show_progress_bar=False)  # type: ignore[index]
            torch.cuda.synchronize()
            last_ok = batch
            batch += step
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                break
            raise
    return last_ok


def measure_throughput(model, texts: list, tok_per_text: int,
                       n_warmup: int = 5, n_measure: int = 20) -> float:
    """Returns tokens/sec at the given batch size."""
    batch = len(texts)
    for _ in range(n_warmup):
        model.encode(texts, batch_size=batch, show_progress_bar=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_measure):
        model.encode(texts, batch_size=batch, show_progress_bar=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (batch * tok_per_text * n_measure) / elapsed


# Run benchmark
results = []
POOL_SIZE = 4096
texts_pool = [SAMPLE_TEXT] * POOL_SIZE

for name, params_b, score, api_price in MODELS:
    print(f"\n{'─' * 70}\n{name}")
    try:
        model = mteb.get_model(name)
        tok_per_text = get_token_count(model, SAMPLE_TEXT)
        print(f"  tokens/text : {tok_per_text}")

        max_b   = find_max_batch(model, texts_pool)
        bench_b = max(1, int(max_b * 0.8))
        print(f"  max batch   : {max_b}  →  benchmarking at {bench_b}")

        tps  = measure_throughput(model, texts_pool[:bench_b], tok_per_text)
        cost = H100_COST_HR * 1e6 / (tps * 3600)
        print(f"  throughput  : {tps:,.0f} tok/s")
        print(f"  cost        : ${cost:.4f}/MTok")
        if api_price:
            print(f"  API price   : ${api_price:.4f}/MTok  ({api_price / cost:.1f}x measured)")

        results.append({
            "model":              name,
            "params_b":           params_b,
            "score":              score,
            "tokens_per_text":    tok_per_text,
            "max_batch":          max_b,
            "bench_batch":        bench_b,
            "tokens_per_sec":     round(tps),
            "cost_per_mtok":      round(cost, 5),
            "api_price_per_mtok": api_price,
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"model": name, "error": str(e)})

    finally:
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

# Save results
FIELDS = ["model", "params_b", "score", "tokens_per_text",
          "max_batch", "bench_batch", "tokens_per_sec",
          "cost_per_mtok", "api_price_per_mtok"]

with open("data/embedding_throughput.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(results)

print("\nSaved → embedding_throughput.csv")

# Validation table (models with known API prices)
validated = [r for r in results if r.get("api_price_per_mtok") and r.get("cost_per_mtok")]
if validated:
    print(f"\n{'Model':<45} {'Measured':>12}  {'API':>12}  {'Markup':>8}")
    print("─" * 82)
    for r in validated:
        short  = r["model"].split("/")[-1]
        markup = r["api_price_per_mtok"] / r["cost_per_mtok"]
        print(f"  {short:<43} ${r['cost_per_mtok']:>10.4f}  ${r['api_price_per_mtok']:>10.4f}  {markup:>6.1f}x")
