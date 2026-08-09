#!/usr/bin/env python3
"""Run the BRIGHT retrieval pipeline via native MTEB.

Three stages, each a separate `mteb.evaluate()` call:

  1. Bi-encoder full-corpus retrieval (writes top-K candidates as predictions JSON).
  2. Cross-encoder rerank of those candidates.
  3. LLM listwise rerank of those candidates.

All three stages share the same MTEB task object — only the model and the
`top_ranked` view of the task change between stages. Each stage's results are
saved by MTEB's standard result cache (under `pipeline_results/<model>/...`) so
`scripts/aggregate_scores.py` picks them up unchanged.

Usage:

  Stage-1 + 2 + 3 (full):
    uv run python scripts/experiments/run_pipeline.py \\
        --bi-encoder BAAI/bge-base-en-v1.5 \\
        --cross-encoder BAAI/bge-reranker-base \\
        --llm-rerank \\
        --top-k 100

  Smoke test (one small subset):
    uv run python scripts/experiments/run_pipeline.py \\
        --bi-encoder sentence-transformers/all-MiniLM-L6-v2 \\
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --llm-rerank \\
        --tasks BRIGHTEconomics \\
        --top-k 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `llm_judge` importable when running this script directly via `uv run python`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mteb
import torch
from sentence_transformers import CrossEncoder

# mteb's confidence_scores crashes (IndexError) when sim_scores is empty — happens
# when a query had no candidates from stage 1 (e.g. a BRIGHT query whose gold docs
# weren't in the bi-encoder's top-K). Patch it to return zeros instead.
import mteb._evaluators.retrieval_metrics as _rm
_orig_confidence_scores = _rm.confidence_scores
def _safe_confidence_scores(sim_scores):
    if not sim_scores:
        return {"max": 0.0, "std": 0.0, "diff1": 0.0}
    return _orig_confidence_scores(sim_scores)
_rm.confidence_scores = _safe_confidence_scores

from llm_judge.llm_reranker import LLMListwiseReranker
from llm_judge.rerankers import GENERATIVE_CE_PRESETS, GenerativeCrossEncoder
from llm_judge.settings import Settings
from llm_judge.tasks.bright import BRIGHT_STEM7

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _instantiate_task(name: str, max_queries: int | None = None):
    """Build a fresh task instance. Supports BRIGHT (custom) + standard mteb tasks.

    For BRIGHT-* names: instantiates our custom class and subsamples queries
    (preserving _excluded_ids). For other names: falls back to mteb.get_task()
    without subsampling (let the standard task use its native splits).

    A fresh instance per stage avoids leaking `top_ranked` / `_top_k` state across
    pipeline runs.
    """
    by_name = {cls.__name__: cls for cls in BRIGHT_STEM7}
    is_bright = name in by_name
    task = by_name[name]() if is_bright else mteb.get_task(task_name=name)

    if max_queries is not None:
        task.load_data()
        # Both BRIGHT (custom) and standard mteb retrieval tasks expose
        # task.dataset["default"]["test"] as a dict with queries/relevant_docs.
        ds = task.dataset["default"]["test"]
        queries = ds["queries"].select(range(min(max_queries, len(ds["queries"]))))
        keep_qids = set(queries["id"])
        ds["queries"] = queries
        ds["relevant_docs"] = {q: r for q, r in ds["relevant_docs"].items() if q in keep_qids}
        # Corpus stays full so retrieval over the whole document set is preserved.
        if is_bright:
            task._excluded_ids = {q: e for q, e in task._excluded_ids.items() if q in keep_qids}
    return task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--first-stage",
        "--bi-encoder",  # backwards-compat alias
        dest="first_stage",
        required=True,
        help=(
            "First-stage retriever model id. Loaded via mteb.get_model() so the "
            "same flag works for dense bi-encoders (e.g. BAAI/bge-base-en-v1.5), "
            "sparse (bm25s, naver/splade-*), or late-interaction (colbert-ir/colbertv2.0)."
        ),
    )
    p.add_argument(
        "--cross-encoder",
        default=None,
        help="HF model id for the cross-encoder reranker. Omit to skip stage 2.",
    )
    p.add_argument(
        "--llm-rerank",
        action="store_true",
        help="Run stage 3: LLM listwise rerank using settings.model.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task class names to run. Default: all BRIGHT STEM-7 tasks.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of stage-1 candidates to pass to rerank stages.",
    )
    p.add_argument(
        "--results-root",
        default="pipeline_results",
        help="Root folder for MTEB result cache + predictions.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encode batch size for the bi-encoder.",
    )
    p.add_argument(
        "--ce-batch-size",
        type=int,
        default=None,
        help="Override batch size for the CE reranker (stage 2). Falls back to --batch-size if unset. Useful when CE model is much larger than bi-encoder (e.g. mxbai-rerank-large-v2 at 2B OOMs at batch=32).",
    )
    p.add_argument(
        "--ce-max-length",
        type=int,
        default=None,
        help="Override max input length for CE reranker. Required for LLM-based CEs (mxbai-rerank-v2, Qwen3-Reranker) whose default 8K-32K context blows up attention on BRIGHT's long docs. BERT-CEs are already 512-capped natively.",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap queries per task for fast smoke testing. Corpus stays full.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override bi-encoder max_seq_length (e.g. 512 to cap long-context models for BRIGHT-paper-aligned protocol).",
    )
    p.add_argument(
        "--attn-impl",
        default=None,
        choices=(None, "flash_attention_2", "sdpa", "eager"),
        help="HF attn_implementation for bi-encoder (e.g. flash_attention_2 to use installed flash-attn).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    task_names = args.tasks or [cls.__name__ for cls in BRIGHT_STEM7]
    device = _device()
    logger.info("Device: %s", device)

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    bi_slug = _slug(args.first_stage)
    # Predictions from stage 1 are shared across stages 2 and 3 for the same first-stage.
    predictions_dir = root / "predictions" / bi_slug
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: first-stage retriever (dense/sparse/late-interaction) ----
    logger.info("=== Stage 1: first-stage = %s ===", args.first_stage)
    # mteb.get_model() handles all paradigms uniformly: SentenceTransformer for
    # dense, BM25 for sparse (needs `mteb[bm25s]`), pylate MultiVectorModel for
    # ColBERT (needs `mteb[pylate]`). All return objects satisfying the right
    # protocol so mteb.evaluate() dispatches correctly with no further code.
    # max_seq_length flows through mteb.get_model() → loader → wrapper (e.g.
    # InstructSentenceTransformerModel.__init__ accepts it natively).
    bi_kwargs = {}
    if args.max_seq_length is not None:
        bi_kwargs["max_seq_length"] = args.max_seq_length
        logger.info("Setting bi-encoder max_seq_length=%d via loader kwarg", args.max_seq_length)
    if args.attn_impl is not None:
        # flash_attention_2 requires fp16/bf16 weights — set dtype too.
        # `dtype` is the modern transformers API (≥4.50); was previously `torch_dtype`.
        bi_kwargs["model_kwargs"] = {
            "attn_implementation": args.attn_impl,
            "dtype": "bfloat16",
        }
        logger.info("Setting bi-encoder attn_implementation=%s, dtype=bfloat16", args.attn_impl)
    if args.first_stage == "bm25s":
        # Standard BM25 protocol (Robertson & Zaragoza 2009; BEIR — Thakur et al. 2021;
        # BRIGHT — Su et al. 2024): default k1=1.5, b=0.75, with NO stopword removal
        # and NO stemming. mteb's bm25s loader defaults to English stopwords + Porter
        # stemmer, which silently destroys BM25 on reasoning-heavy corpora like BRIGHT
        # (Economics nDCG@10: 0.0 with defaults vs 11.1 with this override; paper 14.9).
        bi_kwargs["stopwords"] = None
        bi_kwargs["stemmer_language"] = None
        logger.info("BM25: applying standard-protocol (no stopwords, no stemming)")
    bi_model = mteb.get_model(args.first_stage, **bi_kwargs)
    bi_cache_dir = root / "cache" / bi_slug
    bi_cache_dir.mkdir(parents=True, exist_ok=True)
    bi_cache = mteb.cache.ResultCache(bi_cache_dir)

    for task_name in task_names:
        task = _instantiate_task(task_name, max_queries=args.max_queries)
        mteb.evaluate(
            model=bi_model,
            tasks=task,
            cache=bi_cache,
            prediction_folder=predictions_dir,
            encode_kwargs={"batch_size": args.batch_size},
        )

    # Stage-1 model occupies GPU memory; release before loading the cross-encoder.
    del bi_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Stage 2: cross-encoder ----
    if args.cross_encoder:
        logger.info("=== Stage 2: cross-encoder = %s ===", args.cross_encoder)
        if args.cross_encoder in GENERATIVE_CE_PRESETS:
            # Decoder-LLM reranker (e.g. mxbai-rerank-v2, Qwen3-Reranker).
            # sentence_transformers.CrossEncoder loads these incorrectly
            # (random score head), so use our chat-template + logit-extraction wrapper.
            gce_kwargs = {"device": device}
            if args.ce_max_length is not None:
                gce_kwargs["max_length"] = args.ce_max_length
            logger.info("Using GenerativeCrossEncoder for %s", args.cross_encoder)
            ce_model = GenerativeCrossEncoder(args.cross_encoder, **gce_kwargs)
        else:
            ce_init_kwargs = {"device": device, "trust_remote_code": True}
            if args.ce_max_length is not None:
                ce_init_kwargs["max_length"] = args.ce_max_length
                logger.info("Setting CE max_length=%d", args.ce_max_length)
            ce_model = CrossEncoder(args.cross_encoder, **ce_init_kwargs)
        ce_slug = _slug(args.cross_encoder)
        ce_cache_dir = root / "cache" / f"{bi_slug}__{ce_slug}"
        ce_cache_dir.mkdir(parents=True, exist_ok=True)
        ce_cache = mteb.cache.ResultCache(ce_cache_dir)

        for task_name in task_names:
            task = _instantiate_task(task_name, max_queries=args.max_queries)
            pred_path = predictions_dir / task.prediction_file_name
            if not pred_path.exists():
                logger.warning(
                    "No stage-1 predictions for %s at %s — skipping CE rerank.",
                    task_name,
                    pred_path,
                )
                continue
            task.convert_to_reranking(pred_path, top_k=args.top_k)
            ce_batch = args.ce_batch_size if args.ce_batch_size is not None else args.batch_size
            mteb.evaluate(
                model=ce_model,
                tasks=task,
                cache=ce_cache,
                encode_kwargs={"batch_size": ce_batch},
            )

        del ce_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Stage 3: LLM listwise rerank ----
    if args.llm_rerank:
        settings = Settings()
        logger.info("=== Stage 3: LLM listwise rerank = %s ===", settings.model)
        llm_model = LLMListwiseReranker(model_name=settings.model)
        llm_slug = _slug(settings.model)
        llm_cache_dir = root / "cache" / f"{bi_slug}__llm-{llm_slug}"
        llm_cache_dir.mkdir(parents=True, exist_ok=True)
        llm_cache = mteb.cache.ResultCache(llm_cache_dir)

        for task_name in task_names:
            task = _instantiate_task(task_name, max_queries=args.max_queries)
            pred_path = predictions_dir / task.prediction_file_name
            if not pred_path.exists():
                logger.warning(
                    "No stage-1 predictions for %s at %s — skipping LLM rerank.",
                    task_name,
                    pred_path,
                )
                continue
            task.convert_to_reranking(pred_path, top_k=args.top_k)
            mteb.evaluate(
                model=llm_model,
                tasks=task,
                cache=llm_cache,
                encode_kwargs={"batch_size": 1},  # unused by LLM, but MTEB requires it
            )
            # Persist token/cost/latency usage for this (first-stage, LLM, task)
            # cell next to the scores so it can go on the paper's cost axis without
            # re-running. mteb's ResultCache doesn't track this, so we write our own.
            if getattr(llm_model, "last_usage", None) is not None:
                usage = dict(llm_model.last_usage)
                usage["first_stage"] = args.first_stage
                usage["task"] = task_name
                usage["top_k"] = args.top_k
                usage_dir = root / "usage" / f"{bi_slug}__llm-{llm_slug}"
                usage_dir.mkdir(parents=True, exist_ok=True)
                (usage_dir / f"{task_name}.json").write_text(json.dumps(usage, indent=2))
                logger.info(
                    "Usage[%s/%s]: in=%d out=%d think=%d total=%d calls=%d failed=%d wall=%.1fs",
                    args.first_stage, task_name,
                    usage["input_tokens"], usage["output_tokens"], usage["thinking_tokens"],
                    usage["total_tokens"], usage["n_llm_calls"], usage["n_failed"], usage["wall_time_s"],
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
