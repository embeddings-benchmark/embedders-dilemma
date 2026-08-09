"""LLM listwise reranker implementing MTEB's `SearchProtocol`.

Plugs into `mteb.evaluate(model=LLMListwiseReranker(...), tasks=[...])` after a
first-stage retriever has produced `top_ranked` candidates via
`task.convert_to_reranking(predictions_path, top_k=N)`.

Uses `SearchProtocol` (not `CrossEncoderProtocol`) so MTEB calls our `search()`
directly with per-query candidates — preserving the listwise grouping that
`SearchCrossEncoderWrapper` would otherwise flatten into pointwise pairs.

Concurrency is bounded by `settings.max_concurrency` via the semaphore inside
`llm_judge.llm_client.send_request`; we just fan out with `asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import (
    CorpusDatasetType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


LISTWISE_INSTRUCTIONS = (
    "You are an expert at evaluating document relevance. Given a query and a list of "
    "candidate documents, rank them by relevance to the query. Output ONLY a JSON list "
    "of integer document IDs in order of decreasing relevance. Example output format: "
    "[12, 3, 47, 1, ...]. Include ALL document IDs from the input."
)


def _make_model_meta(model_name: str) -> ModelMeta:
    """Build a minimal ModelMeta for the LLM reranker.

    MTEB's ModelMeta validator requires `name` to contain a '/'. We prefix with
    'llm-rerank/' when the configured model name doesn't already have one.
    """
    safe_name = model_name if "/" in model_name else f"llm-rerank/{model_name}"
    return ModelMeta(
        loader=None,
        name=safe_name,
        revision=model_name,
        release_date=None,
        languages=None,
        framework=[],
        similarity_fn_name=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        open_weights=None,
        public_training_code=None,
        public_training_data=None,
        use_instructions=None,
        training_datasets=None,
        modalities=["text"],
        is_cross_encoder=True,
    )


class LLMListwiseReranker:
    """SearchProtocol-compatible LLM reranker.

    Requires `top_ranked` (set via `task.convert_to_reranking(...)`). For each
    query, builds a listwise prompt over the candidate docs, parses the LLM's
    ranked JSON list, and emits rank-position-based scores (k - rank_pos).
    """

    def __init__(
        self,
        model_name: str,
        max_doc_chars: int = 800,
    ) -> None:
        self.model_name = model_name
        self.max_doc_chars = max_doc_chars
        self.mteb_model_meta = _make_model_meta(model_name)
        self._corpus_texts: list[str] | None = None
        self._corpus_titles: list[str] | None = None
        self._doc_id_to_idx: dict[str, int] | None = None
        # Per-search usage accounting (tokens/cost/latency). search() overwrites
        # this each call; the runner reads it after each task to write a sidecar.
        self.last_usage: dict | None = None

    # ---- SearchProtocol ----

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
    ) -> None:
        # MTEB's HF Datasets give column access in O(1) without materializing the
        # whole row. We only need text + id; titles are kept in case a corpus has them.
        self._corpus_texts = list(corpus["text"])
        self._corpus_titles = (
            list(corpus["title"]) if "title" in corpus.column_names else None
        )
        ids = list(corpus["id"])
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutputType:
        if top_ranked is None:
            raise ValueError(
                "LLMListwiseReranker requires top_ranked candidates. "
                "Call `task.convert_to_reranking(<stage1_predictions.json>, top_k=...)` first."
            )
        if self._corpus_texts is None or self._doc_id_to_idx is None:
            raise RuntimeError("index() must be called before search().")

        # Build per-query work items. top_ranked is already sorted descending by
        # stage-1 score (convert_to_reranking sorts before slicing top_k).
        work: list[tuple[str, str, list[str], list[str]]] = []
        skipped_missing = 0
        for row in queries:
            qid = row["id"]
            qtext = row["text"]
            cand_ids_full = top_ranked.get(qid, [])
            if not cand_ids_full:
                skipped_missing += 1
                continue
            cand_ids = list(cand_ids_full[:top_k])
            try:
                cand_texts = [
                    self._corpus_texts[self._doc_id_to_idx[d]] for d in cand_ids
                ]
            except KeyError as e:
                # A candidate ID not in corpus is a data bug — surface it loudly.
                raise KeyError(
                    f"Candidate doc {e!r} for query {qid!r} not found in indexed corpus."
                ) from e
            work.append((qid, qtext, cand_ids, cand_texts))

        if skipped_missing:
            logger.warning(
                "LLMListwiseReranker: %d queries had no top_ranked candidates and were skipped.",
                skipped_missing,
            )

        if not work:
            self.last_usage = _empty_usage(n_queries=0)
            return {row["id"]: {} for row in queries}

        _t0 = time.monotonic()
        ranked_per_query = asyncio.run(self._run_all(work))
        wall_s = time.monotonic() - _t0

        # Initialize empty result for every query (including skipped ones), then
        # fill from the LLM's output. MTEB's evaluator tolerates empty result dicts.
        results: RetrievalOutputType = {row["id"]: {} for row in queries}
        agg = _empty_usage(n_queries=len(work))
        for qid, scores, usage in ranked_per_query:
            results[qid] = scores
            u = usage or {}
            for k in ("input_tokens", "cached_tokens", "output_tokens",
                      "thinking_tokens", "total_tokens", "cost"):
                agg[k] += u.get(k, 0) or 0
            failed = u.get("call_failed", False)
            agg["n_failed"] += 1 if failed else 0
            agg["n_llm_calls"] += 0 if failed else 1
            # Rank-coverage audit: surface silent degradation (truncated/malformed
            # output → docs kept stage-1 order via backfill).
            kk = u.get("k", 0) or 0
            pc = u.get("parsed_count", 0) or 0
            if not failed:
                if pc >= kk and kk > 0:
                    agg["n_fully_ranked"] += 1
                elif pc > 0:
                    agg["n_partial_ranked"] += 1
                else:
                    agg["n_empty_ranked"] += 1  # parsed nothing → full silent fallback
            agg["docs_backfilled"] += max(0, kk - pc)
        agg["wall_time_s"] = round(wall_s, 2)
        agg["model"] = self.model_name
        agg["top_k"] = len(work[0][2]) if work else None
        agg["max_doc_chars"] = self.max_doc_chars
        self.last_usage = agg
        return results

    # ---- internals ----

    async def _run_all(
        self, work: list[tuple[str, str, list[str], list[str]]]
    ) -> list[tuple[str, dict[str, float], dict | None]]:
        coros = [self._rerank_one(*item) for item in work]
        return await asyncio.gather(*coros)

    async def _rerank_one(
        self,
        qid: str,
        qtext: str,
        cand_doc_ids: list[str],
        cand_texts: list[str],
    ) -> tuple[str, dict[str, float]]:
        from llm_judge.llm_client import send_request

        k = len(cand_doc_ids)
        blocks: list[str] = []
        for i, txt in enumerate(cand_texts):
            t = txt[: self.max_doc_chars]
            if len(txt) > self.max_doc_chars:
                t += "..."
            blocks.append(f"[{i}] {t}")
        docs_str = "\n\n".join(blocks)
        user_msg = (
            f"Query: {qtext}\n\n"
            f"Candidate documents ({k}):\n{docs_str}\n\n"
            f"Output the JSON list of document IDs ranked by decreasing relevance:"
        )

        usage: dict | None = None
        try:
            response, usage = await send_request(
                instructions=LISTWISE_INSTRUCTIONS,
                input=user_msg,
            )
        except Exception as e:
            logger.warning(
                "LLM rerank failed for query %s: %r — falling back to stage-1 order.",
                qid,
                e,
            )
            response = ""
            usage = None  # signals a failed call to the aggregator

        ranked_local = _parse_ranked_list(response, k)
        # Coverage = how many of the k docs the LLM actually ranked BEFORE we
        # backfill. <k means the output was truncated / malformed / refused and
        # those docs keep stage-1 order — a silent degradation we record (in the
        # usage stats) rather than hide.
        parsed_count = len(ranked_local)
        if usage is None:
            usage = {}
        usage = dict(usage)  # don't mutate the client's dict
        usage["parsed_count"] = parsed_count
        usage["k"] = k
        usage["call_failed"] = response == ""  # exception path
        # Fallback for any missing local IDs: append them in original order.
        if parsed_count < k:
            present = set(ranked_local)
            ranked_local.extend(i for i in range(k) if i not in present)

        # Rank-position score: top-ranked gets `k`, next gets `k-1`, etc.
        scores = {
            cand_doc_ids[local_id]: float(k - rank_pos)
            for rank_pos, local_id in enumerate(ranked_local)
        }
        return qid, scores, usage


def _empty_usage(n_queries: int) -> dict:
    return {
        "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
        "thinking_tokens": 0, "total_tokens": 0, "cost": 0.0,
        "n_queries": n_queries, "n_llm_calls": 0, "n_failed": 0,
        # Rank-coverage audit (detect silent degradation from truncation/parse-fail):
        "n_fully_ranked": 0,    # query got a complete ranking of all k docs
        "n_partial_ranked": 0,  # some docs ranked, rest silently kept stage-1 order
        "n_empty_ranked": 0,    # LLM output unparseable → query fully stage-1 (silent)
        "docs_backfilled": 0,   # total docs across queries that kept stage-1 order
        "wall_time_s": 0.0,
    }


def _parse_ranked_list(response: str, k: int) -> list[int]:
    """Extract a list of valid local IDs in [0, k) from the LLM response.

    Robust to extra prose around the JSON list. Deduplicates, ignores out-of-range
    or non-integer entries.
    """
    if not response:
        return []
    match = re.search(r"\[[\s\S]*?\]", response)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    seen: set[int] = set()
    out: list[int] = []
    for x in parsed:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= n < k and n not in seen:
            out.append(n)
            seen.add(n)
    return out
