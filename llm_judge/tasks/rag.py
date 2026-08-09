"""LLM-based RAG tasks (LOFT-style).

Approach (from the LOFT paper):
  - The LLM is given all corpus documents formatted as:
      ID: {id} | TITLE: {title} | CONTENT: {text} | END ID: {id}
  - It is then given the query and asked to generate a short answer.
  - We collect the generated text and compare it with the gold answers
    using Rouge-L and Exact Match.

Dataset format (from loft_to_rag.py):
  - config 'corpus'  → split 'corpus'  : _id, title, text
  - config 'queries' → split 'queries' : _id, text, gold_answers (list of strings)
  - config 'default' → split 'test'    : query-id, corpus-id, score
  HF repo names: mteb/rag-{dataset}-{scale} (e.g. mteb/rag-nq-128k)

Note: Like the retrieval task, we bypass standard MTEB indexing.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata

from llm_judge.evaluators.llm_rag_evaluator import LLMRAGEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AbsTaskLLMRAG(AbsTask):
    """Base class for LLM-based RAG using the LOFT document-reading approach.

    Subclasses must set `metadata` and optionally `num_negatives`.

    The task bypasses the standard MTEB retrieval indexing pipeline entirely.
    Instead it delegates to LLMRAGEvaluator which:
      1. For each query: builds a candidate pool from the entire provided corpus.
      2. Sends the pool + query to the LLM and parses the generated answer.
      3. Computes Exact Match and Rouge-L against the gold_answers.
    """

    is_multi_answer: bool = False

    # Internal storage — populated in load_data()
    _corpus: dict[str, dict]        # {_id: {"title": ..., "text": ...}}
    _queries: dict[str, dict]       # {_id: {"text": ..., "gold_answers": [...]}}
    _qrels: dict[str, dict[str, int]]  # {query_id: {corpus_id: score}}

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        rev = self.metadata.dataset.get("revision", None)

        logger.info(f"Loading RAG dataset {path} ...")

        # Corpus
        corpus_ds = load_dataset(path, "corpus", split="corpus", revision=rev, trust_remote_code=False)
        self._corpus = {
            row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
            for row in corpus_ds
        }

        # Queries (RAG modified: includes gold_answers)
        queries_ds = load_dataset(path, "queries", split="queries", revision=rev, trust_remote_code=False)
        self._queries = {
            row["_id"]: {"text": row["text"], "gold_answers": row.get("gold_answers", [])}
            for row in queries_ds
        }

        # Qrels (default config, test split)
        qrels_ds = load_dataset(path, "default", split="test", revision=rev, trust_remote_code=False)
        self._qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for row in qrels_ds:
            self._qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.data_loaded = True
        logger.info(
            f"Loaded: {len(self._corpus)} docs, {len(self._queries)} queries, "
            f"{len(self._qrels)} queries with qrels"
        )

    def evaluate(
        self,
        model: Any,
        split: str = "test",
        subsets_to_run: list[str] | None = None,
        *,
        encode_kwargs: dict,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs,
    ) -> dict:
        if not self.data_loaded:
            self.load_data(num_proc=num_proc)

        evaluator = LLMRAGEvaluator(
            corpus=self._corpus,
            queries=self._queries,
            qrels=self._qrels,
            is_multi_answer=self.is_multi_answer,
        )

        import asyncio
        scores = asyncio.run(evaluator.evaluate_async())
        scores["hf_subset"] = "default"
        scores["languages"] = list(self.metadata.eval_langs)[:1] if self.metadata.eval_langs else ["eng-Latn"]
        return {"default": scores}

    def _evaluate_subset(self, *args, **kwargs):
        # Never called — we override evaluate() directly
        raise NotImplementedError("AbsTaskLLMRAG overrides evaluate() directly")

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        """Bypass MTEB's automatic dataset profiling."""
        return {"num_samples": len(self._queries) if hasattr(self, "_queries") else 0}


class AbsTaskHybridRAG(AbsTaskLLMRAG):
    """
    Retrieve-then-Read via HybridRAGEvaluator.
    Expects evaluate() to be called with a loaded `model` (the embedding encoder).
    Will retrieve `top_k` documents, and then use the LLM to generate an answer.
    """
    top_k: int = 5

    def evaluate(
        self,
        model: Any,
        split: str = "test",
        subsets_to_run: list[str] | None = None,
        *,
        encode_kwargs: dict,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs,
    ) -> dict:
        if not self.data_loaded:
            self.load_data(num_proc=num_proc)

        # Here `model` must be the embedding encoder, e.g. sentence-transformers
        from llm_judge.evaluators.llm_rag_evaluator import HybridRAGEvaluator
        import copy

        # Spoof the metadata name to "NQ" so MTEB's InstructWrapper doesn't crash 
        # trying to look up our custom Hybrid class name in its native registry.
        encoder_meta = copy.deepcopy(self.metadata)
        encoder_meta.name = "NQ"

        evaluator = HybridRAGEvaluator(
            corpus=self._corpus,
            queries=self._queries,
            qrels=self._qrels,
            encoder=model,
            top_k=self.top_k,
            is_multi_answer=self.is_multi_answer,
            encoder_kwargs={
                "task_name": "NQ",
                "task_metadata": encoder_meta,
                "hf_split": split,
                "hf_subset": "default",
            }
        )

        import asyncio
        scores = asyncio.run(evaluator.evaluate_async())
        scores["hf_subset"] = "default"
        scores["languages"] = list(self.metadata.eval_langs)[:1] if self.metadata.eval_langs else ["eng-Latn"]
        return {"default": scores}


# ---------------------------------------------------------------------------
# Concrete task classes — one per LOFT dataset × scale
# ---------------------------------------------------------------------------
# Only the 6 datasets that have gold answers are supported for RAG:
# NQ, TopiOCQA, HotpotQA, Musique, QAMPARI, Quest
# ---------------------------------------------------------------------------

def _make_metadata(
    name: str,
    hf_path: str,
    revision: str,
    description: str,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        dataset={"path": hf_path, "revision": revision},
        description=description,
        reference="https://arxiv.org/abs/2409.02076",  # LOFT paper
        category="t2t",
        modalities=["text"],
        type="Retrieval",  # MTEB doesn't have a "RAG" type, we masquerade as Retrieval
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="em",
        date=("2024-01-01", "2024-12-31"),
        domains=["Written"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )


# ---- NQ ----
class LLMNQRAG128k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMNQRAG128k",
        hf_path="mteb/rag-nq-128k",
        revision="f0826fbcdb275b3025f466b9148a00469171ffe1",
        description="Natural Questions RAG (LOFT 128k scale, 100 queries).",
    )


class LLMNQRAG32k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMNQRAG32k",
        hf_path="mteb/rag-nq-32k",
        revision="29083af3ad761d1f294ead591bdcfc11a1b4f131",
        description="Natural Questions RAG (LOFT 32k scale, 10 queries).",
    )


# ---- HotpotQA ----
class LLMHotpotQARAG128k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMHotpotQARAG128k",
        hf_path="mteb/rag-hotpotqa-128k",
        revision="44bcbdacd130a538ebbdf436dafbb19ab194f443",
        description="HotpotQA multi-hop RAG (LOFT 128k scale, 100 queries).",
    )


class LLMHotpotQARAG32k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMHotpotQARAG32k",
        hf_path="mteb/rag-hotpotqa-32k",
        revision="c29073d96ad8b2c0179caa561f93fec64a927f4e",
        description="HotpotQA multi-hop RAG (LOFT 32k scale, 10 queries).",
    )


# ---- TopiOCQA ----
class LLMTopiOCQARAG128k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMTopiOCQARAG128k",
        hf_path="mteb/rag-topiocqa-128k",
        revision="835ba23e6136944d1b95837acc1586eae7704c11",
        description="TopiOCQA conversational RAG (LOFT 128k scale, 100 queries per dialog).",
    )

class LLMTopiOCQARAG32k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMTopiOCQARAG32k",
        hf_path="mteb/rag-topiocqa-32k",
        revision="8943d44012596f7a49eb11d4b7bbf7a262e3e38b",
        description="TopiOCQA conversational RAG (LOFT 32k scale, 10 queries per dialog).",
    )

# ---- Musique ----
class LLMMusiqueRAG128k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMMusiqueRAG128k",
        hf_path="mteb/rag-musique-128k",
        revision="c3ce7979ff99ebda432941d48a18b6476fcb5c6f",
        description="MuSiQue multi-hop RAG (LOFT 128k scale, 100 queries).",
    )

class LLMMusiqueRAG32k(AbsTaskLLMRAG):
    metadata = _make_metadata(
        name="LLMMusiqueRAG32k",
        hf_path="mteb/rag-musique-32k",
        revision="fed69e6074dfb7a395cb27ad4ba8670dab569ca6",
        description="MuSiQue multi-hop RAG (LOFT 32k scale, 10 queries).",
    )

# ---- QAMPARI ----
class LLMQAMPARIRAG128k(AbsTaskLLMRAG):
    is_multi_answer = True
    metadata = _make_metadata(
        name="LLMQAMPARIRAG128k",
        hf_path="mteb/rag-qampari-128k",
        revision="ed16daf35f67c2225e6c1e69923ca9d0db918e40",
        description="QAMPARI list-based RAG (LOFT 128k scale, 100 queries).",
    )

class LLMQAMPARIRAG32k(AbsTaskLLMRAG):
    is_multi_answer = True
    metadata = _make_metadata(
        name="LLMQAMPARIRAG32k",
        hf_path="mteb/rag-qampari-32k",
        revision="46c731990293e1e5259d157a9bc40a25ea742aef",
        description="QAMPARI list-based RAG (LOFT 32k scale, 10 queries).",
    )

# ---- Quest ----
class LLMQuestRAG128k(AbsTaskLLMRAG):
    is_multi_answer = True
    metadata = _make_metadata(
        name="LLMQuestRAG128k",
        hf_path="mteb/rag-quest-128k",
        revision="99ce8a17a7822599cb26a3308c1a057cdb1442f5",
        description="QASper/Quest scientific RAG (LOFT 128k scale, 100 queries).",
    )

class LLMQuestRAG32k(AbsTaskLLMRAG):
    is_multi_answer = True
    metadata = _make_metadata(
        name="LLMQuestRAG32k",
        hf_path="mteb/rag-quest-32k",
        revision="f963170004aed23418342f1a92a1489f529ff4b2",
        description="QASper/Quest scientific RAG (LOFT 32k scale, 10 queries).",
    )

# ---------------------------------------------------------------------------
# Hybrid RAG Classes (Retrieve then Read)
# Same metadata and subsets, but inherits from AbsTaskHybridRAG
# ---------------------------------------------------------------------------

class HybridNQRAG128k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridNQRAG128k", "mteb/rag-nq-128k", "f0826fbcdb275b3025f466b9148a00469171ffe1", "NQ Hybrid RAG (128k pool)")

class HybridNQRAG32k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridNQRAG32k", "mteb/rag-nq-32k", "29083af3ad761d1f294ead591bdcfc11a1b4f131", "NQ Hybrid RAG (32k pool)")

class HybridHotpotQARAG128k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridHotpotQARAG128k", "mteb/rag-hotpotqa-128k", "44bcbdacd130a538ebbdf436dafbb19ab194f443", "HotpotQA Hybrid RAG (128k pool)")

class HybridHotpotQARAG32k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridHotpotQARAG32k", "mteb/rag-hotpotqa-32k", "c29073d96ad8b2c0179caa561f93fec64a927f4e", "HotpotQA Hybrid RAG (32k pool)")

class HybridTopiOCQARAG128k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridTopiOCQARAG128k", "mteb/rag-topiocqa-128k", "835ba23e6136944d1b95837acc1586eae7704c11", "TopiOCQA Hybrid RAG (128k pool)")

class HybridTopiOCQARAG32k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridTopiOCQARAG32k", "mteb/rag-topiocqa-32k", "8943d44012596f7a49eb11d4b7bbf7a262e3e38b", "TopiOCQA Hybrid RAG (32k pool)")

class HybridMusiqueRAG128k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridMusiqueRAG128k", "mteb/rag-musique-128k", "c3ce7979ff99ebda432941d48a18b6476fcb5c6f", "MuSiQue Hybrid RAG (128k pool)")

class HybridMusiqueRAG32k(AbsTaskHybridRAG):
    metadata = _make_metadata("HybridMusiqueRAG32k", "mteb/rag-musique-32k", "fed69e6074dfb7a395cb27ad4ba8670dab569ca6", "MuSiQue Hybrid RAG (32k pool)")

class HybridQAMPARIRAG128k(AbsTaskHybridRAG):
    is_multi_answer = True
    metadata = _make_metadata("HybridQAMPARIRAG128k", "mteb/rag-qampari-128k", "ed16daf35f67c2225e6c1e69923ca9d0db918e40", "QAMPARI Hybrid RAG (128k pool)")

class HybridQAMPARIRAG32k(AbsTaskHybridRAG):
    is_multi_answer = True
    metadata = _make_metadata("HybridQAMPARIRAG32k", "mteb/rag-qampari-32k", "46c731990293e1e5259d157a9bc40a25ea742aef", "QAMPARI Hybrid RAG (32k pool)")

class HybridQuestRAG128k(AbsTaskHybridRAG):
    is_multi_answer = True
    metadata = _make_metadata("HybridQuestRAG128k", "mteb/rag-quest-128k", "99ce8a17a7822599cb26a3308c1a057cdb1442f5", "Quest Hybrid RAG (128k pool)")

class HybridQuestRAG32k(AbsTaskHybridRAG):
    is_multi_answer = True
    metadata = _make_metadata("HybridQuestRAG32k", "mteb/rag-quest-32k", "f963170004aed23418342f1a92a1489f529ff4b2", "Quest Hybrid RAG (32k pool)")
