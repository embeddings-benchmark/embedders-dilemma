"""LLM-based retrieval tasks (LOFT-style).

Approach (from the LOFT paper):
  - The LLM is given all corpus documents formatted as:
      ID: {id} | TITLE: {title} | CONTENT: {text} | END ID: {id}
  - It is then asked to identify which documents are relevant to the query.
  - We collect the retrieved IDs, build a results dict {qid: {doc_id: score}},
    and evaluate with standard IR metrics (NDCG, MAP, MRR) via pytrec_eval.

Dataset format (MTEB standard / loft_to_mteb.py output):
  - config 'corpus'  → split 'corpus'  : _id, title, text
  - config 'queries' → split 'queries' : _id, text
  - config 'default' → split 'test'    : query-id, corpus-id, score
  HF repo names: mteb/loft-{dataset}-{scale}  (e.g. mteb/loft-nq-128k)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata

from llm_judge.evaluators.llm_retrieval_evaluator import LLMRetrievalEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AbsTaskLLMRetrieval(AbsTask):
    """Base class for LLM-based retrieval using the LOFT document-reading approach.

    Subclasses must set `metadata` and optionally `num_negatives`.

    The task bypasses the standard MTEB retrieval indexing pipeline entirely.
    Instead it:
      1. Loads corpus/queries/qrels from the HF dataset.
      2. For each query: builds a candidate pool (positives + sampled negatives).
      3. Computes NDCG@10, MAP@10, MRR@10 via pytrec_eval (in the Evaluator).
    """

    num_negatives: int = 19

    # Internal storage — populated in load_data()
    _corpus: dict[str, dict]        # {_id: {"title": ..., "text": ...}}
    _queries: dict[str, str]        # {_id: text}
    _qrels: dict[str, dict[str, int]]  # {query_id: {corpus_id: score}}

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        rev = self.metadata.dataset.get("revision", None)

        logger.info(f"Loading retrieval dataset {path} ...")

        # Corpus
        corpus_ds = load_dataset(path, "corpus", split="corpus", revision=rev, trust_remote_code=False)
        self._corpus = {
            row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
            for row in corpus_ds
        }

        # Queries
        queries_ds = load_dataset(path, "queries", split="queries", revision=rev, trust_remote_code=False)
        self._queries = {row["_id"]: row["text"] for row in queries_ds}

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

        evaluator = LLMRetrievalEvaluator(
            corpus=self._corpus,
            queries=self._queries,
            qrels=self._qrels,
        )

        scores = asyncio.run(evaluator.evaluate_async())
        scores["hf_subset"] = "default"
        scores["languages"] = list(self.metadata.eval_langs)[:1] if self.metadata.eval_langs else ["eng-Latn"]
        return {"default": scores}

    def _evaluate_subset(self, *args, **kwargs):
        # Never called — we override evaluate() directly
        raise NotImplementedError("AbsTaskLLMRetrieval overrides evaluate() directly")

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        """Bypass MTEB's automatic dataset profiling."""
        return {"num_samples": len(self._queries) if hasattr(self, "_queries") else 0}


# ---------------------------------------------------------------------------
# Concrete task classes — one per LOFT dataset × scale
# ---------------------------------------------------------------------------
# Scale key: 32k = 10 queries (dev), 128k = 100 queries (test), 1m = 100 queries (test)
# We default to 128k (100 queries) as the primary evaluation scale.
# ---------------------------------------------------------------------------

def _make_metadata(
    name: str,
    hf_path: str,
    description: str,
    revision: str | None = None,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        dataset={"path": hf_path, "revision": revision},
        description=description,
        reference="https://arxiv.org/abs/2409.02076",  # LOFT paper
        category="t2t",
        modalities=["text"],
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall@1",
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
class LLMNQRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMNQRetrieval128k",
        hf_path="mteb/loft-nq-128k",
        revision="5f96df7e7ded879c1d206e3d9826d929679d1c8c",
        description="Natural Questions retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMNQRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMNQRetrieval32k",
        hf_path="mteb/loft-nq-32k",
        revision="d81b5a78f79a18f870c221347f5e610697c5e63f",
        description="Natural Questions retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- HotpotQA ----
class LLMHotpotQARetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMHotpotQARetrieval128k",
        hf_path="mteb/loft-hotpotqa-128k",
        revision="8c3c58a1167bc2134ce23511062fc211533b266a",
        description="HotpotQA multi-hop retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMHotpotQARetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMHotpotQARetrieval32k",
        hf_path="mteb/loft-hotpotqa-32k",
        revision="8f215c1739abf7761773178a167776ac8fcb2413",
        description="HotpotQA multi-hop retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- FiQA ----
class LLMFiQARetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMFiQARetrieval128k",
        hf_path="mteb/loft-fiqa-128k",
        revision="86b19b8c1cffd10178d2faa7b5b7e1e5eee463b2",
        description="FiQA financial QA retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMFiQARetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMFiQARetrieval32k",
        hf_path="mteb/loft-fiqa-32k",
        revision="068171c1609f2e545a45fe5b56d28c40b06f8d0b",
        description="FiQA financial QA retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- ArguAna ----
class LLMArguAnaRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMArguAnaRetrieval128k",
        hf_path="mteb/loft-arguana-128k",
        revision="645851f6bb1c66f33aa399e2e1ff32ba62c601e4",
        description="ArguAna argument retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMArguAnaRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMArguAnaRetrieval32k",
        hf_path="mteb/loft-arguana-32k",
        revision="main",
        description="ArguAna argument retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- FEVER ----
class LLMFEVERRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMFEVERRetrieval128k",
        hf_path="mteb/loft-fever-128k",
        revision="f11b68afa3fea5aca5049d13ec26e41122ebe6ae",
        description="FEVER fact verification retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMFEVERRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMFEVERRetrieval32k",
        hf_path="mteb/loft-fever-32k",
        revision="main",
        description="FEVER fact verification retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- MSMARCO ----
class LLMMSMARCORetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMMSMARCORetrieval128k",
        hf_path="mteb/loft-msmarco-128k",
        revision="5b00fb471259fb3c627594220e28e42f7282367b",
        description="MS MARCO passage retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMMSMARCORetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMMSMARCORetrieval32k",
        hf_path="mteb/loft-msmarco-32k",
        revision="main",
        description="MS MARCO passage retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- SciFact ----
class LLMSciFactRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMSciFactRetrieval128k",
        hf_path="mteb/loft-scifact-128k",
        revision="1541ed7823c2bb464b00590cd51d59fe8fc01869",
        description="SciFact scientific claim retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMSciFactRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMSciFactRetrieval32k",
        hf_path="mteb/loft-scifact-32k",
        revision="main",
        description="SciFact scientific claim retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- Quora ----
class LLMQuoraRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQuoraRetrieval128k",
        hf_path="mteb/loft-quora-128k",
        revision="9aabd9d1b8ed994f4ec9d8eab0a61d04259b9a58",
        description="Quora duplicate question retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMQuoraRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQuoraRetrieval32k",
        hf_path="mteb/loft-quora-32k",
        revision="main",
        description="Quora duplicate question retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- Musique ----
class LLMMusiqueRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMMusiqueRetrieval128k",
        hf_path="mteb/loft-musique-128k",
        revision="f3cbd14cd5e944b124cc17e46631652406f8f206",
        description="MuSiQue multi-hop retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMMusiqueRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMMusiqueRetrieval32k",
        hf_path="mteb/loft-musique-32k",
        revision="47b75db947774ed7b3a3a1b31c86dde11ec37ce9",
        description="MuSiQue multi-hop retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- TopiOCQA ----
class LLMTopiOCQARetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMTopiOCQARetrieval128k",
        hf_path="mteb/loft-topiocqa-128k",
        revision="9bd8e7636ce11669d1771800dabce1c8c58baace",
        description="TopiOCQA conversational QA retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMTopiOCQARetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMTopiOCQARetrieval32k",
        hf_path="mteb/loft-topiocqa-32k",
        revision="e7d5d0e137e1c143956470526c5cd35dc0107d48",
        description="TopiOCQA conversational QA retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- Touché-2020 ----
class LLMTouche2020Retrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMTouche2020Retrieval128k",
        hf_path="mteb/loft-webis-touche2020-128k",
        revision="main",
        description="Touché-2020 argument retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMTouche2020Retrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMTouche2020Retrieval32k",
        hf_path="mteb/loft-webis-touche2020-32k",
        revision="main",
        description="Touché-2020 argument retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- QAMPARI ----
class LLMQAMPARIRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQAMPARIRetrieval128k",
        hf_path="mteb/loft-qampari-128k",
        revision="3820cdf9b98b637bca152f408eeabdcdf0efcee8",
        description="QAMPARI list question retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMQAMPARIRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQAMPARIRetrieval32k",
        hf_path="mteb/loft-qampari-32k",
        revision="70ea886acfdf519b753730571070a7e36c160c01",
        description="QAMPARI list question retrieval (LOFT 32k scale, 10 queries).",
    )


# ---- QUEST ----
class LLMQUESTRetrieval128k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQUESTRetrieval128k",
        hf_path="mteb/loft-quest-128k",
        revision="fa0b5d677bfb3a129f16f15b78e4ad7b19ab078d",
        description="QUEST long-form QA retrieval (LOFT 128k scale, 100 queries).",
    )


class LLMQUESTRetrieval32k(AbsTaskLLMRetrieval):
    metadata = _make_metadata(
        name="LLMQUESTRetrieval32k",
        hf_path="mteb/loft-quest-32k",
        revision="e40a319e812e259a66b5f11908dc7a92030e981e",
        description="QUEST long-form QA retrieval (LOFT 32k scale, 10 queries).",
    )


# ===========================================================================
# V2 — Standard MTEB datasets, small-corpus set for the LLM-vs-Embeddings paper
# ===========================================================================
# These are uploaded via scripts/sample_retrieval.py as mteb/llm-eval-* repos.
# Update the revision strings below after running the upload script.
# All qrels are normalised to split='test' during upload so the base class
# load_data() works without any changes.
# ===========================================================================

def _make_metadata_v2(
    name: str,
    hf_path: str,
    description: str,
    domains: list[str],
    eval_langs: list[str],
    task_subtypes: list[str],
    reference: str,
    revision: str | None = None,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        dataset={"path": hf_path, "revision": revision},
        description=description,
        reference=reference,
        category="t2t",
        modalities=["text"],
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=eval_langs,
        main_score="ndcg_at_10",
        date=("2019-01-01", "2024-12-31"),
        domains=domains,
        task_subtypes=task_subtypes,
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )


# ---- TempReasonL1 (12504 docs / 200 queries — capped from 4000 during upload) ----
class LLMTempReasonL1(AbsTaskLLMRetrieval):
    """TempReason L1 temporal reasoning retrieval. Corpus: 12504 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMTempReasonL1",
        hf_path="mteb/llm-eval-tempreason-l1",
        revision="1c61e95df3d70835785db4d70f17d3a16a7bd4ce",   
        description="TempReasonL1: retrieve encyclopaedic facts for temporal reasoning questions (200-query sample).",
        domains=["Encyclopaedic", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Reasoning as Retrieval"],
        reference="https://huggingface.co/datasets/mteb/TempReasonL1",
    )


# ---- Hagrid (496 docs / 496 queries) ----
class LLMHagridRetrieval(AbsTaskLLMRetrieval):
    """HAGRID open-domain QA retrieval. Corpus: 496 docs. All queries."""
    metadata = _make_metadata_v2(
        name="LLMHagridRetrieval",
        hf_path="mteb/llm-eval-hagrid",
        revision="6383cc91b1b3f16c528eafdf704d89e7e64f3e75",   
        description="HAGRID generative-retrieval QA over encyclopaedic passages.",
        domains=["Encyclopaedic", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Question answering"],
        reference="https://github.com/project-miracl/hagrid",
    )


# ---- LegalBench Corporate Lobbying (319 docs / 340 queries) ----
class LLMLegalBenchCorporateLobbying(AbsTaskLLMRetrieval):
    """LegalBench bill title → bill summary retrieval. Corpus: 319 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMLegalBenchCorporateLobbying",
        hf_path="mteb/llm-eval-legalbench-corporate-lobbying",
        revision="300cb175d1b608cfc3fdb94fd06b3ceb937f9b6a",   
        description="LegalBench corporate lobbying: retrieve bill summaries from titles.",
        domains=["Legal", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Article retrieval"],
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying",
    )


# ---- AILA Statutes (82 docs / 50 queries) ----
class LLMAILAStatutes(AbsTaskLLMRetrieval):
    """AILA 2019 Indian statute retrieval. Corpus: 82 docs. 50 queries."""
    metadata = _make_metadata_v2(
        name="LLMAILAStatutes",
        hf_path="mteb/llm-eval-aila-statutes",
        revision="a2acf12d293ea934823ca26752a05c5bfab24dff",   
        description="AILA 2019: retrieve the most relevant Indian statute for a legal situation.",
        domains=["Legal", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Article retrieval"],
        reference="https://zenodo.org/records/4063986",
    )


# ---- SpartQA (1592 docs / 200 queries — capped from 3594 during upload) ----
class LLMSpartQA(AbsTaskLLMRetrieval):
    """SpartQA spatial reasoning retrieval. Corpus: 1592 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMSpartQA",
        hf_path="mteb/llm-eval-spartqa",
        revision="d6628df131c3279007b88fe3e057e809dac056af",   
        description="SpartQA: retrieve ground-truth answers to spatial reasoning questions (200-query sample).",
        domains=["Encyclopaedic", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Reasoning as Retrieval"],
        reference="https://github.com/HLR/SpartQA_generation",
    )


# ---- WinoGrande (5095 docs / 200 queries — capped from 1267 during upload) ----
class LLMWinoGrande(AbsTaskLLMRetrieval):
    """WinoGrande commonsense reasoning retrieval. Corpus: 5095 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMWinoGrande",
        hf_path="mteb/llm-eval-winogrande",
        revision="bc2a1e4c24771a6f3432578016cfb737732398fd",   
        description="WinoGrande: retrieve the answer for a commonsense fill-in-the-blank (200-query sample).",
        domains=["Encyclopaedic", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Reasoning as Retrieval"],
        reference="https://winogrande.allenai.org/",
    )


# ---- TwitterHjerneRetrieval (262 docs / 78 queries — Danish) ----
class LLMTwitterHjerneRetrieval(AbsTaskLLMRetrieval):
    """Danish Twitter QA retrieval. Corpus: 262 docs. All 77 queries."""
    metadata = _make_metadata_v2(
        name="LLMTwitterHjerneRetrieval",
        hf_path="mteb/llm-eval-twitter-hjerne",
        revision="31f9b918c30ef94e15a168cb95ca4ebf291396eb",
        description="Danish question answering retrieval from Twitter (#Twitterhjerne). Adds multilingual signal.",
        domains=["Social", "Written"],
        eval_langs=["dan-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne",
    )


# ===========================================================================
# V3 — Post-audit additions (replacing TempReason/SpartQA/WinoGrande in main
# retrieval analysis; those tasks moved to "Reasoning-over-candidates" appendix).
# Each is genuine small-corpus semantic retrieval over informative documents
# (no answer-pool corpora). Some (BuiltBench, FinanceBench) are kept available
# but inactive in the main suite — see per-class NOTE comments for reasons.
# ===========================================================================


# ---- HumanEval (158 docs / 158→100 queries — English code retrieval) ----
class LLMHumanEvalRetrieval(AbsTaskLLMRetrieval):
    """HumanEval text-to-code retrieval. Corpus: 158 code solutions. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMHumanEvalRetrieval",
        hf_path="mteb/llm-eval-humaneval",
        revision="b53fc23a1dfcb4659b288319dce4f5f3e750c73c",
        description="HumanEval: retrieve the Python function body that solves a given natural-language problem (100-query sample).",
        domains=["Programming", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Code retrieval"],
        reference="https://huggingface.co/datasets/embedding-benchmark/HumanEval",
    )


# ---- FQuAD (269 docs / 400→100 queries — French Wikipedia QA) ----
class LLMFQuADRetrieval(AbsTaskLLMRetrieval):
    """French QA retrieval over Wikipedia passages. Corpus: 269 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMFQuADRetrieval",
        hf_path="mteb/llm-eval-fquad",
        revision="dc5443dbfad5cd88c9507d27db4cde4caaede178",
        description="FQuAD: retrieve the French-Wikipedia paragraph that answers a French question (100-query sample).",
        domains=["Encyclopaedic", "Written"],
        eval_langs=["fra-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/manu/fquad2_test",
    )


# ---- LegalBench Consumer Contracts QA (154 docs / 396→100 queries) ----
class LLMLegalBenchConsumerContractsQA(AbsTaskLLMRetrieval):
    """LegalBench consumer-contract Q&A retrieval. Corpus: 154 ToS clauses. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMLegalBenchConsumerContractsQA",
        hf_path="mteb/llm-eval-legalbench-consumer-contracts",
        revision="642870c78f65ed31a2ab0f8ab113f2957ff2df27",
        description="LegalBench consumer contracts QA: retrieve the relevant Terms-of-Service clause (100-query sample).",
        domains=["Legal", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa",
    )


# ---- BuiltBench (2761 docs / 334→100 queries — built environment) ----
# NOTE: corpus is 207K tokens (LoFT-wrapped); exceeds 128K context of most open LLMs.
# Kept available for Gemini-only analyses; not in main retrieval suite.
class LLMBuiltBenchRetrieval(AbsTaskLLMRetrieval):
    """Built-environment / IFC-taxonomy retrieval. Corpus: 2761 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMBuiltBenchRetrieval",
        hf_path="mteb/llm-eval-builtbench",
        revision="134196fb5fb3148a13dacfd539b8cc7f27099ec3",
        description="BuiltBench: retrieve the named built-environment entity (from IFC/Uniclass) matching a textual description (100-query sample).",
        domains=["Engineering", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Article retrieval"],
        reference="https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-retrieval",
    )


# ---- PublicHealthQA English (172→100 queries — medical/public-health) ----
class LLMPublicHealthQA(AbsTaskLLMRetrieval):
    """Public-health QA retrieval (English subset). Corpus: 172 docs. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMPublicHealthQA",
        hf_path="mteb/llm-eval-public-health-qa",
        revision="b05938525381b7aebc079f88fc3ed8f572a80bb9",
        description="PublicHealthQA: retrieve the relevant public-health/COVID-19 information passage for a public-health question (100-query sample, English subset).",
        domains=["Medical", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/xhluca/publichealth-qa",
    )


# ---- FinanceBenchRetrieval (145 docs / 150→100 queries — financial analysis) ----
# NOTE: saturated for top embedders (Octen 0.946, gemini-embed 0.916). Kept available
# but not in main suite — replaced by LLMHC3FinanceRetrieval below.
class LLMFinanceBenchRetrieval(AbsTaskLLMRetrieval):
    """FinanceBench retrieval. Corpus: 145 financial-report passages. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMFinanceBenchRetrieval",
        hf_path="mteb/llm-eval-finance-bench",
        revision="450ac2680b8c18d74e87088ae513a74c7df4d168",
        description="FinanceBench: retrieve the relevant 10-K/balance-sheet/cash-flow passage to answer a financial-analysis question (100-query sample).",
        domains=["Financial", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/PatronusAI/financebench",
    )


# ---- HC3FinanceRetrieval (415 docs / 415→100 queries — finance Q&A) ----
class LLMHC3FinanceRetrieval(AbsTaskLLMRetrieval):
    """HC3 Finance retrieval. Corpus: 415 finance forum/help answers. 100 queries."""
    metadata = _make_metadata_v2(
        name="LLMHC3FinanceRetrieval",
        hf_path="mteb/llm-eval-hc3-finance",
        revision="8733760a6f3e9ddb8bacaaf020906c2ba5a5e30b",
        description="HC3-Finance: retrieve the relevant finance Q&A answer for an open-ended finance question (100-query sample).",
        domains=["Financial", "Written"],
        eval_langs=["eng-Latn"],
        task_subtypes=["Question answering"],
        reference="https://huggingface.co/datasets/Hello-SimpleAI/HC3",
    )

