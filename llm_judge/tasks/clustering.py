import asyncio

import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from mteb import EncoderProtocol
from mteb.abstasks import AbsTaskClustering, AbsTaskClusteringLegacy
from pydantic import BaseModel, Field
from sklearn.metrics import v_measure_score

from llm_judge.llm_client import send_request, send_request_rlm
from mteb.tasks import (
    ArxivClusteringP2P,
    BigPatentClustering,
    RedditClusteringP2P,
    StackExchangeClusteringP2P,
    TwentyNewsgroupsClustering,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixin: all LLM clustering logic lives here, shared by both base variants.
# ---------------------------------------------------------------------------

class LLMClusteringMixin:
    """LLM clustering logic — listed first in MRO so _evaluate_subset takes priority."""

    instruction: str

    async def _cluster_split(
        self,
        sentences: list[str],
        n_clusters: int,
    ) -> tuple[list[int], dict, str]:
        """Cluster documents via LLM or RLM. Returns (assignments, usage, raw_response)."""

        from llm_judge.settings import Settings
        _settings = Settings()

        if _settings.use_rlm:
            return await self._cluster_split_rlm(sentences, n_clusters)
        return await self._cluster_split_llm(sentences, n_clusters)

    async def _cluster_split_rlm(
        self,
        sentences: list[str],
        n_clusters: int,
    ) -> tuple[list[int], dict, str]:
        """RLM-native clustering: give RLM all documents as context and let it reason.

        The RLM already knows how to use its REPL, llm_query, rlm_query, etc.
        We just tell it the task and the output format — it figures out the strategy.
        """
        import re
        import random


        # Pass documents as a Python list — LocalREPL serializes list to JSON then loads it back,
        # so the REPL `context` variable will be a proper Python list from the start.
        context_data = sentences

        root_prompt = (
            f"{self.instruction}\n\n"
            f"The `context` variable contains a list of {len(sentences)} documents (strings). "
            f"Assign each document to exactly one of {n_clusters} clusters "
            f"(numbered 0 to {n_clusters - 1}). "
            f"Documents discussing the same topic should be in the same cluster.\n\n"
            f"Store the result as a Python list called `assignments` — "
            f"exactly {len(sentences)} integers, one per document in order — "
            f"then call FINAL_VAR(assignments)."
        )

        # Run RLM: it will iterate, call llm_query_batched, aggregate in REPL code
        result, usage = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: send_request_rlm(root_prompt, context_data)
        )

        res_text = result or ""
        # Parse result — RLM should produce a Python list repr or JSON array
        if res_text.strip():
            # Try JSON array
            try:
                import json
                parsed = json.loads(res_text.strip())
                if isinstance(parsed, list):
                    assignments = [int(x) for x in parsed]
                    return self._pad_and_clamp_static(assignments, len(sentences), n_clusters), usage, res_text
            except Exception:
                pass
            # Try Python list repr (ast.literal_eval)
            try:
                import ast
                parsed = ast.literal_eval(res_text.strip())
                if isinstance(parsed, list):
                    assignments = [int(x) for x in parsed]
                    return self._pad_and_clamp_static(assignments, len(sentences), n_clusters), usage, res_text
            except Exception:
                pass
            # Fallback: regex extract integers
            nums = re.findall(r'\b(\d+)\b', res_text)
            if len(nums) >= len(sentences) // 2:
                assignments = [max(0, min(int(n), n_clusters - 1)) for n in nums[:len(sentences)]]
                return self._pad_and_clamp_static(assignments, len(sentences), n_clusters), usage, res_text

        logger.warning(f"[RLM] Could not parse assignments from response: {res_text[:200]}")
        return [random.randint(0, n_clusters - 1) for _ in range(len(sentences))], usage, res_text

    @staticmethod
    def _pad_and_clamp_static(assignments: list[int], n_docs: int, n_clusters: int) -> list[int]:
        import random
        if len(assignments) < n_docs:
            assignments.extend([random.randint(0, n_clusters - 1) for _ in range(n_docs - len(assignments))])
        assignments = assignments[:n_docs]
        return [max(0, min(a, n_clusters - 1)) for a in assignments]

    async def _cluster_split_llm(
        self,
        sentences: list[str],
        n_clusters: int,
    ) -> tuple[list[int], dict, str]:
        """Plain LLM clustering: single JSON-schema call, retries + fallback parsing. Returns (assignments, usage, raw_response)."""
        import re
        import random

        class ClusterResponseModel(BaseModel):
            assignments: list[int] = Field(
                description="A list of cluster IDs (0-indexed), one per document in the same order."
            )

        docs_str = "\n".join(f"[Doc {i}]: {s}" for i, s in enumerate(sentences))

        instructions = (
            f"{self.instruction}\n\n"
            f"You will be given {len(sentences)} documents. "
            f"Assign each document to exactly one of {n_clusters} clusters (numbered 0 to {n_clusters - 1}). "
            f"Documents that discuss the same topic should be in the same cluster.\n\n"
            f"Output ONLY a compact, minified JSON object with a single 'assignments' field: "
            f"a list of exactly {len(sentences)} integers. Example: "
            f'{{"assignments":[0,2,1,3,0,...]}}'
        )
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "ClusterResponse",
                "schema": ClusterResponseModel.model_json_schema(),
                "description": "Schema for the clustering response.",
                "strict": True,
            },
        }

        def _parse(text: str) -> list[int] | None:
            if not text or not text.strip():
                return None
            try:
                return ClusterResponseModel.model_validate_json(text).assignments
            except Exception:
                pass
            clean = re.sub(r'```(?:json)?\n(.*?)```', r'\1', text, flags=re.DOTALL).strip()
            try:
                return ClusterResponseModel.model_validate_json(clean).assignments
            except Exception:
                pass
            match = re.search(r'"assignments"\s*:\s*\[([0-9,\s]*)', text)
            if match:
                nums = [int(x.strip()) for x in match.group(1).rstrip(", \n").split(",") if x.strip().isdigit()]
                if nums:
                    return nums
            match = re.search(r'\[\s*(\d+(?:\s*,\s*\d+)*)', text)
            if match:
                nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
                if nums:
                    return nums
            # If all parsing completely fails, return None so _fix() assigns random clusters 
            # (which correctly penalizes V-measure instead of inflating a default).
            return None

        def _fix(assignments: list[int]) -> list[int]:
            if len(assignments) < len(sentences):
                assignments.extend([random.randint(0, n_clusters - 1) for _ in range(len(sentences) - len(assignments))])
            return [max(0, min(a, n_clusters - 1)) for a in assignments[:len(sentences)]]

        last_response = ""
        last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}
        for attempt in range(2):
            response, usage = await send_request(instructions=instructions, input=docs_str, response_format=rf, bypass_rlm=True)
            last_response = response or ""
            last_usage = usage
            result = _parse(last_response)
            if result:
                return _fix(result), usage, last_response
            if attempt == 0:
                logger.warning("JSON parse failed, retrying...")

        logger.warning(f"All parsing failed. Response preview: {last_response[:200]}")
        return _fix(_parse(last_response) or []), last_usage, last_response

    async def _evaluate_subset_async(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Our datasets are already small (sampled), so use all rows directly.
        downsampled_dataset = data_split

        downsampled_dataset = downsampled_dataset.select_columns(
            [self.input_column_name, self.label_column_name]
        )

        all_sentences: list[list[str]] = []
        all_labels: list[list[str]] = []

        for row in downsampled_dataset:
            sents = row[self.input_column_name]
            lbls = row[self.label_column_name]
            if isinstance(sents, str):
                sents = [sents]
            if not isinstance(lbls, list):
                lbls = [lbls]
            all_sentences.append(sents)
            all_labels.append([str(l) for l in lbls])

        all_v_scores: list[float] = []
        all_assignments: list[list[int]] = []

        total_usage = {
            "input_tokens":    0,
            "cached_tokens":   0,
            "output_tokens":   0,
            "thinking_tokens": 0,
            "total_tokens":    0,
            "cost":            0.0,
            "rlm_total_calls": 0,   # total sub-LM calls (RLM path only; 0 for LLM)
            "rlm_wall_time_s": 0.0, # total RLM wall time (RLM path only; 0 for LLM)
        }

        for i, (sentences, cluster_labels) in enumerate(zip(all_sentences, all_labels)):
            n_clusters = len(set(cluster_labels))
            logger.info(
                f"[LLM Clustering] split {i + 1}/{len(all_labels)}: "
                f"{len(sentences)} docs, {n_clusters} clusters"
            )

            try:
                assignments, usage, raw_response = await self._cluster_split(sentences, n_clusters)
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
                
                if i == 0:
                    try:
                        import json
                        import os
                        from llm_judge.llm_client import settings
                        
                        sample_docs = []
                        for idx in range(min(5, len(sentences))):
                            sample_docs.append({
                                "input_text": sentences[idx],
                                "predicted_cluster": assignments[idx],
                                "gold_cluster": int(cluster_labels[idx]) if str(cluster_labels[idx]).isdigit() else cluster_labels[idx]
                            })
                        
                        model_name = settings.model.replace("/", "__")
                        output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        sample_path = os.path.join(output_dir, f"{self.metadata.name}_{hf_subset}_samples.json")
                        with open(sample_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "task": self.metadata.name,
                                "model_raw_output_preview": raw_response[:5000],
                                "samples": sample_docs
                            }, f, indent=4)
                        logger.info(f"Saved enriched clustering samples to {sample_path}")
                    except Exception as ex:
                        logger.warning(f"Failed to save clustering samples: {ex}")

            except Exception as e:
                logger.error(f"Error during async clustering evaluation: {e}")
                logger.warning(f"Assigning random clusters for split {i+1} due to error.")
                import random
                assignments = [random.randint(0, n_clusters - 1) for _ in range(len(sentences))]

            v_score = v_measure_score(cluster_labels, assignments)
            logger.info(f"  V-measure: {v_score:.4f}")
            all_v_scores.append(v_score)
            all_assignments.append(assignments)

        if prediction_folder:
            self._save_task_predictions(
                all_assignments,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        mean_v = float(np.mean(all_v_scores))
        std_v = float(np.std(all_v_scores))
        logger.info(f"Clustering done: V-measure = {mean_v:.4f} ± {std_v:.4f}")

        return {
            "v_measures": all_v_scores,
            "v_measure": mean_v,
            "v_measure_std": std_v,
            "usage_stats": total_usage,
        }

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return asyncio.run(
            self._evaluate_subset_async(
                model=model,
                data_split=data_split,
                encode_kwargs=encode_kwargs,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prediction_folder=prediction_folder,
                num_proc=num_proc,
                **kwargs,
            )
        )


# ---------------------------------------------------------------------------
# Two concrete base classes — same logic, different MTEB parent.
# Mixin is listed FIRST so its _evaluate_subset wins in the MRO.
# ---------------------------------------------------------------------------

class AbsTaskLLMClustering(LLMClusteringMixin, AbsTaskClusteringLegacy):
    """For tasks that inherit from AbsTaskClusteringLegacy:
    TwentyNewsgroups, Reddit, arXiv, BigPatent.
    """


class AbsTaskLLMClusteringFast(LLMClusteringMixin, AbsTaskClustering):
    """For tasks that inherit from AbsTaskClustering: StackExchange."""


# ===================================================================
# Premium Clustering Tasks (10)
# ===================================================================

from mteb.tasks import (
    StackExchangeClustering,
    ArxivClusteringS2S,
    BiorxivClusteringP2P, MedrxivClusteringP2P, MedrxivClusteringS2S
)

class LLMRedditClusteringP2P(AbsTaskLLMClustering, RedditClusteringP2P):
    metadata = RedditClusteringP2P.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-reddit_clustering_p2p", "revision": "298518f04f9f2a5849282dd20e43672172428790"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following Reddit posts by their subreddit community topic."

class LLMBigPatentClustering(AbsTaskLLMClustering, BigPatentClustering):
    metadata = BigPatentClustering.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-big_patent_clustering", "revision": "fbb01d02cb8308089d99223cbcc35c584a67ba77"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following patent abstracts by their technology domain (e.g., chemistry, physics, mechanical engineering, electricity)."

class LLMTwentyNewsgroupsClusteringV2(AbsTaskLLMClustering, TwentyNewsgroupsClustering):
    metadata = TwentyNewsgroupsClustering.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-twenty_newsgroups_v2", "revision": "0187d654bb254527a7050c608ced9967dc04db91"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following Usenet newsgroup posts by topic."

class LLMStackExchangeClusteringP2PV2(AbsTaskLLMClusteringFast, StackExchangeClusteringP2P):
    metadata = StackExchangeClusteringP2P.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-stackexchange_clustering_p2p_v2", "revision": "5de039af34921939fb4a1c8b2e4f9f0a6ed6cd50"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following StackExchange questions (title+body) by their technical subject area."

class LLMStackExchangeClusteringV2(AbsTaskLLMClusteringFast, StackExchangeClustering):
    metadata = StackExchangeClustering.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-stackexchange_clustering_v2", "revision": "2cbee8fd7c715e92a99639d46ee940e4bf64e5d2"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following StackExchange questions (title only) by their technical subject area."

class LLMArxivClusteringP2P(AbsTaskLLMClustering, ArxivClusteringP2P):
    metadata = ArxivClusteringP2P.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-arxiv_clustering_p2p", "revision": "3ce50711e47553e7c9f47b53819ccd64cd0f5b9b"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following arXiv paper titles and abstracts by their research area."

class LLMArxivClusteringS2S(AbsTaskLLMClustering, ArxivClusteringS2S):
    metadata = ArxivClusteringS2S.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-arxiv_clustering_s2s", "revision": "28c899f0bb82d9001804d0f408d66932effc0c1c"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following arXiv paper titles by their research area."

class LLMBiorxivClusteringP2PV2(AbsTaskLLMClusteringFast, BiorxivClusteringP2P):
    metadata = BiorxivClusteringP2P.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-biorxiv_clustering_p2p_v2", "revision": "9e11c95384ef78952ba754f9d8942084ddbb61a7"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following Biorxiv paper titles and abstracts by their main category."

class LLMMedrxivClusteringP2PV2(AbsTaskLLMClusteringFast, MedrxivClusteringP2P):
    metadata = MedrxivClusteringP2P.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-medrxiv_clustering_p2p_v2", "revision": "63c8e6cfbcab3f986291141799fe646c60bb441c"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following Medrxiv paper titles and abstracts by their main category."

class LLMMedrxivClusteringS2SV2(AbsTaskLLMClusteringFast, MedrxivClusteringS2S):
    metadata = MedrxivClusteringS2S.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-medrxiv_clustering_s2s_v2", "revision": "c565eea82f4a8728b8fe5181388b407d305f2647"},
        "eval_splits": ["test"],
    })
    instruction = "Cluster the following Medrxiv paper titles by their main category."
