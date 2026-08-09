import asyncio
import logging
from time import time
from pathlib import Path
from typing import Any

from mteb import SearchProtocol, TaskMetadata
from mteb._evaluators import RetrievalEvaluator
from mteb._evaluators.retrieval_metrics import make_score_dict
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.models import MTEBModels
from mteb.tasks import (
    HUMECore17InstructionReranking,
    HUMENews21InstructionReranking,
    HUMERobust04InstructionReranking,
    HUMEWikipediaRerankingMultilingual,
)
from mteb.types import (
    CorpusDatasetType,
    QueryDatasetType,
    RetrievalOutputType,
    ScoresDict,
    TopRankedDocumentsType,
)
from pydantic import BaseModel, Field

from llm_judge.llm_client import send_request
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


class LLMRetrievalEvaluator(RetrievalEvaluator):
    def __init__(
        self,
        corpus: CorpusDatasetType,
        queries: QueryDatasetType,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        instruction: str,
        top_ranked: TopRankedDocumentsType | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            corpus,
            queries,
            task_metadata,
            hf_split,
            hf_subset,
            top_k,
            top_ranked,
            **kwargs,
        )
        self.instruction = instruction

    async def _score_pair(
        self,
        query_text: str,
        corpus_text: str,
        instruction: str,
    ) -> tuple[float, dict, str, str]:
        """Score a single query–document pair via LLM."""

        class ResponseModel(BaseModel):
            reasoning: str | None = Field(None)
            output: int = Field(ge=0, le=1)

        sentence = f"Query: {query_text}\nDocument: {corpus_text}"

        async with self.semaphore:
            response, usage = await send_request(
                instructions=instruction,
                input=sentence,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ResponseSchema",
                        "schema": ResponseModel.model_json_schema(),
                        "description": "Schema for the retrieval response.",
                        "strict": True,
                    },
                },
            )

        try:
            parsed = ResponseModel.model_validate_json(response)
            output_val = float(parsed.output)
            reasoning = parsed.reasoning or ""
        except Exception as e:
            import re
            logger.warning(f"Reranking JSON parsing failed. Attempting regex extraction. Error: {e}")
            reasoning = ""
            match = re.search(r'"output"\s*:\s*(\d+)', response)
            if match:
                output_val = float(match.group(1))
            else:
                logger.error(f"Fallback regex failed for response: {response}")
                output_val = 0.0  # Default to 0 on total failure
                
        return output_val, usage, response or "", reasoning

    async def _call_async(
        self,
        search_model: SearchProtocol,
        encode_kwargs: dict[str, Any],
    ) -> RetrievalOutputType:
        self.semaphore = asyncio.Semaphore(100)
        logger.info("Running retrieval task - Indexing corpus...")

        # maps
        queries_id2idx = {qid: i for i, qid in enumerate(self.queries["id"])}
        corpus_id2idx = {cid: i for i, cid in enumerate(self.corpus["id"])}

        tasks = []
        index_map = []  # (query_id, corpus_id) for reconstructing result

        for query_id, corpus_ids in self.top_ranked.items():
            query_row = self.queries[queries_id2idx[query_id]]
            query_text = query_row["text"]

            for corpus_id in corpus_ids:
                corpus_row = self.corpus[corpus_id2idx[corpus_id]]
                corpus_text = corpus_row["text"]
                if "title" in corpus_row:
                    corpus_text = corpus_row["title"] + " " + corpus_text

                tasks.append(
                    self._score_pair(
                        query_text=query_text,
                        corpus_text=corpus_text,
                        instruction=self.instruction,
                    )
                )
                index_map.append((query_id, corpus_id))

        # Run all tasks concurrently with tqdm progress bar
        results = await tqdm_asyncio.gather(*tasks)

        # Rebuild prediction dict
        predictions: RetrievalOutputType = {}
        enriched_results = {} # qid -> {cid -> (score, raw, reasoning)}
        total_usage = {
            "input_tokens":    0,
            "cached_tokens":   0,
            "output_tokens":   0,
            "thinking_tokens": 0,
            "total_tokens":    0,
            "cost":            0.0,
        }
        
        for (query_id, corpus_id), (score, usage, raw, reasoning) in zip(index_map, results):
            predictions.setdefault(query_id, {})
            predictions[query_id][corpus_id] = score
            
            enriched_results.setdefault(query_id, {})
            enriched_results[query_id][corpus_id] = (score, raw, reasoning)
            
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

        self.usage_stats = total_usage
        self.enriched_results = enriched_results
        return predictions

    def __call__(  # type: ignore[override]
        self,
        search_model: SearchProtocol,
        encode_kwargs: dict[str, Any],
    ) -> RetrievalOutputType:
        preds = asyncio.run(
            self._call_async(
                search_model=search_model,
                encode_kwargs=encode_kwargs,
            )
        )
        return preds, self.usage_stats


class LLMRetrievalAbsTask(AbsTaskRetrieval):
    instruction: str

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> ScoresDict:
        data_split["relevant_docs"], data_split["queries"] = (
            _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )
        )
        retriever = LLMRetrievalEvaluator(
            corpus=data_split["corpus"],
            queries=data_split["queries"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_ranked=data_split["top_ranked"],
            top_k=self._top_k,
            instruction=self.instruction,
            **kwargs,
        )

        start_time = time()
        results, usage_stats = retriever(
            model,
            encode_kwargs=encode_kwargs,
        )
        end_time = time()
        logger.debug(
            f"Running retrieval task - Time taken to retrieve: {end_time - start_time:.2f} seconds"
        )

        if prediction_folder:
            self._save_task_predictions(
                results,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )
            
        try:
            import json
            import os
            from llm_judge.llm_client import settings
            
            qids = list(results.keys())
            sample_size = min(5, len(qids))
            samples = []
            
            for i in range(sample_size):
                qid = qids[i]
                cids = list(results[qid].keys())
                if not cids: continue
                cid = cids[0] # Just take the first doc for simplicity
                score, raw, reasoning = retriever.enriched_results[qid][cid]
                
                # Check if it was supposed to be relevant
                is_relevant = cid in data_split["relevant_docs"].get(qid, {})
                
                samples.append({
                    "query": data_split["queries"][qid],
                    "document_id": cid,
                    "model_raw_output": raw,
                    "reasoning": reasoning,
                    "predicted_relevance": int(score),
                    "is_gold_relevant": is_relevant,
                    "is_correct": bool(int(score) == int(is_relevant))
                })
            
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                task_name = self.metadata.name
            except Exception:
                task_name = "reranking_task"
                
            sample_path = os.path.join(output_dir, f"{task_name}_samples.json")
            
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} reranking response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        logger.info("Running retrieval task - Evaluating retrieval scores...")
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
        ) = retriever.evaluate(
            data_split["relevant_docs"],
            results,
            self.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
            skip_first_result=self.skip_first_result,
        )
        task_specific_scores = self.task_specific_scores(
            all_scores,
            data_split["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        logger.info("Running retrieval task - Finished.")
        res = make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores,
            self._previous_results_model_meta,
        )
        res["usage_stats"] = usage_stats
        return res


class LLMHUMECore17InstructionReranking(
    HUMECore17InstructionReranking, LLMRetrievalAbsTask
):
    instruction = (
        "You're assistant for information retrieval task. "
        "Given a query and a document, determine if the document is relevant to the query. "
        "Respond with 1 if the document is relevant to the query, otherwise respond with 0. "
        "You should output json with the field 'output' indicating your relevance judgment."
        "Example:\nInput: 'Query: What is the capital of France? Document: Paris is the capital city of France.'\n"
        "Output: {'reasoning': 'The document directly answers the query by stating that Paris is the capital of France.', 'output': 1}"
    )


class LLMHUMENews21InstructionReranking(
    HUMENews21InstructionReranking, LLMRetrievalAbsTask
):
    instruction = (
        "You're assistant for information retrieval task. "
        "Given a query and a document, determine if the document is relevant to the query. "
        "Respond with 1 if the document is relevant to the query, otherwise respond with 0. "
        "You should output json with the field 'output' indicating your relevance judgment."
        "Example:\nInput: 'Query: What is the capital of France? Document: Paris is the capital city of France.'\n"
        "Output: {'reasoning': 'The document directly answers the query by stating that Paris is the capital of France.', 'output': 1}"
    )


class LLMHUMERobust04InstructionReranking(
    HUMERobust04InstructionReranking, LLMRetrievalAbsTask
):
    instruction = (
        "You're assistant for information retrieval task. "
        "Given a query and a document, determine if the document is relevant to the query. "
        "Respond with 1 if the document is relevant to the query, otherwise respond with 0. "
        "You should output json with the field 'output' indicating your relevance judgment."
        "Example:\nInput: 'Query: What is the capital of France? Document: Paris is the capital city of France.'\n"
        "Output: {'reasoning': 'The document directly answers the query by stating that Paris is the capital of France.', 'output': 1}"
    )


class LLMHUMEWikipediaRerankingMultilingual(
    HUMEWikipediaRerankingMultilingual, LLMRetrievalAbsTask
):
    instruction = (
        "You're assistant for information retrieval task. "
        "Given a query and a document, determine if the document is relevant to the query. "
        "Respond with 1 if the document is relevant to the query, otherwise respond with 0. "
        "You should output json with the field 'output' indicating your relevance judgment."
        "Example:\nInput: 'Query: What is the capital of France? Document: Paris is the capital city of France.'\n"
        "Output: {'reasoning': 'The document directly answers the query by stating that Paris is the capital of France.', 'output': 1}"
    )
