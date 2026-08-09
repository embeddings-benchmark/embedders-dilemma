import asyncio
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from mteb import EncoderProtocol
from mteb.abstasks import AbsTaskPairClassification
from mteb.tasks import (
    LegalBenchPC,
    SprintDuplicateQuestionsPC,
    TwitterURLCorpusPC,
    RTE3,
)
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from llm_judge.llm_client import send_request


class AbsTaskLLMPairClassification(AbsTaskPairClassification):
    instruction: str

    async def _score_pair(
        self,
        sentence1: str,
        sentence2: str,
        response_model: type[BaseModel],
    ) -> tuple[float, dict, str, str]:
        """Score a single pair via LLM. Returns a binary similarity score (0 or 1) and usage."""
        sentence = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}"

        async with self._semaphore:
            response, usage = await send_request(
                instructions=self.instruction,
                input=sentence,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ResponseSchema",
                        "schema": response_model.model_json_schema(),
                        "description": "Schema for the pair classification response.",
                        "strict": True,
                    },
                },
            )

        try:
            parsed = response_model.model_validate_json(response)
            output_val = float(parsed.output)
            reasoning = parsed.reasoning or ""
        except Exception as e:
            import re
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Pair Classification JSON parsing failed. Attempting regex extraction. Error: {e}")
            reasoning = ""
            match = re.search(r'"output"\s*:\s*(\d+)', response)
            if match:
                output_val = float(match.group(1))
            else:
                logger.error(f"Fallback regex failed for response: {response}")
                output_val = 0.0  # Default to 0 (no match/duplicate) on total failure
                
        return output_val, usage, response or "", reasoning

    async def _evaluate_subset_async(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self._semaphore = asyncio.Semaphore(10)

        class ResponseModel(BaseModel):
            reasoning: str | None = Field(None)
            output: int = Field(ge=0, le=1)

        if self.metadata.modalities == ["text"]:
            data_split = (
                Dataset.from_dict(data_split[0]) if len(data_split) == 1 else data_split
            )

        tasks = []
        for sentence1, sentence2 in zip(
            data_split[self.input1_column_name], data_split[self.input2_column_name]
        ):
            tasks.append(self._score_pair(sentence1, sentence2, ResponseModel))

        results_with_usage = await tqdm_asyncio.gather(*tasks)

        logger = logging.getLogger(__name__)

        similarity_scores = []
        enriched_results = []
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
        for s, usage, raw, reasoning in results_with_usage:
            similarity_scores.append(s)
            enriched_results.append((s, raw, reasoning))
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

        try:
            import json
            import os
            from llm_judge.llm_client import settings
            
            sample_size = min(5, len(data_split))
            samples = []
            
            for i in range(sample_size):
                p_score, raw, reasoning = enriched_results[i]
                gold = data_split[self.label_column_name][i]
                samples.append({
                    "sentence1": data_split[self.input1_column_name][i],
                    "sentence2": data_split[self.input2_column_name][i],
                    "model_raw_output": raw,
                    "reasoning": reasoning,
                    "predicted_label": int(p_score),
                    "gold_label": int(gold),
                    "is_correct": bool(int(p_score) == int(gold))
                })
            
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                task_name = self.metadata.name
                sample_path = os.path.join(output_dir, f"{task_name}_{hf_subset}_samples.json")
            except Exception:
                sample_path = os.path.join(output_dir, f"pair_cls_task_{hf_subset}_samples.json")
            
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} pair classification response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        if prediction_folder:
            self._save_task_predictions(
                similarity_scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        # Compute metrics directly — can't use parent's _compute_metrics
        # because it expects a dict of distance types (cosine, manhattan, etc.)
        # but we have a flat list of LLM predictions.
        import numpy as np
        from sklearn.metrics import average_precision_score

        np_labels = np.asarray(
            data_split[self.label_column_name], dtype=np.int64
        )
        np_scores = np.asarray(similarity_scores, dtype=np.float64)

        # AP: how well the score ranking separates positives from negatives
        ap = float(average_precision_score(np_labels, np_scores))

        # Best accuracy/F1 at threshold (with binary 0/1 scores there's only one threshold)
        metrics = self._compute_metrics_values(
            list(np_scores), np_labels, high_score_more_similar=True
        )

        return {
            "similarity_ap": ap,
            "similarity_accuracy": metrics["accuracy"],
            "similarity_f1": metrics["f1"],
            "max_ap": ap,
            "max_accuracy": metrics["accuracy"],
            "max_f1": metrics["f1"],
            "max_precision": metrics["precision"],
            "max_recall": metrics["recall"],
            "usage_stats": total_usage,
        }

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return asyncio.run(
            self._evaluate_subset_async(
                model=model,
                data_split=data_split,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                num_proc=num_proc,
                **kwargs,
            )
        )


# ===================================================================
# Pair Classification tasks (Primary Benchmark)
# ===================================================================

class LLMSprintDuplicateQuestionsPC(AbsTaskLLMPairClassification, SprintDuplicateQuestionsPC):
    metadata = SprintDuplicateQuestionsPC.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sprint_duplicate_questions", "revision": "d1c6be04a5f84b606b758024ed9d3b3fb7e4029a"},
        "eval_splits": ["test"],
    })
    label_column_name = "label"
    instruction = (
        "Determine whether two questions are duplicates (asking for the same information). "
        "Output json with fields 'reasoning' and 'output' (1 if duplicate, 0 if not)."
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        pass


class LLMTwitterURLCorpusPC(AbsTaskLLMPairClassification, TwitterURLCorpusPC):
    metadata = TwitterURLCorpusPC.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-twitter_url_corpus", "revision": "741fbe5cf43a594d8c37073d66baf8ab879281ce"},
        "eval_splits": ["test"],
    })
    label_column_name = "label"
    instruction = (
        "Determine whether two tweets convey the same message (are paraphrases of each other). "
        "Output json with fields 'reasoning' and 'output' (1 if paraphrase, 0 if not)."
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        pass


class LLMLegalBenchPC(AbsTaskLLMPairClassification, LegalBenchPC):
    metadata = LegalBenchPC.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-legal_bench_pc", "revision": "a0217dc60ec5a45077e213a9538e845533523ed0"},
        "eval_splits": ["test"],
    })
    label_column_name = "label"
    instruction = (
        "Determine whether two legal text snippets form a matching pair (yes/correct/relevant = 1, no/incorrect/irrelevant = 0). "
        "Output json with fields 'reasoning' and 'output' (1 or 0)."
    )


class LLMRTE3PC(AbsTaskLLMPairClassification, RTE3):
    fast_loading = False
    metadata = RTE3.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-rte3", "revision": "5d745bd9e435cc190599f916f54c0ab197ddeb73"},
        "eval_splits": ["test"],
        "eval_langs": {lang: RTE3.metadata.eval_langs[lang] for lang in ["de", "en", "fr", "it"]},
    })
    label_column_name = "label"
    instruction = (
        "Determine whether the hypothesis is semantically entailed by the premise (i.e., the meaning of the hypothesis follows from the premise). "
        "Output json with fields 'reasoning' and 'output' (1 if entailment, 0 if not)."
    )