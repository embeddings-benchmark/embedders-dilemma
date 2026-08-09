import asyncio
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from mteb import EncoderProtocol
from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.sts import STSMetrics
from mteb.tasks import (
    # HUME tasks (kept as secondary baselines)
    HUMESICKR,
    HUMESTS12,
    HUMESTS22,
    HUMESTSBenchmark,
    # DATASETS.md tasks
    BiossesSTS,
    STS12STS,
    STS16STS,
    STSBenchmarkSTS,
    SickrSTS,
)
from pydantic import BaseModel, Field
from scipy import stats
from tqdm.asyncio import tqdm_asyncio

from llm_judge.llm_client import send_request


class AbsTaskLLMSTS(AbsTaskSTS):
    instruction: str

    async def _score_pair(
        self,
        sentence1: str,
        sentence2: str,
        response_model: type[BaseModel],
    ) -> tuple[float, dict, str, str]:
        """Score a single STS pair via LLM."""
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
                        "description": "Schema for the sts response.",
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
            logger.warning(f"STS JSON parsing failed. Attempting regex extraction. Error: {e}")
            reasoning = ""
            match = re.search(r'"output"\s*:\s*([\d.]+)', response) if response else None
            if match:
                output_val = float(match.group(1))
            else:
                logger.error(f"Fallback regex failed for response: {response}")
                output_val = 0.0  # Default to 0 on total failure
                
        return output_val, usage, response or "", reasoning

    async def _evaluate_subset_async(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None,
        **kwargs: Any,
    ) -> STSMetrics:
        self._semaphore = asyncio.Semaphore(10)

        gold_scores = data_split["score"]

        class ResponseModel(BaseModel):
            reasoning: str | None = Field(None)
            output: float = Field(
                description="The similarity score as a float.",
                ge=self.min_score,
                le=self.max_score,
            )

        tasks = []
        for sentence1, sentence2 in zip(
            data_split[self.column_names[0]], data_split[self.column_names[1]]
        ):
            tasks.append(self._score_pair(sentence1, sentence2, ResponseModel))

        predictions_with_usage = await tqdm_asyncio.gather(*tasks)

        logger = logging.getLogger(__name__)

        predictions = []
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
        for p, usage, raw, reasoning in predictions_with_usage:
            predictions.append(p)
            enriched_results.append((p, raw, reasoning))
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

        try:
            import json
            import os
            from llm_judge.llm_client import settings
            
            sample_size = min(5, len(data_split))
            samples = []
            
            for i in range(sample_size):
                p_sim, raw, reasoning = enriched_results[i]
                samples.append({
                    "sentence1": data_split[self.column_names[0]][i],
                    "sentence2": data_split[self.column_names[1]][i],
                    "model_raw_output": raw,
                    "reasoning": reasoning,
                    "predicted_sim": p_sim,
                    "gold_score": gold_scores[i],
                    "abs_diff": abs(p_sim - gold_scores[i])
                })
            
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                task_name = self.metadata.name
                sample_path = os.path.join(output_dir, f"{task_name}_{hf_subset}_samples.json")
            except Exception:
                sample_path = os.path.join(output_dir, f"sts_task_{hf_subset}_samples.json")
            
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {sample_size} STS response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        if prediction_folder:
            self._save_task_predictions(
                predictions,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return {
            "cosine_spearman": float(stats.spearmanr(gold_scores, predictions)[0]),
            "cosine_pearson": float(stats.pearsonr(gold_scores, predictions)[0]),
            "usage_stats": total_usage,
        }

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> STSMetrics:
        return asyncio.run(
            self._evaluate_subset_async(
                model=model,
                data_split=data_split,
                encode_kwargs=encode_kwargs,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prediction_folder=prediction_folder,
                **kwargs,
            )
        )


# ===================================================================
# HUME tasks (kept as secondary baselines)
# ===================================================================

class LLMHUMESTS22(HUMESTS22, AbsTaskLLMSTS):
    instruction = (
        "You are an expert in semantic textual similarity. Your task is to rate how similar two sentences "
        "are in meaning on a scale from 1 to 4.\n\n"
        "Scoring rubric:\n"
        "1 — Completely unrelated: The sentences have no semantic overlap whatsoever.\n"
        "2 — Somewhat similar: The sentences share some broad concepts but differ in details.\n"
        "3 — Mostly similar: The sentences convey largely the same meaning with minor differences.\n"
        "4 — Semantically identical: The sentences mean the same thing.\n\n"
        "Important: Focus on MEANING, not surface word overlap.\n\n"
        "Output json with fields 'reasoning' and 'output' (floating point 1.0-4.0). "
        "Decimals like 3.5 are encouraged for subtle differences."
    )

class LLMHUMESTSBenchmark(HUMESTSBenchmark, AbsTaskLLMSTS):
    instruction = (
        "You are an expert in semantic textual similarity. Your task is to rate how similar two sentences "
        "are in meaning on a scale from 0 to 5.\n\n"
        "Scoring rubric:\n"
        "0 — Completely unrelated: The sentences have no semantic overlap.\n"
        "1 — Not similar: The sentences are on different topics or say opposite things.\n"
        "2 — Slightly similar: The sentences share a broad topic but have different meanings.\n"
        "3 — Moderately similar: The sentences share key concepts but differ in details or scope.\n"
        "4 — Very similar: The sentences convey nearly the same meaning with minor differences.\n"
        "5 — Semantically identical: The sentences mean exactly the same thing.\n\n"
        "Important: Focus on MEANING, not surface word overlap. Paraphrases score high; "
        "sentences that share words but differ in meaning should score low.\n\n"
        "Output json with fields 'reasoning' and 'output' (floating point 0.0-5.0). "
        "Decimals like 3.5 are encouraged for subtle differences."
    )


class LLMHUMESTS12(HUMESTS12, AbsTaskLLMSTS):
    instruction = (
        "You are an expert in semantic textual similarity. Your task is to rate how similar two sentences "
        "are in meaning on a scale from 0 to 5.\n\n"
        "Scoring rubric:\n"
        "0 — Completely unrelated: The sentences have no semantic overlap.\n"
        "1 — Not similar: The sentences are on different topics or say opposite things.\n"
        "2 — Slightly similar: The sentences share a broad topic but have different meanings.\n"
        "3 — Moderately similar: The sentences share key concepts but differ in details or scope.\n"
        "4 — Very similar: The sentences convey nearly the same meaning with minor differences.\n"
        "5 — Semantically identical: The sentences mean exactly the same thing.\n\n"
        "Output json with fields 'reasoning' and 'output' (floating point 0.0-5.0)."
    )

class LLMHUMESICKR(HUMESICKR, AbsTaskLLMSTS):
    instruction = (
        "You are an expert in semantic textual similarity. Your task is to rate how semantically "
        "related two sentences are on a scale from 1 to 5.\n\n"
        "Scoring rubric:\n"
        "1 — Completely unrelated: The sentences have no semantic connection.\n"
        "2 — Slightly related: The sentences share a very loose topical connection.\n"
        "3 — Moderately related: The sentences share some concepts or describe related situations.\n"
        "4 — Very related: The sentences describe very similar situations with minor differences.\n"
        "5 — Highly related / equivalent: The sentences describe the same situation or mean the same thing.\n\n"
        "Note: This scale starts at 1 (not 0). The sentences come from image captions "
        "so they often describe visual scenes.\n\n"
        "Output json with fields 'reasoning' and 'output' (floating point 1.0-5.0). "
        "Decimals like 3.5 are encouraged for subtle differences."
    )


# ===================================================================
# Premium STS Tasks (9)
# ===================================================================

class LLMSTSBenchmark(AbsTaskLLMSTS, STSBenchmarkSTS):
    metadata = STSBenchmarkSTS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-stsbenchmark", "revision": "86bbaf4470f501ee381411836b3a22f112bfe42a"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

class LLMSICKR(AbsTaskLLMSTS, SickrSTS):
    metadata = SickrSTS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sickr", "revision": "82eb9939fa177ddd94d5ddf4668e011d7446c1da7"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic relatedness of two sentences on a scale from 1 to 5, "
        "where 1 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 1.0-5.0). "
        "Decimals are encouraged."
    )

class LLMSTS12(AbsTaskLLMSTS, STS12STS):
    metadata = STS12STS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts12", "revision": "1559a6e5259e61f06ab1f4ef1c71a921e4877dd1"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

from mteb.tasks import (
    STS13STS, 
    STS14STS, 
    STS15STS, 
    STS17Crosslingual, 
    STS22CrosslingualSTSv2
)

class LLMSTS13(AbsTaskLLMSTS, STS13STS):
    metadata = STS13STS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts13", "revision": "e0251ba4e151f033f74bd5543c5d32706e810eab"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

class LLMSTS14(AbsTaskLLMSTS, STS14STS):
    metadata = STS14STS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts14", "revision": "6237db5b2c4be5e7d4aacad8e850ba8d550093d6"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

class LLMSTS15(AbsTaskLLMSTS, STS15STS):
    metadata = STS15STS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts15", "revision": "6283e03ad9bd28cb1ed3d3a472d08a0e9df58fc1"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

class LLMSTS16(AbsTaskLLMSTS, STS16STS):
    metadata = STS16STS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts16", "revision": "69862fce58ef5e5e9fdfa7c8b321227cc63961ea"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )

class LLMBIOSSES(AbsTaskLLMSTS, BiossesSTS):
    metadata = BiossesSTS.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-biosses", "revision": "cf968edb41fa17a96392f6b373819efac1c2d6d6"},
        "eval_splits": ["test"],
    })
    instruction = (
        "Rate the semantic similarity of two biomedical sentences on a scale from 0 to 4, "
        "where 0 means completely unrelated and 4 means equivalent meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-4.0). "
        "Decimals are encouraged."
    )

class LLMSTS17(AbsTaskLLMSTS, STS17Crosslingual):
    fast_loading = False
    metadata = STS17Crosslingual.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts17", "revision": "fe4f4e1b9fdafeae22df69e66bdc3b634be30e9d"},
        "eval_splits": ["test"],
        "eval_langs": {lang: STS17Crosslingual.metadata.eval_langs[lang] for lang in ["en-en", "en-de", "es-es", "fr-en", "it-en"]},
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 0 to 5, "
        "where 0 means completely unrelated and 5 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 0.0-5.0). "
        "Decimals are encouraged."
    )
class LLMSTS22v2(AbsTaskLLMSTS, STS22CrosslingualSTSv2):
    fast_loading = False
    metadata = STS22CrosslingualSTSv2.metadata.model_copy(update={
        "dataset": {"path": "mteb/llm-eval-sts22_v2", "revision": "7a79fd41b024091522d04e84d8d9dc93d223cf8c"},
        "eval_splits": ["test"],
        "eval_langs": {lang: STS22CrosslingualSTSv2.metadata.eval_langs[lang] for lang in ["en", "de", "es", "fr", "ru", "zh"]},
    })
    instruction = (
        "Rate the semantic similarity of two sentences on a scale from 1 to 4, "
        "where 1 means completely unrelated and 4 means identical meaning. "
        "Output json with fields 'reasoning' and 'output' (float 1.0-4.0). "
        "Decimals are encouraged."
    )
