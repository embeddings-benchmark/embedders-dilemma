import asyncio
from typing import Any, get_args

import numpy as np
from pydantic import BaseModel
from mteb import EncoderProtocol
from mteb._evaluators import SklearnEvaluator
from tqdm.asyncio import tqdm_asyncio

from llm_judge.llm_client import send_request

import logging

logger = logging.getLogger(__name__)

# Number of sentences per LLM call. Lower = fewer tokens per call but more calls.
# 20 keeps each request well under typical token limits.
CLASSIFICATION_BATCH_SIZE = 1


def _valid_labels(response_model: type[BaseModel]) -> list[str]:
    """Extract the valid Literal string values from the response model's 'output' field."""
    field = response_model.model_fields.get("output")
    if field is None:
        return []
    annotation = field.annotation
    # Unwrap Optional if present
    args = get_args(annotation)
    if args:
        # For Literal["a", "b"] the args are the valid values
        return [a for a in args if isinstance(a, str)]
    return []

class LLMClassificationEvaluator(SklearnEvaluator):
    def __init__(self, *args, **kwargs):
        self.llm_evaluator_model = kwargs.get("evaluator_model")
        self.hf_subset = kwargs.get("hf_subset", "default")
        self.hf_split = kwargs.get("hf_split", "test")

        # We are the sklearn model — SklearnEvaluator calls our fit/predict
        kwargs["evaluator_model"] = self
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        return self

    def score(self, X, y):
        return 0.0

    async def _call_async(
        self,
        model: Any,
        *,
        encode_kwargs: dict,
        test_cache: np.ndarray | None = None,
    ) -> np.ndarray:
        test_sentences = self.eval_dataset[self.values_column_name]

        _empty = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                  "thinking_tokens": 0, "total_tokens": 0, "cost": 0.0}
        if not hasattr(self, "_total_usage"):
            self._total_usage = dict(_empty)

        if not hasattr(LLMClassificationEvaluator, "GLOBAL_USAGE") or LLMClassificationEvaluator.GLOBAL_USAGE is None:
            LLMClassificationEvaluator.GLOBAL_USAGE = dict(_empty)

        config_model = getattr(self, "llm_evaluator_model", None)
        if not config_model:
            config_model = self.evaluator_model
            
        async def classify(sentence: str):
            response, usage = await send_request(
                instructions=config_model.instruction,
                input=sentence,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ResponseSchema",
                        "schema": config_model.response_model.model_json_schema(),
                        "description": "Schema for the classification response.",
                        "strict": True,
                    },
                },
            )
            for k in self._total_usage:
                self._total_usage[k] += usage.get(k, 0)
                LLMClassificationEvaluator.GLOBAL_USAGE[k] += usage.get(k, 0)
            
            raw_text = response or ""
            try:
                parsed = config_model.response_model.model_validate_json(raw_text)
                return parsed.output_class, raw_text, parsed.output
            except Exception:
                # Regex fallback for non-JSON responses
                from typing import get_args
                import re
                field = config_model.response_model.model_fields.get("output")
                valid_labels = []
                if field and get_args(field.annotation):
                    valid_labels = [a for get_arg in [get_args(field.annotation)] for a in get_arg if isinstance(a, str)]

                if valid_labels:
                    pattern = r'"(' + "|".join(re.escape(l) for l in valid_labels) + r')"'
                    match = re.search(pattern, raw_text)
                    if match:
                        label = match.group(1)
                        try:
                            dummy = config_model.response_model(output=label)
                            return dummy.output_class, raw_text, label
                        except Exception:
                            try:
                                inst = config_model.response_model.model_construct(output=label)
                                label2idx = {v: k for k, v in inst.idx2label.items()}
                                return label2idx.get(label, -1), raw_text, label
                            except Exception:
                                pass
                return -1, raw_text, "PARSE_FAILURE"

        # Create all tasks at once
        tasks = [classify(sentence) for sentence in test_sentences]

        # Run them concurrently with progress bar
        results = await tqdm_asyncio.gather(*tasks)
        output_classes = [r[0] for r in results]

        try:
            import json
            import os

            test_labels = self.eval_dataset[self.label_column_name]
            dummy_inst = config_model.response_model.model_construct()
            idx2label = dummy_inst.idx2label

            sample_size = min(5, len(test_sentences))
            samples = []
            for i in range(sample_size):
                out_idx, raw, parsed_label = results[i]
                gold_idx = test_labels[i]
                gold_label = idx2label.get(gold_idx, str(gold_idx))
                
                samples.append({
                    "input_text": test_sentences[i],
                    "model_raw_output": raw,
                    "parsed_output": parsed_label,
                    "ground_truth": gold_label,
                    "is_correct": bool(out_idx == gold_idx)
                })
            
            from llm_judge.llm_client import settings
            model_name = settings.model.replace("/", "__")
            output_dir = os.path.join("llm_results", model_name, "results", "no_model_name__available", "no_revision_available")
            os.makedirs(output_dir, exist_ok=True)

            task_name = getattr(LLMClassificationEvaluator, "CURRENT_TASK_NAME", "classification_task")
            sample_path = os.path.join(output_dir, f"{task_name}_{self.hf_subset}_samples.json")
            
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=4)
            logger.info(f"Saved {sample_size} enriched classification response samples to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug samples: {e}")

        return np.array(output_classes)

    def predict(self, X) -> np.ndarray:
        return asyncio.run(
            self._call_async(model=None, encode_kwargs={})
        )

    def __call__(
        self,
        model: EncoderProtocol,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = self.predict(None)
        return y_pred, np.array([])
