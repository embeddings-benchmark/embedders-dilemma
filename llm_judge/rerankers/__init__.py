"""Custom reranker wrappers for mteb-incompatible architectures.

`sentence_transformers.CrossEncoder` assumes BERT-style encoder + linear classifier head
and loads checkpoints via `AutoModelForSequenceClassification`. For decoder-LLM
rerankers (mxbai-rerank-large-v2, Qwen3-Reranker-8B, etc.) the checkpoint is
`*ForCausalLM` and scoring is via `logit(yes_token) - logit(no_token)` after applying
a chat template. `sentence_transformers.CrossEncoder` silently random-inits the missing
classification head, producing garbage scores. This package provides a drop-in
replacement that does it correctly.
"""
from .generative_ce import GenerativeCrossEncoder, GENERATIVE_CE_PRESETS

__all__ = ["GenerativeCrossEncoder", "GENERATIVE_CE_PRESETS"]
