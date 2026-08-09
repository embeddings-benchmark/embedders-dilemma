"""Decoder-LLM cross-encoder rerankers, for models that score a (query, doc)
pair by reading a yes/no-style token logit at the final prompt position rather
than via a BERT classification head. `sentence_transformers.CrossEncoder` loads
these as a SequenceClassification model with a randomly-initialised head (garbage
scores), so we score them directly here.

Compatible with mteb's CE path: exposes `.predict()`, `.tokenizer`, `.model`,
`.mteb_model_meta` so mteb's SearchCrossEncoderWrapper treats it transparently.

Supported (see GENERATIVE_CE_PRESETS):
  * mixedbread-ai/mxbai-rerank-{base,large}-v2 — chat template, logit("1")-logit("0")
  * Qwen/Qwen3-Reranker-{0.6B,4B,8B} — manual prompt w/ empty <think>, logit("yes")-logit("no")
  * BAAI/bge-reranker-v2-gemma — instruction-after-passage prompt, logit("Yes") alone

Add a family by adding a GenerativeCEPreset to GENERATIVE_CE_PRESETS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from mteb.models.model_meta import ModelMeta
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _make_generative_ce_meta(model_name: str) -> ModelMeta:
    """Mteb's `evaluate()` inspects `model.mteb_model_meta` to skip its own load
    path. Without this, mteb sees a placeholder ModelMeta ("no_model_name/available")
    and tries to re-load the model via SentenceTransformer, which fails.
    """
    return ModelMeta(
        loader=None,
        name=model_name,
        revision=model_name,  # not strictly meaningful for our custom path
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


@dataclass(frozen=True)
class GenerativeCEPreset:
    """Per-model-family scoring config.

    Exactly one of `build_messages` / `build_prompt` is used:
      * `build_messages(query, doc) -> list[dict]`: messages fed through the
        tokenizer's chat template (mxbai-rerank-v2 uses this).
      * `build_prompt(query, doc) -> str`: a raw prompt string used verbatim
        (NO chat template). Qwen3-Reranker requires this because its official
        format pre-fills an empty `<think>\\n\\n</think>` block in the suffix,
        which apply_chat_template would not reproduce.

    `positive_token` / `negative_token` surface strings → token ids (resolved at
    init via convert_tokens_to_ids); score = logit(pos) - logit(neg) at the final
    position, monotonic with the model card's softmax-prob-of-yes (same ranking).
    """
    positive_token: str
    negative_token: str
    build_messages: callable | None = None
    build_prompt: callable | None = None
    # For build_prompt presets whose format ends in a fixed suffix that must be
    # preserved under truncation (e.g. Qwen3-Reranker's "...<think>\n\n</think>\n\n"),
    # set prefix/suffix so the wrapper truncates only the middle (the document).
    prompt_prefix: str | None = None
    prompt_suffix: str | None = None
    build_body: callable | None = None  # (query, doc) -> str (goes between prefix/suffix)
    # "difference" → logit(pos) - logit(neg) (Qwen3-Reranker, mxbai).
    # "positive"   → logit(pos) alone (bge-reranker-v2-gemma model-card behavior).
    score_mode: str = "difference"
    # Whether the tokenizer should add special tokens (e.g. Gemma <bos>) to a
    # build_prompt string. False for Qwen3/mxbai (they embed their own control tokens).
    add_special_tokens: bool = False


def _mxbai_v2_messages(query: str, doc: str) -> list[dict]:
    # mxbai-rerank-large-v2 chat_template.jinja iterates over messages and pulls
    # roles "query" + "document". The system prompt is fixed in the template.
    return [
        {"role": "query", "content": query},
        {"role": "document", "content": doc},
    ]


# Qwen3-Reranker official format (from model card). Manual prefix/suffix — the
# suffix closes an empty <think> block so the model emits yes/no immediately.
_QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_RERANK_INSTRUCT = "Given a query, retrieve relevant passages that help answer the query"


def _qwen3_reranker_prompt(query: str, doc: str) -> str:
    return _QWEN3_RERANK_PREFIX + _qwen3_reranker_body(query, doc) + _QWEN3_RERANK_SUFFIX


def _qwen3_reranker_body(query: str, doc: str) -> str:
    return (
        f"<Instruct>: {_QWEN3_RERANK_INSTRUCT}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


_QWEN3_RR = dict(
    positive_token="yes", negative_token="no",
    build_prompt=_qwen3_reranker_prompt,
    prompt_prefix=_QWEN3_RERANK_PREFIX,
    prompt_suffix=_QWEN3_RERANK_SUFFIX,
    build_body=_qwen3_reranker_body,
)


# bge-reranker-v2-gemma (BAAI) — Gemma-based LLM reranker. Single forward pass;
# instruction comes AFTER query+passage; score = logit("Yes") alone (model card).
_BGE_GEMMA_PROMPT = (
    "Given a query A and a passage B, determine whether the passage contains an "
    "answer to the query by providing a prediction of either 'Yes' or 'No'."
)
def _bge_gemma_prompt(query: str, doc: str) -> str:
    return f"A: {query}\nB: {doc}\n{_BGE_GEMMA_PROMPT}"


GENERATIVE_CE_PRESETS: dict[str, GenerativeCEPreset] = {
    "mixedbread-ai/mxbai-rerank-large-v2": GenerativeCEPreset(
        positive_token="1", negative_token="0", build_messages=_mxbai_v2_messages,
    ),
    "mixedbread-ai/mxbai-rerank-base-v2": GenerativeCEPreset(
        positive_token="1", negative_token="0", build_messages=_mxbai_v2_messages,
    ),
    "Qwen/Qwen3-Reranker-0.6B": GenerativeCEPreset(**_QWEN3_RR),
    "Qwen/Qwen3-Reranker-4B": GenerativeCEPreset(**_QWEN3_RR),
    "Qwen/Qwen3-Reranker-8B": GenerativeCEPreset(**_QWEN3_RR),
    "BAAI/bge-reranker-v2-gemma": GenerativeCEPreset(
        positive_token="Yes", negative_token="No",
        build_prompt=_bge_gemma_prompt, score_mode="positive",
        add_special_tokens=True,  # Gemma needs <bos>
    ),
}


class GenerativeCrossEncoder:
    """Drop-in replacement for `sentence_transformers.CrossEncoder` for decoder-LLM rerankers."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        max_length: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = "flash_attention_2",
        preset: GenerativeCEPreset | None = None,
    ):
        if preset is None:
            if model_name_or_path not in GENERATIVE_CE_PRESETS:
                raise ValueError(
                    f"No preset for {model_name_or_path!r}. Pass `preset=GenerativeCEPreset(...)` "
                    f"or add to GENERATIVE_CE_PRESETS in {__name__}."
                )
            preset = GENERATIVE_CE_PRESETS[model_name_or_path]
        self.preset = preset
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # mteb's evaluate() reads this attribute to skip its own SentenceTransformer
        # load path — without it, mteb tries to re-load via "no_model_name/available".
        self.mteb_model_meta = _make_generative_ce_meta(model_name_or_path)
        self.model_name_or_path = model_name_or_path

        logger.info(
            "Loading GenerativeCrossEncoder %s (dtype=%s, attn=%s, max_len=%d)",
            model_name_or_path,
            dtype,
            attn_implementation,
            max_length,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        kw: dict = {"trust_remote_code": True, "dtype": dtype}
        if attn_implementation is not None:
            kw["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kw)
        self.model.to(self.device)
        self.model.eval()

        # Resolve scoring token IDs. Prefer convert_tokens_to_ids (matches the
        # model cards exactly, e.g. Qwen3-Reranker uses ids of "yes"/"no"); fall
        # back to tokenizing if that yields <unk>.
        unk = self.tokenizer.unk_token_id
        pos_id = self.tokenizer.convert_tokens_to_ids(preset.positive_token)
        neg_id = self.tokenizer.convert_tokens_to_ids(preset.negative_token)
        if pos_id is None or pos_id == unk:
            pos_id = self.tokenizer(preset.positive_token, add_special_tokens=False)["input_ids"][0]
        if neg_id is None or neg_id == unk:
            neg_id = self.tokenizer(preset.negative_token, add_special_tokens=False)["input_ids"][0]
        self._pos_token_id = pos_id
        self._neg_token_id = neg_id
        logger.info(
            "Scoring tokens: %r → id %d, %r → id %d",
            preset.positive_token, self._pos_token_id,
            preset.negative_token, self._neg_token_id,
        )

    @torch.inference_mode()
    def predict(
        self,
        sentence_pairs: list[tuple[str, str]] | None = None,
        batch_size: int = 8,
        show_progress_bar: bool = True,
        *,
        inputs1=None,
        inputs2=None,
        **_kwargs,
    ) -> np.ndarray:
        """Score `(query, doc)` pairs.

        Accepts two calling conventions:
          * sentence-transformers style: `predict(sentence_pairs=[(q, d), ...])`
          * mteb's CrossEncoderWrapper style: `predict(inputs1=..., inputs2=...)`
            where inputs1/inputs2 are DataLoaders of batches with a "text" key.

        Returns float scores (one per pair): logit(pos)-logit(neg), or logit(pos)
        alone when the preset uses score_mode="positive", read at the final token.
        """
        if sentence_pairs is None:
            if inputs1 is None or inputs2 is None:
                raise ValueError(
                    "GenerativeCrossEncoder.predict requires either `sentence_pairs` or "
                    "(`inputs1`, `inputs2`) DataLoaders."
                )
            queries = [text for batch in inputs1 for text in batch["text"]]
            docs = [text for batch in inputs2 for text in batch["text"]]
            sentence_pairs = list(zip(queries, docs))
        scores = np.zeros(len(sentence_pairs), dtype=np.float32)

        # Build input token-id lists upfront. Two modes:
        #  (A) suffix-preserving (Qwen3-Reranker): tokenize prefix+body+suffix
        #      separately and truncate ONLY the body so the yes/no read position
        #      (end of suffix) is never chopped.
        #  (B) plain: build the full prompt string, tokenize with right-truncation.
        suffix_preserving = (
            self.preset.prompt_prefix is not None
            and self.preset.prompt_suffix is not None
            and self.preset.build_body is not None
        )
        token_id_lists: list[list[int]] = []
        if suffix_preserving:
            prefix_ids = self.tokenizer(self.preset.prompt_prefix, add_special_tokens=False)["input_ids"]
            suffix_ids = self.tokenizer(self.preset.prompt_suffix, add_special_tokens=False)["input_ids"]
            body_budget = self.max_length - len(prefix_ids) - len(suffix_ids)
            if body_budget <= 0:
                raise ValueError(
                    f"max_length={self.max_length} too small for prefix+suffix "
                    f"({len(prefix_ids)}+{len(suffix_ids)} tokens)."
                )
            for q, d in sentence_pairs:
                body_ids = self.tokenizer(self.preset.build_body(q, d), add_special_tokens=False)["input_ids"]
                body_ids = body_ids[:body_budget]  # keep head of body (doc tail dropped)
                token_id_lists.append(prefix_ids + body_ids + suffix_ids)
        else:
            # add_special_tokens: chat-template / Qwen3 prompts embed their own
            # control tokens, so no extra BOS. bge-reranker-v2-gemma's raw prompt
            # has no control tokens and Gemma relies on <bos> → add_special_tokens=True.
            add_special = getattr(self.preset, "add_special_tokens", False)
            for q, d in sentence_pairs:
                if self.preset.build_prompt is not None:
                    prompt = self.preset.build_prompt(q, d)
                else:
                    messages = self.preset.build_messages(q, d)
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                ids = self.tokenizer(prompt, add_special_tokens=add_special, truncation=True, max_length=self.max_length)["input_ids"]
                token_id_lists.append(ids)

        iterator = range(0, len(token_id_lists), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="GenCE", leave=False)

        # LEFT-pad so the last real token is always the final position. This lets
        # us pass logits_to_keep=1 to the model — it then materialises logits only
        # for that final position instead of [batch, seq, vocab]. Critical for
        # large-vocab rerankers (e.g. Gemma's 256K vocab → a full-seq logits tensor
        # is 8×8192×256000×2 ≈ 34 GB and OOMs; last-position-only is ~4 MB).
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in iterator:
                chunk = token_id_lists[start : start + batch_size]
                enc = self.tokenizer.pad(
                    {"input_ids": chunk}, return_tensors="pt", padding=True,
                ).to(self.device)
                out = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    use_cache=False,
                    logits_to_keep=1,  # only the final position
                )
                # logits: [batch, 1, vocab] → final position for every row (left-padded).
                last_logits = out.logits[:, -1, :]
                pos = last_logits[:, self._pos_token_id]
                if getattr(self.preset, "score_mode", "difference") == "positive":
                    score = pos.float().cpu().numpy()
                else:
                    neg = last_logits[:, self._neg_token_id]
                    score = (pos - neg).float().cpu().numpy()
                scores[start : start + batch_size] = score
        finally:
            self.tokenizer.padding_side = orig_padding_side
        return scores

    # Some mteb code paths inspect model config / device. Provide common attrs.
    @property
    def config(self):  # noqa: D401
        return self.model.config
