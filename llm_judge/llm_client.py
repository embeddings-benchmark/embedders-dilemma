import copy
import json
import asyncio
import logging
import re
import threading
from pathlib import Path

from google.auth import default
from google.auth.transport import requests as google_requests

import aiofiles
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from llm_judge.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()
client = AsyncOpenAI(api_key=settings.token, base_url=settings.base_url)


_vertex_creds = None
_vertex_project_id = None
_vertex_lock = threading.Lock()


_MD_FENCE_LEAD  = re.compile(r"^```\w*\s*\n", flags=re.DOTALL)
_MD_FENCE_TRAIL = re.compile(r"\n```\s*$", flags=re.DOTALL)


def _strip_thinking(content: str) -> tuple[str, int]:
    """Clean a model response: strip CoT prelude AND markdown code fences.

    CoT formats handled:
      - Well-formed pair: <think>...</think>actual_answer
      - Asymmetric closer only (Qwen3.6 etc.): Thinking Process:\\n...\\n</think>\\nanswer

    Markdown formats stripped (Qwen3.6 wraps JSON in ```json ... ```):
      - Leading  ```json\\n  or  ```\\n  (any language tag)
      - Trailing \\n```

    Returns (clean_content, n_thinking_chars). Caller converts n_thinking_chars
    to approximate thinking tokens (~4 chars/token) for thinking-tax tracking.
    """
    if not content:
        return content, 0
    if "</think>" in content:
        thinking, _, after = content.partition("</think>")
        clean = after.strip()
        n_think = len(thinking)
    else:
        clean = content.strip()
        n_think = 0
    # Strip leading/trailing markdown code fences if present (common with Qwen-family JSON output)
    clean = _MD_FENCE_LEAD.sub("", clean)
    clean = _MD_FENCE_TRAIL.sub("", clean).strip()
    return clean, n_think

def get_vertex_credentials():
    """Fetches fresh OAuth2 credentials for Vertex AI. Thread-safe."""
    global _vertex_creds, _vertex_project_id
    with _vertex_lock:
        if _vertex_creds is None:
            _vertex_creds, _vertex_project_id = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

        if _vertex_creds and not _vertex_creds.valid:
            auth_req = google_requests.Request()
            _vertex_creds.refresh(auth_req)

        return _vertex_creds.token if _vertex_creds else None, _vertex_project_id

# Semaphore and cache lock are recreated per event loop (asyncio primitives are
# loop-bound; each asyncio.run() creates a new loop, so we must not reuse them).
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_cache_lock: asyncio.Lock | None = None
_cache_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_event_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(settings.max_concurrency)
        _semaphore_loop = loop
    return _semaphore


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock, _cache_lock_loop
    loop = asyncio.get_event_loop()
    if _cache_lock is None or _cache_lock_loop is not loop:
        _cache_lock = asyncio.Lock()
        _cache_lock_loop = loop
    return _cache_lock


cache_path = Path("cache.json")
cache_data: dict[str, str] = {}

async def _load_cache() -> None:
    """Load cache from disk asynchronously into cache_data."""
    global cache_data
    if not cache_path.exists():
        return
    async with aiofiles.open(cache_path) as f:
        text = await f.read()
        if text.strip():
            cache_data = json.loads(text)


async def _save_cache() -> None:
    """Save cache to disk asynchronously."""
    async with aiofiles.open(cache_path, "w") as f:
        await f.write(json.dumps(cache_data, indent=4))


def _make_schema_strict(schema: dict) -> dict:
    """Recursively patch a JSON schema to satisfy OpenAI strict-mode requirements:
    - Every object gets additionalProperties: false
    - Every object's properties are all listed in required
    Safe to apply for Gemini/Groq — they accept these fields and ignore them.
    """
    schema = copy.deepcopy(schema)

    def _process(obj: dict) -> None:
        if not isinstance(obj, dict):
            return
        if obj.get("type") == "object" and "properties" in obj:
            obj["additionalProperties"] = False
            obj["required"] = list(obj["properties"].keys())
            for v in obj["properties"].values():
                _process(v)
        # Recurse into $defs
        for v in obj.get("$defs", {}).values():
            _process(v)
        # Recurse into anyOf / oneOf / allOf
        for key in ("anyOf", "oneOf", "allOf"):
            for item in obj.get(key, []):
                _process(item)

    _process(schema)
    return schema


_rlm_instance = None


def get_rlm(model_name: str):
    """Create or return a singleton RLM instance with logging and verbose output."""
    global _rlm_instance
    if _rlm_instance is None:
        from rlm import RLM
        from rlm.logger import RLMLogger

        # Refresh token if using Vertex AI, otherwise use static token
        if settings.use_vertex_ai:
            token, _ = get_vertex_credentials()
        else:
            token = settings.token

        _rlm_instance = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": model_name,
                "api_key": token,
                "base_url": settings.base_url,
            },
            max_iterations=10,
            max_errors=4,         # Stop if 4 consecutive REPL errors
            max_timeout=300.0,
            verbose=True,
            logger=RLMLogger(log_dir="rlm_results/trajectories"),  # Saves JSONL per run for post-hoc analysis
        )
        logger.info(f"[RLM] Initialized: model={model_name}, max_iter=10, timeout=300s")
    return _rlm_instance


def create_rlm_instance():
    """Create a fresh RLM instance (not cached). Thread-safe — each call gets its own state."""
    from rlm import RLM
    from rlm.logger import RLMLogger

    if settings.use_vertex_ai:
        token, _ = get_vertex_credentials()
    else:
        token = settings.token

    return RLM(
        backend="openai",
        backend_kwargs={
            "model_name": settings.model,
            "api_key": token,
            "base_url": settings.base_url,
        },
        max_iterations=10,
        max_errors=4,
        max_timeout=300.0,
        verbose=True,
        logger=RLMLogger(log_dir="rlm_results/trajectories"),
    )


def _log_rlm_usage(result) -> dict:
    """Log token usage, cost, and timing from an RLMChatCompletion.
    
    Returns a usage dict with both standard fields (input/output/total/cost)
    and RLM-specific fields (rlm_total_calls, rlm_wall_time_s) so they flow
    into every task's usage_stats in the result JSON.
    """
    usage = result.usage_summary
    in_tok = getattr(usage, "total_input_tokens", 0) or 0
    out_tok = getattr(usage, "total_output_tokens", 0) or 0
    cost = getattr(usage, "total_cost", None)
    wall_time = getattr(result, "execution_time", 0.0) or 0.0

    # Count total sub-LM calls across all models used in this completion
    n_calls = 0
    if usage and hasattr(usage, "model_usage_summaries"):
        n_calls = sum(
            getattr(m, "total_calls", 0) or 0
            for m in usage.model_usage_summaries.values()
        )

    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    logger.info(
        f"[RLM] Tokens: {in_tok:,} in + {out_tok:,} out = {in_tok + out_tok:,} total | "
        f"Sub-calls: {n_calls} | Cost: {cost_str} | Time: {wall_time:.1f}s"
    )
    return {
        "input_tokens":    in_tok,
        "output_tokens":   out_tok,
        "total_tokens":    in_tok + out_tok,
        "cost":            cost or 0.0,
        "rlm_total_calls": n_calls,    # total LLM sub-calls made by the RLM
        "rlm_wall_time_s": wall_time,  # wall-clock time for this completion
    }


def send_request_rlm(instructions: str, input_data) -> tuple[str, dict]:
    """
    Send a request via the RLM engine and return (content, usage).

    The RLM paradigm loads `input_data` (the documents) as the REPL `context`
    variable, and uses `instructions` as the `root_prompt` so the model knows
    what task to perform on that context.

    input_data can be any serialisable type - a string, a list of dicts
    (for retrieval/RAG), or a plain list (for clustering).
    """
    from rlm import (
        BudgetExceededError,
        CancellationError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )
    from openai import RateLimitError, APIStatusError
    import time

    rlm_client = get_rlm(settings.model)

    _zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                   "cost": 0.0, "rlm_total_calls": 0, "rlm_wall_time_s": 0.0}

    for attempt in range(settings.max_retries + 1):
        try:
            # context (input_data) → loaded into REPL `context` variable
            # root_prompt (instructions) → shown to the model as the task description
            result = rlm_client.completion(input_data, root_prompt=instructions)
            if result is None:
                logger.warning(f"[RLM] None result from completion, attempt {attempt + 1}/{settings.max_retries}")
                time.sleep(2)
                continue
            usage = _log_rlm_usage(result)
            if result.response:
                return result.response, usage
            logger.warning("[RLM] Empty response from completion")
            return "", usage

        except (TimeoutExceededError, TokenLimitExceededError,
                ErrorThresholdExceededError, BudgetExceededError) as e:
            # These carry a partial_answer — return it if available
            partial = getattr(e, "partial_answer", None)
            logger.warning(f"[RLM] {type(e).__name__}: {e}")
            if partial:
                logger.info(f"[RLM] Returning partial answer ({len(partial)} chars)")
                return partial, _zero_usage
            return "", _zero_usage

        except CancellationError:
            logger.warning("[RLM] Cancelled by user")
            return "", _zero_usage

        except (TypeError, AttributeError) as e:
            if "NoneType" in str(e) or "'NoneType'" in str(e):
                logger.warning(
                    f"[RLM] None content (safety filter?), "
                    f"attempt {attempt + 1}/{settings.max_retries}: {e}"
                )
                time.sleep(2)
                continue
            raise

        except RateLimitError as e:
            if attempt == settings.max_retries:
                raise
            wait = 15.0
            err_msg = str(e).lower()
            if "retry in" in err_msg:
                try:
                    part = err_msg.split("retry in ")[1]
                    num_str = "".join(c for c in part if c.isdigit() or c == ".")
                    if num_str:
                        wait = float(num_str) + 1.0
                except Exception:
                    pass
            logger.warning(f"[RLM] Rate limit, retry in {wait:.0f}s (attempt {attempt + 1})")
            time.sleep(wait)

        except APIStatusError as e:
            if attempt == settings.max_retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"[RLM] API {e}, retry in {wait}s (attempt {attempt + 1})")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"[RLM] Unexpected error: {type(e).__name__}: {e}")
            raise

    logger.error("[RLM] All retries exhausted")
    return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
               "cost": 0.0, "rlm_total_calls": 0, "rlm_wall_time_s": 0.0}


def send_request_rlm_fresh(instructions: str, input_data) -> tuple[str, dict]:
    """Like send_request_rlm() but creates a fresh RLM instance per call — thread-safe for parallel workers."""
    from rlm import (
        BudgetExceededError,
        CancellationError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )
    from openai import RateLimitError, APIStatusError
    import time

    _zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                   "cost": 0.0, "rlm_total_calls": 0, "rlm_wall_time_s": 0.0}

    for attempt in range(settings.max_retries + 1):
        try:
            rlm_client = create_rlm_instance()
            result = rlm_client.completion(input_data, root_prompt=instructions)
            if result is None:
                logger.warning(f"[RLM-fresh] None result from completion, attempt {attempt + 1}/{settings.max_retries}")
                time.sleep(2)
                continue
            usage = _log_rlm_usage(result)
            if result.response:
                return result.response, usage
            logger.warning("[RLM-fresh] Empty response from completion")
            return "", usage

        except (TimeoutExceededError, TokenLimitExceededError,
                ErrorThresholdExceededError, BudgetExceededError) as e:
            partial = getattr(e, "partial_answer", None)
            logger.warning(f"[RLM-fresh] {type(e).__name__}: {e}")
            if partial:
                logger.info(f"[RLM-fresh] Returning partial answer ({len(partial)} chars)")
                return partial, _zero_usage
            return "", _zero_usage

        except CancellationError:
            logger.warning("[RLM-fresh] Cancelled by user")
            return "", _zero_usage

        except (TypeError, AttributeError) as e:
            if "NoneType" in str(e) or "'NoneType'" in str(e):
                logger.warning(f"[RLM-fresh] None content (safety filter / empty response?), attempt {attempt + 1}/{settings.max_retries}: {e}")
                time.sleep(2)
                continue
            raise

        except RateLimitError as e:
            if attempt == settings.max_retries:
                raise
            wait = 15.0
            err_msg = str(e).lower()
            if "retry in" in err_msg:
                try:
                    part = err_msg.split("retry in ")[1]
                    num_str = "".join(c for c in part if c.isdigit() or c == ".")
                    if num_str:
                        wait = float(num_str) + 1.0
                except Exception:
                    pass
            logger.warning(f"[RLM-fresh] Rate limit, retry in {wait:.0f}s (attempt {attempt + 1})")
            time.sleep(wait)

        except APIStatusError as e:
            if attempt == settings.max_retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"[RLM-fresh] API {e}, retry in {wait}s (attempt {attempt + 1})")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"[RLM-fresh] Unexpected error (attempt {attempt + 1}): {type(e).__name__}: {e}")
            if attempt < settings.max_retries:
                time.sleep(2 ** attempt)
                continue
            return "", _zero_usage

    logger.error("[RLM-fresh] All retries exhausted")
    return "", _zero_usage


async def send_request(instructions: str, input: str, bypass_rlm: bool = False, **kwargs: dict) -> tuple[str, dict]:
    """
    Send a request to the AsyncOpenAI client.
    Returns (content, usage_dict).
    """
    # Auto-bypass RLM for structured-output tasks (json_schema).
    # These are simple per-item scoring (STS, reranking, pair_cls, classification)
    # that don't benefit from iterative REPL reasoning.
    if "response_format" in kwargs:
        rf = kwargs["response_format"]
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            bypass_rlm = True

    if settings.use_rlm and not bypass_rlm:
        return send_request_rlm(instructions, input)

    # Patch response_format schema for strict-mode compatibility if present
    if "response_format" in kwargs:
        rf = kwargs["response_format"]
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            if settings.use_strict_json:
                # OpenAI strict mode: patch schema to be fully strict-compliant
                inner = rf.get("json_schema", {})
                if "schema" in inner:
                    inner["schema"] = _make_schema_strict(inner["schema"])
            else:
                # Gemini/Groq: remove response_format entirely (it caps output tokens)
                # The prompt already instructs JSON output explicitly
                del kwargs["response_format"]

    if settings.use_vertex_ai:
        token, _ = get_vertex_credentials()
        client.api_key = token
    else:
        client.api_key = settings.token

    async with _get_semaphore():
        for attempt in range(settings.max_retries + 1):
            try:
                logger.info(f"Sending request: model={settings.model}, max_tokens={settings.max_tokens}")
                extra = {}
                if settings.reasoning_effort is not None:
                    if "openrouter" in settings.base_url:
                        extra.setdefault("extra_body", {})["reasoning"] = {"effort": settings.reasoning_effort}
                    else:
                        extra["reasoning_effort"] = settings.reasoning_effort
                if not settings.enable_thinking:
                    _eb = extra.setdefault("extra_body", {})
                    if settings.use_vertex_ai:
                        _eb["google"] = {"thinking_config": {"thinking_budget": 0}}
                    elif "openrouter" in settings.base_url:
                        _eb["reasoning"] = {"enabled": False}
                    else:
                        _eb["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
                response = await client.chat.completions.create(
                    messages=[
                        ChatCompletionSystemMessageParam(role="system", content=instructions),
                        ChatCompletionUserMessageParam(role="user", content=input),
                    ],
                    max_tokens=settings.max_tokens,
                    model=settings.model,
                    **extra,
                    **kwargs,
                )
                choice = response.choices[0]
                finish = choice.finish_reason
                usage = response.usage
                content = choice.message.content or ""
                content, n_think_chars = _strip_thinking(content)

                if finish != "stop":
                    if usage:
                        details = getattr(usage, "completion_tokens_details", None)
                        thinking = getattr(details, "reasoning_tokens", None) if details else None
                        tokens_info = (
                            f"usage={usage.completion_tokens}/{settings.max_tokens} completion tokens"
                            + (f" (thinking={thinking})" if thinking else "")
                        )
                    else:
                        tokens_info = "no usage info"
                    logger.warning(f"finish_reason={finish} | {tokens_info}")

                # Some providers return None content on truncation
                in_tok = 0
                out_tok = 0
                tot_tok = 0
                cached_tok = 0

                if usage:
                    in_tok = getattr(usage, "prompt_tokens", 0)
                    out_tok = getattr(usage, "completion_tokens", 0)
                    tot_tok = getattr(usage, "total_tokens", 0)
                    # Cached tokens: vLLM with prefix caching + OpenAI both report here.
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached_tok = getattr(details, "cached_tokens", 0) if details else 0

                # Fallback for Gemini / LiteLLM proxy which might inject token info elsewhere
                if in_tok == 0 and out_tok == 0:
                    usage_metadata = getattr(response, "usage_metadata", None)
                    if usage_metadata:
                        if isinstance(usage_metadata, dict):
                            in_tok = usage_metadata.get("prompt_token_count", 0)
                            out_tok = usage_metadata.get("candidates_token_count", 0)
                            tot_tok = usage_metadata.get("total_token_count", 0)
                            cached_tok = cached_tok or usage_metadata.get("cached_content_token_count", 0)
                        else:
                            in_tok = getattr(usage_metadata, "prompt_token_count", 0)
                            out_tok = getattr(usage_metadata, "candidates_token_count", 0)
                            tot_tok = getattr(usage_metadata, "total_token_count", 0)
                            cached_tok = cached_tok or getattr(usage_metadata, "cached_content_token_count", 0)

                # Reasoning-tokens accounting: prefer the API's own count (OpenAI/Gemini);
                # fall back to inferring from the stripped <think> block for models that
                # don't expose it separately (e.g. Qwen3.6 served via vLLM).
                api_think_tok = 0
                if usage:
                    details = getattr(usage, "completion_tokens_details", None)
                    api_think_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                    api_think_tok = api_think_tok or getattr(usage, "reasoning_tokens", 0) or 0
                inferred_think_tok = round(n_think_chars / 4) if n_think_chars else 0  # ~4 chars/token

                usage_dict = {
                    "input_tokens":    in_tok,
                    "cached_tokens":   cached_tok,
                    "output_tokens":   out_tok,
                    "thinking_tokens": api_think_tok or inferred_think_tok,
                    "total_tokens":    tot_tok,
                    "cost":            0.0,
                }

                return content, usage_dict

            except RateLimitError as e:
                if attempt == settings.max_retries:
                    raise
                # Try to read Retry-After header from the response
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_str = e.response.headers.get("retry-after")
                    if retry_after_str:
                        try:
                            retry_after = float(retry_after_str) + 1.0  # add 1s buffer
                        except ValueError:
                            pass
                wait = retry_after if retry_after else max(5, 2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {wait:.1f}s (attempt {attempt + 1}/{settings.max_retries})")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                # Retry on 5xx, 404 (vLLM restart/cold start), and 408 (timeout).
                # Other 4xx are programming errors and re-raise.
                if (e.status_code >= 500 or e.status_code in (404, 408, 499)) and attempt < settings.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"Transient status {e.status_code}, retrying in {wait}s (attempt {attempt + 1}/{settings.max_retries})")
                    await asyncio.sleep(wait)
                elif e.status_code == 400 and "Developer instruction is not enabled" in str(e):
                    # Model doesn't support system role — merge instruction into user message
                    logger.warning("Model doesn't support system role — merging instruction into user message")
                    merged = f"{instructions}\n\n{input}"
                    response = await client.chat.completions.create(
                        messages=[ChatCompletionUserMessageParam(role="user", content=merged)],
                        max_tokens=settings.max_tokens,
                        model=settings.model,
                        **extra,
                        **kwargs,
                    )
                    content = response.choices[0].message.content or ""
                    
                    usage = response.usage
                    usage_dict = {
                        "input_tokens":    usage.prompt_tokens if usage else 0,
                        "cached_tokens":   0,
                        "output_tokens":   usage.completion_tokens if usage else 0,
                        "thinking_tokens": 0,
                        "total_tokens":    usage.total_tokens if usage else 0,
                        "cost":            0.0,
                    }
                    return content, usage_dict
                elif e.status_code == 400 and "json_validate_failed" in str(e):
                    # The model produced invalid JSON (e.g. escaped apostrophes like \')
                    # Try to salvage the failed_generation text from the error body.
                    body = e.body if isinstance(e.body, dict) else {}
                    failed_gen = body.get("error", {}).get("failed_generation", "")
                    if failed_gen:
                        # Fix common bad escapes: \' -> ' (apostrophes don't need escaping in JSON)
                        repaired = failed_gen.replace("\\'", "'")
                        try:
                            json.loads(repaired)  # validate the repair worked
                            logger.warning(f"json_validate_failed repaired (attempt {attempt + 1}): returning salvaged content")
                            return repaired, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "cost": 0.0}
                        except json.JSONDecodeError:
                            pass
                    # Repair failed — retry if attempts remain, otherwise raise
                    if attempt < settings.max_retries:
                        logger.warning(f"json_validate_failed, retrying (attempt {attempt + 1}/{settings.max_retries})")
                        await asyncio.sleep(1)
                    else:
                        raise
                else:
                    logger.warning(f"Unhandled APIStatusError: status_code={e.status_code} msg={e}")
                    raise
            except APIConnectionError as e:
                # Transient — tunnel blip, vLLM restart, network jitter. Retry with backoff.
                if attempt == settings.max_retries:
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning(f"Connection error, retrying in {wait}s (attempt {attempt + 1}/{settings.max_retries}): {e}")
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                raise
    return "", {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "cost": 0.0}


async def send_request_multi(
    instructions: str,
    context: str,
    query: str,
    **kwargs,
) -> tuple[str, dict]:
    """Send a request structured for implicit Vertex AI caching.

    Messages are ordered so the large, static corpus forms a stable prefix
    that the Gemini backend can cache across multiple queries:

        system  → instructions (static, same for all queries in a task)
        user    → context / corpus (static, same for all queries)
        asst    → "Understood." (static acknowledgement — extends stable prefix)
        user    → query (dynamic, unique per request)

    The threshold for implicit caching is 1024 tokens; corpus prompts for
    retrieval/RAG tasks are typically 10k–100k tokens, so cache hits are
    virtually guaranteed after the first request in each batch.
    """
    async with _get_semaphore():
        for attempt in range(settings.max_retries + 1):
            try:
                if settings.use_vertex_ai:
                    token, _ = get_vertex_credentials()
                    client.api_key = token
                else:
                    client.api_key = settings.token

                logger.info(
                    f"[multi] Sending cached request: model={settings.model}, "
                    f"context_len={len(context)}, query_len={len(query)}"
                )
                extra = {}
                if settings.reasoning_effort is not None:
                    if "openrouter" in settings.base_url:
                        extra.setdefault("extra_body", {})["reasoning"] = {"effort": settings.reasoning_effort}
                    else:
                        extra["reasoning_effort"] = settings.reasoning_effort
                if not settings.enable_thinking:
                    _eb = extra.setdefault("extra_body", {})
                    if settings.use_vertex_ai:
                        _eb["google"] = {"thinking_config": {"thinking_budget": 0}}
                    elif "openrouter" in settings.base_url:
                        _eb["reasoning"] = {"enabled": False}
                    else:
                        _eb["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}

                response = await client.chat.completions.create(
                    messages=[
                        ChatCompletionSystemMessageParam(
                            role="system", content=instructions
                        ),
                        ChatCompletionUserMessageParam(
                            role="user", content=context           # corpus — stable prefix ✅
                        ),
                        {"role": "assistant", "content": "Understood, I have read all the documents carefully."},
                        ChatCompletionUserMessageParam(
                            role="user", content=query             # dynamic per request ✅
                        ),
                    ],
                    max_tokens=settings.max_tokens,
                    model=settings.model,
                    **extra,
                    **kwargs,
                )

                choice = response.choices[0]
                content = choice.message.content or ""
                content, n_think_chars = _strip_thinking(content)
                finish = choice.finish_reason
                usage = response.usage

                if finish != "stop":
                    logger.warning(f"[multi] finish_reason={finish}")

                in_tok = out_tok = tot_tok = cached_tok = 0
                if usage:
                    in_tok = getattr(usage, "prompt_tokens", 0)
                    out_tok = getattr(usage, "completion_tokens", 0)
                    tot_tok = getattr(usage, "total_tokens", 0)
                    # Check for cache hits in OpenAI-style prompt_tokens_details
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached_tok = getattr(details, "cached_tokens", 0) if details else 0

                # Fallback for Gemini / LiteLLM proxy
                if in_tok == 0 and out_tok == 0:
                    usage_metadata = getattr(response, "usage_metadata", None)
                    if usage_metadata:
                        if isinstance(usage_metadata, dict):
                            in_tok = usage_metadata.get("prompt_token_count", 0)
                            out_tok = usage_metadata.get("candidates_token_count", 0)
                            tot_tok = usage_metadata.get("total_token_count", 0)
                            cached_tok = usage_metadata.get("cached_content_token_count", 0)
                        else:
                            in_tok = getattr(usage_metadata, "prompt_token_count", 0)
                            out_tok = getattr(usage_metadata, "candidates_token_count", 0)
                            tot_tok = getattr(usage_metadata, "total_token_count", 0)
                            cached_tok = getattr(usage_metadata, "cached_content_token_count", 0)

                if cached_tok:
                    logger.info(f"[multi] Cache hit: {cached_tok:,} cached tokens")

                # Reasoning tokens: API-reported preferred; fall back to inferred from </think> strip.
                api_think_tok = 0
                if usage:
                    details = getattr(usage, "completion_tokens_details", None)
                    api_think_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                    api_think_tok = api_think_tok or getattr(usage, "reasoning_tokens", 0) or 0
                inferred_think_tok = round(n_think_chars / 4) if n_think_chars else 0

                usage_dict = {
                    "input_tokens":    in_tok,
                    "cached_tokens":   cached_tok,
                    "output_tokens":   out_tok,
                    "thinking_tokens": api_think_tok or inferred_think_tok,
                    "total_tokens":    tot_tok,
                    "cost":            0.0,
                }
                return content, usage_dict

            except RateLimitError as e:
                if attempt == settings.max_retries:
                    raise
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_str = e.response.headers.get("retry-after")
                    if retry_after_str:
                        try:
                            retry_after = float(retry_after_str) + 1.0
                        except ValueError:
                            pass
                wait = retry_after if retry_after else max(5, 2 ** attempt)
                logger.warning(
                    f"[multi] Rate limit, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{settings.max_retries})"
                )
                await asyncio.sleep(wait)

            except APIStatusError as e:
                # Retry on 5xx, 404 (vLLM restart/cold start), 408 (timeout).
                if (e.status_code >= 500 or e.status_code in (404, 408, 499)) and attempt < settings.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        f"[multi] Transient status {e.status_code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{settings.max_retries})"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

            except APIConnectionError as e:
                # Transient connectivity (tunnel blip, vLLM restart). Retry with backoff.
                if attempt == settings.max_retries:
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning(
                    f"[multi] Connection error, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{settings.max_retries}): {e}"
                )
                await asyncio.sleep(wait)

            except Exception as e:
                logger.error(f"[multi] Unexpected error: {type(e).__name__}: {e}")
                raise

    return "", {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "cost": 0.0}
