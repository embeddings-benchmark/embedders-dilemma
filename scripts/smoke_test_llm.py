#!/usr/bin/env python3
"""Smoke-test the LLM endpoint defined by .env before running the full 37-task eval.

Checks, in order:
  1. /v1/models       — endpoint is reachable, served model id matches MODEL
  2. /v1/chat/completions on a 1-line prompt — sanity-check that completions work
  3. /v1/chat/completions on a small JSON-schema prompt — verifies the path
                                                          used by classification/STS

Reads BASE_URL / TOKEN / MODEL from .env (via llm_judge.settings.Settings).

Usage:
    uv run python scripts/smoke_test_llm.py
    uv run python scripts/smoke_test_llm.py --skip-json   # skip the json-schema check
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Allow running directly with `python scripts/smoke_test_llm.py` (no `-m` needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from openai import AsyncOpenAI

from llm_judge.settings import Settings


def check_models_endpoint(settings: Settings) -> None:
    print(f"[1/3] GET {settings.base_url}/models  (timeout 10s)")
    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {settings.token}"} if settings.token else {}
            r = client.get(f"{settings.base_url}/models", headers=headers)
            r.raise_for_status()
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

    data = r.json()
    served = [m.get("id") for m in data.get("data", [])]
    print(f"      served models: {served}")

    if settings.model not in served:
        print(f"      WARNING: MODEL={settings.model!r} not in served list. The endpoint")
        print("               may still accept it (some proxies are permissive), but it's")
        print("               worth confirming.")
    else:
        print(f"      MODEL={settings.model!r} matches a served id  ✓")

    # Try to surface context-window metadata, varies by server
    for m in data.get("data", []):
        if m.get("id") == settings.model:
            for k in ("max_model_len", "context_length", "max_tokens", "context_window"):
                if k in m:
                    print(f"      reported {k}: {m[k]:,}")


async def check_basic_completion(settings: Settings) -> None:
    print("\n[2/3] basic chat completion (1-line prompt)")
    client = AsyncOpenAI(api_key=settings.token or "dummy", base_url=settings.base_url)
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the single word: OK"},
            ],
            max_tokens=16,
        )
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        sys.exit(2)
    elapsed = time.time() - t0
    msg = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    print(f"      latency: {elapsed:.2f}s")
    print(f"      reply:   {msg!r}")
    if usage:
        print(f"      tokens:  in={usage.prompt_tokens} out={usage.completion_tokens}")
    if not msg:
        print("      WARNING: empty reply. Model may be misconfigured or refusing.")


async def check_json_schema(settings: Settings) -> None:
    print("\n[3/3] structured-output (JSON, same path as classification/STS)")
    client = AsyncOpenAI(api_key=settings.token or "dummy", base_url=settings.base_url)

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
    }

    kwargs = {
        "model": settings.model,
        "messages": [
            {"role": "system",
             "content": "Answer with a JSON object: {\"answer\": \"yes\"} or {\"answer\": \"no\"}."},
            {"role": "user", "content": "Is the sky blue?"},
        ],
        "max_tokens": 64,
    }
    if settings.use_strict_json:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "yes_no", "schema": schema, "strict": True},
        }
    # If strict JSON is off, we rely on the prompt to get well-formed JSON (matches
    # what llm_client.send_request does for Gemini/Groq-style endpoints).

    t0 = time.time()
    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        sys.exit(3)
    elapsed = time.time() - t0
    msg = (resp.choices[0].message.content or "").strip()
    print(f"      latency: {elapsed:.2f}s")
    print(f"      reply:   {msg!r}")
    if settings.use_strict_json:
        print("      mode:    strict json_schema")
    else:
        print("      mode:    prompt-only (USE_STRICT_JSON=false)")


async def amain(skip_json: bool) -> None:
    settings = Settings()
    print("=== smoke test ===")
    print(f"BASE_URL={settings.base_url}")
    print(f"MODEL={settings.model}")
    print(f"USE_STRICT_JSON={settings.use_strict_json}")
    print()

    check_models_endpoint(settings)
    await check_basic_completion(settings)
    if not skip_json:
        await check_json_schema(settings)

    print("\nAll checks passed.  ✓")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-json", action="store_true", help="Skip the structured-output check.")
    args = p.parse_args()
    asyncio.run(amain(skip_json=args.skip_json))


if __name__ == "__main__":
    main()
