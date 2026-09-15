from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
from typing import Any, Awaitable, Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)


T = TypeVar("T")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-")


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def effective_reasoning_effort(model: str, requested: str) -> str:
    del model
    return requested if requested != "none" else "not_requested"


def reasoning_arguments(model: str, requested: str) -> dict[str, Any]:
    effective = effective_reasoning_effort(model, requested)
    return (
        {"extra_body": {"reasoning": {"effort": effective}}}
        if effective != "not_requested"
        else {}
    )


def usage_fields(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", 0)
            or 0
        ),
        "output_tokens": int(
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", 0)
            or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def completion_text(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(
                item.get("text", "")
                if isinstance(item, dict)
                else getattr(item, "text", "")
            )
            for item in content
        ).strip()
    return ""


def json_object_text(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError("model returned no JSON object")
    return match.group(0)


def response_route(response) -> dict[str, str]:
    return {
        "response_model": str(getattr(response, "model", "") or ""),
        "response_provider": str(getattr(response, "provider", "") or ""),
    }


async def with_retries(
    operation: Callable[[], Awaitable[T]], attempts: int = 8
) -> T:
    for attempt in range(attempts):
        try:
            return await operation()
        except (RateLimitError, APIConnectionError, APITimeoutError) as error:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        except APIStatusError as error:
            if error.status_code < 500 or attempt + 1 == attempts:
                raise
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
    raise RuntimeError("retry loop exhausted")


def client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    headers = {
        "X-OpenRouter-Title": os.environ.get(
            "OPENROUTER_APP_NAME", "rag-eval-handoff"
        )
    }
    if referer := os.environ.get("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    return AsyncOpenAI(
        base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        default_headers=headers,
        max_retries=0,
        timeout=120.0,
    )
