"""Thin wrapper around the Anthropic SDK with logging + prompt loading.

All LLM calls in the agent go through ``call_messages`` or ``call_messages_parsed``
so effort, refusal handling, fallbacks and structured logs are uniform.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .. import config


T = TypeVar("T", bound=BaseModel)


_logger = logging.getLogger("recommender.llm")
_client: Optional[Any] = None

# Server-side refusal fallback (re-runs a declined request on another model inside the
# same call). Only these models accept the "default" form on the Claude API.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


class LLMRefusalError(RuntimeError):
    """The model (and any fallback) declined. Callers treat it like any LLM failure."""


def get_client() -> Any:
    """Lazy-init shared Anthropic client.

    Credentials resolve the SDK's usual way: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    or an ``ant auth login`` profile.
    """
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic is not installed. Install project dependencies before using LLM features."
            ) from exc
        _client = anthropic.Anthropic(timeout=config.REQUEST_TIMEOUT_SECONDS, max_retries=2)
    return _client


def is_configured() -> bool:
    """True when Claude credentials are available. Lets callers pick the offline path up front."""
    if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read a prompt file from src/recommender/prompts/. ``name`` is the stem (e.g. ``"plan"``)."""
    path = config.PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _logs_path() -> Path:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return config.LOGS_DIR / "llm_calls.jsonl"


def _log_call(record: Dict[str, Any]) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with _logs_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        _logger.warning("could not write LLM call log: %s", e)


def _usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def _generate(
    *,
    system: str,
    user: str,
    label: str,
    max_tokens: int,
    effort: Optional[str],
    model: Optional[str],
    schema: Optional[Type[BaseModel]] = None,
) -> Any:
    """One Messages API request. Returns the parsed model when ``schema`` is given,
    otherwise the response text."""
    client = get_client()
    model = model or config.MODEL
    effort = effort if effort in EFFORT_LEVELS else "low"

    # Thinking is left at the model default (adaptive on current models); effort is
    # the latency/cost knob. Sampling params are not sent (removed on current models).
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort},
    }
    if model in FALLBACK_MODELS:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"

    started = time.monotonic()
    try:
        if schema is not None:
            response = client.beta.messages.parse(output_format=schema, **kwargs)
        else:
            response = client.beta.messages.create(**kwargs)
    except Exception as e:
        _log_call({"label": label, "model": model, "error": str(e), "status": "error"})
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)

    _log_call({
        "label": label,
        "model": model,
        "served_by": getattr(response, "model", model),
        "effort": effort,
        "elapsed_ms": elapsed_ms,
        "stop_reason": response.stop_reason,
        **_usage(response),
    })

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise LLMRefusalError(f"{label}: model declined (category={category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{label}: hit max_tokens={max_tokens} before finishing")

    if schema is not None:
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(f"{label}: no structured output returned")
        return parsed
    return "".join(b.text for b in response.content if b.type == "text").strip()


def call_messages(
    *,
    system: str,
    user: str,
    label: str,
    max_tokens: int,
    effort: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Plain text response."""
    return _generate(
        system=system, user=user, label=label, max_tokens=max_tokens,
        effort=effort, model=model,
    )


def call_messages_parsed(
    *,
    system: str,
    user: str,
    label: str,
    schema: Type[T],
    max_tokens: int,
    effort: Optional[str] = None,
    model: Optional[str] = None,
) -> T:
    """Structured-output call. Returns a validated Pydantic model."""
    return _generate(
        system=system, user=user, label=label, max_tokens=max_tokens,
        effort=effort, model=model, schema=schema,
    )
