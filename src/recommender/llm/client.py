"""Thin wrapper around the Google Gemini SDK with logging + prompt loading.

All LLM calls in the agent go through ``call_messages`` or ``call_messages_parsed``
so timeouts, retries, and structured logs are uniform.
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
_genai_types: Optional[Any] = None


def _load_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Install project dependencies before using LLM features."
        ) from exc
    return genai, types


def get_client() -> Any:
    """Lazy-init shared Gemini client. Reads GEMINI_API_KEY from env."""
    global _client, _genai_types
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export it in your shell."
            )
        genai, types = _load_genai()
        _genai_types = types
        _client = genai.Client(api_key=api_key)
    return _client


def _get_genai_types() -> Any:
    global _genai_types
    if _genai_types is None:
        _, _genai_types = _load_genai()
    return _genai_types


def _build_contents(system: str, user: str) -> str:
    return f"{system}\n\n{user}" if system else user


def _stop_reason(response: Any) -> str:
    if not getattr(response, "candidates", None):
        return "unknown"
    finish_reason = getattr(response.candidates[0], "finish_reason", None)
    if finish_reason is None:
        return "unknown"
    return getattr(finish_reason, "name", str(finish_reason))


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


_BUDGET_FOR_EFFORT = {"minimal": 0, "low": 1024, "medium": 4096, "high": 8192}


def _thinking_config(types: Any, model: str, effort: Optional[str]) -> Any:
    """Map our effort level onto the model's thinking knob.

    Gemini 3 takes ``thinking_level``; Gemini 2.5 takes a token ``thinking_budget``.
    Leaving it unset lets the model think as long as it likes, which is the main
    source of latency, so every call passes an explicit level.
    """
    effort = effort or config.EFFORT
    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=effort)
    return types.ThinkingConfig(thinking_budget=_BUDGET_FOR_EFFORT.get(effort, 1024))


def _usage(response: Any) -> Dict[str, Any]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "thinking_tokens": getattr(meta, "thoughts_token_count", None),
    }


def _generate(
    *,
    system: str,
    user: str,
    label: str,
    max_tokens: int,
    effort: Optional[str],
    temperature: float,
    model: Optional[str],
    schema: Optional[Type[BaseModel]] = None,
) -> str:
    client = get_client()
    types = _get_genai_types()
    model = model or config.MODEL

    extra: Dict[str, Any] = {}
    if schema is not None:
        extra = {
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
        }
    generation_config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        thinking_config=_thinking_config(types, model, effort),
        **extra,
    )

    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=_build_contents(system, user),
            config=generation_config,
        )
    except Exception as e:
        _log_call({"label": label, "model": model, "error": str(e), "status": "error"})
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)

    _log_call({
        "label": label,
        "model": model,
        "effort": effort,
        "elapsed_ms": elapsed_ms,
        "stop_reason": _stop_reason(response),
        **_usage(response),
    })
    return response.text.strip() if response.text else ""


def call_messages(
    *,
    system: str,
    user: str,
    label: str,
    max_tokens: int,
    effort: Optional[str] = None,
    temperature: float = 1.0,
    model: Optional[str] = None,
) -> str:
    """Plain text response. Returns text from Gemini API."""
    return _generate(
        system=system, user=user, label=label, max_tokens=max_tokens,
        effort=effort, temperature=temperature, model=model,
    )


def call_messages_parsed(
    *,
    system: str,
    user: str,
    label: str,
    schema: Type[T],
    max_tokens: int,
    effort: Optional[str] = None,
    temperature: float = 1.0,
    model: Optional[str] = None,
) -> T:
    """Structured-output call. Returns a validated Pydantic model."""
    text = _generate(
        system=system, user=user, label=label, max_tokens=max_tokens,
        effort=effort, temperature=temperature, model=model, schema=schema,
    )
    return schema.model_validate(json.loads(text or "{}"))


def is_configured() -> bool:
    """True when an API key is present. Lets callers pick the offline path up front."""
    return bool(os.environ.get("GEMINI_API_KEY"))
