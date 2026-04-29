"""Thin wrapper around the Google Gemini SDK with logging + prompt loading.

All LLM calls in the agent go through ``call_messages`` or ``call_messages_parsed``
so timeouts, retries, and structured logs are uniform.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
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


def call_messages(
    *,
    system: str,
    user: str,
    label: str,
    max_tokens: int,
    effort: Optional[str] = None,
    thinking: bool = False,
) -> str:
    """Plain text response. Returns text from Gemini API."""
    client = get_client()
    types = _get_genai_types()

    full_message = _build_contents(system, user)
    generation_config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=1.0,
    )

    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=config.MODEL,
            contents=full_message,
            config=generation_config,
        )
    except Exception as e:
        _log_call({"label": label, "error": str(e), "status": "error"})
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)

    text = response.text.strip() if response.text else ""
    _log_call({
        "label": label,
        "model": config.MODEL,
        "elapsed_ms": elapsed_ms,
        "stop_reason": _stop_reason(response),
    })
    return text


def call_messages_parsed(
    *,
    system: str,
    user: str,
    label: str,
    schema: Type[T],
    max_tokens: int,
    effort: Optional[str] = None,
    thinking: bool = False,
) -> T:
    """Structured-output call. Returns a validated Pydantic model."""
    client = get_client()
    types = _get_genai_types()

    full_message = _build_contents(system, user)
    generation_config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=1.0,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
    )

    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=config.MODEL,
            contents=full_message,
            config=generation_config,
        )
    except Exception as e:
        _log_call({"label": label, "error": str(e), "status": "error"})
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)

    response_text = response.text.strip() if response.text else "{}"
    response_json = json.loads(response_text)
    parsed_output = schema.model_validate(response_json)

    _log_call({
        "label": label,
        "model": config.MODEL,
        "elapsed_ms": elapsed_ms,
        "stop_reason": _stop_reason(response),
    })
    return parsed_output
