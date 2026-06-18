"""LLM client tests against a fake Anthropic SDK client (no network)."""

from types import SimpleNamespace

import pytest

from src.recommender.llm import client
from src.recommender.llm.types import CheckVerdict


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        return self.response

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self.response


def _install(monkeypatch, response):
    messages = FakeMessages(response)
    fake = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(client, "_client", fake)
    monkeypatch.setattr(client, "_log_call", lambda record: None)
    return messages


def _response(stop_reason="end_turn", parsed=None, text="hi", model="claude-opus-5-5"):
    return SimpleNamespace(
        stop_reason=stop_reason, parsed_output=parsed, model=model,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
    )


def test_parsed_call_sends_schema_effort_and_fallbacks(monkeypatch):
    verdict = CheckVerdict(confidence=0.9, note="ok")
    messages = _install(monkeypatch, _response(parsed=verdict))

    out = client.call_messages_parsed(
        system="sys", user="u", label="check", schema=CheckVerdict,
        max_tokens=100, effort="low", model="claude-opus-5-5",
    )

    assert out is verdict
    kind, kwargs = messages.calls[0]
    assert kind == "parse"
    assert kwargs["output_format"] is CheckVerdict
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == [client.FALLBACK_BETA]
    assert "temperature" not in kwargs and "thinking" not in kwargs


def test_no_fallbacks_for_models_without_server_side_fallback(monkeypatch):
    messages = _install(monkeypatch, _response())
    client.call_messages(system="s", user="u", label="x", max_tokens=10, model="claude-haiku-5-5")
    _, kwargs = messages.calls[0]
    assert "fallbacks" not in kwargs and "betas" not in kwargs


def test_text_call_joins_text_blocks_and_maps_unknown_effort(monkeypatch):
    messages = _install(monkeypatch, _response(text="  hello  "))
    out = client.call_messages(system="s", user="u", label="x", max_tokens=10, effort="minimal")
    assert out == "hello"
    assert messages.calls[0][1]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("stop_reason,error", [
    ("refusal", client.LLMRefusalError),
    ("max_tokens", RuntimeError),
])
def test_bad_stop_reasons_raise(monkeypatch, stop_reason, error):
    _install(monkeypatch, _response(stop_reason=stop_reason))
    with pytest.raises(error):
        client.call_messages_parsed(system="s", user="u", label="x", schema=CheckVerdict, max_tokens=10)


def test_is_configured_reads_anthropic_env(monkeypatch, tmp_path):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(client.Path, "home", lambda: tmp_path)
    assert client.is_configured() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert client.is_configured() is True
