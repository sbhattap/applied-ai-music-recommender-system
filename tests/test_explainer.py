"""Tests for the LLM explainer (mocked) and its templated fallback."""

from src.recommender.llm import explainer
from src.recommender.llm.types import CheckFlag, CheckVerdict


SONG = {
    "id": 1, "title": "Lofi Loop", "artist": "A",
    "genre": "lofi", "mood": "chill", "tempo_bpm": 75, "energy": 0.30,
}


def test_templated_fallback_no_hedge_on_high_confidence():
    verdict = CheckVerdict(confidence=0.85, flags=[], note="ok")
    out = explainer.templated_fallback(
        song=SONG, breakdown="it's lofi (your favorite genre)", verdict=verdict
    )
    assert "lofi" in out
    assert "Closest match" not in out


def test_templated_fallback_hedges_on_low_confidence():
    verdict = CheckVerdict(
        confidence=0.4,
        flags=[CheckFlag.INTENT_MISMATCH],
        note="catalog leans intense",
    )
    out = explainer.templated_fallback(
        song=SONG, breakdown="it's lofi (your favorite genre)", verdict=verdict
    )
    assert out.startswith("Closest match")


def test_explain_calls_llm_with_breakdown_and_verdict(monkeypatch):
    captured = {}

    def fake_text(*, system, user, label, max_tokens, **kwargs):
        captured["system"] = system
        captured["user"] = user
        captured["label"] = label
        return "Because it's a perfect lofi fit."

    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", fake_text)
    monkeypatch.setattr(
        "src.recommender.llm.explainer.client.load_prompt",
        lambda name: f"<<{name}-prompt>>",
    )

    verdict = CheckVerdict(confidence=0.85, flags=[], note="ok")
    out = explainer.explain(
        query="chill study music",
        profile_summary="genre=lofi, mood=chill, energy=0.35",
        song=SONG,
        score=0.91,
        breakdown="it's lofi (your favorite genre), and energy matches",
        verdict=verdict,
    )

    assert out == "Because it's a perfect lofi fit."
    assert captured["label"] == "explainer"
    assert "0.85" in captured["user"]  # verdict confidence reaches the prompt
    assert "lofi (your favorite genre)" in captured["user"]  # breakdown reaches the prompt
    assert "chill study music" in captured["user"]  # query reaches the prompt
