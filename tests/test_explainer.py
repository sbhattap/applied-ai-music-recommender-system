"""Tests for the LLM explainer (mocked) and its templated fallback."""

from src.recommender.llm import explainer
from src.recommender.llm.types import (
    BatchExplanations,
    CheckFlag,
    CheckVerdict,
    Claim,
    SongExplanation,
)


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


SIGNALS = [{"name": "energy", "label": "Energy", "value": 0.95, "weight": 0.16,
            "contribution": 0.152, "detail": "energy 0.30 vs your target 0.35"}]
EVIDENCE = [{"doc_id": "song:1", "kind": "song", "source": "catalog", "title": "Lofi Loop",
             "text": "Lofi Loop by A. lofi song, chill mood.", "url": "", "score": 0.8}]


def test_explain_batch_makes_one_call_with_numbered_evidence(monkeypatch):
    calls = []

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        calls.append({"system": system, "user": user, "label": label, "schema": schema})
        return BatchExplanations(explanations=[
            SongExplanation(song_id=1, claims=[Claim(text="Energy fits.", evidence_ids=["sig:energy"])]),
            SongExplanation(song_id=99, claims=[Claim(text="Not ours.", evidence_ids=["song:99"])]),
        ])

    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr(
        "src.recommender.llm.explainer.client.load_prompt",
        lambda name: f"<<{name}-prompt>>",
    )

    verdict = CheckVerdict(confidence=0.85, flags=[], note="ok")
    song2 = {**SONG, "id": 2, "title": "Second"}
    out = explainer.explain_batch(
        query="chill study music",
        profile_summary="genre=lofi, mood=chill, energy=0.35",
        items=[
            {"song": SONG, "score": 0.91, "signals": SIGNALS, "evidence": EVIDENCE},
            {"song": song2, "score": 0.80, "signals": SIGNALS, "evidence": []},
        ],
        verdict=verdict,
    )

    assert len(calls) == 1
    assert calls[0]["label"] == "explainer"
    assert calls[0]["schema"] is BatchExplanations
    user = calls[0]["user"]
    assert "song_id=1" in user and "song_id=2" in user
    assert "[sig:energy]" in user and "[song:1]" in user
    assert "confidence=0.85" in user
    # song 2 was skipped by the model; unknown id 99 is dropped
    assert out == {1: [{"text": "Energy fits.", "evidence_ids": ["sig:energy"]}]}
