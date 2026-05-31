"""Tests for the LLM explainer (mocked) and its templated fallback."""

from src.recommender.llm import explainer
from src.recommender.llm.types import BatchNarratives, CheckFlag, CheckVerdict, SongNarrative


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


def test_explain_batch_makes_one_call_for_all_songs(monkeypatch):
    calls = []

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        calls.append({"system": system, "user": user, "label": label})
        return BatchNarratives(narratives=[
            SongNarrative(song_id=1, narrative="Because it's a lofi fit."),
            SongNarrative(song_id=99, narrative="Not one of ours."),
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
            {"song": SONG, "score": 0.91, "breakdown": "it's lofi (your favorite genre)"},
            {"song": song2, "score": 0.80, "breakdown": "energy matches"},
        ],
        verdict=verdict,
    )

    assert len(calls) == 1
    assert calls[0]["label"] == "explainer"
    assert calls[0]["system"] == "<<explainer-prompt>>"
    assert "song_id=1" in calls[0]["user"] and "song_id=2" in calls[0]["user"]
    assert "confidence=0.85" in calls[0]["user"]
    # song 2 was skipped by the model; unknown id 99 is dropped
    assert out == {1: "Because it's a lofi fit."}
