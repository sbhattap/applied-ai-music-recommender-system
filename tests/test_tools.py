"""Tests for the agent's tools. Pure-logic tools run as-is; semantic_search is mocked."""

import pytest

from src.recommender.llm.types import Plan, SemanticPicks
from src.recommender.tools import dispatch, hybrid, semantic_search, structured_filter


@pytest.fixture
def songs():
    return [
        {"id": 1, "title": "Lofi Loop", "artist": "A", "genre": "lofi",
         "mood": "chill", "mood_tags": "chill|study", "energy": 0.30,
         "tempo_bpm": 75, "valence": 0.5, "danceability": 0.4, "acousticness": 0.7,
         "instrumentalness": 0.92,
         "popularity": 60, "release_decade": "2020s", "liveness": 0.1,
         "duration_sec": 200, "explicit": 0},
        {"id": 2, "title": "Hard Rock", "artist": "B", "genre": "rock",
         "mood": "intense", "mood_tags": "intense|driving", "energy": 0.95,
         "tempo_bpm": 140, "valence": 0.6, "danceability": 0.5, "acousticness": 0.05,
         "instrumentalness": 0.01,
         "popularity": 70, "release_decade": "2010s", "liveness": 0.2,
         "duration_sec": 200, "explicit": 0},
        {"id": 3, "title": "Ambient Drift", "artist": "C", "genre": "ambient",
         "mood": "chill", "mood_tags": "chill|sleep", "energy": 0.20,
         "tempo_bpm": 60, "valence": 0.4, "danceability": 0.2, "acousticness": 0.9,
         "instrumentalness": 0.97,
         "popularity": 30, "release_decade": "2020s", "liveness": 0.05,
         "duration_sec": 300, "explicit": 0},
    ]


def _plan(**overrides) -> Plan:
    base = dict(intent="x", chosen_tool="structured_filter",
                expected_count_min=1, expected_count_max=10)
    base.update(overrides)
    return Plan(**base)


def test_structured_filter_genre(songs):
    out = structured_filter.run(_plan(genre_filter=["lofi"]), songs)
    assert [s["id"] for s in out] == [1]


def test_structured_filter_mood_via_tag(songs):
    # song 1 has mood_tags "chill|study" — a "study" mood request should match via tags
    out = structured_filter.run(_plan(mood_filter=["study"]), songs)
    assert {s["id"] for s in out} == {1}


def test_structured_filter_bpm_range(songs):
    out = structured_filter.run(_plan(bpm_max=80), songs)
    assert {s["id"] for s in out} == {1, 3}


def test_structured_filter_energy_range(songs):
    out = structured_filter.run(_plan(energy_min=0.5), songs)
    assert {s["id"] for s in out} == {2}


def test_structured_filter_combined(songs):
    out = structured_filter.run(
        _plan(genre_filter=["lofi", "ambient"], energy_max=0.4),
        songs,
    )
    assert {s["id"] for s in out} == {1, 3}


def test_structured_filter_instrumentalness(songs):
    out = structured_filter.run(_plan(instrumentalness_min=0.9), songs)
    assert {s["id"] for s in out} == {1, 3}


def test_dispatch_routes_to_structured_filter(songs):
    out = dispatch.call("structured_filter", _plan(genre_filter=["rock"]), songs)
    assert [s["id"] for s in out] == [2]


def test_dispatch_unknown_tool_raises(songs):
    with pytest.raises(ValueError, match="unknown tool"):
        dispatch.call("nonexistent", _plan(), songs)  # type: ignore[arg-type]


def test_hybrid_skips_semantic_when_no_query(songs):
    # No semantic_query → hybrid should just return the structured-filter output
    out = hybrid.run(_plan(genre_filter=["lofi"]), songs)
    assert [s["id"] for s in out] == [1]


def test_hybrid_calls_semantic_after_filter(songs, monkeypatch):
    captured = {}

    def fake_semantic(plan, songs):
        captured["plan"] = plan
        captured["songs"] = songs
        # Simulate the LLM picking song id 3 from the narrowed set
        return [s for s in songs if s["id"] == 3]

    monkeypatch.setattr(semantic_search, "run", fake_semantic)
    plan = _plan(
        chosen_tool="hybrid",
        genre_filter=["lofi", "ambient"],
        semantic_query="something to fall asleep to",
    )
    out = hybrid.run(plan, songs)

    # structured_filter should have narrowed to [lofi, ambient] — i.e. ids 1 and 3
    narrowed_ids = [s["id"] for s in captured["songs"]]
    assert sorted(narrowed_ids) == [1, 3]
    # And semantic_search picked id 3 from those
    assert [s["id"] for s in out] == [3]


def test_semantic_search_passes_query_through(songs, monkeypatch):
    """semantic_search should call the LLM client with the plan's semantic_query."""
    captured = {}

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        captured["system"] = system
        captured["user"] = user
        captured["label"] = label
        return SemanticPicks(song_ids=[1, 3])

    monkeypatch.setattr(
        "src.recommender.tools.semantic_search.client.call_messages_parsed",
        fake_parsed,
    )
    monkeypatch.setattr("src.recommender.tools.semantic_search.config.SEMANTIC_BACKEND", "llm")
    plan = _plan(chosen_tool="semantic_search", semantic_query="quiet music")
    out = semantic_search.run(plan, songs)

    assert captured["label"] == "semantic_search"
    assert "quiet music" in captured["user"]
    assert "instrumental=" in captured["user"]
    assert {s["id"] for s in out} == {1, 3}


def test_semantic_search_without_query_returns_all(songs):
    plan = _plan(chosen_tool="semantic_search", semantic_query=None)
    out = semantic_search.run(plan, songs)
    assert len(out) == len(songs)
