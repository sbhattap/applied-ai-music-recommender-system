"""End-to-end tests for the four-act loop with the LLM mocked."""

import pytest

from src.recommender.llm import agent
from src.recommender.llm.types import CheckFlag, CheckVerdict, Plan


@pytest.fixture
def songs():
    return [
        {"id": 1, "title": "Lofi Loop", "artist": "A", "genre": "lofi",
         "mood": "chill", "mood_tags": "chill|study", "energy": 0.30,
         "tempo_bpm": 75, "valence": 0.5, "danceability": 0.4, "acousticness": 0.7,
         "instrumentalness": 0.92,
         "popularity": 60, "release_decade": "2020s", "liveness": 0.1,
         "duration_sec": 200, "explicit": 0},
        {"id": 2, "title": "Library Rain", "artist": "B", "genre": "lofi",
         "mood": "chill", "mood_tags": "chill|focus", "energy": 0.25,
         "tempo_bpm": 70, "valence": 0.4, "danceability": 0.3, "acousticness": 0.8,
         "instrumentalness": 0.95,
         "popularity": 55, "release_decade": "2020s", "liveness": 0.1,
         "duration_sec": 220, "explicit": 0},
        {"id": 3, "title": "Hard Rock", "artist": "C", "genre": "rock",
         "mood": "intense", "mood_tags": "intense|driving", "energy": 0.95,
         "tempo_bpm": 140, "valence": 0.6, "danceability": 0.5, "acousticness": 0.05,
         "instrumentalness": 0.01,
         "popularity": 70, "release_decade": "2010s", "liveness": 0.2,
         "duration_sec": 200, "explicit": 0},
    ]


@pytest.fixture
def profile():
    return {
        "name": "Chill Lofi",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
    }


def _stub_llm(monkeypatch, *, plan: Plan, verdict: CheckVerdict, narrative: str = "Because lofi."):
    calls = []

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        calls.append({"label": label, "schema": schema.__name__})
        if schema.__name__ == "Plan":
            return plan
        if schema.__name__ == "CheckVerdict":
            return verdict
        raise AssertionError(f"unexpected schema: {schema.__name__}")

    def fake_text(*, system, user, label, max_tokens, **kwargs):
        calls.append({"label": label})
        return narrative

    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages", fake_text)
    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", fake_text)
    return calls


def test_full_loop_structured_filter_path(monkeypatch, songs, profile):
    plan = Plan(
        intent="chill lofi for studying",
        chosen_tool="structured_filter",
        expected_count_min=2, expected_count_max=5,
        genre_filter=["lofi"],
    )
    verdict = CheckVerdict(confidence=0.85, flags=[CheckFlag.LOOKS_GOOD],
                           note="lofi tracks for chill study match well")
    calls = _stub_llm(monkeypatch, plan=plan, verdict=verdict)

    result = agent.run("chill study music", profile, songs, k=3)

    assert result.plan.chosen_tool == "structured_filter"
    assert result.candidate_count == 2
    assert result.verdict.confidence == pytest.approx(0.85)
    assert len(result.recommendations) == 2  # only 2 lofi songs in the catalog
    # plan, check, and one explainer call per recommendation
    labels = [c["label"] for c in calls]
    assert labels.count("plan") == 1
    assert labels.count("check") == 1
    assert labels.count("explainer") == 2


def test_loop_emits_plan_act_check_respond_in_order(monkeypatch, songs, profile):
    plan = Plan(
        intent="lofi", chosen_tool="structured_filter",
        expected_count_min=1, expected_count_max=5, genre_filter=["lofi"],
    )
    verdict = CheckVerdict(confidence=0.8, flags=[], note="ok")
    _stub_llm(monkeypatch, plan=plan, verdict=verdict)

    result = agent.run("anything", profile, songs)
    kinds = [ev["kind"] for ev in result.trace_events]
    # plan, act, check, respond — in that order
    assert kinds[0] == "plan"
    assert kinds[1] == "act"
    assert kinds[2] == "check"
    assert kinds[-1] == "respond"


def test_check_step_skipped_when_no_candidates(monkeypatch, songs, profile):
    """If the tool returns no candidates, we synthesize a low-confidence verdict
    instead of calling the LLM check (to save the round trip)."""
    plan = Plan(
        intent="impossible", chosen_tool="structured_filter",
        expected_count_min=5, expected_count_max=10,
        genre_filter=["nonexistent_genre"],
    )
    # The check verdict here would never be emitted by the LLM — assert it's synthesized
    verdict_should_not_be_used = CheckVerdict(confidence=1.0, flags=[], note="LLM check ran")
    calls = _stub_llm(monkeypatch, plan=plan, verdict=verdict_should_not_be_used)

    result = agent.run("songs from a fake genre", profile, songs)
    assert result.candidate_count == 0
    assert result.verdict.confidence == 0.0
    assert CheckFlag.LOW_RESULT_COUNT in result.verdict.flags
    # 'check' label should never appear — we short-circuited
    assert all(c["label"] != "check" for c in calls)


def test_explainer_fallback_on_llm_failure(monkeypatch, songs, profile):
    plan = Plan(
        intent="lofi", chosen_tool="structured_filter",
        expected_count_min=1, expected_count_max=5, genre_filter=["lofi"],
    )
    verdict = CheckVerdict(confidence=0.8, flags=[], note="ok")

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        if schema.__name__ == "Plan":
            return plan
        if schema.__name__ == "CheckVerdict":
            return verdict
        raise AssertionError(schema.__name__)

    def fake_text_explode(*, system, user, label, max_tokens, **kwargs):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", fake_text_explode)

    result = agent.run("chill music", profile, songs)
    assert len(result.recommendations) == 2
    # All narratives should be the templated fallback (no LLM text)
    for rec in result.recommendations:
        assert rec.narrative  # non-empty
    # And the trace should record the fallback
    assert any(ev["kind"] == "explainer_fallback" for ev in result.trace_events)


def test_hybrid_retries_with_relaxed_vibe_filters_after_zero_candidates(monkeypatch, songs, profile):
    plan = Plan(
        intent="instrumental chill study music",
        chosen_tool="hybrid",
        expected_count_min=2,
        expected_count_max=5,
        semantic_query="chill study music with no vocals",
        mood_filter=["chill"],
        instrumentalness_min=0.9,
    )
    verdict = CheckVerdict(confidence=0.75, flags=[], note="reasonable instrumental study matches")

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        if schema.__name__ == "Plan":
            return plan
        if schema.__name__ == "CheckVerdict":
            return verdict
        raise AssertionError(schema.__name__)

    def fake_semantic(plan_arg, songs_arg):
        if plan_arg.mood_filter:
            return []
        return [s for s in songs_arg if s["id"] == 1]

    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", lambda **kwargs: "Because instrumental.")
    monkeypatch.setattr("src.recommender.tools.hybrid.semantic_search.run", fake_semantic)

    result = agent.run("chill study music, no vocals", profile, songs)

    assert result.candidate_count == 1
    assert result.plan.mood_filter == []
    assert result.recommendations[0].song["id"] == 1
    assert any(ev["kind"] == "refine" for ev in result.trace_events)


def test_refine_switches_structured_filter_to_hybrid_on_intent_mismatch(monkeypatch, songs, profile):
    plan = Plan(
        intent="chill study music",
        chosen_tool="structured_filter",
        expected_count_min=1,
        expected_count_max=5,
        mood_filter=["focus"],
        energy_max=0.3,
    )
    verdicts = iter([
        CheckVerdict(confidence=0.45, flags=[CheckFlag.INTENT_MISMATCH], note="too narrow"),
        CheckVerdict(confidence=0.82, flags=[CheckFlag.LOOKS_GOOD], note="better fit"),
    ])

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        if schema.__name__ == "Plan":
            return plan
        if schema.__name__ == "CheckVerdict":
            return next(verdicts)
        raise AssertionError(schema.__name__)

    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", lambda **kwargs: "Because better.")
    monkeypatch.setattr(
        "src.recommender.tools.hybrid.semantic_search.run",
        lambda plan_arg, songs_arg: songs_arg,
    )

    result = agent.run("chill study music", profile, songs)

    assert result.plan.chosen_tool == "hybrid"
    assert result.plan.semantic_query == "chill study music"
    assert result.plan.energy_max == pytest.approx(0.4)
    refine_events = [ev for ev in result.trace_events if ev["kind"] == "refine"]
    assert refine_events
    assert refine_events[0]["from_tool"] == "structured_filter"
    assert refine_events[0]["to_tool"] == "hybrid"


def test_refine_switches_to_structured_filter_on_constraint_violated(monkeypatch, songs, profile):
    plan = Plan(
        intent="instrumental chill study music",
        chosen_tool="semantic_search",
        expected_count_min=1,
        expected_count_max=5,
        semantic_query="instrumental chill study music",
        instrumentalness_min=0.9,
    )
    verdicts = iter([
        CheckVerdict(confidence=0.55, flags=[CheckFlag.CONSTRAINT_VIOLATED], note="returned vocals"),
        CheckVerdict(confidence=0.85, flags=[CheckFlag.LOOKS_GOOD], note="constraints enforced"),
    ])

    def fake_parsed(*, system, user, label, schema, max_tokens, **kwargs):
        if schema.__name__ == "Plan":
            return plan
        if schema.__name__ == "CheckVerdict":
            return next(verdicts)
        raise AssertionError(schema.__name__)

    monkeypatch.setattr("src.recommender.llm.agent.client.call_messages_parsed", fake_parsed)
    monkeypatch.setattr("src.recommender.llm.explainer.client.call_messages", lambda **kwargs: "Because instrumental.")
    monkeypatch.setitem(agent.dispatch.TOOLS, "semantic_search", lambda plan_arg, songs_arg: songs_arg)

    result = agent.run("chill study music, no vocals", profile, songs)

    assert result.plan.chosen_tool == "structured_filter"
    refine_events = [ev for ev in result.trace_events if ev["kind"] == "refine"]
    assert refine_events
    assert refine_events[0]["to_tool"] == "structured_filter"
