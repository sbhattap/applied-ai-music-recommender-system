"""Agent orchestrator: plan -> act -> check -> respond.

The four-act loop, with a small recovery step when hybrid retrieval becomes too
strict for a small catalog.
"""

from typing import Any, Dict, List, Optional

from .. import config
from ..core.ranking import recommend_songs
from ..tools import dispatch
from ..trace.recorder import TraceRecorder
from . import client, explainer
from .types import (
    AgentResult,
    CheckFlag,
    CheckVerdict,
    Plan,
    Recommendation,
)


def _profile_to_prefs(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "genre": profile.get("genre"),
        "mood": profile.get("mood"),
        "energy": profile.get("energy"),
        "likes_acoustic": profile.get("likes_acoustic", False),
        "target_valence": profile.get("target_valence", 0.5),
        "prefer_popular": profile.get("prefer_popular", True),
    }


def _profile_summary(profile: Dict[str, Any]) -> str:
    return (
        f"genre={profile.get('genre')}, mood={profile.get('mood')}, "
        f"energy={profile.get('energy')}, "
        f"likes_acoustic={profile.get('likes_acoustic', False)}"
    )


def _candidate_summary(songs: List[Dict[str, Any]], limit: int = 12) -> str:
    sample = songs[:limit]
    lines = [
        f"  id={s['id']:<3} | {s['title']!r} | genre={s['genre']}, "
        f"mood={s['mood']}, bpm={s['tempo_bpm']:.0f}, energy={s['energy']:.2f}, "
        f"instrumental={s.get('instrumentalness', 0.0):.2f}"
        for s in sample
    ]
    suffix = "" if len(songs) <= limit else f"\n  ... and {len(songs) - limit} more"
    return "\n".join(lines) + suffix


def plan_query(query: str, profile: Dict[str, Any]) -> Plan:
    """Act 1: parse the query into a structured Plan."""
    system = client.load_prompt("plan")
    user = (
        f"Query: {query}\n"
        f"Listener profile: {_profile_summary(profile)}\n\n"
        f"Emit the Plan."
    )
    return client.call_messages_parsed(
        system=system,
        user=user,
        label="plan",
        schema=Plan,
        max_tokens=config.PLAN_MAX_TOKENS,
        thinking=True,
    )


def check_candidates(plan: Plan, candidates: List[Dict[str, Any]]) -> CheckVerdict:
    """Act 3: judge whether the candidates plausibly satisfy the plan."""
    system = client.load_prompt("check")
    user = (
        f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
        f"Candidates returned ({len(candidates)} songs):\n{_candidate_summary(candidates)}\n\n"
        f"Emit the CheckVerdict."
    )
    return client.call_messages_parsed(
        system=system,
        user=user,
        label="check",
        schema=CheckVerdict,
        max_tokens=config.CHECK_MAX_TOKENS,
    )


def _relaxed_hybrid_plan(plan: Plan) -> Optional[Plan]:
    """Retry hybrid with only hard constraints preserved.

    In small catalogs, hybrid can become too narrow when the plan contains both
    semantic intent and structured vibe hints. Keep explicit "hard" constraints
    like instrumentalness/BPM, but let semantic_search carry the vibe.
    """
    if plan.chosen_tool != "hybrid":
        return None

    relaxed = plan.model_copy(deep=True)
    relaxed.genre_filter = []
    relaxed.mood_filter = []
    relaxed.energy_min = None
    relaxed.energy_max = None

    if relaxed == plan:
        return None
    return relaxed


def run(
    query: str,
    profile: Dict[str, Any],
    songs: List[Dict[str, Any]],
    *,
    k: int = 5,
    use_llm_explainer: bool = True,
    trace: Optional[TraceRecorder] = None,
) -> AgentResult:
    """Run the full four-act loop for a single query."""
    if trace is None:
        trace = TraceRecorder(query)

    # Act 1 - Plan
    plan = plan_query(query, profile)
    trace.record("plan", plan=plan.model_dump())

    # Act 2 - Act (call the chosen tool)
    plan_used = plan
    candidates = dispatch.call(plan_used.chosen_tool, plan_used, songs)
    if not candidates:
        relaxed_plan = _relaxed_hybrid_plan(plan)
        if relaxed_plan is not None:
            relaxed_candidates = dispatch.call(relaxed_plan.chosen_tool, relaxed_plan, songs)
            if relaxed_candidates:
                plan_used = relaxed_plan
                candidates = relaxed_candidates
                trace.record(
                    "act_retry",
                    reason="relaxed hybrid constraints after zero-candidate first pass",
                    tool=relaxed_plan.chosen_tool,
                    candidate_count=len(candidates),
                    candidate_ids=[s["id"] for s in candidates],
                    relaxed_fields=["genre_filter", "mood_filter", "energy_min", "energy_max"],
                )
    trace.record(
        "act",
        tool=plan_used.chosen_tool,
        candidate_count=len(candidates),
        candidate_ids=[s["id"] for s in candidates],
    )

    # Act 3 - Check (passive)
    if candidates:
        verdict = check_candidates(plan_used, candidates)
    else:
        verdict = CheckVerdict(
            confidence=0.0,
            flags=[CheckFlag.LOW_RESULT_COUNT],
            note="No candidates returned by the tool.",
        )
    trace.record("check", verdict=verdict.model_dump())

    # Act 4 - Respond: deterministic re-rank, then LLM explanations
    prefs = _profile_to_prefs(profile)
    ranked = recommend_songs(prefs, candidates, k=k) if candidates else []
    profile_summary = _profile_summary(profile)

    recommendations: List[Recommendation] = []
    for song, score, breakdown in ranked:
        if use_llm_explainer:
            try:
                narrative = explainer.explain(
                    query=query,
                    profile_summary=profile_summary,
                    song=song,
                    score=score,
                    breakdown=breakdown,
                    verdict=verdict,
                )
            except Exception as e:
                trace.record("explainer_fallback", song_id=song["id"], error=str(e))
                narrative = explainer.templated_fallback(
                    song=song, breakdown=breakdown, verdict=verdict
                )
        else:
            narrative = explainer.templated_fallback(
                song=song, breakdown=breakdown, verdict=verdict
            )
        recommendations.append(Recommendation(
            song=song, score=score, signal_explanation=breakdown, narrative=narrative,
        ))
    trace.record("respond", recommendation_count=len(recommendations))

    return AgentResult(
        query=query,
        profile=profile,
        plan=plan_used,
        candidate_count=len(candidates),
        verdict=verdict,
        recommendations=recommendations,
        trace_events=trace.snapshot(),
    )
