"""Agent orchestrator: plan -> act -> check -> refine -> respond."""

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


def _synthesize_empty_verdict() -> CheckVerdict:
    return CheckVerdict(
        confidence=0.0,
        flags=[CheckFlag.LOW_RESULT_COUNT],
        note="No candidates returned by the tool.",
    )


def _has_hard_constraints(plan: Plan) -> bool:
    return any([
        bool(plan.genre_filter),
        bool(plan.mood_filter),
        plan.bpm_min is not None,
        plan.bpm_max is not None,
        plan.energy_min is not None,
        plan.energy_max is not None,
        plan.instrumentalness_min is not None,
        plan.instrumentalness_max is not None,
    ])


def _widen_numeric_range(
    low: Optional[float],
    high: Optional[float],
    *,
    step: float,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    widened_low = low
    widened_high = high
    if low is not None:
        widened_low = low - step
        if floor is not None:
            widened_low = max(floor, widened_low)
    if high is not None:
        widened_high = high + step
        if ceiling is not None:
            widened_high = min(ceiling, widened_high)
    return widened_low, widened_high


def _relax_hybrid_constraints(plan: Plan) -> Optional[Plan]:
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


def _refine_plan(
    plan: Plan,
    candidates: List[Dict[str, Any]],
    verdict: CheckVerdict,
) -> Optional[Plan]:
    flags = set(verdict.flags)

    if not candidates and CheckFlag.LOW_RESULT_COUNT in flags:
        return _relax_hybrid_constraints(plan)

    if CheckFlag.CONSTRAINT_VIOLATED in flags:
        if plan.chosen_tool != "structured_filter" and _has_hard_constraints(plan):
            refined = plan.model_copy(deep=True)
            refined.chosen_tool = "structured_filter"
            refined.semantic_query = None
            if refined != plan:
                return refined

    if verdict.confidence < 0.6 or CheckFlag.INTENT_MISMATCH in flags:
        refined = plan.model_copy(deep=True)
        if refined.chosen_tool == "structured_filter":
            refined.chosen_tool = "hybrid"
            if not refined.semantic_query:
                refined.semantic_query = refined.intent
        elif refined.chosen_tool == "hybrid":
            refined.chosen_tool = "semantic_search"
            if not refined.semantic_query:
                refined.semantic_query = refined.intent
            refined.genre_filter = []
            refined.mood_filter = []
        else:
            refined.semantic_query = refined.semantic_query or refined.intent

        refined.energy_min, refined.energy_max = _widen_numeric_range(
            refined.energy_min, refined.energy_max, step=0.1, floor=0.0, ceiling=1.0
        )
        refined.bpm_min, refined.bpm_max = _widen_numeric_range(
            refined.bpm_min, refined.bpm_max, step=10.0, floor=0.0
        )

        if refined != plan:
            return refined

    return None


def _should_refine(verdict: CheckVerdict) -> bool:
    flags = set(verdict.flags)
    return (
        verdict.confidence < 0.6
        or CheckFlag.LOW_RESULT_COUNT in flags
        or CheckFlag.CONSTRAINT_VIOLATED in flags
        or CheckFlag.INTENT_MISMATCH in flags
    )


def _execute_pass(
    plan: Plan,
    songs: List[Dict[str, Any]],
    trace: TraceRecorder,
    *,
    iteration: int,
) -> tuple[Plan, List[Dict[str, Any]], CheckVerdict]:
    trace.record("plan", iteration=iteration, plan=plan.model_dump())
    candidates = dispatch.call(plan.chosen_tool, plan, songs)
    trace.record(
        "act",
        iteration=iteration,
        tool=plan.chosen_tool,
        candidate_count=len(candidates),
        candidate_ids=[s["id"] for s in candidates],
    )

    verdict = check_candidates(plan, candidates) if candidates else _synthesize_empty_verdict()
    trace.record("check", iteration=iteration, verdict=verdict.model_dump())
    return plan, candidates, verdict


def run(
    query: str,
    profile: Dict[str, Any],
    songs: List[Dict[str, Any]],
    *,
    k: int = 5,
    use_llm_explainer: bool = True,
    trace: Optional[TraceRecorder] = None,
    max_refines: int = 2,
) -> AgentResult:
    """Run the full four-act loop for a single query."""
    if trace is None:
        trace = TraceRecorder(query)

    current_plan = plan_query(query, profile)
    plan_used = current_plan
    candidates: List[Dict[str, Any]] = []
    verdict = _synthesize_empty_verdict()
    seen_plans = {current_plan.model_dump_json()}

    for iteration in range(max_refines + 1):
        plan_used, candidates, verdict = _execute_pass(
            current_plan,
            songs,
            trace,
            iteration=iteration,
        )
        if iteration >= max_refines or not _should_refine(verdict):
            break

        refined_plan = _refine_plan(plan_used, candidates, verdict)
        trace.record(
            "refine",
            iteration=iteration,
            reason=verdict.note,
            confidence=verdict.confidence,
            flags=[flag.value for flag in verdict.flags],
            candidate_count=len(candidates),
            from_tool=plan_used.chosen_tool,
            to_tool=refined_plan.chosen_tool if refined_plan else None,
            refined_plan=refined_plan.model_dump() if refined_plan else None,
        )
        if refined_plan is None:
            break

        refined_key = refined_plan.model_dump_json()
        if refined_key in seen_plans:
            trace.record(
                "refine_stop",
                iteration=iteration,
                reason="Refined plan matches a previous plan.",
            )
            break
        seen_plans.add(refined_key)
        current_plan = refined_plan

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
