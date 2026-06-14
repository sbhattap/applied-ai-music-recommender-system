"""Agent orchestrator: plan -> act -> check -> refine -> respond."""

from typing import Any, Dict, List, Optional

from .. import config
from ..core.rerank import rerank
from ..rag import citations
from ..rag.documents import Document
from ..rag.retriever import SongHit, get_retriever
from ..tools import dispatch, structured_filter
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
    # Missing keys stay None: the reranker ignores preferences the listener never stated.
    return {
        "genre": profile.get("genre"),
        "mood": profile.get("mood"),
        "energy": profile.get("energy"),
        "likes_acoustic": profile.get("likes_acoustic"),
        "target_valence": profile.get("target_valence"),
        "prefer_popular": profile.get("prefer_popular"),
    }


GENRE_FOCUS_SHARE = 0.6


def genre_cap(plan: Plan, hits: Dict[int, SongHit], songs_by_id: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Per-genre diversity cap, or None when the request is about a genre.

    Off if the plan filters on genre, or if most of the top retrieved songs share one
    genre (the query itself targets it, e.g. "heavy metal").
    """
    if plan.genre_filter:
        return None
    top = sorted(hits.values(), key=lambda h: -h.rrf)[:10]
    genres = [songs_by_id[h.song_id]["genre"] for h in top if h.song_id in songs_by_id]
    if genres and max(genres.count(g) for g in set(genres)) / len(genres) >= GENRE_FOCUS_SHARE:
        return None
    return 2


def _profile_summary(profile: Dict[str, Any]) -> str:
    return (
        f"genre={profile.get('genre')}, mood={profile.get('mood')}, "
        f"energy={profile.get('energy')}, "
        f"likes_acoustic={profile.get('likes_acoustic', False)}"
    )


def _candidate_summary(
    songs: List[Dict[str, Any]],
    hits: Optional[Dict[int, SongHit]] = None,
    limit: int = 12,
) -> str:
    hits = hits or {}
    sample = songs[:limit]

    def retrieval(s: Dict[str, Any]) -> str:
        h = hits.get(s["id"])
        return f", query_match={h.query_match:.2f} ({h.found_by})" if h else ""

    lines = [
        f"  id={s['id']:<3} | {s['title']!r} | genre={s['genre']}, "
        f"mood={s['mood']}, bpm={s['tempo_bpm']:.0f}, energy={s['energy']:.2f}, "
        f"instrumental={s.get('instrumentalness', 0.0):.2f}{retrieval(s)}"
        for s in sample
    ]
    suffix = "" if len(songs) <= limit else f"\n  ... and {len(songs) - limit} more"
    return "\n".join(lines) + suffix


_INSTRUMENTAL_CUES = ("no vocals", "instrumental", "no lyrics", "lyrics-free", "without vocals")


def heuristic_plan(query: str) -> Plan:
    """Offline planner used when the LLM is unavailable: semantic retrieval on the raw
    query, plus the one hard constraint that's easy to spot reliably (instrumental)."""
    lowered = query.lower()
    instrumental = any(cue in lowered for cue in _INSTRUMENTAL_CUES)
    return Plan(
        intent=query.strip(),
        chosen_tool="hybrid" if instrumental else "semantic_search",
        expected_count_min=3,
        expected_count_max=12,
        semantic_query=query.strip(),
        instrumentalness_min=0.8 if instrumental else None,
    )


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
        effort=config.PLAN_EFFORT,
        temperature=config.PLAN_TEMPERATURE,
    )


def check_candidates(
    plan: Plan,
    candidates: List[Dict[str, Any]],
    hits: Optional[Dict[int, SongHit]] = None,
) -> CheckVerdict:
    """Act 3: judge whether the candidates plausibly satisfy the plan."""
    system = client.load_prompt("check")
    user = (
        f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
        f"Candidates returned ({len(candidates)} songs):\n{_candidate_summary(candidates, hits)}\n\n"
        f"Emit the CheckVerdict."
    )
    return client.call_messages_parsed(
        system=system,
        user=user,
        label="check",
        schema=CheckVerdict,
        max_tokens=config.CHECK_MAX_TOKENS,
        effort=config.CHECK_EFFORT,
        temperature=config.CHECK_TEMPERATURE,
        model=config.FAST_MODEL,
    )


def _synthesize_empty_verdict() -> CheckVerdict:
    return CheckVerdict(
        confidence=0.0,
        flags=[CheckFlag.LOW_RESULT_COUNT],
        note="No candidates returned by the tool.",
    )


def deterministic_verdict(plan: Plan, candidates: List[Dict[str, Any]]) -> Optional[CheckVerdict]:
    """Verdict we can compute without the LLM, or None when judgment is needed.

    structured_filter enforces every constraint by construction, so the only open
    question is count. Semantic/hybrid results still need the LLM to judge intent fit.
    """
    if plan.chosen_tool != "structured_filter":
        return None
    if len(candidates) >= plan.expected_count_min:
        return CheckVerdict(
            confidence=0.8,
            flags=[CheckFlag.LOOKS_GOOD],
            note=f"Deterministic check: all constraints enforced by the filter; "
                 f"{len(candidates)} candidates within expected range.",
        )
    return CheckVerdict(
        confidence=0.45,
        flags=[CheckFlag.LOW_RESULT_COUNT],
        note=f"Deterministic check: only {len(candidates)} candidates, "
             f"expected at least {plan.expected_count_min}.",
    )


def _count_only_verdict(plan: Plan, candidates: List[Dict[str, Any]]) -> CheckVerdict:
    """Fallback when the LLM check fails: judge on count alone, at middling confidence."""
    short = len(candidates) < plan.expected_count_min
    return CheckVerdict(
        confidence=0.45 if short else 0.6,
        flags=[CheckFlag.LOW_RESULT_COUNT] if short else [],
        note="LLM check unavailable; verdict based on candidate count only.",
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


def _ground(
    query: str,
    plan: Plan,
    songs: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Dict[int, SongHit]:
    """Score every candidate against the query (and taste, if known) with the hybrid
    retriever. Runs for all tools, so structured-filter results get evidence too."""
    if not candidates:
        return {}
    retriever = get_retriever(songs, _history_docs(profile))
    hits = retriever.search(
        plan.semantic_query or query,
        candidates,
        taste_vector=profile.get("taste_vector"),
    )
    return {h.song_id: h for h in hits}


PERSONAL_CANDIDATES = 6
TASTE_BLEND = 0.6


def _personal_candidates(
    query: str,
    plan: Plan,
    songs: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extra candidates retrieved with the query pulled toward the listener's taste.

    Query-only retrieval never surfaces songs that fit the *listener* but not the
    query wording. Structured constraints still hold: for hybrid the pool is the
    filtered set; structured_filter results are left alone.
    """
    taste = profile.get("taste_vector")
    if taste is None or plan.chosen_tool == "structured_filter":
        return []
    pool = structured_filter.run(plan, songs) if plan.chosen_tool == "hybrid" else songs
    have = {s["id"] for s in candidates}
    hits = get_retriever(songs, _history_docs(profile)).search(
        plan.semantic_query or query, pool, top_n=PERSONAL_CANDIDATES + len(have),
        taste_vector=taste, taste_blend=TASTE_BLEND,
    )
    by_id = {s["id"]: s for s in pool}
    return [by_id[h.song_id] for h in hits if h.song_id not in have][:PERSONAL_CANDIDATES]


def _history_docs(profile: Dict[str, Any]) -> List[Document]:
    return [Document(**d) for d in profile.get("history_docs") or []]


def _execute_pass(
    plan: Plan,
    songs: List[Dict[str, Any]],
    trace: TraceRecorder,
    *,
    query: str,
    profile: Dict[str, Any],
    iteration: int,
    fast_check: bool,
    llm_available: bool,
) -> tuple[Plan, List[Dict[str, Any]], CheckVerdict, Dict[int, SongHit]]:
    trace.record("plan", iteration=iteration, plan=plan.model_dump())
    with trace.stage("retrieve"):
        candidates = dispatch.call(plan.chosen_tool, plan, songs)
        personal = _personal_candidates(query, plan, songs, candidates, profile)
        if personal:
            trace.record("personalize", iteration=iteration, added_ids=[s["id"] for s in personal])
        candidates = candidates + personal
        hits = _ground(query, plan, songs, candidates, profile)
    trace.record(
        "act",
        iteration=iteration,
        tool=plan.chosen_tool,
        candidate_count=len(candidates),
        candidate_ids=[s["id"] for s in candidates],
        retrieval=[
            {k: v for k, v in h.to_dict().items() if k != "evidence"}
            for h in sorted(hits.values(), key=lambda h: -h.rrf)[:10]
        ],
    )

    with trace.stage("check"):
        verdict, source = _check(plan, candidates, hits, fast_check=fast_check, llm_available=llm_available)
    trace.record("check", iteration=iteration, source=source, verdict=verdict.model_dump())
    return plan, candidates, verdict, hits


def _check(
    plan: Plan,
    candidates: List[Dict[str, Any]],
    hits: Dict[int, SongHit],
    *,
    fast_check: bool,
    llm_available: bool,
) -> tuple[CheckVerdict, str]:
    """Cheapest verdict that's still honest: synthesized → deterministic → LLM → count-only."""
    if not candidates:
        return _synthesize_empty_verdict(), "synthesized"
    if fast_check:
        verdict = deterministic_verdict(plan, candidates)
        if verdict is not None:
            return verdict, "deterministic"
    if llm_available:
        try:
            return check_candidates(plan, candidates, hits), "llm"
        except Exception:
            pass
    return _count_only_verdict(plan, candidates), "count_only"


def _plan(query: str, profile: Dict[str, Any], trace: TraceRecorder, llm_available: bool) -> Plan:
    with trace.stage("plan"):
        if llm_available:
            try:
                return plan_query(query, profile)
            except Exception as e:
                trace.record("plan_fallback", error=str(e))
        else:
            trace.record("plan_fallback", error="LLM not configured")
        return heuristic_plan(query)


def run(
    query: str,
    profile: Dict[str, Any],
    songs: List[Dict[str, Any]],
    *,
    k: int = 5,
    use_llm_explainer: bool = True,
    trace: Optional[TraceRecorder] = None,
    max_refines: int = config.MAX_REFINES,
    fast_check: bool = True,
) -> AgentResult:
    """Run plan -> act -> check -> refine -> rank, then (optionally) explain.

    Set ``use_llm_explainer=False`` to get ranked results with templated narratives
    right away, then call ``explain_result`` to fill in LLM narratives; the UI does
    this so cards render before the explainer finishes.
    """
    if trace is None:
        trace = TraceRecorder(query)
    llm_available = client.is_configured()

    current_plan = _plan(query, profile, trace, llm_available)
    plan_used = current_plan
    candidates: List[Dict[str, Any]] = []
    hits: Dict[int, SongHit] = {}
    verdict = _synthesize_empty_verdict()
    seen_plans = {current_plan.model_dump_json()}

    for iteration in range(max_refines + 1):
        plan_used, candidates, verdict, hits = _execute_pass(
            current_plan,
            songs,
            trace,
            query=query,
            profile=profile,
            iteration=iteration,
            fast_check=fast_check,
            llm_available=llm_available,
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

    with trace.stage("rank"):
        cap = genre_cap(plan_used, hits, {s["id"]: s for s in candidates})
        trace.record("rank", diversity_cap=cap,
                     reason="per-genre cap" if cap else "request targets a genre; no cap")
        recommendations = _rank(profile, candidates, hits, verdict, k=k, max_per_genre=cap)

    result = AgentResult(
        query=query,
        profile=profile,
        plan=plan_used,
        candidate_count=len(candidates),
        verdict=verdict,
        recommendations=recommendations,
        candidates=candidates,
        hits={sid: h.to_dict() for sid, h in hits.items()},
    )
    if use_llm_explainer and llm_available:
        explain_result(result, trace=trace)
    trace.record("respond", recommendation_count=len(recommendations))
    result.trace_events = trace.snapshot()
    result.timings_ms = {**trace.timings, "total": trace.total_ms()}
    return result


def _rank(
    profile: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    hits: Dict[int, SongHit],
    verdict: CheckVerdict,
    *,
    k: int,
    max_per_genre: Optional[int] = 2,
) -> List[Recommendation]:
    """Additive rerank, then attach evidence and a grounded (non-LLM) explanation."""
    if not candidates:
        return []
    recently_played = profile.get("recently_played")
    ranked = rerank(
        _profile_to_prefs(profile),
        candidates,
        {sid: h.to_dict() for sid, h in hits.items()},
        k=k,
        has_taste=profile.get("taste_vector") is not None,
        recently_played=set(recently_played) if recently_played is not None else None,
        max_per_genre=max_per_genre,
    )
    low_confidence = _is_low_confidence(verdict)
    out = []
    for song, score, signals, breakdown_text in ranked:
        hit = hits.get(song["id"])
        signal_dicts = [s.to_dict() for s in signals]
        evidence = [vars(e).copy() for e in hit.evidence] if hit else []
        claims = citations.grounded_claims(signal_dicts, evidence, low_confidence=low_confidence)
        out.append(Recommendation(
            song=song,
            score=score,
            signal_explanation=breakdown_text,
            narrative=citations.render(claims),
            signals=signal_dicts,
            evidence=evidence,
            retrieval={k_: v for k_, v in hit.to_dict().items() if k_ != "evidence"} if hit else None,
            claims=claims,
        ))
    return out


def what_if(result: AgentResult, exclude: set, *, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Re-rank the final candidate pool without some signals. Returns rank movements."""
    profile = result.profile
    recently_played = profile.get("recently_played")
    rank_event = next((e for e in result.trace_events if e["kind"] == "rank"), {})
    kwargs = dict(
        has_taste=profile.get("taste_vector") is not None,
        recently_played=set(recently_played) if recently_played is not None else None,
        max_per_genre=rank_event.get("diversity_cap", 2),
    )
    k = k or len(result.recommendations)
    prefs = _profile_to_prefs(profile)
    before = rerank(prefs, result.candidates, result.hits, k=k, **kwargs)
    after = rerank(prefs, result.candidates, result.hits, k=k, exclude=exclude, **kwargs)
    old_rank = {song["id"]: i for i, (song, *_rest) in enumerate(before, 1)}
    return [
        {
            "rank": i,
            "song_id": song["id"],
            "title": song["title"],
            "score": score,
            "previous_rank": old_rank.get(song["id"]),
        }
        for i, (song, score, *_rest) in enumerate(after, 1)
    ]


def _is_low_confidence(verdict: CheckVerdict) -> bool:
    return verdict.confidence < 0.6 or bool(
        {CheckFlag.INTENT_MISMATCH, CheckFlag.CONSTRAINT_VIOLATED} & set(verdict.flags)
    )


def explain_result(result: AgentResult, *, trace: Optional[TraceRecorder] = None) -> AgentResult:
    """Replace templated narratives with LLM ones, in a single batched call.

    Songs the LLM skips, or every song if the call fails, keep the templated text.
    Called standalone (no ``trace``), it appends its events/timing to ``result``.
    """
    standalone = trace is None
    trace = trace or TraceRecorder(result.query)
    if not result.recommendations:
        return result
    try:
        _explain_into(result, trace)
    finally:
        if standalone:
            result.trace_events.extend(trace.snapshot())
            result.timings_ms["explain"] = trace.timings.get("explain", 0)
            result.timings_ms["total"] = result.timings_ms.get("total", 0) + result.timings_ms["explain"]
    return result


def _explain_into(result: AgentResult, trace: TraceRecorder) -> None:
    items = [
        {"song": rec.song, "score": rec.score, "signals": rec.signals, "evidence": rec.evidence}
        for rec in result.recommendations
    ]
    with trace.stage("explain"):
        try:
            raw = explainer.explain_batch(
                query=result.query,
                profile_summary=_profile_summary(result.profile),
                items=items,
                verdict=result.verdict,
            )
        except Exception as e:
            trace.record("explainer_fallback", error=str(e))
            return

    kept_total = dropped_total = 0
    fallback_ids = []
    for rec in result.recommendations:
        allowed = citations.evidence_ids(rec.signals, rec.evidence)
        kept, dropped = citations.validate(raw.get(rec.song["id"], []), allowed)
        kept_total += len(kept)
        dropped_total += len(dropped)
        rec.dropped_claims = dropped
        if kept:
            rec.claims = kept
            rec.narrative = citations.render(kept)
        else:
            fallback_ids.append(rec.song["id"])  # keep the grounded claims from _rank
    trace.record(
        "citations",
        kept=kept_total,
        dropped=dropped_total,
        dropped_claims=[
            {"song_id": rec.song["id"], **d} for rec in result.recommendations for d in rec.dropped_claims
        ],
    )
    if fallback_ids:
        trace.record("explainer_fallback", song_ids=fallback_ids, error="no valid cited claims returned")
    result.narratives_from_llm = len(fallback_ids) < len(result.recommendations)
