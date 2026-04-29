"""Deterministic filter on song attributes. Pure logic — no LLM."""

from typing import Any, Dict, List

from ..llm.types import Plan


def run(plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the songs that satisfy every constraint declared in the plan."""
    out: List[Dict[str, Any]] = []
    genres = {g.lower() for g in plan.genre_filter}
    moods = {m.lower() for m in plan.mood_filter}

    for song in songs:
        if genres and song["genre"].lower() not in genres:
            continue
        if moods:
            primary = song["mood"].lower()
            tags = {t.lower() for t in song.get("mood_tags", "").split("|") if t}
            if primary not in moods and not (tags & moods):
                continue
        if plan.bpm_min is not None and song["tempo_bpm"] < plan.bpm_min:
            continue
        if plan.bpm_max is not None and song["tempo_bpm"] > plan.bpm_max:
            continue
        if plan.energy_min is not None and song["energy"] < plan.energy_min:
            continue
        if plan.energy_max is not None and song["energy"] > plan.energy_max:
            continue
        instrumentalness = song.get("instrumentalness", 0.0)
        if plan.instrumentalness_min is not None and instrumentalness < plan.instrumentalness_min:
            continue
        if plan.instrumentalness_max is not None and instrumentalness > plan.instrumentalness_max:
            continue
        out.append(song)
    return out
