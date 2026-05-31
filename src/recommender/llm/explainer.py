"""Batched narrative explainer.

One LLM call covers every recommended song (previously one call per song). Narratives
are conditioned on the deterministic score breakdown AND the agent's check verdict, so
they can hedge ("closest match available, though the catalog leans X") when the check
flagged low confidence.
"""

from typing import Any, Dict, List

from .. import config
from . import client
from .types import BatchNarratives, CheckVerdict


def _format_song(song: Dict[str, Any], score: float, breakdown: str) -> str:
    return (
        f"- song_id={song['id']}: {song['title']!r} by {song['artist']} "
        f"(genre={song['genre']}, mood={song['mood']}, "
        f"bpm={song['tempo_bpm']:.0f}, energy={song['energy']:.2f})\n"
        f"  score={score:.3f}; breakdown: {breakdown}"
    )


def explain_batch(
    *,
    query: str,
    profile_summary: str,
    items: List[Dict[str, Any]],
    verdict: CheckVerdict,
) -> Dict[int, str]:
    """Return ``{song_id: narrative}`` for every item in one call.

    ``items`` are ``{"song", "score", "breakdown"}`` dicts. Songs the model skipped are
    simply missing from the result; the caller falls back to the template for those.
    """
    system = client.load_prompt("explainer")
    songs_block = "\n".join(
        _format_song(it["song"], it["score"], it["breakdown"]) for it in items
    )
    user = (
        f"User query: {query}\n"
        f"Profile: {profile_summary}\n\n"
        f"Check verdict: confidence={verdict.confidence:.2f}, "
        f"flags={[f.value for f in verdict.flags]}, note={verdict.note!r}\n\n"
        f"Recommended songs:\n{songs_block}\n\n"
        f"Write one narrative per song."
    )
    out = client.call_messages_parsed(
        system=system,
        user=user,
        label="explainer",
        schema=BatchNarratives,
        max_tokens=config.EXPLAINER_MAX_TOKENS,
        effort=config.EXPLAINER_EFFORT,
        temperature=config.EXPLAINER_TEMPERATURE,
        model=config.FAST_MODEL,
    )
    wanted = {it["song"]["id"] for it in items}
    return {n.song_id: n.narrative.strip() for n in out.narratives if n.song_id in wanted}


def templated_fallback(
    *,
    song: Dict[str, Any],
    breakdown: str,
    verdict: CheckVerdict,
) -> str:
    """Used when the LLM is unavailable. Reuses the deterministic breakdown text."""
    if verdict.confidence < 0.6:
        prefix = "Closest match available — "
    else:
        prefix = ""
    return prefix + breakdown
