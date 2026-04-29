"""Per-recommendation narrative explainer.

Conditions the narrative on the deterministic score breakdown AND the agent's check
verdict, so it can hedge ("closest match available, though the catalog leans X")
when the check flagged low confidence.
"""

from typing import Any, Dict

from .. import config
from . import client
from .types import CheckVerdict


def _format_breakdown(breakdown: str) -> str:
    return breakdown


def explain(
    *,
    query: str,
    profile_summary: str,
    song: Dict[str, Any],
    score: float,
    breakdown: str,
    verdict: CheckVerdict,
) -> str:
    """One- or two-sentence narrative for a single recommended song."""
    system = client.load_prompt("explainer")
    user = (
        f"User query: {query}\n"
        f"Profile: {profile_summary}\n\n"
        f"Song: {song['title']!r} by {song['artist']} "
        f"(genre={song['genre']}, mood={song['mood']}, "
        f"bpm={song['tempo_bpm']:.0f}, energy={song['energy']:.2f})\n"
        f"Deterministic score: {score:.3f}\n"
        f"Score breakdown:\n{_format_breakdown(breakdown)}\n\n"
        f"Check verdict: confidence={verdict.confidence:.2f}, "
        f"flags={[f.value for f in verdict.flags]}, note={verdict.note!r}\n\n"
        f"Write the narrative now."
    )
    return client.call_messages(
        system=system,
        user=user,
        label="explainer",
        max_tokens=config.EXPLAINER_MAX_TOKENS,
    )


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
