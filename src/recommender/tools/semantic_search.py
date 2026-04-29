"""LLM-driven match against the song catalog.

For 30 songs we just pass the full catalog to the model and ask for matching ids.
No embeddings, no vector store — the corpus fits in a prompt.
"""

from typing import Any, Dict, List

from .. import config
from ..llm import client
from ..llm.types import Plan, SemanticPicks


_SYSTEM = """\
You select songs from a catalog that match a user's intent. You will receive:
- The intent (free-text)
- The full catalog of songs with their attributes

Return the ids of the songs that best match the intent. Order does not matter — a
deterministic re-ranker scores them downstream. Pick between 5 and 12 songs unless
the intent is very narrow. Do NOT pick songs that clearly don't fit just to hit a
count target — fewer good matches is better than padded results.
"""


def _format_catalog(songs: List[Dict[str, Any]]) -> str:
    lines = []
    for s in songs:
        tags = s.get("mood_tags", "")
        lines.append(
            f"id={s['id']:<3} | {s['title']!r} by {s['artist']} | "
            f"genre={s['genre']}, mood={s['mood']}, tags=[{tags}], "
            f"bpm={s['tempo_bpm']:.0f}, energy={s['energy']:.2f}, "
            f"valence={s['valence']:.2f}, acoustic={s['acousticness']:.2f}, "
            f"instrumental={s.get('instrumentalness', 0.0):.2f}"
        )
    return "\n".join(lines)


def run(plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the catalog subset whose ids the LLM picked for ``plan.semantic_query``."""
    if not plan.semantic_query:
        return list(songs)

    user_msg = (
        f"Intent: {plan.semantic_query}\n\n"
        f"Catalog ({len(songs)} songs):\n{_format_catalog(songs)}\n\n"
        f"Return the ids of the songs that match the intent."
    )
    picks = client.call_messages_parsed(
        system=_SYSTEM,
        user=user_msg,
        label="semantic_search",
        schema=SemanticPicks,
        max_tokens=config.SEMANTIC_SEARCH_MAX_TOKENS,
    )
    chosen = set(picks.song_ids)
    return [s for s in songs if s["id"] in chosen]
