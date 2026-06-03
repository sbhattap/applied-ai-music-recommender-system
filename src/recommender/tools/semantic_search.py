"""Semantic match against the song catalog.

Default backend is the local hybrid retriever (dense + BM25, ~5 ms). The original
LLM backend — the full catalog in a prompt, one model call per search — is kept behind
``RECOMMENDER_SEMANTIC_BACKEND=llm`` so evals can compare the two.
"""

from typing import Any, Dict, List

from .. import config
from ..llm import client
from ..llm.types import Plan, SemanticPicks
from ..rag import get_retriever


MIN_RESULTS = 8
MAX_RESULTS = 15


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
    """Return the songs that best match ``plan.semantic_query``."""
    if not plan.semantic_query:
        return list(songs)
    if config.SEMANTIC_BACKEND == "llm":
        return run_llm(plan, songs)
    return run_rag(plan, songs)


def run_rag(plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Top-N by hybrid retrieval. N follows the planner's expected count, within bounds."""
    top_n = max(MIN_RESULTS, min(MAX_RESULTS, plan.expected_count_max))
    hits = get_retriever(songs).search(plan.semantic_query, songs, top_n=top_n)
    by_id = {s["id"]: s for s in songs}
    return [by_id[h.song_id] for h in hits]


def run_llm(plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the catalog subset whose ids the LLM picked for ``plan.semantic_query``."""
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
