"""Batched, citation-grounded explainer.

One LLM call covers every recommended song. Each song comes with a numbered evidence
list — its additive score signals (``sig:*``) and the documents the retriever found for
it (``song:*``, ``artist:*``, ``genre:*``, ``tags:*``, ``hist:*``). The model returns
short claims that each cite evidence ids; ``rag.citations.validate`` then drops any
claim citing nothing or citing evidence that wasn't offered for that song.
"""

from typing import Any, Dict, List, Optional

from .. import config
from . import client
from .types import BatchExplanations, CheckVerdict


EVIDENCE_TEXT_CHARS = 280


def _format_item(item: Dict[str, Any]) -> str:
    song = item["song"]
    lines = [
        f"- song_id={song['id']}: {song['title']!r} by {song['artist']} "
        f"(genre={song['genre']}, mood={song['mood']}, "
        f"bpm={song['tempo_bpm']:.0f}, energy={song['energy']:.2f})",
        f"  score={item['score']:.3f}",
        "  evidence:",
    ]
    for s in item["signals"]:
        lines.append(
            f"    [sig:{s['name']}] {s['label']}: {s['detail']} "
            f"(weight {s['weight']:.2f}, contributes {s['contribution']:+.3f})"
        )
    for d in item["evidence"]:
        text = " ".join(d["text"].split())[:EVIDENCE_TEXT_CHARS]
        lines.append(f"    [{d['doc_id']}] {d['source']}: {text}")
    return "\n".join(lines)


def explain_batch(
    *,
    query: str,
    profile_summary: str,
    items: List[Dict[str, Any]],
    verdict: CheckVerdict,
    model: Optional[str] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Return ``{song_id: [{"text", "evidence_ids"}, ...]}`` (unvalidated) in one call.

    ``items`` are ``{"song", "score", "signals", "evidence"}`` dicts. Songs the model
    skipped are missing from the result; the caller falls back for those.
    """
    system = client.load_prompt("explainer")
    user = (
        f"User query: {query}\n"
        f"Profile: {profile_summary}\n\n"
        f"Check verdict: confidence={verdict.confidence:.2f}, "
        f"flags={[f.value for f in verdict.flags]}, note={verdict.note!r}\n\n"
        f"Recommended songs:\n" + "\n".join(_format_item(it) for it in items) + "\n\n"
        f"Write cited claims for every song."
    )
    out = client.call_messages_parsed(
        system=system,
        user=user,
        label="explainer",
        schema=BatchExplanations,
        max_tokens=config.EXPLAINER_MAX_TOKENS,
        effort=config.EXPLAINER_EFFORT,
        model=model or config.FAST_MODEL,
    )
    wanted = {it["song"]["id"] for it in items}
    return {
        e.song_id: [c.model_dump() for c in e.claims]
        for e in out.explanations
        if e.song_id in wanted
    }


def templated_fallback(
    *,
    song: Dict[str, Any],
    breakdown: str,
    verdict: CheckVerdict,
) -> str:
    """Plain-text fallback from the deterministic breakdown (legacy CLI path)."""
    if verdict.confidence < 0.6:
        prefix = "Closest match available — "
    else:
        prefix = ""
    return prefix + breakdown
