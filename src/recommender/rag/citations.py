"""Citation validation and grounded fallback explanations.

The explainer must attach evidence ids to every claim. ``validate`` keeps a claim only
if it cites at least one id and every id it cites was actually offered for that song.
Dropped claims are kept (with a reason) for the trace and eval: a high drop rate means
the model is drifting from the evidence.
"""

import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def evidence_ids(signals: Sequence[Dict[str, Any]], docs: Sequence[Dict[str, Any]]) -> Set[str]:
    return {f"sig:{s['name']}" for s in signals} | {d["doc_id"] for d in docs}


def validate(
    claims: Iterable[Dict[str, Any]],
    allowed: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept, dropped = [], []
    for claim in claims:
        text = (claim.get("text") or "").strip()
        ids = [i.strip() for i in claim.get("evidence_ids") or [] if i and i.strip()]
        if not text:
            continue
        if not ids:
            dropped.append({"text": text, "evidence_ids": [], "reason": "no citation"})
            continue
        unknown = [i for i in ids if i not in allowed]
        if unknown:
            dropped.append({"text": text, "evidence_ids": ids,
                            "reason": f"cites evidence not retrieved for this song: {unknown}"})
            continue
        kept.append({"text": text, "evidence_ids": ids})
    return kept, dropped


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _first_sentence(text: str, limit: int = 320) -> str:
    sentence = _SENTENCE_END.split(text.strip(), maxsplit=1)[0].rstrip("…")
    return sentence if len(sentence) <= limit else sentence[:limit].rstrip() + "…"


def grounded_claims(
    signals: Sequence[Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
    *,
    low_confidence: bool = False,
) -> List[Dict[str, Any]]:
    """Deterministic, cited explanation built only from evidence. Used when the LLM is
    unavailable or every LLM claim for a song was dropped."""
    claims: List[Dict[str, Any]] = []
    top = [s for s in signals if s["contribution"] > 0][:2]
    for s in top:
        detail = s["detail"][:1].upper() + s["detail"][1:]
        claims.append({"text": f"{detail}.", "evidence_ids": [f"sig:{s['name']}"]})

    context = next((d for d in docs if d["kind"] in ("hist", "tags", "artist")), None)
    if context:
        claims.append({"text": _first_sentence(context["text"]), "evidence_ids": [context["doc_id"]]})

    if low_confidence and claims:
        claims[0] = {**claims[0], "text": "Closest match available — " + claims[0]["text"]}
    return claims


def render(claims: Sequence[Dict[str, Any]]) -> str:
    return " ".join(c["text"] for c in claims)
