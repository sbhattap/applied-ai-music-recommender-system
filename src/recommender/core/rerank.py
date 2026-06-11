"""Additive reranker for the RAG pipeline.

score = sum(weight_i * value_i) over every active signal, so the per-signal bars the
UI draws add up exactly to the score. Retrieval signals (how well the song matches
the *query* and the listener's *history*) sit next to the original 7 profile signals.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .scoring import DEFAULT_WEIGHTS, explain_song, signal_values


QUERY_WEIGHT = 0.5
TASTE_WEIGHT = 0.15
NOVELTY_WEIGHT = 0.05

SIGNAL_LABELS = {
    "query_match": "Query match",
    "taste_match": "Listening-history match",
    "novelty": "New to you",
    "genre": "Genre",
    "mood": "Mood",
    "energy": "Energy",
    "acousticness": "Acousticness",
    "danceability": "Danceability",
    "valence": "Valence",
    "popularity": "Popularity",
}


# Profile signals only count when the listener actually stated that preference
# (None = no preference). Danceability has no preference key and always counts.
PREFERENCE_KEYS = {
    "genre": "genre",
    "mood": "mood",
    "energy": "energy",
    "acousticness": "likes_acoustic",
    "danceability": None,
    "valence": "target_valence",
    "popularity": "prefer_popular",
}


def stated_signals(prefs: Dict[str, Any]) -> List[str]:
    return [name for name, key in PREFERENCE_KEYS.items() if key is None or prefs.get(key) is not None]


@dataclass
class Signal:
    name: str
    label: str
    value: float
    weight: float
    contribution: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def active_weights(
    *,
    has_query: bool,
    has_taste: bool,
    has_history: bool,
    profile_signals: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Retrieval signals get fixed shares; the stated profile signals split the remainder
    in DEFAULT_WEIGHTS proportions. Always sums to 1."""
    profile = list(profile_signals) if profile_signals is not None else list(DEFAULT_WEIGHTS)
    weights: Dict[str, float] = {}
    if has_query:
        weights["query_match"] = QUERY_WEIGHT
    if has_taste:
        weights["taste_match"] = TASTE_WEIGHT
    if has_history:
        weights["novelty"] = NOVELTY_WEIGHT
    if not profile:
        total = sum(weights.values()) or 1.0
        return {n: w / total for n, w in weights.items()}
    remainder = 1.0 - sum(weights.values())
    total = sum(DEFAULT_WEIGHTS[n] for n in profile)
    for name in profile:
        weights[name] = remainder * DEFAULT_WEIGHTS[name] / total
    return weights


def _detail(name: str, value: float, prefs: Dict[str, Any], song: Dict[str, Any],
            hit: Optional[Dict[str, Any]]) -> str:
    if name == "query_match" and hit:
        ranks = [f"semantic #{hit['dense_rank']}"]
        if hit.get("bm25_rank"):
            ranks.append(f"keyword #{hit['bm25_rank']}")
        terms = f"; matched: {', '.join(hit['matched_terms'])}" if hit.get("matched_terms") else ""
        return f"retrieval match {value:.2f} ({', '.join(ranks)}{terms})"
    if name == "taste_match":
        return f"similarity {value:.2f} to songs you listen to"
    if name == "novelty":
        return "not in your recent plays" if value else "you've played this recently"
    if name == "genre":
        return (f"{song['genre']} is your favorite genre" if value
                else f"{song['genre']} (you prefer {prefs.get('genre')})")
    if name == "mood":
        if value == 1.0:
            return f"{song['mood']} mood matches yours"
        if value == 0.5:
            return f"tagged {prefs.get('mood')} (primary mood {song['mood']})"
        return f"{song['mood']} mood (you prefer {prefs.get('mood')})"
    if name == "energy":
        return f"energy {song['energy']:.2f} vs your target {prefs['energy']:.2f}"
    if name == "acousticness":
        want = "acoustic" if prefs.get("likes_acoustic") else "produced"
        return f"acousticness {song['acousticness']:.2f} (you prefer {want})"
    if name == "danceability":
        return f"danceability {song['danceability']:.2f}"
    if name == "valence":
        return f"valence {song.get('valence', 0.5):.2f} vs target {prefs['target_valence']:.2f}"
    if name == "popularity":
        return f"popularity {song.get('popularity', 50)}"
    return f"{value:.2f}"


def breakdown(
    prefs: Dict[str, Any],
    song: Dict[str, Any],
    weights: Dict[str, float],
    *,
    hit: Optional[Dict[str, Any]] = None,
    recently_played: Optional[Set[int]] = None,
) -> List[Signal]:
    values: Dict[str, float] = {}
    if "query_match" in weights:
        values["query_match"] = float(hit["query_match"]) if hit else 0.0
    if "taste_match" in weights:
        # cosine sims of related music cluster in ~[0.5, 1]; rescale so the signal spreads
        raw = hit.get("taste_sim") if hit else None
        values["taste_match"] = max(0.0, min(1.0, (raw - 0.5) * 2)) if raw is not None else 0.0
    if "novelty" in weights:
        values["novelty"] = 0.0 if recently_played and song["id"] in recently_played else 1.0
    values.update({n: v for n, v in signal_values(prefs, song).items() if n in weights})

    return [
        Signal(
            name=name,
            label=SIGNAL_LABELS[name],
            value=round(values[name], 4),
            weight=round(weights[name], 4),
            contribution=round(values[name] * weights[name], 4),
            detail=_detail(name, values[name], prefs, song, hit),
        )
        for name in weights
    ]


def rerank(
    prefs: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    hits: Dict[int, Dict[str, Any]],
    *,
    k: int = 5,
    has_taste: bool = False,
    recently_played: Optional[Set[int]] = None,
    max_per_genre: Optional[int] = 2,
    exclude: Set[str] = frozenset(),
) -> List[Tuple[Dict[str, Any], float, List[Signal], str]]:
    """Top-k with the same diversity cap as ``recommend_songs``.

    ``exclude`` drops signals and renormalizes the rest — the UI's "what if" view.
    Returns ``(song, score, signals, breakdown_text)``; signals are sorted by contribution.
    """
    weights = active_weights(
        has_query=bool(hits), has_taste=has_taste, has_history=recently_played is not None,
        profile_signals=stated_signals(prefs),
    )
    if exclude:
        weights = {n: w for n, w in weights.items() if n not in exclude}
        total = sum(weights.values()) or 1.0
        weights = {n: w / total for n, w in weights.items()}
    scored = []
    for song in candidates:
        signals = breakdown(prefs, song, weights, hit=hits.get(song["id"]),
                            recently_played=recently_played)
        score = round(sum(s.contribution for s in signals), 4)
        scored.append((song, score, sorted(signals, key=lambda s: -s.contribution)))
    scored.sort(key=lambda t: -t[1])

    out = []
    per_genre: Dict[str, int] = {}
    for song, score, signals in scored:
        if max_per_genre is not None and per_genre.get(song["genre"], 0) >= max_per_genre:
            continue
        per_genre[song["genre"]] = per_genre.get(song["genre"], 0) + 1
        out.append((song, score, signals, explain_song(prefs, song)))
        if len(out) == k:
            break
    return out
