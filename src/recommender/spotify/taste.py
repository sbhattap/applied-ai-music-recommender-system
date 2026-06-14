"""Listening history -> taste profile the recommender can use.

1. Weight every track the listener has touched (recent plays, top tracks by time range,
   saved tracks) into one play-weight per track.
2. Match tracks to the catalog by normalized title + artist. Matched songs carry real
   audio features; unmatched tracks are embedded from their text (title, artists,
   artist genres) with the same embedder as the catalog.
3. Taste vector = weight-averaged embedding of everything above.
4. Derived profile: genre/mood/energy/valence/acoustic from matched songs plus the
   nearest catalog neighbours of unmatched ones, so the 7 profile signals work too.
5. Evidence: ``hist:{id}`` docs for catalog songs the listener actually plays, and
   ``similar:{id}`` docs linking catalog songs to the history track they're closest to.
   Both are citable by the explainer ("because you play X").
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.io import _normalize_text, split_artists
from ..rag.documents import Document
from ..rag.embedder import get_vector_cache
from ..rag.retriever import get_retriever


# How much each signal says about taste.
WEIGHT_RECENT_PLAY = 1.0
WEIGHT_TOP = {"short_term": 3.0, "medium_term": 2.0, "long_term": 1.5}
WEIGHT_SAVED = 1.0
NEIGHBOURS = 3
SIMILAR_MIN = 0.80          # cosine floor for a "similar to X you play" evidence doc

_SUFFIX = re.compile(r"\s*[\(\[\-–].*?(feat\.?|ft\.?|with|remaster(ed)?|version|edit|mix|live).*$", re.I)


def track_key(title: str, artist: str) -> Tuple[str, str]:
    base = _SUFFIX.sub("", title) or title
    return _normalize_text(base), _normalize_text(artist)


@dataclass
class TrackStats:
    name: str
    artists: List[str]
    url: str = ""
    weight: float = 0.0
    plays: int = 0
    top_ranges: List[str] = field(default_factory=list)
    saved: bool = False
    genres: List[str] = field(default_factory=list)


def aggregate(history: Dict[str, Any]) -> Dict[Tuple[str, str], TrackStats]:
    artist_genres = {a["name"]: a.get("genres", []) for a in history.get("top_artists", [])}
    tracks: Dict[Tuple[str, str], TrackStats] = {}

    def touch(t: Dict[str, Any]) -> TrackStats:
        names = [a["name"] for a in t["artists"]] or [""]
        key = track_key(t["name"], names[0])
        if key not in tracks:
            genres = sorted({g for n in names for g in artist_genres.get(n, [])})
            tracks[key] = TrackStats(name=t["name"], artists=names, url=t.get("url", ""), genres=genres)
        return tracks[key]

    for t in history.get("recently_played", []):
        s = touch(t)
        s.plays += 1
        s.weight += WEIGHT_RECENT_PLAY
    for time_range, items in history.get("top_tracks", {}).items():
        for pos, t in enumerate(items):
            s = touch(t)
            s.top_ranges.append(time_range)
            s.weight += WEIGHT_TOP.get(time_range, 1.0) * (1 - pos / (2 * max(len(items), 1)))
    for t in history.get("saved_tracks", []):
        s = touch(t)
        s.saved = True
        s.weight += WEIGHT_SAVED
    return tracks


def _catalog_index(songs: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    index = {}
    for s in songs:
        for name in split_artists(s["artist"])[:1]:
            index[track_key(s["title"], name)] = s["id"]
    return index


def _history_text(t: TrackStats) -> str:
    genres = f" Genres: {', '.join(t.genres)}." if t.genres else ""
    return f"{t.name} by {', '.join(t.artists)}.{genres}"


_RANGE_LABEL = {"short_term": "last 4 weeks", "medium_term": "last 6 months", "long_term": "all time"}


def _hist_text(t: TrackStats) -> str:
    parts = []
    if t.plays:
        parts.append(f"played {t.plays} time{'s' if t.plays != 1 else ''} in your recent history")
    if t.top_ranges:
        parts.append("in your top tracks (" + ", ".join(_RANGE_LABEL[r] for r in t.top_ranges) + ")")
    if t.saved:
        parts.append("saved in your library")
    return f"{t.name} by {', '.join(t.artists)}: " + "; ".join(parts) + "."


def build_profile(
    history: Dict[str, Any],
    songs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Profile dict for ``agent.run``: profile prefs + taste_vector, recently_played,
    history_docs, and a ``history_summary`` for the UI."""
    tracks = aggregate(history)
    if not tracks:
        raise ValueError("No listening history returned by Spotify.")
    retriever = get_retriever(songs)
    by_id = {s["id"]: s for s in songs}
    catalog = _catalog_index(songs)
    card_matrix = np.stack([retriever.song_vector(s["id"]) for s in songs])

    stats = list(tracks.values())
    keys = list(tracks.keys())
    weights = np.array([t.weight for t in stats], dtype=np.float32)
    matched = {i: catalog[k] for i, k in enumerate(keys) if k in catalog}

    vecs = get_vector_cache().get_many([_history_text(t) for t in stats])
    for i, sid in matched.items():
        vecs[i] = retriever.song_vector(sid)        # matched: use the catalog card
    taste = (weights[:, None] * vecs).sum(axis=0)
    taste = taste / (np.linalg.norm(taste) or 1.0)

    # Feature profile from matched songs + nearest catalog neighbours of the rest.
    feature_votes: List[Tuple[Dict[str, Any], float]] = []
    for i, t in enumerate(stats):
        if i in matched:
            feature_votes.append((by_id[matched[i]], t.weight))
        else:
            sims = card_matrix @ vecs[i]
            for j in np.argsort(-sims)[:NEIGHBOURS]:
                feature_votes.append((songs[j], t.weight * float(sims[j]) / NEIGHBOURS))
    total = sum(w for _, w in feature_votes) or 1.0
    genre_w: Counter = Counter()
    mood_w: Counter = Counter()
    for song, w in feature_votes:
        genre_w[song["genre"]] += w
        mood_w[song["mood"]] += w
    avg = lambda key: round(sum(s[key] * w for s, w in feature_votes) / total, 3)  # noqa: E731

    recently_played = sorted({
        catalog[track_key(t["name"], (t["artists"] or [{"name": ""}])[0]["name"])]
        for t in history.get("recently_played", [])
        if track_key(t["name"], (t["artists"] or [{"name": ""}])[0]["name"]) in catalog
    })

    docs: List[Document] = []
    for i, sid in matched.items():
        docs.append(Document(
            doc_id=f"hist:{sid}", kind="hist", source="spotify", title=stats[i].name,
            text=_hist_text(stats[i]), song_ids=[sid], url=stats[i].url,
        ))
    # "Similar to X you play": each catalog song's closest history track.
    sims = card_matrix @ vecs.T                       # songs x history
    matched_ids = set(matched.values())
    for j, song in enumerate(songs):
        if song["id"] in matched_ids:
            continue
        i = int(np.argmax(sims[j]))
        if sims[j, i] >= SIMILAR_MIN:
            t = stats[i]
            docs.append(Document(
                doc_id=f"similar:{song['id']}", kind="similar", source="spotify",
                title=f"Similar to {t.name}",
                text=f"Close to {t.name} by {', '.join(t.artists)}, which you listen to "
                     f"(similarity {sims[j, i]:.2f}).",
                song_ids=[song["id"]], url=t.url,
            ))

    top_tracks = sorted(stats, key=lambda t: -t.weight)[:10]
    top_genres = Counter(g for t in stats for g in t.genres).most_common(8)
    user = history.get("user", {})
    return {
        "name": f"{user.get('display_name') or 'Your'} (Spotify)",
        "genre": genre_w.most_common(1)[0][0],
        "mood": mood_w.most_common(1)[0][0],
        "energy": avg("energy"),
        "target_valence": avg("valence"),
        "likes_acoustic": avg("acousticness") >= 0.5,
        "taste_vector": [round(float(x), 6) for x in taste],
        "recently_played": recently_played,
        "history_docs": [
            {"doc_id": d.doc_id, "kind": d.kind, "source": d.source, "title": d.title,
             "text": d.text, "song_ids": d.song_ids, "url": d.url}
            for d in docs
        ],
        "history_summary": {
            "user": user,
            "tracks": len(stats),
            "matched_to_catalog": len(matched),
            "top_tracks": [f"{t.name} — {', '.join(t.artists)}" for t in top_tracks],
            "top_artist_genres": [g for g, _ in top_genres],
            "genre_mix": {g: round(w / total, 3) for g, w in genre_w.most_common(6)},
        },
    }


def summary_lines(profile: Dict[str, Any]) -> List[str]:
    s = profile.get("history_summary") or {}
    out = [f"{s.get('tracks', 0)} tracks analysed, {s.get('matched_to_catalog', 0)} in the catalog"]
    if s.get("top_artist_genres"):
        out.append("Your genres: " + ", ".join(s["top_artist_genres"][:5]))
    return out
