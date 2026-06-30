"""Catalog-style labels for history tracks the catalog doesn't contain.

Spotify no longer returns audio features, and returns artist genres for only a few
artists, so an unmatched track is often just "Title by Artist". Embedding that text
against catalog cards gives near-random neighbours. One batched fast-model call labels
those tracks with the catalog's own vocabulary (genre, mood, energy, valence,
acousticness). Labels are cached on disk per track, so a profile rebuild only labels
new tracks. Without credentials, or on any LLM error, it returns what's cached and the
caller falls back to embedding neighbours.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

from pydantic import BaseModel, Field

from .. import config
from ..llm import client

log = logging.getLogger(__name__)

MAX_TRACKS = 120            # top tracks by weight; covers ~all of a typical history
CACHE_PATH = config.CACHE_DIR / "spotify_track_labels.json"

Genre = Literal[
    "pop", "hip-hop", "latin", "edm", "lofi", "indie", "rock", "folk", "reggae", "funk",
    "metal", "k-pop", "country", "r&b", "ambient", "blues", "jazz", "classical",
]
Mood = Literal["happy", "melancholic", "intense", "chill", "relaxed"]


class TrackLabel(BaseModel):
    i: int = Field(description="index of the track in the input list")
    genre: Genre
    mood: Mood
    energy: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)


class TrackLabels(BaseModel):
    labels: List[TrackLabel]


Key = Tuple[str, str]


def _cache_key(key: Key) -> str:
    return f"{key[0]}|{key[1]}"


def _load(path: Path) -> Dict[str, Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def label_tracks(
    tracks: Sequence[Tuple[Key, str, List[str], List[str]]],
    *,
    cache_path: Path = CACHE_PATH,
) -> Dict[Key, Dict]:
    """``tracks``: (key, title, artists, spotify_genres), most important first.
    Returns {key: {genre, mood, energy, valence, acousticness}} for labelled tracks."""
    tracks = list(tracks)[:MAX_TRACKS]
    cache = _load(cache_path)
    todo = [t for t in tracks if _cache_key(t[0]) not in cache]

    if todo and client.is_configured():
        lines = []
        for i, (_, title, artists, genres) in enumerate(todo):
            extra = f" [Spotify genres: {', '.join(genres)}]" if genres else ""
            lines.append(f"{i}. {title} — {', '.join(artists)}{extra}")
        try:
            out = client.call_messages_parsed(
                system=client.load_prompt("track_labels"),
                user="\n".join(lines),
                label="spotify_labels",
                schema=TrackLabels,
                max_tokens=12000,
                effort="low",
                model=config.FAST_MODE_MODEL,
            )
            for lab in out.labels:
                if 0 <= lab.i < len(todo):
                    cache[_cache_key(todo[lab.i][0])] = lab.model_dump(exclude={"i"})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        except Exception as e:  # labelling is an enhancement; never block the profile
            log.warning("track labelling failed, using embedding neighbours: %s", e)

    return {t[0]: cache[_cache_key(t[0])] for t in tracks if _cache_key(t[0]) in cache}
