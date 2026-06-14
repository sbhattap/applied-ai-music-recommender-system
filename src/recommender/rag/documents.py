"""Evidence documents: everything the retriever can surface and the explainer can cite.

Each document has a stable ``doc_id`` (what citations point at), a ``source`` for
provenance, and the ``song_ids`` it supports. Kinds:

  song:{id}      catalog card — attributes rendered as words, so both BM25 and the
                 embedder can match "slow", "no vocals", "upbeat"
  tags:{id}      Last.fm crowd tags for one track
  artist:{slug}  Wikipedia summary, linked to every song the artist is credited on
  genre:{slug}   Wikipedia summary, linked to every song in that genre
  hist:{id}      the listener's own play history for a track (added per user)
  similar:{id}   which of the listener's tracks a catalog song is closest to
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.io import split_artists


# Bios are long; the lead is what's relevant and what fits the embedder.
MAX_CONTEXT_CHARS = 600


@dataclass
class Document:
    doc_id: str
    kind: str                # song | tags | artist | genre | hist | similar
    source: str              # catalog | lastfm | wikipedia | spotify
    title: str
    text: str
    song_ids: List[int] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id, "kind": self.kind, "source": self.source,
            "title": self.title, "text": self.text, "url": self.url,
        }


def slug(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9&]+", "-", ascii_text.lower()).strip("-")


def describe(song: Dict[str, Any]) -> List[str]:
    """Numeric features -> plain-language descriptors."""
    out: List[str] = []
    energy = song["energy"]
    out.append("low energy, calm" if energy < 0.35 else
               "moderate energy" if energy < 0.65 else "high energy, intense")
    bpm = song["tempo_bpm"]
    pace = "slow" if bpm < 85 else "mid" if bpm < 115 else "fast"
    out.append(f"{pace} tempo ({bpm:.0f} BPM)")
    valence = song.get("valence", 0.5)
    if valence < 0.35:
        out.append("dark, sad tone")
    elif valence > 0.65:
        out.append("upbeat, positive tone")
    acoustic = song["acousticness"]
    if acoustic >= 0.6:
        out.append("acoustic")
    elif acoustic <= 0.2:
        out.append("electronic, produced sound")
    instrumental = song.get("instrumentalness", 0.0)
    out.append("instrumental, no vocals" if instrumental >= 0.7 else "has singing")
    if song.get("danceability", 0.0) >= 0.7:
        out.append("danceable")
    if song.get("liveness", 0.0) >= 0.6:
        out.append("live recording")
    popularity = song.get("popularity", 50)
    if popularity >= 80:
        out.append("popular hit")
    elif popularity <= 40:
        out.append("niche, lesser-known")
    if song.get("explicit"):
        out.append("explicit lyrics")
    return out


def song_card(song: Dict[str, Any]) -> Document:
    tags = ", ".join(t for t in song.get("mood_tags", "").split("|") if t)
    text = (
        f"{song['title']} by {song['artist']}. {song['genre']} song, {song['mood']} mood. "
        + (f"Tags: {tags}. " if tags else "")
        + "; ".join(describe(song)) + "."
    )
    return Document(
        doc_id=f"song:{song['id']}", kind="song", source="catalog",
        title=song["title"], text=text, song_ids=[song["id"]],
    )


def keyword_text(song: Dict[str, Any]) -> str:
    """BM25 text for a song: everything on the card except the title. Vibe queries
    ("something to get pumped up") otherwise match title words ("Pumped Up Kicks")."""
    tags = " ".join(t for t in song.get("mood_tags", "").split("|") if t)
    return f"{song['artist']}. {song['genre']} {song['mood']} {tags}. " + "; ".join(describe(song))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _lead(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[: end + 1] if end > limit // 2 else cut.rstrip() + "…"


def build_documents(
    songs: Iterable[Dict[str, Any]],
    knowledge_dir: Optional[Path] = None,
) -> List[Document]:
    """Song cards for ``songs`` plus whatever external evidence exists for them."""
    songs = list(songs)
    docs = [song_card(s) for s in songs]
    if knowledge_dir is None:
        return docs

    by_id = {s["id"]: s for s in songs}
    for row in _read_jsonl(knowledge_dir / "lastfm_tags.jsonl"):
        song = by_id.get(row["song_id"])
        if song and row.get("tags"):
            docs.append(Document(
                doc_id=f"tags:{song['id']}", kind="tags", source="lastfm",
                title=f"Last.fm tags for {song['title']}",
                text=f"Listeners tag {song['title']} as: {', '.join(row['tags'])}.",
                song_ids=[song["id"]], url=row.get("url", ""),
            ))

    credits: Dict[str, List[int]] = {}
    for s in songs:
        for name in split_artists(s["artist"]):
            credits.setdefault(name, []).append(s["id"])
    for row in _read_jsonl(knowledge_dir / "artists.jsonl"):
        ids = credits.get(row["name"])
        if ids and row.get("found"):
            docs.append(Document(
                doc_id=f"artist:{slug(row['name'])}", kind="artist", source="wikipedia",
                title=row["name"], text=_lead(row["summary"]), song_ids=ids,
                url=row.get("url", ""),
            ))

    by_genre: Dict[str, List[int]] = {}
    for s in songs:
        by_genre.setdefault(s["genre"], []).append(s["id"])
    for row in _read_jsonl(knowledge_dir / "genres.jsonl"):
        ids = by_genre.get(row["genre"])
        if ids:
            docs.append(Document(
                doc_id=f"genre:{slug(row['genre'])}", kind="genre", source="wikipedia",
                title=row["title"], text=_lead(row["summary"]), song_ids=ids,
                url=row.get("url", ""),
            ))
    return docs
