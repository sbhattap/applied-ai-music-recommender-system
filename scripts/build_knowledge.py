"""Fetch external evidence for the RAG knowledge store.

Writes JSONL files under data/rag/ that the retriever turns into evidence documents:
  - artists.jsonl      Wikipedia summary per catalog artist
  - genres.jsonl       Wikipedia summary per catalog genre
  - lastfm_tags.jsonl  Last.fm crowd tags per song (only if LASTFM_API_KEY is set)

Usage:
  uv run python scripts/build_knowledge.py            # fetch what's missing
  uv run python scripts/build_knowledge.py --refresh  # refetch everything

Wikipedia text is CC BY-SA 4.0; every record keeps its source URL for attribution.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender import config  # noqa: E402
from src.recommender.core.io import load_songs, split_artists  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


USER_AGENT = "applied-ai-music-recommender/0.2 (educational project)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
LASTFM_API = "https://ws.audioscrobbler.com/2.0/"

# A search hit only counts as the artist's page if its summary reads like a music bio.
MUSIC_WORDS = re.compile(
    r"\b(singer|songwriter|rapper|band|musician|composer|pianist|cellist|violinist|"
    r"guitarist|DJ|producer|record producer|duo|group|orchestra|ensemble|vocalist|"
    r"conductor|artist|boy band|girl group|music)\b",
    re.IGNORECASE,
)

# Stage names whose Wikipedia page title is something else entirely.
ARTIST_PAGES = {
    "2Pac": "Tupac Shakur",
    "Drake": "Drake (musician)",
    "IVE": "Ive (group)",
    "ROSALÍA": "Rosalía",
    "Khalid": "Khalid (singer)",
    "Bicep": "Bicep (duo)",
    "Eagles": "Eagles (band)",
    "Keane": "Keane (band)",
    "Future": "Future (rapper)",
    "Ghost": "Ghost (Swedish band)",
    "Crush": "Crush (singer)",
    "Paul Lewis": "Paul Lewis (pianist)",
}

# Names whose only Wikipedia hits are known false positives.
ARTIST_SKIP = {"FAST BOY"}

GENRE_PAGES = {
    "ambient": "Ambient music",
    "blues": "Blues",
    "classical": "Classical music",
    "country": "Country music",
    "edm": "Electronic dance music",
    "folk": "Folk music",
    "funk": "Funk",
    "hip-hop": "Hip-hop",
    "indie": "Indie rock",
    "jazz": "Jazz",
    "k-pop": "K-pop",
    "latin": "Latin music",
    "lofi": "Lo-fi music",
    "metal": "Heavy metal music",
    "pop": "Pop music",
    "r&b": "Contemporary R&B",
    "reggae": "Reggae",
    "rock": "Rock music",
}


def _get_json(url: str, params: Optional[Dict[str, Any]] = None, *, retries: int = 5) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"  rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def wiki_extracts(titles: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch lookup (<=20 titles): lead-section text, URL and disambiguation flag.

    Returns ``{requested_title: page}``, following normalization and redirects.
    """
    data = _get_json(WIKI_API, {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "titles": "|".join(titles), "prop": "extracts|pageprops|info",
        "exintro": 1, "explaintext": 1, "exlimit": "max", "inprop": "url",
        "ppprop": "disambiguation",
    })
    query = data.get("query", {})
    alias = {t: t for t in titles}
    for step in ("normalized", "redirects"):
        for m in query.get(step, []):
            for req, cur in alias.items():
                if cur == m["from"]:
                    alias[req] = m["to"]
    pages = {p["title"]: p for p in query.get("pages", []) if not p.get("missing")}
    out = {}
    for req, title in alias.items():
        page = pages.get(title)
        if not page or "disambiguation" in page.get("pageprops", {}) or not page.get("extract"):
            continue
        out[req] = {"title": page["title"], "summary": page["extract"].strip(), "url": page.get("fullurl", "")}
    return out


def _summary(title: str) -> Optional[Dict[str, str]]:
    try:
        data = _get_json(WIKI_SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe=""))
    except urllib.error.HTTPError:
        return None
    if data.get("type") != "standard" or not data.get("extract"):
        return None
    return {
        "title": data["title"],
        "summary": data["extract"],
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


def wiki_artist(name: str) -> Optional[Dict[str, str]]:
    """Best-effort: search, then take the first hit whose summary reads like a music bio."""
    hits = _get_json(WIKI_API, {
        "action": "query", "list": "search", "format": "json", "srlimit": 3,
        "srsearch": f'"{name}" (musician OR band OR singer OR composer OR rapper)',
    }).get("query", {}).get("search", [])
    titles = [h["title"] for h in hits]
    if not titles:
        return None
    pages = wiki_extracts(titles)
    for title in titles:
        if _is_music_page(name, pages.get(title)):
            return pages[title]
    return None


def _name_matches(name: str, page: Dict[str, str]) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    target = norm(name)
    # Title must carry the name; matching on summary text let "Drake" resolve to a
    # song page that merely mentions him.
    return target in norm(page["title"])


def lastfm_tags(api_key: str, artist: str, title: str) -> List[str]:
    data = _get_json(LASTFM_API, {
        "method": "track.gettoptags", "artist": artist, "track": title,
        "api_key": api_key, "format": "json", "autocorrect": 1,
    })
    tags = data.get("toptags", {}).get("tag", [])
    return [t["name"].lower() for t in tags if int(t.get("count", 0)) >= 5][:8]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_music_page(name: str, page: Optional[Dict[str, str]]) -> bool:
    return bool(page) and bool(MUSIC_WORDS.search(page["summary"][:400])) and _name_matches(name, page)


def build_artists(songs: List[Dict[str, Any]], refresh: bool) -> None:
    """Exact-title batch lookup first (cheap), search only for names that miss."""
    path = config.RAG_DATA_DIR / "artists.jsonl"
    existing = {} if refresh else {r["name"]: r for r in _read_jsonl(path) if r.get("found")}
    names = sorted({a for s in songs for a in split_artists(s["artist"])})
    todo = [n for n in names if n not in existing and n not in ARTIST_SKIP]

    found: Dict[str, Dict[str, str]] = {}
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        pages = wiki_extracts([ARTIST_PAGES.get(n, n) for n in batch])
        for name in batch:
            page = pages.get(ARTIST_PAGES.get(name, name))
            if name in ARTIST_PAGES and page or _is_music_page(name, page):
                found[name] = page
        time.sleep(0.5)
    print(f"  exact-title hits: {len(found)}/{len(todo)}")

    for name in [n for n in todo if n not in found]:
        try:
            page = wiki_artist(name)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ! {name}: {e}")
            continue
        if page:
            found[name] = page
        print(f"  search {name}: {page['title'] if page else '—'}")
        time.sleep(0.5)

    out = []
    for name in names:
        if name in existing:
            out.append(existing[name])
        else:
            page = found.get(name)
            out.append({"name": name, "source": "wikipedia", "found": page is not None, **(page or {})})
    _write_jsonl(path, out)
    print(f"artists: {sum(r['found'] for r in out)}/{len(out)} matched -> {path}")


def build_genres(songs: List[Dict[str, Any]], refresh: bool) -> None:
    path = config.RAG_DATA_DIR / "genres.jsonl"
    existing = {} if refresh else {r["genre"]: r for r in _read_jsonl(path)}
    out = []
    for genre in sorted({s["genre"] for s in songs}):
        if genre in existing:
            out.append(existing[genre])
            continue
        page = _summary(GENRE_PAGES.get(genre, genre))
        if page:
            out.append({"genre": genre, "source": "wikipedia", **page})
        print(f"  {genre}: {page['title'] if page else '—'}")
    _write_jsonl(path, out)
    print(f"genres: {len(out)} -> {path}")


def build_lastfm(songs: List[Dict[str, Any]], refresh: bool) -> None:
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print("lastfm: LASTFM_API_KEY not set, skipping")
        return
    path = config.RAG_DATA_DIR / "lastfm_tags.jsonl"
    existing = {} if refresh else {r["song_id"]: r for r in _read_jsonl(path)}
    out = []
    for song in songs:
        if song["id"] in existing:
            out.append(existing[song["id"]])
            continue
        artist = split_artists(song["artist"])[0]
        try:
            tags = lastfm_tags(api_key, artist, song["title"])
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"  ! {song['title']}: {e}")
            continue
        url = f"https://www.last.fm/music/{urllib.parse.quote(artist)}/_/{urllib.parse.quote(song['title'])}"
        out.append({"song_id": song["id"], "source": "lastfm", "tags": tags, "url": url})
        time.sleep(0.25)  # Last.fm asks for <5 req/s
    _write_jsonl(path, out)
    print(f"lastfm: {sum(bool(r['tags']) for r in out)}/{len(out)} tagged -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="refetch existing records")
    parser.add_argument("--songs", default=str(config.SONGS_CSV))
    args = parser.parse_args()

    songs = load_songs(args.songs)
    build_genres(songs, args.refresh)
    build_artists(songs, args.refresh)
    build_lastfm(songs, args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
