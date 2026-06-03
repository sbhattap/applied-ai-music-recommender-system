"""CSV catalog loader. Single source of truth for song attributes."""

import csv
import unicodedata
from typing import Dict, List


INSTRUMENTALNESS_OVERRIDES = {
    ("experience", "ludovico einaudi, daniel hope, i virtuosi italiani"): 0.93,
    ("glue", "bicep"): 0.82,
    ("river flows in you", "yiruma"): 0.98,
    ("gymnopedie no. 1", "erik satie, philippe entremont"): 0.99,
    (
        "the carnival of the animals, r. 125: xiii. the swan (arr. for cello and piano)",
        "camille saint-saens, yo-yo ma, kathryn stott",
    ): 0.99,
    ("nowesind", "jacob david"): 0.90,
    ("cello suite no. 1 in g major, bwv 1007: i. prelude", "johann sebastian bach, yo-yo ma"): 0.98,
    (
        'sonata no. 14 "moonlight" in c-sharp minor", op. 27 no. 2: i. adagio sostenuto',
        "ludwig van beethoven, paul lewis",
    ): 0.99,
    ("vaseegara", "bombay jayashri"): 0.02,
    ("merry-go-round of life - from 'howl's moving castle'", "joe hisaishi"): 0.97,
    ("clair de lune, l. 32", "claude debussy, martin jones"): 0.99,
    ("nocturne no. 2 in e-flat major, op. 9 no. 2", "frederic chopin, daniel barenboim"): 0.99,
}


# Artist names that contain a comma; everything else in the ``artist`` column is a
# comma-separated list of credited artists.
_COMMA_NAMES = (
    "Earth, Wind & Fire",
    "Crosby, Stills, Nash & Young",
    "Tyler, The Creator",
    "Grover Washington, Jr.",
)


def split_artists(artist: str) -> List[str]:
    """Split a credited-artists string into individual names."""
    protected = artist
    for i, name in enumerate(_COMMA_NAMES):
        protected = protected.replace(name, f"\x00{i}\x00")
    names = []
    for part in protected.split(","):
        part = part.strip()
        for i, name in enumerate(_COMMA_NAMES):
            part = part.replace(f"\x00{i}\x00", name)
        if part:
            names.append(part)
    return names


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split())


def _infer_instrumentalness(row: Dict[str, str]) -> float:
    explicit_value = row.get("instrumentalness")
    if explicit_value not in (None, ""):
        return float(explicit_value)

    key = (_normalize_text(row["title"]), _normalize_text(row["artist"]))
    return INSTRUMENTALNESS_OVERRIDES.get(key, 0.0)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV with the schema documented in data/songs.csv."""
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                "instrumentalness": _infer_instrumentalness(row),
                "popularity": int(row["popularity"]),
                "release_decade": row["release_decade"],
                "mood_tags": row["mood_tags"],
                "liveness": float(row["liveness"]),
                "duration_sec": int(row["duration_sec"]),
                "explicit": int(row["explicit"]),
            })
    return songs
