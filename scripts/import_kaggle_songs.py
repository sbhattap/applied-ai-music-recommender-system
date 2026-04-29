"""Build data/songs.csv from the Kaggle "Spotify Tracks Dataset".

One-off utility, not part of the runtime package. Downloads the dataset via
kagglehub (reads KAGGLE_USERNAME / KAGGLE_KEY from .env), filters to the
genres in the recommender's prompt vocabulary, takes the top-N most popular
per genre, derives mood + mood_tags from valence/energy quadrants, and
writes a CSV in the same schema as the existing demo catalog.

Usage:
    python scripts/import_kaggle_songs.py
    python scripts/import_kaggle_songs.py --songs-per-genre 15

The first run downloads ~80MB of CSV; kagglehub caches it under
~/.cache/kagglehub on subsequent runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "songs.csv"
DEMO_CSV = DATA_DIR / "songs_demo.csv"


# Map Kaggle's `track_genre` values → genres in our prompt vocabulary.
# Kaggle has 125 genres; we keep the subset that maps onto genres the
# planner already knows about. "chill" is the closest Kaggle equivalent
# of our "lofi" bucket.
GENRE_MAP = {
    "pop": "pop",
    "rock": "rock",
    "jazz": "jazz",
    "classical": "classical",
    "edm": "edm",
    "hip-hop": "hip-hop",
    "folk": "folk",
    "funk": "funk",
    "latin": "latin",
    "k-pop": "k-pop",
    "ambient": "ambient",
    "country": "country",
    "metal": "metal",
    "indie": "indie",
    "r-n-b": "r&b",
    "reggae": "reggae",
    "blues": "blues",
    "chill": "lofi",
}


def derive_mood(valence: float, energy: float) -> Tuple[str, List[str]]:
    """Quadrant-based mood inference from Spotify audio features.

    Valence ("musical positiveness") and energy (intensity/activity) form a 2D
    space; we split it into five regions. High energy → happy or intense by
    valence; low energy → relaxed or melancholic by valence; the mid-mid
    "chill" zone is kept narrow so the agent's mood filter actually selects.
    """
    if energy >= 0.65:
        if valence >= 0.5:
            return "happy", ["happy", "uplifting", "energetic"]
        return "intense", ["intense", "driving", "energetic"]
    if energy < 0.4:
        if valence >= 0.5:
            return "relaxed", ["relaxed", "warm", "mellow"]
        return "melancholic", ["melancholic", "introspective", "somber"]
    # Mid energy band — let valence still split out clear-positive / clear-negative
    if valence >= 0.65:
        return "happy", ["happy", "warm"]
    if valence < 0.35:
        return "melancholic", ["melancholic", "moody"]
    return "chill", ["chill", "calm"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import the Kaggle Spotify Tracks Dataset")
    parser.add_argument("--songs-per-genre", type=int, default=10,
                        help="How many songs to pull per genre (default: 10 → ~180 total)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the songs.csv → songs_demo.csv backup step")
    args = parser.parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print(
            "ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set. Add them to .env "
            "(format: get them from https://www.kaggle.com/settings → Create New Token).",
            file=sys.stderr,
        )
        return 2

    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed. Run: pip install kagglehub", file=sys.stderr)
        return 2

    print("Downloading Kaggle 'Spotify Tracks Dataset' (cached after first run)...")
    download_path = Path(kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset"))
    csv_path = download_path / "dataset.csv"
    if not csv_path.exists():
        # The dataset layout has shifted in the past — be forgiving.
        candidates = list(download_path.rglob("*.csv"))
        if not candidates:
            print(f"ERROR: no CSV found under {download_path}", file=sys.stderr)
            return 2
        csv_path = candidates[0]
    print(f"  Source: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Loaded {len(df):,} tracks across {df['track_genre'].nunique()} genres")

    # Keep only genres we map to, dedupe (artist, title) collabs, take top N by popularity per genre.
    df = df[df["track_genre"].isin(GENRE_MAP)].copy()
    df = df.drop_duplicates(subset=["artists", "track_name"])
    df = df.sort_values("popularity", ascending=False)
    sampled = (
        df.groupby("track_genre", group_keys=False)
          .head(args.songs_per_genre)
          .reset_index(drop=True)
    )
    print(f"  Sampled {len(sampled)} songs ({args.songs_per_genre} per genre x "
          f"{sampled['track_genre'].nunique()} genres)")

    # Preserve the hand-crafted demo catalog at songs_demo.csv before overwriting.
    if not args.no_backup and OUTPUT_CSV.exists() and not DEMO_CSV.exists():
        shutil.copy2(OUTPUT_CSV, DEMO_CSV)
        print(f"  Preserved original {OUTPUT_CSV.name} -> {DEMO_CSV.name}")

    rows = []
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        valence = float(row["valence"])
        energy = float(row["energy"])
        mood, mood_tags = derive_mood(valence, energy)
        artists = str(row["artists"]).replace(";", ", ")
        rows.append({
            "id": i,
            "title": row["track_name"],
            "artist": artists,
            "genre": GENRE_MAP[row["track_genre"]],
            "mood": mood,
            "energy": round(energy, 3),
            "tempo_bpm": round(float(row["tempo"]), 1),
            "valence": round(valence, 3),
            "danceability": round(float(row["danceability"]), 3),
            "acousticness": round(float(row["acousticness"]), 3),
            "instrumentalness": round(float(row["instrumentalness"]), 3),
            "popularity": int(row["popularity"]),
            "release_decade": "unknown",  # Kaggle dataset has no release year
            "mood_tags": "|".join(mood_tags),
            "liveness": round(float(row["liveness"]), 3),
            "duration_sec": int(row["duration_ms"] / 1000),
            "explicit": int(bool(row["explicit"])),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Wrote {len(out)} songs to {OUTPUT_CSV}")
    print("\nGenre breakdown:")
    print(out["genre"].value_counts().to_string())
    print("\nMood breakdown:")
    print(out["mood"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
