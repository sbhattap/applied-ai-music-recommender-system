"""
Rule-based baseline demo (Modules 1-3 deliverable).

Runs the deterministic rule-based recommender against six fixed user profiles —
three normal, three adversarial/edge-case — under two configurations side by side:

  Left column  — ORIGINAL_WEIGHTS (balanced, sums to 1.0)
  Right column — ACTIVE_STRATEGY  (one of the named ranking strategies below)

This is the baseline the eval section compares the agent-orchestrated path against.
It does not use the LLM at all.

To switch strategies, change ACTIVE_STRATEGY to one of:
  "genre_first"      — genre dominates (50%), everything else breaks ties
  "mood_first"       — mood/vibe dominates (50%), genre is nearly ignored
  "energy_similarity"— energy dominates (55%), genre/mood labels are nearly ignored

Usage (from project root):
  python scripts/baseline_demo.py
  python scripts/baseline_demo.py -o results.html
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

# Make `src` importable when this script is run as a file (python scripts/baseline_demo.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from tabulate import tabulate

from src.recommender import STRATEGY_WEIGHTS, load_songs, recommend_songs  # noqa: E402,F401


SONGS_CSV = PROJECT_ROOT / "data" / "songs.csv"


# ---------------------------------------------------------------------------
# Weight configurations
# ---------------------------------------------------------------------------

ORIGINAL_WEIGHTS = {
    "genre":        0.20,   # genre exact match
    "mood":         0.15,   # primary mood / tag match
    "energy":       0.25,   # energy proximity
    "acousticness": 0.12,   # acoustic vs produced preference
    "danceability": 0.08,   # raw danceability score
    "valence":      0.12,   # emotional positiveness proximity
    "popularity":   0.08,   # mainstream vs niche preference
    # sum = 1.00
}

# ---------------------------------------------------------------------------
# Active ranking strategy — change this one line to switch modes.
# Options: "genre_first" | "mood_first" | "energy_similarity"
# ---------------------------------------------------------------------------
ACTIVE_STRATEGY = "genre_first"


# ---------------------------------------------------------------------------
# Normal user profiles
# ---------------------------------------------------------------------------

HIGH_ENERGY_POP = {
    "name": "High-Energy Pop",
    "genre": "pop",
    "mood": "happy",
    "energy": 0.88,
    "likes_acoustic": False,
}

CHILL_LOFI = {
    "name": "Chill Lofi",
    "genre": "lofi",
    "mood": "chill",
    "energy": 0.38,
    "likes_acoustic": True,
}

DEEP_INTENSE_ROCK = {
    "name": "Deep Intense Rock",
    "genre": "rock",
    "mood": "intense",
    "energy": 0.91,
    "likes_acoustic": False,
}


# ---------------------------------------------------------------------------
# Adversarial / edge-case profiles
# ---------------------------------------------------------------------------

# EDGE CASE 1 — Conflicting energy vs. mood. High energy + melancholic mood:
# the energy weight (25-30%) and mood weight (15-20%) pull in opposite directions.
CONFLICTING_ENERGY_MOOD = {
    "name": "Conflicting Energy + Mood (adversarial)",
    "genre": "classical",
    "mood": "melancholic",
    "energy": 0.9,
    "likes_acoustic": True,
}

# EDGE CASE 2 — Acoustic EDM contradiction. EDM tracks have low acousticness,
# so likes_acoustic=True directly fights the genre preference.
ACOUSTIC_EDM_CONTRADICTION = {
    "name": "Acoustic EDM Contradiction (adversarial)",
    "genre": "edm",
    "mood": "intense",
    "energy": 0.97,
    "likes_acoustic": True,
}

# EDGE CASE 3 — No exact genre+mood match exists in the catalogue.
# Tests graceful degradation to partial-match scoring.
IMPOSSIBLE_MATCH = {
    "name": "No Exact Match in Catalogue (adversarial)",
    "genre": "jazz",
    "mood": "happy",
    "energy": 0.9,
    "likes_acoustic": False,
}


ALL_PROFILES = [
    HIGH_ENERGY_POP,
    CHILL_LOFI,
    DEEP_INTENSE_ROCK,
    CONFLICTING_ENERGY_MOOD,
    ACOUSTIC_EDM_CONTRADICTION,
    IMPOSSIBLE_MATCH,
]


_TABLE_HEADERS = ["#", "Title", "Artist", "Genre", "Mood", "Energy", "Score", "Why"]
_COL_WIDTHS = [None, 22, 16, 10, 10, None, None, 48]


def _recs_to_rows(recs):
    rows = []
    for rank, (song, score, explanation) in enumerate(recs, 1):
        body = explanation.removeprefix("Recommended because ").removesuffix(".")
        why = "\n".join(f"• {r.strip()}" for r in body.split(", and "))
        rows.append([
            rank,
            song["title"],
            song["artist"],
            song["genre"],
            song["mood"],
            f"{song['energy']:.2f}",
            f"{score:.4f}",
            why,
        ])
    return rows


def _print_table(recs, label):
    print(f"  ── {label} ──")
    if not recs:
        print("  (no recommendations returned)\n")
        return
    rows = _recs_to_rows(recs)
    table = tabulate(rows, headers=_TABLE_HEADERS, tablefmt="rounded_grid",
                     maxcolwidths=_COL_WIDTHS)
    for line in table.splitlines():
        print("  " + line)
    print()


def run_for_profile(songs, profile, k=5):
    name = profile["name"]
    prefs = {key: v for key, v in profile.items() if key != "name"}
    strategy_label = ACTIVE_STRATEGY.replace("_", " ").upper()

    print("=" * 82)
    print(f"  Profile : {name}")
    print(
        f"  genre={prefs['genre']},  mood={prefs['mood']},  "
        f"energy={prefs['energy']},  likes_acoustic={prefs.get('likes_acoustic', False)}"
    )
    print("=" * 82)

    orig_recs = recommend_songs(prefs, songs, k=k, weights=ORIGINAL_WEIGHTS)
    strategy_recs = recommend_songs(prefs, songs, k=k, strategy=ACTIVE_STRATEGY)

    _print_table(orig_recs, "ORIGINAL WEIGHTS")
    _print_table(strategy_recs, f"Strategy: {strategy_label}")


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _save_html(content: str, path: str) -> None:
    import html as _html
    escaped = _html.escape(content)
    strategy_label = ACTIVE_STRATEGY.replace("_", " ").title()
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Music Recommender — {strategy_label}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1e1e2e;
      color: #cdd6f4;
      font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
      font-size: 13.5px;
      line-height: 1.45;
      padding: 2rem 2.5rem;
    }}
    h1 {{
      color: #89b4fa;
      font-size: 1.1rem;
      margin-bottom: 1.2rem;
      letter-spacing: .04em;
    }}
    pre {{
      white-space: pre;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <h1>🎵 Music Recommender — Strategy: {strategy_label}</h1>
  <pre>{escaped}</pre>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser(description="Rule-based baseline demo")
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="Save output to a styled HTML file (e.g. results.html)",
    )
    parser.add_argument(
        "--songs", default=str(SONGS_CSV),
        help="Path to songs.csv (default: data/songs.csv)",
    )
    args = parser.parse_args()

    songs = load_songs(args.songs)

    if args.output:
        buffer = io.StringIO()
        tee = _TeeStream(sys.stdout, buffer)
        with contextlib.redirect_stdout(tee):
            for profile in ALL_PROFILES:
                run_for_profile(songs, profile)
        _save_html(buffer.getvalue(), args.output)
        print(f"\n  Saved -> {args.output}")
    else:
        for profile in ALL_PROFILES:
            run_for_profile(songs, profile)


if __name__ == "__main__":
    main()
