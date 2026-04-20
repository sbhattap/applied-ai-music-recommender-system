"""
Command line runner for the Music Recommender Simulation.

Defines six user preference profiles — three normal, three adversarial/edge-case —
and runs recommendations under two configurations side-by-side:

  Left column  — ORIGINAL_WEIGHTS (balanced, sums to 1.0)
  Right column — ACTIVE_STRATEGY  (one of the named ranking strategies below)

To switch strategies, change ACTIVE_STRATEGY to one of:
  "genre_first"      — genre dominates (50%), everything else breaks ties
  "mood_first"       — mood/vibe dominates (50%), genre is nearly ignored
  "energy_similarity"— energy dominates (55%), genre/mood labels are nearly ignored
"""

import sys
import io
import argparse
import contextlib
sys.stdout.reconfigure(encoding="utf-8")

from tabulate import tabulate
from recommender import load_songs, recommend_songs, STRATEGY_WEIGHTS


# ---------------------------------------------------------------------------
# Weight configurations
# To revert to originals: pass ORIGINAL_WEIGHTS (or None) to recommend_songs.
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
# The right column in the output will reflect whatever strategy is set here.
# Options: "genre_first" | "mood_first" | "energy_similarity"
# ---------------------------------------------------------------------------
ACTIVE_STRATEGY = "genre_first"


# ---------------------------------------------------------------------------
# Normal user profiles
# ---------------------------------------------------------------------------

# Genre match: pop (25%)  |  Mood match: happy (20%)
# Energy proximity to 0.88 (30%)  |  Non-acoustic preferred (15%)
# Expects high-danceability pop songs like "Sunrise City" and "Gym Hero" to
# dominate the results.
HIGH_ENERGY_POP = {
    "name": "High-Energy Pop",
    "genre": "pop",
    "mood": "happy",
    "energy": 0.88,
    "likes_acoustic": False,
}

# Genre match: lofi (25%)  |  Mood match: chill (20%)
# Low energy target 0.38 (30%)  |  Acoustic preferred (15%)
# "Library Rain" and "Midnight Coding" are near-perfect fits; the diversity cap
# forces the recommender to branch out after 2 lofi songs.
CHILL_LOFI = {
    "name": "Chill Lofi",
    "genre": "lofi",
    "mood": "chill",
    "energy": 0.38,
    "likes_acoustic": True,
}

# Genre match: rock (25%)  |  Mood match: intense (20%)
# Energy target 0.91 (30%)  |  Non-acoustic (15%)
# Only one pure rock song ("Storm Runner") matches genre; the diversity cap
# then forces the recommender to pull intense songs from other genres.
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

# EDGE CASE 1 — Conflicting energy vs. mood
# energy: 0.9 signals a high-intensity listener, but mood: "melancholic"
# describes a sad/introspective feel.  High-energy songs (edm, metal) have
# non-melancholic moods, so the mood weight (20%) and energy weight (30%)
# pull in opposite directions.  The recommender will optimise for energy
# first and likely never surface the one truly melancholic song
# ("Copper Cathedral", energy 0.30) despite its mood match.
CONFLICTING_ENERGY_MOOD = {
    "name": "Conflicting Energy + Mood (adversarial)",
    "genre": "classical",
    "mood": "melancholic",
    "energy": 0.9,
    "likes_acoustic": True,
}

# EDGE CASE 2 — Acoustic EDM contradiction
# EDM tracks in the dataset have acousticness ≈ 0.03, so likes_acoustic: True
# directly fights the genre preference.  The single EDM song ("Bass Drop
# Kingdom") earns the 25% genre-match bonus but is punished hard by the
# acousticness weight (15% × 0.03 ≈ 0.004 vs 15% × 0.97 ≈ 0.145 for a
# folk song).  Watch whether the recommender surfaces the requested genre at
# all, or whether acoustic folk/ambient songs outrank it entirely.
ACOUSTIC_EDM_CONTRADICTION = {
    "name": "Acoustic EDM Contradiction (adversarial)",
    "genre": "edm",
    "mood": "intense",
    "energy": 0.97,
    "likes_acoustic": True,
}

# EDGE CASE 3 — No exact genre + mood match exists in the catalogue
# There is no jazz song tagged "happy" (the only jazz entry is "Coffee Shop
# Stories", mood: relaxed).  With 0% genre-mood overlap, the scoring logic
# falls back entirely on energy/acousticness/danceability proximity.  This
# tests whether the recommender gracefully surfaces the best partial-match
# or whether it silently returns low-quality recommendations.
IMPOSSIBLE_MATCH = {
    "name": "No Exact Match in Catalogue (adversarial)",
    "genre": "jazz",
    "mood": "happy",
    "energy": 0.9,
    "likes_acoustic": False,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_PROFILES = [
    HIGH_ENERGY_POP,
    CHILL_LOFI,
    DEEP_INTENSE_ROCK,
    CONFLICTING_ENERGY_MOOD,
    ACOUSTIC_EDM_CONTRADICTION,
    IMPOSSIBLE_MATCH,
]


_TABLE_HEADERS = ["#", "Title", "Artist", "Genre", "Mood", "Energy", "Score", "Why"]
# Max character widths per column (None = unlimited)
_COL_WIDTHS     = [None, 22, 16, 10, 10, None, None, 48]


def _recs_to_rows(recs):
    """Convert a list of (song, score, explanation) tuples into tabulate rows."""
    rows = []
    for rank, (song, score, explanation) in enumerate(recs, 1):
        # Strip boilerplate and split on ", and " so each reason gets its own line
        body = explanation.removeprefix("Recommended because ").removesuffix(".")
        why  = "\n".join(f"• {r.strip()}" for r in body.split(", and "))
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
    """Print one labelled recommendation table."""
    strategy_label = ACTIVE_STRATEGY.replace("_", " ").upper()
    print(f"  ── {label} ──")
    if not recs:
        print("  (no recommendations returned)\n")
        return
    rows  = _recs_to_rows(recs)
    table = tabulate(rows, headers=_TABLE_HEADERS, tablefmt="rounded_grid",
                     maxcolwidths=_COL_WIDTHS)
    for line in table.splitlines():
        print("  " + line)
    print()


def run_for_profile(songs, profile, k=5):
    name  = profile["name"]
    prefs = {key: v for key, v in profile.items() if key != "name"}
    strategy_label = ACTIVE_STRATEGY.replace("_", " ").upper()

    print("=" * 82)
    print(f"  Profile : {name}")
    print(
        f"  genre={prefs['genre']},  mood={prefs['mood']},  "
        f"energy={prefs['energy']},  likes_acoustic={prefs.get('likes_acoustic', False)}"
    )
    print("=" * 82)

    orig_recs     = recommend_songs(prefs, songs, k=k, weights=ORIGINAL_WEIGHTS)
    strategy_recs = recommend_songs(prefs, songs, k=k, strategy=ACTIVE_STRATEGY)

    _print_table(orig_recs,     "ORIGINAL WEIGHTS")
    _print_table(strategy_recs, f"Strategy: {strategy_label}")


class _TeeStream:
    """Writes to multiple streams simultaneously (stdout + capture buffer)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _save_html(content: str, path: str) -> None:
    """Wrap plain-text terminal output in a styled HTML file."""
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
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save output to a styled HTML file (e.g. results.html)",
    )
    args = parser.parse_args()

    songs = load_songs("data/songs.csv")

    if args.output:
        buffer = io.StringIO()
        tee = _TeeStream(sys.stdout, buffer)
        with contextlib.redirect_stdout(tee):
            for profile in ALL_PROFILES:
                run_for_profile(songs, profile)
        _save_html(buffer.getvalue(), args.output)
        print(f"\n  Saved → {args.output}")
    else:
        for profile in ALL_PROFILES:
            run_for_profile(songs, profile)


if __name__ == "__main__":
    main()
