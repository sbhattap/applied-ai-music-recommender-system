"""
Command line runner for the Music Recommender Simulation.

Defines six user preference profiles — three normal, three adversarial/edge-case —
and runs recommendations under two weight configurations side-by-side:

  ORIGINAL_WEIGHTS  — genre 25 %, mood 20 %, energy 30 %, acousticness 15 %, danceability 10 %
  SHIFTED_WEIGHTS   — genre 12.5 % (halved), energy 60 % (doubled), rest unchanged
                      (weights intentionally do not sum to 1.0; only rank order matters)
"""

from recommender import load_songs, recommend_songs


# ---------------------------------------------------------------------------
# Weight configurations
# To revert to originals: pass ORIGINAL_WEIGHTS (or None) to recommend_songs.
# ---------------------------------------------------------------------------

ORIGINAL_WEIGHTS = {
    "genre":        0.25,   # original
    "mood":         0.20,   # original
    "energy":       0.30,   # original
    "acousticness": 0.15,   # original
    "danceability": 0.10,   # original
}

SHIFTED_WEIGHTS = {
    "genre":        0.125,  # halved  (was 0.25)
    "mood":         0.20,   # unchanged
    "energy":       0.60,   # doubled (was 0.30)
    "acousticness": 0.15,   # unchanged
    "danceability": 0.10,   # unchanged
    # sum = 1.175 — scores are higher overall but ranking is still valid
}


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


def _fmt_recs(recs):
    """Return a list of concise strings for one set of recommendations."""
    if not recs:
        return ["  (no recommendations returned)"]
    lines = []
    for rank, (song, score, _) in enumerate(recs, 1):
        lines.append(
            f"  {rank}. {song['title']} [{song['genre']}, {song['mood']}, "
            f"e={song['energy']:.2f}]  score={score:.4f}"
        )
    return lines


def run_for_profile(songs, profile, k=5):
    name = profile["name"]
    prefs = {key: v for key, v in profile.items() if key != "name"}

    print("=" * 70)
    print(f"Profile : {name}")
    print(
        f"  genre={prefs['genre']}, mood={prefs['mood']}, "
        f"energy={prefs['energy']}, likes_acoustic={prefs.get('likes_acoustic', False)}"
    )
    print("-" * 70)

    orig_recs  = recommend_songs(prefs, songs, k=k, weights=ORIGINAL_WEIGHTS)
    shift_recs = recommend_songs(prefs, songs, k=k, weights=SHIFTED_WEIGHTS)

    orig_lines  = _fmt_recs(orig_recs)
    shift_lines = _fmt_recs(shift_recs)

    col = 35  # left-column width
    header_l = "ORIGINAL weights".center(col)
    header_r = "SHIFTED weights (energy x2, genre /2)".center(col)
    print(f"  {header_l}  |  {header_r}")
    print(f"  {'-'*col}  |  {'-'*col}")

    for i in range(max(len(orig_lines), len(shift_lines))):
        left  = orig_lines[i]  if i < len(orig_lines)  else ""
        right = shift_lines[i] if i < len(shift_lines) else ""
        # flag rank changes
        changed = (left != right)
        marker = " <" if changed else ""
        print(f"  {left:<{col}}  |  {right}{marker}")

    print()


def main():
    songs = load_songs("data/songs.csv")

    for profile in ALL_PROFILES:
        run_for_profile(songs, profile)


if __name__ == "__main__":
    main()
