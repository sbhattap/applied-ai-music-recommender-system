from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Extended features (default to neutral so existing test fixtures still work)
    popularity: int = 50          # 0–100 chart popularity
    release_decade: str = ""      # e.g. "2020s"
    mood_tags: str = ""           # pipe-separated tags e.g. "happy|uplifting|summer"
    liveness: float = 0.1         # 0=studio, 1=live recording
    duration_sec: int = 0
    explicit: int = 0             # 0=clean, 1=explicit

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Extended preferences (optional — defaults to neutral values)
    target_valence: float = 0.5   # 0=prefers darker/sadder, 1=prefers upbeat/positive
    prefer_popular: bool = True   # True=mainstream, False=underground/niche

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initialize the recommender with a list of available songs."""
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        """
        Compute a weighted compatibility score between a user profile and a song.

        Scoring weights:
          - Genre match:        20%
          - Mood match:         15%  (full credit for primary mood; 0.5x for tag match)
          - Energy proximity:   25%
          - Acousticness fit:   12%
          - Danceability:        8%
          - Valence proximity:  12%
          - Popularity fit:      8%

        Returns a float in [0.0, 1.0] where higher means a better match.
        """
        genre_match = 1.0 if song.genre == user.favorite_genre else 0.0

        # Soft mood match: full credit for primary mood, half credit if mood
        # appears anywhere in the pipe-separated mood_tags.
        if song.mood == user.favorite_mood:
            mood_match = 1.0
        elif song.mood_tags and user.favorite_mood in song.mood_tags.split("|"):
            mood_match = 0.5
        else:
            mood_match = 0.0

        energy_proximity = 1.0 - abs(song.energy - user.target_energy)
        acousticness_fit = song.acousticness if user.likes_acoustic else (1.0 - song.acousticness)
        danceability_score = song.danceability
        valence_proximity = 1.0 - abs(song.valence - user.target_valence)
        pop_norm = song.popularity / 100
        popularity_fit = pop_norm if user.prefer_popular else (1.0 - pop_norm)

        return (
            genre_match        * 0.20 +
            mood_match         * 0.15 +
            energy_proximity   * 0.25 +
            acousticness_fit   * 0.12 +
            danceability_score * 0.08 +
            valence_proximity  * 0.12 +
            popularity_fit     * 0.08
        )

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Return the top-k songs for a user, ranked by compatibility score.

        Applies a diversity cap of at most 2 songs per genre so results
        span multiple genres. If fewer than k qualifying songs exist,
        returns all that match the cap.
        """
        scored = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)

        # Diversity cap: max 2 songs per genre
        results = []
        genre_counts: Dict[str, int] = {}
        for song in scored:
            if genre_counts.get(song.genre, 0) < 2:
                results.append(song)
                genre_counts[song.genre] = genre_counts.get(song.genre, 0) + 1
            if len(results) == k:
                break

        return results

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Generate a human-readable explanation for why a song was recommended.

        Checks genre, mood, energy proximity, and acousticness against the
        user profile and returns a sentence listing the matching factors.
        Falls back to a generic message if no specific factor matches.
        """
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"it's {song.genre} (your favorite genre)")
        if song.mood == user.favorite_mood:
            reasons.append(f"it matches your {song.mood} mood preference")
        if abs(song.energy - user.target_energy) <= 0.15:
            reasons.append(f"its energy ({song.energy}) is close to your target ({user.target_energy})")
        if user.likes_acoustic and song.acousticness >= 0.6:
            reasons.append("it has a strong acoustic feel")
        if not user.likes_acoustic and song.acousticness <= 0.3:
            reasons.append("it has a non-acoustic, produced sound")
        if abs(song.valence - user.target_valence) <= 0.15:
            reasons.append(f"its emotional tone (valence {song.valence:.2f}) matches your preference")
        if user.prefer_popular and song.popularity >= 70:
            reasons.append(f"it's a popular track (popularity {song.popularity})")
        elif not user.prefer_popular and song.popularity <= 40:
            reasons.append("it's an underground or niche pick")

        if not reasons:
            reasons.append("it closely matches your overall listening profile")

        return "Recommended because " + ", and ".join(reasons) + "."

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv
    songs = []
    with open(csv_path, newline="") as f:
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
                "popularity": int(row["popularity"]),
                "release_decade": row["release_decade"],
                "mood_tags": row["mood_tags"],
                "liveness": float(row["liveness"]),
                "duration_sec": int(row["duration_sec"]),
                "explicit": int(row["explicit"]),
            })
    return songs

DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre":        0.20,
    "mood":         0.15,
    "energy":       0.25,
    "acousticness": 0.12,
    "danceability": 0.08,
    "valence":      0.12,
    "popularity":   0.08,
}

# Named ranking strategies — each gives one signal a dominant share so the
# output reflects a clear intent (browse by genre, match a vibe, find an energy
# twin) rather than a balanced compromise.
STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Lock into the user's preferred genre first; everything else breaks ties.
    "genre_first": {
        "genre":        0.50,
        "mood":         0.20,
        "energy":       0.15,
        "acousticness": 0.07,
        "danceability": 0.04,
        "valence":      0.03,
        "popularity":   0.01,
    },
    # Match the user's vibe/emotional state above genre — good for "I'm feeling
    # X today, anything goes."  Soft tag matching means partial mood credit too.
    "mood_first": {
        "genre":        0.08,
        "mood":         0.50,
        "energy":       0.20,
        "acousticness": 0.08,
        "danceability": 0.05,
        "valence":      0.07,
        "popularity":   0.02,
    },
    # Pure audio-feature similarity — find the closest "feel" regardless of
    # genre or mood labels.  Good for workout, focus, or driving playlists.
    "energy_similarity": {
        "genre":        0.03,
        "mood":         0.05,
        "energy":       0.55,
        "acousticness": 0.15,
        "danceability": 0.12,
        "valence":      0.08,
        "popularity":   0.02,
    },
}

# def _score_song(user_prefs: Dict, song: Dict) -> float:          # ORIGINAL signature
def _score_song(user_prefs: Dict, song: Dict,
                weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute a weighted compatibility score between a user preference dict and a song dict.

    Expected keys in user_prefs: 'genre', 'mood', 'energy' (float), 'likes_acoustic' (bool).
    Default weighting: genre 25%, mood 20%, energy 30%, acousticness 15%, danceability 10%.

    Pass a custom ``weights`` dict with the same keys to override the defaults.
    Note: weights need not sum to 1.0 — only relative magnitude matters for ranking.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS

    genre_match = 1.0 if song["genre"] == user_prefs.get("genre") else 0.0

    # Soft mood match: full credit for primary mood, half for a tag match
    mood = user_prefs.get("mood", "")
    if song["mood"] == mood:
        mood_match = 1.0
    elif mood in song.get("mood_tags", "").split("|"):
        mood_match = 0.5
    else:
        mood_match = 0.0

    energy_proximity = 1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5))
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    acousticness_fit = song["acousticness"] if likes_acoustic else (1.0 - song["acousticness"])
    danceability_score = song["danceability"]

    target_valence = user_prefs.get("target_valence", 0.5)
    valence_proximity = 1.0 - abs(song.get("valence", 0.5) - target_valence)

    prefer_popular = user_prefs.get("prefer_popular", True)
    pop_norm = song.get("popularity", 50) / 100
    popularity_fit = pop_norm if prefer_popular else (1.0 - pop_norm)

    # Use .get() so caller-supplied weight dicts that omit new keys still work
    return (
        genre_match        * w.get("genre", 0) +
        mood_match         * w.get("mood", 0) +
        energy_proximity   * w.get("energy", 0) +
        acousticness_fit   * w.get("acousticness", 0) +
        danceability_score * w.get("danceability", 0) +
        valence_proximity  * w.get("valence", 0) +
        popularity_fit     * w.get("popularity", 0)
    )

def _explain_song(user_prefs: Dict, song: Dict) -> str:
    """
    Generate a human-readable explanation for why a song matches a preference dict.

    Mirrors the logic of Recommender.explain_recommendation but operates on
    plain dicts instead of dataclass instances. Falls back to a generic message
    if no specific factor matches.
    """
    reasons = []
    if song["genre"] == user_prefs.get("genre"):
        reasons.append(f"it's {song['genre']} (your favorite genre)")
    if song["mood"] == user_prefs.get("mood"):
        reasons.append(f"it matches your {song['mood']} mood preference")
    if abs(song["energy"] - user_prefs.get("energy", 0.5)) <= 0.15:
        reasons.append(f"its energy ({song['energy']}) is close to your target")
    if user_prefs.get("likes_acoustic") and song["acousticness"] >= 0.6:
        reasons.append("it has a strong acoustic feel")
    if not user_prefs.get("likes_acoustic") and song["acousticness"] <= 0.3:
        reasons.append("it has a non-acoustic, produced sound")
    target_valence = user_prefs.get("target_valence", 0.5)
    if abs(song.get("valence", 0.5) - target_valence) <= 0.15:
        reasons.append(f"its emotional tone (valence {song.get('valence', 0.5):.2f}) matches your preference")
    if user_prefs.get("prefer_popular", True) and song.get("popularity", 50) >= 70:
        reasons.append(f"it's a popular track (popularity {song.get('popularity', 50)})")
    elif not user_prefs.get("prefer_popular", True) and song.get("popularity", 50) <= 40:
        reasons.append("it's an underground or niche pick")
    if not reasons:
        reasons.append("it closely matches your overall listening profile")
    return "Recommended because " + ", and ".join(reasons) + "."

# def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> ...:   # ORIGINAL signature
def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5,
                    weights: Optional[Dict[str, float]] = None,
                    strategy: Optional[str] = None) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Pass a custom ``weights`` dict to experiment with different scoring priorities,
    or pass a ``strategy`` name ("genre_first", "mood_first", "energy_similarity")
    to use a preset. If both are given, ``weights`` takes precedence.
    Omit both to use DEFAULT_WEIGHTS.
    """
    if weights is None and strategy is not None:
        weights = STRATEGY_WEIGHTS.get(strategy, DEFAULT_WEIGHTS)

    scored = sorted(songs, key=lambda s: _score_song(user_prefs, s, weights), reverse=True)

    results = []
    genre_counts: Dict[str, int] = {}
    for song in scored:
        genre = song["genre"]
        if genre_counts.get(genre, 0) < 2:
            score = _score_song(user_prefs, song, weights)
            explanation = _explain_song(user_prefs, song)
            results.append((song, score, explanation))
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if len(results) == k:
            break

    return results
