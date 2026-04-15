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
          - Genre match:        25%
          - Mood match:         20%
          - Energy proximity:   30%
          - Acousticness fit:   15%
          - Danceability:       10%

        Returns a float in [0.0, 1.0] where higher means a better match.
        """
        genre_match = 1.0 if song.genre == user.favorite_genre else 0.0
        mood_match = 1.0 if song.mood == user.favorite_mood else 0.0
        energy_proximity = 1.0 - abs(song.energy - user.target_energy)
        acousticness_fit = song.acousticness if user.likes_acoustic else (1.0 - song.acousticness)
        danceability_score = song.danceability

        return (
            genre_match       * 0.25 +
            mood_match        * 0.20 +
            energy_proximity  * 0.30 +
            acousticness_fit  * 0.15 +
            danceability_score * 0.10
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
            })
    return songs

DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre":        0.25,
    "mood":         0.20,
    "energy":       0.30,
    "acousticness": 0.15,
    "danceability": 0.10,
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
    mood_match = 1.0 if song["mood"] == user_prefs.get("mood") else 0.0
    energy_proximity = 1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5))
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    acousticness_fit = song["acousticness"] if likes_acoustic else (1.0 - song["acousticness"])
    danceability_score = song["danceability"]

    return (
        genre_match        * w["genre"] +        # was * 0.25
        mood_match         * w["mood"] +          # was * 0.20
        energy_proximity   * w["energy"] +        # was * 0.30
        acousticness_fit   * w["acousticness"] +  # was * 0.15
        danceability_score * w["danceability"]    # was * 0.10
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
    if not reasons:
        reasons.append("it closely matches your overall listening profile")
    return "Recommended because " + ", and ".join(reasons) + "."

# def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> ...:   # ORIGINAL signature
def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5,
                    weights: Optional[Dict[str, float]] = None) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Pass a custom ``weights`` dict to experiment with different scoring priorities.
    Omit it (or pass None) to use DEFAULT_WEIGHTS.
    """
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
