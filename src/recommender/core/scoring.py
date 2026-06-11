"""7-signal compatibility scoring + per-song explanation.

The scorer is the deterministic ranking authority. The agent does not score —
it picks which candidates to feed in, then this module ranks them.
"""

from typing import Dict, Optional


DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre":        0.20,
    "mood":         0.15,
    "energy":       0.25,
    "acousticness": 0.12,
    "danceability": 0.08,
    "valence":      0.12,
    "popularity":   0.08,
}

STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "genre_first": {
        "genre":        0.50,
        "mood":         0.20,
        "energy":       0.15,
        "acousticness": 0.07,
        "danceability": 0.04,
        "valence":      0.03,
        "popularity":   0.01,
    },
    "mood_first": {
        "genre":        0.08,
        "mood":         0.50,
        "energy":       0.20,
        "acousticness": 0.08,
        "danceability": 0.05,
        "valence":      0.07,
        "popularity":   0.02,
    },
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


def _pref(user_prefs: Dict, key: str, default):
    value = user_prefs.get(key)
    return default if value is None else value


def signal_values(user_prefs: Dict, song: Dict) -> Dict[str, float]:
    """Per-signal match values in [0.0, 1.0], before weighting.

    Unstated preferences (missing or None) fall back to neutral defaults; the RAG
    reranker additionally drops those signals from its weights.
    """
    genre_match = 1.0 if song["genre"] == user_prefs.get("genre") else 0.0

    mood = _pref(user_prefs, "mood", "")
    if song["mood"] == mood:
        mood_match = 1.0
    elif mood in song.get("mood_tags", "").split("|"):
        mood_match = 0.5
    else:
        mood_match = 0.0

    energy_proximity = 1.0 - abs(song["energy"] - _pref(user_prefs, "energy", 0.5))
    likes_acoustic = _pref(user_prefs, "likes_acoustic", False)
    acousticness_fit = song["acousticness"] if likes_acoustic else (1.0 - song["acousticness"])

    target_valence = _pref(user_prefs, "target_valence", 0.5)
    valence_proximity = 1.0 - abs(song.get("valence", 0.5) - target_valence)

    prefer_popular = _pref(user_prefs, "prefer_popular", True)
    pop_norm = song.get("popularity", 50) / 100
    popularity_fit = pop_norm if prefer_popular else (1.0 - pop_norm)

    return {
        "genre": genre_match,
        "mood": mood_match,
        "energy": energy_proximity,
        "acousticness": acousticness_fit,
        "danceability": song["danceability"],
        "valence": valence_proximity,
        "popularity": popularity_fit,
    }


def score_song(user_prefs: Dict, song: Dict,
               weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted compatibility score in [0.0, 1.0].

    Pass a custom ``weights`` dict to override DEFAULT_WEIGHTS. Weights need
    not sum to 1.0 — only relative magnitude matters for ranking.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    values = signal_values(user_prefs, song)
    return sum(values[name] * w.get(name, 0) for name in values)


def explain_song(user_prefs: Dict, song: Dict) -> str:
    """Human-readable list of factors that matched between profile and song."""
    reasons = []
    if song["genre"] == user_prefs.get("genre"):
        reasons.append(f"it's {song['genre']} (your favorite genre)")
    if song["mood"] == user_prefs.get("mood"):
        reasons.append(f"it matches your {song['mood']} mood preference")
    if abs(song["energy"] - _pref(user_prefs, "energy", 0.5)) <= 0.15:
        reasons.append(f"its energy ({song['energy']}) is close to your target")
    if user_prefs.get("likes_acoustic") and song["acousticness"] >= 0.6:
        reasons.append("it has a strong acoustic feel")
    if not user_prefs.get("likes_acoustic") and song["acousticness"] <= 0.3:
        reasons.append("it has a non-acoustic, produced sound")
    target_valence = _pref(user_prefs, "target_valence", 0.5)
    if abs(song.get("valence", 0.5) - target_valence) <= 0.15:
        reasons.append(f"its emotional tone (valence {song.get('valence', 0.5):.2f}) matches your preference")
    prefer_popular = _pref(user_prefs, "prefer_popular", True)
    if prefer_popular and song.get("popularity", 50) >= 70:
        reasons.append(f"it's a popular track (popularity {song.get('popularity', 50)})")
    elif not prefer_popular and song.get("popularity", 50) <= 40:
        reasons.append("it's an underground or niche pick")
    if not reasons:
        reasons.append("it closely matches your overall listening profile")
    return "Recommended because " + ", and ".join(reasons) + "."


# Underscored aliases retained — older callers and the module-private convention
_score_song = score_song
_explain_song = explain_song
