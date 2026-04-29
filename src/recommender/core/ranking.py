"""Ranking layer: turns scores into a top-k list with a per-genre diversity cap.

Exposes both the functional API (``recommend_songs``) used by the CLI and the
OOP API (``Recommender``) used by tests.
"""

from typing import Dict, List, Optional, Tuple

from .models import Song, UserProfile
from .scoring import (
    DEFAULT_WEIGHTS,
    STRATEGY_WEIGHTS,
    score_song,
    explain_song,
)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5,
                    weights: Optional[Dict[str, float]] = None,
                    strategy: Optional[str] = None) -> List[Tuple[Dict, float, str]]:
    """Top-k recommendations with a diversity cap of at most 2 songs per genre.

    Pass ``weights`` to override scoring priorities, or ``strategy`` to pick a
    preset ("genre_first" | "mood_first" | "energy_similarity"). If both are
    given, ``weights`` wins. Omit both to use DEFAULT_WEIGHTS.
    """
    if weights is None and strategy is not None:
        weights = STRATEGY_WEIGHTS.get(strategy, DEFAULT_WEIGHTS)

    scored = sorted(songs, key=lambda s: score_song(user_prefs, s, weights), reverse=True)

    results: List[Tuple[Dict, float, str]] = []
    genre_counts: Dict[str, int] = {}
    for song in scored:
        genre = song["genre"]
        if genre_counts.get(genre, 0) < 2:
            score = score_song(user_prefs, song, weights)
            explanation = explain_song(user_prefs, song)
            results.append((song, score, explanation))
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if len(results) == k:
            break
    return results


class Recommender:
    """OOP facade over the functional API. Used by the existing test suite."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "target_valence": user.target_valence,
            "prefer_popular": user.prefer_popular,
        }
        song_dict = song.__dict__
        return score_song(prefs, song_dict)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        results: List[Song] = []
        genre_counts: Dict[str, int] = {}
        for song in scored:
            if genre_counts.get(song.genre, 0) < 2:
                results.append(song)
                genre_counts[song.genre] = genre_counts.get(song.genre, 0) + 1
            if len(results) == k:
                break
        return results

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "target_valence": user.target_valence,
            "prefer_popular": user.prefer_popular,
        }
        return explain_song(prefs, song.__dict__)
