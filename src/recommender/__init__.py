"""Music recommender package.

Re-exports the deterministic core API at the package root so existing imports
(``from src.recommender import Song, UserProfile, Recommender, ...``) keep
working while the extension layers (llm, tools, trace, output) live alongside.
"""

from .core import (
    DEFAULT_WEIGHTS,
    Recommender,
    STRATEGY_WEIGHTS,
    Song,
    UserProfile,
    explain_song,
    load_songs,
    recommend_songs,
    score_song,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "Recommender",
    "STRATEGY_WEIGHTS",
    "Song",
    "UserProfile",
    "explain_song",
    "load_songs",
    "recommend_songs",
    "score_song",
]
