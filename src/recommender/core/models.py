from dataclasses import dataclass


@dataclass
class Song:
    """Represents a song and its attributes."""
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
    instrumentalness: float = 0.0
    # Extended features (default to neutral so existing test fixtures still work)
    popularity: int = 50          # 0–100 chart popularity
    release_decade: str = ""      # e.g. "2020s"
    mood_tags: str = ""           # pipe-separated tags e.g. "happy|uplifting|summer"
    liveness: float = 0.1         # 0=studio, 1=live recording
    duration_sec: int = 0
    explicit: int = 0             # 0=clean, 1=explicit


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Extended preferences (optional — defaults to neutral values)
    target_valence: float = 0.5   # 0=prefers darker/sadder, 1=prefers upbeat/positive
    prefer_popular: bool = True   # True=mainstream, False=underground/niche
