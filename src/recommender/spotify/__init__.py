"""Spotify login (OAuth PKCE), listening-history fetch, and taste profiling."""

from .api import SpotifyAPIError, SpotifyClient
from .auth import SpotifyAuthError, TokenStore, Tokens, is_configured
from .taste import build_profile

__all__ = [
    "SpotifyAPIError",
    "SpotifyAuthError",
    "SpotifyClient",
    "TokenStore",
    "Tokens",
    "build_profile",
    "is_configured",
]
