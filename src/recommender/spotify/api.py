"""Minimal Spotify Web API client for listening history.

Only endpoints still open to new Development Mode apps (Feb 2026 changes): /me,
/me/player/recently-played, /me/top/{tracks,artists}, /me/tracks. Batch lookups
(/tracks?ids=, /artists?ids=) were removed, so artist genres come from top artists
instead of per-artist calls. Responses are normalized to plain dicts.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .auth import Tokens


API = "https://api.spotify.com/v1"
TIME_RANGES = ("short_term", "medium_term", "long_term")


class SpotifyAPIError(RuntimeError):
    pass


def _track(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name", ""),
        "artists": [{"id": a.get("id"), "name": a.get("name", "")} for a in item.get("artists", [])],
        "url": item.get("external_urls", {}).get("spotify", ""),
    }


class SpotifyClient:
    def __init__(self, tokens: Tokens, on_refresh: Optional[Callable[[Tokens], Tokens]] = None) -> None:
        self.tokens = tokens
        self._on_refresh = on_refresh
        self.calls = 0

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, *, retries: int = 3) -> Dict[str, Any]:
        url = f"{API}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        for attempt in range(retries):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.tokens.access_token}"})
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 401 and self._on_refresh and attempt == 0:
                    self.tokens = self._on_refresh(self.tokens)
                    continue
                if e.code == 429 and attempt < retries - 1:
                    if "QUOTA_EXCEEDED" in body:
                        raise SpotifyAPIError("Spotify development-mode quota exceeded.") from e
                    time.sleep(min(float(e.headers.get("Retry-After") or 2), 10))
                    continue
                raise SpotifyAPIError(f"GET {path} failed ({e.code}): {body[:300]}") from e
        raise SpotifyAPIError(f"GET {path} failed after {retries} attempts")

    def me(self) -> Dict[str, Any]:
        data = self._get("/me")
        return {"id": data.get("id"), "display_name": data.get("display_name") or data.get("id")}

    def recently_played(self, limit: int = 50) -> List[Dict[str, Any]]:
        data = self._get("/me/player/recently-played", {"limit": limit})
        return [{**_track(i["track"]), "played_at": i.get("played_at")} for i in data.get("items", [])]

    def top_tracks(self, time_range: str = "medium_term", limit: int = 50) -> List[Dict[str, Any]]:
        data = self._get("/me/top/tracks", {"time_range": time_range, "limit": limit})
        return [_track(i) for i in data.get("items", [])]

    def top_artists(self, time_range: str = "medium_term", limit: int = 50) -> List[Dict[str, Any]]:
        data = self._get("/me/top/artists", {"time_range": time_range, "limit": limit})
        return [
            {"id": a.get("id"), "name": a.get("name", ""), "genres": a.get("genres") or []}
            for a in data.get("items", [])
        ]

    def saved_tracks(self, limit: int = 50) -> List[Dict[str, Any]]:
        data = self._get("/me/tracks", {"limit": limit})
        return [{**_track(i["track"]), "added_at": i.get("added_at")} for i in data.get("items", [])]

    def fetch_history(self) -> Dict[str, Any]:
        """Everything the taste model uses: ~8 requests total."""
        return {
            "user": self.me(),
            "recently_played": self.recently_played(),
            "top_tracks": {r: self.top_tracks(r) for r in TIME_RANGES},
            "top_artists": self.top_artists("medium_term"),
            "saved_tracks": self.saved_tracks(),
            "fetched_at": time.time(),
        }
