"""Spotify OAuth: Authorization Code with PKCE (no client secret needed).

Spotify rules for new apps (2025-26): redirect URIs must use a loopback IP
(``http://127.0.0.1:PORT``), never ``localhost``; Development Mode apps need a Premium
owner and an allowlist of at most 5 users.
"""

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config


AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = ("user-read-recently-played", "user-top-read", "user-library-read")
TOKENS_PATH = config.PROJECT_ROOT / ".spotify_tokens.json"   # gitignored


class SpotifyAuthError(RuntimeError):
    pass


def client_id() -> Optional[str]:
    return os.environ.get("SPOTIFY_CLIENT_ID") or None


def redirect_uri() -> str:
    return os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")


def is_configured() -> bool:
    return client_id() is not None


# ── PKCE ─────────────────────────────────────────────────────────────────────

def new_verifier() -> str:
    return secrets.token_urlsafe(64)[:96]


def challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorize_url(state: str, verifier: str, *, redirect: Optional[str] = None) -> str:
    cid = client_id()
    if not cid:
        raise SpotifyAuthError("SPOTIFY_CLIENT_ID is not set.")
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "client_id": cid,
        "response_type": "code",
        "redirect_uri": redirect or redirect_uri(),
        "code_challenge_method": "S256",
        "code_challenge": challenge(verifier),
        "state": state,
        "scope": " ".join(SCOPES),
    })


# ── tokens ───────────────────────────────────────────────────────────────────

@dataclass
class Tokens:
    access_token: str
    refresh_token: str
    expires_at: float
    scope: str = ""

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at - 60


def _post_token(form: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(
        TOKEN_URL,
        data=urllib.parse.urlencode(form).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SpotifyAuthError(f"Spotify token request failed ({e.code}): {detail}") from e


def _tokens(data: Dict[str, Any], previous_refresh: str = "") -> Tokens:
    return Tokens(
        access_token=data["access_token"],
        # Spotify may omit refresh_token on refresh; keep the old one then.
        refresh_token=data.get("refresh_token") or previous_refresh,
        expires_at=time.time() + int(data.get("expires_in", 3600)),
        scope=data.get("scope", ""),
    )


def exchange_code(code: str, verifier: str, *, redirect: Optional[str] = None) -> Tokens:
    return _tokens(_post_token({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect or redirect_uri(),
        "client_id": client_id() or "",
        "code_verifier": verifier,
    }))


def refresh(tokens: Tokens) -> Tokens:
    return _tokens(_post_token({
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "client_id": client_id() or "",
    }), previous_refresh=tokens.refresh_token)


class TokenStore:
    """Tokens on disk (0600) so the CLI and app share one login."""

    def __init__(self, path: Path = TOKENS_PATH) -> None:
        self.path = path

    def load(self) -> Optional[Tokens]:
        if not self.path.exists():
            return None
        try:
            return Tokens(**json.loads(self.path.read_text()))
        except (ValueError, TypeError):
            return None

    def save(self, tokens: Tokens) -> None:
        self.path.write_text(json.dumps(asdict(tokens)))
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def valid(self) -> Optional[Tokens]:
        """Stored tokens, refreshed if expired. None when logged out or refresh fails."""
        tokens = self.load()
        if tokens is None:
            return None
        if tokens.expired:
            try:
                tokens = refresh(tokens)
            except SpotifyAuthError:
                return None
            self.save(tokens)
        return tokens
