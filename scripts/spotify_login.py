"""Log in to Spotify from the terminal (PKCE) and preview your taste profile.

  uv run python scripts/spotify_login.py            # log in, then print profile
  uv run python scripts/spotify_login.py --profile  # reuse saved login

Needs SPOTIFY_CLIENT_ID in .env and this redirect URI registered on the app:
  http://127.0.0.1:8888/callback
Tokens are saved to .spotify_tokens.json (gitignored), shared with the Streamlit app.
"""

import argparse
import http.server
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.recommender import config  # noqa: E402
from src.recommender.core.io import load_songs  # noqa: E402
from src.recommender.spotify import SpotifyClient, TokenStore, build_profile  # noqa: E402
from src.recommender.spotify import auth  # noqa: E402

CLI_REDIRECT = "http://127.0.0.1:8888/callback"


def login() -> None:
    state, verifier = secrets.token_urlsafe(16), auth.new_verifier()
    result = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>Spotify connected. You can close this tab.</h3>")
            done.set()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 8888), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = auth.authorize_url(state, verifier, redirect=CLI_REDIRECT)
    print(f"Opening Spotify login...\nIf the browser doesn't open, visit:\n{url}\n")
    webbrowser.open(url)
    done.wait(timeout=300)
    server.shutdown()

    if result.get("state") != state or "code" not in result:
        raise SystemExit(f"Login failed: {result.get('error', 'no code returned')}")
    TokenStore().save(auth.exchange_code(result["code"], verifier, redirect=CLI_REDIRECT))
    print("Logged in. Tokens saved to .spotify_tokens.json")


def show_profile() -> None:
    store = TokenStore()
    tokens = store.valid()
    if tokens is None:
        raise SystemExit("Not logged in. Run without --profile first.")
    client = SpotifyClient(tokens, on_refresh=lambda t: (store.save(r := auth.refresh(t)), r)[1])
    history = client.fetch_history()
    profile = build_profile(history, load_songs(str(config.SONGS_CSV)))
    s = profile["history_summary"]
    print(f"\n{profile['name']}: {s['tracks']} tracks, {s['matched_to_catalog']} in catalog "
          f"({client.calls} API calls)")
    print(f"Derived profile: genre={profile['genre']}, mood={profile['mood']}, "
          f"energy={profile['energy']}, valence={profile['target_valence']}, "
          f"likes_acoustic={profile['likes_acoustic']}")
    print("Genre mix:", s["genre_mix"])
    print("Spotify genres:", ", ".join(s["top_artist_genres"]) or "(none returned)")
    print("Top tracks:\n  " + "\n  ".join(s["top_tracks"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", action="store_true", help="skip login, reuse saved tokens")
    args = parser.parse_args()
    if not auth.is_configured():
        raise SystemExit("Set SPOTIFY_CLIENT_ID in .env first.")
    if not args.profile:
        login()
    show_profile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
