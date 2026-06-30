"""Log in to Spotify from the terminal (PKCE) and preview your taste profile.

  uv run python scripts/spotify_login.py            # log in, then print profile
  uv run python scripts/spotify_login.py --profile  # reuse saved login
  uv run python scripts/spotify_login.py --port 8889  # use another registered port

Needs SPOTIFY_CLIENT_ID in .env and this redirect URI registered on the app:
  http://127.0.0.1:<port>/callback   (default port 8888)
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

DEFAULT_PORT = 8888


def login(port: int = DEFAULT_PORT) -> None:
    redirect = f"http://127.0.0.1:{port}/callback"
    state, verifier = secrets.token_urlsafe(16), auth.new_verifier()
    login_url = auth.authorize_url(state, verifier, redirect=redirect)
    result = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            if url.path == "/":  # convenience entry point (e.g. VS Code "Open in Browser")
                self.send_response(302)
                self.send_header("Location", login_url)
                self.end_headers()
                return
            # Only the OAuth redirect ends the wait; favicon or stray requests don't.
            if url.path != "/callback" or not ({"code", "error"} & query.keys()):
                self.send_response(404)
                self.end_headers()
                return
            result.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            message = b"Spotify connected. You can close this tab." if "code" in query else b"Login cancelled."
            self.wfile.write(b"<h3>" + message + b"</h3>")
            done.set()

        def log_message(self, *args):
            pass

    def read_pasted() -> None:
        """Fallback for browsers that won't follow the redirect to 127.0.0.1 (e.g.
        an in-app browser): paste the address Spotify tried to open."""
        for line in sys.stdin:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(line.strip()).query)
            if "code" in query or "error" in query:
                result.update({k: v[0] for k, v in query.items()})
                done.set()
                return
            print("That doesn't look like the redirect address (it should contain ?code=...).", flush=True)

    # Threaded: an idle connection (browser pre-connect, VS Code port forwarding)
    # must not block the real callback request.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=read_pasted, daemon=True).start()
    print(f"Open this link in your regular browser to log in to Spotify (waiting 5 minutes):\n\n{login_url}\n\n"
          f"(or http://127.0.0.1:{port}/ if this port is forwarded to your browser's machine)\n", flush=True)
    print(f"If the browser says it can't open 127.0.0.1:{port} after you click Agree, copy the\n"
          f"full address it tried to open ({redirect}?code=...) and paste it here.\n",
          flush=True)
    webbrowser.open(login_url)
    done.wait(timeout=300)
    server.shutdown()

    if result.get("state") != state or "code" not in result:
        raise SystemExit(f"Login failed: {result.get('error', 'no code returned')}")
    TokenStore().save(auth.exchange_code(result["code"], verifier, redirect=redirect))
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
    print(f"\n{profile['name']}: {s['tracks']} tracks, {s['matched_to_catalog']} in catalog, "
          f"{s.get('labelled', 0)} labelled ({client.calls} API calls)")
    print(f"Derived profile: genre={profile['genre']}, mood={profile['mood']}, "
          f"energy={profile['energy']}, valence={profile['target_valence']}, "
          f"likes_acoustic={profile['likes_acoustic']}")
    print("Genre mix:", s["genre_mix"])
    print("Spotify genres:", ", ".join(s["top_artist_genres"]) or "(none returned)")
    print("Top tracks:\n  " + "\n  ".join(s["top_tracks"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", action="store_true", help="skip login, reuse saved tokens")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="local callback port; http://127.0.0.1:<port>/callback must be registered")
    args = parser.parse_args()
    if not auth.is_configured():
        raise SystemExit("Set SPOTIFY_CLIENT_ID in .env first.")
    if not args.profile:
        login(args.port)
    show_profile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
