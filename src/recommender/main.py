"""CLI entry point for the agent-orchestrated recommender.

Usage:
  python -m recommender.main --query "chill study music, no vocals"

Uses Claude when credentials are available (ANTHROPIC_API_KEY in the environment or
.env, or an `ant auth login` profile); otherwise runs fully offline with a heuristic
planner and grounded template explanations.
"""

import argparse
import sys
from typing import Any, Dict

from . import config
from .core.io import load_songs
from .llm import agent


PROFILES: Dict[str, Dict[str, Any]] = {
    "high_energy_pop": {
        "name": "High-Energy Pop",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.88,
        "likes_acoustic": False,
    },
    "chill_lofi": {
        "name": "Chill Lofi",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
    },
    "deep_intense_rock": {
        "name": "Deep Intense Rock",
        "genre": "rock",
        "mood": "intense",
        "energy": 0.91,
        "likes_acoustic": False,
    },
}


def _print_trace(events):
    print("\n  ── Agent trace ──")
    for ev in events:
        kind = ev["kind"]
        if kind == "plan":
            p = ev["plan"]
            print(f"  [PLAN]    intent={p['intent']!r}")
            print(f"            tool={p['chosen_tool']}, "
                  f"expected={p['expected_count_min']}-{p['expected_count_max']}, "
                  f"genre={p['genre_filter']}, mood={p['mood_filter']}, "
                  f"bpm=[{p['bpm_min']},{p['bpm_max']}], "
                  f"energy=[{p['energy_min']},{p['energy_max']}]")
            if p.get("semantic_query"):
                print(f"            semantic_query={p['semantic_query']!r}")
        elif kind == "act":
            print(f"  [ACT]     tool={ev['tool']}, "
                  f"candidates={ev['candidate_count']}, ids={ev['candidate_ids']}")
        elif kind == "check":
            v = ev["verdict"]
            print(f"  [CHECK]   confidence={v['confidence']:.2f}, "
                  f"flags={v['flags']}, note={v['note']!r}")
        elif kind == "refine":
            print(f"  [REFINE]  confidence={ev['confidence']:.2f}, "
                  f"flags={ev['flags']}, from={ev['from_tool']} -> {ev['to_tool']}")
        elif kind == "refine_stop":
            print(f"  [STOP]    {ev['reason']}")
        elif kind == "respond":
            print(f"  [RESPOND] returned {ev['recommendation_count']} recommendations")
        elif kind == "explainer_fallback":
            ids = ev.get("song_ids", "all")
            print(f"  [WARN]    explainer fell back (songs={ids}): {ev['error']}")
        elif kind == "plan_fallback":
            print(f"  [WARN]    planner fell back to heuristic plan: {ev['error']}")
    print()


def _spotify_profile(songs):
    from .spotify import SpotifyClient, TokenStore, build_profile
    from .spotify import auth

    store = TokenStore()
    tokens = store.valid()
    if tokens is None:
        raise RuntimeError("Not logged in to Spotify. Run: uv run python scripts/spotify_login.py")

    def on_refresh(t):
        fresh = auth.refresh(t)
        store.save(fresh)
        return fresh

    return build_profile(SpotifyClient(tokens, on_refresh=on_refresh).fetch_history(), songs)


def _print_timings(timings):
    if not timings:
        return
    stages = ", ".join(f"{k}={v}ms" for k, v in timings.items() if k != "total")
    print(f"  ── Latency: total={timings.get('total', 0)}ms ({stages})\n")


def _print_recs(result):
    if not result.recommendations:
        print("  (no recommendations)\n")
        return
    print("  ── Recommendations ──")
    for i, rec in enumerate(result.recommendations, 1):
        s = rec.song
        print(f"  {i}. {s['title']!r} by {s['artist']} "
              f"[{s['genre']}/{s['mood']}, score={rec.score:.3f}]")
        print(f"     {rec.narrative}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agent-orchestrated music recommender")
    parser.add_argument("--query", "-q", required=True,
                        help="Free-text query, e.g. \"chill study music, no vocals\"")
    parser.add_argument("--profile", "-p", default="chill_lofi", choices=list(PROFILES),
                        help="Pre-defined listener profile (default: chill_lofi)")
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--no-llm-explainer", action="store_true",
                        help="Skip the LLM explainer; use templated narratives")
    parser.add_argument("--spotify", action="store_true",
                        help="Use your Spotify listening history (run scripts/spotify_login.py first)")
    parser.add_argument("--songs", default=str(config.SONGS_CSV),
                        help="Path to songs.csv")
    args = parser.parse_args(argv)

    songs = load_songs(args.songs)
    profile = PROFILES[args.profile]
    if args.spotify:
        try:
            profile = _spotify_profile(songs)
        except (RuntimeError, ValueError) as e:
            print(f"\n  ERROR: {e}", file=sys.stderr)
            return 2

    print("=" * 82)
    print(f"  Query   : {args.query!r}")
    print(f"  Profile : {profile['name']} "
          f"(genre={profile['genre']}, mood={profile['mood']}, energy={profile['energy']})")
    print(f"  Catalog : {len(songs)} songs")
    print("=" * 82)

    try:
        result = agent.run(
            args.query, profile, songs,
            k=args.k,
            use_llm_explainer=not args.no_llm_explainer,
        )
    except RuntimeError as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        return 2

    _print_trace(result.trace_events)
    _print_recs(result)
    _print_timings(result.timings_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
