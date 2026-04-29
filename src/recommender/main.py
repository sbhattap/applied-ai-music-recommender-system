"""CLI entry point for the agent-orchestrated recommender.

Usage:
  python -m recommender.main --query "chill study music, no vocals"

Reads GEMINI_API_KEY from the environment (or .env). Exits non-zero with a
clear message if the key is missing.
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
        elif kind == "respond":
            print(f"  [RESPOND] returned {ev['recommendation_count']} recommendations")
        elif kind == "explainer_fallback":
            print(f"  [WARN]    explainer fell back for song id={ev['song_id']}: {ev['error']}")
    print()


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
    parser.add_argument("--songs", default=str(config.SONGS_CSV),
                        help="Path to songs.csv")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    songs = load_songs(args.songs)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
