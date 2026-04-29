"""Hybrid tool: structured filter narrows the candidate set, then semantic search ranks within it.

Useful when the user states both attribute constraints and a vibe (e.g. "chill study
music, no vocals" - vibe + instrumental constraint).
"""

from typing import Any, Dict, List

from ..llm.types import Plan
from . import semantic_search, structured_filter


def run(plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    narrowed = structured_filter.run(plan, songs)
    if not plan.semantic_query or not narrowed:
        return narrowed
    return semantic_search.run(plan, narrowed)
