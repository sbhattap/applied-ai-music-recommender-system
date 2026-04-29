"""Tool registry — maps the agent's chosen tool name to the callable."""

from typing import Any, Callable, Dict, List

from ..llm.types import Plan, ToolName
from . import hybrid, semantic_search, structured_filter


ToolFn = Callable[[Plan, List[Dict[str, Any]]], List[Dict[str, Any]]]

TOOLS: Dict[ToolName, ToolFn] = {
    "semantic_search": semantic_search.run,
    "structured_filter": structured_filter.run,
    "hybrid": hybrid.run,
}


def call(name: ToolName, plan: Plan, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if name not in TOOLS:
        raise ValueError(f"unknown tool: {name}. valid: {list(TOOLS)}")
    return TOOLS[name](plan, songs)
