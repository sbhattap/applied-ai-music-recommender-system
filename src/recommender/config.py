"""Centralized config: model IDs, effort levels, retry caps, paths."""

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"
SONGS_CSV = DATA_DIR / "songs.csv"

# RAG knowledge store: fetched evidence lives in data/rag/, the built index in .cache/
RAG_DATA_DIR = DATA_DIR / "rag"
RAG_INDEX_DIR = CACHE_DIR / "rag_index"


MODEL = os.environ.get("RECOMMENDER_MODEL", "claude-opus-5-5")
# Model for check + explain; defaults to MODEL. Set e.g. claude-haiku-5-5 to trade
# some quality for lower latency and cost on those steps.
FAST_MODEL = os.environ.get("RECOMMENDER_FAST_MODEL", MODEL)

# Per-step effort (low | medium | high | xhigh | max). Effort is the main latency
# knob: these steps are short, structured tasks, so they all run at "low".
PLAN_EFFORT = os.environ.get("RECOMMENDER_PLAN_EFFORT", "low")
CHECK_EFFORT = "low"
EXPLAINER_EFFORT = "low"

# max_tokens covers thinking + output; generous caps avoid truncation and cost
# nothing unless used.
PLAN_MAX_TOKENS = 8000
CHECK_MAX_TOKENS = 8000
EXPLAINER_MAX_TOKENS = 16000   # one batched call covers all k songs
SEMANTIC_SEARCH_MAX_TOKENS = 8000

MAX_REFINES = 1
RETRY_CAP = 2  # only used when refine-loop stretch is enabled

REQUEST_TIMEOUT_SECONDS = 60.0

# Retrieval backend for semantic_search/hybrid: "rag" (local hybrid retriever) or
# "llm" (legacy: full catalog in the prompt). "llm" is kept for ablations.
SEMANTIC_BACKEND = os.environ.get("RECOMMENDER_SEMANTIC_BACKEND", "rag")
EMBEDDING_MODEL = os.environ.get("RECOMMENDER_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
