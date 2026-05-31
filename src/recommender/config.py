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


MODEL = os.environ.get("RECOMMENDER_MODEL", "gemini-3-flash-preview")
# Cheaper/faster model for check + explain; defaults to MODEL so one env var still works.
FAST_MODEL = os.environ.get("RECOMMENDER_FAST_MODEL", MODEL)
EFFORT = os.environ.get("RECOMMENDER_EFFORT", "medium")

# Per-step thinking effort. "minimal" ~= thinking off; extra thinking is the biggest
# latency cost on Gemini 3, so only the planner gets any.
PLAN_EFFORT = os.environ.get("RECOMMENDER_PLAN_EFFORT", "low")
CHECK_EFFORT = "minimal"
EXPLAINER_EFFORT = "minimal"

# Low temperature keeps plans/verdicts stable, which also means fewer refine loops.
PLAN_TEMPERATURE = 0.3
CHECK_TEMPERATURE = 0.2
EXPLAINER_TEMPERATURE = 0.7

PLAN_MAX_TOKENS = 4096
CHECK_MAX_TOKENS = 2048
EXPLAINER_MAX_TOKENS = 4096   # one batched call covers all k songs
SEMANTIC_SEARCH_MAX_TOKENS = 2048

MAX_REFINES = 1
RETRY_CAP = 2  # only used when refine-loop stretch is enabled

REQUEST_TIMEOUT_SECONDS = 60.0

# Retrieval backend for semantic_search/hybrid: "rag" (local hybrid retriever) or
# "llm" (legacy: full catalog in the prompt). "llm" is kept for ablations.
SEMANTIC_BACKEND = os.environ.get("RECOMMENDER_SEMANTIC_BACKEND", "rag")
EMBEDDING_MODEL = os.environ.get("RECOMMENDER_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
