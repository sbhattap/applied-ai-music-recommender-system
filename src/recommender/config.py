"""Centralized config: model IDs, effort levels, retry caps, paths."""

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SONGS_CSV = DATA_DIR / "songs.csv"


MODEL = os.environ.get("RECOMMENDER_MODEL", "gemini-3-flash-preview")
EFFORT = os.environ.get("RECOMMENDER_EFFORT", "medium")

PLAN_MAX_TOKENS = 4096
CHECK_MAX_TOKENS = 2048
EXPLAINER_MAX_TOKENS = 1024
SEMANTIC_SEARCH_MAX_TOKENS = 2048

RETRY_CAP = 2  # only used when refine-loop stretch is enabled

REQUEST_TIMEOUT_SECONDS = 60.0
