"""Shared types for the agent — Pydantic for LLM I/O, dataclasses for internals."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


ToolName = Literal["semantic_search", "structured_filter", "hybrid"]


class Plan(BaseModel):
    intent: str = Field(description="One-line restatement of what the user wants.")
    chosen_tool: ToolName
    expected_count_min: int = Field(ge=0, le=30)
    expected_count_max: int = Field(ge=1, le=30)

    semantic_query: Optional[str] = None
    genre_filter: List[str] = Field(default_factory=list)
    mood_filter: List[str] = Field(default_factory=list)
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    energy_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    energy_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    instrumentalness_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    instrumentalness_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CheckFlag(str, Enum):
    LOW_RESULT_COUNT = "low_result_count"
    CONSTRAINT_VIOLATED = "constraint_violated"
    INTENT_MISMATCH = "intent_mismatch"
    LOOKS_GOOD = "looks_good"


class CheckVerdict(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    flags: List[CheckFlag] = Field(default_factory=list)
    note: str


class SemanticPicks(BaseModel):
    """Wire format for semantic_search: just the song ids the LLM picked."""
    song_ids: List[int] = Field(default_factory=list)


class Claim(BaseModel):
    text: str = Field(description="One short factual sentence about why the song fits.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Ids of the evidence items that support this claim, e.g. 'sig:energy', 'artist:bicep'.",
    )


class SongExplanation(BaseModel):
    song_id: int
    claims: List[Claim] = Field(default_factory=list)


class BatchExplanations(BaseModel):
    """Wire format for the batched explainer: cited claims for every recommended song."""
    explanations: List[SongExplanation] = Field(default_factory=list)


@dataclass
class Recommendation:
    song: Dict[str, Any]
    score: float
    signal_explanation: str       # from the deterministic rule-based scorer
    narrative: str                 # rendered from claims (LLM or grounded fallback)
    signals: List[Dict[str, Any]] = field(default_factory=list)      # additive score parts
    evidence: List[Dict[str, Any]] = field(default_factory=list)     # retrieved docs
    retrieval: Optional[Dict[str, Any]] = None                       # ranks, matched terms
    claims: List[Dict[str, Any]] = field(default_factory=list)       # validated, cited
    dropped_claims: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    query: str
    profile: Dict[str, Any]
    plan: Plan
    candidate_count: int
    verdict: CheckVerdict
    recommendations: List[Recommendation]
    trace_events: List[Dict[str, Any]] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(default_factory=dict)
    narratives_from_llm: bool = False
    # Full final candidate pool + retrieval hits, so the UI can re-rank ("what if")
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    hits: Dict[int, Dict[str, Any]] = field(default_factory=dict)
