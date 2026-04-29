"""Per-query trace recorder — captures plan/act/check/explain steps for observability."""

from datetime import datetime, timezone
from typing import Any, Dict, List


class TraceRecorder:
    def __init__(self, query: str) -> None:
        self.query = query
        self.events: List[Dict[str, Any]] = []
        self._started = datetime.now(timezone.utc)

    def record(self, kind: str, **payload: Any) -> None:
        self.events.append({
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        })

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.events)
