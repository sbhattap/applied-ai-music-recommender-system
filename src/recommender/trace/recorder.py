"""Per-query trace recorder — captures plan/act/check/explain steps for observability."""

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List


class TraceRecorder:
    def __init__(self, query: str) -> None:
        self.query = query
        self.events: List[Dict[str, Any]] = []
        self.timings: Dict[str, int] = {}
        self._started = datetime.now(timezone.utc)
        self._t0 = time.monotonic()

    def record(self, kind: str, **payload: Any) -> None:
        self.events.append({
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        })

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a pipeline stage. Repeated stages (e.g. check per iteration) accumulate."""
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = int((time.monotonic() - started) * 1000)
            self.timings[name] = self.timings.get(name, 0) + elapsed

    def total_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.events)
