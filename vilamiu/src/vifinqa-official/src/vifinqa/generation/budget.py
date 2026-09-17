"""Shared runtime budget for question-generation pipelines.

The budget counts calls immediately before ``ChatLLM.complete`` and remains independent
of the concrete LLM implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    calls: int
    max_calls: int | None
    elapsed_seconds: float


class GenerationBudgetExceeded(RuntimeError):
    """Intentional stop when the pipeline is about to exceed its LLM budget."""

    def __init__(self, *, candidate: object, stage: str, snapshot: BudgetSnapshot) -> None:
        self.candidate = candidate
        self.stage = stage
        self.snapshot = snapshot
        super().__init__(
            f"LLM call budget exhausted: candidate={candidate!r} stage={stage} "
            f"calls={snapshot.calls}/{snapshot.max_calls} "
            f"elapsed={snapshot.elapsed_seconds:.3f}s"
        )


class LLMCallBudget:
    """Thread-safe counter that blocks the next call after ``max_calls`` is reached.

    ``max_calls=None`` enables telemetry without a limit. In-flight calls are not
    cancelled; the limit is checked immediately before each completion call.
    """

    def __init__(self, max_calls: int | None = None) -> None:
        if max_calls is not None and max_calls < 1:
            raise ValueError("max_calls must be at least 1 or None")
        self._max_calls = max_calls
        self._calls = 0
        self._started_at = time.monotonic()
        self._lock = Lock()

    def before_call(self, *, candidate: object, stage: str) -> None:
        with self._lock:
            if self._max_calls is not None and self._calls >= self._max_calls:
                raise GenerationBudgetExceeded(
                    candidate=candidate,
                    stage=stage,
                    snapshot=self._snapshot_unlocked(),
                )
            self._calls += 1

    def exhausted(self) -> bool:
        with self._lock:
            return self._max_calls is not None and self._calls >= self._max_calls

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            calls=self._calls,
            max_calls=self._max_calls,
            elapsed_seconds=time.monotonic() - self._started_at,
        )
