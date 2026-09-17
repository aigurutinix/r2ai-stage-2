"""Typed evidence requirements and exact-cell selection.

This is deliberately small: retrieval can expose candidate cells without giving
an LLM permission to invent operands.  The public pipeline can adopt the
contract incrementally while its current ranking remains the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .answers import fold


OPERATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("growth", ("tang truong", "toc do tang", "tang giam")),
    ("difference", ("chenh lech", "khac biet", "hieu so")),
    ("ratio", ("ty le", "ty trong", "bien loi nhuan", "bao nhieu lan")),
    ("average", ("trung binh", "binh quan")),
    ("sum", ("tong", "cong")),
    ("max", ("lon nhat", "cao nhat", "nhieu nhat")),
    ("min", ("nho nhat", "thap nhat", "it nhat")),
)


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    """One required metric/year/role that must bind to a source cell."""

    slot_id: str
    ticker: str | None
    requested_year: int | None
    scope: str | None
    metric: str
    role: str = "value"
    stage: str = "direct"


@dataclass(frozen=True, slots=True)
class EvidenceCell:
    """A source cell candidate offered to the deterministic planner or LLM."""

    slot_id: str
    table_id: str
    row: int
    column: int
    value: float
    score: float = 0.0
    year: int | None = None
    scope: str | None = None
    unit: str | None = None


def operation_hint(question: str) -> str:
    """Return a conservative operation label from arithmetic wording."""
    text = f" {fold(question)} "
    for operation, phrases in OPERATION_PATTERNS:
        if any(f" {phrase} " in text for phrase in phrases):
            return operation
    return "lookup"


def build_evidence_slots(
    question: str,
    metadata: Mapping[str, object],
    metrics: Iterable[str],
) -> tuple[EvidenceSlot, ...]:
    """Build stable typed slots from already-parsed report metadata.

    The function does not infer tickers or years; callers must pass the output
    of the existing document parser.  That keeps this layer from weakening the
    document gate.
    """
    tickers = tuple(str(value) for value in metadata.get("tickers", ()) if value)
    years = tuple(
        int(value)
        for value in metadata.get("slot_years", metadata.get("years", ()))
        if value is not None
    )
    scopes = metadata.get("scope")
    scope = str(scopes) if scopes else None
    normalized_metrics = tuple(dict.fromkeys(fold(metric).strip() for metric in metrics if fold(metric).strip()))
    if not normalized_metrics:
        normalized_metrics = (fold(question).strip(),)
    tickers = tickers or (None,)
    years = years or (None,)
    operation = operation_hint(question)
    slots: list[EvidenceSlot] = []

    def add(ticker: str | None, year: int | None, metric: str, role: str) -> None:
        slot_id = "|".join((
            f"ticker={ticker or '*'}", f"year={year or '*'}", f"scope={scope or '*'}",
            f"metric={metric}", f"role={role}",
        ))
        slots.append(EvidenceSlot(slot_id, ticker, year, scope, metric, role))

    for ticker in tickers:
        for metric in normalized_metrics:
            if operation == "growth" and len(years) == 1 and years[0] is not None:
                add(ticker, years[0] - 1, metric, "prior")
                add(ticker, years[0], metric, "current")
            elif operation == "ratio" and len(normalized_metrics) >= 2:
                for year in years:
                    role = "numerator" if metric == normalized_metrics[0] else "denominator"
                    add(ticker, year, metric, role)
            else:
                for year in years:
                    add(ticker, year, metric, "value")
    return tuple(slots)


def select_best_cells(
    slots: Iterable[EvidenceSlot],
    candidates: Mapping[str, Iterable[EvidenceCell]],
) -> dict[str, EvidenceCell]:
    """Select the highest-scoring candidate independently for each slot."""
    selected: dict[str, EvidenceCell] = {}
    for slot in slots:
        options = tuple(candidates.get(slot.slot_id, ()))
        if not options:
            continue
        selected[slot.slot_id] = max(
            options,
            key=lambda cell: (cell.score, cell.year == slot.requested_year, -cell.row, -cell.column),
        )
    return selected


def complete_slots(slots: Iterable[EvidenceSlot], selected: Mapping[str, EvidenceCell]) -> bool:
    """Whether every required slot has a concrete source cell."""
    return all(slot.slot_id in selected for slot in slots)
