"""Source-derived evidence slots for table selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .retrieval import tokenize


PERIOD_WORDS = {"nam", "quy", "thang", "ngay", "ky", "tai", "cuoi", "dau"}
ENTITY_PHRASES = ("cong ty", "ngan hang", "tap doan", "ctcp", "chi nhanh", "quy dau tu")


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def load_line_items(path: Path) -> tuple[str, ...]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    normalized = {normalize(label) for label in labels} - {""}
    return tuple(sorted(normalized, key=lambda label: (-len(label), label)))


def named_line_items(question: str, labels: tuple[str, ...], limit: int = 3) -> list[str]:
    """Return the longest corpus line items explicitly named by a question."""
    text = f" {normalize(question)} "
    found = []
    for label in labels:
        words = label.split()
        if not words or all(word in PERIOD_WORDS or word.isdigit() for word in words):
            continue
        if any(phrase in label for phrase in ENTITY_PHRASES):
            continue
        if f" {label} " not in text:
            continue
        if any(label in longer for longer in found):
            continue
        found.append(label)
        if len(found) == limit:
            break
    return found


def select_slot_tables(
    record: dict, base_order: list[str], line_items: list[str], budget: int,
) -> tuple[list[str], list[dict]]:
    """Reserve the best exact carrier for every report/item slot, then fill."""
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    ordered = [table_id for table_id in base_order if table_id in candidates]
    seen = set(ordered)
    ordered.extend(table_id for table_id in candidates if table_id not in seen)
    position = {table_id: rank for rank, table_id in enumerate(ordered)}

    slots = []
    reserved = set()
    for report_id in record.get("selected_docs", []):
        report_tables = [
            table_id for table_id in ordered
            if table_id.rsplit("|", 1)[0] == report_id
        ]
        for item in line_items:
            matches = [
                table_id for table_id in report_tables
                if f" {item} " in f" {normalize(candidates[table_id].get('text', ''))} "
            ]
            chosen = min(matches, key=position.get) if matches else None
            slots.append({"report_id": report_id, "line_item": item, "table_id": chosen})
            if chosen:
                reserved.add(chosen)

    ranked_reserved = sorted(reserved, key=position.get)
    selected = ranked_reserved + [table_id for table_id in ordered if table_id not in reserved]
    return selected[:budget], slots


def select_confidence_slot_tables(
    record: dict,
    base_order: list[str],
    line_items: list[str],
    scores: dict[str, float],
    logit_gap: float = 4.0,
    max_per_report: int = 2,
    require_item_for_extras: bool = False,
) -> tuple[list[str], list[dict]]:
    """Keep one report floor, then only high-confidence item carriers."""
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    ordered = [table_id for table_id in base_order if table_id in candidates]
    seen = set(ordered)
    ordered.extend(table_id for table_id in candidates if table_id not in seen)
    position = {table_id: rank for rank, table_id in enumerate(ordered)}

    scored = {table_id: float(value) for table_id, value in scores.items()
              if table_id in candidates}
    top_logit = max((_logit(value) for value in scored.values()), default=None)
    eligible = {
        table_id for table_id, value in scored.items()
        if top_logit is not None and top_logit - _logit(value) <= logit_gap
    }

    def report(table_id: str) -> str:
        return table_id.rsplit("|", 1)[0]

    selected: set[str] = set()
    slots = []
    for report_id in record.get("selected_docs", []):
        report_tables = [table_id for table_id in ordered if report(table_id) == report_id]
        for item in line_items:
            matches = [
                table_id for table_id in report_tables
                if table_id in eligible and
                f" {item} " in f" {normalize(candidates[table_id].get('text', ''))} "
            ]
            chosen = min(matches, key=position.get) if matches else None
            slots.append({"report_id": report_id, "line_item": item, "table_id": chosen})
            if chosen:
                selected.add(chosen)

    # An unscored candidate may satisfy the report floor, but never become an extra.
    for report_id in record.get("selected_docs", []):
        report_tables = [table_id for table_id in ordered if report(table_id) == report_id]
        if report_tables:
            selected.add(report_tables[0])

    for table_id in ordered:
        if table_id not in eligible or table_id in selected:
            continue
        report_id = report(table_id)
        if require_item_for_extras and line_items:
            text = f" {normalize(candidates[table_id].get('text', ''))} "
            if not any(f" {item} " in text for item in line_items):
                continue
        if sum(report(item) == report_id for item in selected) < max_per_report:
            selected.add(table_id)

    return [table_id for table_id in ordered if table_id in selected], slots


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(probability / (1.0 - probability))


def append_confidence_item_tables(
    record: dict,
    starting_order: list[str],
    candidate_order: list[str],
    line_items: list[str],
    scores: dict[str, float],
    logit_gap: float = 4.0,
    max_per_report: int = 3,
) -> list[str]:
    """Append scored item carriers to complex-question rankings only."""
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    selected = list(dict.fromkeys(starting_order))
    seen = set(selected)
    reports = set(record.get("selected_docs", []))
    counts = {report_id: sum(
        table_id.rsplit("|", 1)[0] == report_id for table_id in selected
    ) for report_id in reports}
    scored = {table_id: float(value) for table_id, value in scores.items()
              if table_id in candidates}
    top_logit = max((_logit(value) for value in scored.values()), default=None)
    for table_id in candidate_order:
        if table_id in seen or table_id not in scored:
            continue
        report_id = table_id.rsplit("|", 1)[0]
        if report_id not in reports or counts.get(report_id, 0) >= max_per_report:
            continue
        if top_logit is not None and top_logit - _logit(scored[table_id]) > logit_gap:
            continue
        text = f" {normalize(candidates[table_id].get('text', ''))} "
        if line_items and not any(f" {item} " in text for item in line_items):
            continue
        selected.append(table_id)
        seen.add(table_id)
        counts[report_id] = counts.get(report_id, 0) + 1
    return selected
