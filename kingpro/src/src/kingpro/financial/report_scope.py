"""Resolve report scope from the nearest physical financial-statement masthead.

Filename/container scope remains a fallback only.  OCR corpora can contain a
misnamed container (for example HUT 2024), while the statement masthead and
form code inside the report remain authoritative.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from kingpro.retrieval.bm25_index import fold


_MASTHEAD = re.compile(
    r"^(?:bang can doi ke toan|bao cao tinh hinh tai chinh|"
    r"bao cao ket qua hoat dong kinh doanh|bao cao luu chuyen tien te|"
    r"ban thuyet minh bao cao tai chinh|thuyet minh bao cao tai chinh|"
    r"bao cao tai chinh)"
    r".{0,80}\b(?P<scope>hop nhat|rieng)\b"
)


@lru_cache(maxsize=512)
def _lines(path: str) -> tuple[str, ...]:
    try:
        return tuple(Path(path).read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError):
        return ()


def container_scope(report_id: str) -> str:
    name = str(report_id).casefold()
    if "consolidated" in name:
        return "consolidated"
    if "separate" in name:
        return "separate"
    if "aggregated" in name:
        return "aggregated"
    return "unknown"


def extracted_report_path(entry: dict[str, Any], root: str | Path) -> Path:
    report_id = str(entry["report_id"])
    return (
        Path(root)
        / "data"
        / "financial_statements"
        / str(entry["ticker"])
        / str(entry["year"])
        / report_id
        / f"{report_id}_extracted.txt"
    )


def physical_scope(
    entry: dict[str, Any],
    root: str | Path,
    *,
    backward_window: int = 120,
) -> tuple[str, dict[str, Any] | None]:
    """Return physical scope and the masthead evidence, falling back safely.

    ``entry['line']`` is the one-based source line containing the table.  The
    nearest preceding explicit statement/report masthead wins.  A bounded
    backwards search avoids treating remote note prose as a scope declaration.
    """

    fallback = container_scope(str(entry.get("report_id", "")))
    report_path = extracted_report_path(entry, root)
    lines = _lines(str(report_path.resolve()))
    if not lines:
        return fallback, None
    anchor = min(max(int(entry.get("line", 1)) - 1, 0), len(lines))
    lower = max(0, anchor - max(1, int(backward_window)))
    for index in range(anchor - 1, lower - 1, -1):
        text = lines[index].strip()
        if not text:
            continue
        match = _MASTHEAD.search(fold(text))
        if match is None:
            continue
        scope = "consolidated" if match.group("scope") == "hop nhat" else "separate"
        return scope, {
            "report_path": str(report_path),
            "line": index + 1,
            "text": text,
            "container_scope": fallback,
            "physical_scope": scope,
            "overrides_container": scope != fallback,
        }
    return fallback, None
