"""Exact table-local currency-unit markers from extracted financial reports."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable


def plain(text: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).lower().replace("đ", "d")


_PREFIX = r"(?:(?:don vi(?: tinh)?|dvt|unit|currency)\s*[:\-]?\s*)?"
_SUFFIX = r"\s*[.]?"
_MARKERS = (
    (
        1_000.0,
        re.compile(
            rf"^{_PREFIX}(?:(?:ngan|nghin)\s*(?:vnd|dong)|"
            rf"(?:vnd|dong)\s*(?:ngan|nghin)|thousand\s*(?:vnd|dong)){_SUFFIX}$",
            re.I,
        ),
    ),
    (
        1_000_000.0,
        re.compile(
            rf"^{_PREFIX}(?:trieu\s*(?:vnd|dong)|(?:vnd|dong)\s*trieu|"
            rf"million\s*(?:vnd|dong)){_SUFFIX}$",
            re.I,
        ),
    ),
    (
        1_000_000_000.0,
        re.compile(
            rf"^{_PREFIX}(?:(?<!cong )ty\s*(?:vnd|dong)|"
            rf"(?:vnd|dong)\s*ty|billion\s*(?:vnd|dong)){_SUFFIX}$",
            re.I,
        ),
    ),
)
_BASE = re.compile(
    r"^(?:(?:don vi(?: tinh)?|dvt|unit|currency)\s*[:\-]?\s*)?"
    r"(?:vnd|dong|dong viet nam|viet nam dong)\s*[.]?$",
    re.I,
)
_HEADER_MARKERS = (
    (1_000.0, re.compile(r"(?:ngan|nghin)\s*(?:vnd|dong)|(?:vnd|dong)\s*(?:ngan|nghin)(?!\s*han)", re.I)),
    (1_000_000.0, re.compile(r"trieu\s*(?:vnd|dong)|(?:vnd|dong)\s*trieu", re.I)),
    (1_000_000_000.0, re.compile(r"(?<!cong )ty\s*(?:vnd|dong)|(?:vnd|dong)\s*ty(?!\s*le)", re.I)),
)


def unit_factor_from_marker(text: object) -> float | None:
    normalized = " ".join(plain(re.sub(r"<[^>]+>", " ", str(text))).split())
    for factor, pattern in _MARKERS:
        if pattern.fullmatch(normalized):
            return factor
    if len(normalized) <= 100 and _BASE.fullmatch(normalized):
        return 1.0
    return None


def unit_factor_from_table_header(text: object) -> float | None:
    header_rows = "</tr>".join(str(text).split("</tr>")[:2])
    normalized = " ".join(plain(re.sub(r"<[^>]+>", " ", header_rows)).split())
    for factor, pattern in _HEADER_MARKERS:
        if pattern.search(normalized):
            return factor
    return None


def nearest_unit_marker(
    lines: list[str], table_line: int, *, max_distance: int = 60
) -> dict | None:
    """Return the nearest explicit marker at or before a 1-based table line."""

    if not lines or table_line < 1:
        return None
    end = min(table_line, len(lines))
    start = max(1, end - max_distance)
    for line_number in range(end, start - 1, -1):
        text = lines[line_number - 1].strip()
        if not text:
            continue
        factor = (
            unit_factor_from_table_header(text)
            if "<table" in text.casefold()
            else unit_factor_from_marker(text)
        )
        if factor is not None:
            return {
                "factor": factor,
                "line": line_number,
                "distance": table_line - line_number,
                "text": text,
            }
    return None


def extracted_text_index(data_root: str | Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in Path(data_root).rglob("*_extracted.txt"):
        document = path.parent.name
        if path.name == f"{document}_extracted.txt":
            index[document] = path
    return index


def build_local_unit_factors(
    entries: Iterable[dict],
    data_root: str | Path,
    *,
    max_distance: int = 60,
) -> tuple[dict[str, float], dict[str, int]]:
    """Build a conservative ``table_ref -> factor`` map from explicit markers."""

    grouped: dict[str, list[tuple[str, int]]] = {}
    for entry in entries:
        report = str(entry.get("report_id", ""))
        table_ref = str(entry.get("table_ref", ""))
        try:
            line = int(entry.get("line", table_ref.rsplit("|", 1)[-1]))
        except (TypeError, ValueError):
            continue
        if report and table_ref:
            grouped.setdefault(report, []).append((table_ref, line))
    index = extracted_text_index(data_root)
    factors: dict[str, float] = {}
    documents_read = 0
    for report, refs in grouped.items():
        path = index.get(report)
        if path is None:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        documents_read += 1
        for table_ref, line in refs:
            marker = nearest_unit_marker(lines, line, max_distance=max_distance)
            if marker is not None:
                factors[table_ref] = float(marker["factor"])
    return factors, {
        "unit_documents_indexed": len(index),
        "unit_documents_read": documents_read,
        "local_unit_tables": len(factors),
    }

