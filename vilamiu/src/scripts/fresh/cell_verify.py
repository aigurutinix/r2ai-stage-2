"""Arithmetic verification gates for BCTC cell reads.

Human readers sanity-check: roll-ups (270=100+200), note Cộng ties, scope/year
metadata. This module refuses patches that fail cheap offline checks — it never
invents numbers.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import parse_statements as ps
from identity_check import doc_union_rollup_ok, table_identity_score

PERIOD_LABEL_RE = re.compile(
    r"^\s*(?:năm|số|tại ngày|cuối|đầu|kỳ)\s*(?:nay|trước|này|cuối năm|đầu năm)?\s*"
    r"[\d/.\s-]*$", re.I,
)
CONG_RE = re.compile(r"\bcộng\b|\btổng\b|\btổng cộng\b", re.I)


def _read_grid(blob: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(blob.decode("utf-8-sig"))))


def verify_maso_table(path: Path, code: str, kind: str) -> str | None:
    """Reject CDKT tables whose doc union fails TT200 roll-ups."""

    if kind.lower() != "cdkt":
        return None
    if not path.is_file():
        return "missing_csv"
    parent = path.parent
    dok = doc_union_rollup_ok(parent)
    if dok is False:
        return "cdkt_rollup_fail"
    scored = table_identity_score(path)
    if scored is None:
        return None
    code = str(code).lstrip("t")
    if code not in scored["stmt"].current:
        return "code_missing"
    if scored.get("applicable") and float(scored.get("score") or 0) < 0.5:
        return "table_identity_weak"
    return None


def verify_note_cong(blob: bytes, row: int, col: int, label: str) -> str | None:
    """If the row looks like Cộng, require sum(children) ≈ total in same column."""

    if not CONG_RE.search(label or ""):
        return None
    grid = _read_grid(blob)
    if row + 1 >= len(grid):
        return None
    header_rows = min(3, len(grid))
    body_start = 1
    children: list[float] = []
    for ridx in range(body_start, len(grid)):
        if ridx == row + 1:
            continue
        row_cells = grid[ridx]
        if not row_cells:
            continue
        row_label = max(
            (str(c).strip() for c in row_cells if not any(ch.isdigit() for ch in str(c))),
            key=len, default="",
        )
        if not row_label or PERIOD_LABEL_RE.match(row_label):
            continue
        if CONG_RE.search(row_label):
            continue
        if col >= len(row_cells):
            continue
        val = ps.parse_vn_number(str(row_cells[col]))
        if val is not None:
            children.append(val)
    if len(children) < 2:
        return None
    total_cell = grid[row + 1][col] if col < len(grid[row + 1]) else ""
    total = ps.parse_vn_number(str(total_cell))
    if total is None:
        return None
    rhs = sum(children)
    denom = max(abs(total), abs(rhs), 1.0)
    if abs(total - rhs) / denom > 0.02:
        return "cong_mismatch"
    return None


def verify_patch(
    layer: str,
    blob: bytes,
    path: Path | None,
    meta: dict,
    question: str,
) -> str | None:
    """Return reject reason or None if checks pass / not applicable."""

    if layer == "maso_plan":
        code = str(meta.get("code") or "")
        kind = str(meta.get("source") or meta.get("kind") or "cdkt")
        if path is not None:
            reason = verify_maso_table(path, code, kind)
            if reason:
                return reason
        if meta.get("identity_doc_ok") is False and not meta.get("identity_flipped"):
            return "identity_doc_fail"
    elif layer == "note_plan":
        label = str(meta.get("row_label") or "")
        row = int(meta.get("row") or 0)
        col = int(meta.get("col") or 1)
        reason = verify_note_cong(blob, row, col, label)
        if reason:
            return reason
    elif layer == "dual_tab":
        if path is not None:
            reason = verify_maso_table(
                path, str(meta.get("code") or ""), str(meta.get("kind") or "cdkt"))
            if reason:
                return reason
    return None
