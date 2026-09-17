"""BCTC table normalizer — unit / scope / period helpers for address specialists.

Problem this solves: OCR tables mix raw VND with headers like `31/12/2022Triệu VND`.
Question units (`triệu đồng`, `tỷ đồng`, …) must not be applied twice.
"""

from __future__ import annotations

import re
from pathlib import Path

from build_submission import UNIT_SCALES, unit_of

# Displayed cell is already in this many VND.
_TABLE_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"nghìn\s*tỷ", re.I), 1e12),
    (re.compile(r"trăm\s*tỷ", re.I), 1e11),
    (re.compile(r"(?<![a-zà-ỹ])tỷ\s*(?:vnd|đồng)?|(?:vnd|đồng)?\s*tỷ(?![a-zà-ỹ])", re.I), 1e9),
    (re.compile(r"triệu\s*(?:vnd|đồng)?|triệuvnd", re.I), 1e6),
    (re.compile(r"(?<![a-z])vnd(?![a-z])|(?<![a-zà-ỹ])đồng", re.I), 1.0),
)


def header_text(blob: bytes, limit: int = 6000) -> str:
    return blob[:limit].decode("utf-8", errors="ignore")


def detect_table_unit_vnd(blob_or_text: bytes | str) -> float | None:
    """Scale of one displayed unit relative to VND, or None if unclear."""

    text = header_text(blob_or_text) if isinstance(blob_or_text, bytes) else blob_or_text
    folded = text.casefold()
    # Prefer the strongest (largest) explicit unit marker in the header strip.
    hits: list[float] = []
    for pat, scale in _TABLE_UNIT_PATTERNS:
        if pat.search(folded):
            hits.append(scale)
    if not hits:
        return None
    # If both Triệu and VND appear, Triệu wins (VND is substring noise).
    return max(hits)


def question_unit(question: str) -> tuple[str, float]:
    return unit_of(question)


def effective_divide_unit(question: str, blob: bytes, plan_scale: float = 1.0) -> float:
    """Denominator for `cell * plan_scale / unit` so answer matches question unit.

    When the CSV is already denominated in the same unit the question asks for,
    returns 1.0 (modulo plan_scale handling left to caller).
    """

    _name, q_unit = question_unit(question)
    if not q_unit:
        return 0.0
    table = detect_table_unit_vnd(blob)
    if table is None or table <= 0:
        return q_unit
    # cell_display * table ≈ VND; want VND / q_unit
    # → cell_display * plan_scale * table / q_unit
    # PROGRAM: cell * plan_scale / unit  ⇒  unit = q_unit / table
    return q_unit / table


def wants_beginning(question: str) -> bool:
    q = question.casefold()
    return bool(re.search(
        r"đầu năm|đầu kỳ|số đầu|tại ngày 01[/.-]01|ngày 1[/.-]1[/.-]|1/1/",
        q,
    ))


def wants_ending(question: str) -> bool:
    q = question.casefold()
    if wants_beginning(q):
        return False
    return bool(re.search(
        r"cuối năm|cuối kỳ|số cuối|đến ngày 31[/.-]12|31/12|tại ngày 31",
        q,
    ))


def wants_separate(question: str) -> bool:
    return bool(re.search(
        r"công ty mẹ|báo cáo riêng|riêng lẻ|bctc riêng",
        question, re.I,
    ))


def wants_consolidated(question: str) -> bool:
    return bool(re.search(
        r"hợp nhất|tập đoàn(?!.*công ty mẹ)",
        question, re.I,
    ))


def scope_matches(question: str, doc: str) -> bool:
    """False when question demands mẹ/riêng but doc is consolidated (or reverse)."""

    d = doc.casefold()
    is_cons = "consolidated" in d or "hợp nhất" in d
    is_sep = "separate" in d or "riêng" in d
    if wants_separate(question) and is_cons:
        return False
    if wants_consolidated(question) and is_sep and not is_cons:
        return False
    return True


def label_overlaps_question(label: str, question: str, min_token: int = 2) -> bool:
    """Loose token overlap: row label should appear in the question."""

    if not label or not label.strip():
        return False
    q = question.casefold()
    lab = label.casefold()
    if lab in q:
        return True
    tokens = [t for t in re.split(r"[^0-9a-zà-ỹ]+", lab) if len(t) >= 3]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in q)
    return hits >= min(min_token, max(1, len(tokens) // 2 + 1))


def resolve_csv(path: str | Path, root: Path) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    cand = root / path
    if cand.is_file():
        return cand
    return None
