"""Vietnamese-style number parsing for OCR'd financial statements.

Handles:
- thousand separator dots:  "1.234.567"      -> 1234567.0
- decimal commas:           "12,5", "5,832" -> 12.5, 5.832
- negatives in parentheses: "(1.839)"        -> -1839.0   (Thong tu 200 style)
- EN-style fallback:        "1,234,567.89"   -> 1234567.89
- OCR whitespace / weird dashes / footnote marks / currency suffixes
"""
from __future__ import annotations

import re

_WS = re.compile(r"[\s   ]+")
_FOOTNOTE = re.compile(r"\(\*+\)")
_CURRENCY_TAIL = re.compile(r"(?:đ|₫|vnđ|vnd|đồng|dong)\s*$", re.IGNORECASE)
_LETTER = re.compile(r"[A-Za-zÀ-ỹà-ỹĐđ]")


def parse_vn_number(raw) -> float | None:
    """Parse a table cell into a float. Returns None if not a number."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            f = float(raw)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # reject NaN
    s = str(raw).strip()
    if not s:
        return None
    s = _FOOTNOTE.sub("", s)
    s = _WS.sub("", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = _CURRENCY_TAIL.sub("", s)

    neg = False
    m = re.fullmatch(r"\((.+)\)", s)
    if m:
        neg = True
        s = m.group(1)
    if s.startswith("-"):
        neg = True
        s = s.lstrip("-")
    s = s.replace("%", "")

    # any remaining letter -> it's text, not a number
    if _LETTER.search(s):
        return None
    digits = re.sub(r"[^0-9.,]", "", s)
    if not re.search(r"\d", digits):
        return None

    try:
        val = _resolve_separators(digits)
    except (ValueError, OverflowError):
        return None
    if val is None:
        return None
    return -val if neg else val


def _resolve_separators(s: str) -> float | None:
    dots, commas = s.count("."), s.count(",")
    if dots and commas:
        # the rightmost separator is the decimal separator
        if s.rfind(".") > s.rfind(","):
            return float(s.replace(",", ""))
        return float(s.replace(".", "").replace(",", "."))
    if dots:
        parts = s.split(".")
        # A leading zero makes the intent unambiguous: ratios such as 0.123 or
        # 0.9237 are decimals, never thousands groupings.
        if len(parts) == 2 and parts[0] == "0" and parts[1] != "":
            return float(s)
        if len(parts) > 1 and parts[0] != "" and len(parts[0]) <= 3 and all(
            len(p) == 3 for p in parts[1:]
        ):
            return float("".join(parts))          # 1.234.567 -> thousands
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            return float(s)                        # 10.5 -> decimal
        if len(parts) == 2 and parts[0] == "":
            return float(s)                        # .5
        # weird grouping: strip all dots (OCR-mangled thousands)
        return float("".join(parts))
    if commas:
        parts = s.split(",")
        if len(parts) == 2 and parts[0] != "" and parts[1] != "":
            # Locale-first for this Vietnamese corpus: a lone comma is the
            # decimal mark, including ratios with 3+ fractional digits
            # ("5,832 tỷ", "0,9237"). Multiple commas remain EN grouping.
            return float(parts[0] + "." + parts[1])
        return float("".join(parts))                  # 1,234,567 -> thousands
    return float(s)


def looks_numeric(raw) -> bool:
    return parse_vn_number(raw) is not None


def is_year_like(val: float) -> bool:
    return val == int(val) and 1990 <= val <= 2035
