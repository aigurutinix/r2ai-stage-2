"""Vietnamese text and financial-number normalisation helpers."""

from __future__ import annotations

import html
import math
import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLAIN_NUMBER_RE = re.compile(r"^\d+$")
_VN_GROUPED_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_EN_GROUPED_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DECIMAL_RE = re.compile(r"^\d+[.,]\d{1,2}$")
_SPACE_GROUPED_RE = re.compile(r"^\d{1,3}(?:\s\d{3})+$")


def clean_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value))
    return _SPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def fold_text(value: object) -> str:
    text = clean_text(value).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def tokens(value: object) -> tuple[str, ...]:
    return tuple(fold_text(value).split())


def parse_vn_number(value: object) -> float | None:
    """Parse common Vietnamese financial-table numbers without guessing prose."""

    raw = clean_text(value)
    if not raw or raw.casefold() in {"-", "–", "—", "n/a", "na", "nil"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1].strip()
    raw = raw.replace("−", "-").replace("–", "-").strip()
    if raw.startswith(('+', '-')):
        if raw.startswith('-'):
            negative = True
        raw = raw[1:].strip()
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1].strip()
    valid = any(
        pattern.fullmatch(raw)
        for pattern in (_PLAIN_NUMBER_RE, _VN_GROUPED_RE, _EN_GROUPED_RE, _DECIMAL_RE, _SPACE_GROUPED_RE)
    )
    if not valid:
        return None

    # Vietnamese percentages use a decimal comma even when three decimal
    # places are shown (for example ``99,999%``).  Treating that form as an
    # English thousands group silently turns a valid ownership percentage into
    # 99,999.  Percent context is unambiguous, so resolve it before the generic
    # grouped-number branches.
    if is_percent and "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    elif _SPACE_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(" ", "")
    elif _EN_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(",", "")
    elif _VN_GROUPED_RE.fullmatch(raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        left, right = raw.rsplit(",", 1)
        if is_percent or len(right) < 3:
            raw = left.replace(",", "") + "." + right
        else:
            raw = left.replace(",", "") + right
    elif "." in raw:
        parts = raw.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            raw = "".join(parts)
    try:
        number = float(raw)
    except ValueError:
        return None
    if negative:
        number = -abs(number)
    if not math.isfinite(number):
        return None
    return number


def english_vnd_scale(text: object) -> float | None:
    """Recognise explicit English VND unit labels in either word order.

    Currency and multiplier must be adjacent (possibly in parentheses), so a
    narrative mention of millions of shares cannot scale a VND column.
    """
    folded = fold_text(text)
    multiplier = r"(thousands?|millions?|billions?|trillions?)"
    separator = r"[\s()\[\]:]*"
    match = re.search(r"(?<![a-z])vnd" + separator + multiplier + r"\b", folded)
    if match is None:
        match = re.search(r"\b" + multiplier + separator + r"vnd\b", folded)
    if match is None:
        return None
    return {"thousand": 1e3, "million": 1e6, "billion": 1e9,
            "trillion": 1e12}[match.group(1).rstrip("s")]


def source_scale(text: object) -> float:
    folded = fold_text(text)
    english_scale = english_vnd_scale(folded)
    if english_scale is not None:
        return english_scale
    if "nghin ty" in folded:
        return 1_000_000_000_000.0
    if "tram ty" in folded:
        return 100_000_000_000.0
    if "ty dong" in folded or "ty vnd" in folded or folded.endswith(" ty"):
        return 1_000_000_000.0
    if "trieu dong" in folded or "trieu vnd" in folded:
        return 1_000_000.0
    if any(unit in folded for unit in ("nghin dong", "ngan dong", "nghin vnd", "ngan vnd")):
        return 1_000.0
    return 1.0


def requested_scale(question: str) -> float:
    folded = fold_text(question)
    # Report tables can be denominated directly in a foreign currency.  In
    # that case ``source_scale`` correctly stays at one, while the requested
    # answer may still ask for (for example) *million USD*.
    if "trieu usd" in folded or "trieu do la my" in folded:
        return 1_000_000.0
    return source_scale(question)
