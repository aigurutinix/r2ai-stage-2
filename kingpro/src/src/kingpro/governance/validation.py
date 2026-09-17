"""Conservative financial-domain validation independent of the LLM."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


_PART_OF_TOTAL = re.compile(
    r"(?:ty trong|chiem bao nhieu|phan tram (?:cua|trong)|co cau|ty le dong gop)"
)
_COUNT = re.compile(r"(?:bao nhieu (?:cong ty|doanh nghiep|nam|quy)|so luong)")


def validate_financial_result(question: str, answer: Any, unit: str | None) -> dict[str, Any]:
    """Return hard failures only where the requested quantity has a closed domain.

    Growth, margins and leverage can legitimately be negative or exceed 100%, so
    they are not globally clipped.  This validator intentionally prefers a
    warning over a false rejection outside closed-domain intents.
    """
    hard: list[str] = []
    warnings: list[str] = []
    try:
        value = float(answer)
    except (TypeError, ValueError):
        return {"passed": False, "hard_failures": ["non_numeric"], "warnings": []}
    if not math.isfinite(value):
        hard.append("non_finite")
    folded = _fold(question)
    if _PART_OF_TOTAL.search(folded):
        if value < -1e-9 or value > 100.000001:
            hard.append("part_of_total_outside_0_100")
    if _COUNT.search(folded):
        if value < 0 or abs(value - round(value)) > 1e-9:
            hard.append("count_not_nonnegative_integer")
    normalized_unit = _fold(unit or "")
    if "phan tram" in folded and normalized_unit and "phan tram" not in normalized_unit:
        warnings.append("question_unit_output_unit_mismatch")
    if abs(value) > 1e18:
        warnings.append("extreme_magnitude_review")
    return {
        "passed": not hard,
        "hard_failures": hard,
        "warnings": warnings,
        "closed_domain_checks": {
            "part_of_total": bool(_PART_OF_TOTAL.search(folded)),
            "count": bool(_COUNT.search(folded)),
        },
        "policy": "conservative-code-validation-no-global-financial-clipping",
    }

