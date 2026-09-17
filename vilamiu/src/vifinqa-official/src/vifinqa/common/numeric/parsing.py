"""Coerce answers to numbers before comparison.

Parsing failures return ``None`` and emit a warning instead of crashing or silently
skipping the value.
"""

from __future__ import annotations

import logging
import math

from vifinqa.constants import ANSWER_ABS_TOL
from vifinqa.common.filtering.table_filters import is_numeric_cell

logger = logging.getLogger(__name__)


def coerce_number(value: object) -> float | None:
    """Coerce a number or localized numeric string to float; return None on failure."""
    if isinstance(value, bool):
        logger.warning("Boolean answers are not treated as numbers: %r", value)
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        logger.warning("Answer is neither a number nor a string and cannot be coerced: %r", value)
        return None

    text = value.strip()
    if not text:
        logger.warning("Empty answer cannot be coerced")
        return None

    try:
        return float(text)
    except ValueError:
        pass

    if is_numeric_cell(text):
        cleaned = text
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        if negative:
            cleaned = cleaned[1:-1]
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        cleaned = cleaned.replace(".", "").replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            logger.warning("Answer matches a localized number format but parsing failed: %r", value)
            return None
        return -num if negative else num

    logger.warning("Could not coerce answer to a number: %r", value)
    return None


def is_correct(expected: object, actual: object, *, abs_tol: float = ANSWER_ABS_TOL) -> bool:
    exp = coerce_number(expected)
    act = coerce_number(actual)
    if exp is None or act is None:
        return False
    return math.isclose(exp, act, rel_tol=0.0, abs_tol=abs_tol)


answers_match = is_correct
