"""Answer-unit normalisation.

CONFIRMED BY THE ORGANIZERS: `answer` must be expressed in the unit the QUESTION
asks for. For "... là bao nhiêu phần trăm?" the correct answer is 90, not 0.9.

Two distinct cases must not be confused:

  A) the source cell already holds a percentage    -> value 90   , keep 90
  B) the source cell holds a ratio (0..1)          -> value 0.9  , emit 90

We cannot tell A from B from the number alone (0.9% is a legal percentage), so
the decision uses the SOURCE CELL's context:
  - label/col_name containing '%' or 'ty le'/'phan tram' -> already a percentage
  - a value > 1.5 for a percent question                 -> already a percentage
  - otherwise a bare ratio in [0, 1.5]                   -> multiply by 100

`check_answer_unit` never rewrites silently at submission time; it reports so a
human can audit. `percent_from_cell` is used by the deterministic rule path
where the provenance of the number is known exactly.
"""
from __future__ import annotations

import re

from ..utils.viet_text import norm

_PCT_HINT = re.compile(r"%|\bty le\b|\bphan tram\b|\bti le\b|\bty trong\b")

# output_type -> (low, high) plausible magnitude for a CORRECT answer
PLAUSIBLE = {
    # Returns, margins and growth rates can legitimately be negative.
    "percent": (-1000.0, 1000.0),
    "percentage_point": (-500.0, 500.0),
    "ratio": (-1000.0, 1000.0),
    "year": (1990.0, 2035.0),
    "count": (0.0, 1e6),
}


def cell_is_already_percent(label: str, col_name: str, value: float) -> bool:
    """Is the stored number ALREADY expressed in percent units?

    Careful: a label like "Tỷ lệ sở hữu" says the row is a RATIO CONCEPT, it
    says nothing about the stored scale — the cell may hold 0.9 or 90. Only an
    explicit '%' marker or the magnitude itself is evidence about the scale.
    """
    ctx = f"{label} {col_name}"
    if "%" in ctx:
        return True
    return abs(value) > 1.5


def percent_from_cell(value: float, label: str = "", col_name: str = "") -> float:
    """Return the value expressed in PERCENT units (90.0, not 0.9)."""
    if cell_is_already_percent(label, col_name, value):
        return float(value)
    return float(value) * 100.0


def check_answer_unit(answer: float, output_type: str) -> str | None:
    """Return a human-readable warning when the magnitude looks wrong."""
    try:
        a = float(answer)
    except (TypeError, ValueError):
        return "answer is not a float"
    if a != a:
        return "answer is NaN"
    rng = PLAUSIBLE.get(output_type)
    if rng and not (rng[0] <= a <= rng[1]):
        return f"{output_type} answer {a:g} outside plausible range {rng}"
    if output_type == "percent" and 0 < abs(a) <= 1.0:
        return (f"percent answer {a:g} looks like a RATIO — the organizers "
                f"expect 90 for 90%, not 0.9")
    if output_type == "year" and a != int(a):
        return f"year answer {a:g} is not an integer"
    return None


def audit_entries(entries) -> dict:
    """Count unit-suspicious answers in a built submission (id -> warning)."""
    warnings = {}
    for e in entries:
        ot = e.get("output_type") or ""
        if not ot:
            continue
        w = check_answer_unit(e.get("answer", 0.0), ot)
        if w:
            warnings[e["id"]] = w
    return warnings
