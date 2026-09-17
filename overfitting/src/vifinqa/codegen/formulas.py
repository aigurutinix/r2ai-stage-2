"""Formula registry: how to combine facts once they are located.

Each entry provides (a) a Vietnamese-facing description injected into the prompt
so the model computes the intended quantity, and (b) a deterministic python
implementation used by the rule path when every operand is known.

Unit convention (ORGANIZER-CONFIRMED): the emitted answer is already in the unit
the question asks. Percent-valued operations therefore multiply by 100 here —
`growth_pct` on 100 -> 120 returns 20.0, not 0.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Formula:
    name: str
    arity: str                 # "2" | "n"
    output_type: str           # number | percent | percentage_point | ratio | count | year
    describe: str              # prompt text (Vietnamese question -> English hint)
    fn: Callable | None = None


def _growth_pct(end: float, base: float) -> float:
    if base == 0:
        raise ZeroDivisionError("growth base is zero")
    return (end - base) / abs(base) * 100.0


def _difference(a: float, b: float) -> float:
    return a - b


def _ratio_pct(num: float, den: float) -> float:
    if den == 0:
        raise ZeroDivisionError("ratio denominator is zero")
    return num / den * 100.0


def _ratio_times(num: float, den: float) -> float:
    if den == 0:
        raise ZeroDivisionError("ratio denominator is zero")
    return num / den


def _cagr_pct(end: float, base: float, years: float) -> float:
    if base <= 0 or years <= 0:
        raise ValueError("CAGR needs a positive base and horizon")
    return ((end / base) ** (1.0 / years) - 1.0) * 100.0


def _average(values) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("average of an empty set")
    return sum(vals) / len(vals)


REGISTRY: dict[str, Formula] = {
    "lookup": Formula(
        "lookup", "1", "number",
        "Return the single located figure, converted to the requested unit."),
    "growth_pct": Formula(
        "growth_pct", "2", "percent",
        "Growth in PERCENT: (end - base) / abs(base) * 100. "
        "'base' is the earlier period, 'end' the later one. Answer 20 for +20%.",
        _growth_pct),
    "difference": Formula(
        "difference", "2", "number",
        "Difference A - B in the requested unit. Keep the sign the question "
        "implies ('A cao hơn B bao nhiêu' -> A - B).", _difference),
    "ratio": Formula(
        "ratio", "2", "percent",
        "Share in PERCENT: numerator / denominator * 100. Answer 90 for 90%.",
        _ratio_pct),
    "margin": Formula(
        "margin", "2", "percent",
        "Margin/ROE/ROA/ROS in PERCENT: profit / base * 100.", _ratio_pct),
    "ratio_times": Formula(
        "ratio_times", "2", "ratio",
        "Ratio expressed in times ('lần') or turns ('vòng'): a / b, NOT x100.",
        _ratio_times),
    "cagr": Formula(
        "cagr", "2", "percent",
        "CAGR in PERCENT: ((end/base)**(1/n_years) - 1) * 100.", _cagr_pct),
    "average": Formula(
        "average", "n", "number",
        "Arithmetic mean of the located figures, in the requested unit.",
        _average),
    "sum": Formula(
        "sum", "n", "number",
        "Sum of the located figures, in the requested unit.", sum),
    "ranking": Formula(
        "ranking", "n", "number",
        "Rank the entities by the metric and return the value asked for "
        "(largest/smallest/2nd...). If the question asks WHICH company, return "
        "the numeric figure of that company."),
    "count": Formula(
        "count", "n", "count",
        "Count how many entities satisfy the stated condition; return an integer."),
    "percentage_point": Formula(
        "percentage_point", "2", "percentage_point",
        "Difference between two PERCENTAGES, in percentage points: p1 - p2."),
    "hypothetical": Formula(
        "hypothetical", "n", "number",
        "Apply the stated hypothetical adjustment ('giả sử ...') to the located "
        "figures, then answer in the requested unit."),
}


def get(op: str) -> Formula:
    return REGISTRY.get(op, REGISTRY["lookup"])


def describe_for_prompt(op: str, n_facts: int = 1) -> str:
    f = get(op)
    head = f"OPERATION: {f.name} (expected output type: {f.output_type})"
    body = f.describe
    tail = ""
    if f.arity == "2" and n_facts >= 2:
        tail = (" You must locate BOTH operands — they usually live in different "
                "tables/reports (different company or different year).")
    elif f.arity == "n" and n_facts >= 2:
        tail = (f" You must locate the figure for EACH of the {n_facts} "
                f"entity/period combinations before combining.")
    return f"{head}\n{body}{tail}"
