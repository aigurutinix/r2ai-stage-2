"""A small plan schema the model fills in, and a compiler that executes it.

The model's first attempt at this pool wrote free-form pandas and 9 of 12
programs died on a style rule (`portability_problems` rejects comprehensions,
which py37's `exec` scoping makes unsafe). Nothing about the reasoning was wrong;
the format was.

So the split moves one notch further. The model does the part our regexes are
brittle at — reading which phrase is the filter, which is the answer, what
operation, over which axis — and returns a few fields. Everything after that is
ours: selection, arithmetic, unit conversion, and the emitted pandas. That makes
malformed syntax, hard-coded constants and py37 breakage impossible by
construction rather than by inspection.

The schema is also the label a synthetic training set would carry: every field
here is known by construction when a question is generated from the corpus, so
the same JSON serves as prompt target for fine-tuning later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Everything the dead zone asks for. `screen` is the two-stage shape — rank the
# axis by one metric, report another from the winner — which is the single
# largest unserved family.
OPS = frozenset({
    "value", "diff", "growth", "sum", "avg", "max", "min", "ratio", "screen",
    "count",
})
AXES = frozenset({"nam", "ma"})
TAKES = frozenset({"max", "min", "pos", "neg"})

UNIT_DIVISOR = {
    "dong": 1.0, "nghin": 1e3, "trieu": 1e6, "ty": 1e9,
    "tram_ty": 1e11, "nghin_ty": 1e12,
}
RATIO_UNITS = frozenset({"phan_tram", "lan", "vong"})


@dataclass(frozen=True, slots=True)
class Plan:
    op: str
    metric: str
    axis: str = "nam"
    denominator: str = ""
    filter_metric: str = ""
    filter_take: str = "max"
    keys: tuple[str, ...] = ()


def parse(reply: str, metrics: list[str] | None = None) -> Plan | None:
    """The plan the model returned, or None if it is not usable.

    When `metrics` is given, the names are snapped onto that vocabulary. It is
    omitted on the first pass: the model names the metrics in its own words and
    the label matcher resolves them afterwards, which is what lets a screen ranked
    by a phrase no regex extracted still find its rows.
    """

    text = reply.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None

    op = str(raw.get("op", "")).strip().lower()
    if op not in OPS:
        return None
    filter_metric = _snap(raw.get("filter_metric", ""), metrics) or ""
    denominator = _snap(raw.get("denominator", ""), metrics) or ""
    metric = _snap(raw.get("metric", ""), metrics)
    if not metric:
        # count may only name the filter; screen-ratio may name num/den only.
        if op == "count" and filter_metric:
            metric = filter_metric
        else:
            return None
    axis = str(raw.get("axis", "nam")).strip().lower()
    if axis not in AXES:
        axis = "nam"
    take = str(raw.get("filter_take", "max")).strip().lower()
    if take not in TAKES:
        take = "max"

    if op == "ratio" and not denominator:
        return None
    if op == "screen" and not filter_metric:
        return None
    if op == "count" and not filter_metric:
        return None

    keys = raw.get("keys") or []
    if not isinstance(keys, list):
        keys = []
    return Plan(
        op=op, metric=metric, axis=axis, denominator=denominator,
        filter_metric=filter_metric, filter_take=take,
        keys=tuple(str(k) for k in keys),
    )


def _snap(value, metrics: list[str] | None) -> str | None:
    """Map a model-written phrase onto a phrase the panel actually contains."""

    text = str(value or "").strip(" ,.;:")
    if not text:
        return None
    if metrics is None:
        return text
    if text in metrics:
        return text
    folded = text.casefold()
    for candidate in metrics:
        if candidate.casefold() == folded:
            return candidate
    # Longest containment either way — the model shortens as often as it pads.
    best, best_len = None, 0
    for candidate in metrics:
        low = candidate.casefold()
        if (low in folded or folded in low) and len(low) > best_len:
            best, best_len = candidate, len(low)
    return best


def _rows_for(rows, metric, axis_key=None, axis=None):
    picked = []
    for row in rows:
        if row[2] != metric:
            continue
        if axis_key is not None and str(row[0 if axis == "ma" else 1]) != str(axis_key):
            continue
        picked.append(row)
    return picked


def _raw_series(rows, metric, axis):
    column = 0 if axis == "ma" else 1
    out: dict[str, float] = {}
    for row in rows:
        if row[2] != metric:
            continue
        key = str(row[column])
        try:
            value = abs(float(row[4]))
        except (TypeError, ValueError):
            continue
        # First wins: the panel is built best-match-first per (company, year).
        out.setdefault(key, value)
    return out


def _raw_series_signed(rows, metric, axis):
    column = 0 if axis == "ma" else 1
    out: dict[str, float] = {}
    for row in rows:
        if row[2] != metric:
            continue
        key = str(row[column])
        try:
            value = float(row[4])
        except (TypeError, ValueError):
            continue
        out.setdefault(key, value)
    return out


def _series(rows, metric, axis):
    """`{axis key: amount}` for one metric, computing it if it is a named ratio.

    Screens are usually ranked by something that is not a line item at all —
    "biên lợi nhuận gộp", "hệ số thanh toán nhanh", ROA. Those can never resolve
    against a row label, which is why the first pass left ten plans unresolved
    even though the model had read every one of them correctly. `ratio.FORMULAS`
    already knows how to rewrite them into two line items, so the ranking series
    is computed from those instead of looked up.
    """

    direct = _raw_series(rows, metric, axis)
    if direct:
        return direct

    from vifin.answering.ratio import FORMULAS

    for pattern, numerator, denominator in FORMULAS:
        if not pattern.search(metric):
            continue
        top = _raw_series(rows, numerator, axis)
        bottom = _raw_series(rows, denominator, axis)
        computed = {}
        for key, value in top.items():
            divisor = bottom.get(key)
            if divisor:
                computed[key] = value / divisor
        if computed:
            return computed
    return {}


def execute(plan: Plan, panel_rows, target_unit: str) -> tuple[float, list[str]] | None:
    """The answer, plus the panel keys the program has to read to get it."""

    rows = panel_rows[1:]
    if plan.op == "count":
        ranking = _raw_series_signed(rows, plan.filter_metric, plan.axis)
        if not ranking:
            return None
        if plan.filter_take == "pos":
            hits = [k for k, v in ranking.items() if v > 0]
        elif plan.filter_take == "neg":
            hits = [k for k, v in ranking.items() if v < 0]
        else:
            return None
        if plan.metric and plan.denominator and plan.metric != plan.filter_metric:
            top = _raw_series_signed(rows, plan.metric, plan.axis)
            bot = _raw_series_signed(rows, plan.denominator, plan.axis)
            hits = [k for k in hits if k in top and k in bot and top[k] < bot[k]]
        return float(len(hits)), hits

    series = _series(rows, plan.metric, plan.axis)
    if plan.keys:
        wanted = {str(k) for k in plan.keys}
        narrowed = {k: v for k, v in series.items() if k in wanted}
        series = narrowed or series
    if not series:
        return None

    order = sorted(series)
    if plan.op == "screen":
        ranking = _series(rows, plan.filter_metric, plan.axis)
        shared = [k for k in order if k in ranking]
        if not shared:
            return None
        winner = (max(shared, key=lambda k: ranking[k]) if plan.filter_take == "max"
                  else min(shared, key=lambda k: ranking[k]))
        amount, used = series[winner], [winner]
        # Optional: asked figure is a ratio at the winner (biên gộp, ROE…).
        if plan.denominator:
            bottom = _series(rows, plan.denominator, plan.axis)
            if winner not in bottom or not bottom[winner]:
                return None
            amount = series[winner] / bottom[winner]
    elif plan.op == "ratio":
        bottom = _series(rows, plan.denominator, plan.axis)
        shared = [k for k in order if k in bottom and bottom[k]]
        if not shared:
            return None
        key = shared[-1]
        amount, used = series[key] / bottom[key], [key]
    elif plan.op in ("diff", "growth"):
        if len(order) < 2:
            return None
        low, high = order[0], order[-1]
        if plan.op == "diff":
            amount = abs(series[high] - series[low])
        else:
            if not series[low]:
                return None
            amount = (series[high] - series[low]) / abs(series[low]) * 100.0
        used = [low, high]
    elif plan.op == "value":
        key = order[-1]
        amount, used = series[key], [key]
    elif plan.op == "sum":
        amount, used = sum(series.values()), order
    elif plan.op == "avg":
        amount, used = sum(series.values()) / len(series), order
    elif plan.op == "max":
        key = max(order, key=lambda k: series[k])
        amount, used = series[key], [key]
    else:  # min
        key = min(order, key=lambda k: series[k])
        amount, used = series[key], [key]

    if plan.op == "growth":
        result = amount
    elif plan.op == "ratio" or (plan.op == "screen" and plan.denominator):
        result = amount * (100.0 if target_unit == "phan_tram" else 1.0)
    elif target_unit in RATIO_UNITS:
        # A đồng amount cannot answer "bao nhiêu phần trăm"; refuse rather than
        # report a currency figure as a rate, the error this whole pool is full of.
        return None
    else:
        result = amount / UNIT_DIVISOR.get(target_unit, 1.0)
    return round(result, 2), used


def compile_query(plan: Plan, target_unit: str) -> str:
    """Pandas over the shipped panel CSV, written by us rather than the model.

    Deliberately plain: no comprehensions, no lambdas, no imports, and every
    figure read from the frame. That is what makes the program both py37-safe and
    defensible on the organisers' manual review.
    """

    axis_col = "ma" if plan.axis == "ma" else "nam"
    divisor = 1.0 if plan.op in ("growth", "ratio", "count") or (
        plan.op == "screen" and plan.denominator) else UNIT_DIVISOR.get(target_unit, 1.0)
    lines = [
        "def series(frame, metric):",
        "    out = {}",
        f"    sub = frame[frame['chi_tieu'] == metric]",
        "    for i in range(len(sub)):",
        f"        key = str(sub.iloc[i]['{axis_col}'])",
        "        if key in out:",
        "            continue",
        "        out[key] = abs(float(sub.iloc[i]['gia_tri_dong']))",
        "    return out",
        "",
    ]

    if plan.op == "count":
        # Sign filters need the raw (signed) series, not abs.
        lines = [
            "def series_signed(frame, metric):",
            "    out = {}",
            "    sub = frame[frame['chi_tieu'] == metric]",
            "    for i in range(len(sub)):",
            f"        key = str(sub.iloc[i]['{axis_col}'])",
            "        if key in out:",
            "            continue",
            "        out[key] = float(sub.iloc[i]['gia_tri_dong'])",
            "    return out",
            "",
            f"rank = series_signed(df, {plan.filter_metric!r})",
            "hits = []",
            "for k in sorted(rank):",
            ("    if rank[k] > 0:" if plan.filter_take == "pos"
             else "    if rank[k] < 0:"),
            "        hits.append(k)",
        ]
        if plan.metric and plan.denominator and plan.metric != plan.filter_metric:
            lines += [
                f"top = series_signed(df, {plan.metric!r})",
                f"bot = series_signed(df, {plan.denominator!r})",
                "kept = []",
                "for k in hits:",
                "    if k in top and k in bot and top[k] < bot[k]:",
                "        kept.append(k)",
                "hits = kept",
            ]
        lines += ["result = float(len(hits))"]
        return "\n".join(lines)

    lines += [
        f"vals = series(df, {plan.metric!r})",
        "keys = sorted(vals)",
    ]

    if plan.op == "screen":
        lines += [
            f"rank = series(df, {plan.filter_metric!r})",
            "shared = []",
            "for k in keys:",
            "    if k in rank:",
            "        shared.append(k)",
            "win = shared[0]",
            "for k in shared:",
            f"    if rank[k] {'>' if plan.filter_take == 'max' else '<'} rank[win]:",
            "        win = k",
            "amount = vals[win]",
        ]
        if plan.denominator:
            lines += [
                f"bottom = series(df, {plan.denominator!r})",
                "amount = amount / bottom[win]",
            ]
    elif plan.op == "ratio":
        lines += [
            f"bottom = series(df, {plan.denominator!r})",
            "shared = []",
            "for k in keys:",
            "    if k in bottom and bottom[k] != 0:",
            "        shared.append(k)",
            "k = shared[-1]",
            "amount = vals[k] / bottom[k]",
        ]
    elif plan.op == "diff":
        lines.append("amount = abs(vals[keys[-1]] - vals[keys[0]])")
    elif plan.op == "growth":
        lines.append("amount = (vals[keys[-1]] - vals[keys[0]]) / abs(vals[keys[0]]) * 100.0")
    elif plan.op == "value":
        lines.append("amount = vals[keys[-1]]")
    elif plan.op == "sum":
        lines += ["amount = 0.0", "for k in keys:", "    amount = amount + vals[k]"]
    elif plan.op == "avg":
        lines += ["amount = 0.0", "for k in keys:", "    amount = amount + vals[k]",
                  "amount = amount / len(keys)"]
    else:
        comparison = ">" if plan.op == "max" else "<"
        lines += ["win = keys[0]", "for k in keys:",
                  f"    if vals[k] {comparison} vals[win]:", "        win = k",
                  "amount = vals[win]"]

    if plan.op == "ratio" and target_unit == "phan_tram":
        lines.append("amount = amount * 100.0")
    elif plan.op == "screen" and plan.denominator and target_unit == "phan_tram":
        lines.append("amount = amount * 100.0")
    lines.append(f"result = round(amount / {divisor!r}, 2)")
    return "\n".join(lines)
