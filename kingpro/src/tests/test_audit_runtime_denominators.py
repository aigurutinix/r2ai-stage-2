from __future__ import annotations

import ast

import pandas as pd

from scripts.audit_runtime_denominators import denominator_stats, instrument


def _execute(code: str, namespace: dict) -> list[dict]:
    seen: list[dict] = []

    def recorder(value, line, expression):
        seen.append(denominator_stats(value, line=line, expression=expression))
        return value

    namespace = {**namespace, "_audit_denominator": recorder}
    exec(instrument(code), namespace)
    return seen


def test_instruments_executed_scalar_denominator_once() -> None:
    calls = {"count": 0}

    def denominator() -> int:
        calls["count"] += 1
        return 2

    seen = _execute("result = 10 / denominator()", {"denominator": denominator})
    assert calls["count"] == 1
    assert seen[0]["expression"] == "denominator()"
    assert seen[0]["zero_count"] == 0


def test_records_zero_and_nan_in_series_denominator() -> None:
    seen = _execute(
        "result = numerator / denominator",
        {
            "numerator": pd.Series([1.0, 2.0, 3.0]),
            "denominator": pd.Series([2.0, 0.0, float('nan')]),
        },
    )
    assert seen[0]["kind"] == "series"
    assert seen[0]["zero_count"] == 1
    assert seen[0]["nan_count"] == 1


def test_dead_branch_denominator_is_not_recorded() -> None:
    seen = _execute("if False:\n    result = 1 / 0\nresult = 1", {})
    assert seen == []


def test_nested_divisions_keep_both_observations() -> None:
    seen = _execute("result = (12 / 3) / 2", {})
    assert len(seen) == 2
    assert [item["minimum_finite"] for item in seen] == [3.0, 2.0]


def test_empty_dataframe_denominator_is_reported() -> None:
    stats = denominator_stats(pd.DataFrame(), line=1, expression="df")
    assert stats["empty"] is True
    assert stats["value_count"] == 0
