from __future__ import annotations

import pandas as pd

from scripts.audit_runtime_extrema_ties import _iterable_tie, _series_tie, instrument


def test_series_idxmax_tie_preserves_labels() -> None:
    series = pd.Series([4.0, 9.0, 9.0], index=[2020, 2021, 2022])
    tie = _series_tie(series, "idxmax", series.idxmax())
    assert tie is not None
    assert tie["tie_count"] == 2
    assert tie["tied_labels"] == [2021, 2022]
    assert tie["returned"] == 2021


def test_series_unique_extreme_is_not_a_finding() -> None:
    series = pd.Series([4.0, 9.0, 8.0])
    assert _series_tie(series, "max", series.max()) is None


def test_builtin_tie_reports_positions() -> None:
    tie = _iterable_tie(([3, 7, 7],), "max", 7)
    assert tie is not None
    assert tie["tied_labels"] == [1, 2]


def test_parser_helper_body_is_not_instrumented() -> None:
    code = "def parse(x):\n    return min(x, 2)\nresult = max([1, 2])"
    compiled = instrument(code)
    assert compiled is not None


def test_instrumented_method_executes_once_and_records_tie() -> None:
    seen = []

    def method_extreme(obj, method, line, expression, *args, **kwargs):
        result = getattr(obj, method)(*args, **kwargs)
        seen.append((method, expression, result))
        return result

    namespace = {
        "series": pd.Series([5, 5], index=[2019, 2020]),
        "_audit_method_extreme": method_extreme,
    }
    exec(instrument("result = series.idxmax()"), namespace)
    assert namespace["result"] == 2019
    assert seen == [("idxmax", "series.idxmax()", 2019)]
