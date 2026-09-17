from __future__ import annotations

from scripts.audit_rounding_order_sensitivity import defer_intermediate_rounding


def _execute(code: str) -> tuple[float, list[dict]]:
    compiled, stripped = defer_intermediate_rounding(code)
    namespace: dict = {}
    exec(compiled, namespace)
    return float(namespace["result"]), stripped


def test_preserves_outermost_result_round() -> None:
    result, stripped = _execute("result = round(1 / 3, 2)")
    assert result == 0.33
    assert stripped == []


def test_removes_round_from_intermediate_assignment() -> None:
    result, stripped = _execute("x = round(1.234, 2)\nresult = round(x * 3, 2)")
    assert result == 3.7
    assert len(stripped) == 1
    assert stripped[0]["expression"] == "round(1.234, 2)"


def test_removes_nested_round_but_keeps_final_round() -> None:
    result, stripped = _execute("result = round(round(1.236, 2) * 2, 2)")
    assert result == 2.47
    assert len(stripped) == 1


def test_does_not_rewrite_parser_helper() -> None:
    result, stripped = _execute(
        "def parse(x):\n    return round(x, 1)\nx = parse(1.26)\nresult = round(x, 2)"
    )
    assert result == 1.3
    assert stripped == []


def test_does_not_rewrite_recall_guard() -> None:
    result, stripped = _execute(
        "result = round(1.234, 2)\nif round(result, 2) != 1.23:\n    result = 99"
    )
    assert result == 1.23
    assert stripped == []


def test_preserves_conditional_final_round() -> None:
    result, stripped = _execute("x = 1.236\nresult = round(x, 2) if x is not None else None")
    assert result == 1.24
    assert stripped == []
