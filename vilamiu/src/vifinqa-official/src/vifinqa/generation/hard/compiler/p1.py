
from __future__ import annotations

from dataclasses import dataclass

from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.compiler.common import locate_cell
from vifinqa.generation.hard.numbers import VN_NUMBER_PARSER_SOURCE, parse_vn_number
from vifinqa.generation.hard.schemas import Comparison, HardPlan, MetricBinding

__all__ = ["P1CompileError", "P1CompileResult", "P1Compiler", "locate_cell"]


class P1CompileError(Exception):
    pass


def _compare(value: float, threshold: float, comparison: Comparison) -> bool:
    if comparison == "gt":
        return value > threshold
    if comparison == "gte":
        return value >= threshold
    if comparison == "lt":
        return value < threshold
    return value <= threshold


_COMPARATOR_TOKEN: dict[Comparison, str] = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


@dataclass(slots=True)
class P1CompileResult:
    pandas_query: str
    expected_answer: object
    values_by_ticker: dict[str, float]


class P1Compiler:

    def compile(
        self,
        plan: HardPlan,
        bindings: list[MetricBinding],
        tables: dict[str, TableAsset],
    ) -> P1CompileResult:
        if plan.threshold is None:
            raise P1CompileError("Plan has not locked its threshold.")
        draft = plan.draft

        values_by_ticker: dict[str, float] = {}
        cell_by_ticker: dict[str, tuple[int, int]] = {}
        for binding in bindings:
            table = tables.get(binding.table_ref)
            if table is None:
                raise P1CompileError(f"Missing TableAsset for {binding.table_ref}.")
            location = locate_cell(table, binding.row_label, binding.column_label)
            if location is None:
                raise P1CompileError(
                    f"Could not locate a cell for {binding.table_ref} "
                    f"(row={binding.row_label!r}, column={binding.column_label!r})."
                )
            cell_by_ticker[binding.ticker] = location
            row_idx, col_idx = location
            raw = table.rows[row_idx][col_idx]
            values_by_ticker[binding.ticker] = parse_vn_number(raw) * binding.scale

        threshold = plan.threshold
        passed = {
            ticker: value
            for ticker, value in values_by_ticker.items()
            if _compare(value, threshold, draft.comparison)
        }
        if draft.final_operation == "count":
            expected: object = len(passed)
        elif draft.final_operation == "sum":
            expected = sum(passed.values())
        else:
            expected = len(passed) > 0

        query = self._render_query(bindings, cell_by_ticker, threshold, draft.comparison, draft.final_operation)
        return P1CompileResult(pandas_query=query, expected_answer=expected, values_by_ticker=values_by_ticker)

    def _render_query(
        self,
        bindings: list[MetricBinding],
        cell_by_ticker: dict[str, tuple[int, int]],
        threshold: float,
        comparison: Comparison,
        final_operation: str,
    ) -> str:
        comparator = _COMPARATOR_TOKEN[comparison]
        lines = [VN_NUMBER_PARSER_SOURCE.rstrip("\n"), "", "values = {}"]
        for binding in bindings:
            row_idx, col_idx = cell_by_ticker[binding.ticker]
            lines.append(
                f"values[{binding.ticker!r}] = _parse_vn_number("
                f"dfs[{binding.table_ref!r}].iloc[{row_idx}, {col_idx}]) * {binding.scale!r}"
            )
        lines.append(f"threshold = {threshold!r}")
        lines.append(f"passed = {{k: v for k, v in values.items() if v {comparator} threshold}}")
        if final_operation == "count":
            lines.append("result = len(passed)")
        elif final_operation == "sum":
            lines.append("result = sum(passed.values())")
        else:
            lines.append("result = len(passed) > 0")
        return "\n".join(lines) + "\n"
