
from __future__ import annotations

from dataclasses import dataclass

from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.compiler.common import locate_cell
from vifinqa.generation.hard.numbers import VN_NUMBER_PARSER_SOURCE, parse_vn_number
from vifinqa.generation.hard.schemas import HardP3Plan, MetricBinding


class P3CompileError(Exception):
    pass


@dataclass(slots=True)
class P3CompileResult:
    pandas_query: str
    expected_answer: object
    selected_period: str
    selector_values: dict[str, float]
    answer_values: dict[str, float]


class P3Compiler:

    def compile(
        self,
        plan: HardP3Plan,
        bindings: list[MetricBinding],
        tables: dict[str, TableAsset],
    ) -> P3CompileResult:
        draft = plan.draft
        # Keep period handling explicit and deterministic.
        selector_bindings = sorted(
            (b for b in bindings if b.metric_role == draft.selector_metric_role), key=lambda b: int(b.period)
        )
        answer_bindings = sorted(
            (b for b in bindings if b.metric_role == draft.answer_metric_role), key=lambda b: int(b.period)
        )
        if not selector_bindings or not answer_bindings:
            raise P3CompileError("Missing binding for selector_metric_role or answer_metric_role.")

        selector_cells, selector_values = self._resolve_values(selector_bindings, tables)
        answer_cells, answer_values = self._resolve_values(answer_bindings, tables)

        if draft.selector_operation == "argmax":
            selected_period = max(selector_values, key=selector_values.get)
        else:
            selected_period = min(selector_values, key=selector_values.get)
        if selected_period not in answer_values:
            raise P3CompileError(f"No answer binding for the selected period: {selected_period!r}.")
        expected = answer_values[selected_period]

        query = self._render_query(
            selector_bindings=selector_bindings,
            selector_cells=selector_cells,
            answer_bindings=answer_bindings,
            answer_cells=answer_cells,
            selector_operation=draft.selector_operation,
        )
        return P3CompileResult(
            pandas_query=query,
            expected_answer=expected,
            selected_period=selected_period,
            selector_values=selector_values,
            answer_values=answer_values,
        )

    def _resolve_values(
        self, bindings: list[MetricBinding], tables: dict[str, TableAsset]
    ) -> tuple[dict[str, tuple[int, int]], dict[str, float]]:
        cells: dict[str, tuple[int, int]] = {}
        values: dict[str, float] = {}
        for binding in bindings:
            table = tables.get(binding.table_ref)
            if table is None:
                raise P3CompileError(f"Missing TableAsset for {binding.table_ref}.")
            location = locate_cell(table, binding.row_label, binding.column_label)
            if location is None:
                raise P3CompileError(
                    f"Could not locate a cell for {binding.table_ref} "
                    f"(row={binding.row_label!r}, column={binding.column_label!r})."
                )
            cells[binding.period] = location
            row_idx, col_idx = location
            raw = table.rows[row_idx][col_idx]
            values[binding.period] = parse_vn_number(raw) * binding.scale
        return cells, values

    def _render_query(
        self,
        *,
        selector_bindings: list[MetricBinding],
        selector_cells: dict[str, tuple[int, int]],
        answer_bindings: list[MetricBinding],
        answer_cells: dict[str, tuple[int, int]],
        selector_operation: str,
    ) -> str:
        picker = "max" if selector_operation == "argmax" else "min"
        lines = [VN_NUMBER_PARSER_SOURCE.rstrip("\n"), "", "selector_values = {}"]
        for binding in selector_bindings:
            row_idx, col_idx = selector_cells[binding.period]
            lines.append(
                f"selector_values[{binding.period!r}] = _parse_vn_number("
                f"dfs[{binding.table_ref!r}].iloc[{row_idx}, {col_idx}]) * {binding.scale!r}"
            )
        lines.append(f"selected_period = {picker}(selector_values, key=selector_values.get)")
        lines.append("")
        lines.append("answer_values = {}")
        for binding in answer_bindings:
            row_idx, col_idx = answer_cells[binding.period]
            lines.append(
                f"answer_values[{binding.period!r}] = _parse_vn_number("
                f"dfs[{binding.table_ref!r}].iloc[{row_idx}, {col_idx}]) * {binding.scale!r}"
            )
        lines.append("result = answer_values[selected_period]")
        return "\n".join(lines) + "\n"
