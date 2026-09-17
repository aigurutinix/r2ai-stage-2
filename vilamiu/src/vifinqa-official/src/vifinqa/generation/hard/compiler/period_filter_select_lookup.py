"""Compiler walking skeleton: filter period -> select period -> lookup scalar."""

from __future__ import annotations

from dataclasses import dataclass

from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.compiler.common import locate_cell
from vifinqa.generation.hard.depth3_schemas import PeriodFilterSelectLookupDraft, ResolvedMetricExpression
from vifinqa.generation.hard.keyed import KeyedOperationError, Series, filter_keys, lookup, select_key
from vifinqa.generation.hard.numbers import VN_NUMBER_PARSER_SOURCE, parse_vn_number
from vifinqa.generation.hard.schemas import MetricBinding


class PeriodFilterSelectLookupCompileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PeriodFilterSelectLookupCompileResult:
    pandas_query: str
    expected_answer: int | float | bool
    allowed_periods: frozenset[str]
    selected_period: str
    role_values: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class _LocatedExpression:
    operation: str
    cells: tuple[tuple[MetricBinding, int, int], ...]


class PeriodFilterSelectLookupCompiler:
    def compile(
        self,
        plan: PeriodFilterSelectLookupDraft,
        expressions: list[ResolvedMetricExpression],
        tables: dict[str, TableAsset],
    ) -> PeriodFilterSelectLookupCompileResult:
        roles = (plan.filter_role, plan.selector_role, plan.answer_role)
        values_by_role: dict[str, dict[str, float]] = {}
        cells_by_role: dict[str, dict[str, _LocatedExpression]] = {}
        for role in roles:
            role_expressions = sorted(
                (expression for expression in expressions if expression.metric_role == role.metric_role),
                key=lambda expression: int(expression.period),
            )
            if not role_expressions:
                raise PeriodFilterSelectLookupCompileError(f"Missing binding for role {role.metric_role!r}")
            values, cells = self._resolve_role(role_expressions, tables)
            values_by_role[role.metric_role] = values
            cells_by_role[role.metric_role] = cells

        periods = set(values_by_role[plan.filter_role.metric_role])
        for role in roles[1:]:
            if set(values_by_role[role.metric_role]) != periods:
                raise PeriodFilterSelectLookupCompileError("The three roles do not cover the same period set")

        try:
            allowed = filter_keys(
                Series(values_by_role[plan.filter_role.metric_role]),
                comparison=plan.filter_comparison,
                threshold=plan.filter_threshold,
            )
            selected = select_key(
                Series(values_by_role[plan.selector_role.metric_role]),
                allowed,
                operation=plan.selector_operation,
            )
            expected = lookup(Series(values_by_role[plan.answer_role.metric_role]), selected)
        except KeyedOperationError as exc:
            raise PeriodFilterSelectLookupCompileError(str(exc)) from exc

        query = self._render_query(plan, cells_by_role)
        return PeriodFilterSelectLookupCompileResult(
            pandas_query=query,
            expected_answer=expected,
            allowed_periods=allowed.keys,
            selected_period=selected.key,
            role_values=values_by_role,
        )

    @staticmethod
    def _resolve_role(
        expressions: list[ResolvedMetricExpression], tables: dict[str, TableAsset]
    ) -> tuple[dict[str, float], dict[str, _LocatedExpression]]:
        values: dict[str, float] = {}
        cells: dict[str, _LocatedExpression] = {}
        for expression in expressions:
            if expression.period in values:
                raise PeriodFilterSelectLookupCompileError(
                    f"Role {expression.metric_role!r} has multiple expressions in period {expression.period!r}"
                )
            located: list[tuple[MetricBinding, int, int]] = []
            cell_values: list[float] = []
            for binding in expression.cells:
                table = tables.get(binding.table_ref)
                if table is None:
                    raise PeriodFilterSelectLookupCompileError(f"Missing TableAsset for {binding.table_ref}")
                location = locate_cell(table, binding.row_label, binding.column_label)
                if location is None:
                    raise PeriodFilterSelectLookupCompileError(
                        f"Could not locate a cell {binding.table_ref}: {binding.row_label!r}/{binding.column_label!r}"
                    )
                row_idx, col_idx = location
                located.append((binding, row_idx, col_idx))
                cell_values.append(parse_vn_number(table.rows[row_idx][col_idx]) * binding.scale)
            value = cell_values[0] if expression.operation == "direct" else sum(cell_values)
            values[expression.period] = value
            cells[expression.period] = _LocatedExpression(expression.operation, tuple(located))
        return values, cells

    @staticmethod
    def _render_query(
        plan: PeriodFilterSelectLookupDraft,
        cells_by_role: dict[str, dict[str, _LocatedExpression]],
    ) -> str:
        variable_names = {
            plan.filter_role.metric_role: "filter_values",
            plan.selector_role.metric_role: "selector_values",
            plan.answer_role.metric_role: "answer_values",
        }
        lines = [VN_NUMBER_PARSER_SOURCE.rstrip("\n"), ""]
        for role in (plan.filter_role, plan.selector_role, plan.answer_role):
            variable = variable_names[role.metric_role]
            lines.append(f"{variable} = {{}}")
            for period, expression in sorted(
                cells_by_role[role.metric_role].items(), key=lambda item: int(item[0])
            ):
                terms = [
                    f"_parse_vn_number(dfs[{binding.table_ref!r}].iloc[{row_idx}, {col_idx}]) * {binding.scale!r}"
                    for binding, row_idx, col_idx in expression.cells
                ]
                rhs = terms[0] if expression.operation == "direct" else "sum([" + ", ".join(terms) + "])"
                lines.append(f"{variable}[{period!r}] = {rhs}")
            lines.append("")

        operator = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[plan.filter_comparison]
        picker = "max" if plan.selector_operation == "argmax" else "min"
        lines.extend(
            [
                "allowed_periods = {period for period, value in filter_values.items() "
                f"if value {operator} {plan.filter_threshold!r}}}",
                f"best_value = {picker}(selector_values[period] for period in allowed_periods)",
                "winner_periods = sorted(period for period in allowed_periods "
                "if selector_values[period] == best_value)",
                "selected_period = winner_periods[0]",
                "result = answer_values[selected_period]",
            ]
        )
        return "\n".join(lines) + "\n"
