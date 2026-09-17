
from __future__ import annotations

import re

from vifinqa.generation.intermediate_formulas.base import FormulaScenarioPlan
from vifinqa.generation.intermediate_formulas.longitudinal import (
    REDUCER_EXPRESSIONS,
    TRANSFORM_EXPRESSIONS,
    LongitudinalScenarioPlan,
)
from vifinqa.generation.intermediate_formulas.registry import FORMULA_EXPRESSIONS

_VN_NUMBER_PARSER_SOURCE = '''\
def _parse_vn_number(raw):
    s = str(raw).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    s = s.replace(".", "").replace(",", ".")
    value = float(s)
    return -value if negative else value
'''

# Keep unit and scale handling explicit.
_PERCENTAGE_FORMULAS = frozenset({"roa", "roe"})


def compile_formula_query(plan: FormulaScenarioPlan) -> str:
    lines = [_VN_NUMBER_PARSER_SOURCE]
    for role_id, cell in plan.bindings.items():
        lines.append(
            f"{role_id} = _parse_vn_number(dfs[{cell.table_ref!r}].iloc[{cell.row_idx}, {cell.col_idx}]) * {cell.scale!r}"
        )
    expression = FORMULA_EXPRESSIONS[plan.formula_id]
    if plan.formula_id in _PERCENTAGE_FORMULAS:
        lines.append(f"result = round(({expression}) * 100, 2)")
    else:
        lines.append(f"result = round({expression}, 2)")
    return "\n".join(lines) + "\n"


_PERIOD_TOKEN_PATTERNS = {token: re.compile(rf"\b{token}\b") for token in ("x1", "x2", "x3")}


def compile_longitudinal_query(plan: LongitudinalScenarioPlan) -> str:
    lines = [_VN_NUMBER_PARSER_SOURCE]
    transform_expression = TRANSFORM_EXPRESSIONS[plan.per_entity_transform]
    entity_transform_vars: list[str] = []
    for entity in plan.entities:
        period_vars: list[str] = []
        for index, period in enumerate(plan.periods, start=1):
            cell = plan.bindings[(entity, period)]
            var_name = f"x{index}_{entity}"
            period_vars.append(var_name)
            lines.append(
                f"{var_name} = _parse_vn_number(dfs[{cell.table_ref!r}].iloc[{cell.row_idx}, {cell.col_idx}]) * {cell.scale!r}"
            )
        expr = transform_expression
        for generic_token, actual_var in zip(("x1", "x2", "x3"), period_vars, strict=True):
            expr = _PERIOD_TOKEN_PATTERNS[generic_token].sub(actual_var, expr)
        transform_var = f"t_{entity}"
        lines.append(f"{transform_var} = {expr}")
        entity_transform_vars.append(transform_var)

    lines.append(f"_values = [{', '.join(entity_transform_vars)}]")
    reducer_expression = REDUCER_EXPRESSIONS[plan.terminal_reducer]
    lines.append(f"result = round(({reducer_expression}) * 100, 2)")
    return "\n".join(lines) + "\n"
