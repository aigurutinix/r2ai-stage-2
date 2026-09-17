
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.table import TableAsset
from vifinqa.common.filtering.table_filters import is_numeric_cell
from vifinqa.generation.intermediate_formulas.base import (
    FormulaDefinition,
    FormulaScenarioPlan,
    ResolvedCell,
)
from vifinqa.generation.intermediate_formulas.longitudinal import (
    LongitudinalScenarioPlan,
    transform_missing_period_tokens,
)
from vifinqa.generation.intermediate_formulas.registry import formula_excludes_company
from vifinqa.generation.schemas import FormulaRoleCellMapping

_DENOMINATOR_ROLES: dict[str, tuple[str, ...]] = {
    "roa": ("total_assets_begin", "total_assets_end"),
    "roe": ("equity_begin", "equity_end"),
    "quick_ratio": ("current_liabilities",),
}
_NON_NEGATIVE_ROLES: dict[str, tuple[str, ...]] = {
    "quick_ratio": ("current_assets", "inventory", "current_liabilities"),
}


# Keep LLM behavior within the declared contract.
# Keep unit and scale handling explicit.
_SCALE_TOKEN_RE = {
    1_000_000_000.0: re.compile(r"\btỷ\b", re.IGNORECASE),
    1_000_000.0: re.compile(r"\btriệu\b", re.IGNORECASE),
    1_000.0: re.compile(r"\bnghìn\b", re.IGNORECASE),
}


def detect_scale_multiplier(*texts: str) -> float:
    combined = " ".join(texts)
    for multiplier, pattern in _SCALE_TOKEN_RE.items():
        if pattern.search(combined):
            return multiplier
    return 1.0


def resolve_role_cell(
    mapping: FormulaRoleCellMapping, table: TableAsset, *, unit_context: str = ""
) -> ResolvedCell | str:
    if not mapping.found:
        return f"Role {mapping.role_id}: LLM reported no match ({mapping.reason})."
    row_label = mapping.row_label.strip()
    column_label = mapping.column_label.strip()
    if not row_label or not column_label:
        return f"Role {mapping.role_id}: missing row_label/column_label."

    col_indices = [i for i, header in enumerate(table.header) if header.strip() == column_label]
    if not col_indices:
        return f"Role {mapping.role_id}: column_label {column_label!r} does not match the actual header."
    if len(col_indices) > 1:
        return f"Role {mapping.role_id}: column_label {column_label!r} matches multiple columns and is ambiguous."
    col_idx = col_indices[0]

    row_indices = [
        i
        for i, row in enumerate(table.rows)
        if any(cell.strip() == row_label for cell in row if not is_numeric_cell(cell))
    ]
    if not row_indices:
        return f"Role {mapping.role_id}: row_label {row_label!r} does not match any row in the actual table."
    if len(row_indices) > 1:
        return f"Role {mapping.role_id}: row_label {row_label!r} matches multiple rows and is ambiguous."
    row_idx = row_indices[0]

    row = table.rows[row_idx]
    if col_idx >= len(row):
        return f"Role {mapping.role_id}: column_label {column_label!r} exceeds the selected row's column count."
    raw_value = row[col_idx]
    if not is_numeric_cell(raw_value):
        return (
            f"Role {mapping.role_id}: cell at (row={row_label!r}, col={column_label!r}) "
            f"is not a valid number: {raw_value!r}."
        )

    scale = detect_scale_multiplier(column_label, unit_context)

    return ResolvedCell(
        table_ref=mapping.table_ref,
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=row_label,
        column_label=column_label,
        raw_value=raw_value,
        scale=scale,
    )


def formula_role_coverage_error(
    formula: FormulaDefinition, bindings: Mapping[str, ResolvedCell]
) -> str:
    required_ids = set(formula.role_ids())
    missing = [role_id for role_id in required_ids if role_id not in bindings]
    if missing:
        return f"Formula {formula.formula_id}: missing bindings for roles {sorted(missing)}."
    extra = [role_id for role_id in bindings if role_id not in required_ids]
    if extra:
        return f"Formula {formula.formula_id}: unknown role bindings outside the registry: {sorted(extra)}."
    return ""


def formula_scale_consistency_error(bindings: Mapping[str, ResolvedCell]) -> str:
    scales = {cell.scale for cell in bindings.values()}
    if len(scales) > 1:
        detail = {role_id: cell.scale for role_id, cell in bindings.items()}
        return f"Roles use inconsistent scales within one document: {detail}."
    return ""


def formula_industry_policy_error(formula: FormulaDefinition, company: CompanyInfo | None) -> str:
    if formula_excludes_company(formula, company):
        name = company.name if company else "unknown"
        return f"Formula {formula.formula_id}: {formula.industry_policy} (company: {name})."
    return ""


def formula_report_scope_error(formula: FormulaDefinition, report_scope: str) -> str:
    if report_scope not in formula.allowed_report_scopes:
        return f"Formula {formula.formula_id}: report_scope={report_scope} is not allowed."
    return ""


def formula_denominator_positive_error(formula_id: str, values: Mapping[str, float]) -> str:
    for role_id in _DENOMINATOR_ROLES.get(formula_id, ()):
        if values.get(role_id, 0.0) <= 0:
            return f"Formula {formula_id}: role {role_id} must be > 0; got {values.get(role_id)!r}."
    return ""


def formula_sign_policy_error(formula_id: str, values: Mapping[str, float]) -> str:
    for role_id in _NON_NEGATIVE_ROLES.get(formula_id, ()):
        if values.get(role_id, 0.0) < 0:
            return f"Formula {formula_id}: role {role_id} must be >= 0; got {values.get(role_id)!r}."
    return ""


def formula_answer_finite_error(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"Formula result is not finite: {value!r}."
    return ""


def formula_question_alignment_error(question: str, plan: FormulaScenarioPlan) -> str:
    normalized = question.casefold()
    if plan.entity_ticker.casefold() not in normalized and plan.entity_name.casefold() not in normalized:
        return "Question does not clearly identify the company by name or ticker."
    if plan.year not in question:
        return "Question does not clearly identify the reporting year."
    return ""


class VnNumberError(ValueError):
    pass


def parse_vn_number(raw: str) -> float:
    s = raw.strip()
    if not s or not is_numeric_cell(s):
        raise VnNumberError(f"Invalid Vietnamese number format: {raw!r}")
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




def longitudinal_transform_dependency_error(transform_id: str) -> str:
    missing = transform_missing_period_tokens(transform_id)
    if missing:
        return f"Transform {transform_id} does not reference all three periods (missing {missing})."
    return ""


DEFAULT_NEAR_ZERO_BASE_MIN_RATIO = 0.05


def longitudinal_near_zero_base_error(
    x1: float, x2: float, x3: float, *, min_ratio: float = DEFAULT_NEAR_ZERO_BASE_MIN_RATIO
) -> str:
    values = (x1, x2, x3)
    max_abs = max(abs(v) for v in values)
    if max_abs == 0:
        return "All three levels are zero, so no meaningful growth rate can be computed."
    for value in values:
        if abs(value) < min_ratio * max_abs:
            return (
                f"One of the three levels ({value!r}) is too small relative to the others for the same entity "
                f"(near-zero base; threshold {min_ratio:.0%} of {max_abs!r}), so the growth rate/CoV "
                "is not analytically meaningful even if mathematically valid."
            )
    return ""


def longitudinal_coverage_error(
    entities: tuple[str, str, str],
    periods: tuple[str, str, str],
    bindings: Mapping[tuple[str, str], ResolvedCell],
) -> str:
    required_keys = {(entity, period) for entity in entities for period in periods}
    missing = required_keys - set(bindings)
    if missing:
        return f"Missing bindings for cells: {sorted(missing)}."
    extra = set(bindings) - required_keys
    if extra:
        return f"Unexpected cell bindings outside the locked 3x3 grid: {sorted(extra)}."
    return ""


def longitudinal_scale_consistency_error(
    entities: tuple[str, str, str],
    periods: tuple[str, str, str],
    bindings: Mapping[tuple[str, str], ResolvedCell],
) -> str:
    for entity in entities:
        scales = {bindings[(entity, period)].scale for period in periods}
        if len(scales) > 1:
            detail = {period: bindings[(entity, period)].scale for period in periods}
            return f"Entity {entity}: scale is inconsistent across three periods: {detail}."
    return ""


def longitudinal_question_alignment_error(question: str, plan: LongitudinalScenarioPlan) -> str:
    normalized = question.casefold()
    missing_entities = [
        ticker
        for ticker in plan.entities
        if ticker.casefold() not in normalized
        and plan.entity_names.get(ticker, "").casefold() not in normalized
    ]
    if missing_entities:
        return f"Question does not mention all required companies: {missing_entities}."

    missing_periods = [period for period in plan.periods if period not in question]
    if not missing_periods:
        return ""

    # Keep period handling explicit and deterministic.
    first_period, last_period = plan.periods[0], plan.periods[-1]
    range_pattern = re.compile(rf"{first_period}\s*(?:-|–|—|đến)\s*{last_period}")
    if range_pattern.search(question):
        return ""

    return f"Question does not mention all required years: {missing_periods}."
