
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from vifinqa.generation.schemas import AnswerType

ReportScope = Literal["consolidated", "parent"]


@dataclass(frozen=True, slots=True)
class FormulaRole:

    role_id: str
    label_vi: str
    concept_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormulaDefinition:
    formula_id: str
    name_vi: str
    formula_text: str
    required_roles: tuple[FormulaRole, ...]
    answer_type: AnswerType
    unit: str
    time_basis: str  # Keep period handling explicit and deterministic.
    allowed_report_scopes: tuple[ReportScope, ...]
    industry_policy: str
    denominator_policy: str
    sign_policy: str
    terminology_notes: str
    source_urls: tuple[str, ...]
    enabled: bool

    @property
    def required_observation_count(self) -> int:
        return len(self.required_roles)

    def role_ids(self) -> tuple[str, ...]:
        return tuple(role.role_id for role in self.required_roles)


@dataclass(frozen=True, slots=True)
class ResolvedCell:

    table_ref: str
    row_idx: int
    col_idx: int
    row_label: str
    column_label: str
    raw_value: str
    # Keep unit and scale handling explicit.
    # Keep unit and scale handling explicit.
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class FormulaScenarioPlan:

    scenario: Literal["same_doc_multi_role_formula"]
    formula_id: str
    entity_ticker: str
    entity_name: str
    document_doc_name: str
    year: str
    report_scope: ReportScope
    answer_type: AnswerType
    unit: str
    bindings: Mapping[str, ResolvedCell] = field(default_factory=dict)

    def relevant_tables(self) -> list[str]:
        seen: dict[str, None] = {}
        for cell in self.bindings.values():
            seen.setdefault(cell.table_ref, None)
        return list(seen)
