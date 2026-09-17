
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vifinqa.generation.hard.schemas import MetricBinding, UnitClaim
from vifinqa.generation.schemas import AnswerType, MeasurementBasis


class KeyedMetricRoleDraft(BaseModel):
    anchor_ref: str = ""
    concept_name: str = ""
    metric_role: str = ""
    concept_formula: str = ""
    financial_rationale: str = ""
    unit: str = ""
    value_kind: Literal["money", "percentage", "number"] = "number"
    measurement_basis: MeasurementBasis = "unknown"


class PeriodFilterSelectLookupDraft(BaseModel):
    feasible: bool
    reason: str = ""
    filter_role: KeyedMetricRoleDraft = Field(default_factory=KeyedMetricRoleDraft)
    filter_comparison: Literal["gt", "gte", "lt", "lte"] = "gt"
    filter_threshold: float = 0.0
    selector_role: KeyedMetricRoleDraft = Field(default_factory=KeyedMetricRoleDraft)
    selector_operation: Literal["argmax", "argmin"] = "argmax"
    answer_role: KeyedMetricRoleDraft = Field(default_factory=KeyedMetricRoleDraft)
    answer_type: AnswerType = "number"
    population: str = ""
    analysis_intent: str = ""
    relationship_rationale: str = ""

    @model_validator(mode="after")
    def require_distinct_complete_roles(self) -> PeriodFilterSelectLookupDraft:
        if not self.feasible:
            return self
        roles = (self.filter_role, self.selector_role, self.answer_role)
        missing: list[str] = []
        for index, role in enumerate(roles):
            for field_name in (
                "anchor_ref",
                "concept_name",
                "metric_role",
                "concept_formula",
                "financial_rationale",
                "unit",
            ):
                if not getattr(role, field_name).strip():
                    missing.append(f"role[{index}].{field_name}")
        if missing:
            raise ValueError(f"Role is missing required fields: {missing}")
        metric_roles = [role.metric_role for role in roles]
        if len(set(metric_roles)) != 3:
            raise ValueError("filter, selector, and answer must use three distinct metric roles")
        if self.answer_type == "company":
            raise ValueError("Period-based lookup recipes do not return a company")
        return self


class RoleCandidateMapping(BaseModel):
    table_ref: str
    row_label: str
    column_label: str
    unit: UnitClaim = Field(default_factory=UnitClaim)


class RolePeriodMetricExpression(BaseModel):
    metric_role: str
    period: str
    operation: Literal["direct", "sum"] = "direct"
    cells: list[RoleCandidateMapping] = Field(default_factory=list)
    measurement_basis: MeasurementBasis = "unknown"

    @model_validator(mode="after")
    def require_operation_arity(self) -> RolePeriodMetricExpression:
        expected = 1 if self.operation == "direct" else 2
        if len(self.cells) < expected or (self.operation == "direct" and len(self.cells) != 1):
            requirement = "exactly one cell" if self.operation == "direct" else "at least two cells"
            raise ValueError(f"operation={self.operation} requires {requirement}")
        cell_keys = [(cell.table_ref, cell.row_label, cell.column_label) for cell in self.cells]
        if len(cell_keys) != len(set(cell_keys)):
            raise ValueError("An expression must not repeat the same cell")
        return self


class MultiRoleMappingResult(BaseModel):
    feasible: bool = True
    reason: str = ""
    expressions: list[RolePeriodMetricExpression] = Field(default_factory=list)


class ResolvedMetricExpression(BaseModel):
    metric_role: str
    period: str
    operation: Literal["direct", "sum"]
    cells: list[MetricBinding]

    @model_validator(mode="after")
    def require_consistent_cells(self) -> ResolvedMetricExpression:
        expected = 1 if self.operation == "direct" else 2
        if len(self.cells) < expected or (self.operation == "direct" and len(self.cells) != 1):
            raise ValueError("Resolved expression has invalid arity")
        if any(cell.metric_role != self.metric_role or cell.period != self.period for cell in self.cells):
            raise ValueError("Resolved cell does not match the expression role/period")
        if len({cell.normalized_unit for cell in self.cells}) != 1:
            raise ValueError("Expression cells do not share one normalized unit")
        if len({cell.measurement_basis for cell in self.cells}) != 1:
            raise ValueError("Expression cells do not share one measurement basis")
        return self
