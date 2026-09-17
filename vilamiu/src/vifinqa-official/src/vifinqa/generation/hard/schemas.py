
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vifinqa.generation.hard.numbers import UnitKind, UnitSource
from vifinqa.generation.schemas import AnswerType, MeasurementBasis

StepOperation = Literal[
    "extract",
    "lookup",
    "difference",
    "growth",
    "ratio",
    "filter",
    "count",
    "sum",
    "boolean",
    "argmin",
    "argmax",
]
FinalOperation = Literal["count", "sum", "boolean"]
Comparison = Literal["gt", "gte", "lt", "lte"]
GroupBy = Literal["company", "period", "none"]


class UnitClaim(BaseModel):

    kind: UnitKind = "unknown"
    evidence: str = ""
    source: UnitSource = "none"


class MetricRoleInput(BaseModel):
    kind: Literal["metric_role"] = "metric_role"
    metric_role: str


class StepOutputInput(BaseModel):
    kind: Literal["step_output"] = "step_output"
    step_id: str


StepInput = MetricRoleInput | StepOutputInput


class ReasoningStep(BaseModel):

    step_id: str
    operation: StepOperation
    inputs: list[StepInput] = Field(default_factory=list)
    group_by: GroupBy = "none"
    output: str
    description: str = ""


class BaseHardPlan(BaseModel):

    reasoning_steps: list[ReasoningStep]
    calculation_steps: list[str] = Field(default_factory=list)


class HardPlanDraft(BaseModel):

    feasible: bool
    reason: str = ""
    concept_name: str = ""
    metric_role: str = ""
    concept_formula: str = ""
    financial_rationale: str = ""
    population: str = ""
    unit: str = ""
    answer_type: AnswerType = "number"
    # Keep unit and scale handling explicit.
    # (xem `hard/p1.py::_resolve_scale`).
    metric_value_kind: Literal["money", "percentage", "number"] = "money"
    measurement_basis: MeasurementBasis = "unknown"
    final_operation: FinalOperation = "count"
    comparison: Comparison = "gt"


class HardPlan(BaseHardPlan):

    draft: HardPlanDraft
    threshold: float | None = None


SelectorOperation = Literal["argmax", "argmin"]


class HardMultiMetricPlanDraft(BaseModel):

    feasible: bool
    reason: str = ""
    selector_concept_name: str = ""
    selector_metric_role: str = ""
    selector_concept_formula: str = ""
    selector_financial_rationale: str = ""
    selector_unit: str = ""
    # Keep unit and scale handling explicit.
    selector_value_kind: Literal["money", "percentage", "number"] = "number"
    selector_measurement_basis: MeasurementBasis = "unknown"
    selector_operation: SelectorOperation = "argmax"
    answer_concept_name: str = ""
    answer_metric_role: str = ""
    answer_concept_formula: str = ""
    answer_financial_rationale: str = ""
    answer_unit: str = ""
    answer_measurement_basis: MeasurementBasis = "unknown"
    answer_type: AnswerType = "number"
    population: str = ""
    # Keep period handling explicit and deterministic.
    # Keep LLM behavior within the declared contract.
    selector_anchor_ref: str = ""
    answer_anchor_ref: str = ""
    analysis_intent: str = ""
    relationship_rationale: str = ""


class HardP3PairJudgment(BaseModel):

    valid: bool
    reason: str = ""


class HardP3QuestionJudgment(BaseModel):

    valid: bool
    reason: str = ""


class HardP3Plan(BaseHardPlan):

    draft: HardMultiMetricPlanDraft


def derive_calculation_steps(steps: list[ReasoningStep]) -> list[str]:
    return [step.description for step in steps if step.description]


class HardTableMapping(BaseModel):

    table_ref: str
    has_concept: bool
    row_label: str = ""
    column_label: str = ""
    measurement_basis: MeasurementBasis = "unknown"
    unit: UnitClaim = Field(default_factory=UnitClaim)


class HardMappingResult(BaseModel):
    mappings: list[HardTableMapping] = Field(default_factory=list)


class HardMultiRoleTableMapping(BaseModel):

    table_ref: str
    has_selector: bool = False
    selector_row_label: str = ""
    selector_column_label: str = ""
    selector_measurement_basis: MeasurementBasis = "unknown"
    selector_unit: UnitClaim = Field(default_factory=UnitClaim)
    has_answer: bool = False
    answer_row_label: str = ""
    answer_column_label: str = ""
    answer_measurement_basis: MeasurementBasis = "unknown"
    answer_unit: UnitClaim = Field(default_factory=UnitClaim)

    @model_validator(mode="after")
    def require_location_for_present_roles(self) -> HardMultiRoleTableMapping:
        missing: list[str] = []
        if self.has_selector:
            if not self.selector_row_label.strip():
                missing.append("selector_row_label")
            if not self.selector_column_label.strip():
                missing.append("selector_column_label")
        if self.has_answer:
            if not self.answer_row_label.strip():
                missing.append("answer_row_label")
            if not self.answer_column_label.strip():
                missing.append("answer_column_label")
        if missing:
            raise ValueError(f"has_selector/has_answer=true requires location: {missing}")
        return self


class HardMultiRoleMappingResult(BaseModel):
    mappings: list[HardMultiRoleTableMapping] = Field(default_factory=list)


class MetricBinding(BaseModel):

    metric_role: str
    table_ref: str
    ticker: str
    period: str
    report_scope: Literal["consolidated", "parent"]
    row_label: str
    column_label: str
    raw_unit: str
    scale: float
    normalized_unit: str
    measurement_basis: MeasurementBasis
    evidence: str = ""


@dataclass(slots=True)
class RejectionCounters:

    retrieval: int = 0
    window: int = 0
    inventory: int = 0
    plan: int = 0
    pair_judge: int = 0
    role_retrieval: int = 0
    mapping: int = 0
    scope_or_basis: int = 0
    unit_gate: int = 0
    graph_gate: int = 0
    coverage_gate: int = 0
    threshold: int = 0
    compile: int = 0
    execute: int = 0
    finance_judge: int = 0
    alignment_judge: int = 0
    naturalness_judge: int = 0
    question_judge: int = 0
    question_style: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "retrieval": self.retrieval,
            "window": self.window,
            "inventory": self.inventory,
            "plan": self.plan,
            "pair_judge": self.pair_judge,
            "role_retrieval": self.role_retrieval,
            "mapping": self.mapping,
            "scope_or_basis": self.scope_or_basis,
            "unit_gate": self.unit_gate,
            "graph_gate": self.graph_gate,
            "coverage_gate": self.coverage_gate,
            "threshold": self.threshold,
            "compile": self.compile,
            "execute": self.execute,
            "finance_judge": self.finance_judge,
            "alignment_judge": self.alignment_judge,
            "naturalness_judge": self.naturalness_judge,
            "question_judge": self.question_judge,
            "question_style": self.question_style,
        }


@dataclass(slots=True)
class ExecutionTrace:

    scenario: str
    plan: HardPlan | HardP3Plan
    bindings: list[MetricBinding] = field(default_factory=list)
    accessed_refs: frozenset[str] = frozenset()
