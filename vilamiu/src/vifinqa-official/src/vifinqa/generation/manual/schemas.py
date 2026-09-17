"""Schemas persisted by the independent Hard manual author/audit lane."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdeationNote(StrictModel):
    intent: str = Field(min_length=1)
    financial_rationale: str = Field(min_length=1)
    candidate_metrics: list[str] = Field(min_length=1)
    adaptive_dependency: str = Field(min_length=1)


class ReasoningTrace(StrictModel):
    intent: str = Field(min_length=1)
    reasoning_steps: list[str] = Field(min_length=2)
    adaptive_dependencies: list[str] = Field(min_length=1)
    dynamic_outputs: dict[str, object] = Field(default_factory=dict)
    contributing_bindings: list[str] = Field(min_length=1)


class AuthorRubric(StrictModel):
    terminal_numeric: bool
    arithmetic: bool
    data_grounding: bool
    finance_formula: bool
    terminology: bool
    dependency: bool
    hardness: bool
    alignment: bool
    naturalness: bool
    tier: bool
    no_forced_formula: bool
    notes: str = ""

    def failed_items(self) -> list[str]:
        return [
            name
            for name in (
                "terminal_numeric",
                "arithmetic",
                "data_grounding",
                "finance_formula",
                "terminology",
                "dependency",
                "hardness",
                "alignment",
                "naturalness",
                "tier",
                "no_forced_formula",
            )
            if not getattr(self, name)
        ]


class CsvMutation(StrictModel):
    table_ref: str = Field(min_length=1)
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    replacement: str


class MetamorphicSpec(StrictModel):
    name: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    mutations: list[CsvMutation] = Field(min_length=1)
    expect_result_change: bool = True


class ManualRecordDraft(StrictModel):
    question: str = Field(min_length=1)
    answer: object
    relevant_docs: list[str] = Field(min_length=1)
    relevant_tables: list[str] = Field(min_length=1)
    pandas_query: str = Field(min_length=1)
    csv_path: str | list[str]
    difficulty: Literal["hard"] = "hard"
    trace: ReasoningTrace
    self_review: AuthorRubric
    metamorphic_specs: list[MetamorphicSpec] = Field(min_length=1)


class ManualAuthorDraft(StrictModel):
    work_id: str
    pack_id: str
    batch_id: str
    author_id: str
    ideation: list[IdeationNote] = Field(default_factory=list)
    records: list[ManualRecordDraft] = Field(default_factory=list, max_length=5)
    no_yield_reason: str = ""

    @model_validator(mode="after")
    def require_output_or_reason(self) -> ManualAuthorDraft:
        if not self.records and not self.no_yield_reason.strip():
            raise ValueError("A batch without records must provide no_yield_reason")
        if self.records and not self.ideation:
            raise ValueError("A batch with records must save ideation before terminal-value inspection")
        return self


class AuditReconstruction(StrictModel):
    record_id: int
    intent: str = Field(min_length=1)
    computation: str = Field(min_length=1)
    adaptive_dependency: str = Field(min_length=1)


class AuditRubric(StrictModel):
    terminal_numeric: bool
    arithmetic: bool
    data_grounding: bool
    finance_formula: bool
    terminology: bool
    dependency: bool
    hardness: bool
    alignment: bool
    naturalness: bool
    tier: bool
    no_forced_formula: bool

    def failed_items(self) -> list[str]:
        return [name for name in type(self).model_fields if not getattr(self, name)]


class AuditRecordReview(StrictModel):
    record_id: int
    verdict: Literal["accept", "reject"]
    rubric: AuditRubric
    issues: list[str] = Field(default_factory=list)
    root_layer: str = ""
    terminology_or_formula_sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_rubric(self) -> AuditRecordReview:
        failed = self.rubric.failed_items()
        if self.verdict == "accept" and failed:
            raise ValueError(f"verdict=accept but the rubric failed: {failed}")
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict=reject must provide issues")
        return self


class ManualAuditDraft(StrictModel):
    work_id: str
    batch_id: str
    audit_id: str
    reviewer_id: str
    reconstructions: list[AuditReconstruction] = Field(default_factory=list)
    reviews: list[AuditRecordReview] = Field(default_factory=list)
    batch_notes: str = ""
