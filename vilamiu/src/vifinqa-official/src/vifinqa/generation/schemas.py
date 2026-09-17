
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "intermediate", "hard"]
Answer = str | float | int | bool
Operation = Literal[
    "difference",
    "ratio",
    "growth",
    "share",
    "sum",
    "average",
    "minimum",
    "maximum",
    "argmin",
    "argmax",
    "count",
    "boolean",
]
AnswerType = Literal["money", "percentage", "number", "year", "company", "boolean"]
MeasurementBasis = Literal["gross", "net", "not_applicable", "unknown"]


class GeneratedQA(BaseModel):

    feasible: bool = True
    reason: str | None = None
    question: str = ""
    answer: Answer | None = None
    pandas_query: str = (
        ""  # Keep LLM behavior within the declared contract.
    )
    relevant_docs: list[str] = Field(default_factory=list)
    relevant_tables: list[str] = Field(default_factory=list)


class NaturalnessJudgment(BaseModel):

    natural: bool
    reason: str = ""
    rewrite: str | None = None


class ConceptSelection(BaseModel):

    feasible: bool
    concept_name: str = ""
    concept_formula: str = ""
    operation: Operation = "difference"
    answer_type: AnswerType = "number"
    unit: str = ""
    table_topic: str = ""
    population: str = ""
    financial_rationale: str = ""
    calculation_steps: list[str] = Field(default_factory=list)
    target_periods: list[str] = Field(default_factory=list)
    time_basis: Literal["period", "point_in_time", "unknown"] = "unknown"
    measurement_basis: MeasurementBasis = "unknown"
    observation_count: int = 0
    reason: str = ""


class FinancialValidityJudgment(BaseModel):

    valid: bool
    reason: str = ""


class TableConceptMapping(BaseModel):

    table_ref: str
    has_concept: bool
    row_or_column_hint: str = ""
    measurement_basis: MeasurementBasis = "unknown"


class ConceptMappingResult(BaseModel):
    mappings: list[TableConceptMapping] = Field(default_factory=list)


class QuestionDraft(BaseModel):

    question: str


class PandasQuery(BaseModel):

    pandas_query: str


class QueryAnswer(PandasQuery):

    answer: Answer


class EasyFactQuery(PandasQuery):

    metric_name: str
    period_label: str
    time_basis: Literal["period", "point_in_time"]
    unit: str
    report_scope: Literal["consolidated", "parent"]
    answer: Answer | None = None


class FormulaRoleCellMapping(BaseModel):

    role_id: str
    found: bool = False
    table_ref: str = ""
    row_label: str = ""
    column_label: str = ""
    reason: str = ""


class FormulaMappingResult(BaseModel):
    mappings: list[FormulaRoleCellMapping] = Field(default_factory=list)


class QARecord(BaseModel):

    id: int
    question: str
    answer: Answer
    relevant_docs: list[str]
    relevant_tables: list[str]
    pandas_query: str
    csv_path: str | list[str]
    difficulty: Difficulty
    # Present for reviewed Hard Cube intents.  Optional keeps the shared schema compatible with
    # easy/medium and legacy hard generators.
    template_id: str | None = None
