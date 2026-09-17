
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class DependencyStatus(str, Enum):
    VALID = "VALID"
    REJECTED = "REJECTED"
    REVIEW = "REVIEW"


class RejectionReason(str, Enum):
    FISCAL_YEAR_MISMATCH = "fiscal_year_mismatch"
    ADJUSTMENT_ONLY_TABLE = "adjustment_only_table"
    UNSUPPORTED_CHART_OF_ACCOUNTS = "unsupported_chart_of_accounts"
    METRIC_LABEL_MISMATCH = "metric_label_mismatch"
    UNIT_CONFLICT = "unit_conflict"
    CURRENT_COLUMN_UNRESOLVED = "current_column_unresolved"
    RAGGED_VALUE_DETACHED = "ragged_value_detached"
    DUPLICATE_PRIMARY_CONFLICT = "duplicate_primary_conflict"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    PARTIAL_REPORTING_PERIOD = "partial_reporting_period"
    PERIOD_BASIS_UNRESOLVED = "period_basis_unresolved"


@dataclass(frozen=True, slots=True)
class AuditVerdict:
    status: DependencyStatus
    reason: RejectionReason | None
    interpretation_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DependencyReviewItem:

    dependency_id: str
    ticker: str
    period: str
    metric_key: str
    table_ref: str
    reason_hint: RejectionReason
    evidence: str
    shortlist: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DependencyReviewResult:
    dependency_id: str
    accept: bool
    interpretation_id: str | None
    reason: RejectionReason | None
    detail: str = ""


class DependencyAuditor(Protocol):
    prompt_version: str
    model_id: str

    def audit(self, items: tuple[DependencyReviewItem, ...]) -> tuple[DependencyReviewResult, ...]:
        ...
