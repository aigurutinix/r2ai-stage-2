
from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.hard.numbers import VnNumberError, parse_vn_number
from vifinqa.generation.hard.recipe.audit.base import (
    AuditVerdict,
    DependencyAuditor,
    DependencyReviewItem,
    DependencyStatus,
    RejectionReason,
)
from vifinqa.generation.hard.recipe.audit.cache import AuditCache, cache_key
from vifinqa.generation.hard.recipe.audit.deterministic import run_deterministic_checks
from vifinqa.generation.hard.recipe.audit.dependencies import (
    DependencyRecord,
    collect_dependency_closure,
)
from vifinqa.generation.hard.recipe.audit.period import (
    PeriodEvidence,
    build_period_evidence,
    deterministic_period_verdict,
    period_subject_for,
)
from vifinqa.generation.hard.recipe.base import (
    MetricRoleInput,
    MetricTerm,
    MetricTerms,
    ReasoningGraph,
    ReasoningNode,
)

logger = logging.getLogger(__name__)

_REVIEW_SHORTLIST = ("confirm", "reject")


class DependencyAuditRejected(ValueError):

    def __init__(self, reasons: dict[str, RejectionReason]) -> None:
        self.reasons = reasons
        super().__init__(f"{len(reasons)} dependency bị reject: {reasons}")


@dataclass(slots=True)
class AuditStats:

    dependencies_total: int = 0
    cache_hits: int = 0
    deterministic_valid: int = 0
    deterministic_rejected: int = 0
    review_sent_to_llm: int = 0
    llm_calls: int = 0
    llm_confirmed: int = 0
    llm_rejected: int = 0
    period_deterministic_valid: int = 0
    period_deterministic_rejected: int = 0
    period_review_sent: int = 0
    period_cache_hits: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuditedValues:
    values: dict[str, float]
    stats: AuditStats


@dataclass(frozen=True, slots=True)
class DependencyAuditReport:
    """Complete closure audit, including valid values and explicit rejected dependencies.

    This report is used only while constructing an audited eligible universe, before candidate
    identity/PublicSpec are locked. ``audit_dependency_closure`` remains the strict all-or-nothing
    API for an already formed candidate.
    """

    values: dict[str, float]
    rejection_reasons: dict[str, RejectionReason]
    stats: AuditStats


def _table_id_from_ref(table_ref: str) -> int:
    return int(table_ref.rsplit("_", 1)[-1])


def _load_row(record: DependencyRecord, doc: DocumentRef) -> tuple[str, ...]:
    table_id = _table_id_from_ref(record.binding.table_ref)
    table = load_table(
        doc.table_csv_path(table_id),
        ticker=doc.ticker,
        year=doc.year,
        doc_name=doc.doc_name,
        table_id=table_id,
    )
    return table.rows[record.binding.row_idx]


def _source_content_hash(record: DependencyRecord, doc: DocumentRef) -> str:
    row = _load_row(record, doc)
    raw = row[record.binding.col_idx] if record.binding.col_idx < len(row) else ""
    raw_key = f"{record.binding.table_ref}:{record.binding.row_idx}:{record.binding.col_idx}:{raw}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _period_cache_key(evidence: PeriodEvidence, *, auditor: DependencyAuditor) -> str:
    return cache_key(
        source_content_hash=f"period:{evidence.stable_hash()}",
        dependency_id=evidence.subject.identity,
        interpretation_id=None,
        prompt_version=auditor.prompt_version,
        model_id=auditor.model_id,
    )


def _materialize_value(record: DependencyRecord, doc: DocumentRef) -> float:
    row = _load_row(record, doc)
    raw = row[record.binding.col_idx]
    try:
        return parse_vn_number(raw) * record.binding.scale
    except VnNumberError as exc:
        raise DependencyAuditRejected(
            {record.dependency_id: RejectionReason.RAGGED_VALUE_DETACHED}
        ) from exc


def _review_item(
    record: DependencyRecord,
    doc: DocumentRef,
    reason_hint: RejectionReason,
    detail: str,
) -> DependencyReviewItem:
    row = _load_row(record, doc)
    evidence = f"{detail} | dòng thô: {list(row)}"
    return DependencyReviewItem(
        dependency_id=record.dependency_id,
        ticker=record.ticker,
        period=record.period,
        metric_key=record.binding.metric_key,
        table_ref=record.binding.table_ref,
        reason_hint=reason_hint,
        evidence=evidence,
        shortlist=_REVIEW_SHORTLIST,
    )


def _period_review_item(
    record: DependencyRecord,
    evidence: PeriodEvidence,
    reason_hint: RejectionReason,
    detail: str,
) -> DependencyReviewItem:
    return DependencyReviewItem(
        dependency_id=f"period:{evidence.subject.identity}",
        ticker=record.ticker,
        period=record.period,
        metric_key=record.binding.metric_key,
        table_ref=record.binding.table_ref,
        reason_hint=reason_hint,
        evidence=f"{detail}\n\n{evidence.to_prompt_text()}",
        shortlist=_REVIEW_SHORTLIST,
    )


def _audit_period_subjects(
    records: tuple[DependencyRecord, ...],
    *,
    docs_by_name: dict[str, DocumentRef],
    cache: AuditCache,
    auditor: DependencyAuditor,
    stats: AuditStats,
) -> dict[str, RejectionReason]:
    """Audit each table-column period subject once, then fan out reject reasons to dependency IDs."""
    grouped: dict[str, list[DependencyRecord]] = defaultdict(list)
    for record in records:
        grouped[period_subject_for(record).identity].append(record)

    rejected_subjects: dict[str, RejectionReason] = {}
    review_items: list[DependencyReviewItem] = []
    review_evidence_by_id: dict[str, PeriodEvidence] = {}

    for subject_id, subject_records in grouped.items():
        record = subject_records[0]
        doc_name = record.binding.table_ref.split("|", 1)[0]
        doc = docs_by_name.get(doc_name)
        if doc is None or doc.text_path is None:
            rejected_subjects[subject_id] = RejectionReason.PERIOD_BASIS_UNRESOLVED
            continue
        table_id = _table_id_from_ref(record.binding.table_ref)
        table = load_table(
            doc.table_csv_path(table_id),
            ticker=doc.ticker,
            year=doc.year,
            doc_name=doc.doc_name,
            table_id=table_id,
        )
        document = parse_document(doc.text_path)
        evidence = build_period_evidence(
            record, doc=doc, document=document, table=table
        )
        key = _period_cache_key(evidence, auditor=auditor)
        cached = cache.get(key)
        if cached is not None:
            stats.period_cache_hits += 1
            if cached.status != DependencyStatus.VALID:
                rejected_subjects[subject_id] = (
                    cached.reason or RejectionReason.PERIOD_BASIS_UNRESOLVED
                )
            continue

        verdict = deterministic_period_verdict(evidence)
        if verdict.status == DependencyStatus.VALID:
            stats.period_deterministic_valid += 1
            cache.put(key, verdict)
        elif verdict.status == DependencyStatus.REJECTED:
            stats.period_deterministic_rejected += 1
            cache.put(key, verdict)
            rejected_subjects[subject_id] = (
                verdict.reason or RejectionReason.PERIOD_BASIS_UNRESOLVED
            )
        else:
            item = _period_review_item(
                record,
                evidence,
                verdict.reason or RejectionReason.PERIOD_BASIS_UNRESOLVED,
                verdict.detail,
            )
            review_items.append(item)
            review_evidence_by_id[item.dependency_id] = evidence

    stats.period_review_sent = len(review_items)
    if review_items:
        results = auditor.audit(tuple(review_items))
        stats.llm_calls += 1
        results_by_id = {r.dependency_id: r for r in results}
        for item in review_items:
            evidence = review_evidence_by_id[item.dependency_id]
            subject_id = evidence.subject.identity
            result = results_by_id.get(item.dependency_id)
            key = _period_cache_key(evidence, auditor=auditor)
            if result is None or not result.accept:
                stats.llm_rejected += 1
                reason = (
                    result.reason
                    if result and result.reason
                    else RejectionReason.PERIOD_BASIS_UNRESOLVED
                )
                rejected_subjects[subject_id] = reason
                cache.put(
                    key, AuditVerdict(status=DependencyStatus.REJECTED, reason=reason)
                )
            else:
                stats.llm_confirmed += 1
                cache.put(
                    key,
                    AuditVerdict(
                        status=DependencyStatus.VALID,
                        reason=None,
                        interpretation_id=result.interpretation_id,
                        detail=result.detail,
                    ),
                )

    rejected_dependencies: dict[str, RejectionReason] = {}
    for subject_id, reason in rejected_subjects.items():
        for record in grouped[subject_id]:
            rejected_dependencies[record.dependency_id] = reason
    return rejected_dependencies


def audit_dependency_report(
    graph: ReasoningGraph,
    *,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    cache: AuditCache,
    auditor: DependencyAuditor,
) -> DependencyAuditReport:
    """Audit a provisional universe and return every valid/rejected dependency explicitly."""
    records = collect_dependency_closure(graph, docs_by_name=docs_by_name)
    stats = AuditStats(dependencies_total=len(records))
    rejection_reasons: dict[str, RejectionReason] = _audit_period_subjects(
        records, docs_by_name=docs_by_name, cache=cache, auditor=auditor, stats=stats
    )
    values: dict[str, float] = {}
    to_review: dict[str, tuple[DependencyRecord, DocumentRef]] = {}
    review_items_by_doc: dict[str, list[DependencyReviewItem]] = defaultdict(list)

    for record in records:
        if record.dependency_id in rejection_reasons:
            continue
        doc_name = record.binding.table_ref.split("|", 1)[0]
        doc = docs_by_name.get(doc_name)
        if doc is None:
            rejection_reasons[record.dependency_id] = (
                RejectionReason.PROVENANCE_MISMATCH
            )
            continue

        content_hash = _source_content_hash(record, doc)
        key = cache_key(
            source_content_hash=content_hash,
            dependency_id=record.dependency_id,
            interpretation_id=None,
            prompt_version=auditor.prompt_version,
            model_id=auditor.model_id,
        )
        cached = cache.get(key)
        if cached is not None:
            stats.cache_hits += 1
            if cached.status == DependencyStatus.VALID:
                values[record.dependency_id] = _materialize_value(record, doc)
            else:
                rejection_reasons[record.dependency_id] = (
                    cached.reason or RejectionReason.PROVENANCE_MISMATCH
                )
            continue

        verdict = run_deterministic_checks(record, doc=doc, company_meta=company_meta)
        if verdict.status == DependencyStatus.VALID:
            stats.deterministic_valid += 1
            cache.put(key, verdict)
            values[record.dependency_id] = _materialize_value(record, doc)
        elif verdict.status == DependencyStatus.REJECTED:
            stats.deterministic_rejected += 1
            cache.put(key, verdict)
            rejection_reasons[record.dependency_id] = (
                verdict.reason or RejectionReason.PROVENANCE_MISMATCH
            )
        else:  # REVIEW
            to_review[record.dependency_id] = (record, doc)
            review_items_by_doc[doc_name].append(
                _review_item(
                    record,
                    doc,
                    verdict.reason or RejectionReason.PROVENANCE_MISMATCH,
                    verdict.detail,
                )
            )

    stats.review_sent_to_llm = sum(len(v) for v in review_items_by_doc.values())

    for items in review_items_by_doc.values():
        results = auditor.audit(tuple(items))
        stats.llm_calls += 1
        results_by_id = {r.dependency_id: r for r in results}
        for item in items:
            dependency_id = item.dependency_id
            record, doc = to_review[dependency_id]
            result = results_by_id.get(dependency_id)
            content_hash = _source_content_hash(record, doc)

            if result is None or not result.accept:
                stats.llm_rejected += 1
                reason = (
                    result.reason
                    if result and result.reason
                    else RejectionReason.PROVENANCE_MISMATCH
                )
                rejection_reasons[dependency_id] = reason
                key = cache_key(
                    source_content_hash=content_hash,
                    dependency_id=dependency_id,
                    interpretation_id=None,
                    prompt_version=auditor.prompt_version,
                    model_id=auditor.model_id,
                )
                cache.put(
                    key, AuditVerdict(status=DependencyStatus.REJECTED, reason=reason)
                )
            else:
                stats.llm_confirmed += 1
                values[dependency_id] = _materialize_value(record, doc)
                key = cache_key(
                    source_content_hash=content_hash,
                    dependency_id=dependency_id,
                    interpretation_id=result.interpretation_id,
                    prompt_version=auditor.prompt_version,
                    model_id=auditor.model_id,
                )
                cache.put(
                    key,
                    AuditVerdict(
                        status=DependencyStatus.VALID,
                        reason=None,
                        interpretation_id=result.interpretation_id,
                    ),
                )

    stats.rejection_reasons = dict(Counter(r.value for r in rejection_reasons.values()))

    return DependencyAuditReport(
        values=values,
        rejection_reasons=rejection_reasons,
        stats=stats,
    )


def audit_dependency_closure(
    graph: ReasoningGraph,
    *,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    cache: AuditCache,
    auditor: DependencyAuditor,
) -> AuditedValues:
    """Strict audit for an already formed candidate: any rejected dependency rejects it whole."""
    report = audit_dependency_report(
        graph,
        docs_by_name=docs_by_name,
        company_meta=company_meta,
        cache=cache,
        auditor=auditor,
    )
    if report.rejection_reasons:
        raise DependencyAuditRejected(report.rejection_reasons)
    return AuditedValues(values=report.values, stats=report.stats)


def _dependency_id(role_id: str, term: MetricTerm) -> str:
    return f"{role_id}:{term.cell.table_ref}:{term.cell.row_idx}:{term.cell.col_idx}"


def _rebuild_term(
    role_id: str, term: MetricTerm, values: dict[str, float]
) -> MetricTerm:
    dependency_id = _dependency_id(role_id, term)
    if dependency_id not in values:
        raise DependencyAuditRejected(
            {dependency_id: RejectionReason.PROVENANCE_MISMATCH}
        )
    validated_cell = replace(term.cell, value=values[dependency_id])
    return replace(term, cell=validated_cell)


def _rebuild_role(role: MetricRoleInput, values: dict[str, float]) -> MetricRoleInput:
    new_bindings = {
        key: MetricTerms(
            numerator=tuple(
                _rebuild_term(role.role_id, term, values) for term in terms.numerator
            ),
            denominator=tuple(
                _rebuild_term(role.role_id, term, values) for term in terms.denominator
            ),
            requires_positive_denominator=terms.requires_positive_denominator,
        )
        for key, terms in role.bindings.items()
    }
    return replace(role, bindings=new_bindings)


def _rebuild_node(
    node: ReasoningNode, role_map: dict[str, MetricRoleInput]
) -> ReasoningNode:
    new_inputs = tuple(
        role_map[node_input.role_id]
        if isinstance(node_input, MetricRoleInput)
        else node_input
        for node_input in node.inputs
    )
    return replace(node, inputs=new_inputs)


def rebuild_validated_graph(
    graph: ReasoningGraph, audited: AuditedValues
) -> ReasoningGraph:
    new_roles = tuple(
        _rebuild_role(role, audited.values) for role in graph.metric_roles
    )
    role_map = {role.role_id: role for role in new_roles}
    new_nodes = tuple(_rebuild_node(node, role_map) for node in graph.nodes)
    return replace(graph, metric_roles=new_roles, nodes=new_nodes)
