"""Hard Cube V4 CP6 capacity/catalog audit.

This module is read-only with respect to production generation: it enumerates the
analytical frame catalog, runs deterministic feasibility through the production
planner/finalizer, and optionally sends the post-gate candidate specs to the
production semantic auditor under an explicit LLM-call budget.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.generation.budget import GenerationBudgetExceeded, LLMCallBudget
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.llm import FakeDependencyAuditor
from vifinqa.generation.hard.recipe.audit.service import AuditStats, DependencyAuditRejected
from vifinqa.generation.hard.recipe.base import CandidateRejected
from vifinqa.generation.hard.recipe.evaluator import EvaluationError, evaluate_graph
from vifinqa.generation.hard.recipe.finalize import finalize_candidate
from vifinqa.generation.hard.recipe.grounded.analytical_planner import (
    FRAMES,
    AnalyticalFrame,
    _build_attempt,
    _public_spec,
)
from vifinqa.generation.hard.recipe.grounded.domains import (
    adjacent_period_domains,
    cross_entity_period_window_domains,
    same_period_domains,
    single_entity_period_window_domains,
)
from vifinqa.generation.hard.recipe.planner import RecipeCandidate, table_ref_to_path_map
from vifinqa.generation.hard.recipe.semantic_auditor import (
    LLMSemanticAuditor,
    MAX_SEMANTIC_AUDIT_BATCH_SIZE,
    SemanticAuditResponseError,
)
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.panel.catalog import get_ratio, metric_name
from vifinqa.generation.panel.store import JsonCubeStore


@dataclass(slots=True)
class FrameCapacity:
    frame_id: str
    story_family: str
    topology: str
    domain_kind: str
    metric_keys: tuple[str, ...]
    metric_families: tuple[str, ...]
    meaning: str
    finance_rationale: str
    interpretation_limits: tuple[str, ...]
    raw_domains: int = 0
    build_ok: int = 0
    preflight_ok: int = 0
    finalized_ok: int = 0
    projected_population_ok: int = 0
    semantic_audited: int = 0
    semantic_accepted: int = 0
    semantic_rejected: int = 0
    semantic_needs_review: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)
    evidence_tables: Counter[str] = field(default_factory=Counter)
    evidence_domains: set[str] = field(default_factory=set)
    audit_stats: AuditStats = field(default_factory=AuditStats)

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "story_family": self.story_family,
            "topology": self.topology,
            "domain_kind": self.domain_kind,
            "metric_keys": self.metric_keys,
            "metric_families": self.metric_families,
            "meaning": self.meaning,
            "finance_rationale": self.finance_rationale,
            "interpretation_limits": self.interpretation_limits,
            "stage_counts": {
                "raw_domains": self.raw_domains,
                "build_ok": self.build_ok,
                "preflight_ok": self.preflight_ok,
                "finalized_ok": self.finalized_ok,
                "projected_population_ok": self.projected_population_ok,
                "semantic_audited": self.semantic_audited,
                "semantic_accepted": self.semantic_accepted,
            },
            "semantic_rejected": self.semantic_rejected,
            "semantic_needs_review": self.semantic_needs_review,
            "reject_reasons": dict(self.reject_reasons.most_common()),
            "evidence": {
                "distinct_domain_keys": len(self.evidence_domains),
                "distinct_tables": len(self.evidence_tables),
                "top_tables": self.evidence_tables.most_common(5),
            },
            "audit_stats": _audit_stats_payload(self.audit_stats),
        }


@dataclass(slots=True)
class CapacityAuditResult:
    frames: list[FrameCapacity]
    candidates: list[RecipeCandidate]
    semantic_errors: list[str]
    semantic_budget_calls: int
    semantic_budget_max: int | None
    cp5b_report_found: bool

    def to_payload(self) -> dict[str, Any]:
        accepted_total = sum(frame.semantic_accepted for frame in self.frames)
        projected_total = sum(frame.projected_population_ok for frame in self.frames)
        families_with_projected = {
            frame.story_family for frame in self.frames if frame.projected_population_ok > 0
        }
        families_with_semantic_accept = {
            frame.story_family for frame in self.frames if frame.semantic_accepted > 0
        }
        return {
            "summary": {
                "frames": len(self.frames),
                "story_families": len({frame.story_family for frame in self.frames}),
                "story_families_with_projected_candidates": len(families_with_projected),
                "story_families_with_semantic_accepts": len(families_with_semantic_accept),
                "projected_population_ok": projected_total,
                "semantic_accepted": accepted_total,
                "semantic_budget_calls": self.semantic_budget_calls,
                "semantic_budget_max": self.semantic_budget_max,
                "dependency_verified": False,
                "dependency_review_mode": "fake_auto_confirm_projection",
                "cp5b_report_found": self.cp5b_report_found,
            },
            "frames": [frame.to_payload() for frame in self.frames],
            "semantic_errors": self.semantic_errors,
        }


def run_capacity_audit(
    *,
    settings: Settings,
    llm: ChatLLM | None = None,
    semantic_max_calls: int | None = None,
    semantic_candidate_limit: int | None = None,
) -> CapacityAuditResult:
    docs = scan_catalog(settings.data_root)
    company_meta = load_company_meta(settings.company_meta_path)
    cube = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir).load_or_build()
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)
    audit_cache = AuditCache(cache_dir=settings.cache_dir / "hard_cube_v4_cp6_capacity")
    dependency_auditor = FakeDependencyAuditor()

    frame_reports = [
        FrameCapacity(
            frame_id=frame.frame_id,
            story_family=frame.story_family,
            topology=frame.topology,
            domain_kind=frame.domain_kind,
            metric_keys=frame.metric_keys,
            metric_families=tuple(_metric_family(key) for key in frame.metric_keys),
            meaning=frame.meaning,
            finance_rationale=frame.finance_rationale or frame.meaning,
            interpretation_limits=frame.interpretation_limits,
        )
        for frame in FRAMES
    ]
    by_id = {frame.frame_id: frame for frame in frame_reports}
    candidates: list[RecipeCandidate] = []

    for frame in FRAMES:
        report = by_id[frame.frame_id]
        for entities, periods in _domains_for_frame(frame, cube, company_meta):
            report.raw_domains += 1
            try:
                attempt = _build_attempt(cube, frame, entities, periods)
            except CandidateRejected as exc:
                report.reject_reasons[_reason("build", exc)] += 1
                continue
            if attempt is None:
                report.reject_reasons["coverage_or_domain_gate"] += 1
                continue
            selected, graph = attempt
            report.build_ok += 1
            try:
                evaluate_graph(graph)
            except EvaluationError as exc:
                report.reject_reasons[_reason("preflight", exc)] += 1
                continue
            report.preflight_ok += 1
            try:
                graph, trace, compiled, answer, audit_stats = finalize_candidate(
                    provisional_graph=graph,
                    terminal_metric_key=frame.terminal_key,
                    docs_by_name=docs_by_name,
                    company_meta=company_meta,
                    table_ref_to_path=table_ref_to_path,
                    auditor=dependency_auditor,
                    audit_cache=audit_cache,
                    enforce_objective_gates=True,
                )
            except DependencyAuditRejected as exc:
                for reason, count in Counter(str(reason) for reason in exc.reasons.values()).items():
                    report.reject_reasons[f"dependency:{reason}"] += count
                continue
            except (EvaluationError, CandidateRejected) as exc:
                report.reject_reasons[_reason("finalize", exc)] += 1
                continue
            report.finalized_ok += 1
            report.projected_population_ok += 1
            _add_audit_stats(report.audit_stats, audit_stats)
            candidate_id = _candidate_id(frame.frame_id, selected, periods)
            public_spec = _public_spec(
                frame,
                candidate_id=candidate_id,
                entities=selected,
                periods=periods,
                company_meta=company_meta,
            )
            candidate = RecipeCandidate(
                candidate_id=candidate_id,
                recipe_id=frame.frame_id,
                entities=selected,
                graph=graph,
                trace=trace,
                compiled=compiled,
                formatted_answer=answer,
                audit_stats=audit_stats,
                public_spec=public_spec,
            )
            candidates.append(candidate)
            report.evidence_domains.add(f"{','.join(selected)}|{','.join(periods)}")
            report.evidence_tables.update(compiled.relevant_tables)

    semantic_errors: list[str] = []
    budget = LLMCallBudget(semantic_max_calls) if llm is not None and semantic_max_calls else None
    if llm is not None and budget is not None:
        _run_semantic_audit(
            _semantic_order(candidates, limit=semantic_candidate_limit),
            frame_reports=by_id,
            auditor=LLMSemanticAuditor(llm),
            budget=budget,
            errors=semantic_errors,
        )

    snapshot = budget.snapshot() if budget is not None else None
    return CapacityAuditResult(
        frames=frame_reports,
        candidates=candidates,
        semantic_errors=semantic_errors,
        semantic_budget_calls=snapshot.calls if snapshot is not None else 0,
        semantic_budget_max=snapshot.max_calls if snapshot is not None else semantic_max_calls,
        cp5b_report_found=Path("data/generated/hard_cube_v4_cp5b_report.md").exists(),
    )


def write_capacity_report(
    result: CapacityAuditResult,
    *,
    json_out: Path,
    md_out: Path,
) -> None:
    payload = result.to_payload()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: CapacityAuditResult) -> str:
    payload = result.to_payload()
    summary = payload["summary"]
    lines = [
        "# Hard Cube V4 CP6 Capacity Report",
        "",
        "## Summary",
        "",
        f"- Frames audited: {summary['frames']}",
        f"- Story families in catalog: {summary['story_families']}",
        f"- Story families with projected candidates: {summary['story_families_with_projected_candidates']}",
        f"- Projected population/finance candidates: {summary['projected_population_ok']}",
        f"- Semantic accepted among projected candidates: {summary['semantic_accepted']}",
        "- Real dependency audit verified: no (dependency REVIEW items are fake-auto-confirmed for projection)",
        f"- Semantic audit budget used: {summary['semantic_budget_calls']}/{summary['semantic_budget_max']}",
        f"- CP5B report found: {summary['cp5b_report_found']}",
        "",
        "Projected counts are measured after production finalize with objective runtime gates enabled,",
        "but FakeDependencyAuditor auto-confirms REVIEW items; they are not real dependency-accepted capacity.",
        "CP7 diversity quotas are not applied in this capacity report.",
        "",
        "## Catalog Matrix",
        "",
        "| Frame | Story family | Topology | Domain | Metrics | Projected | Semantic accept |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for frame in result.frames:
        lines.append(
            "| "
            + " | ".join(
                [
                    frame.frame_id,
                    frame.story_family,
                    frame.topology,
                    frame.domain_kind,
                    ", ".join(frame.metric_keys),
                    str(frame.projected_population_ok),
                    str(frame.semantic_accepted),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Stage Counts",
            "",
            "| Frame | Raw domains | Build OK | Preflight OK | Projected OK | Top reject reasons |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for frame in result.frames:
        top_reasons = ", ".join(f"{reason}={count}" for reason, count in frame.reject_reasons.most_common(4))
        lines.append(
            f"| {frame.frame_id} | {frame.raw_domains} | {frame.build_ok} | "
            f"{frame.preflight_ok} | {frame.projected_population_ok} | {top_reasons} |"
        )

    lines.extend(
        [
            "",
            "## Finance Rationale And Limits",
            "",
        ]
    )
    for frame in result.frames:
        lines.append(f"### {frame.frame_id}")
        lines.append("")
        lines.append(f"- Meaning: {frame.meaning}")
        lines.append(f"- Finance rationale: {frame.finance_rationale}")
        lines.append("- Interpretation limits: " + " / ".join(frame.interpretation_limits))
        lines.append(
            f"- Evidence: {len(frame.evidence_domains)} domain(s), "
            f"{len(frame.evidence_tables)} distinct table ref(s)."
        )
        lines.append("")

    if result.semantic_errors:
        lines.extend(["## Semantic Errors", ""])
        lines.extend(f"- {error}" for error in result.semantic_errors)
        lines.append("")

    lines.extend(
        [
            "## Blocker Check",
            "",
            _blocker_line(result),
            "",
        ]
    )
    return "\n".join(lines)


def _domains_for_frame(frame: AnalyticalFrame, cube: Any, company_meta: Any) -> Iterable[tuple[tuple[str, ...], tuple[str, ...]]]:
    if frame.domain_kind == "same_period":
        for _industry, entities, period in same_period_domains(cube, company_meta):
            yield entities, (period,)
    elif frame.domain_kind == "adjacent_period":
        for _industry, entities, prior, current in adjacent_period_domains(cube, company_meta):
            yield entities, (prior, current)
    elif frame.domain_kind == "single_entity_period_window":
        for _ticker, entities, periods in single_entity_period_window_domains(cube):
            yield entities, periods
    else:
        for _industry, entities, periods in cross_entity_period_window_domains(cube, company_meta):
            yield entities, periods


def _semantic_order(candidates: list[RecipeCandidate], *, limit: int | None) -> list[RecipeCandidate]:
    grouped: dict[str, dict[str, list[RecipeCandidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        grouped[candidate.public_spec.story_family or ""][candidate.recipe_id].append(candidate)
    ordered: list[RecipeCandidate] = []
    progressed = True
    while progressed and (limit is None or len(ordered) < limit):
        progressed = False
        for family in sorted(grouped):
            for frame_id in sorted(grouped[family]):
                bucket = grouped[family][frame_id]
                if not bucket:
                    continue
                ordered.append(bucket.pop(0))
                progressed = True
                if limit is not None and len(ordered) >= limit:
                    return ordered
    return ordered


def _run_semantic_audit(
    candidates: list[RecipeCandidate],
    *,
    frame_reports: dict[str, FrameCapacity],
    auditor: LLMSemanticAuditor,
    budget: LLMCallBudget,
    errors: list[str],
) -> None:
    idx = 0
    while idx < len(candidates):
        batch = candidates[idx : idx + MAX_SEMANTIC_AUDIT_BATCH_SIZE]
        idx += len(batch)
        try:
            budget.before_call(candidate="cp6_semantic_audit", stage="semantic_audit")
        except GenerationBudgetExceeded:
            break
        for candidate in batch:
            frame_reports[candidate.recipe_id].semantic_audited += 1
        try:
            assessments = auditor.audit(tuple(candidate.public_spec for candidate in batch))
        except SemanticAuditResponseError as exc:
            errors.append(str(exc))
            continue
        for assessment in assessments:
            candidate = next(c for c in batch if c.candidate_id == assessment.candidate_id)
            frame = frame_reports[candidate.recipe_id]
            if assessment.decision == "accept":
                frame.semantic_accepted += 1
            elif assessment.decision == "reject":
                frame.semantic_rejected += 1
            else:
                frame.semantic_needs_review += 1


def _metric_family(metric_key: str) -> str:
    ratio = get_ratio(metric_key)
    if ratio is not None:
        return f"ratio:{ratio.value_kind}"
    if metric_key in {"roa", "roe"}:
        return "average_balance_return_ratio"
    if metric_key.startswith("kqkd:"):
        return "income_statement"
    if metric_key.startswith("cdkt:"):
        return "balance_sheet"
    if metric_key.startswith("lctt:"):
        return "cash_flow_statement"
    if metric_name(metric_key) is not None:
        return "raw_statement_metric"
    return "derived_metric"


def _candidate_id(frame_id: str, entities: tuple[str, ...], periods: tuple[str, ...]) -> str:
    from vifinqa.generation.hard.recipe.grounded.analytical_planner import _signature

    return _signature(frame_id, entities, periods)


def _reason(stage: str, exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    head = text.split(":", 1)[0]
    return f"{stage}:{head[:120]}"


def _add_audit_stats(total: AuditStats, stats: AuditStats) -> None:
    total.dependencies_total += stats.dependencies_total
    total.cache_hits += stats.cache_hits
    total.deterministic_valid += stats.deterministic_valid
    total.deterministic_rejected += stats.deterministic_rejected
    total.review_sent_to_llm += stats.review_sent_to_llm
    total.llm_calls += stats.llm_calls
    total.llm_confirmed += stats.llm_confirmed
    total.llm_rejected += stats.llm_rejected
    total.period_deterministic_valid += stats.period_deterministic_valid
    total.period_deterministic_rejected += stats.period_deterministic_rejected
    total.period_review_sent += stats.period_review_sent
    total.period_cache_hits += stats.period_cache_hits
    for reason, count in stats.rejection_reasons.items():
        total.rejection_reasons[reason] = total.rejection_reasons.get(reason, 0) + count


def _audit_stats_payload(stats: AuditStats) -> dict[str, Any]:
    return {
        "dependencies_total": stats.dependencies_total,
        "cache_hits": stats.cache_hits,
        "deterministic_valid": stats.deterministic_valid,
        "deterministic_rejected": stats.deterministic_rejected,
        "review_sent_to_llm": stats.review_sent_to_llm,
        "llm_calls": stats.llm_calls,
        "llm_confirmed": stats.llm_confirmed,
        "llm_rejected": stats.llm_rejected,
        "period_deterministic_valid": stats.period_deterministic_valid,
        "period_deterministic_rejected": stats.period_deterministic_rejected,
        "period_review_sent": stats.period_review_sent,
        "period_cache_hits": stats.period_cache_hits,
        "rejection_reasons": dict(stats.rejection_reasons),
    }


def _blocker_line(result: CapacityAuditResult) -> str:
    projected = sum(frame.projected_population_ok for frame in result.frames)
    semantic = sum(frame.semantic_accepted for frame in result.frames)
    if result.semantic_budget_max in (None, 0):
        return (
            f"Capacity blocker: real dependency headroom is unmeasured. Deterministic projection is {projected}; "
            "semantic audit was not run."
        )
    return (
        f"Capacity blocker: real dependency headroom is unmeasured because REVIEW items used a fake auditor. "
        f"Semantic accepted {semantic} projected candidates; deterministic projection was {projected}."
    )
