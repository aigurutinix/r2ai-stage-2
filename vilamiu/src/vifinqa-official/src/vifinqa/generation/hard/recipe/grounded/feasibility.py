
from __future__ import annotations

from dataclasses import dataclass, field

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe import operations  # noqa: F401 — side-effect: register operations
from vifinqa.generation.hard.recipe.base import GraphError
from vifinqa.generation.hard.recipe.evaluator import EvaluationError, evaluate_graph
from vifinqa.generation.hard.recipe.grounded.domains import adjacent_period_domains, same_period_domains
from vifinqa.generation.hard.recipe.grounded.planner import GROUNDED_ATTEMPT_BUILDERS, GroundedDomainAttempt
from vifinqa.generation.hard.recipe.grounded.recipes import GROUNDED_RECIPE_IDS
from vifinqa.generation.panel.base import Cube

MIN_FINAL_CANDIDATES_FOR_PRODUCTION = 8

_REASON_INSUFFICIENT_COVERAGE = "insufficient_coverage"
_REASON_CONDITION_GATE_NON_TRIVIAL = "condition_gate_non_trivial"
_REASON_TIE = "tie_no_unique_winner"
_REASON_OTHER_EVALUATION_ERROR = "other_evaluation_error"
_REASON_GRAPH_ERROR = "graph_error"


@dataclass(frozen=True, slots=True)
class GroundedFeasibilityReport:
    intent_id: str
    domain_basis: str  # "same_period" | "adjacent_period"
    total_domains: int
    coverage_ok: int
    condition_gate_pass: int
    unique_winner_pass: int
    final_candidates: int
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.final_candidates >= MIN_FINAL_CANDIDATES_FOR_PRODUCTION


def _classify_evaluation_error(exc: EvaluationError) -> str:
    msg = str(exc)
    if "no unique winner" in msg:
        return _REASON_TIE
    if "filter output is empty" in msg or "equals the entire universe" in msg or "requires >=" in msg:
        return _REASON_CONDITION_GATE_NON_TRIVIAL
    return _REASON_OTHER_EVALUATION_ERROR


def audit_grounded_feasibility(intent_id: str, cube: Cube, company_meta: dict[str, CompanyInfo]) -> GroundedFeasibilityReport:
    domain_basis, attempt_fn = GROUNDED_ATTEMPT_BUILDERS[intent_id]
    if domain_basis == "same_period":
        domains = same_period_domains(cube, company_meta)
        attempts: list[GroundedDomainAttempt] = [
            attempt_fn(cube, industry, entities, period) for industry, entities, period in domains
        ]
    else:
        domains = adjacent_period_domains(cube, company_meta)
        attempts = [
            attempt_fn(cube, industry, entities, prior, current) for industry, entities, prior, current in domains
        ]

    reasons: dict[str, int] = {}
    coverage_ok = 0
    final_candidates = 0
    for attempt in attempts:
        if attempt.graph is None:
            reasons[_REASON_INSUFFICIENT_COVERAGE] = reasons.get(_REASON_INSUFFICIENT_COVERAGE, 0) + 1
            continue
        coverage_ok += 1
        try:
            evaluate_graph(attempt.graph)
        except EvaluationError as exc:
            reason = _classify_evaluation_error(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        except GraphError:
            reasons[_REASON_GRAPH_ERROR] = reasons.get(_REASON_GRAPH_ERROR, 0) + 1
            continue
        final_candidates += 1

    condition_gate_pass = coverage_ok - reasons.get(_REASON_CONDITION_GATE_NON_TRIVIAL, 0) - reasons.get(_REASON_GRAPH_ERROR, 0)
    return GroundedFeasibilityReport(
        intent_id=intent_id,
        domain_basis=domain_basis,
        total_domains=len(attempts),
        coverage_ok=coverage_ok,
        condition_gate_pass=condition_gate_pass,
        unique_winner_pass=final_candidates,
        final_candidates=final_candidates,
        reject_reasons=reasons,
    )


def audit_all_grounded(cube: Cube, company_meta: dict[str, CompanyInfo]) -> tuple[GroundedFeasibilityReport, ...]:
    return tuple(audit_grounded_feasibility(intent_id, cube, company_meta) for intent_id in GROUNDED_RECIPE_IDS)


def enabled_recipe_ids(reports: tuple[GroundedFeasibilityReport, ...]) -> tuple[str, ...]:
    return tuple(r.intent_id for r in reports if r.enabled)


def build_report_payload(reports: tuple[GroundedFeasibilityReport, ...]) -> dict:
    return {
        "min_final_candidates_for_production": MIN_FINAL_CANDIDATES_FOR_PRODUCTION,
        "recipes": [
            {
                "intent_id": r.intent_id,
                "domain_basis": r.domain_basis,
                "total_domains": r.total_domains,
                "coverage_ok": r.coverage_ok,
                "condition_gate_pass": r.condition_gate_pass,
                "unique_winner_pass": r.unique_winner_pass,
                "final_candidates": r.final_candidates,
                "enabled": r.enabled,
                "reject_reasons": dict(sorted(r.reject_reasons.items())),
            }
            for r in reports
        ],
    }


def render_markdown(reports: tuple[GroundedFeasibilityReport, ...]) -> str:
    lines = [
        "# Hard Cube — Grounded Recipe Feasibility (Step 1, engineering inventory gate)",
        "",
        f"Production registration threshold: >= {MIN_FINAL_CANDIDATES_FOR_PRODUCTION} final candidates "
        "(an engineering threshold, not a financial threshold). Domain: complete industry_l2 groups "
        "excluding credit institutions, with no industry_l1 fallback and at least three companies.",
        "",
        "| Intent | Domain basis | Total domains | Coverage met | Condition gate passed | Unique winner | Final candidates | Enabled |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.intent_id} | {r.domain_basis} | {r.total_domains} | {r.coverage_ok} | "
            f"{r.condition_gate_pass} | {r.unique_winner_pass} | {r.final_candidates} | "
            f"{'✅' if r.enabled else '❌'} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## {r.intent_id}")
        lines.append("")
        lines.append(f"- Domain basis: `{r.domain_basis}`; total domains evaluated: {r.total_domains}")
        lines.append(f"- Coverage met (>=3 entities pass the denominator gate): {r.coverage_ok}")
        lines.append(f"- Filter condition/non-triviality gate passed: {r.condition_gate_pass}")
        lines.append(f"- Unique winner (no tie): {r.unique_winner_pass}")
        lines.append(f"- Final candidates: {r.final_candidates}")
        lines.append(f"- Enabled: {'YES' if r.enabled else 'NO'}")
        if r.reject_reasons:
            reasons_text = ", ".join(f"{k}={v}" for k, v in sorted(r.reject_reasons.items()))
            lines.append(f"- Rejection reasons: {reasons_text}")
        lines.append("")
    return "\n".join(lines) + "\n"
