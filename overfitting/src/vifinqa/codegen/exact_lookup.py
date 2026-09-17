"""Fail-closed canonical challenger for successful single-cell lookups."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import (
    extract_metric_qualifiers,
    get_metric,
    metric_keys,
    metric_uses_absolute_value,
)
from ..utils.viet_text import norm
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import cell_is_already_percent, check_answer_unit, percent_from_cell


_OPENING_PERIOD_MARKERS = (
    "dau nam", "dau ky", "ngay 01 thang 01",
)
_DETAIL_MARKERS = (
    "tu ngan hang", "tu cong ty", "den cong ty", "thuoc chi phi", "ben lien quan",
    "khau hao", "ngoai te", "lai tien gui", "co dong pho thong",
    "han muc", "lai du thu", "nguyen vat lieu", "du an", "gia goc",
    "cho vay ca nhan", "noi dia", "phai thu ve cho vay",
    "du phong rui ro", "thue hoat dong ngan han",
    "vay dai han", "no vay dai han", "trich lap", "den han trong nam",
    "nguyen lieu", "vat lieu va cong cu", "so du vay ngan hang",
    "thuong mai dich vu", "du phong chung", "nguyen gia",
    "vo hinh", "nguoi ban", "khu vuc",
    "phai tra", "co ky han", "hoat dong xay dung",
)


@dataclass
class ExactLookupAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_lookup_answer(route: dict, tables: list[dict]) -> ExactLookupAnswer:
    """Build a shadow challenger only for one canonical, non-derived fact.

    This is intentionally stricter than the normal lookup fallback. It is used
    to challenge already-successful LLM rows, where a false replacement is more
    costly than refusing to emit a candidate.
    """
    plan = route.get("plan") or {}
    if plan.get("op", "lookup") != "lookup":
        return ExactLookupAnswer(False, detail="not a lookup route")
    output_type = str(route.get("output_type") or "number")
    if output_type not in {"number", "percent"}:
        return ExactLookupAnswer(
            False, detail=f"unsupported lookup output={output_type}")

    tickers = list(dict.fromkeys(route.get("tickers") or []))
    years = list(dict.fromkeys(route.get("years") or []))
    if len(tickers) != 1 or len(years) != 1:
        return ExactLookupAnswer(
            False, detail=f"lookup scope entities={len(tickers)} years={len(years)}")

    requirements = route.get("evidence_requirements") or []
    requirement_source = "route"
    if not requirements:
        inferred = _infer_single_requirement(route, tickers[0], years[0])
        if inferred is not None:
            requirements = [inferred]
            requirement_source = "question"
    if len(requirements) != 1:
        return ExactLookupAnswer(
            False, detail=f"canonical requirements={len(requirements)}")
    requirement = requirements[0]
    metric_key = str(requirement.get("metric_key") or "")
    if not metric_key:
        return ExactLookupAnswer(False, detail="canonical metric missing")
    metric = get_metric(metric_key)
    if metric.components:
        return ExactLookupAnswer(
            False, detail=f"derived metric components={len(metric.components)}")

    scope_error = exact_metric_scope_error(route.get("question", ""), metric)
    if scope_error:
        return ExactLookupAnswer(False, detail=scope_error)

    found = resolve_requirement(
        requirement, tables,
        question=str(requirement.get("metric_label") or metric.label),
    )
    if found is None:
        return ExactLookupAnswer(False, detail="exact canonical cell unresolved")

    raw_expr = found.expr()
    if output_type == "percent":
        answer = percent_from_cell(found.value, found.label, found.col_name)
        scale_expr = ("" if cell_is_already_percent(
            found.label, found.col_name, found.value) else " * 100")
        answer_expr = f"{raw_expr}{scale_expr}"
    else:
        absolute = metric_uses_absolute_value(
            " ".join((str(route.get("question") or ""), metric.label)),
            (metric_key,),
        )
        value_vnd = abs(found.value_vnd) if absolute else found.value_vnd
        answer = value_vnd / float(route.get("unit_scale", 1.0) or 1.0)
        if absolute:
            raw_expr = f"abs({raw_expr})"
        answer_expr = (
            f"({raw_expr} * {found.unit_scale:g} / "
            f"{float(route.get('unit_scale', 1.0) or 1.0):g})"
        )

    answer = round(float(answer), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactLookupAnswer(False, detail=f"unit guard: {warning}")
    question_scale = float(route.get("unit_scale", 1.0) or 1.0)
    if output_type == "number" and question_scale >= 1e6 and abs(answer) > 1e10:
        return ExactLookupAnswer(False, detail="unit guard: implausible scaled amount")

    report_year = _report_year(found.report_id)
    target_year = int(requirement["year"])
    code = re.sub(r"\.0$", "", str(found.code or "").strip())
    expected_codes = set(metric.codes)
    vas_exact = bool(expected_codes and code in expected_codes)
    if vas_exact and report_year == target_year:
        tier, confidence = "vas_current", 99.0
    elif vas_exact and report_year == target_year + 1:
        tier, confidence = "vas_prior", 97.0
    else:
        tier, confidence = "note_exact", 94.0

    query = f"round({answer_expr}, 2)"
    return ExactLookupAnswer(
        True, answer, query, confidence,
        detail=(f"exact_lookup metric={metric_key} tier={tier} "
                f"requirement={requirement_source} "
                f"cell={found.report_id}|{found.table_pos}|r{found.row}c{found.col}"),
        tier=tier, resolved=[found],
    )


def _infer_single_requirement(route: dict, ticker: str, year: int) -> dict | None:
    """Recover one atomic canonical lookup from the literal question text."""
    keys = metric_keys([str(route.get("question") or "")], expand_derived=False)
    if len(keys) != 1:
        return None
    metric = get_metric(keys[0])
    if metric.components:
        return None
    return {
        "requirement_id": f"{ticker}|{year}|{metric.key}",
        "ticker": ticker,
        "year": year,
        "doc_type": str(route.get("doc_type") or ""),
        "metric_key": metric.key,
        "metric_label": metric.label,
        "metric_variants": list(metric.variants),
        "statement": metric.statement,
    }


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None


def exact_metric_scope_error(question: str, metric) -> str:
    """Reject periods or child qualifiers absent from a canonical metric."""
    question_norm = norm(str(question or ""))
    opening_date = re.search(
        r"(?<!\d)0?1\s*/\s*0?1(?:\s*/\s*20\d{2}|(?!\d))",
        question_norm,
    )
    if (any(marker in question_norm for marker in _OPENING_PERIOD_MARKERS)
            or opening_date):
        return "opening-period lookup is ambiguous"
    unsupported_detail = _unsupported_detail_marker(question_norm, metric)
    if unsupported_detail:
        return f"canonical metric misses detail={unsupported_detail}"
    asked = extract_metric_qualifiers(
        question_norm, include_defaults=False)
    for field in ("gross_net", "maturity"):
        wanted = getattr(asked, field)
        canonical = getattr(metric.qualifiers, field)
        if wanted and canonical != wanted:
            return (f"canonical metric qualifier mismatch "
                    f"{field}={wanted}/{canonical or 'missing'}")
    return ""


def _unsupported_detail_marker(question_norm: str, metric) -> str:
    """Return a child-detail phrase not represented by the canonical metric."""
    metric_text = " ".join((
        metric.label, *metric.aliases, *metric.row_aliases,
        *metric.qualifier_phrases, *metric.context_phrases,
    ))
    for marker in _DETAIL_MARKERS:
        if (_contains_phrase(question_norm, marker)
                and not _contains_phrase(metric_text, marker)):
            return marker
    return ""


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))
