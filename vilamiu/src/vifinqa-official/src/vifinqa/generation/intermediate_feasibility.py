
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import CompanyInfo, load_company_meta
from vifinqa.common.corpus.table import load_table
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import table_index_text
from vifinqa.generation.retrieval.time_series import enumerate_time_series_windows


@dataclass(frozen=True, slots=True)
class FormulaRoleCandidateSpec:
    formula_id: str
    role_keywords: dict[str, tuple[str, ...]]


FORMULA_CANDIDATES: tuple[FormulaRoleCandidateSpec, ...] = (
    FormulaRoleCandidateSpec(
        formula_id="roa",
        role_keywords={
            "net_income": ("lợi nhuận sau thuế",),
            "total_assets": ("tổng cộng tài sản", "tổng tài sản"),
        },
    ),
    FormulaRoleCandidateSpec(
        formula_id="roe",
        role_keywords={
            "net_income": ("lợi nhuận sau thuế",),
            "equity": ("vốn chủ sở hữu",),
        },
    ),
    FormulaRoleCandidateSpec(
        formula_id="quick_ratio",
        role_keywords={
            "current_assets": ("tài sản ngắn hạn",),
            "inventory": ("hàng tồn kho",),
            "current_liabilities": ("nợ ngắn hạn",),
        },
    ),
)


def _document_label_text(doc: DocumentRef) -> str:
    if doc.tables_dir is None or doc.text_path is None:
        return ""
    parts: list[str] = []
    for table_id in doc.table_ids:
        table = load_table(
            doc.table_csv_path(table_id),
            ticker=doc.ticker,
            year=doc.year,
            doc_name=doc.doc_name,
            table_id=table_id,
        )
        if not is_table_eligible(table):
            continue
        parts.append(table_index_text(table))
    return " \n ".join(parts).casefold()


def _formula_covered(text: str, spec: FormulaRoleCandidateSpec) -> bool:
    return all(any(kw in text for kw in keywords) for keywords in spec.role_keywords.values())


@dataclass(frozen=True, slots=True)
class FormulaFeasibility:
    formula_id: str
    documents_with_all_roles: int
    documents_total: int


def measure_formula_role_feasibility(docs: list[DocumentRef]) -> list[FormulaFeasibility]:
    texts = [_document_label_text(doc) for doc in docs]
    results: list[FormulaFeasibility] = []
    for spec in FORMULA_CANDIDATES:
        covered = sum(1 for text in texts if text and _formula_covered(text, spec))
        results.append(
            FormulaFeasibility(
                formula_id=spec.formula_id,
                documents_with_all_roles=covered,
                documents_total=len(docs),
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class RectangleFeasibility:
    industry_l3: str
    report_scope: str
    periods: tuple[str, ...]
    ticker_count: int
    tickers: tuple[str, ...]


def measure_rectangle_feasibility(
    docs: list[DocumentRef],
    companies: dict[str, CompanyInfo],
    *,
    period_count: int = 3,
    min_entities: int = 3,
) -> list[RectangleFeasibility]:
    windows = enumerate_time_series_windows(docs, period_count=period_count)
    by_group: dict[tuple[str, str, tuple[str, ...]], set[str]] = {}
    for window in windows:
        company = companies.get(window.ticker)
        if company is None or not company.industry_l3:
            continue
        key = (company.industry_l3, window.report_scope, window.periods)
        by_group.setdefault(key, set()).add(window.ticker)

    results: list[RectangleFeasibility] = []
    for (industry, scope, periods), tickers in sorted(by_group.items()):
        if len(tickers) >= min_entities:
            results.append(
                RectangleFeasibility(
                    industry_l3=industry,
                    report_scope=scope,
                    periods=periods,
                    ticker_count=len(tickers),
                    tickers=tuple(sorted(tickers)),
                )
            )
    return results


def _formula_payload(results: list[FormulaFeasibility]) -> list[dict]:
    return [
        {
            "formula_id": r.formula_id,
            "documents_with_all_roles": r.documents_with_all_roles,
            "documents_total": r.documents_total,
        }
        for r in results
    ]


def _rectangle_payload(results: list[RectangleFeasibility]) -> list[dict]:
    return [
        {
            "industry_l3": r.industry_l3,
            "report_scope": r.report_scope,
            "periods": list(r.periods),
            "ticker_count": r.ticker_count,
            "tickers": list(r.tickers),
        }
        for r in results
    ]


def render_markdown(
    formula_results: list[FormulaFeasibility], rectangle_results: list[RectangleFeasibility]
) -> str:
    lines = [
        "# CP0 — Intermediate new scenarios feasibility",
        "",
        "## Formula-role coverage (heuristic label match; not 100% accurate)",
        "",
    ]
    for r in formula_results:
        lines.append(
            f"- `{r.formula_id}`: {r.documents_with_all_roles}/{r.documents_total} documents "
            "contain all required role keywords."
        )
    lines.append("")
    lines.append("## 3x3 rectangle coverage (industry_l3 x report_scope x 3 consecutive years)")
    lines.append("")
    if not rectangle_results:
        lines.append("- No rectangle contains >=3 tickers in the same level-three industry.")
    for r in rectangle_results:
        tickers = ", ".join(r.tickers)
        lines.append(
            f"- {r.industry_l3} / {r.report_scope} / {list(r.periods)}: "
            f"{r.ticker_count} tickers ({tickers})"
        )
    return "\n".join(lines) + "\n"


def write_feasibility_report(
    *,
    data_root: Path,
    company_meta_path: Path,
    json_out: Path,
    md_out: Path,
) -> tuple[list[FormulaFeasibility], list[RectangleFeasibility]]:
    docs = scan_catalog(data_root)
    companies = load_company_meta(company_meta_path)
    formula_results = measure_formula_role_feasibility(docs)
    rectangle_results = measure_rectangle_feasibility(docs, companies)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "formula_candidates": _formula_payload(formula_results),
        "rectangles": _rectangle_payload(rectangle_results),
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(formula_results, rectangle_results), encoding="utf-8")

    return formula_results, rectangle_results
