"""Offline hard-negative evaluation for financial row schema linking."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from ..finance.metrics import METRICS
from ..retrieval.shortlist import build_shortlist
from ..utils.viet_text import norm, tokens


@dataclass(frozen=True)
class RowLinkCase:
    case_id: str
    category: str
    metric_variants: tuple[str, ...]
    question: str
    rows: tuple[tuple[str, str], ...]
    expected_label: str
    years: tuple[int, ...] = ()
    columns: tuple[str, ...] = ("So cuoi nam",)
    expected_column: str = ""


def _canonical_parent_child_cases() -> list[RowLinkCase]:
    """Derive sibling cases whose labels share a parent/child token span."""
    atomic = [metric for metric in METRICS.values() if not metric.is_derived]
    cases = []
    for metric in atomic:
        wanted = set(tokens(metric.label))
        if len(wanted) < 2:
            continue
        siblings = []
        for other in atomic:
            if other.key == metric.key:
                continue
            have = set(tokens(other.label))
            overlap = len(wanted & have)
            if overlap < 2:
                continue
            if wanted <= have or have <= wanted or overlap / min(len(wanted), len(have)) >= 0.75:
                siblings.append(other)
        if not siblings:
            continue
        siblings.sort(key=lambda item: (-len(wanted & set(tokens(item.label))), item.key))
        options = [metric, *siblings[:5]]
        rows = tuple((item.label, item.codes[0] if item.codes else "") for item in options)
        cases.append(RowLinkCase(
            case_id=f"canonical:{metric.key}",
            category="canonical_parent_child",
            metric_variants=(metric.label,),
            question=metric.label,
            rows=rows,
            expected_label=metric.label,
        ))
    return cases


def _curated_cases() -> list[RowLinkCase]:
    return [
        RowLinkCase(
            case_id="parent_child:intangible_net",
            category="parent_child",
            metric_variants=("gia tri con lai cua tai san co dinh vo hinh",),
            question="Giá trị còn lại của tài sản cố định vô hình tổng cộng",
            rows=(("X Tài sản cố định", "220"),
                  ("Tài sản cố định hữu hình", "221"),
                  ("Tài sản cố định vô hình", "227")),
            expected_label="Tài sản cố định vô hình",
        ),
        RowLinkCase(
            case_id="parent_child:interbank_borrowing",
            category="parent_child",
            metric_variants=("vay cac TCTD khac",),
            question="Vay các TCTD khác cuối năm",
            rows=(("Tiền gửi và vay các TCTD khác", ""),
                  ("Tiền gửi của các TCTD khác", ""),
                  ("Vay các TCTD khác", "")),
            expected_label="Vay các TCTD khác",
        ),
        RowLinkCase(
            case_id="gross_net:investment_property_net",
            category="gross_net",
            metric_variants=("gia tri con lai cua bat dong san dau tu",),
            question="Giá trị còn lại của bất động sản đầu tư",
            rows=(("Bất động sản đầu tư", ""),
                  ("Nguyên giá bất động sản đầu tư", ""),
                  ("Giá trị còn lại của bất động sản đầu tư", "")),
            expected_label="Giá trị còn lại của bất động sản đầu tư",
        ),
        RowLinkCase(
            case_id="gross_net:investment_property_gross",
            category="gross_net",
            metric_variants=("nguyen gia bat dong san dau tu",),
            question="Nguyên giá bất động sản đầu tư",
            rows=(("Bất động sản đầu tư", ""),
                  ("Nguyên giá bất động sản đầu tư", ""),
                  ("Giá trị còn lại của bất động sản đầu tư", "")),
            expected_label="Nguyên giá bất động sản đầu tư",
        ),
        RowLinkCase(
            case_id="counterparty:hag_long_term_loan",
            category="counterparty",
            metric_variants=("vay dai han voi hoang anh gia lai",),
            question=("Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai "
                      "cuối năm 2017"),
            rows=(("Vay dài hạn", ""),
                  ("Vay dài hạn đến hạn trả", ""),
                  ("Công ty Cổ phần HoàngAnh Gia Lai Công ty mẹ Vay dài hạn", ""),
                  ("Công ty Cổ phần Thủyđiện Hoàng Anh Gia Lai Vay dài hạn", "")),
            expected_label="Công ty Cổ phần HoàngAnh Gia Lai Công ty mẹ Vay dài hạn",
            years=(2017,),
            columns=("31/12/2017",),
        ),
        RowLinkCase(
            case_id="counterparty:alpha_receivable",
            category="counterparty",
            metric_variants=("phai thu dai han voi cong ty alpha",),
            question="Phải thu dài hạn với Công ty Alpha cuối năm 2024",
            rows=(("Phải thu dài hạn", ""),
                  ("Công ty Alpha Phải thu dài hạn", ""),
                  ("Công ty Beta Phải thu dài hạn", "")),
            expected_label="Công ty Alpha Phải thu dài hạn",
            years=(2024,),
            columns=("31/12/2024",),
        ),
        RowLinkCase(
            case_id="counterparty:beta_payable",
            category="counterparty",
            metric_variants=("phai tra ngan han cho cong ty beta",),
            question="Phải trả ngắn hạn cho Công ty Beta cuối năm 2024",
            rows=(("Phải trả ngắn hạn", ""),
                  ("Công ty Alpha Phải trả ngắn hạn", ""),
                  ("Công ty Beta Phải trả ngắn hạn", "")),
            expected_label="Công ty Beta Phải trả ngắn hạn",
            years=(2024,),
            columns=("31/12/2024",),
        ),
        RowLinkCase(
            case_id="period:opening",
            category="opening_closing",
            metric_variants=("no ngan han",),
            question="Nợ ngắn hạn đầu năm 2024",
            rows=(("Nợ ngắn hạn", "310"),),
            expected_label="Nợ ngắn hạn",
            years=(2024,),
            columns=("Số cuối năm", "Số đầu năm"),
            expected_column="Số đầu năm",
        ),
        RowLinkCase(
            case_id="period:closing",
            category="opening_closing",
            metric_variants=("no ngan han",),
            question="Nợ ngắn hạn cuối năm 2024",
            rows=(("Nợ ngắn hạn", "310"),),
            expected_label="Nợ ngắn hạn",
            years=(2024,),
            columns=("Số cuối năm", "Số đầu năm"),
            expected_column="Số cuối năm",
        ),
    ]


def default_hard_negative_cases() -> list[RowLinkCase]:
    return [*_canonical_parent_child_cases(), *_curated_cases()]


def _table_for(case: RowLinkCase) -> dict:
    rows = []
    for row_i, (label, code) in enumerate(case.rows, start=1):
        for col_i, col_name in enumerate(case.columns, start=1):
            rows.append({
                "row": row_i,
                "label": label,
                "code": code,
                "col": col_i,
                "col_name": col_name,
                "value": float(row_i * 100 + col_i),
                "unit_scale": 1.0,
            })
    return {
        "var": "df1",
        "report_id": f"offline_{case.case_id}",
        "table_pos": 0,
        "report_year": case.years[0] if case.years else 2024,
        "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def evaluate_row_linking(cases: list[RowLinkCase]) -> dict:
    by_category: dict[str, list[int]] = defaultdict(list)
    details = []
    for case in cases:
        ranked = build_shortlist(
            [_table_for(case)],
            list(case.metric_variants),
            list(case.years),
            top_n=max(10, len(case.rows) * len(case.columns)),
            min_score=0.0,
            question=case.question,
        )
        rank = 0
        for index, candidate in enumerate(ranked, start=1):
            label_ok = norm(candidate.label) == norm(case.expected_label)
            column_ok = (not case.expected_column
                         or norm(candidate.col_name) == norm(case.expected_column))
            if label_ok and column_ok:
                rank = index
                break
        by_category[case.category].append(rank)
        details.append({
            "case_id": case.case_id,
            "category": case.category,
            "rank": rank,
            "expected_label": case.expected_label,
            "expected_column": case.expected_column,
            "top_label": ranked[0].label if ranked else "",
            "top_column": ranked[0].col_name if ranked else "",
        })

    def metrics(ranks: list[int]) -> dict:
        n = max(1, len(ranks))
        return {
            "n": len(ranks),
            "top1": round(sum(rank == 1 for rank in ranks) / n, 4),
            "mrr": round(sum(1.0 / rank for rank in ranks if rank) / n, 4),
            "recall5": round(sum(0 < rank <= 5 for rank in ranks) / n, 4),
        }

    all_ranks = [rank for ranks in by_category.values() for rank in ranks]
    return {
        "overall": metrics(all_ranks),
        "per_category": {
            category: metrics(ranks)
            for category, ranks in sorted(by_category.items())
        },
        "failures": [detail for detail in details if detail["rank"] != 1],
    }
