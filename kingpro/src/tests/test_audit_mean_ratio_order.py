from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_mean_ratio_order import audit


def _candidate(tmp_path: Path, question: str, code: str) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "submission.json").write_text(
        json.dumps(
            [{"id": 1, "question": question, "answer": 1, "pandas_query": code}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_flags_ratio_of_sums_for_mean_ratio_question(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Tỷ lệ lợi nhuận trên doanh thu trung bình của các công ty là bao nhiêu %?",
        "result = df['profit'].sum() / df['revenue'].sum() * 100",
    )
    report = audit(root)
    assert report["question_ids"] == [1]
    assert "sum" in report["findings"][0]["expressions"][0]


def test_accepts_mean_of_ratio_derived_dataframe_column(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Biên lợi nhuận gộp bình quân của các công ty là bao nhiêu phần trăm?",
        "df['gross_margin_pct'] = df['gross_profit'] / df['revenue'] * 100\n"
        "result = df['gross_margin_pct'].mean()",
    )
    report = audit(root)
    assert report["finding_count"] == 0
    assert report["unresolved_count"] == 0
    assert report["proven_question_ids"] == [1]


def test_accepts_explicit_arithmetic_mean_of_ratios(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Giá trị trung bình của tỷ lệ hao mòn trên nguyên giá là bao nhiêu %?",
        "result = round((abs(v0) / v3 + abs(v1) / v4 + abs(v2) / v5) / 3 * 100, 2)",
    )
    report = audit(root)
    assert report["finding_count"] == 0
    assert report["unresolved_count"] == 0


def test_excludes_average_balance_used_inside_single_company_ratio(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        (
            "ROA của VNM tính bằng lợi nhuận sau thuế chia cho tổng tài sản "
            "bình quân đầu và cuối kỳ là bao nhiêu %?"
        ),
        "result = npat / ((assets_open + assets_close) / 2) * 100",
    )
    report = audit(root)
    assert report["questions_with_mean_ratio_language"] == 0


def test_excludes_average_before_balance_noun(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "ROA trên trung bình tổng tài sản đầu và cuối kỳ là bao nhiêu %?",
        "result = npat / ((assets_open + assets_close) / 2) * 100",
    )
    report = audit(root)
    assert report["questions_with_mean_ratio_language"] == 0


def test_keeps_opaque_program_unresolved_instead_of_calling_it_wrong(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Tỷ lệ bình quân của nhóm là bao nhiêu %?",
        "result = precomputed_ratio",
    )
    report = audit(root)
    assert report["finding_count"] == 0
    assert report["unresolved_count"] == 1
