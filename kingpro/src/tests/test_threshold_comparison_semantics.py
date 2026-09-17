from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_threshold_comparison_semantics import audit


def _candidate(tmp_path: Path, question: str, code: str) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "submission.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": question,
                    "answer": 1,
                    "pandas_query": code,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_ignores_parser_comparisons_and_accepts_strict_boundary(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Có bao nhiêu công ty có lợi nhuận lớn hơn 10 tỷ đồng?",
        """def parse(x):
    if len(x) <= 2:
        return 0
    return float(x)
result = df[df['profit'] > 10].shape[0]
""",
    )
    report = audit(root)
    assert report["finding_count"] == 0
    assert report["unresolved_count"] == 0


def test_flags_strict_operator_for_inclusive_lower_boundary(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Có bao nhiêu công ty có doanh thu ít nhất 10 tỷ đồng?",
        "result = df[df['revenue'] > 10].shape[0]",
    )
    report = audit(root)
    assert report["question_ids"] == [1]
    assert report["findings"][0]["contradictions"][0]["expected_operator"] == ">="


def test_accepts_pandas_inclusive_method(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Các công ty có tỷ lệ từ 1 lần trở lên có bình quân bao nhiêu?",
        "result = df.loc[df['ratio'].ge(1), 'value'].mean()",
    )
    report = audit(root)
    assert report["finding_count"] == 0


def test_flags_inclusive_operator_for_strict_upper_boundary(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Các năm có tỷ lệ thấp hơn mức trung vị đạt giá trị bao nhiêu?",
        "threshold = df['ratio'].median()\nresult = df[df['ratio'] <= threshold].value.mean()",
    )
    report = audit(root)
    assert report["question_ids"] == [1]
    assert report["findings"][0]["contradictions"][0]["expected_operator"] == "<"


def test_does_not_treat_minimum_rent_noun_as_threshold(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        "Tổng số tiền thuê tối thiểu trong tương lai là bao nhiêu?",
        "result = df['rent'].sum()",
    )
    report = audit(root)
    assert report["questions_with_threshold_language"] == 0
    assert report["finding_count"] == 0


def test_understands_equal_or_lower_compound_phrase(tmp_path: Path) -> None:
    root = _candidate(
        tmp_path,
        (
            "Nhóm cao hơn mức trung vị gấp bao nhiêu lần nhóm có tỷ lệ "
            "bằng hoặc thấp hơn mức trung vị?"
        ),
        (
            "med = df['ratio'].median()\n"
            "result = df[df['ratio'] > med].value.sum() / "
            "df[df['ratio'] <= med].value.sum()"
        ),
    )
    report = audit(root)
    assert report["finding_count"] == 0
