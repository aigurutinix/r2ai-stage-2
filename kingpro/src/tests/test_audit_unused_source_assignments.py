import json
from pathlib import Path

from scripts.audit_unused_source_assignments import audit


def _candidate(tmp_path: Path, queries: list[str]) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    rows = [
        {"id": index + 1, "question": "test", "answer": 0, "pandas_query": query}
        for index, query in enumerate(queries)
    ]
    (candidate / "submission.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    return candidate


def test_used_source_assignment_is_not_flagged(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "a = _source_value('AAA', 2022, 'revenue')\n"
                "b = _source_value('BBB', 2022, 'revenue')\n"
                "result = a + b"
            ],
        )
    )
    assert report["finding_count"] == 0


def test_omitted_source_assignment_is_flagged(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "a = _source_value('AAA', 2022, 'revenue')\n"
                "b = _source_value('BBB', 2022, 'revenue')\n"
                "result = a"
            ],
        )
    )
    assert report["question_ids"] == [1]
    assert report["findings"][0]["unused_bindings"][0]["binding"] == "b"


def test_all_branch_assignments_are_followed(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "if condition:\n"
                "    selected = _source_value('AAA', 2022, 'revenue')\n"
                "else:\n"
                "    selected = _source_value('BBB', 2022, 'revenue')\n"
                "result = selected"
            ],
        )
    )
    assert report["finding_count"] == 0


def test_panel_container_is_not_misclassified_as_scalar(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "_rows = [{'ticker': 'AAA', 'value': "
                "_source_value('AAA', 2022, 'revenue')}]\n"
                "df = pd.DataFrame(_rows)\n"
                "result = float(df['value'].sum())"
            ],
        )
    )
    assert report["source_binding_count"] == 0
    assert report["finding_count"] == 0


def test_omitted_dataframe_scalar_is_flagged(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "v0 = _btc_number(df1.iloc[0]['raw'])\n"
                "v1 = _btc_number(df1.iloc[1]['raw'])\n"
                "result = v0"
            ],
        )
    )
    assert report["question_ids"] == [1]
    assert report["findings"][0]["unused_bindings"][0]["binding"] == "v1"
    assert report["findings"][0]["unused_bindings"][0]["evidence_reads"][0]["kind"] == "dataframe_scalar"


def test_source_used_in_result_guard_is_not_flagged(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "check = _btc_number(df1.iloc[0]['raw'])\n"
                "value = _btc_number(df1.iloc[1]['raw'])\n"
                "if check == value:\n"
                "    result = value\n"
                "else:\n"
                "    result = None"
            ],
        )
    )
    assert report["finding_count"] == 0


def test_named_recall_crosscheck_is_excluded(tmp_path: Path) -> None:
    report = audit(
        _candidate(
            tmp_path,
            [
                "_recall_source = _btc_number(df1.iloc[0]['raw'])\n"
                "value = _btc_number(df1.iloc[1]['raw'])\n"
                "result = value"
            ],
        )
    )
    assert report["diagnostic_bindings_excluded"] == 1
    assert report["finding_count"] == 0
