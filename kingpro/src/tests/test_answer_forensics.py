import csv
import json
from pathlib import Path

from kingpro.forensics import AnswerForensicsEngine, detect_intents, parse_btc_number
from kingpro.forensics.engine import REVIEWED_VERDICTS, _manifest_features


def test_counterfactual_rejected_is_a_durable_review_verdict():
    assert "counterfactual_rejected" in REVIEWED_VERDICTS
    assert "source_confirmed_scope_repair" in REVIEWED_VERDICTS
    assert "source_confirmed_keep_baseline" in REVIEWED_VERDICTS
    assert "source_confirmed_no_change" in REVIEWED_VERDICTS
    assert "source_formula_confirmed" in REVIEWED_VERDICTS


def test_parse_btc_number_preserves_accounting_sign_and_locale():
    assert parse_btc_number("(1.234.567)") == -1234567.0
    assert parse_btc_number("12,34") == 12.34
    assert parse_btc_number("1,234.56") == 1234.56
    assert parse_btc_number("-") is None


def test_detect_intents_handles_vietnamese_diacritics():
    intents = detect_intents("Tỷ lệ tăng trưởng cao nhất và chênh lệch là bao nhiêu?")
    assert {"ratio", "growth", "extreme", "difference"}.issubset(set(intents))


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker",
        "year",
        "metric_key",
        "raw",
        "typed_factor",
        "scale",
        "source_table",
        "source_csv",
        "row_idx",
        "col_idx",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "ticker": "AAA",
                "year": 2024,
                "metric_key": "value:a",
                "raw": "100",
                "typed_factor": 1,
                "scale": 1,
                "source_table": "AAA_financial_statements_2024|10",
                "source_csv": "table.csv",
                "row_idx": 1,
                "col_idx": 1,
            }
        )
        writer.writerow(
            {
                "ticker": "AAA",
                "year": 2024,
                "metric_key": "value:b",
                "raw": "40",
                "typed_factor": 1,
                "scale": 1,
                "source_table": "AAA_financial_statements_2024|10",
                "source_csv": "table.csv",
                "row_idx": 2,
                "col_idx": 1,
            }
        )


def test_engine_combines_audits_and_excludes_durable_review(tmp_path):
    candidate = tmp_path / "sub_top123_candidate_v200_test"
    candidate.mkdir()
    rows = [
        {
            "id": 1,
            "question": "Tổng hai khoản là bao nhiêu?",
            "answer": 140.0,
            "pandas_query": "v0 = _source_value('AAA', 2024, 'value:a')\nv1 = _source_value('AAA', 2024, 'value:b')\nresult = v0 + v1",
            "relevant_docs": ["AAA_financial_statements_2024"],
            "relevant_tables": ["AAA_financial_statements_2024|10"],
        },
        {
            "id": 2,
            "question": "Giá trị là bao nhiêu?",
            "answer": 1.0,
            "pandas_query": "result = 1.0",
            "relevant_docs": [],
            "relevant_tables": [],
        },
    ]
    (candidate / "submission.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    _write_manifest(candidate / "data" / "q1_source_cells.csv")
    audit = tmp_path / "v200_period.json"
    audit.write_text(
        json.dumps(
            {
                "submission": str(candidate),
                "findings": [
                    {"id": 1, "high_priority": True, "reasons": ["period"]},
                    {"id": 1, "high_priority": False, "reasons": ["second row, same report"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"schema_version": 1, "reviews": [{"id": 2, "verdict": "hypothesis_rejected", "summary": "done"}]}),
        encoding="utf-8",
    )

    report = AnswerForensicsEngine(candidate, [audit], ledger, tmp_path, 200).run(top=10)

    assert report["top_ids"] == [1]
    assert report["counts"]["durably_reviewed"] == 1
    assert len(report["prioritized"][0]["audit_signals"]) == 1
    assert report["prioritized"][0]["audit_signals"][0]["stale"] is False
    assert any(item["operation"] == "signed_sum_all_cells" for item in report["prioritized"][0]["counterfactuals"])


def test_manifest_does_not_double_apply_typed_dtype_factor(tmp_path):
    path = tmp_path / "q3_source_cells.csv"
    _write_manifest(path)
    text = path.read_text(encoding="utf-8").replace("100,1,1,", "784.295,1000,1,")
    path.write_text(text, encoding="utf-8")

    features, writers = _manifest_features(path)

    assert features["distinct_scales"] == [1.0]
    assert writers[0]["scale"] == 1.0
    assert parse_btc_number(writers[0]["raw"]) * writers[0]["scale"] == 784295.0


def test_engine_excludes_trusted_vothuong_reviews(tmp_path):
    candidate = tmp_path / "sub_top123_candidate_v200_test"
    candidate.mkdir()
    rows = [
        {
            "id": 1,
            "question": "Tổng hai khoản là bao nhiêu?",
            "answer": 140.0,
            "pandas_query": "result = 100 + 40",
            "relevant_docs": [],
            "relevant_tables": [],
        }
    ]
    (candidate / "submission.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    audit = tmp_path / "v200_period.json"
    audit.write_text(
        json.dumps({"findings": [{"id": 1, "high_priority": True, "reasons": ["period"]}]}),
        encoding="utf-8",
    )
    review_log = tmp_path / "experiments.jsonl"
    review_log.write_text(
        json.dumps(
            {
                "kind": "review",
                "id": "R0001",
                "question_ids": [1],
                "verdict": "source_confirmed",
                "summary": "verified",
                "oracle_trust": "that",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = AnswerForensicsEngine(
        candidate,
        [audit],
        history_root=tmp_path,
        history_min_version=200,
        review_log_path=review_log,
    ).run(top=10)

    assert report["top_ids"] == []
    assert report["counts"]["durably_reviewed"] == 1
    assert report["reviewed"][0]["review"]["event_id"] == "R0001"


def test_engine_normalizes_terminal_review_verdicts(tmp_path):
    candidate = tmp_path / "sub_top123_candidate_v200_test"
    candidate.mkdir()
    rows = [
        {
            "id": qid,
            "question": "Câu đã được kiểm tra nguồn",
            "answer": 1.0,
            "pandas_query": "result = 1.0",
            "relevant_docs": [],
            "relevant_tables": [],
        }
        for qid in (1, 2, 3)
    ]
    (candidate / "submission.json").write_text(json.dumps(rows), encoding="utf-8")
    audit = tmp_path / "v200_period.json"
    audit.write_text(
        json.dumps({"findings": [{"id": 1}, {"id": 2}, {"id": 3}]}), encoding="utf-8"
    )
    review_log = tmp_path / "experiments.jsonl"
    review_log.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "review",
                    "id": "R{:04d}".format(qid),
                    "question_ids": [qid],
                    "verdict": verdict,
                    "summary": "verified",
                    "oracle_trust": "that",
                }
            )
            for qid, verdict in (
                (1, "source-confirmed-fix"),
                (2, "ambiguous_keep_baseline"),
                (3, "false_positive"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = AnswerForensicsEngine(
        candidate,
        [audit],
        history_root=tmp_path,
        history_min_version=200,
        review_log_path=review_log,
    ).run(top=10)

    assert report["top_ids"] == []
    assert report["counts"]["durably_reviewed"] == 3
