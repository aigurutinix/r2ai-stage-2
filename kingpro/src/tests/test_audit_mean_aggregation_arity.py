import csv
import json

from scripts.audit_mean_aggregation_arity import audit


def _manifest(path, tickers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "year", "raw"])
        writer.writeheader()
        for index, ticker in enumerate(tickers, 1):
            writer.writerow({"ticker": ticker, "year": "2024", "raw": str(index)})


def test_mean_arity_flags_only_wrong_output_divisor(tmp_path):
    candidate = tmp_path / "candidate"
    (candidate / "data").mkdir(parents=True)
    rows = [
        {
            "id": 1,
            "question": "Giá trị trung bình của A, B và C là bao nhiêu?",
            "answer": 2.0,
            "pandas_query": "result = (v0 + v1 + v2) / 3",
        },
        {
            "id": 2,
            "question": "Giá trị trung bình của A, B và C là bao nhiêu?",
            "answer": 3.0,
            "pandas_query": "result = (v0 + v1 + v2) / 2",
        },
        {
            "id": 3,
            "question": "ROA trên tổng tài sản bình quân của A là bao nhiêu?",
            "answer": 1.0,
            "pandas_query": "result = profit / ((opening + closing) / 2)",
        },
    ]
    (candidate / "submission.json").write_text(json.dumps(rows), encoding="utf-8")
    for qid in (1, 2):
        _manifest(candidate / "data" / "q{}_source_cells.csv".format(qid), ["A", "B", "C"])
    _manifest(candidate / "data" / "q3_source_cells.csv", ["A", "A"])

    report = audit(candidate)

    assert report["question_ids"] == [2]
    assert report["findings"][0]["source_group_count"] == 3
    assert report["findings"][0]["small_constant_divisors"] == [2.0]
