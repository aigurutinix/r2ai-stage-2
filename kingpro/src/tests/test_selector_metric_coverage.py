import csv
import json

from scripts.audit_selector_metric_coverage import audit


def test_cross_year_metric_key_comparison_is_opt_in(tmp_path):
    candidate = tmp_path / "candidate"
    data = candidate / "data"
    data.mkdir(parents=True)
    (candidate / "submission.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Năm nào có giá trị cao nhất?",
                    "answer": 2025,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (data / "q1_source_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ticker", "year", "metric_key"))
        writer.writeheader()
        writer.writerows(
            [
                {"ticker": "AAA", "year": "2024", "metric_key": "note:old-layout"},
                {"ticker": "AAA", "year": "2025", "metric_key": "note:new-layout"},
            ]
        )

    default_report = audit(candidate)
    research_report = audit(candidate, include_cross_year=True)

    assert default_report["finding_count"] == 0
    assert default_report["include_cross_year"] is False
    assert research_report["finding_count"] == 2
    assert {item["axis"] for item in research_report["findings"]} == {"years_within_company"}
