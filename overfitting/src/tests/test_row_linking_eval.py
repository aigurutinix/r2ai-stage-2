from vifinqa.validation.row_linking_eval import (
    RowLinkCase,
    default_hard_negative_cases,
    evaluate_row_linking,
)


def test_default_suite_is_batch_sized_and_covers_hard_negative_families():
    cases = default_hard_negative_cases()

    assert len(cases) >= 20
    assert {case.category for case in cases} >= {
        "canonical_parent_child",
        "parent_child",
        "gross_net",
        "counterparty",
        "opening_closing",
    }


def test_eval_reports_rank_metrics_and_failure_details():
    cases = [
        RowLinkCase(
            case_id="easy",
            category="test",
            metric_variants=("doanh thu thuan",),
            question="Doanh thu thuần năm 2024",
            rows=(("Doanh thu thuần", "10"), ("Chi phí khác", "32")),
            expected_label="Doanh thu thuần",
            years=(2024,),
            columns=("Năm 2024",),
        ),
        RowLinkCase(
            case_id="missing",
            category="test",
            metric_variants=("doanh thu thuan",),
            question="Doanh thu thuần năm 2024",
            rows=(("Doanh thu thuần", "10"),),
            expected_label="Không tồn tại",
            years=(2024,),
            columns=("Năm 2024",),
        ),
    ]

    report = evaluate_row_linking(cases)

    assert report["overall"] == {
        "n": 2, "top1": 0.5, "mrr": 0.5, "recall5": 0.5,
    }
    assert [failure["case_id"] for failure in report["failures"]] == ["missing"]
