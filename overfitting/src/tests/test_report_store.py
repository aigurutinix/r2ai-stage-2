from pathlib import Path

import pandas as pd

from vifinqa.extraction.build_store import Store
from vifinqa.extraction.report_parser import detect_unit, report_meta_from_path


def _report_path(root: Path, ticker: str, year: int, report_id: str) -> Path:
    return root / ticker / str(year) / report_id / f"{report_id}_extracted.txt"


def test_report_doc_type_uses_name_tokens(tmp_path):
    cases = {
        "VPB_financial_statements_2022_separate_1": "separate",
        "VPB_financial_statements_2022_consolidated": "consolidated",
        "VSF_financial_statements_2024_aggregated": "aggregated",
        "PRT_2023_explanatory_letters_1": "other",
    }
    for report_id, expected in cases.items():
        path = _report_path(tmp_path, "VPB", 2022, report_id)
        assert report_meta_from_path(path)["doc_type"] == expected


def test_detects_hundred_billion_table_units():
    assert detect_unit("Đơn vị tính: trăm tỷ đồng") == (1e11, "explicit")
    assert detect_unit("", "Trăm tỷ VND") == (1e11, "header")


def test_store_preserves_one_to_many_report_index_without_breaking_find_report(tmp_path):
    reports = pd.DataFrame(
        [
            {
                "report_id": "VPB_financial_statements_2022_separate_1",
                "ticker": "VPB",
                "year": 2022,
                "doc_type": "separate",
                "path": "a",
                "n_tables": 1,
            },
            {
                "report_id": "VPB_financial_statements_2022_separate_2",
                "ticker": "VPB",
                "year": 2022,
                "doc_type": "separate",
                "path": "b",
                "n_tables": 1,
            },
            {
                "report_id": "VPB_financial_statements_2023_aggregated",
                "ticker": "VPB",
                "year": 2023,
                "doc_type": "aggregated",
                "path": "c",
                "n_tables": 1,
            },
        ]
    )
    reports.to_parquet(tmp_path / "reports.parquet", index=False)
    store = Store(tmp_path)

    expected = [
        "VPB_financial_statements_2022_separate_1",
        "VPB_financial_statements_2022_separate_2",
    ]
    assert store.find_reports("VPB", 2022, "separate", allow_fallback=False) == expected
    # Existing callers still get one deterministic string, matching old last-row wins.
    assert store.find_report("VPB", 2022, "separate") == expected[-1]
    # Unlabelled/aggregated reports remain reachable when a conventional type is absent.
    assert store.find_reports("VPB", 2023, "consolidated") == [
        "VPB_financial_statements_2023_aggregated"
    ]


def test_store_repairs_explicit_doc_type_in_stale_parquet_metadata(tmp_path):
    reports = pd.DataFrame(
        [
            {
                "report_id": "MBB_financial_statements_2022_separate_1",
                "ticker": "MBB",
                "year": 2022,
                "doc_type": "consolidated",
                "path": "a",
                "n_tables": 1,
            },
            {
                "report_id": "ACV_financial_statements_2022_aggregated",
                "ticker": "ACV",
                "year": 2022,
                "doc_type": "consolidated",
                "path": "b",
                "n_tables": 1,
            },
        ]
    )
    reports.to_parquet(tmp_path / "reports.parquet", index=False)

    store = Store(tmp_path)

    assert store.find_reports(
        "MBB", 2022, "separate", allow_fallback=False
    ) == ["MBB_financial_statements_2022_separate_1"]
    assert store.find_reports(
        "ACV", 2022, "aggregated", allow_fallback=False
    ) == ["ACV_financial_statements_2022_aggregated"]
