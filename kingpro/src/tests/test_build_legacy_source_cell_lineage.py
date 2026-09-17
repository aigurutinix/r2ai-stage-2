from scripts.build_legacy_source_cell_lineage import convert


def test_convert_scalar_legacy_read() -> None:
    payload = {
        "submission": "candidate",
        "records": [
            {
                "id": 1,
                "question": "Công ty mẹ là bao nhiêu?",
                "answer": 2.0,
                "terminal_reads": [
                    {
                        "source_row": 4,
                        "source_column": 2,
                        "source_table": "ABC_financial_statements_2024_separate|12",
                        "source_label": "Tổng cộng",
                        "raw": "2",
                        "csv": "data/x.csv",
                    }
                ],
            }
        ],
    }
    result = convert(payload)
    assert result["physical_cells_resolved"] == 1
    assert result["records"][0]["cells"][0]["row_idx"] == 4
    assert result["records"][0]["cells"][0]["col_idx"] == 2


def test_convert_skips_non_scalar_read() -> None:
    payload = {
        "records": [
            {
                "id": 2,
                "terminal_reads": [{"source_rows": [1, 2], "source_table": "D|3"}],
            }
        ]
    }
    result = convert(payload)
    assert result["records_with_scalar_cells"] == 0
    assert result["skipped_terminal_read_count"] == 1
