import csv
import json

from scripts.audit_unit_dimension_contract import audit_candidate, requested_currency_factor


def test_requested_currency_factor_does_not_parse_cong_ty_as_billion():
    assert requested_currency_factor("Công ty có bao nhiêu triệu đồng?") == 1e6
    assert requested_currency_factor("Công ty có bao nhiêu nghìn đồng?") == 1e3
    assert requested_currency_factor("Công ty có bao nhiêu tỷ đồng?") == 1e9
    assert requested_currency_factor("Công ty tăng bao nhiêu phần trăm?") is None


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _source_row(ticker, document, raw):
    return {
        "ticker": ticker,
        "year": "2025",
        "metric_key": "note:value",
        "raw": raw,
        "typed_factor": "1",
        "scale": "1",
        "source_table": f"{document}|2",
        "source_csv": "unused.csv",
        "row_idx": "1",
        "col_idx": "1",
    }


def _program(second_multiplier):
    return f"""\
df1 = list(dfs.values())[0]
v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])
v1 = _btc_number(df1.iloc[1]['raw'], df1.iloc[1]['typed_factor']) * float(df1.iloc[1]['scale'])
result = round((v0 + v1 * {second_multiplier}) / 1e6, 2)
"""


def test_operand_level_contract_distinguishes_fixed_and_missing_unit_conversion(tmp_path):
    candidate = tmp_path / "candidate"
    data_root = tmp_path / "financial_statements"
    base_document = "AAA_financial_statements_2025"
    million_document = "BBB_financial_statements_2025"
    for document, header in ((base_document, "VND"), (million_document, "Triệu VND")):
        folder = data_root / document
        folder.mkdir(parents=True)
        (folder / f"{document}_extracted.txt").write_text(
            f"Đơn vị tính: {header}\n<table><tr><td></td><td>{header}</td></tr></table>\n",
            encoding="utf-8",
        )

    records = []
    for qid, multiplier in ((1, "1"), (2, "1e6")):
        records.append(
            {
                "id": qid,
                "question": "Tổng hai giá trị là bao nhiêu triệu đồng?",
                "answer": 1.0,
                "pandas_query": _program(multiplier),
            }
        )
        _write_manifest(
            candidate / "data" / f"q{qid}_source_cells.csv",
            [
                _source_row("AAA", base_document, "1000000"),
                _source_row("BBB", million_document, "2"),
            ],
        )
    (candidate / "submission.json").write_text(json.dumps(records), encoding="utf-8")

    report = audit_candidate(candidate, data_root)

    assert report["eligible_programs_checked"] == 2
    assert report["question_ids"] == [1]
    assert report["findings"][0]["relative_unit_mismatch"] is True


def test_single_operand_output_scale_contract(tmp_path):
    candidate = tmp_path / "candidate"
    data_root = tmp_path / "financial_statements"
    document = "AAA_financial_statements_2025"
    folder = data_root / document
    folder.mkdir(parents=True)
    (folder / f"{document}_extracted.txt").write_text(
        "Đơn vị tính: VND\n<table><tr><td></td><td>VND</td></tr></table>\n",
        encoding="utf-8",
    )
    (candidate / "submission.json").parent.mkdir(parents=True, exist_ok=True)
    (candidate / "submission.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Giá trị là bao nhiêu tỷ đồng?",
                    "answer": 1000.0,
                    "pandas_query": "df1=list(dfs.values())[0]\nv0=_btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor'])*float(df1.iloc[0]['scale'])\nresult=v0/1e6",
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_manifest(
        candidate / "data" / "q1_source_cells.csv",
        [_source_row("AAA", document, "1000000000")],
    )

    report = audit_candidate(candidate, data_root)

    assert report["question_ids"] == [1]
    assert report["findings"][0]["output_scale_mismatch"] is True


def test_ocr_normalized_operand_is_not_a_false_unit_finding(tmp_path):
    candidate = tmp_path / "candidate"
    data_root = tmp_path / "financial_statements"
    document = "AAA_financial_statements_2025"
    folder = data_root / document
    folder.mkdir(parents=True)
    (folder / f"{document}_extracted.txt").write_text("VND\n", encoding="utf-8")
    candidate.mkdir(parents=True)
    (candidate / "submission.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Gia tri trung binh la bao nhieu trieu \u0111ong?",
                    "answer": 34.51,
                    "pandas_query": (
                        "df1=list(dfs.values())[0]\n"
                        "v0=_btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor'])*float(df1.iloc[0]['scale'])\n"
                        "result=(v0/100)"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    row = _source_row("AAA", document, "3451")
    row["metric_key"] = "note:average_employee_income_monthly_ocr"
    _write_manifest(candidate / "data" / "q1_source_cells.csv", [row])

    report = audit_candidate(candidate, data_root)

    assert report["finding_count"] == 0
    assert report["skipped"]["ocr_normalized_operand"] == 1
