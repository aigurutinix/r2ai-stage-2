"""End-to-end contracts for the deterministic submission path."""

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports, parse_question, retrieve_docs
from validate_submission import validate


def make_dataset(root: Path) -> Path:
    dataset = root / "vifinqa"
    (dataset / "questions").mkdir(parents=True)
    (dataset / "code_stock.csv").write_text("ticker,name\nABC,Công ty ABC\n", encoding="utf-8")
    (dataset / "questions" / "questions.jsonl").write_text(
        json.dumps({"id": 1, "question": "Doanh thu ABC năm 2024 là bao nhiêu?"}) + "\n",
        encoding="utf-8",
    )
    report_id = "ABC_financial_statements_2024_consolidated"
    source = dataset / "financial_statements" / "ABC" / "2024" / report_id
    source.mkdir(parents=True)
    (source / f"{report_id}_extracted.txt").write_text(
        "Báo cáo kết quả kinh doanh\nĐơn vị: triệu VND\n"
        "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr>"
        "<tr><td>Doanh thu</td><td>42</td></tr></table>\n",
        encoding="utf-8",
    )
    return dataset


def test_document_gate_selects_matching_report(tmp_path):
    dataset = make_dataset(tmp_path)
    parsed = parse_question("Doanh thu ABC năm 2024 là bao nhiêu?", load_companies(dataset / "code_stock.csv"))
    docs, _ = retrieve_docs(parsed, load_reports(dataset / "financial_statements"))
    assert docs == ["ABC_financial_statements_2024_consolidated"]


def test_submission_command_runs_strict_gate_and_creates_valid_zip(tmp_path):
    dataset = make_dataset(tmp_path)
    output = tmp_path / "output"
    completed = subprocess.run(
        [sys.executable, "run.py", "--data-root", str(dataset), "--output-dir", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert completed.stdout.strip().endswith("submission.zip")
    package = output / "package"
    assert validate(package, {1}, {"ABC_financial_statements_2024_consolidated|3"}) == []
    with ZipFile(output / "submission.zip") as archive:
        assert "submission.json" in archive.namelist()
        assert "data/tables/table_1_0.csv" in archive.namelist()


def test_validator_rejects_duplicate_tables_and_missing_evidence(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "submission.json").write_text(json.dumps([{
        "id": 1, "question": "q", "answer": 0.0, "relevant_docs": ["R"],
        "relevant_tables": ["R|1", "R|1"],
        "evidence": [{"variable": "df0", "csv_path": "data/missing.csv"}],
        "pandas_query": "result = df0",
    }]), encoding="utf-8")
    errors = validate(package, {1}, {"R|1"})
    assert any("missing evidence file" in error for error in errors)
