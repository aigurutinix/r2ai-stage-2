from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.report_scope import physical_scope
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.retrieval.bm25_index import extract_all_facets


def test_physical_masthead_overrides_swapped_container_name() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report_id = "ABC_financial_statements_2024_consolidated"
        report = (
            root / "data" / "financial_statements" / "ABC" / "2024"
            / report_id / f"{report_id}_extracted.txt"
        )
        report.parent.mkdir(parents=True)
        report.write_text(
            "CÔNG TY ABC\nBẢNG CÂN ĐỐI KẾ TOÁN RIÊNG\n"
            "Tại ngày 31 tháng 12 năm 2024\n\n<table>...</table>\n",
            encoding="utf-8",
        )
        scope, evidence = physical_scope(
            {
                "report_id": report_id,
                "ticker": "ABC",
                "year": "2024",
                "line": 5,
            },
            root,
        )
        assert scope == "separate"
        assert evidence is not None
        assert evidence["container_scope"] == "consolidated"
        assert evidence["overrides_container"] is True


def test_missing_physical_report_falls_back_to_container_scope() -> None:
    with tempfile.TemporaryDirectory() as directory:
        scope, evidence = physical_scope(
            {
                "report_id": "ABC_financial_statements_2024_separate",
                "ticker": "ABC",
                "year": "2024",
                "line": 1,
            },
            directory,
        )
        assert scope == "separate"
        assert evidence is None


def test_q98_compiler_uses_physical_parent_inventory() -> None:
    rows = json.loads(
        (ROOT / "sub_v290_scope2" / "submission.json").read_text(encoding="utf-8")
    )
    row = next(item for item in rows if int(item["id"]) == 98)
    compiler = DeterministicFinancialCompiler(ROOT)
    compiled = compiler.compile(row["question"], extract_all_facets(row["question"]))
    assert compiled is not None
    assert compiled.metric == "inventory"
    assert compiled.answer == 146.469679444
    assert compiled.table_refs == ["HUT_financial_statements_2024_consolidated|325"]
    assert compiled.source_cells[0]["raw"] == "146.469.679.444"


def test_q714_unqualified_query_preserves_measured_catalog_default() -> None:
    rows = json.loads(
        (ROOT / "sub_v290_scope2" / "submission.json").read_text(encoding="utf-8")
    )
    row = next(item for item in rows if int(item["id"]) == 714)
    compiler = DeterministicFinancialCompiler(ROOT)
    compiled = compiler.compile(row["question"], extract_all_facets(row["question"]))
    assert compiled is not None
    assert compiled.metric == "net_finance_result"
    assert compiled.answer == 238.891842241
    assert compiled.table_refs == ["HUT_financial_statements_2024_consolidated|376"]
