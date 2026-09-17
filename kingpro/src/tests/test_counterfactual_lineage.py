from __future__ import annotations

import json
from pathlib import Path

from kingpro.product.counterfactual_lineage import CounterfactualLineageAnalyzer


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "root"
    candidate = root / "candidate"
    physical = root / "build" / "tables" / "AAA_report"
    physical.mkdir(parents=True)
    candidate.joinpath("data").mkdir(parents=True)
    source = "0,1,2\n,2024VND,2023VND\nDoanh thu thuần,1.234,900\nChi phí,700,600\n"
    physical.joinpath("table_1.csv").write_text(source, encoding="utf-8")
    candidate.joinpath("data", "AAA_10.csv").write_text(source, encoding="utf-8")
    root.joinpath("build", "catalog.jsonl").write_text(
        json.dumps(
            {"table_ref": "AAA_report|10", "csv_path": "AAA_report/table_1.csv"}
        )
        + "\n",
        encoding="utf-8",
    )
    row = {
        "id": 1,
        "question": "Doanh thu AAA năm 2024 là bao nhiêu?",
        "answer": 1234.0,
        "relevant_tables": ["AAA_report|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/AAA_10.csv"}],
        "pandas_query": (
            "df1 = list(dfs.values())[0]\n"
            "result = float(str(df1.iloc[1, 1]).replace('.', ''))"
        ),
    }
    candidate.joinpath("submission.json").write_text(
        json.dumps([row], ensure_ascii=False), encoding="utf-8"
    )
    return root, candidate, row


def test_counterfactual_dependency_finds_only_result_sensitive_cell(tmp_path: Path) -> None:
    root, candidate, row = _fixture(tmp_path)
    report = CounterfactualLineageAnalyzer(root, candidate).analyze(row)
    assert report["status"] == "verified"
    assert report["active_variables"] == ["df1"]
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert (cell["table_ref"], cell["row_idx"], cell["col_idx"], cell["raw"]) == (
        "AAA_report|10",
        1,
        1,
        "1.234",
    )
    assert len(cell["counterfactuals"]) >= 2
    assert all(item["result"] != row["answer"] for item in cell["counterfactuals"])


def test_counterfactual_lineage_refuses_non_relevant_physical_binding(tmp_path: Path) -> None:
    root, candidate, row = _fixture(tmp_path)
    row = {**row, "relevant_tables": ["UNRELATED|99"]}
    report = CounterfactualLineageAnalyzer(root, candidate).analyze(row)
    assert report == {
        "status": "skipped",
        "reason": "no_unique_physical_binding",
        "cells": [],
    }


def test_backward_slice_avoids_perturbing_unused_frames() -> None:
    code = """
df1 = list(dfs.values())[0]
df2 = list(dfs.values())[1]
picked = df2.iloc[0, 0]
result = float(picked)
"""
    assert CounterfactualLineageAnalyzer._referenced_frames(code, {"df1", "df2"}) == {"df2"}
