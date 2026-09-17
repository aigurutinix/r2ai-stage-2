from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v274_q986_nan_guard as builder
from scripts.grader_check import run_one


def _rows(path: Path) -> dict[int, dict]:
    return {
        int(row["id"]): row
        for row in json.loads((path / "submission.json").read_text(encoding="utf-8"))
    }


def test_only_q986_submission_row_changes() -> None:
    before = _rows(builder.SOURCE)
    after = _rows(builder.OUTPUT)
    assert before.keys() == after.keys()
    assert [qid for qid in before if before[qid] != after[qid]] == [986]
    assert before[986]["answer"] == after[986]["answer"] == 2016.0
    for key in ("relevant_docs", "relevant_tables", "evidence"):
        assert before[986][key] == after[986][key]


def test_non_submission_payload_shape_and_critical_lineage_are_preserved() -> None:
    added = Path("v274_q986_nan_guard_audit.json")
    before = {
        path.relative_to(builder.SOURCE)
        for path in builder.SOURCE.rglob("*")
        if path.is_file() and path.relative_to(builder.SOURCE) != Path("submission.json")
    }
    after = {
        path.relative_to(builder.OUTPUT)
        for path in builder.OUTPUT.rglob("*")
        if path.is_file() and path.relative_to(builder.OUTPUT) not in {Path("submission.json"), added}
    }
    assert before == after
    for relative in (
        Path("source_audit.json"),
        Path("data/q749_source_cells.csv"),
        Path("data/q780_source_cells.csv"),
        Path("data/q986_source_cells.csv"),
        Path("v272_q749_q780_cleanup_audit.json"),
    ):
        assert hashlib.sha256((builder.SOURCE / relative).read_bytes()).digest() == hashlib.sha256(
            (builder.OUTPUT / relative).read_bytes()
        ).digest()


def test_q986_nan_guard_matches_in_string_and_typed_modes() -> None:
    row = _rows(builder.OUTPUT)[986]
    paths = {
        item["variable"]: str((builder.OUTPUT / item["csv_path"]).resolve())
        for item in row["evidence"]
    }
    assert float(run_one(row["pandas_query"], paths, typed_dfs=False)) == 2016.0
    typed_result, namespace = run_one(
        row["pandas_query"], paths, typed_dfs=True, return_namespace=True
    )
    assert float(typed_result) == 2016.0
    assert namespace["v3"] == 0.0
    assert "if value != value" in row["pandas_query"]


def test_builder_refuses_overwrite() -> None:
    try:
        builder.build()
    except FileExistsError:
        pass
    else:
        raise AssertionError("builder must refuse an existing output")
