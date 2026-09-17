from __future__ import annotations

import csv
import json
from pathlib import Path

from kingpro.product.cell_lineage import RuntimeCellLineageIndex
from kingpro.product.service import ProductService


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, *, manifest_raw: str = "1.234") -> RuntimeCellLineageIndex:
    root = tmp_path / "root"
    artifact = root / "candidate"
    table = root / "build" / "tables" / "AAA_report"
    table.mkdir(parents=True)
    artifact.joinpath("data").mkdir(parents=True)
    table.joinpath("table_1.csv").write_text(
        "0,1,2\n,2024VND,2023VND\nDoanh thu thuần,1.234,900\n",
        encoding="utf-8",
    )
    root.joinpath("build", "catalog.jsonl").write_text(
        json.dumps(
            {
                "table_ref": "AAA_report|10",
                "csv_path": "AAA_report/table_1.csv",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact.joinpath("submission.json").write_text(
        json.dumps([{"id": 1}], ensure_ascii=False), encoding="utf-8"
    )
    with artifact.joinpath("data", "q1_source_cells.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker", "year", "metric_key", "raw", "typed_factor",
                "scale", "source_table", "source_csv", "row_idx", "col_idx",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ticker": "AAA",
                "year": "2024",
                "metric_key": "kqkd:10",
                "raw": manifest_raw,
                "typed_factor": "1",
                "scale": "1",
                "source_table": "AAA_report|10",
                "source_csv": "table_1.csv",
                "row_idx": "1",
                "col_idx": "1",
            }
        )
    return RuntimeCellLineageIndex(root, artifact)


def test_manifest_coordinate_is_reproved_against_physical_csv(tmp_path: Path) -> None:
    index = _fixture(tmp_path)
    cells = index.for_question(1)
    assert len(cells) == 1
    assert cells[0]["table_ref"] == "AAA_report|10"
    assert cells[0]["row_idx"] == 1
    assert cells[0]["col_idx"] == 1
    assert cells[0]["source_label"] == "Doanh thu thuần"
    assert cells[0]["header_path"] == ["2024VND"]
    assert cells[0]["raw_manifest"] == cells[0]["raw_physical"] == "1.234"
    assert cells[0]["verification"] == "coordinate_and_raw_match"


def test_stale_manifest_fails_closed(tmp_path: Path) -> None:
    index = _fixture(tmp_path, manifest_raw="9.999")
    assert index.for_question(1) == []


def test_v297_registry_replay_exposes_verified_cell_lineage() -> None:
    rows = json.loads(
        (ROOT / "sub_v297_scope2" / "submission.json").read_text(encoding="utf-8")
    )
    question = next(row["question"] for row in rows if int(row["id"]) == 69)
    service = ProductService(
        root=ROOT,
        replay_submission=ROOT / "sub_v297_scope2" / "submission.json",
    )
    response = service.ask(question)
    assert response["status"] == "answered"
    cells = response["verification"]["source_cells"]
    assert len(cells) == 1
    assert cells[0]["table_ref"] == "SHB_financial_statements_2019_consolidated|2372"
    assert cells[0]["row_idx"] == 17
    assert cells[0]["col_idx"] == 8
    assert cells[0]["raw_physical"] == "259.236.746"
    assert cells[0]["header_path"][-1] == "Tổng cộng"
    assert response["verification"]["cell_lineage"]["status"] == "verified"


def test_v297_coverage_is_measured_not_inferred() -> None:
    index = RuntimeCellLineageIndex(ROOT, ROOT / "sub_v297_scope2")
    assert index.coverage() == {
        "manifest_questions": 814,
        "source_audit_questions": 643,
        "audited_union_questions": 817,
        "counterfactual_questions": 195,
        "runtime_lineage_questions": 1012,
        "submission_questions": 1012,
    }


def test_v297_legacy_program_uses_hash_bound_counterfactual_lineage() -> None:
    index = RuntimeCellLineageIndex(ROOT, ROOT / "sub_v297_scope2")
    cells = index.for_question(1)
    assert len(cells) == 1
    assert cells[0]["table_ref"] == "VJC_financial_statements_2018_separate|1179"
    assert cells[0]["row_idx"] == 1
    assert cells[0]["col_idx"] == 1
    assert cells[0]["raw_physical"] == "208.253.201.298"
    assert cells[0]["verification"] == (
        "counterfactual_result_dependency_and_coordinate_raw_match"
    )
    assert len(cells[0]["counterfactuals"]) == 3
