from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_manifest_dtype_inference import accounting_number, audit


HEADER = "ticker,year,metric_key,raw,typed_factor,scale,source_table,source_csv,row_idx,col_idx\n"


def _candidate(tmp_path: Path, factor: float) -> Path:
    candidate = tmp_path / "candidate"
    data = candidate / "data"
    data.mkdir(parents=True)
    (candidate / "submission.json").write_text("[]\n", encoding="utf-8")
    (data / "q749_source_cells.csv").write_text(
        HEADER
        + "EIB,2025,note:x,380,1,1,EIB|1,a.csv,0,1\n"
        + f"ACB,2025,note:x,450.276,{factor},1,ACB|2,b.csv,0,1\n",
        encoding="utf-8",
    )
    return candidate


def test_accounting_parser_distinguishes_decimal_and_thousands() -> None:
    assert accounting_number("0.56") == 0.56
    assert accounting_number("450.276") == 450276.0
    assert accounting_number("(210.684)") == -210684.0


def test_latent_single_token_inference_catches_wrong_factor(tmp_path: Path) -> None:
    report = audit(_candidate(tmp_path, 1.0))
    assert report["passed"] is False
    assert report["finding_ids"] == [749]
    assert any(
        item["kind"] == "single-token-numeric-inference-mismatch"
        for item in report["findings"]
    )


def test_correct_factor_is_invariant(tmp_path: Path) -> None:
    report = audit(_candidate(tmp_path, 1000.0))
    assert report["passed"] is True
    assert report["finding_count"] == 0


def test_legacy_manifest_is_unsupported_not_a_numeric_finding(tmp_path: Path) -> None:
    candidate = tmp_path / "legacy"
    data = candidate / "data"
    data.mkdir(parents=True)
    (candidate / "submission.json").write_text("[]\n", encoding="utf-8")
    (data / "q1_source_cells.csv").write_text(
        "ticker,year,metric_key,raw,scale,source_table,source_csv,row_idx,col_idx\n"
        "AAA,2025,x,123,1,AAA|1,a.csv,0,1\n",
        encoding="utf-8",
    )
    report = audit(candidate)
    assert report["finding_count"] == 0
    assert report["unsupported_ids"] == [1]
    assert report["complete_coverage"] is False
    assert report["passed"] is False
