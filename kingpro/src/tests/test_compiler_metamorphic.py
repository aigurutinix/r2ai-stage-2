from __future__ import annotations

from pathlib import Path

from scripts.audit_compiler_metamorphic import audit


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "sub_top123_candidate_v269_source_lineage_control" / "submission.json"


def test_metamorphic_subset_rebuilds_and_injects_without_disagreement() -> None:
    report = audit(REGISTRY, root=ROOT, max_cases=3)
    assert report["compiled_entries"] >= 1
    assert report["passed"], report
    assert report["disagreement_ids"] == []
    assert report["variants"]["cube_reversed"]["status"] == "invariant"
    assert report["variants"]["irrelevant_cell_appended"]["status"] == "invariant"
    assert report["variants"]["registry_reversed"]["status"] == "invariant"
    assert "unique unused ticker" in report["coverage"]["irrelevant_cell_injection"]


def test_metamorphic_report_hashes_immutable_inputs() -> None:
    report = audit(REGISTRY, root=ROOT, max_cases=1)
    hashes = report["input_hashes"]
    assert set(hashes) == {
        "registry_sha256",
        "compiler_sha256",
        "catalog_sha256",
        "statement_cube_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())
