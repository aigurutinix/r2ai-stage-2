from __future__ import annotations

from scripts.audit_compiler_schema_robustness import audit


def test_compiler_schema_robustness_suite_passes_all_named_classes() -> None:
    report = audit()
    assert report["passed"], report
    assert report["case_count"] == 10
    assert report["passed_count"] == 10
    assert report["failed_count"] == 0
    assert all(report["coverage"].values())

