from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_bm25_source_disagreements.py"
SPEC = importlib.util.spec_from_file_location("audit_bm25_source_disagreements", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_tables_prefers_executed_sources_and_ignores_recall_rows() -> None:
    row = {"relevant_tables": ["R|legacy"]}
    audit = {
        "sources": [
            {"table_ref": "R|executed", "metric": "note:cash"},
            {"table_ref": "R|retrieval", "metric": "recall:extra"},
        ]
    }

    assert MODULE.target_tables(row, audit) == ["R|executed"]


def test_trusted_review_zeroes_priority() -> None:
    assert MODULE.priority_score(
        has_source_audit=False,
        trusted_review=True,
        target_count=1,
        target_rank=None,
        alternative_label_score=2.0,
        target_label_score=0.0,
    ) == 0.0


def test_legacy_single_table_miss_has_high_review_priority() -> None:
    assert MODULE.priority_score(
        has_source_audit=False,
        trusted_review=False,
        target_count=1,
        target_rank=None,
        alternative_label_score=1.5,
        target_label_score=0.0,
    ) == 10.0


def test_exact_panel_audit_reduces_review_priority() -> None:
    assert MODULE.priority_score(
        has_source_audit=True,
        trusted_review=False,
        target_count=8,
        target_rank=None,
        alternative_label_score=1.5,
        target_label_score=0.0,
    ) == 5.5
