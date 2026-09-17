from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import export_rerank_pairs
from export_rerank_pairs import dense_candidates, dense_report_ids, merge_dense_candidates
from rerank_qwen_8b import to_judge
from vifinqa.retrieval import Table
from vifinqa.tables import ReportIdentity


def report(ticker: str, year: int, scope: str):
    return SimpleNamespace(identity=SimpleNamespace(ticker=ticker, year=year, scope=scope))


def test_same_triple_policy_keeps_duplicate_filing_but_not_another_scope():
    reports = {
        "AAA_2024_consolidated": report("AAA", 2024, "consolidated"),
        "AAA_2024_consolidated_2": report("AAA", 2024, "consolidated"),
        "AAA_2024_separate": report("AAA", 2024, "separate"),
        "BBB_2024_consolidated": report("BBB", 2024, "consolidated"),
    }

    assert dense_report_ids(reports, ["AAA_2024_consolidated"], "same-triple") == {
        "AAA_2024_consolidated", "AAA_2024_consolidated_2",
    }


def test_selected_reports_and_any_policies_are_explicit():
    reports = {"AAA": report("AAA", 2024, "consolidated")}
    assert dense_report_ids(reports, ["AAA"], "selected-reports") == {"AAA"}
    assert dense_report_ids(reports, ["AAA"], "any") is None


def test_scorer_accepts_dense_candidates_without_a_sparse_rank():
    record = {"candidates": [
        {"table_id": "sparse", "sparse_rank": 1},
        {"table_id": "dense", "sparse_rank": None, "dense_rank": 2},
    ]}

    assert to_judge(record) == [0, 1]


def test_dense_similarity_is_computed_inside_the_metadata_boundary(monkeypatch):
    identities = {
        name: ReportIdentity(name, name, 2024, "consolidated", f"{name}.txt")
        for name in ("R1", "R2")
    }
    reports = {name: SimpleNamespace(identity=identity) for name, identity in identities.items()}
    tables = {
        table_id: Table(
            table_id, report_id, None, 1, (("Chỉ tiêu", "2024"), ("Doanh thu", "1")),
            "", (), (("Chỉ tiêu", "2024"),), ("2024",), "VND",
        )
        for report_id, table_id in (("R1", "A"), ("R1", "B"), ("R2", "C"))
    }
    monkeypatch.setattr(
        export_rerank_pairs, "report_tables_cached",
        lambda report: tuple(table for table in tables.values()
                             if table.report_id == report.identity.report_id),
    )
    dense = {
        "order": {7: 0},
        "ids": [("R1", "A"), ("R1", "B"), ("R2", "C")],
        "tables": np.array(((1.0, 0.0), (0.0, 1.0), (2.0, 0.0)), dtype=np.float32),
        "questions": np.array(((1.0, 0.0),), dtype=np.float32),
        "report_rows": {"R1": [0, 1], "R2": [2]},
    }

    selected = dense_candidates(dense, 7, reports, 1, {"R1"})

    assert [candidate["table_id"] for candidate in selected] == ["A"]


def test_dense_overlap_adds_a_rank_without_duplicating_the_sparse_candidate():
    sparse = [
        {"table_id": "A", "sparse_rank": 1, "text": "sparse A"},
        {"table_id": "B", "sparse_rank": 2, "text": "sparse B"},
    ]
    dense = [
        {"table_id": "B", "report_id": "R", "dense_rank": 1,
         "dense_score": 0.9, "text": "dense B"},
        {"table_id": "C", "report_id": "R", "dense_rank": 2,
         "dense_score": 0.8, "text": "dense C"},
    ]

    merged = merge_dense_candidates(sparse, dense)

    assert [candidate["table_id"] for candidate in merged] == ["A", "B", "C"]
    assert merged[1]["sparse_rank"] == 2
    assert merged[1]["dense_rank"] == 1
    # Use the sparse rendering for overlaps so one table has one deterministic
    # cross-encoder input.
    assert merged[1]["text"] == "sparse B"


def test_equal_dense_scores_break_ties_by_global_vector_row(monkeypatch):
    identities = {
        name: ReportIdentity(name, name, 2024, "consolidated", f"{name}.txt")
        for name in ("R1", "R2")
    }
    reports = {name: SimpleNamespace(identity=identity) for name, identity in identities.items()}
    tables = {
        table_id: Table(
            table_id, report_id, None, 1, (("Metric", "2024"), ("Value", "1")),
            "", (), (("Metric", "2024"),), ("2024",), "VND",
        )
        for report_id, table_id in (("R1", "A"), ("R2", "B"))
    }
    monkeypatch.setattr(
        export_rerank_pairs, "report_tables_cached",
        lambda report: (tables["A"],) if report.identity.report_id == "R1" else (tables["B"],),
    )
    dense = {
        "order": {7: 0},
        "ids": [("R1", "A"), ("R2", "B")],
        "tables": np.array(((1.0, 0.0), (1.0, 0.0)), dtype=np.float32),
        "questions": np.array(((1.0, 0.0),), dtype=np.float32),
        # Deliberately oppose report insertion order to global vector order.
        "report_rows": {"R2": [1], "R1": [0]},
    }

    first = dense_candidates(dense, 7, reports, 2, {"R2", "R1"})
    second = dense_candidates(dense, 7, reports, 2, {"R1", "R2"})

    assert [candidate["table_id"] for candidate in first] == ["A", "B"]
    assert [candidate["table_id"] for candidate in second] == ["A", "B"]
