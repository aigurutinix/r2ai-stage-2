"""Tests for the dense-artifact verifier and the sidecar merger."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_table_sidecars
import verify_dense_artifacts


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


# --- merge_table_sidecars ---------------------------------------------------


def test_merge_unions_disjoint_lanes(tmp_path):
    dense = write_json(tmp_path / "dense.json", {"R|1": "R"})
    graph = write_json(tmp_path / "graph.json", {"R|2": "R"})
    merged, conflicts = merge_table_sidecars.merge(
        [(dense, {"R|1": "R"}), (graph, {"R|2": "R"})]
    )
    assert conflicts == []
    assert merged == {"R|1": "R", "R|2": "R"}


def test_merge_accepts_the_same_table_from_two_lanes():
    # A note table is often also a graph target; agreeing twice is not an error.
    merged, conflicts = merge_table_sidecars.merge(
        [(Path("a"), {"R|9": "R"}), (Path("b"), {"R|9": "R"})]
    )
    assert conflicts == []
    assert merged == {"R|9": "R"}


def test_merge_reports_a_conflict_rather_than_letting_the_last_lane_win():
    merged, conflicts = merge_table_sidecars.merge(
        [(Path("a"), {"R|9": "R"}), (Path("b"), {"R|9": "OTHER"})]
    )
    assert len(conflicts) == 1
    assert "R|9" in conflicts[0]
    assert merged["R|9"] == "R"


def test_merge_exits_non_zero_on_a_conflict(tmp_path, monkeypatch, capsys):
    first = write_json(tmp_path / "a.json", {"R|9": "R"})
    second = write_json(tmp_path / "b.json", {"R|9": "OTHER"})
    output = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", [
        "merge_table_sidecars.py",
        "--input", str(first), "--input", str(second), "--output", str(output),
    ])
    assert merge_table_sidecars.main() == 1
    assert "conflicting report ids" in capsys.readouterr().out
    assert not output.exists()


def test_merge_writes_the_union_and_exits_zero(tmp_path, monkeypatch):
    first = write_json(tmp_path / "a.json", {"R|1": "R"})
    second = write_json(tmp_path / "b.json", {"R|2": "R"})
    output = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", [
        "merge_table_sidecars.py",
        "--input", str(first), "--input", str(second), "--output", str(output),
    ])
    assert merge_table_sidecars.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"R|1": "R", "R|2": "R"}


def test_merge_rejects_an_entry_without_a_report(tmp_path):
    path = write_json(tmp_path / "bad.json", {"R|1": ""})
    with pytest.raises(SystemExit):
        merge_table_sidecars.load_map(path)


# --- verify_dense_artifacts -------------------------------------------------


def unit_vectors(rows: int, dims: int = verify_dense_artifacts.EXPECTED_DIMS):
    vectors = np.zeros((rows, dims), dtype=np.float32)
    vectors[:, 0] = 1.0
    return vectors.astype(np.float16)


def test_healthy_vectors_report_no_failures():
    failures = []
    verify_dense_artifacts.check_vectors(unit_vectors(4), ["a", "b", "c", "d"],
                                         "v", failures)
    assert failures == []


def test_unnormalised_vectors_are_rejected():
    # The Matryoshka prefix of a unit vector is not itself a unit vector; this
    # is what catches a truncation that skipped renormalisation.
    vectors = unit_vectors(4).astype(np.float32) * 0.5
    failures = []
    verify_dense_artifacts.check_vectors(vectors.astype(np.float16),
                                         ["a", "b", "c", "d"], "v", failures)
    assert any("L2 norms" in failure for failure in failures)


def test_id_count_must_match_the_vector_rows():
    failures = []
    verify_dense_artifacts.check_vectors(unit_vectors(4), ["a", "b"], "v", failures)
    assert any("against" in failure for failure in failures)


def test_duplicate_ids_are_rejected():
    failures = []
    verify_dense_artifacts.check_vectors(unit_vectors(2), ["a", "a"], "v", failures)
    assert any("not unique" in failure for failure in failures)


def test_wrong_width_is_rejected_before_the_norm_check():
    failures = []
    verify_dense_artifacts.check_vectors(unit_vectors(2, dims=8), ["a", "b"],
                                         "v", failures)
    assert len(failures) == 1
    assert "shape" in failures[0]


def test_non_finite_vectors_are_rejected():
    vectors = unit_vectors(2).astype(np.float32)
    vectors[0, 0] = np.inf
    failures = []
    verify_dense_artifacts.check_vectors(vectors.astype(np.float16), ["a", "b"],
                                         "v", failures)
    assert any("NaN or Inf" in failure for failure in failures)


def test_builder_configuration_is_read_from_the_source():
    config = verify_dense_artifacts.embedding_config(
        Path(__file__).resolve().parents[1] / "kaggle" / "embed_tables.py"
    )
    assert config["MODEL_NAME"] == "Qwen/Qwen3-Embedding-4B"
    assert config["MODEL_REVISION"] == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert config["DIMS"] == verify_dense_artifacts.EXPECTED_DIMS
    assert config["INSTRUCT"].startswith("Given a Vietnamese financial question")


def test_builder_configuration_does_not_execute_source(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(
        'MODEL_NAME = "model"\n'
        'MODEL_REVISION = "revision"\n'
        'MAX_LENGTH = 256\n'
        'DIMS = 1024\n'
        'INSTRUCT = __import__("os").getcwd()\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="INSTRUCT must be a literal"):
        verify_dense_artifacts.embedding_config(builder)


def test_table_report_pairing_is_validated():
    failures = []
    table_ids = verify_dense_artifacts.validate_table_ids(
        [["report", "report|10"], ["wrong", "report|11"]], failures
    )
    assert table_ids == ["report|10", "report|11"]
    assert any("pairs" in failure for failure in failures)
