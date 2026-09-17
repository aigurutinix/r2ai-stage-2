import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_rerank_scores.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("validate_rerank_scores", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PAIR_SCRIPT = SCRIPT.with_name("validate_rerank_pairs.py")
PAIR_SPEC = importlib.util.spec_from_file_location("validate_rerank_pairs", PAIR_SCRIPT)
PAIR_MODULE = importlib.util.module_from_spec(PAIR_SPEC)
assert PAIR_SPEC.loader is not None
PAIR_SPEC.loader.exec_module(PAIR_MODULE)


PAIRS = [
    {
        "id": 1,
        "candidates": [{"table_id": "a"}, {"table_id": "b"}],
    }
]


def test_complete_rerank_scores_are_accepted():
    assert MODULE.validate(PAIRS, [{"id": 1, "scores": {"a": 0.9, "b": 0.1}}]) == {
        "questions": 1,
        "candidates": 2,
    }


@pytest.mark.parametrize(
    "scores",
    (
        [],
        [{"id": 1, "scores": {"a": 0.9}}],
        [
            {"id": 1, "scores": {"a": 0.9, "b": 0.1}},
            {"id": 1, "scores": {"a": 0.8, "b": 0.2}},
        ],
    ),
)
def test_incomplete_or_duplicate_rerank_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        MODULE.validate(PAIRS, scores)


@pytest.mark.parametrize("value", (math.nan, math.inf, -0.1, 1.1, "0.5", True))
def test_invalid_score_values_are_rejected(value):
    with pytest.raises(ValueError):
        MODULE.validate(PAIRS, [{"id": 1, "scores": {"a": value, "b": 0.1}}])


def test_duplicate_pair_candidates_are_rejected():
    pairs = [{"id": 1, "candidates": [{"table_id": "a"}, {"table_id": "a"}]}]
    with pytest.raises(ValueError, match="duplicate candidate"):
        MODULE.validate(pairs, [{"id": 1, "scores": {"a": 0.9}}])


def report(report_id="ACB/2024/con"):
    ticker, year, scope = report_id.split("/")
    identity = SimpleNamespace(ticker=ticker, year=int(year), scope=scope)
    return SimpleNamespace(identity=identity)


def test_full_pair_gate_accepts_aligned_dense_and_graph_candidates():
    report_id = "ACB/2024/con"
    records = [{
        "id": 1,
        "selected_docs": [report_id],
        "candidates": [
            {"table_id": f"{report_id}|10", "text": "sparse", "sparse_rank": 1},
            {"table_id": f"{report_id}|20", "text": "dense", "dense_rank": 1},
            {"table_id": f"{report_id}|30", "text": "graph", "graph_rank": 1},
        ],
    }]
    traces = [{"id": 1, "added": [{"table_id": f"{report_id}|30"}]}]
    sidecar = {f"{report_id}|20": report_id, f"{report_id}|30": report_id}
    result = PAIR_MODULE.validate(
        records, [{"id": 1}], {report_id: report(report_id)}, sidecar, traces,
    )
    assert result["pairs"] == 3
    assert result["dense_only_candidates"] == 1
    assert result["graph_candidates"] == 1


def test_full_pair_gate_rejects_dense_candidate_outside_same_triple():
    selected, wrong = "ACB/2024/con", "VCB/2024/con"
    records = [{
        "id": 1,
        "selected_docs": [selected],
        "candidates": [{"table_id": f"{wrong}|20", "text": "dense", "dense_rank": 1}],
    }]
    with pytest.raises(ValueError, match="same-triple"):
        PAIR_MODULE.validate(
            records, [{"id": 1}],
            {selected: report(selected), wrong: report(wrong)},
            {f"{wrong}|20": wrong}, [{"id": 1, "added": []}],
        )


def test_reranker_model_sizes_have_independent_pinned_revisions():
    source = (Path(__file__).resolve().parents[1] / "kaggle" / "rerank_qwen_8b.py")
    text = source.read_text(encoding="utf-8")
    assert '"Qwen/Qwen3-Reranker-8B": "77d193c' in text
    assert '"Qwen/Qwen3-Reranker-4B": "22e6836' in text
    assert "MODEL_REVISION = MODEL_REVISIONS[MODEL_NAME]" in text


def test_score_manifest_pins_pairs_scores_and_configuration(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    scores = tmp_path / "scores.jsonl"
    manifest = tmp_path / "scores.manifest.json"
    pairs.write_text("pairs\n", encoding="utf-8")
    scores.write_text("scores\n", encoding="utf-8")
    value = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "revision": "revision",
        "quantization": "int8",
        "adapter": None,
        "max_length": 1024,
        "rerank_depth": 130,
        "prompt_sha256": "prompt",
        "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    result = MODULE.validate_manifests(pairs, [scores], [manifest])
    assert result["model"] == "Qwen/Qwen3-Reranker-8B"


def test_score_manifest_rejects_wrong_score_hash(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    scores = tmp_path / "scores.jsonl"
    manifest = tmp_path / "scores.manifest.json"
    pairs.write_text("pairs\n", encoding="utf-8")
    scores.write_text("scores\n", encoding="utf-8")
    manifest.write_text(json.dumps({
        "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
        "scores_sha256": "wrong",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="score hash"):
        MODULE.validate_manifests(pairs, [scores], [manifest])
