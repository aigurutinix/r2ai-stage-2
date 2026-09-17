import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11_merge_codegen.py"
SPEC = importlib.util.spec_from_file_location("merge_codegen_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(qid, *, status="ok", confidence=99, detail="", source="rule"):
    return {
        "id": qid,
        "status": status,
        "detail_conf": confidence,
        "detail": detail,
        "source": source,
        "answer": float(qid),
    }


def test_safe_merge_only_fills_failed_high_confidence_rows():
    base = [_row(1), _row(2, status="failed"), _row(3, status="failed")]
    candidates = [_row(1), _row(2), _row(3, confidence=60)]

    merged, accepted = MODULE.merge_codegen(base, candidates)

    assert accepted == [2]
    assert merged[0]["source"] == "rule"
    assert merged[1]["source"] == "canonical_v2_blend:rule"
    assert merged[2]["status"] == "failed"


def test_safe_merge_rejects_ambiguous_and_unit_warning_rows():
    base = [_row(1, status="failed"), _row(2, status="failed")]
    candidates = [
        _row(1, detail="AMBIGUOUS"),
        _row(2, detail="UNIT-WARN: ratio is implausible"),
    ]

    merged, accepted = MODULE.merge_codegen(base, candidates)

    assert accepted == []
    assert all(row["status"] == "failed" for row in merged)


def test_allowlist_limits_an_explicit_audited_merge():
    base = [_row(1, status="failed"), _row(2, status="failed")]
    candidates = [_row(1, confidence=20), _row(2, confidence=20)]

    merged, accepted = MODULE.merge_codegen(
        base, candidates, min_confidence=0, allow_ids={2})

    assert accepted == [2]
    assert merged[0]["status"] == "failed"
    assert merged[1]["source"] == "canonical_v2_blend:rule"
