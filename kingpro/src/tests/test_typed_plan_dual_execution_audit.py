from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.product.typed_plan_audit import TypedPlanAuditor
from kingpro.retrieval.bm25_index import extract_all_facets


REGISTRY = ROOT / "sub_top123_candidate_v269_source_lineage_control" / "submission.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(*ids: int) -> list[dict]:
    wanted = set(ids)
    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [row for row in rows if int(row["id"]) in wanted]


def test_direct_plan_oracle_is_independent_of_compiler_answer() -> None:
    auditor = TypedPlanAuditor(ROOT)
    row = _rows(662)[0]
    facets = extract_all_facets(row["question"])
    compiled = auditor.compiler.compile(row["question"], facets)
    assert compiled is not None
    oracle = auditor.typed_oracle(
        row["question"], facets, replace(compiled, answer=compiled.answer + 12345)
    )

    assert oracle.plan.kind == "direct"
    assert oracle.plan.operation == "linear_ratio"
    assert oracle.plan.metric == "npat_to_assets_pct"
    assert len(oracle.plan.operands) == 2
    assert oracle.value == compiled.answer


def test_panel_and_universe_plans_capture_selector_filter_and_provenance() -> None:
    auditor = TypedPlanAuditor(ROOT)
    panel_row, universe_row = _rows(363, 464)

    panel = auditor.audit_row(panel_row)
    assert panel["status"] == "verified"
    assert panel["typed_plan"]["kind"] == "panel"
    assert panel["typed_plan"]["operation"] in {"select_max", "select_min"}
    assert panel["typed_plan"]["parameters"]["selector_metric"] == "liabilities_to_equity"
    assert panel["provenance"]["exact_match"] is True

    universe = auditor.audit_row(universe_row)
    assert universe["status"] == "verified"
    assert universe["typed_plan"]["kind"] == "universe"
    assert universe["typed_plan"]["operation"] == "filter_threshold_then_max"
    assert universe["typed_plan"]["parameters"]["threshold"] == -10.0
    assert universe["typed_plan"]["parameters"]["eligible_count"] > 0
    assert universe["provenance"]["exact_match"] is True


def test_refused_query_is_emitted_instead_of_dropped() -> None:
    auditor = TypedPlanAuditor(ROOT)
    refused = auditor.audit_row(_rows(1)[0])

    assert refused["compiler_accepted"] is False
    assert refused["status"] == "refused"
    assert refused["refusal"]["code"] == "unsupported_or_ambiguous_query"
    assert refused["typed_plan"] is None


def test_focused_audit_is_read_only_and_summarises_coverage() -> None:
    cube = ROOT / "build" / "statement_cube.jsonl"
    before = {str(cube): _sha256(cube), str(REGISTRY): _sha256(REGISTRY)}
    report = TypedPlanAuditor(ROOT).audit_rows(_rows(1, 363, 464, 662))
    after = {str(cube): _sha256(cube), str(REGISTRY): _sha256(REGISTRY)}

    assert before == after
    assert report["mode"] == "read_only_offline"
    assert report["verification_scope"]["blind_second_solver"] is False
    assert "shared_semantic_gate" in report["verification_scope"]["dependency_groups"]
    assert report["summary"]["total_questions"] == 4
    assert report["summary"]["compiler_accepted"] == 3
    assert report["summary"]["compiler_refused"] == 1
    assert report["summary"]["accepted_uncovered"] == 0
    assert report["summary"]["verified"] == 3
    assert report["summary"]["plan_kind_counts"] == {
        "direct": 1,
        "panel": 1,
        "universe": 1,
    }
