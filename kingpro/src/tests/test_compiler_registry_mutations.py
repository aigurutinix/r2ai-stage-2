from __future__ import annotations

import json
import warnings
from pathlib import Path

from scripts.audit_compiler_registry_mutations import audit
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.retrieval.bm25_index import extract_all_facets


ROOT = Path(__file__).resolve().parents[1]


def test_registry_compiler_replay_rejects_changed_bound_operands() -> None:
    report = audit(
        ROOT
        / "sub_top123_candidate_v217_missing_panel_operand_batch3"
        / "submission.json",
        root=ROOT,
        max_cases=1,
    )
    assert report["passed"], report
    assert report["compiled_entries"] == 1
    assert report["baseline_replays_passed"] == 1
    assert report["mutated_replays_rejected"] == 1
    assert report["protected_operand_coordinates"] >= 1
    assert not report["failures"]


def test_registry_ocr_backslashes_emit_warning_free_python() -> None:
    registry = json.loads(
        (
            ROOT
            / "sub_top123_candidate_v217_missing_panel_operand_batch3"
            / "submission.json"
        ).read_text(encoding="utf-8")
    )
    row = next(item for item in registry if int(item["id"]) == 381)
    compiler = DeterministicFinancialCompiler(ROOT)
    compiled = compiler.compile(
        str(row["question"]), extract_all_facets(str(row["question"]))
    )
    assert compiled is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(compiled.pandas_query, "<registry-compiler>", "exec")
