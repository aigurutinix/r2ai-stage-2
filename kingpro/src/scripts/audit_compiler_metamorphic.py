"""Read-only metamorphic audit for the deterministic compiler registry.

For every registry row accepted by the compiler, this audit rebuilds a
temporary compiler twice: once with ``statement_cube.jsonl`` reversed and once
with one syntactically valid, irrelevant cell appended. Acceptance, answer,
metric, unit, and source bindings must remain unchanged. Real build files,
runtime code, and submissions are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler  # noqa: E402
from kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402

DEFAULT_REGISTRY = ROOT / "sub_top123_candidate_v269_source_lineage_control" / "submission.json"
DEFAULT_OUTPUT = ROOT / "build" / "v272_compiler_metamorphic_v269.json"
TOLERANCE = 0.0050001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _same_answer(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(float(left), float(right), rel_tol=0, abs_tol=TOLERANCE)
    )


def _compile_map(compiler: DeterministicFinancialCompiler, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        question_id = int(row.get("id", -1))
        question = str(row.get("question", ""))
        compiled = compiler.compile(question, extract_all_facets(question))
        if compiled is None:
            result[question_id] = {"accepted": False}
            continue
        result[question_id] = {
            "accepted": True,
            "answer": float(compiled.answer),
            "metric": compiled.metric,
            "unit": compiled.unit,
            "table_refs": list(compiled.table_refs),
            "source_cells": [
                {
                    "ticker": cell.get("ticker"),
                    "year": cell.get("year"),
                    "metric_key": cell.get("metric_key"),
                    "table_ref": cell.get("table_ref"),
                    "row_idx": cell.get("row_idx"),
                    "col_idx": cell.get("col_idx"),
                    "raw": cell.get("raw"),
                }
                for cell in compiled.source_cells
            ],
        }
    return result


def _compare(baseline: dict[int, dict[str, Any]], variant: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    disagreements: list[dict[str, Any]] = []
    for row in rows:
        question_id = int(row.get("id", -1))
        left = baseline.get(question_id, {"accepted": False})
        right = variant.get(question_id, {"accepted": False})
        if left.get("accepted") != right.get("accepted"):
            disagreements.append({"id": question_id, "kind": "acceptance", "baseline": left, "variant": right})
            continue
        if not left.get("accepted"):
            continue
        fields = ("metric", "unit", "table_refs", "source_cells")
        changed = [field for field in fields if left.get(field) != right.get(field)]
        if not _same_answer(left.get("answer"), right.get("answer")):
            changed.append("answer")
        if changed:
            disagreements.append({"id": question_id, "kind": "disagreement", "fields": changed, "baseline": left, "variant": right})
    return ("disagreement" if disagreements else "invariant", disagreements)


def _temp_variant(root: Path, *, reverse: bool, append_irrelevant: bool) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="compiler_metamorphic_"))
    build = temp / "build"
    build.mkdir(parents=True)
    cube = root / "build" / "statement_cube.jsonl"
    catalog = root / "build" / "catalog.jsonl"
    lines = [line for line in cube.read_text(encoding="utf-8").splitlines() if line.strip()]
    if reverse:
        lines.reverse()
    if append_irrelevant:
        sample = json.loads(lines[0])
        sample.update(
            {
                "ticker": "__METAMORPHIC_UNUSED__",
                "year": "2099",
                "scope": "consolidated",
                # Use a metric the compiler actually loads.  A made-up key
                # would be filtered by _REQUIRED_KEYS and would not exercise
                # distractor isolation at all.
                "metric_key": "kqkd:10",
                "ma_so": "10",
                "label": "__metamorphic_unused_cell__",
                "value": 123.0,
                "raw": "123",
                "table_ref": "__METAMORPHIC_UNUSED__|999",
                "csv_path": str(temp / "unused.csv"),
                "row_idx": 0,
                "col_idx": 0,
                "scale": 1.0,
            }
        )
        lines.append(json.dumps(sample, ensure_ascii=False))
    (build / "statement_cube.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(catalog, build / "catalog.jsonl")
    return temp


def audit(registry: Path = DEFAULT_REGISTRY, *, root: Path = ROOT, max_cases: int | None = None) -> dict[str, Any]:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("registry must be a JSON list")
    if max_cases is not None:
        # Bound accepted compiler rows, not raw registry position: registry
        # order commonly starts with unsupported analytic questions.
        probe = DeterministicFinancialCompiler(root)
        selected: list[dict[str, Any]] = []
        accepted = 0
        for row in rows:
            selected.append(row)
            question = str(row.get("question", ""))
            if probe.compile(question, extract_all_facets(question)) is not None:
                accepted += 1
                if accepted >= max_cases:
                    break
        rows = selected
    baseline = _compile_map(DeterministicFinancialCompiler(root), rows)
    accepted_rows = [row for row in rows if baseline[int(row.get("id", -1))].get("accepted")]
    variants: dict[str, dict[str, Any]] = {}
    temp_dirs: list[Path] = []
    try:
        for name, kwargs in (("cube_reversed", {"reverse": True, "append_irrelevant": False}), ("irrelevant_cell_appended", {"reverse": False, "append_irrelevant": True})):
            variant_root = _temp_variant(root, **kwargs)
            temp_dirs.append(variant_root)
            variant_map = _compile_map(DeterministicFinancialCompiler(variant_root), rows)
            status, disagreements = _compare(baseline, variant_map, rows)
            variants[name] = {"status": status, "accepted_entries": sum(item.get("accepted", False) for item in variant_map.values()), "disagreements": disagreements}
    finally:
        for directory in temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)

    reversed_registry = _compile_map(DeterministicFinancialCompiler(root), list(reversed(rows)))
    order_status, order_disagreements = _compare(baseline, reversed_registry, rows)
    variants["registry_reversed"] = {"status": order_status, "accepted_entries": sum(item.get("accepted", False) for item in reversed_registry.values()), "disagreements": order_disagreements}
    all_disagreements = [item for variant in variants.values() for item in variant["disagreements"]]
    return {
        "schema_version": 1,
        "audit": "compiler_registry_metamorphic_read_only",
        "registry": str(registry.resolve()),
        "registry_entries": len(rows),
        "compiled_entries": len(accepted_rows),
        "input_hashes": {
            "registry_sha256": sha256(registry),
            "compiler_sha256": sha256(root / "src" / "kingpro" / "product" / "deterministic_compiler.py"),
            "catalog_sha256": sha256(root / "build" / "catalog.jsonl"),
            "statement_cube_sha256": sha256(root / "build" / "statement_cube.jsonl"),
        },
        "variants": variants,
        "status_counts": {
            "invariant": sum(item["status"] == "invariant" for item in variants.values()),
            "disagreement": sum(item["status"] == "disagreement" for item in variants.values()),
            "unsupported": 0,
        },
        "disagreement_ids": sorted({int(item["id"]) for item in all_disagreements}),
        "coverage": {
            "accepted_registry_rows": len(accepted_rows),
            "cube_row_reversal": "all accepted rows recompiled in a temporary root with statement_cube.jsonl reversed",
            "irrelevant_cell_injection": "all accepted rows recompiled in a temporary root with one loaded metric cell under a unique unused ticker/year",
            "registry_row_reversal": "registry order reversed in memory and recompiled",
        },
        "passed": bool(accepted_rows) and not all_disagreements,
        "claim_limit": "Read-only local metamorphic audit; temporary cube variants only. It does not modify runtime/submission or prove unrestricted-language accuracy.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--fail-on-disagreement", action="store_true")
    args = parser.parse_args()
    report = audit(args.registry.resolve(), root=args.root.resolve(), max_cases=args.max_cases)
    output = args.out.resolve()
    output.relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_disagreement and report["disagreement_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
