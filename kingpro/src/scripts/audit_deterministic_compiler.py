"""Audit the constrained live-demo compiler against a verified registry.

The report measures only questions the compiler elects to support.  It is a
local regression check, not a hidden-set or leaderboard claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))

from kingpro.evaluation.metrics import coerce_number  # noqa: E402
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler  # noqa: E402
from kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402


def audit(registry: Path, *, root: Path, tolerance: float = 0.0050001) -> dict:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    compiler = DeterministicFinancialCompiler(root)
    by_metric: dict[str, Counter] = {}
    mismatches: list[dict] = []
    provenance_mismatches: list[dict] = []
    compiled_count = 0
    matched_count = 0
    binding_valid_count = 0
    for row in rows:
        question = str(row.get("question", ""))
        compiled = compiler.compile(question, extract_all_facets(question))
        if compiled is None:
            continue
        compiled_count += 1
        try:
            submission_evidence = compiled.submission_evidence()
        except ValueError as exc:
            submission_evidence = []
            provenance_mismatches.append(
                {"id": row.get("id"), "kind": "binding-error", "detail": str(exc)}
            )
        expected_bindings = list(zip(compiled.csv_paths, compiled.table_refs))
        actual_bindings = [
            (item.get("variable"), item.get("table_ref"))
            for item in submission_evidence
        ]
        if actual_bindings != expected_bindings and submission_evidence:
            provenance_mismatches.append(
                {
                    "id": row.get("id"),
                    "kind": "binding-mismatch",
                    "expected": expected_bindings,
                    "actual": actual_bindings,
                }
            )
        elif actual_bindings == expected_bindings:
            binding_valid_count += 1
        expected = coerce_number(row.get("answer"))
        matched = expected is not None and abs(compiled.answer - expected) <= tolerance
        matched_count += int(matched)
        metric_counts = by_metric.setdefault(compiled.metric, Counter())
        metric_counts["compiled"] += 1
        metric_counts["matched"] += int(matched)
        if not matched:
            mismatches.append(
                {
                    "id": row.get("id"),
                    "metric": compiled.metric,
                    "question": question,
                    "compiled": compiled.answer,
                    "registry_answer": expected,
                }
            )
    precision = matched_count / compiled_count if compiled_count else 0.0
    return {
        "registry": str(registry.resolve()),
        "registry_entries": len(rows),
        "compiled_entries": compiled_count,
        "coverage": compiled_count / len(rows) if rows else 0.0,
        "matched_entries": matched_count,
        "submission_evidence_bindings": binding_valid_count,
        "provenance_mismatches": provenance_mismatches,
        "precision_on_compiled_subset": precision,
        "tolerance_in_requested_unit": tolerance,
        "by_metric": {
            metric: {
                "compiled": counts["compiled"],
                "matched": counts["matched"],
                "precision": counts["matched"] / counts["compiled"],
            }
            for metric, counts in sorted(by_metric.items())
        },
        "mismatches": mismatches,
        "scope_note": (
            "Local regression on compiler-selected questions from the verified registry; "
            "not a hidden-set, private-score or unrestricted-language claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio" / "submission.json",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-compiled", type=int, default=1)
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args()

    report = audit(args.registry, root=args.root.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["compiled_entries"] < args.min_compiled:
        return 2
    if args.require_perfect and (
        report["mismatches"] or report["provenance_mismatches"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
