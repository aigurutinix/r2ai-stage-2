"""Replay every accepted registry query, then corrupt its bound operands.

The ordinary compiler regression compares the compiler's in-memory answer with
the verified registry.  This audit exercises the emitted Pandas program in the
real sandbox and then changes every source cell it attested.  A safe program
must reject the changed source instead of silently producing a plausible value.

The registry is used only as a local replay oracle.  This is not hidden-label
validation and the mutations do not predict a private leaderboard score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.sandbox import run_pandas_code  # noqa: E402
from kingpro.product.deterministic_compiler import (  # noqa: E402
    CompiledFinancialQuery,
    DeterministicFinancialCompiler,
)
from kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402


DEFAULT_REGISTRY = (
    ROOT
    / "sub_top123_candidate_v217_missing_panel_operand_batch3"
    / "submission.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build"
    / "demo_compliance"
    / "compiler_registry_mutations_hanoi_v217_v2.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _different_numeric(raw: Any, question_id: int) -> str:
    """Return a simple numeric token that cannot equal the attested raw token."""
    candidates = (
        str(900_000_000_000 + question_id),
        str(800_000_000_000 + question_id),
    )
    digits = "".join(char for char in str(raw) if char.isdigit())
    for candidate in candidates:
        if digits != candidate:
            return candidate
    raise AssertionError("failed to construct a distinct mutation")


def _mutated_csv_paths(
    compiled: CompiledFinancialQuery,
    directory: Path,
    *,
    question_id: int,
) -> tuple[dict[str, str], int]:
    """Copy compiler inputs and replace every attested operand in the copies."""
    ref_to_variable = {
        table_ref: variable
        for variable, table_ref in zip(compiled.csv_paths, compiled.table_refs)
    }
    cells_by_variable: dict[str, list[dict[str, Any]]] = {}
    for cell in compiled.source_cells:
        variable = ref_to_variable.get(str(cell.get("table_ref", "")))
        if variable is None:
            raise ValueError("source cell has no bound compiler variable")
        cells_by_variable.setdefault(variable, []).append(cell)

    mutated_paths: dict[str, str] = {}
    mutated_coordinates: set[tuple[str, int, int]] = set()
    for variable, original in compiled.csv_paths.items():
        source = Path(original)
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        for cell in cells_by_variable.get(variable, []):
            # Pandas row zero follows the CSV header, hence the +1 here.
            csv_row = int(cell["row_idx"]) + 1
            csv_col = int(cell["col_idx"])
            if csv_row <= 0 or csv_row >= len(rows):
                raise ValueError("attested row is outside source CSV")
            if csv_col < 0 or csv_col >= len(rows[csv_row]):
                raise ValueError("attested column is outside source CSV")
            coordinate = (variable, csv_row, csv_col)
            if coordinate in mutated_coordinates:
                continue
            rows[csv_row][csv_col] = _different_numeric(
                cell.get("raw"), question_id
            )
            mutated_coordinates.add(coordinate)

        target = directory / f"{variable}.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        mutated_paths[variable] = str(target)
    return mutated_paths, len(mutated_coordinates)


def audit(
    registry: Path = DEFAULT_REGISTRY,
    *,
    root: Path = ROOT,
    max_cases: int | None = None,
    tolerance: float = 0.0050001,
    execution_timeout: float = 10.0,
) -> dict[str, Any]:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    compiler = DeterministicFinancialCompiler(root)
    by_metric: dict[str, Counter] = {}
    failures: list[dict[str, Any]] = []
    compiled_count = 0
    baseline_passed = 0
    mutation_rejected = 0
    protected_operands = 0

    for row in rows:
        question = str(row.get("question", ""))
        compiled = compiler.compile(question, extract_all_facets(question))
        if compiled is None:
            continue
        if max_cases is not None and compiled_count >= max_cases:
            break
        compiled_count += 1
        question_id = int(row.get("id", compiled_count))
        metric_counts = by_metric.setdefault(compiled.metric, Counter())
        metric_counts["compiled"] += 1

        baseline = run_pandas_code(
            compiled.pandas_query,
            compiled.csv_paths,
            timeout=execution_timeout,
        )
        baseline_value = baseline.get("result")
        baseline_ok = bool(
            baseline.get("ok")
            and isinstance(baseline_value, (int, float))
            and math.isclose(
                float(baseline_value),
                float(compiled.answer),
                rel_tol=0,
                abs_tol=tolerance,
            )
        )
        baseline_passed += int(baseline_ok)
        metric_counts["baseline_passed"] += int(baseline_ok)

        mutation: dict[str, Any]
        mutated_cells = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                mutated_paths, mutated_cells = _mutated_csv_paths(
                    compiled, Path(temporary), question_id=question_id
                )
                mutation = run_pandas_code(
                    compiled.pandas_query,
                    mutated_paths,
                    timeout=execution_timeout,
                )
        except Exception as exc:  # report the exact audit failure; never guess
            mutation = {
                "ok": True,
                "result": None,
                "error": f"audit setup {type(exc).__name__}: {exc}",
            }
        mutation_ok = bool(
            mutated_cells > 0
            and mutation.get("ok") is False
            and mutation.get("result") is None
        )
        mutation_rejected += int(mutation_ok)
        protected_operands += mutated_cells
        metric_counts["mutation_rejected"] += int(mutation_ok)
        metric_counts["protected_operands"] += mutated_cells

        if not baseline_ok or not mutation_ok:
            failures.append(
                {
                    "id": question_id,
                    "metric": compiled.metric,
                    "baseline": baseline,
                    "baseline_expected": compiled.answer,
                    "mutated_cells": mutated_cells,
                    "mutation": mutation,
                }
            )

    passed = bool(
        compiled_count > 0
        and baseline_passed == compiled_count
        and mutation_rejected == compiled_count
        and protected_operands >= compiled_count
        and not failures
    )
    return {
        "schema_version": 1,
        "registry": str(registry.resolve()),
        "input_hashes": {
            "registry_sha256": _sha256(registry),
            "compiler_sha256": _sha256(
                root / "src" / "kingpro" / "product" / "deterministic_compiler.py"
            ),
            "sandbox_sha256": _sha256(
                root / "src" / "kingpro" / "answering" / "sandbox.py"
            ),
        },
        "registry_entries": len(rows),
        "compiled_entries": compiled_count,
        "execution_timeout_seconds": execution_timeout,
        "baseline_replays_passed": baseline_passed,
        "mutated_replays_rejected": mutation_rejected,
        "protected_operand_coordinates": protected_operands,
        "passed": passed,
        "by_metric": {
            metric: dict(sorted(counts.items()))
            for metric, counts in sorted(by_metric.items())
        },
        "failures": failures,
        "claim_limit": (
            "Real-registry sandbox mutation regression over compiler-accepted "
            "questions. It proves bound operands fail closed when changed; it "
            "does not expose private labels or prove unrestricted-table accuracy."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.registry.resolve(),
        root=args.root.resolve(),
        max_cases=args.max_cases,
        execution_timeout=args.timeout,
    )
    output = args.out.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("output must remain inside the project") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_failure and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
