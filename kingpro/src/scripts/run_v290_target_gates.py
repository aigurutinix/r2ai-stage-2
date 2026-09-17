"""Run deterministic target/full gates for unpackaged V290."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "sub_v290_scope2"
BASELINE = ROOT / "sub_v276_q638_fix"
OUTPUT = ROOT / "build" / "v290_target_gates.json"
ALLOWLIST = ROOT / "config" / "v290_provenance_allowlist.json"
COMPLIANCE_OUTPUT = ROOT / "build" / "v290_compliance_with_allowlist.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(arguments: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"gate failed: {' '.join(arguments)}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def run_json(arguments: list[str]) -> dict:
    return json.loads(run(arguments))


def keyed(path: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in json.loads(path.read_text(encoding="utf-8"))}


def main() -> int:
    if not CANDIDATE.is_dir():
        raise FileNotFoundError(CANDIDATE)
    test_output = run(["-m", "pytest", "scripts/test_v290_scope2.py", "-q"])
    match = re.search(r"(\d+) passed", test_output)
    if match is None or int(match.group(1)) != 5:
        raise AssertionError(f"unexpected test result: {test_output!r}")

    base, candidate = keyed(BASELINE / "submission.json"), keyed(CANDIDATE / "submission.json")
    semantic_diff = []
    for question_id in base:
        fields = sorted(
            field
            for field in set(base[question_id]) | set(candidate[question_id])
            if base[question_id].get(field) != candidate[question_id].get(field)
        )
        if fields:
            semantic_diff.append({"id": question_id, "fields": fields})
    expected_diff = [
        {"id": 98, "fields": ["answer", "relevant_docs", "relevant_tables"]},
        {"id": 764, "fields": ["answer", "relevant_tables"]},
    ]
    if semantic_diff != expected_diff:
        raise AssertionError(f"semantic diff escaped: {semantic_diff}")
    if candidate[714] != base[714]:
        raise AssertionError("q714 changed")
    if candidate[98]["relevant_docs"] != [
        "HUT_financial_statements_2024_separate",
        "HUT_financial_statements_2024_consolidated",
    ] or candidate[98]["relevant_tables"] != [
        "HUT_financial_statements_2024_separate|328",
        "HUT_financial_statements_2024_consolidated|325",
    ]:
        raise AssertionError("q98 measured union drifted")

    runtimes = {}
    for name, flags in (
        ("string", []),
        ("typed", ["--typed-dfs"]),
        ("official", ["--official"]),
    ):
        result = run_json(
            ["scripts/grader_check.py", str(CANDIDATE), "--ids", "98,764", *flags]
        )
        if not (
            result["entries"] == result["ran_no_exception"]
            == result["numeric_result"] == result["match_stored_answer"] == 2
            and not result["errors"] and not result["mismatch_ids"]
        ):
            raise AssertionError(f"target runtime {name} failed: {result}")
        runtimes[name] = {
            key: result[key]
            for key in (
                "entries", "with_query", "ran_no_exception", "numeric_result",
                "match_stored_answer", "mode", "errors", "error_ids", "mismatch_ids",
            )
        }
    target_sources = run_json(
        ["scripts/verify_source_audit.py", str(CANDIDATE), "--ids", "98,764"]
    )
    full_sources = run_json(["scripts/verify_source_audit.py", str(CANDIDATE)])
    if target_sources["source_cells_checked"] != 3 or target_sources["issues"]:
        raise AssertionError(f"target source audit failed: {target_sources}")
    if full_sources["issues"]:
        raise AssertionError(f"full source audit failed: {full_sources}")
    compliance = run_json(
        [
            "scripts/check_submission_compliance.py",
            str(CANDIDATE),
            "--out", str(COMPLIANCE_OUTPUT),
            "--allow-relevant-table-order",
            "--provenance-allowlist", str(ALLOWLIST),
        ]
    )
    if not (
        compliance["status"] == "PASS"
        and compliance["raw_issue_count"] == 2
        and compliance["accepted_issue_count"] == 2
        and compliance["issues"] == 0
        and compliance["unused_allowlist_count"] == 0
    ):
        raise AssertionError(f"q98 exact provenance allowlist failed: {compliance}")
    audit_path = CANDIDATE / "v290_scope2_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(audit["physical_proof"]) != 4:
        raise AssertionError("physical proof incomplete")
    if (ROOT / "sub_v290_scope2.zip").exists():
        raise AssertionError("V290 must remain unpackaged")

    report = {
        "schema_version": "v290-target-gates/v1",
        "candidate": CANDIDATE.name,
        "baseline": BASELINE.name,
        "candidate_submission_sha256": sha(CANDIDATE / "submission.json"),
        "builder_and_exact_diff_tests": {"status": "PASS", "tests_passed": 5},
        "semantic_diff": semantic_diff,
        "q98_measured_union": {
            "status": "PASS",
            "relevant_docs": candidate[98]["relevant_docs"],
            "relevant_tables": candidate[98]["relevant_tables"],
        },
        "q714_preservation": {
            "status": "PASS",
            "row_byte_semantics_equal": True,
            "compact_manifest_sha256": sha(CANDIDATE / "data" / "q714_source_cells.csv"),
        },
        "target_runtime": runtimes,
        "target_source_audit": {
            "status": "PASS",
            "audited_ids": target_sources["audited_ids"],
            "source_cells_checked": target_sources["source_cells_checked"],
            "issues": target_sources["issues"],
        },
        "full_source_audit": {
            "status": "PASS",
            "audited_question_count": len(full_sources["audited_ids"]),
            "source_cells_checked": full_sources["source_cells_checked"],
            "issues": full_sources["issues"],
        },
        "provenance_compliance": {
            "status": "PASS",
            "policy": compliance["relevant_table_order_policy"],
            "raw_issue_count": compliance["raw_issue_count"],
            "accepted_issue_count": compliance["accepted_issue_count"],
            "remaining_issues": compliance["issues"],
            "unused_allowlist_count": compliance["unused_allowlist_count"],
            "allowlist": str(ALLOWLIST.relative_to(ROOT)),
            "allowlist_sha256": sha(ALLOWLIST),
        },
        "physical_proof": {
            "status": "PASS",
            "checks": audit["physical_proof"],
            "audit_sha256": sha(audit_path),
        },
        "packaging": {"status": "NOT_PACKAGED", "exists": False},
        "submission": {"status": "NOT_SUBMITTED"},
        "overall": "PASS",
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert reloaded["overall"] == "PASS"
    assert reloaded["semantic_diff"] == expected_diff
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha(OUTPUT),
                "overall": "PASS",
                "runtime_matches": {
                    name: gate["match_stored_answer"] for name, gate in runtimes.items()
                },
                "full_source_cells_checked": full_sources["source_cells_checked"],
                "packaged": False,
                "submitted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
