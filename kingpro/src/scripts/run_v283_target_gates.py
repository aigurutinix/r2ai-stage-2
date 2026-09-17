"""Run deterministic release-target gates for unpackaged V283."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "sub_v283_hut_scope"
BASELINE = ROOT / "sub_v276_q638_fix"
OUTPUT = ROOT / "build" / "v283_target_gates.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def command(arguments: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"gate failed: {' '.join(arguments)}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def json_command(arguments: list[str]) -> dict:
    return json.loads(command(arguments))


def semantic_diff() -> list[dict]:
    base = {
        int(row["id"]): row
        for row in json.loads((BASELINE / "submission.json").read_text(encoding="utf-8"))
    }
    candidate = {
        int(row["id"]): row
        for row in json.loads((CANDIDATE / "submission.json").read_text(encoding="utf-8"))
    }
    output = []
    for question_id in base:
        fields = sorted(
            key
            for key in set(base[question_id]) | set(candidate[question_id])
            if base[question_id].get(key) != candidate[question_id].get(key)
        )
        if fields:
            output.append({"id": question_id, "fields": fields})
    return output


def main() -> int:
    if not CANDIDATE.is_dir():
        raise FileNotFoundError(CANDIDATE)
    tests = command(["-m", "pytest", "scripts/test_v283_hut_scope.py", "-q"])
    passed_match = re.search(r"(\d+) passed", tests)
    if passed_match is None or int(passed_match.group(1)) != 5:
        raise AssertionError(f"unexpected test result: {tests!r}")

    runtimes = {}
    for name, flags in (
        ("string", []),
        ("typed", ["--typed-dfs"]),
        ("official", ["--official"]),
    ):
        result = json_command(
            ["scripts/grader_check.py", str(CANDIDATE), "--ids", "98,714", *flags]
        )
        if not (
            result["entries"] == 2
            and result["ran_no_exception"] == 2
            and result["numeric_result"] == 2
            and result["match_stored_answer"] == 2
            and not result["errors"]
            and not result["mismatch_ids"]
        ):
            raise AssertionError(f"target runtime failed in {name}: {result}")
        runtimes[name] = {
            key: result[key]
            for key in (
                "entries", "with_query", "ran_no_exception", "numeric_result",
                "match_stored_answer", "mode", "errors", "error_ids", "mismatch_ids",
            )
        }

    target_sources = json_command(
        ["scripts/verify_source_audit.py", str(CANDIDATE), "--ids", "98,714"]
    )
    full_sources = json_command(
        ["scripts/verify_source_audit.py", str(CANDIDATE)]
    )
    if target_sources["source_cells_checked"] != 3 or target_sources["issues"] != 0:
        raise AssertionError(f"target source audit failed: {target_sources}")
    if full_sources["issues"] != 0:
        raise AssertionError(f"full source audit failed: {full_sources}")

    diffs = semantic_diff()
    expected_diff = [
        {"id": 98, "fields": ["answer", "relevant_docs", "relevant_tables"]},
        {"id": 714, "fields": ["answer", "relevant_docs", "relevant_tables"]},
    ]
    if diffs != expected_diff:
        raise AssertionError(f"semantic diff escaped target: {diffs}")
    audit_path = CANDIDATE / "v283_hut_scope_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mastheads = audit["physical_masthead_proof"]
    if len(mastheads["checks"]) != 3:
        raise AssertionError("masthead proof incomplete")
    zip_path = ROOT / "sub_v283_hut_scope.zip"
    if zip_path.exists():
        raise AssertionError("V283 must remain unpackaged")

    report = {
        "schema_version": "v283-target-gates/v1",
        "candidate": CANDIDATE.name,
        "baseline": BASELINE.name,
        "candidate_submission_sha256": sha(CANDIDATE / "submission.json"),
        "builder_overwrite_guard_and_exact_diff_tests": {
            "status": "PASS",
            "tests_passed": 5,
        },
        "semantic_diff": diffs,
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
        "extracted_masthead_proof": {
            "status": "PASS",
            "container_swap": mastheads["container_swap"],
            "checks": mastheads["checks"],
            "audit_sha256": sha(audit_path),
        },
        "packaging": {
            "status": "NOT_PACKAGED",
            "zip_path": "sub_v283_hut_scope.zip",
            "exists": False,
        },
        "submission": {"status": "NOT_SUBMITTED"},
        "overall": "PASS",
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert reloaded["overall"] == "PASS"
    assert reloaded["semantic_diff"] == expected_diff
    assert reloaded["full_source_audit"]["issues"] == 0
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha(OUTPUT),
                "overall": "PASS",
                "target_runtime": {
                    name: result["match_stored_answer"]
                    for name, result in runtimes.items()
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
