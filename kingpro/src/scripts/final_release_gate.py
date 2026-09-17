"""Run the non-negotiable release gates for a competition artifact.

This is intentionally a read-only verifier.  It never builds, modifies or
uploads a submission.  A candidate is releasable only if every subprocess,
report invariant, archive check and optional SHA lock passes.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


MAX_UPLOAD_FILENAME_CHARACTERS = 64


def run(command, cwd):
    started = time.time()
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    elapsed = round(time.time() - started, 2)
    if completed.returncode:
        raise RuntimeError(
            "gate failed with exit code {}: {}".format(
                completed.returncode, " ".join(str(part) for part in command)
            )
        )
    return {"command": [str(part) for part in command], "seconds": elapsed}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(report, key, expected):
    actual = report.get(key)
    if actual != expected:
        raise RuntimeError(
            "{} expected {!r}, got {!r}".format(key, expected, actual)
        )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_product_python(explicit):
    """Find the application interpreter without contaminating the grader venv."""

    if explicit:
        return explicit.resolve()

    probe = subprocess.run(
        ["py", "-3.14", "-c", "import sys; print(sys.executable)"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return Path(probe.stdout.strip()).resolve()
    return Path(sys.executable).resolve()


def validate_upload_filename(archive):
    """Enforce the competition portal's upload-field filename limit."""

    length = len(archive.name)
    if length > MAX_UPLOAD_FILENAME_CHARACTERS:
        raise RuntimeError(
            "upload filename has {} characters; portal limit is {}: {}".format(
                length, MAX_UPLOAD_FILENAME_CHARACTERS, archive.name
            )
        )
    return {
        "upload_filename": archive.name,
        "upload_filename_characters": length,
        "upload_filename_limit": MAX_UPLOAD_FILENAME_CHARACTERS,
        "upload_filename_within_limit": True,
    }


def validate_archive(candidate, archive, expected_sha):
    if not archive.exists():
        raise RuntimeError("archive is missing: {}".format(archive))
    upload_name_report = validate_upload_filename(archive)
    digest = sha256(archive)
    if expected_sha and digest != expected_sha.upper():
        raise RuntimeError(
            "SHA-256 mismatch: expected {}, got {}".format(
                expected_sha.upper(), digest
            )
        )

    with zipfile.ZipFile(str(archive), "r") as bundle:
        names = bundle.namelist()
        duplicate_count = len(names) - len(set(names))
        if duplicate_count:
            raise RuntimeError("archive has {} duplicate names".format(duplicate_count))
        corrupt = bundle.testzip()
        if corrupt:
            raise RuntimeError("archive CRC/read failure at {}".format(corrupt))
        if "submission.json" not in names:
            raise RuntimeError("archive does not contain root submission.json")
        packed_submission = json.loads(bundle.read("submission.json").decode("utf-8"))

        lower_names = {}
        for name in names:
            lower_names.setdefault(name.casefold(), []).append(name)
        case_collisions = {
            key: values
            for key, values in lower_names.items()
            if len(values) > 1
        }
        if case_collisions:
            sample = next(iter(case_collisions.values()))
            raise RuntimeError("archive has case-colliding paths: {}".format(sample))

        referenced = []
        for row in packed_submission:
            for evidence in row.get("evidence", []):
                csv_path = evidence.get("csv_path") if isinstance(evidence, dict) else None
                if not isinstance(csv_path, str) or not csv_path:
                    raise RuntimeError("submission contains an invalid evidence csv_path")
                if "\\" in csv_path:
                    raise RuntimeError(
                        "evidence path is not portable: {}".format(csv_path)
                    )
                referenced.append(csv_path)
        missing_references = sorted(set(referenced) - set(names))
        if missing_references:
            raise RuntimeError(
                "archive misses exact evidence paths: {}".format(
                    missing_references[:10]
                )
            )

    disk_submission = load_json(candidate / "submission.json")
    if packed_submission != disk_submission:
        raise RuntimeError("archive submission.json differs from candidate directory")

    return {
        "path": str(archive),
        **upload_name_report,
        "sha256": digest,
        "bytes": archive.stat().st_size,
        "entries": len(names),
        "duplicate_names": 0,
        "case_collisions": 0,
        "crc_and_full_read": "pass",
        "submission_json_matches_directory": True,
        "evidence_references": len(referenced),
        "unique_evidence_references": len(set(referenced)),
        "exact_evidence_paths_present": True,
    }


def parse_protected_artifacts(values, root):
    reports = []
    for value in values or []:
        if "=" not in value:
            raise RuntimeError(
                "protected artifact must use PATH=SHA256: {}".format(value)
            )
        name, expected = value.rsplit("=", 1)
        path = (root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            raise RuntimeError("protected artifact is outside project: {}".format(path))
        if not path.is_file():
            raise RuntimeError("protected artifact is missing: {}".format(path))
        actual = sha256(path)
        if actual != expected.upper():
            raise RuntimeError(
                "protected artifact hash mismatch for {}: expected {}, got {}".format(
                    path, expected.upper(), actual
                )
            )
        reports.append(
            {
                "path": str(path),
                "expected_sha256": expected.upper(),
                "sha256_before": actual,
                "bytes_before": path.stat().st_size,
            }
        )
    return reports


def recheck_protected_artifacts(reports):
    for report in reports:
        path = Path(report["path"])
        actual = sha256(path)
        if actual != report["expected_sha256"]:
            raise RuntimeError("protected artifact changed during gate: {}".format(path))
        report["sha256_after"] = actual
        report["bytes_after"] = path.stat().st_size
        report["unchanged"] = (
            report["sha256_before"] == report["sha256_after"]
            and report["bytes_before"] == report["bytes_after"]
        )


def reconcile_legacy_provenance(legacy_report, compliance_report):
    """Accept only legacy mismatches already matched by the exact allowlist."""

    accepted = set()
    for issue in compliance_report.get("accepted_issues", []):
        if issue.get("kind") != "provenance-tables":
            continue
        detail = issue.get("detail") or {}
        accepted.add(
            (
                issue.get("id"),
                tuple(detail.get("declared") or []),
                tuple(detail.get("actual") or []),
            )
        )
    mismatches = legacy_report.get("provenance_mismatches") or []
    unexpected = []
    adjudicated = []
    for item in mismatches:
        key = (
            item.get("id"),
            tuple(item.get("declared") or []),
            tuple(item.get("actual") or []),
        )
        if key in accepted:
            adjudicated.append(item)
        else:
            unexpected.append(item)
    if unexpected:
        raise RuntimeError(
            "legacy provenance has {} mismatch(es) outside the exact allowlist: {}".format(
                len(unexpected), unexpected[:3]
            )
        )
    return {
        "raw_mismatch_count": len(mismatches),
        "adjudicated_mismatch_count": len(adjudicated),
        "unexpected_mismatch_count": 0,
        "adjudicated_mismatches": adjudicated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--expected-submission-sha256",
        help="optional SHA-256 lock for candidate submission.json",
    )
    parser.add_argument(
        "--audit-ledger",
        type=Path,
        help="durable per-question Markdown ledger that must cover IDs 1..1012",
    )
    parser.add_argument(
        "--provenance-allowlist",
        type=Path,
        help="exact allowlist for independently adjudicated provenance warnings",
    )
    parser.add_argument(
        "--protected-artifact",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="artifact that must exist and remain byte-identical during the gate",
    )
    parser.add_argument(
        "--product-python",
        type=Path,
        help="Interpreter containing product dependencies such as bm25s",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--allow-relevant-table-order",
        action="store_true",
        help=(
            "permit a candidate to reorder the exact compact-manifest table "
            "set; use only with an independently verified ordering audit"
        ),
    )
    parser.add_argument(
        "--require-data-derived-labels",
        action="store_true",
        help="require year-returning programs to derive labels from evidence",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidate = args.submission_dir.resolve()
    if not (candidate / "submission.json").is_file():
        raise SystemExit("missing submission.json in {}".format(candidate))

    archive = (
        args.archive.resolve()
        if args.archive
        else candidate.with_suffix(".zip")
    )
    tag = candidate.name
    report_dir = root / "build" / "release_gate" / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).resolve()
    product_python = resolve_product_python(args.product_python)
    protected_artifacts = parse_protected_artifacts(args.protected_artifact, root)

    legacy = report_dir / "legacy_sources.json"
    reports = {
        "compliance": report_dir / "compliance.json",
        "individual_audit": report_dir / "individual_question_audit.json",
        "python37": report_dir / "python37.json",
        "runtime_attributes": report_dir / "runtime_attribute_hazards.json",
        "audited_period": report_dir / "audited_period.json",
        "legacy": legacy,
        "source_scope": report_dir / "source_scope.json",
        "period": report_dir / "period.json",
        "direct_units": report_dir / "direct_units.json",
        "intent": report_dir / "intent.json",
        "threshold_semantics": report_dir / "threshold_semantics.json",
        "mean_ratio_order": report_dir / "mean_ratio_order.json",
        "rounding_order": report_dir / "rounding_order.json",
        "structural": report_dir / "structural.json",
        "table_family": report_dir / "table_family.json",
        "cross": report_dir / "cross_question.json",
        "cross_currency": report_dir / "cross_currency.json",
    }

    compliance_command = [
        python,
        root / "scripts" / "check_submission_compliance.py",
        candidate,
        "--out",
        reports["compliance"],
    ]
    panel_source_command = [
        python,
        root / "scripts" / "verify_panel_source_cells.py",
        candidate,
    ]
    if args.allow_relevant_table_order:
        compliance_command.append("--allow-relevant-table-order")
        panel_source_command.append("--allow-relevant-table-order")
    if args.provenance_allowlist:
        compliance_command.extend(
            ["--provenance-allowlist", args.provenance_allowlist.resolve()]
        )

    commands = [
        compliance_command,
        [python, root / "scripts" / "grader_check.py", candidate],
        [python, root / "scripts" / "grader_check.py", candidate, "--typed-dfs"],
        [python, "-W", "error", root / "scripts" / "grader_check.py", candidate],
        [python, "-W", "error", root / "scripts" / "grader_check.py", candidate, "--typed-dfs"],
        [python, root / "scripts" / "verify_source_audit.py", candidate],
        panel_source_command,
        [python, root / "scripts" / "audit_metric_codes.py", candidate, "--fail-on-findings"],
        [python, root / "scripts" / "audit_python37_compat.py", candidate, "--out", reports["python37"], "--fail-on-findings"],
        [python, root / "scripts" / "audit_runtime_attribute_hazards.py", candidate, "--out", reports["runtime_attributes"], "--fail-on-findings"],
        [python, root / "scripts" / "audit_audited_period_columns.py", candidate, "--out", reports["audited_period"], "--fail-on-findings"],
        [python, root / "scripts" / "audit_legacy_query_sources.py", candidate, "--out", reports["legacy"]],
        [python, root / "scripts" / "audit_legacy_source_scope.py", candidate, reports["legacy"], "--out", reports["source_scope"], "--fail-on-high"],
        [python, root / "scripts" / "audit_period_columns.py", candidate, "--out", reports["period"]],
        [python, root / "scripts" / "audit_direct_units.py", candidate, "--out", reports["direct_units"]],
        [python, root / "scripts" / "audit_program_intent.py", candidate, "--out", reports["intent"]],
        [python, root / "scripts" / "audit_threshold_comparison_semantics.py", candidate, "--out", reports["threshold_semantics"]],
        [python, root / "scripts" / "audit_mean_ratio_order.py", candidate, "--out", reports["mean_ratio_order"]],
        [python, root / "scripts" / "audit_rounding_order_sensitivity.py", candidate, "--out", reports["rounding_order"]],
        [python, root / "scripts" / "audit_structural_risks.py", candidate, "--out", reports["structural"]],
        [python, root / "scripts" / "audit_table_family_collisions.py", candidate, "--out", reports["table_family"], "--fail-on-findings"],
        [product_python, root / "scripts" / "audit_cross_question_consistency.py", candidate, "--out", reports["cross"]],
        [product_python, root / "scripts" / "audit_cross_question_consistency.py", candidate, "--cross-currency", "--out", reports["cross_currency"]],
    ]
    if args.audit_ledger:
        audit_command = [
            python,
            root / "scripts" / "verify_individual_question_audit.py",
            candidate,
            args.audit_ledger.resolve(),
            "--out",
            reports["individual_audit"],
        ]
        if args.expected_submission_sha256:
            audit_command.extend(
                ["--expected-sha256", args.expected_submission_sha256]
            )
        commands.insert(1, audit_command)
    if args.require_data_derived_labels:
        commands.append(
            [python, root / "scripts" / "audit_data_derived_labels.py", candidate]
        )
    if not args.skip_pytest:
        commands.append([product_python, "-m", "pytest", "-q"])

    completed = []
    for command in commands:
        completed.append(run(command, root))

    compliance_report = load_json(reports["compliance"])
    require_equal(compliance_report, "status", "PASS")
    require_equal(compliance_report, "issues", 0)
    require_equal(compliance_report, "unused_allowlist_count", 0)
    if args.audit_ledger:
        require_equal(load_json(reports["individual_audit"]), "status", "PASS")

    require_equal(load_json(reports["python37"]), "finding_count", 0)
    require_equal(load_json(reports["runtime_attributes"]), "finding_count", 0)
    require_equal(load_json(reports["audited_period"]), "finding_count", 0)

    legacy_report = load_json(reports["legacy"])
    require_equal(legacy_report, "unresolved_row_count", 0)
    legacy_provenance = reconcile_legacy_provenance(
        legacy_report, compliance_report
    )

    source_scope = load_json(reports["source_scope"])
    require_equal(source_scope, "unresolved_report_count", 0)
    require_equal(source_scope, "high_count", 0)

    for key in (
        "period",
        "direct_units",
        "intent",
        "threshold_semantics",
        "mean_ratio_order",
        "rounding_order",
        "structural",
        "table_family",
        "cross",
        "cross_currency",
    ):
        require_equal(load_json(reports[key]), "finding_count", 0)
    require_equal(load_json(reports["rounding_order"]), "execution_error_count", 0)

    archive_report = validate_archive(candidate, archive, args.expected_sha256)
    recheck_protected_artifacts(protected_artifacts)
    final_report = {
        "status": "PASS",
        "submission": str(candidate),
        "grader_interpreter": str(python),
        "product_interpreter": str(product_python),
        "commands": completed,
        "reports": {key: str(path) for key, path in reports.items()},
        "archive": archive_report,
        "protected_artifacts": protected_artifacts,
        "legacy_provenance": legacy_provenance,
        "individual_question_audit_required": bool(args.audit_ledger),
        "provenance_allowlist": (
            str(args.provenance_allowlist.resolve())
            if args.provenance_allowlist
            else None
        ),
        "data_derived_labels_required": args.require_data_derived_labels,
        "relevant_table_order_policy": (
            "membership" if args.allow_relevant_table_order else "exact-order"
        ),
        "note": "Read-only gate; no build or upload was performed.",
    }
    rendered = json.dumps(final_report, ensure_ascii=False, indent=2)
    output = args.out.resolve() if args.out else report_dir / "release.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print("\nFINAL RELEASE GATE: PASS")
    print("Report: {}".format(output))
    print("SHA-256: {}".format(archive_report["sha256"]))


if __name__ == "__main__":
    main()
