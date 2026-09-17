"""Static checks for the BTC rule that answers must be derived from evidence."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from build_compliance_safe_candidate import evidence_tables, numeric_literal, unique


def result_assignments(tree: ast.AST) -> list[ast.AST]:
    values = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "result" for target in targets):
                values.append(node.value)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument(
        "--allow-relevant-table-order",
        action="store_true",
        help=(
            "allow relevant_tables to reorder the exact evidence table set; "
            "all content, multiplicity and document provenance remain strict"
        ),
    )
    parser.add_argument(
        "--provenance-allowlist",
        type=Path,
        help=(
            "JSON file containing exact, independently audited provenance "
            "issues. Any new issue or stale allowlist entry fails closed."
        ),
    )
    parser.add_argument("--out", type=Path, help="optionally write the JSON report")
    return parser.parse_args()


def issue_key(issue: dict) -> str:
    """Return a stable, exact identity for a compliance issue."""

    return json.dumps(issue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_allowlist(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = payload.get("issues") if isinstance(payload, dict) else payload
    if not isinstance(allowed, list) or not all(isinstance(item, dict) for item in allowed):
        raise ValueError("provenance allowlist must be a JSON list or an object with an issues list")
    keys = [issue_key(item) for item in allowed]
    if len(keys) != len(set(keys)):
        raise ValueError("provenance allowlist contains duplicate entries")
    return allowed


def main() -> None:
    args = parse_args()
    subdir = args.submission_dir
    rows = json.loads((subdir / "submission.json").read_text(encoding="utf-8"))
    issues = []
    checked = 0
    for row in rows:
        code = (row.get("pandas_query") or "").strip()
        if not code:
            continue
        checked += 1
        qid = row.get("id")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            issues.append({"id": qid, "kind": "syntax", "detail": str(exc)})
            continue
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        has_data_dependency = bool({"df", "dfs"} & names)
        if not has_data_dependency:
            issues.append({"id": qid, "kind": "no-data-dependency"})
        assignments = result_assignments(tree)
        direct_literals = [value for value in assignments if numeric_literal(value)]
        if direct_literals:
            issues.append(
                {
                    "id": qid,
                    "kind": "direct-literal-result",
                    "detail": [value.value for value in direct_literals],
                }
            )
        if not has_data_dependency and assignments and all(isinstance(value, ast.Constant) for value in assignments):
            issues.append({"id": qid, "kind": "constant-result"})
        evidence = row.get("evidence") or []
        if not evidence:
            issues.append({"id": qid, "kind": "no-evidence"})
        for item in evidence:
            target = subdir / item.get("csv_path", "")
            if not target.is_file():
                issues.append({"id": qid, "kind": "missing-evidence", "detail": str(target)})
        try:
            actual_tables = evidence_tables(subdir, row)
        except (FileNotFoundError, ValueError, SyntaxError) as exc:
            issues.append({"id": qid, "kind": "untraceable-evidence", "detail": str(exc)})
            continue
        actual_docs = unique([table.split("|", 1)[0] for table in actual_tables])
        declared_tables = row.get("relevant_tables") or []
        declared_docs = row.get("relevant_docs") or []
        tables_match = (
            Counter(declared_tables) == Counter(actual_tables)
            if args.allow_relevant_table_order
            else declared_tables == actual_tables
        )
        if not tables_match:
            issues.append(
                {
                    "id": qid,
                    "kind": "provenance-tables",
                    "detail": {"declared": declared_tables, "actual": actual_tables},
                }
            )
        if declared_docs != actual_docs:
            issues.append(
                {
                    "id": qid,
                    "kind": "provenance-docs",
                    "detail": {"declared": declared_docs, "actual": actual_docs},
                }
            )
    try:
        allowed = load_allowlist(args.provenance_allowlist)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit("invalid provenance allowlist: {}".format(exc))
    actual_by_key = {issue_key(issue): issue for issue in issues}
    allowed_by_key = {issue_key(issue): issue for issue in allowed}
    accepted = [actual_by_key[key] for key in actual_by_key.keys() & allowed_by_key.keys()]
    unexpected = [actual_by_key[key] for key in actual_by_key.keys() - allowed_by_key.keys()]
    unused = [allowed_by_key[key] for key in allowed_by_key.keys() - actual_by_key.keys()]

    by_kind = {}
    for issue in unexpected:
        by_kind[issue["kind"]] = by_kind.get(issue["kind"], 0) + 1
    report = {
        "entries": len(rows),
        "queries_checked": checked,
        "relevant_table_order_policy": (
            "membership" if args.allow_relevant_table_order else "exact-order"
        ),
        "issues": len(unexpected),
        "by_kind": by_kind,
        "samples": unexpected[:20],
        "raw_issue_count": len(issues),
        "accepted_issue_count": len(accepted),
        "accepted_issues": accepted,
        "unused_allowlist_count": len(unused),
        "unused_allowlist": unused,
        "allowlist": str(args.provenance_allowlist.resolve()) if args.provenance_allowlist else None,
        "status": "PASS" if not unexpected and not unused else "FAIL",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(1 if unexpected or unused else 0)


if __name__ == "__main__":
    main()
