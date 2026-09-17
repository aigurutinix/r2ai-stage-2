"""Verify source-audit coordinates against the original BTC table CSVs.

The candidate builder records every source token together with a Pandas
``iloc`` coordinate.  This gate reloads the corresponding original table with
the same string-preserving options used by the local grader and proves that
the recorded token is still present at that coordinate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument(
        "--ids",
        default="",
        help="optional comma-separated question IDs; default verifies every audit entry",
    )
    return parser.parse_args()


def source_path(submission_dir: Path, source: dict) -> Path | None:
    direct = submission_dir / "data" / source["csv"]
    if direct.is_file():
        return direct

    document, line = source["table_ref"].split("|", 1)
    table_dir = ROOT / "build" / "tables" / document
    matches = sorted(table_dir.glob(f"*_line{line}.csv"))
    if len(matches) == 1:
        return matches[0]
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    submission_dir = args.submission_dir.resolve()
    requested = {
        int(value.strip()) for value in args.ids.split(",") if value.strip()
    }
    audit_path = submission_dir / "source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if requested:
        audit = [entry for entry in audit if int(entry["id"]) in requested]

    checked = 0
    issues = []
    seen_ids = set()
    cache: dict[Path, pd.DataFrame] = {}
    for entry in audit:
        qid = int(entry["id"])
        seen_ids.add(qid)
        for source in entry.get("sources", []):
            path = source_path(submission_dir, source)
            if path is None:
                issues.append(
                    {
                        "id": qid,
                        "kind": "source-not-found",
                        "table_ref": source.get("table_ref"),
                        "csv": source.get("csv"),
                    }
                )
                continue
            try:
                if path not in cache:
                    cache[path] = pd.read_csv(
                        path,
                        encoding="utf-8-sig",
                        dtype=str,
                        keep_default_na=False,
                        index_col=None,
                    )
                frame = cache[path]
                actual = str(frame.iloc[int(source["row"]), int(source["column"])])
            except Exception as exc:  # report the exact source/coordinate failure
                issues.append(
                    {
                        "id": qid,
                        "kind": "coordinate-error",
                        "path": str(path),
                        "row": source.get("row"),
                        "column": source.get("column"),
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            checked += 1
            expected = str(source.get("raw", ""))
            if actual != expected:
                issues.append(
                    {
                        "id": qid,
                        "kind": "token-mismatch",
                        "path": str(path),
                        "row": source["row"],
                        "column": source["column"],
                        "expected": expected,
                        "actual": actual,
                    }
                )

    missing_ids = sorted(requested - seen_ids)
    for qid in missing_ids:
        issues.append({"id": qid, "kind": "audit-entry-not-found"})

    report = {
        "submission": submission_dir.name,
        "requested_ids": sorted(requested),
        "audited_ids": sorted(seen_ids),
        "source_cells_checked": checked,
        "issues": len(issues),
        "findings": issues,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
