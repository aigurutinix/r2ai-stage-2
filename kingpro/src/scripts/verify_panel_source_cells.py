"""Verify every compact panel manifest against the original BTC table cell."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "source_table", "source_csv", "row_idx", "col_idx", "raw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument(
        "--allow-relevant-table-order",
        action="store_true",
        help=(
            "allow submission relevant_tables to reorder the exact manifest "
            "table set; source-cell content and membership remain strict"
        ),
    )
    return parser.parse_args()


def original_source(row: pd.Series) -> Path | None:
    document, line = str(row["source_table"]).split("|", 1)
    table_dir = ROOT / "build" / "tables" / document
    named = table_dir / str(row["source_csv"])
    if named.is_file():
        return named
    matches = sorted(table_dir.glob(f"*_line{line}.csv"))
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    args = parse_args()
    submission_dir = args.submission_dir.resolve()
    submission = {
        int(row["id"]): row
        for row in json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    }
    audit = json.loads((submission_dir / "panel_source_audit.json").read_text(encoding="utf-8"))
    cache: dict[Path, pd.DataFrame] = {}
    issues: list[dict] = []
    checked = 0
    for entry in audit:
        qid = int(entry["id"])
        row = submission.get(qid)
        if row is None:
            issues.append({"id": qid, "kind": "submission-row-missing"})
            continue
        evidence = row.get("evidence") or []
        expected_name = f"q{qid}_source_cells.csv"
        matches = [item for item in evidence if Path(item.get("csv_path", "")).name == expected_name]
        if len(matches) != 1:
            issues.append({"id": qid, "kind": "compact-manifest-count", "count": len(matches)})
            continue
        manifest_path = submission_dir / matches[0]["csv_path"]
        try:
            manifest = pd.read_csv(
                manifest_path,
                encoding="utf-8-sig",
                dtype=str,
                keep_default_na=False,
                index_col=None,
            )
        except Exception as exc:
            issues.append({"id": qid, "kind": "manifest-read", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
        if missing:
            issues.append({"id": qid, "kind": "manifest-columns", "missing": missing})
            continue
        table_refs = list(dict.fromkeys(manifest["source_table"].tolist()))
        submitted_refs = row.get("relevant_tables") or []
        if args.allow_relevant_table_order:
            if Counter(table_refs) != Counter(submitted_refs):
                issues.append({"id": qid, "kind": "relevant-table-membership"})
        elif table_refs != submitted_refs:
            issues.append({"id": qid, "kind": "relevant-table-order"})
        if entry.get("source_cells") is not None and int(entry["source_cells"]) != len(manifest):
            issues.append({"id": qid, "kind": "audit-cell-count", "audit": entry["source_cells"], "manifest": len(manifest)})
        for _, source in manifest.iterrows():
            path = original_source(source)
            if path is None:
                issues.append({"id": qid, "kind": "source-not-found", "source_table": source["source_table"]})
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
                actual = str(cache[path].iloc[int(source["row_idx"]), int(source["col_idx"])])
            except Exception as exc:
                issues.append({"id": qid, "kind": "coordinate-error", "path": str(path), "detail": f"{type(exc).__name__}: {exc}"})
                continue
            checked += 1
            if actual != str(source["raw"]):
                issues.append({
                    "id": qid,
                    "kind": "token-mismatch",
                    "path": str(path),
                    "expected": str(source["raw"]),
                    "actual": actual,
                })
    report = {
        "submission": submission_dir.name,
        "panel_questions": len(audit),
        "source_cells_checked": checked,
        "relevant_table_order_policy": (
            "membership" if args.allow_relevant_table_order else "exact-order"
        ),
        "issues": len(issues),
        "findings": issues[:50],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
