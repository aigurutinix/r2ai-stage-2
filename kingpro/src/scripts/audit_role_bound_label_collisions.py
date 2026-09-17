"""Find exact-name reads bound to the wrong governance-role table.

Financial notes often repeat one person's name under the Board, executive
management and supervisory board with different remuneration values.  A plain
row-label match can therefore be lexically exact and semantically wrong.  This
read-only audit requires an explicit role in the question, checks the role
markers of the currently read table, and searches the same report for an exact
label in a table carrying the requested role.

Findings are review candidates only.  The script never rewrites a submission.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "build" / "tables"

ROLE_PATTERNS = {
    "board": re.compile(r"\b(?:hdqt|hoi dong quan tri)\b"),
    "management": re.compile(r"\b(?:ban tong giam doc|ban giam doc|ban dieu hanh)\b"),
    "supervisory": re.compile(r"\bban kiem soat\b"),
}


def fold(text: object) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value.replace("đ", "d")).strip()


def roles(text: object) -> set[str]:
    normalized = fold(text)
    return {name for name, pattern in ROLE_PATTERNS.items() if pattern.search(normalized)}


def load_catalog(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_ref: dict[str, dict[str, Any]] = {}
    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = str(row.get("table_ref", ""))
            report = str(row.get("report_id", ""))
            by_ref[ref] = row
            by_report[report].append(row)
    return by_ref, by_report


def exact_rows(meta: dict[str, Any], label: str) -> list[list[str]]:
    path = TABLES / str(meta.get("csv_path", ""))
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        return []
    target = fold(label)
    mask = frame.iloc[:, 0].map(fold) == target
    return [list(map(str, values)) for values in frame.loc[mask].values.tolist()]


def audit(legacy_path: Path, catalog_path: Path) -> dict[str, Any]:
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    by_ref, by_report = load_catalog(catalog_path)
    findings: list[dict[str, Any]] = []
    checked_role_questions = 0

    for record in legacy.get("records", []):
        requested = roles(record.get("question", ""))
        if not requested:
            continue
        checked_role_questions += 1
        for read in record.get("terminal_reads", []):
            current_ref = str(read.get("source_table", ""))
            current = by_ref.get(current_ref, {})
            current_roles = roles(current.get("search_text", ""))
            if requested & current_roles:
                continue
            report = current_ref.rsplit("|", 1)[0] if "|" in current_ref else ""
            label = str(read.get("source_label", ""))
            alternatives: list[dict[str, Any]] = []
            for meta in by_report.get(report, []):
                table_ref = str(meta.get("table_ref", ""))
                if table_ref == current_ref:
                    continue
                table_roles = roles(meta.get("search_text", ""))
                if not (requested & table_roles):
                    continue
                rows = exact_rows(meta, label)
                if rows:
                    alternatives.append({
                        "table_ref": table_ref,
                        "roles": sorted(table_roles),
                        "section_title": meta.get("section_title", ""),
                        "csv_path": meta.get("csv_path", ""),
                        "exact_label_rows": rows,
                    })
            if not alternatives:
                continue
            findings.append({
                "id": int(record["id"]),
                "question": record.get("question", ""),
                "answer": record.get("answer"),
                "requested_roles": sorted(requested),
                "current": {
                    "table_ref": current_ref,
                    "roles": sorted(current_roles),
                    "source_label": label,
                    "source_row_values": read.get("source_row_values", []),
                    "raw": read.get("raw", ""),
                    "section_title": current.get("section_title", ""),
                },
                "alternatives": alternatives,
                "claim_limit": "Role-bound exact-label collision; source and runtime review required before mutation.",
            })

    findings.sort(key=lambda item: item["id"])
    return {
        "schema_version": 1,
        "legacy_audit": str(legacy_path.resolve()),
        "catalog": str(catalog_path.resolve()),
        "records_checked": len(legacy.get("records", [])),
        "explicit_role_questions_checked": checked_role_questions,
        "finding_count": len(findings),
        "automatic_answer_changes": False,
        "findings": findings,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_audit", type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "build" / "catalog_enriched.jsonl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(args.legacy_audit, args.catalog)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
