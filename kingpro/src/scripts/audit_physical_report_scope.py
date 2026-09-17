"""Audit the *physical* report scope of every source table used by a submission.

Financial-statement OCR containers are not trustworthy scope labels: a path
ending in ``_consolidated`` can contain a separate/parent report and vice
versa.  This read-only audit walks backwards from the exact serialized table
to the nearest explicit masthead and compares that physical scope with the
question.  Compact source-cell manifests are preferred; resolved terminal
reads are used as a fallback for older evidence.

The output is a review queue.  It never changes a submission.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "financial_statements"


def fold(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text.replace("đ", "d")).strip()


SEPARATE_RE = re.compile(
    r"^(?:bao cao tai chinh rieng|bao cao tai chinh tong hop|"
    r"bao cao tai chinh.{0,45}cua rieng cong ty me|bang can doi ke toan rieng|"
    r"bao cao tinh hinh tai chinh rieng|bao cao ket qua hoat dong kinh doanh rieng|"
    r"bao cao luu chuyen tien te rieng|ban thuyet minh bao cao tai chinh rieng|"
    r"thuyet minh bao cao tai chinh rieng)\b"
)
CONSOLIDATED_RE = re.compile(
    r"^(?:bao cao tai chinh hop nhat|bang can doi ke toan hop nhat|"
    r"bao cao tinh hinh tai chinh hop nhat|bao cao ket qua hoat dong kinh doanh hop nhat|"
    r"bao cao luu chuyen tien te hop nhat|ban thuyet minh bao cao tai chinh hop nhat|"
    r"thuyet minh bao cao tai chinh hop nhat)\b"
)
SEPARATE_QUESTION_RE = re.compile(
    r"\b(?:cong ty me|ngan hang me|khoi ngan hang me|bao cao tai chinh rieng|"
    r"bctc rieng|xet rieng|rieng cua cong ty)\b"
)
CONSOLIDATED_QUESTION_RE = re.compile(r"\b(?:hop nhat|toan tap doan)\b")


def extracted_index() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in CORPUS.rglob("*_extracted.txt"):
        result.setdefault(path.name.removesuffix("_extracted.txt"), path)
    return result


@lru_cache(maxsize=None)
def lines(path_text: str) -> tuple[str, ...]:
    return tuple(Path(path_text).read_text(encoding="utf-8", errors="replace").splitlines())


def line_scope(line: str) -> str:
    normalized = fold(line)
    separate = SEPARATE_RE.search(normalized)
    consolidated = CONSOLIDATED_RE.search(normalized)
    if separate and not consolidated:
        return "separate"
    if consolidated and not separate:
        return "consolidated"
    if separate and consolidated:
        # Parent-only reports are often called "BCTC tổng hợp" and their
        # disclaimer then tells readers to also consult a separately issued
        # consolidated report.  The first mentioned report is the subject of
        # the current container.  Exact "hợp nhất và riêng" boilerplate is too
        # ambiguous and is intentionally left unknown.
        if "bao cao tai chinh tong hop" in normalized:
            return "separate" if separate.start() < consolidated.start() else "consolidated"
    return ""


def nearest_scope(path: Path, table_line: int, window: int = 120) -> dict[str, object]:
    content = lines(str(path))
    start = max(0, table_line - 1 - window)
    for index in range(min(table_line - 2, len(content) - 1), start - 1, -1):
        scope = line_scope(content[index])
        if scope:
            return {
                "scope": scope,
                "scope_line": index + 1,
                "distance": table_line - (index + 1),
                "marker": content[index].strip()[:500],
            }
    return {"scope": "unknown", "scope_line": None, "distance": None, "marker": ""}


def selected_tables(submission: Path, legacy: dict[str, object]) -> dict[int, set[str]]:
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    result: dict[int, set[str]] = {int(row["id"]): set() for row in rows}

    # Current compact manifests are the strongest lineage because they are
    # shipped with the exact candidate and include post-v218 repairs.
    for row in rows:
        qid = int(row["id"])
        for evidence in row.get("evidence", []):
            csv_path = submission / str(evidence.get("csv_path", ""))
            if not csv_path.is_file():
                continue
            header = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:1]
            if not header or "source_table" not in header[0].split(","):
                continue
            import csv
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for source in csv.DictReader(handle):
                    table = str(source.get("source_table", ""))
                    if "|" in table:
                        result[qid].add(table)

    # Fallback for programs whose evidence CSV predates compact lineage.
    for record in legacy.get("records", []):
        qid = int(record["id"])
        if result.get(qid):
            continue
        for read in record.get("terminal_reads", []):
            table = str(read.get("source_table", ""))
            if "|" in table:
                result.setdefault(qid, set()).add(table)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("legacy_audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    legacy = json.loads(args.legacy_audit.read_text(encoding="utf-8"))
    tables_by_id = selected_tables(submission, legacy)
    reports = extracted_index()

    records: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for qid, refs in sorted(tables_by_id.items()):
        question = str(by_id[qid].get("question", ""))
        q = fold(question)
        asks_separate = bool(SEPARATE_QUESTION_RE.search(q))
        asks_consolidated = bool(CONSOLIDATED_QUESTION_RE.search(q))
        expected = (
            "mixed" if asks_separate and asks_consolidated else
            "separate" if asks_separate else
            "consolidated" if asks_consolidated else
            "unqualified"
        )
        sources: list[dict[str, object]] = []
        for ref in sorted(refs):
            document, line_text = ref.rsplit("|", 1)
            path = reports.get(document)
            if path is None:
                unresolved.append({"id": qid, "source_table": ref, "reason": "document_not_found"})
                continue
            physical = nearest_scope(path, int(line_text))
            sources.append({"source_table": ref, **physical})
        if sources:
            records.append({
                "id": qid,
                "question": question,
                "answer": by_id[qid].get("answer"),
                "expected_scope": expected,
                "sources": sources,
            })

    findings: list[dict[str, object]] = []
    for record in records:
        known = [source for source in record["sources"] if source["scope"] != "unknown"]
        scopes = {str(source["scope"]) for source in known}
        expected = str(record["expected_scope"])
        reasons: list[str] = []
        severity = ""
        if expected in {"separate", "consolidated"} and any(
            source["scope"] != expected for source in known
        ):
            reasons.append(f"explicit question scope {expected} conflicts with a physical source")
            severity = "high"
        if expected == "mixed" and known and scopes != {"separate", "consolidated"}:
            reasons.append("explicit mixed-scope question does not bind both physical scopes")
            severity = "high"
        if expected != "mixed" and len(scopes) > 1:
            reasons.append("selected operands mix physical separate and consolidated reports")
            severity = "high" if expected != "unqualified" else "medium"
        # Unqualified questions normally target consolidated figures.  Keep
        # this as a medium review hint: some issuers genuinely publish only a
        # separate report or a metric may exist only there.
        if expected == "unqualified" and scopes == {"separate"}:
            reasons.append("unqualified question is answered only from a physical separate report")
            severity = severity or "medium"
        if reasons:
            findings.append({**record, "severity": severity, "reasons": reasons})

    findings.sort(key=lambda item: (0 if item["severity"] == "high" else 1, int(item["id"])))
    payload = {
        "submission": str(submission),
        "legacy_audit": str(args.legacy_audit.resolve()),
        "questions_with_selected_tables": len(records),
        "selected_table_count": sum(len(record["sources"]) for record in records),
        "unknown_scope_table_count": sum(
            source["scope"] == "unknown" for record in records for source in record["sources"]
        ),
        "unresolved_count": len(unresolved),
        "high_count": sum(item["severity"] == "high" for item in findings),
        "medium_count": sum(item["severity"] == "medium" for item in findings),
        "finding_count": len(findings),
        "policy": "Read-only queue; inspect the full table, formula, and alternate physical report before mutation.",
        "findings": findings,
        "unresolved": unresolved,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"findings", "unresolved", "records"}}, ensure_ascii=False, indent=2))
    print(json.dumps({"finding_ids": [item["id"] for item in findings]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
