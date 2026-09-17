"""Audit source-cell scales against the unit label nearest the cited table.

The older cross-document audit profiles a whole report.  That is useful for
triage, but reports can change units between statements and notes.  This audit
resolves every ``document|line`` source reference to its extracted text and
uses the nearest explicit unit marker before that exact table.

Findings are review candidates only.  A query may deliberately keep the
printed unit or apply its own conversion after reading the source cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "financial_statements"


def plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).lower()
    # Vietnamese đ/Đ is a distinct letter, not a base character plus a
    # combining mark, so NFKD alone does not turn "đồng" into "dong".
    return stripped.replace("đ", "d")


UNIT_PREFIX = r"(?:(?:don vi(?: tinh)?|dvt|unit|currency)\s*[:\-]?\s*)?"
UNIT_SUFFIX = r"\s*[.]?"
SCALED_UNIT_PATTERNS = (
    (
        1_000.0,
        re.compile(
            rf"^{UNIT_PREFIX}(?:(?:ngan|nghin)\s*(?:vnd|dong)|(?:vnd|dong)\s*(?:ngan|nghin)|thousand\s*(?:vnd|dong)){UNIT_SUFFIX}$",
            re.I,
        ),
    ),
    (
        1_000_000.0,
        re.compile(
            rf"^{UNIT_PREFIX}(?:trieu\s*(?:vnd|dong)|(?:vnd|dong)\s*trieu|million\s*(?:vnd|dong)){UNIT_SUFFIX}$",
            re.I,
        ),
    ),
    (
        1_000_000_000.0,
        re.compile(
            rf"^{UNIT_PREFIX}(?:(?<!cong )ty\s*(?:vnd|dong)|(?:vnd|dong)\s*ty|billion\s*(?:vnd|dong)){UNIT_SUFFIX}$",
            re.I,
        ),
    ),
)
BASE_UNIT_PATTERN = re.compile(
    r"^(?:(?:don vi(?: tinh)?|dvt|unit|currency)\s*[:\-]?\s*)?"
    r"(?:vnd|dong|dong viet nam|viet nam dong)\s*[.]?$",
    re.I,
)


def unit_factor_from_marker(text: str) -> float | None:
    """Return a currency factor only for a line that looks like a unit marker."""

    normalized = plain(re.sub(r"<[^>]+>", " ", text))
    normalized = " ".join(normalized.split())
    for factor, pattern in SCALED_UNIT_PATTERNS:
        if pattern.fullmatch(normalized):
            return factor
    if len(normalized) <= 100 and BASE_UNIT_PATTERN.fullmatch(normalized):
        return 1.0
    return None


TABLE_HEADER_UNIT_PATTERNS = (
    (1_000.0, re.compile(r"(?:ngan|nghin)\s*(?:vnd|dong)|(?:vnd|dong)\s*(?:ngan|nghin)(?!\s*han)", re.I)),
    (1_000_000.0, re.compile(r"trieu\s*(?:vnd|dong)|(?:vnd|dong)\s*trieu", re.I)),
    (1_000_000_000.0, re.compile(r"(?<!cong )ty\s*(?:vnd|dong)|(?:vnd|dong)\s*ty(?!\s*le)", re.I)),
)


def unit_factor_from_table_header(text: str) -> float | None:
    """Read unit phrases embedded in the first row of a one-line HTML table."""

    # Units are often placed in a second header row beneath the dates.
    header_rows = "</tr>".join(text.split("</tr>")[:2])
    normalized = plain(re.sub(r"<[^>]+>", " ", header_rows))
    normalized = " ".join(normalized.split())
    for factor, pattern in TABLE_HEADER_UNIT_PATTERNS:
        if pattern.search(normalized):
            return factor
    return None


def nearest_unit_marker(
    lines: list[str], table_line: int, *, max_distance: int = 60
) -> dict | None:
    """Find the nearest explicit marker at or before a 1-based table line."""

    if not lines or table_line < 1:
        return None
    end = min(table_line, len(lines))
    start = max(1, end - max_distance)
    for line_number in range(end, start - 1, -1):
        text = lines[line_number - 1].strip()
        if not text:
            continue
        if "<table" in text.lower():
            factor = unit_factor_from_table_header(text)
        else:
            factor = unit_factor_from_marker(text)
        if factor is not None:
            return {
                "factor": factor,
                "line": line_number,
                "distance": table_line - line_number,
                "text": text,
            }
    return None


def extracted_text_index(data_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in data_root.rglob("*_extracted.txt"):
        document = path.parent.name
        expected = f"{document}_extracted.txt"
        if path.name == expected:
            index[document] = path
    return index


def parse_table_ref(ref: str) -> tuple[str, int] | None:
    if "|" not in ref:
        return None
    document, raw_line = ref.rsplit("|", 1)
    try:
        return document, int(raw_line)
    except ValueError:
        return None


def manifest_sources(candidate: Path, question_id: int) -> list[dict]:
    path = candidate / "data" / f"q{question_id}_source_cells.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "table_ref": row.get("source_table", ""),
            "raw": row.get("raw", ""),
            "scale": row.get("scale", 1.0),
            "typed_factor": row.get("typed_factor", 1.0),
            "metric": row.get("metric_key", ""),
        }
        for row in rows
        if row.get("source_table")
    ]


def load_audit_rows(candidate: Path) -> list[dict]:
    rows = []
    for audit_name in ("source_audit.json", "panel_source_audit.json"):
        path = candidate / audit_name
        if not path.is_file():
            continue
        for source_row in json.loads(path.read_text(encoding="utf-8")):
            row = dict(source_row)
            row["_audit_source"] = audit_name
            if not row.get("sources") and audit_name == "panel_source_audit.json":
                row["sources"] = manifest_sources(candidate, int(row["id"]))
            rows.append(row)
    return rows


def same_scale(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-9)


def audit_candidate(candidate: Path, data_root: Path, max_distance: int) -> dict:
    candidate = candidate.resolve()
    submissions = json.loads(
        (candidate / "submission.json").read_text(encoding="utf-8")
    )
    questions = {int(row["id"]): row.get("question", "") for row in submissions}
    rows = load_audit_rows(candidate)
    index = extracted_text_index(data_root)
    text_cache: dict[str, list[str]] = {}
    resolved_sources = []
    mismatches = []
    missing_documents = set()
    missing_markers = []

    for row in rows:
        question_id = int(row["id"])
        for source in row.get("sources", []):
            ref = str(source.get("table_ref", ""))
            parsed = parse_table_ref(ref)
            if parsed is None:
                continue
            document, table_line = parsed
            path = index.get(document)
            if path is None:
                missing_documents.add(document)
                continue
            if document not in text_cache:
                text_cache[document] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            marker = nearest_unit_marker(
                text_cache[document], table_line, max_distance=max_distance
            )
            base = {
                "id": question_id,
                "question": questions.get(question_id, ""),
                "answer": row.get("answer"),
                "metric": source.get("metric", ""),
                "table_ref": ref,
                "raw": source.get("raw"),
                "actual_scale": float(source.get("scale", 1.0)),
                "typed_factor": float(source.get("typed_factor", 1.0)),
                "audit_source": row.get("_audit_source", ""),
            }
            if marker is None:
                missing_markers.append(base)
                continue
            resolved = {
                **base,
                "local_unit_factor": marker["factor"],
                "unit_marker_line": marker["line"],
                "unit_marker_distance": marker["distance"],
                "unit_marker_text": marker["text"],
            }
            resolved_sources.append(resolved)
            if not same_scale(base["actual_scale"], marker["factor"]):
                mismatches.append(
                    {
                        **resolved,
                        "severity": "triage",
                        "claim_limit": (
                            "Verify query-level conversions and requested output unit "
                            "before changing the answer."
                        ),
                    }
                )

    by_question: dict[int, list[dict]] = {}
    for source in resolved_sources:
        by_question.setdefault(source["id"], []).append(source)
    mixed_local_units = []
    for question_id, sources in sorted(by_question.items()):
        factors = {source["local_unit_factor"] for source in sources}
        companies = {
            source["table_ref"].split("_financial_statements_", 1)[0]
            for source in sources
        }
        if len(factors) > 1 and len(companies) > 1:
            mixed_local_units.append(
                {
                    "id": question_id,
                    "question": questions.get(question_id, ""),
                    "local_unit_factors": sorted(factors),
                    "sources": sources,
                    "severity": "triage",
                }
            )

    return {
        "submission": str(candidate),
        "data_root": str(data_root.resolve()),
        "audit_rows": len(rows),
        "documents_indexed": len(index),
        "sources_with_local_unit": len(resolved_sources),
        "missing_document_count": len(missing_documents),
        "missing_documents": sorted(missing_documents),
        "missing_local_marker_count": len(missing_markers),
        "local_scale_mismatch_count": len(mismatches),
        "local_scale_mismatch_ids": sorted({item["id"] for item in mismatches}),
        "local_scale_mismatches": mismatches,
        "cross_company_mixed_local_unit_count": len(mixed_local_units),
        "cross_company_mixed_local_unit_ids": [
            item["id"] for item in mixed_local_units
        ],
        "cross_company_mixed_local_units": mixed_local_units,
        "claim_limit": (
            "Nearest-table unit triage only. Exact query semantics and requested "
            "answer unit remain authoritative."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-distance", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_candidate(
        args.submission_dir, args.data_root.resolve(), args.max_distance
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
