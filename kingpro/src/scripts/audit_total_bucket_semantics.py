"""Flag unqualified totals read from a dimensional bucket column.

Financial-note tables often put maturity, currency, geography, or status
buckets on one row and a ``Tổng cộng`` column at the end.  A label-only
retriever can select the correct row but the wrong column.  This audit accepts
either an executed legacy-source report or a submission directory.  In
submission mode it checks compact ``q*_source_cells.csv`` operands directly
and can additionally merge the legacy terminal reads.  It emits a conservative
review queue and never modifies a submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"
TABLES = ROOT / "build" / "tables"

TOTAL_HEADER_RE = re.compile(r"\b(tong cong|tong so)\b")
BUCKET_HEADER_RE = re.compile(
    r"\b(qua han|trong han|thang|duoi|tren|tu 1|tu 3|tu 5|"
    r"vnd|usd|eur|ngoai te|noi te|trong nuoc|nuoc ngoai|"
    r"mien bac|mien trung|mien nam|ngan han|dai han)\b"
)
UNQUALIFIED_TOTAL_RE = re.compile(
    r"\b(so du|tong|cuoi nam|cuoi ky|tai ngay|la bao nhieu)\b"
)

EXPLICIT_BUCKET_PATTERNS = (
    r"qua\s*han",
    r"trong\s*han",
    r"duoi\s*\d+\s*thang",
    r"den\s*\d+\s*thang",
    r"tu\s*\d+\s*den\s*\d+\s*thang",
    r"tu\s*\d+\s*den\s*\d+\s*nam",
    r"tren\s*\d+\s*nam",
    r"ngan\s*han",
    r"dai\s*han",
    r"trong\s*nuoc",
    r"nuoc\s*ngoai",
    r"mien\s*bac",
    r"mien\s*trung",
    r"mien\s*nam",
    r"\bvnd\b",
    r"\busd\b",
    r"\beur\b",
)

HEADER_STOPWORDS = {
    "tai", "san", "gia", "tri", "so", "du", "cuoi", "nam", "ky",
    "ngay", "trieu", "ty", "dong", "vnd", "usd", "eur", "tong",
    "cong", "trong", "cua", "va", "khac",
}

MERGED_HEADER_UNIT_RE = re.compile(
    r"(?<=[a-z0-9])(?=(?:nghin|trieu|ty|dong|vnd|usd|eur)\b)"
)


def fold_header(value: object) -> str:
    """Fold a header and restore boundaries lost by merged-cell OCR.

    Extracted headers commonly concatenate a label and unit, for example
    ``Tổng cộngTriệu đồng``.  Without the restored boundary neither the total
    nor the bucket detector can recognise the semantic column.
    """

    return MERGED_HEADER_UNIT_RE.sub(" ", fold(value))


def load_catalog_paths(path: Path = CATALOG) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["table_ref"])] = TABLES / str(row["csv_path"])
    return result


def read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [list(row) for row in csv.reader(handle)]


def header_text(rows: list[list[str]], source_row: int, column: int) -> str:
    """Collect textual header cells above a selected data cell."""

    parts: list[str] = []
    for row in rows[:source_row]:
        if column >= len(row):
            continue
        cell = str(row[column]).strip()
        normalized = fold(cell)
        if not normalized or not re.search(r"[a-z]", normalized):
            continue
        # Numeric data with a unit suffix is not a header.
        if re.fullmatch(r"[()\-.,\d\s]+(?:vnd|usd|eur)?", normalized):
            continue
        parts.append(cell)
    return " | ".join(dict.fromkeys(parts[-5:]))


def numeric_cell(value: object) -> bool:
    compact = re.sub(r"\s+", "", str(value))
    return bool(
        compact not in {"", "-", "--"}
        and re.fullmatch(r"\(?-?\d+(?:[.,]\d+)*\)?%?", compact)
    )


def financial_number(value: object) -> Decimal | None:
    compact = re.sub(r"\s+", "", str(value))
    negative = compact.startswith("(") and compact.endswith(")")
    compact = compact.strip("()")
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)*%?", compact):
        return None
    compact = compact.rstrip("%").replace(".", "").replace(",", "")
    try:
        result = Decimal(compact)
    except InvalidOperation:
        return None
    return -result if negative else result


def question_names_selected_dimension(question: object, selected_header: object) -> bool:
    """Return true when the question explicitly names the chosen column."""

    question_folded = fold(question)
    header_folded = fold_header(selected_header)
    for pattern in EXPLICIT_BUCKET_PATTERNS:
        if re.search(pattern, header_folded) and re.search(pattern, question_folded):
            return True

    # Treat typographic variants such as ``Tá»« 1 Ä‘áº¿n3 thÃ¡ng`` and
    # ``1-3 thÃ¡ng`` as the same explicitly requested maturity bucket.
    range_re = re.compile(r"(?:tu\s*)?(\d+)\s*(?:den\s*|-)\s*(\d+)\s*thang")
    header_range = range_re.search(header_folded)
    question_range = range_re.search(question_folded)
    if header_range and question_range and header_range.groups() == question_range.groups():
        return True

    # Distinctive report acronyms such as BOT describe a business segment.
    # Currency units are deliberately excluded.
    acronyms = {
        token.casefold()
        for token in re.findall(r"\b[A-ZÄ]{2,8}\b", str(selected_header))
        if token.casefold() not in {"vnd", "usd", "eur"}
    }
    if any(re.search(rf"\b{re.escape(token)}\b", question_folded) for token in acronyms):
        return True

    # Header paths use ``|`` between levels.  Compare each level separately so
    # a long report title cannot suppress a real bucket mismatch merely because
    # both title and question contain a year.  Only trailing unit tokens are
    # removed; words such as ``tÃ i chÃ­nh`` must stay contiguous.
    for segment in str(selected_header).split("|"):
        segment_folded = fold_header(segment)
        tokens = re.findall(r"[a-z0-9]+", segment_folded)
        while tokens and tokens[-1] in {"trieu", "ty", "dong", "vnd", "usd", "eur"}:
            tokens.pop()
        if len(tokens) < 2 or all(token in HEADER_STOPWORDS for token in tokens):
            continue
        if any(token.isalpha() and token not in HEADER_STOPWORDS for token in tokens):
            compact_segment = "".join(tokens)
            compact_question = re.sub(r"[^a-z0-9]+", "", question_folded)
            if len(compact_segment) >= 6 and compact_segment in compact_question:
                return True
        for size in range(min(7, len(tokens)), 1, -1):
            for start in range(len(tokens) - size + 1):
                phrase_tokens = tokens[start:start + size]
                # A shared period such as ``năm 2022`` does not qualify the
                # selected dimensional bucket.  Require a distinctive word;
                # digits must not turn an otherwise generic period phrase
                # into a semantic match.
                if not any(
                    token.isalpha() and token not in HEADER_STOPWORDS
                    for token in phrase_tokens
                ):
                    continue
                phrase = " ".join(phrase_tokens)
                if re.search(rf"\b{re.escape(phrase)}\b", question_folded):
                    return True
    return False


def suppress_additive_total_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop cases where the program explicitly sums buckets to the total."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        key = (
            finding.get("id"),
            finding.get("source_table"),
            finding.get("source_label"),
            finding.get("total_column"),
            finding.get("total_raw"),
        )
        groups[key].append(finding)

    suppressed: set[int] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        selected = [financial_number(item.get("selected_raw")) for item in group]
        total = financial_number(group[0].get("total_raw"))
        if total is not None and all(value is not None for value in selected):
            if sum(value for value in selected if value is not None) == total:
                suppressed.update(id(item) for item in group)
    return [item for item in findings if id(item) not in suppressed]


def is_total_header(value: object) -> bool:
    parts = [fold_header(part).strip() for part in str(value).split("|")]
    return any(part == "cong" or TOTAL_HEADER_RE.search(part) for part in parts)


def resolve_source_row(rows: list[list[str]], read: dict[str, Any]) -> int | None:
    """Map pandas' zero-based data row back to the physical CSV row.

    ``audit_legacy_query_sources`` records a pandas row index, while the raw
    CSV still contains its header line.  Matching the captured label/value is
    safer than assuming a fixed offset because a few extractors emit multiple
    header rows.
    """

    label = fold(read.get("source_label", ""))
    raw = re.sub(r"\s+", "", str(read.get("raw", "")))
    column = int(read.get("source_column", -1))
    candidates: list[int] = []
    for index, row in enumerate(rows):
        if column < 0 or column >= len(row) or not row:
            continue
        if fold(row[0]) != label:
            continue
        if re.sub(r"\s+", "", str(row[column])) == raw:
            candidates.append(index)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        expected = int(read.get("source_row", -1)) + 1
        return min(candidates, key=lambda index: abs(index - expected))
    return None


def resolve_compact_source_row(
    rows: list[list[str]], source: dict[str, Any]
) -> int | None:
    """Map a compact source-cell row to the catalog CSV's physical row.

    Compact manifests store the zero-based pandas data-row index.  Catalog
    CSVs preserve one leading column-name row, so ``row_idx + 1`` is normally
    exact.  Matching the captured raw value keeps the mapping safe if an
    extractor ever adds or removes a header row.
    """

    try:
        expected = int(source.get("row_idx", -1)) + 1
        column = int(source.get("col_idx", -1))
    except (TypeError, ValueError):
        return None
    if column < 0:
        return None
    raw = re.sub(r"\s+", "", str(source.get("raw", "")))
    if (
        0 <= expected < len(rows)
        and column < len(rows[expected])
        and re.sub(r"\s+", "", str(rows[expected][column])) == raw
    ):
        return expected
    candidates = [
        index
        for index, row in enumerate(rows)
        if column < len(row)
        and re.sub(r"\s+", "", str(row[column])) == raw
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: abs(index - expected))


def compact_findings(
    submission: Path,
    catalog: dict[str, Path],
    table_cache: dict[str, list[list[str]]],
    *,
    include_equal_values: bool = False,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Inspect compact exact-cell operands in a submission directory."""

    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    reads_checked = 0
    errors: list[dict[str, Any]] = []
    for record in rows:
        for evidence in record.get("evidence", []):
            csv_path = submission / str(evidence.get("csv_path", ""))
            if not csv_path.is_file():
                continue
            with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                sources = list(csv.DictReader(handle))
            required = {"source_table", "row_idx", "col_idx", "raw"}
            if not sources or not required.issubset(sources[0]):
                continue
            for source in sources:
                table_ref = str(source.get("source_table", ""))
                source_csv = catalog.get(table_ref)
                if source_csv is None or not source_csv.is_file():
                    errors.append({
                        "id": int(record["id"]),
                        "source_kind": "compact",
                        "source_table": table_ref,
                        "reason": "catalog table CSV not found",
                    })
                    continue
                if table_ref not in table_cache:
                    table_cache[table_ref] = read_csv(source_csv)
                reads_checked += 1
                source_row = resolve_compact_source_row(table_cache[table_ref], source)
                if source_row is None:
                    errors.append({
                        "id": int(record["id"]),
                        "source_kind": "compact",
                        "source_table": table_ref,
                        "row_idx": source.get("row_idx"),
                        "col_idx": source.get("col_idx"),
                        "raw": source.get("raw"),
                        "reason": "compact source row not resolved",
                    })
                    continue
                source_column = int(source["col_idx"])
                conflict = total_bucket_conflict(
                    record.get("question", ""),
                    table_cache[table_ref],
                    source_row,
                    source_column,
                    include_equal_values=include_equal_values,
                )
                if conflict is None:
                    continue
                row = table_cache[table_ref][source_row]
                findings.append(
                    {
                        "id": int(record["id"]),
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "source_kind": "compact",
                        "source_table": table_ref,
                        "source_label": row[0] if row else "",
                        **conflict,
                    }
                )
    return findings, reads_checked, errors


def legacy_findings(
    audit: dict[str, Any],
    catalog: dict[str, Path],
    table_cache: dict[str, list[list[str]]],
    *,
    include_equal_values: bool = False,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], int]:
    """Inspect terminal reads from ``audit_legacy_query_sources`` output."""

    findings: list[dict[str, Any]] = []
    reads_checked = 0
    errors: list[dict[str, Any]] = []
    non_cell_reads_skipped = 0
    for record in audit.get("records", []):
        for read in record.get("terminal_reads", []):
            if (
                read.get("access_kind") == "selector-exists"
                or read.get("source_column") is None
                or not read.get("source_table")
            ):
                # Existence/count predicates inspect a matched row label, not
                # a numeric bucket cell, so total-vs-bucket semantics do not
                # apply.  Keep the skip visible without calling it an error.
                non_cell_reads_skipped += 1
                continue
            table_ref = str(read.get("source_table", ""))
            csv_path = catalog.get(table_ref)
            if csv_path is None or not csv_path.is_file():
                errors.append({
                    "id": int(record["id"]),
                    "source_kind": "legacy",
                    "source_table": table_ref,
                    "reason": "catalog table CSV not found",
                })
                continue
            if table_ref not in table_cache:
                table_cache[table_ref] = read_csv(csv_path)
            reads_checked += 1
            source_row = resolve_source_row(table_cache[table_ref], read)
            if source_row is None:
                errors.append({
                    "id": int(record["id"]),
                    "source_kind": "legacy",
                    "source_table": table_ref,
                    "source_row": read.get("source_row"),
                    "source_column": read.get("source_column"),
                    "source_label": read.get("source_label"),
                    "raw": read.get("raw"),
                    "reason": "legacy source row not resolved",
                })
                continue
            conflict = total_bucket_conflict(
                record.get("question", ""),
                table_cache[table_ref],
                source_row,
                int(read.get("source_column", -1)),
                include_equal_values=include_equal_values,
            )
            if conflict is None:
                continue
            findings.append(
                {
                    "id": int(record["id"]),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "source_kind": "legacy",
                    "source_table": table_ref,
                    "source_label": read.get("source_label"),
                    **conflict,
                }
            )
    return findings, reads_checked, errors, non_cell_reads_skipped


def total_bucket_conflict(
    question: object,
    rows: list[list[str]],
    source_row: int,
    source_column: int,
    *,
    include_equal_values: bool = False,
) -> dict[str, Any] | None:
    """Return a finding when a bucket was selected instead of total."""

    if source_row >= len(rows) or source_column <= 0:
        return None
    row = rows[source_row]
    if source_column >= len(row):
        return None

    headers = {
        column: header_text(rows, source_row, column)
        for column in range(1, len(row))
    }
    total_columns = [
        column
        for column, text in headers.items()
        if is_total_header(text) and numeric_cell(row[column])
    ]
    if not total_columns or source_column in total_columns:
        return None

    selected_header = headers.get(source_column, "")
    selected_folded = fold_header(selected_header)
    question_folded = fold(question)
    if not BUCKET_HEADER_RE.search(selected_folded):
        return None
    if not UNQUALIFIED_TOTAL_RE.search(question_folded):
        return None

    total_column = total_columns[-1]
    selected_value = financial_number(row[source_column])
    total_value = financial_number(row[total_column])
    if (
        not include_equal_values
        and selected_value is not None
        and selected_value == total_value
    ):
        return None
    if question_names_selected_dimension(question, selected_header):
        return None

    # A question explicitly naming the chosen dimension is not unqualified.
    selected_phrases = [
        phrase
        for phrase in (
            "qua han", "trong han", "duoi 1 thang", "den 1 thang",
            "tu 1 den 3 thang", "tu 3 den 12 thang", "tu 1 den 5 nam",
            "tren 5 nam", "vnd", "usd", "eur", "ngoai te", "noi te",
            "trong nuoc", "nuoc ngoai", "mien bac", "mien trung",
            "mien nam", "ngan han", "dai han",
        )
        if phrase in selected_folded
    ]
    if any(phrase in question_folded for phrase in selected_phrases):
        return None

    return {
        "reason": "unqualified total/balance reads a dimensional bucket column",
        "selected_column": source_column,
        "selected_header": selected_header,
        "selected_raw": row[source_column],
        "total_column": total_column,
        "total_header": headers[total_column],
        "total_raw": row[total_column],
        "source_row_values": row,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="legacy_sources.json or a submission directory",
    )
    parser.add_argument(
        "--legacy-report",
        type=Path,
        help="optional legacy_sources.json to merge in submission mode",
    )
    parser.add_argument(
        "--include-equal-values",
        action="store_true",
        help=(
            "report latent bucket-vs-total semantic mismatches even when the "
            "two cells currently contain the same numeric value"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    catalog = load_catalog_paths()
    table_cache: dict[str, list[list[str]]] = {}
    findings: list[dict[str, Any]] = []
    compact_reads = 0
    legacy_reads = 0
    non_cell_reads_skipped = 0
    errors: list[dict[str, Any]] = []
    input_path = args.input_path.resolve()
    if input_path.is_dir():
        compact, compact_reads, compact_errors = compact_findings(
            input_path,
            catalog,
            table_cache,
            include_equal_values=args.include_equal_values,
        )
        findings.extend(compact)
        errors.extend(compact_errors)
        if args.legacy_report:
            audit = json.loads(args.legacy_report.read_text(encoding="utf-8"))
            legacy, legacy_reads, legacy_errors, non_cell_reads_skipped = legacy_findings(
                audit,
                catalog,
                table_cache,
                include_equal_values=args.include_equal_values,
            )
            findings.extend(legacy)
            errors.extend(legacy_errors)
        input_kind = "submission"
    else:
        audit = json.loads(input_path.read_text(encoding="utf-8"))
        legacy, legacy_reads, errors, non_cell_reads_skipped = legacy_findings(
            audit,
            catalog,
            table_cache,
            include_equal_values=args.include_equal_values,
        )
        findings.extend(legacy)
        input_kind = "legacy_report"

    findings = suppress_additive_total_findings(findings)
    findings.sort(key=lambda item: int(item["id"]))
    payload = {
        "kind": "total_vs_dimensional_bucket_review_queue",
        "input_kind": input_kind,
        "input_path": str(input_path),
        "legacy_report": str(args.legacy_report.resolve()) if args.legacy_report else None,
        "include_equal_values": args.include_equal_values,
        "compact_reads_checked": compact_reads,
        "legacy_terminal_reads_checked": legacy_reads,
        "non_cell_terminal_reads_skipped": non_cell_reads_skipped,
        "reads_checked": compact_reads + legacy_reads,
        "resolution_error_count": len(errors),
        "finding_count": len(findings),
        "question_count": len({item["id"] for item in findings}),
        "findings": findings,
        "resolution_errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "findings"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
