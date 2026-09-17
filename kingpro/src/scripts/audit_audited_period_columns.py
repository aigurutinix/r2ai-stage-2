"""Check source-audited operands against the original table period headers.

``verify_source_audit.py`` proves that every recorded raw token exists at its
declared source coordinate.  This complementary audit asks a different
question: does that coordinate belong to the operand year?  It catches a
perfectly traceable but semantically wrong selection of an adjacent current or
comparative column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from audit_period_columns import descriptor_explicit_year, fold, report_year


ROOT = Path(__file__).resolve().parents[1]


def expected_column_year(
    path: Path, descriptor: str, metric: str = "", report_context: int | None = None,
) -> tuple[int, str] | None:
    """Return the year represented by an unambiguous source column."""

    report = report_context if report_context is not None else report_year(path)
    end_years = {
        int(value)
        for pattern in (
            r"31/12/((?:19|20)\d{2})",
            r"31 thang 12 nam ((?:19|20)\d{2})",
        )
        for value in re.findall(pattern, descriptor)
    }
    start_years = {
        int(value)
        for value in re.findall(r"(?:^|\b)0?1/0?1/((?:19|20)\d{2})", descriptor)
    }
    if len(end_years) == 1:
        explicit = next(iter(end_years))
    elif len(start_years) == 1:
        start = next(iter(start_years))
        explicit = start if "opening" in metric.casefold() else start - 1
    else:
        explicit = descriptor_explicit_year(descriptor)
    if explicit is not None:
        if (
            report is not None
            and "prior" in metric.casefold()
            and explicit == report - 1
        ):
            return report, "prior_operand_report_context"
        return explicit, "explicit_header_year"
    if report is None:
        return None
    # Audit manifests also include wide movement tables where an account name
    # such as "Lợi nhuận sau thuế chưa phân phối năm trước" is a category, not
    # a comparative-period header.  Accept generic roles only at the beginning
    # of a column descriptor, optionally after a unit prefix.
    unit_prefix = r"(?:(?:vnd|trieu dong|nghin vnd|nghin dong)\s+)?"
    role_match = re.match(
        rf"^{unit_prefix}(nam nay|nam truoc|ky nay|ky truoc)(?=\b|trieu|vnd|nghin)", descriptor
    )
    role = None
    if role_match:
        role = "prior" if "truoc" in role_match.group(1) else "current"
    if role == "current":
        return report, "current_year_header"
    if role == "prior":
        return report - 1, "prior_year_header"
    period_match = re.match(
        rf"^{unit_prefix}(so dau nam|so du dau nam|dau ky|so cuoi nam|so du cuoi nam|cuoi ky)\b",
        descriptor,
    )
    period = None
    if period_match:
        period = "opening" if any(
            token in period_match.group(1) for token in ("dau", "opening")
        ) else "ending"
    if period == "ending":
        return report, "ending_header"
    if period == "opening":
        if "opening" in metric.casefold():
            return report, "opening_report_context"
        return report - 1, "opening_header"
    return None


def source_operands(submission_dir: Path, qid: int) -> list[dict]:
    path = submission_dir / "data" / f"q{qid}_source_cells.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_path_for(submission_dir: Path, source: dict) -> Path | None:
    direct = submission_dir / "data" / str(source.get("csv", ""))
    if direct.exists():
        return direct
    table_ref = str(source.get("table_ref", ""))
    if "|" not in table_ref:
        return None
    document, line = table_ref.rsplit("|", 1)
    matches = sorted((ROOT / "build" / "tables" / document).glob(f"*_line{line}.csv"))
    return matches[0] if matches else None


def cell_period_descriptor(path: Path, row: int, column: int) -> str:
    """Return the nearest period-bearing header above a source cell.

    BTC extraction tables can stack two annual blocks vertically under the
    same CSV columns.  Looking only at the first header rows then assigns every
    cell to the earlier block.  Search upward from the exact audited row and
    stop at the closest unambiguous period label instead.
    """

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        data = list(reader)
    # ``row`` is a pandas/DictReader data-row index; account for the CSV field
    # name line at position zero.
    physical_row = row + 1
    flow_section = False
    for candidate in range(physical_row - 1, max(0, physical_row - 16), -1):
        if not (0 <= candidate < len(data) and 0 <= column < len(data[candidate])):
            continue
        descriptor = fold(str(data[candidate][column]))
        if not descriptor:
            continue
        if "trong nam" in descriptor and any(token in descriptor for token in (
            "trich lap", "hoan nhap", "phat sinh", "bien dong",
        )):
            flow_section = True
            continue
        explicit_header = any(token in descriptor for token in (
            "nam tai chinh ket thuc", "tai ngay 31", "vao ngay 31",
            "31/12/", "1/1/", "01/01/",
        )) or re.match(
            r"^(?:nam\s+)?(?:19|20)\d{2}(?=\b|vnd|trieu|nghin)", descriptor
        )
        if explicit_header and descriptor_explicit_year(descriptor) is not None:
            return descriptor
        generic_match = re.match(
            r"^(?:(?:vnd|trieu dong|nghin vnd|nghin dong)\s+)?"
            r"(?:(?:nam nay|nam truoc|ky nay|ky truoc)(?=\b|trieu|vnd|nghin)|"
            r"(?:so dau nam|so du dau nam|dau ky|so cuoi nam|so du cuoi nam|cuoi ky)\b)",
            descriptor,
        )
        if generic_match:
            if flow_section and any(token in descriptor for token in ("so dau", "so cuoi", "dau ky", "cuoi ky")):
                continue
            return descriptor
    return ""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    rows = {
        int(row["id"]): row
        for row in json.loads(
            (args.submission_dir / "submission.json").read_text(encoding="utf-8")
        )
    }
    audit_path = args.submission_dir / "source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    findings: list[dict] = []
    checked = 0
    unresolved = 0
    for record in audit:
        qid = int(record["id"])
        operands = source_operands(args.submission_dir, qid)
        sources = list(record.get("sources", []))
        if len(operands) != len(sources):
            unresolved += 1
            continue
        for index, (operand, source) in enumerate(zip(operands, sources)):
            try:
                operand_year = int(operand["year"])
                column_index = int(source["column"])
            except (KeyError, TypeError, ValueError):
                unresolved += 1
                continue
            source_path = source_path_for(args.submission_dir, source)
            if source_path is None:
                unresolved += 1
                continue
            try:
                source_row = int(source["row"])
            except (KeyError, TypeError, ValueError):
                unresolved += 1
                continue
            descriptor = cell_period_descriptor(
                source_path, source_row, column_index
            )
            expected = expected_column_year(
                source_path,
                descriptor,
                str(source.get("metric", "")),
                report_year(Path(str(source.get("table_ref", "")).split("|", 1)[0] + ".csv")),
            )
            if expected is None:
                continue
            checked += 1
            expected_year, rule = expected
            if operand_year == expected_year:
                continue
            findings.append({
                "id": qid,
                "operand_index": index,
                "operand_year": operand_year,
                "column_year": expected_year,
                "rule": rule,
                "descriptor": descriptor,
                "source_csv": str(source_path),
                "row": source.get("row"),
                "column": column_index,
                "raw": source.get("raw"),
                "metric": source.get("metric"),
                "question": rows.get(qid, {}).get("question"),
                "answer": rows.get(qid, {}).get("answer"),
            })

    payload = {
        "submission": str(args.submission_dir),
        "audited_questions": len(audit),
        "period_operands_checked": checked,
        "unresolved_record_count": unresolved,
        "finding_count": len(findings),
        "findings": findings,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_findings and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
