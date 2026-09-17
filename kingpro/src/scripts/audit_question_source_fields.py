"""Build a field-level source audit for one submission question.

This checks provenance mechanics (scope, physical table, row label/code,
year/column, raw value and scale).  It never edits a submission and does not
claim that a numerically reproducible program is semantically correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"

EXPECTED_LABELS = {
    "kqkd:01": ("doanh thu ban hang va cung cap dich vu",),
    "cdkt:100": ("tai san ngan han",),
    "cdkt:110": ("tien", "cac khoan tuong duong tien"),
    "cdkt:111": ("tien",),
    "cdkt:140": ("hang ton kho",),
    "cdkt:220": ("tai san co dinh",),
    "cdkt:242": ("chi phi xay dung co ban do dang",),
    "cdkt:300": ("no phai tra",),
    "cdkt:310": ("no ngan han",),
    "cdkt:320": ("vay", "ngan han"),
    "cdkt:400": ("von chu so huu",),
    "kqkd:10": ("doanh thu thuan",),
    "kqkd:11": ("gia von",),
    "kqkd:20": ("loi nhuan gop",),
    "kqkd:23": ("chi phi lai vay",),
    "kqkd:25": ("chi phi ban hang",),
    "kqkd:26": ("chi phi quan ly",),
    "kqkd:50": ("loi nhuan", "truoc thue"),
    "kqkd:51": ("chi phi thue", "hien hanh"),
    "kqkd:60": ("loi nhuan", "sau thue"),
    "lctt:20": ("luu chuyen tien thuan", "hoat dong kinh doanh"),
    "note:short_term_borrowings_ending": ("vay", "ngan han"),
    "note:short_term_supplier_advances": ("tra truoc cho nguoi ban", "ngan han"),
    "note:cash_beginning": ("tien", "dau nam"),
    "note:cash_ending": ("tien", "cuoi nam"),
    "note:credit_provision_expense": ("chi phi du phong rui ro tin dung",),
    "note:current_income_tax_expense_million": ("chi phi thue thu nhap", "hien hanh"),
    "note:owner_invested_capital_ending": ("von dau tu cua chu so huu",),
    "note:premises_tax_ending": ("chi phi thue mat bang", "so du cuoi nam"),
    "note:q195_fx_transaction_commitments_million": ("cam ket giao dich hoi doai",),
    "note:q338_pvoil_supplier_payable": ("tong cong ty dau viet nam", "phai tra nha cung cap"),
    "note:q125_operating_lease_income_commitments": ("tong cong",),
    "note:cip_additions": ("tang",),
    "note:personal_loans": ("vay ca nhan",),
    "note:short_term_loan_provision": ("du phong", "cho vay", "ngan han"),
    "note:q612_short_term_bank_loans": ("vay ngan hang",),
    "note:long_term_land_infra_cost": ("gia von", "dat", "co so ha tang", "cho thue"),
    "note:ending_common_shares_outstanding": ("co phieu", "dang luu hanh"),
}


def repair_mojibake(text: object) -> str:
    value = str(text or "")
    for encoding in ("cp1252", "latin1"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
            if repaired.count("�") <= value.count("�"):
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return value


def fold(text: object) -> str:
    value = repair_mojibake(text).lower().replace("đ", "d")
    value = "".join(ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn")
    # Stray OCR punctuation can split a word (for example ``HỮ'U``).  Turn it
    # into whitespace; semantic matching also compares whitespace-compacted
    # forms, which recovers the intended token without weakening field names.
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_question(submission_dir: Path, qid: int) -> dict:
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    for row in rows:
        if int(row.get("id", -1)) == qid:
            return row
    raise KeyError(f"question {qid} not found")


def infer_scope(question: str) -> str:
    normalized = fold(question)
    if any(token in normalized for token in ("cong ty me", "bao cao rieng", "rieng le")):
        return "separate"
    return "consolidated"


def infer_document_scope(document: str) -> str:
    """Infer report scope while preserving numbered extraction suffixes.

    Some source documents are split into variants such as ``*_separate_1``.
    Treating only a literal ``_separate`` suffix as parent-company scope made
    those documents look unknown and weakened the five-layer scope gate (q104).
    """

    if re.search(r"_separate(?:_\d+)?$", document):
        return "separate"
    if re.search(r"_consolidated(?:_\d+)?$", document):
        return "consolidated"
    return "unknown"


def row_label(cells: list[str], metric_key: str) -> str:
    for cell in cells:
        normalized = fold(cell)
        if semantic_label_ok(metric_key, normalized) is True:
            return cell
    non_numeric = [cell for cell in cells if re.search(r"[A-Za-zÀ-ỹ]", cell)]
    return max(non_numeric, key=len) if non_numeric else ""


def semantic_ancestor_label(
    rows: list[list[str]], row_idx: int, metric_key: str, max_rows: int = 8
) -> str:
    """Find a semantic section ancestor for an unlabeled subtotal row.

    Financial notes often put a section label in a repeated-text row, followed
    by components and then an unlabeled subtotal.  The immediate prior row is
    merely the last component (q89), so it must not be used as the subtotal's
    semantic label when an explicit section ancestor exists nearby.
    """

    for candidate_row in reversed(rows[max(0, row_idx - max_rows) : row_idx]):
        for cell in candidate_row:
            if semantic_label_ok(metric_key, fold(cell)) is True:
                return cell
    return ""


def column_semantic_context(rows: list[list[str]], col_idx: int) -> str:
    """Return the earliest textual column header, excluding pure period/unit cells."""

    period_or_unit = (
        "nam nay",
        "nam truoc",
        "cuoi nam",
        "dau nam",
        "vnd",
        "dong",
        "trieu",
        "nghin",
        "ty",
    )
    for row in rows[:4]:
        if col_idx >= len(row):
            continue
        cell = repair_mojibake(row[col_idx]).strip()
        normalized = fold(cell)
        if not re.search(r"[A-Za-zÀ-ỹ]", cell):
            continue
        if any(token in normalized for token in period_or_unit) and not any(
            token in normalized
            for token in ("chi phi", "doanh thu", "loi nhuan", "tai san", "no ", "von ")
        ):
            continue
        return cell
    return ""


def semantic_label_ok(metric_key: str, normalized_label: str) -> bool | None:
    compact_label = normalized_label.replace(" ", "")
    if metric_key == "cdkt:270":
        # Total assets and total sources are numerically identical by the
        # accounting equation, so value equality cannot validate lineage.
        # Require the asset-side label and explicitly reject code-440 wording.
        if "nguon von" in normalized_label:
            return False
        return "tong tai san" in normalized_label or "tong cong tai san" in normalized_label
    if metric_key == "kqkd:01":
        return (
            "doanh thu" in normalized_label
            and "ban hang" in normalized_label
            and "cung cap dich vu" in normalized_label
            and "thuan" not in normalized_label
        )
    if metric_key == "note:depreciation_amortisation_expense":
        return (
            "chi phi" in normalized_label
            and "khau hao" in normalized_label
            and ("hao mon" in normalized_label or "phan bo" in normalized_label)
        )
    if metric_key == "note:long_term_borrowings":
        return "vay" in normalized_label and "dai han" in normalized_label
    if metric_key == "kqkd:23":
        return "chi phi" in normalized_label and ("lai vay" in normalized_label or "di vay" in normalized_label)
    if metric_key == "kqkd:50":
        return "truoc thue" in normalized_label and ("loi nhuan" in normalized_label or "lo ke toan" in normalized_label)
    if metric_key == "kqkd:60":
        return "sau thue" in normalized_label and ("loi nhuan" in normalized_label or normalized_label.startswith("lo "))
    if metric_key == "note:cash_beginning":
        return "tien" in normalized_label and ("dau nam" in normalized_label or "dau ky" in normalized_label)
    if metric_key == "note:cash_ending":
        return "tien" in normalized_label and ("cuoi nam" in normalized_label or "cuoi ky" in normalized_label)
    expected = EXPECTED_LABELS.get(metric_key)
    if not expected:
        return None
    # Financial PDFs often join words at a line break (for example,
    # ``hoat dongkinh doanh``).  Require every semantic token after removing
    # whitespace as well as in the ordinary normalized form so OCR layout does
    # not become a false source-field failure.
    return all(
        token in normalized_label or token.replace(" ", "") in compact_label
        for token in expected
    )


def year_header(
    rows: list[list[str]], col_idx: int, selected_row_idx: int | None = None
) -> str | None:
    signals = ("nam nay", "nam truoc", "cuoi nam", "dau nam", "31/12", "01/01", "1/1")
    # Extractors can concatenate unrelated logical blocks into one CSV.  The
    # first rows may describe taxes while the selected row belongs to a loan
    # block much later (q612).  Prefer the nearest preceding local header and
    # only then fall back to the global first-six-row convention.
    local_rows: list[list[str]] = []
    if selected_row_idx is not None:
        local_rows = list(
            reversed(rows[max(0, selected_row_idx - 8) : selected_row_idx])
        )
    bands = [local_rows, rows[:6]] if local_rows else [rows[:6]]
    for header_rows in bands:
        for row in header_rows:
            if col_idx >= len(row):
                continue
            cell = fold(row[col_idx])
            # A frequent extracted header is ``2022Trieu VND`` with no word
            # boundary after the year, so a plain four-digit match is intentional.
            if any(signal in cell for signal in signals) or re.search(r"20\d{2}", cell):
                parts = [repair_mojibake(row[col_idx])]
                # Keep unit repair inside the same local logical block; mixing
                # with the global band attached tax headers to loan rows (q612).
                for unit_row in header_rows:
                    if col_idx >= len(unit_row):
                        continue
                    unit = repair_mojibake(unit_row[col_idx])
                    normalized_unit = fold(unit)
                    if unit not in parts and any(token in normalized_unit for token in ("vnd", "dong", "trieu", "nghin", "ty")):
                        parts.append(unit)
                return " | ".join(part for part in parts if part)
    return None


def split_header_column(
    body_rows: list[list[str]], header_rows: list[list[str]], body_col_idx: int
) -> int:
    """Map a body value column onto a narrower split-header fragment.

    Extractors may keep a leading body label column while a detached header
    begins at code/note columns.  The financial value columns remain
    right-aligned; using the same index then reads the prior-year header.
    Shift only for a strictly narrower detached fragment.
    """

    body_width = max((len(row) for row in body_rows), default=0)
    header_width = max((len(row) for row in header_rows), default=0)
    if 0 < header_width < body_width:
        return max(0, body_col_idx - (body_width - header_width))
    return body_col_idx


def header_matches_year(header: str | None, target_year: int, report_year: int | None) -> bool | None:
    if not header:
        return None
    normalized = fold(header)
    if str(target_year) in normalized:
        return True
    if report_year is None:
        return None
    if ("nam nay" in normalized or "cuoi nam" in normalized) and target_year == report_year:
        return True
    opening = any(
        token in normalized
        for token in ("nam truoc", "dau nam", "01/01", "1/1")
    ) or bool(re.search(r"(?:^| )0?1 0?1 20\d{2}(?: |$)", normalized))
    if opening and target_year == report_year - 1:
        return True
    return False


def main() -> None:
    args = parse_args()
    submission_dir = args.submission.resolve()
    question = load_question(submission_dir, args.qid)
    evidence_path = submission_dir / "data" / f"q{args.qid}_source_cells.csv"
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)
    expected_scope = infer_scope(question["question"])
    program_refs = {
        (ticker, int(year), metric)
        for ticker, year, metric in re.findall(
            r"_source_value\(['\"]([^'\"]+)['\"],\s*(\d{4}),\s*['\"]([^'\"]+)['\"]\)",
            question.get("pandas_query", ""),
        )
    }

    records: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    with evidence_path.open(encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    for evidence in evidence_rows:
        ticker = evidence["ticker"]
        year = int(evidence["year"])
        metric_key = evidence["metric_key"]
        key = (ticker, year, metric_key)
        seen.add(key)
        table_ref = evidence["source_table"]
        doc = table_ref.split("|", 1)[0]
        scope = infer_document_scope(doc)
        report_match = re.search(r"_financial_statements_(\d{4})(?:_|$)", doc)
        report_year = int(report_match.group(1)) if report_match else None
        source_dir = ROOT / "build" / "tables" / doc
        source_path = source_dir / Path(evidence["source_csv"]).name
        source_path_fallback = False
        if not source_path.exists() and "|" in table_ref:
            line_number = table_ref.rsplit("|", 1)[1]
            alternatives = list(source_dir.glob(f"*line{line_number}.csv"))
            if len(alternatives) == 1:
                source_path = alternatives[0]
                source_path_fallback = True
        physical_rows: list[list[str]] = []
        if source_path.exists():
            with source_path.open(encoding="utf-8-sig", newline="") as source_handle:
                physical_rows = list(csv.reader(source_handle))
        row_idx = int(evidence["row_idx"]) + 1
        col_idx = int(evidence["col_idx"])
        selected_row = physical_rows[row_idx] if row_idx < len(physical_rows) else []
        selected_raw = selected_row[col_idx] if col_idx < len(selected_row) else None
        context_cells: list[str] = []
        for context_idx in range(max(0, row_idx - 2), row_idx):
            context_cells.extend(physical_rows[context_idx])
        # Prefer the selected physical row.  Falling back to preceding rows is
        # only valid when the selected row has no textual label at all; mixing
        # both first can let a longer bucket label override a selected
        # ``TỔNG CỘNG`` row (q125).
        label = row_label(selected_row, metric_key)
        label_source = str(source_path) if label else None
        ancestor = ""
        if not label or semantic_label_ok(metric_key, fold(label)) is not True:
            ancestor = semantic_ancestor_label(physical_rows, row_idx, metric_key)
        if ancestor:
            label = f"{ancestor} | {label}" if label and ancestor != label else ancestor
            label_source = str(source_path)
        elif not label:
            label = row_label(context_cells, metric_key)
            if label:
                label_source = str(source_path)
        # Some reports split row labels into a one-column table immediately
        # after the numeric value table.  Reattach only the nearest following
        # one-column fragment within ten extracted lines and preserve the same
        # physical row offset (q500 loan components).
        if not label and source_path.exists() and "|" in table_ref:
            try:
                current_line = int(table_ref.rsplit("|", 1)[1])
            except ValueError:
                current_line = -1
            following: list[tuple[int, Path]] = []
            for candidate in source_dir.glob("*line*.csv"):
                match = re.search(r"line(\d+)\.csv$", candidate.name)
                if match and int(match.group(1)) > current_line:
                    following.append((int(match.group(1)), candidate))
            if following:
                following_line, following_path = min(following)
                if following_line - current_line <= 10:
                    with following_path.open(encoding="utf-8-sig", newline="") as following_handle:
                        following_rows = list(csv.reader(following_handle))
                    if following_rows and max(len(row) for row in following_rows) == 1 and row_idx < len(following_rows):
                        following_label = row_label(following_rows[row_idx], metric_key)
                        if following_label:
                            label = following_label
                            label_source = str(following_path)
        column_context = column_semantic_context(physical_rows, col_idx)
        semantic_context = " | ".join(part for part in (label, column_context) if part)
        semantic_ok = semantic_label_ok(metric_key, fold(semantic_context))
        header = year_header(physical_rows, col_idx, selected_row_idx=row_idx)
        header_source = str(source_path) if header else None
        # In matrix-style note tables the year is inherited from the report,
        # while the selected row carries only a relative-period label such as
        # ``Số dư cuối năm`` (q194).  That row label is sufficient to bind the
        # report year and should be checked before searching split headers.
        if not header and any(
            signal in fold(label)
            for signal in ("nam nay", "nam truoc", "cuoi nam", "dau nam", "cuoi ky", "dau ky")
        ):
            header = label
            header_source = str(source_path)
        # Some PDF extractors split a multi-row header into its own tiny table.
        # Reattach only the nearest preceding table when it is on the same
        # document and within ten extracted lines, keeping the fallback local.
        if not header and source_path.exists() and "|" in table_ref:
            try:
                current_line = int(table_ref.rsplit("|", 1)[1])
            except ValueError:
                current_line = -1
            preceding: list[tuple[int, Path]] = []
            for candidate in source_dir.glob("*line*.csv"):
                match = re.search(r"line(\d+)\.csv$", candidate.name)
                if match and int(match.group(1)) < current_line:
                    preceding.append((int(match.group(1)), candidate))
            if preceding:
                previous_line, previous_path = max(preceding)
                if current_line - previous_line <= 10:
                    with previous_path.open(encoding="utf-8-sig", newline="") as previous_handle:
                        previous_rows = list(csv.reader(previous_handle))
                    positional_header = year_header(previous_rows, col_idx)
                    header_col_idx = split_header_column(
                        physical_rows, previous_rows, col_idx
                    )
                    aligned_header = year_header(previous_rows, header_col_idx)
                    # Preserve a same-index header when it already proves the
                    # requested period.  Use right alignment only to repair a
                    # mismatch, never merely because fragment widths differ.
                    if header_matches_year(positional_header, year, report_year) is True:
                        header = positional_header
                    elif header_matches_year(aligned_header, year, report_year) is True:
                        header = aligned_header
                    else:
                        # A shifted header that still does not prove the target
                        # period is worse than unknown: it can attach a legal
                        # citation or prior-year bucket to the selected value.
                        header = positional_header
                    if header:
                        header_source = str(previous_path)
        record = {
            "ticker": ticker,
            "year": year,
            "metric_key": metric_key,
            "scope": scope,
            "scope_ok": scope == expected_scope or scope == "unknown",
            "report_year": report_year,
            "table_ref": table_ref,
            "source_csv": str(source_path),
            "source_exists": source_path.exists(),
            "source_path_fallback": source_path_fallback,
            "row_idx": int(evidence["row_idx"]),
            "col_idx": col_idx,
            "label": repair_mojibake(label),
            "label_source": label_source,
            "column_semantic_context": repair_mojibake(column_context),
            "semantic_context": repair_mojibake(semantic_context),
            "semantic_ok": semantic_ok,
            "year_header": repair_mojibake(header),
            "year_header_source": header_source,
            "year_header_ok": header_matches_year(header, year, report_year),
            "evidence_raw": evidence["raw"],
            "physical_raw": selected_raw,
            "raw_match": selected_raw == evidence["raw"],
            "typed_factor": float(evidence.get("typed_factor") or 1),
            "scale": float(evidence.get("scale") or 1),
            "program_reference": key in program_refs,
        }
        records.append(record)

    missing_refs = sorted(program_refs - seen)
    duplicated = sorted(key for key in seen if sum(1 for row in records if (row["ticker"], row["year"], row["metric_key"]) == key) > 1)
    failures = [
        row for row in records
        if not row["source_exists"]
        or row["scope_ok"] is False
        or row["semantic_ok"] is False
        or row["year_header_ok"] is False
        or row["raw_match"] is False
    ]
    output = args.output or ROOT / "build" / f"v227_q{args.qid}_field_audit_v217.json"
    report = {
        "question_id": args.qid,
        "question": repair_mojibake(question["question"]),
        "current_answer": question["answer"],
        "expected_scope": expected_scope,
        "submission": str(submission_dir),
        "evidence_csv": str(evidence_path),
        "counts": {
            "program_refs": len(program_refs),
            "evidence_cells": len(records),
            "missing_program_refs": len(missing_refs),
            "duplicate_keys": len(duplicated),
            "field_failures": len(failures),
        },
        "missing_program_refs": [list(key) for key in missing_refs],
        "duplicate_keys": [list(key) for key in duplicated],
        "field_failures": failures,
        "records": records,
        "mutation_authority": False,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
