"""Enrich compact q-source-cell evidence with physical BTC row/header labels.

The source-cell builders preserve document, table line, row and column but use
compact English ``metric_key`` values.  That is enough to replay a number, yet
not enough for a human to spot a semantically wrong row quickly.  This audit
reopens the original extracted HTML table and records the physical label and
header path for every selected operand.

It is read-only.  Findings are review hints, never automatic answer repairs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    from scripts.audit_legacy_source_scope import fold, nearby_context
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from audit_legacy_source_scope import fold, nearby_context


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "financial_statements"

CONTRASTS = [
    ("phải thu", "phải trả"),
    ("ngắn hạn", "dài hạn"),
    ("đầu kỳ", "cuối kỳ"),
    ("đầu năm", "cuối năm"),
    ("trong nước", "nước ngoài"),
    ("nội địa", "xuất khẩu"),
    ("nguyên giá", "giá trị còn lại"),
    ("phát hành", "lưu hành"),
    ("tiền gốc", "tiền lãi"),
]

CONTEXT_CONTRASTS = [
    ("du phong chung", "du phong cu the"),
    ("du phong cu the", "du phong chung"),
    ("hien hanh", "hoan lai"),
    ("hoan lai", "hien hanh"),
    ("ngan han", "dai han"),
    ("dai han", "ngan han"),
    ("phai thu", "phai tra"),
    ("phai tra", "phai thu"),
    ("trong nuoc", "nuoc ngoai"),
    ("nuoc ngoai", "trong nuoc"),
    ("noi dia", "xuat khau"),
    ("xuat khau", "noi dia"),
    ("nguyen gia", "gia tri con lai"),
    ("gia tri con lai", "nguyen gia"),
    ("tien goc", "tien lai"),
    ("tien lai", "tien goc"),
]

# Directional pairs whose two disclosures often reuse the same row labels and
# column layout.  The local section/paragraph ancestry, not the row alone, is
# the discriminating evidence (q638: thuê versus cho thuê).
DIRECTION_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "lease_role": {
        "lessee": (
            r"\bcam ket thue\b",
            r"\bcong ty thue\b",
            r"\btien thue toi thieu phai tra\b",
            r"\bthue dat\b",
        ),
        "lessor": (
            r"\bcam ket cho thue\b",
            r"\bcong ty cho thue\b",
            r"\btien cho thue toi thieu\b",
        ),
    },
    "receivable_payable": {
        "receivable": (r"\bphai thu\b",),
        "payable": (r"\bphai tra\b",),
    },
    "loan_role": {
        "lender": (r"\bcho vay\b", r"\bkhoan cho vay\b"),
        "borrower": (
            r"\bkhoan vay\b",
            r"\bvay ngan han\b",
            r"\bvay dai han\b",
            r"\bno vay\b",
        ),
    },
    "trade_role": {
        "purchase": (r"\bgia tri mua\b", r"\bmua hang(?: hoa)?\b", r"\bchi phi mua\b"),
        "sale": (r"\bgia tri ban\b", r"\bban hang\b", r"\bdoanh thu ban\b"),
    },
}


def _direction_labels(value: object, family: dict[str, tuple[str, ...]]) -> set[str]:
    text = fold(value).replace("_", " ").replace(":", " ")
    # “mua bán” is one combined activity, not simultaneous purchase and sale
    # roles.  Remove it before classifying the trade family.
    text = re.sub(r"\bmua ban\b", " ", text)
    return {
        label
        for label, patterns in family.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def semantic_direction_conflicts(
    question: object, ancestry: object, source_semantic: object = ""
) -> list[str]:
    reasons: list[str] = []
    for family_name, family in DIRECTION_FAMILIES.items():
        requested = _direction_labels(question, family)
        direct = _direction_labels(source_semantic, family)
        # A physical row/metric is stronger than broad note ancestry. If it
        # names a role, never let nearby prose from a sibling section override
        # it (q534). For non-lease families, no direct role means “not
        # applicable”, because a selector phrase in the question may belong to
        # a different DAG stage (q515/q635).
        selected = (
            direct
            if direct or family_name != "lease_role"
            else _direction_labels(ancestry, family)
        )
        if len(requested) == 1 and len(selected) == 1 and requested != selected:
            reasons.append(
                "{0}: question={1} source={2}".format(
                    family_name,
                    next(iter(requested)),
                    next(iter(selected)),
                )
            )
    return reasons

SEPARATE_CONTEXT_RE = re.compile(
    r"\b(?:bao cao tai chinh rieng|bang can doi ke toan rieng|"
    r"bao cao ket qua hoat dong kinh doanh rieng|bao cao tinh hinh tai chinh rieng|"
    r"thuyet minh bao cao tai chinh rieng)\b"
)
CONSOLIDATED_CONTEXT_RE = re.compile(
    r"\b(?:bao cao tai chinh hop nhat|bang can doi ke toan hop nhat|"
    r"bao cao ket qua hoat dong hop nhat|bao cao tinh hinh tai chinh hop nhat|"
    r"thuyet minh bao cao tai chinh hop nhat)\b"
)


def context_conflicts(
    question: object, context: object, source_semantic: object | None = None
) -> list[str]:
    q = fold(question)
    scope = fold(context if source_semantic is None else source_semantic)
    return [
        f"question asks '{expected}' but source context only signals '{opposite}'"
        for expected, opposite in CONTEXT_CONTRASTS
        if expected in q
        and opposite not in q
        and opposite in scope
        and expected not in scope
    ]


SEPARATE_QUESTION_RE = re.compile(
    r"\b(?:cong ty me|ngan hang me|khoi ngan hang me|bctc rieng|"
    r"bao cao tai chinh rieng|xet rieng)\b"
)


def document_scope_conflicts(
    question: object, source_table: object, context: object,
) -> list[str]:
    """Catch a table whose physical masthead contradicts requested scope.

    Some BTC files contain both a parent-only report and a consolidated report.
    A filename/content mismatch alone is not an answer error: in swapped HUT
    containers the physically correct parent table is under ``*_consolidated``.
    Flag only when the physical table also contradicts the question's scope.
    """

    document = str(source_table).split("|", 1)[0].casefold()
    q = fold(question)
    nearby = fold(context)
    signals_separate = bool(SEPARATE_CONTEXT_RE.search(nearby))
    signals_consolidated = bool(CONSOLIDATED_CONTEXT_RE.search(nearby))
    asks_separate = bool(SEPARATE_QUESTION_RE.search(q))
    asks_consolidated = "hop nhat" in q
    if (
        asks_separate
        and not asks_consolidated
        and signals_consolidated
        and not signals_separate
    ):
        return ["question asks parent/separate scope but physical table is consolidated"]
    if (
        not asks_separate
        and document.endswith("_consolidated")
        and signals_separate
        and not signals_consolidated
    ):
        return ["unqualified consolidated question uses a physical separate table"]
    return []


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extracted_path(document: str) -> Path:
    ticker = document.split("_", 1)[0]
    match = re.search(r"_financial_statements_(\d{4})", document)
    if not match:
        raise ValueError(f"cannot infer year from {document}")
    year = match.group(1)
    return DATA / ticker / year / document / f"{document}_extracted.txt"


@lru_cache(maxsize=None)
def local_ancestry(document: str, line_number: int, window: int = 16) -> str:
    """Return text after the closest preceding table within a small window."""

    lines = extracted_path(document).read_text(encoding="utf-8").splitlines()
    before = lines[max(0, line_number - window - 1) : max(0, line_number - 1)]
    last_table = -1
    for index, value in enumerate(before):
        if "<table" in value.casefold():
            last_table = index
    local = before[last_table + 1 :]
    return " | ".join(value.strip() for value in local if value.strip())


@lru_cache(maxsize=None)
def parsed_table(source_table: str) -> pd.DataFrame:
    document, line_text = source_table.rsplit("|", 1)
    line_number = int(line_text)
    path = extracted_path(document)
    lines = path.read_text(encoding="utf-8").splitlines()
    html = lines[line_number - 1]
    tables = pd.read_html(StringIO(html), header=None, keep_default_na=False)
    if not tables:
        raise ValueError("no HTML table parsed")
    return tables[0]


def physical_label(frame: pd.DataFrame, row: int, column: int) -> str:
    """Pick the nearest descriptive cell to the left of a numeric cell."""
    candidates: list[str] = []
    for index in range(min(column - 1, len(frame.columns) - 1), -1, -1):
        value = str(frame.iloc[row, index]).strip()
        if not value or value.casefold() == "nan":
            continue
        # Accounting codes and note numbers are not semantic row labels.
        compact = re.sub(r"[().,%\s+-]", "", value)
        if compact.isdigit() or re.fullmatch(r"\d+[a-z]?", value.casefold()):
            continue
        candidates.append(value)
    return candidates[0] if candidates else ""


def is_financial_number(value: object) -> bool:
    """Return true for a scalar accounting value, not a year/date header."""

    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return False
    compact = re.sub(r"\s+", "", text)
    return bool(re.fullmatch(r"\(?-?\d+(?:[.,]\d+)*\)?%?", compact))


def header_row_indexes(frame: pd.DataFrame, source_row: int) -> list[int]:
    """Locate the leading header band before the first accounting data row.

    ``pandas.read_html(..., header=None)`` expands rowspan/colspan cells but it
    does not label which leading rows are headers.  The previous implementation
    simply took the first three rows, which polluted lineage with values from
    the first two data rows.  A financial data row has at least one scalar
    accounting value; title/date/header rows do not.  Section-only rows between
    the header band and the first data row are harmless because their selected
    column is normally empty, while keeping them preserves merged ancestry.
    """

    indexes: list[int] = []
    for row in range(max(0, min(source_row, len(frame)))):
        values = [frame.iloc[row, column] for column in range(len(frame.columns))]
        if any(is_financial_number(value) for value in values):
            break
        indexes.append(row)
    return indexes


def header_path(frame: pd.DataFrame, column: int, source_row: int) -> list[str]:
    values: list[str] = []
    for row in header_row_indexes(frame, source_row):
        if column >= len(frame.columns):
            break
        value = str(frame.iloc[row, column]).strip()
        if value and value.casefold() != "nan" and value not in values:
            values.append(value)
    return values


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    errors: list[dict] = []
    findings: list[dict] = []

    for item in rows:
        qid = int(item["id"])
        question = norm(item.get("question", ""))
        enriched: list[dict] = []
        for evidence in item.get("evidence", []):
            csv_path = submission / str(evidence.get("csv_path", ""))
            if not csv_path.is_file():
                continue
            try:
                compact = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            except Exception as error:
                errors.append({"id": qid, "csv": str(csv_path), "error": f"read csv: {error}"})
                continue
            required = {"source_table", "row_idx", "col_idx", "raw"}
            if not required.issubset(compact.columns):
                continue
            for source_index, source in compact.iterrows():
                source_table = str(source["source_table"])
                try:
                    frame = parsed_table(source_table)
                    document, line_text = source_table.rsplit("|", 1)
                    row_index = int(source["row_idx"])
                    column_index = int(source["col_idx"])
                    raw_physical = str(frame.iloc[row_index, column_index])
                    label = physical_label(frame, row_index, column_index)
                    headers = header_path(frame, column_index, row_index)
                    context = nearby_context(extracted_path(document), int(line_text))
                    ancestry = local_ancestry(document, int(line_text))
                except Exception as error:
                    errors.append({
                        "id": qid,
                        "csv": str(csv_path.relative_to(submission)),
                        "source_index": int(source_index),
                        "source_table": source_table,
                        "error": f"resolve cell: {type(error).__name__}: {error}",
                    })
                    continue
                enriched.append({
                    "csv": str(csv_path.relative_to(submission)),
                    "source_index": int(source_index),
                    "ticker": source.get("ticker", ""),
                    "year": source.get("year", ""),
                    "metric_key": source.get("metric_key", ""),
                    "source_table": source_table,
                    "row_idx": row_index,
                    "col_idx": column_index,
                    "source_label": label,
                    "header_path": headers,
                    "source_context": context,
                    "local_ancestry": ancestry,
                    "raw_manifest": str(source["raw"]),
                    "raw_physical": raw_physical,
                })

        if not enriched:
            continue
        records.append({
            "id": qid,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "cells": enriched,
        })

        labels = " | ".join(norm(cell["source_label"]) for cell in enriched)
        reasons: list[str] = []
        for expected, opposite in CONTRASTS:
            if expected in question and opposite in labels and expected not in labels:
                reasons.append(f"question asks '{expected}' but source labels only signal '{opposite}'")
        for cell in enriched:
            reasons.extend(
                context_conflicts(
                    item.get("question", ""),
                    cell["local_ancestry"],
                    "{0} {1}".format(cell["source_label"], cell["metric_key"]),
                )
            )
            reasons.extend(
                semantic_direction_conflicts(
                    item.get("question", ""),
                    cell["local_ancestry"],
                    "{0} {1}".format(cell["source_label"], cell["metric_key"]),
                )
            )
            reasons.extend(document_scope_conflicts(
                item.get("question", ""), cell["source_table"], cell["local_ancestry"]
            ))
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            findings.append({
                "id": qid,
                "reasons": reasons,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "source_labels": list(dict.fromkeys(cell["source_label"] for cell in enriched)),
                "source_contexts": list(dict.fromkeys(cell["source_context"] for cell in enriched)),
                "local_ancestries": list(dict.fromkeys(cell["local_ancestry"] for cell in enriched)),
            })

    payload = {
        "submission": str(submission),
        "questions_with_compact_source_cells": len(records),
        "physical_cells_resolved": sum(len(record["cells"]) for record in records),
        "unique_physical_tables": parsed_table.cache_info().currsize,
        "finding_count": len(findings),
        "resolution_error_count": len(errors),
        "policy": "Read-only semantic evidence. Inspect full table and program before changing any answer.",
        "findings": findings,
        "errors": errors,
        "records": records,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"findings", "errors", "records"}}, ensure_ascii=False, indent=2))
    if findings:
        print(json.dumps({"findings": findings}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"first_errors": errors[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
