"""Find same-report, same-label rows whose values differ across table scopes.

Exact lexical retrieval can confidently select the wrong table when several
disclosures reuse a generic row label (``Sá»‘ dÆ° cuá»‘i nÄƒm`` is common).  This
read-only audit starts from executed terminal cells, searches every extracted
table in the same report for the same folded row label, and ranks alternatives
whose section scope better matches the question.  It never rewrites answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold, report_index


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "build" / "tables"
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"


def numeric_signature(cells: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for cell in cells[1:]:
        compact = re.sub(r"\s+", "", str(cell))
        if re.search(r"\d", compact):
            values.append(compact)
    return tuple(values)


def scope_signals(text: object) -> set[str]:
    value = fold(text)
    patterns = {
        "general_provision": r"du phong chung",
        "specific_provision": r"du phong cu the",
        "provision_movement": r"bien dong du phong|thay doi du phong",
        "prior_year": r"nam truoc",
        "current_year": r"trong nam nhu sau|nam nay|nam hien hanh",
        "opening": r"dau nam|dau ky|so du dau",
        "ending": r"cuoi nam|cuoi ky|so du cuoi|31/12",
        "short_term": r"ngan han",
        "long_term": r"dai han",
        "gross": r"nguyen gia|menh gia|gia tri ghi so gop",
        "net": r"gia tri con lai|gia tri thuan|sau du phong",
    }
    return {name for name, pattern in patterns.items() if re.search(pattern, value)}


STOPWORDS = {
    "bao", "nhieu", "cong", "ty", "ngan", "hang", "thuong", "mai", "co",
    "phan", "viet", "nam", "nam", "cua", "la", "trong", "tai", "ngay",
    "den", "theo", "bao", "cao", "tai", "chinh", "hop", "nhat", "me",
    "trieu", "dong", "nghin", "ty", "vao", "va", "so", "du",
}


def content_terms(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fold(text))
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    }


def similar_label(left: object, right: object) -> bool:
    a = content_terms(left)
    b = content_terms(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.8


def collision_priority(
    question: object,
    current_context: object,
    alternative_context: object,
    current_label: object = "",
    alternative_label: object = "",
) -> tuple[int, list[str]]:
    q = fold(question)
    question_scope = scope_signals(question)
    current = scope_signals(current_context)
    alternative = scope_signals(alternative_context)
    score = 0
    reasons: list[str] = []

    pairs = [
        ("general_provision", "specific_provision"),
        ("specific_provision", "general_provision"),
        ("short_term", "long_term"),
        ("long_term", "short_term"),
        ("gross", "net"),
        ("net", "gross"),
    ]
    for expected, opposite in pairs:
        phrase = expected.replace("_", " ")
        # ``expected`` is an internal English signal name while competition
        # questions are Vietnamese.  Rely on the same folded-language signal
        # extractor used for table contexts; retain the literal phrase check
        # for synthetic English fixtures and future bilingual questions.
        if (
            (expected in question_scope or phrase in q)
            and opposite in current
            and expected in alternative
        ):
            score += 10
            reasons.append(f"alternative matches {expected}; current context signals {opposite}")

    if "nam truoc" not in q and "prior_year" in current and "prior_year" not in alternative:
        score += 10
        reasons.append("current table is prior-year scope; alternative is not")

    risk_exemption = re.search(
        r"\bno (?:du tieu chuan|can chu y|duoi tieu chuan|nghi ngo|co kha nang mat von)\b",
        q,
    )
    if (
        "so du" in q
        and "du phong" not in q
        and not risk_exemption
        and "provision_movement" in current
        and "provision_movement" not in alternative
    ):
        score += 8
        reasons.append("asset balance currently comes from provision movement; alternative does not")

    if "cuoi" in q and "opening" in current and "ending" in alternative:
        score += 8
        reasons.append("question asks ending balance; alternative has ending scope")
    if "dau" in q and "ending" in current and "opening" in alternative:
        score += 8
        reasons.append("question asks opening balance; alternative has opening scope")

    q_terms = content_terms(question)
    current_overlap = len(q_terms & content_terms(current_context))
    alternative_overlap = len(q_terms & content_terms(alternative_context))
    if alternative_overlap >= current_overlap + 2:
        gain = min(6, 2 * (alternative_overlap - current_overlap))
        score += gain
        reasons.append(
            f"alternative section gains {alternative_overlap - current_overlap} question terms"
        )

    qualifier_gain = (
        content_terms(alternative_label)
        & q_terms
        - content_terms(current_label)
    )
    if len(qualifier_gain) >= 2:
        gain = min(8, 2 * len(qualifier_gain))
        score += gain
        reasons.append(
            "alternative row adds question qualifiers: " + ", ".join(sorted(qualifier_gain))
        )
    return score, reasons


def load_catalog() -> dict[str, list[dict[str, Any]]]:
    """Index catalog rows by report once; full audit touches hundreds of reports."""

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["report_id"])].append(row)
    return dict(result)


def table_rows(
    report: str,
    catalog: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta in catalog.get(report, []):
        table_ref = str(meta["table_ref"])
        path = TABLES / str(meta.get("csv_path", ""))
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle)):
                if not row:
                    continue
                label = str(row[0]).strip()
                key = fold(label)
                signature = numeric_signature(row)
                if len(key) < 4 or not signature:
                    continue
                index[key].append(
                    {
                        "table_ref": table_ref,
                        "row": row_number,
                        "label": label,
                        "values": list(signature),
                    }
                )
    return index


def nearby_context_from_lines(lines: list[str], line_number: int, window: int = 24) -> str:
    start = max(0, line_number - 1 - window)
    preceding = [line.strip() for line in lines[start : line_number - 1] if line.strip()][-8:]
    heading_index: int | None = None
    for index, line in enumerate(preceding):
        normalized = fold(line)
        if (
            re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", normalized)
            or re.match(r"^[a-z][.)]\s+\S", normalized)
            or normalized.startswith("thuyet minh ve ")
        ):
            heading_index = index
    if heading_index is not None:
        preceding = preceding[heading_index:]
    context = " | ".join(preceding)
    if "</table>" in context:
        context = context.rsplit("</table>", 1)[-1]
    return context.strip(" |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-priority", type=int, default=1)
    parser.add_argument(
        "--question-ids",
        help="optional comma-separated IDs for a targeted audit",
    )
    args = parser.parse_args()

    audit = json.loads(args.legacy_report.read_text(encoding="utf-8"))
    selected_ids = (
        {int(value) for value in args.question_ids.split(",") if value.strip()}
        if args.question_ids
        else None
    )
    catalog = load_catalog()
    reports = report_index()
    row_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    context_cache: dict[str, str] = {}
    report_lines: dict[str, list[str]] = {}

    def context(table_ref: str) -> str:
        if table_ref in context_cache:
            return context_cache[table_ref]
        report, line = table_ref.rsplit("|", 1)
        path = reports.get(report)
        if path and report not in report_lines:
            report_lines[report] = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        result = (
            nearby_context_from_lines(report_lines[report], int(line))
            if report in report_lines
            else ""
        )
        context_cache[table_ref] = result
        return result

    findings: list[dict[str, Any]] = []
    checked = 0
    for record in audit.get("records", []):
        if selected_ids is not None and int(record["id"]) not in selected_ids:
            continue
        for read in record.get("terminal_reads", []):
            current_ref = str(read.get("source_table", ""))
            if "|" not in current_ref:
                continue
            report = current_ref.rsplit("|", 1)[0]
            if report not in row_cache:
                row_cache[report] = table_rows(report, catalog)
            source_label = read.get("source_label", "")
            source_key = fold(source_label)
            alternatives = list(row_cache[report].get(source_key, []))
            q_terms = content_terms(record.get("question", ""))
            source_terms = content_terms(source_label)
            for label_key, matches in row_cache[report].items():
                added_qualifiers = content_terms(label_key) & q_terms - source_terms
                if label_key != source_key and (
                    similar_label(source_label, label_key) or len(added_qualifiers) >= 2
                ):
                    alternatives.extend(matches)
            checked += 1
            current_raw = re.sub(r"\s+", "", str(read.get("raw", "")))
            for alternative in alternatives:
                alt_ref = str(alternative["table_ref"])
                if alt_ref == current_ref or current_raw in alternative["values"]:
                    continue
                priority, reasons = collision_priority(
                    record.get("question", ""),
                    context(current_ref),
                    context(alt_ref),
                    source_label,
                    alternative.get("label", ""),
                )
                if priority < args.min_priority:
                    continue
                findings.append(
                    {
                        "id": int(record["id"]),
                        "priority": priority,
                        "reasons": reasons,
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "source_label": read.get("source_label"),
                        "current_table": current_ref,
                        "current_raw": read.get("raw"),
                        "current_context": context(current_ref),
                        "alternative": alternative,
                        "alternative_context": context(alt_ref),
                    }
                )

    findings.sort(key=lambda row: (-int(row["priority"]), int(row["id"])))
    payload = {
        "kind": "same_label_value_collision_review_queue",
        "legacy_report": str(args.legacy_report.resolve()),
        "terminal_reads_checked": checked,
        "finding_count": len(findings),
        "question_count": len({row["id"] for row in findings}),
        "findings": findings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "findings"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
