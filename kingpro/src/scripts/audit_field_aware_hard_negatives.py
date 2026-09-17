"""Find high-precision same-report hard negatives using semantic facets.

The older same-label collision audit deliberately has high recall.  It can
rank unrelated rows because company names, years, and report boilerplate are
mixed with row-label terms.  This read-only audit keeps fields separate.  It
emits a finding only when all of these conditions hold:

* the question explicitly requests exactly one facet in a contrast group;
* the executed source carries a different facet from that group;
* a row in the same report carries the requested facet;
* current and alternative row labels still describe the same core metric; and
* the raw numeric signatures differ.

It never changes a query or submission.  The output is a manual-review queue.
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


FACET_GROUPS: dict[str, dict[str, re.Pattern[str]]] = {
    "governance_role": {
        "board": re.compile(r"\b(hoi dong quan tri|hdqt)\b"),
        "management": re.compile(
            r"\b(ban (?:tong )?giam doc|ban dieu hanh|tong giam doc)\b"
        ),
        "supervisory": re.compile(r"\b(ban kiem soat|bks)\b"),
    },
    "provision_type": {
        "general": re.compile(r"\bdu phong chung\b"),
        "specific": re.compile(r"\bdu phong cu the\b"),
    },
    "term": {
        "short_term": re.compile(r"\bngan han\b"),
        "long_term": re.compile(r"\bdai han\b"),
    },
    "valuation_basis": {
        "gross": re.compile(r"\b(nguyen gia|gia tri gop|truoc du phong)\b"),
        "net": re.compile(r"\b(gia tri con lai|gia tri thuan|sau du phong)\b"),
    },
    "counterparty": {
        "related_party": re.compile(r"\b(ben lien quan|noi bo)\b"),
        "third_party": re.compile(r"\b(ben thu ba|khach hang ben ngoai)\b"),
    },
    "geography": {
        "domestic": re.compile(r"\b(trong nuoc|noi dia)\b"),
        "foreign": re.compile(r"\b(nuoc ngoai|quoc te)\b"),
    },
}


CORE_STOPWORDS = {
    "bao", "bao nhieu", "cao", "cong", "cong ty", "cua", "cho", "cac",
    "co", "phan", "ngan", "hang", "thuong", "mai", "viet", "nam", "tai",
    "chinh", "hop", "nhat", "rieng", "me", "trieu", "nghin", "dong", "ty",
    "vnd", "usd", "eur", "vao", "va", "la", "so", "du", "gia", "tri",
    "cuoi", "dau", "ky", "ngay", "theo", "trong", "den", "bao", "nhieu",
    "hoi", "dong", "quan", "tri", "hdqt", "ban", "tong", "giam", "doc",
    "dieu", "hanh", "kiem", "soat", "bks", "du", "phong", "chung", "cu",
    "the", "ngan", "dai", "han", "nguyen", "con", "lai", "thuan", "sau",
    "truoc", "ben", "lien", "quan", "thu", "ba", "noi", "dia", "nuoc",
    "ngoai", "quoc", "te",
}


def tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fold(value))
        if len(token) >= 3 and token not in CORE_STOPWORDS and not token.isdigit()
    }


def numeric_signature(row: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for cell in row[1:]:
        compact = re.sub(r"\s+", "", str(cell))
        if re.search(r"\d", compact):
            values.append(compact)
    return tuple(values)


def facets(value: object) -> dict[str, set[str]]:
    normalized = fold(value)
    result: dict[str, set[str]] = {}
    for group, patterns in FACET_GROUPS.items():
        matches = {name for name, pattern in patterns.items() if pattern.search(normalized)}
        if matches:
            result[group] = matches
    return result


def requested_facets(question: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for group, matches in facets(question).items():
        if len(matches) == 1:
            result[group] = next(iter(matches))
    return result


def effective_group_facets(label: object, context: object, group: str) -> set[str]:
    """Prefer the row label; only fall back to its local section context."""

    label_matches = facets(label).get(group, set())
    if label_matches:
        return label_matches
    return facets(context).get(group, set())


def labels_share_metric(question: object, current: object, alternative: object) -> bool:
    if fold(current) == fold(alternative):
        return True
    q_terms = tokens(question)
    current_terms = tokens(current)
    alternative_terms = tokens(alternative)
    shared = current_terms & alternative_terms & q_terms
    if not shared:
        return False
    denominator = max(1, min(len(current_terms), len(alternative_terms)))
    return len(current_terms & alternative_terms) / denominator >= 0.5


def nearby_context(lines: list[str], line_number: int, window: int = 28) -> str:
    start = max(0, line_number - 1 - window)
    preceding = [line.strip() for line in lines[start : line_number - 1] if line.strip()][-10:]
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


def load_catalog() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ref: dict[str, dict[str, Any]] = {}
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            report = str(row["report_id"])
            ref = str(row["table_ref"])
            by_report[report].append(row)
            by_ref[ref] = row
    return dict(by_report), by_ref


def load_rows(
    report: str,
    catalog: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for meta in catalog.get(report, []):
        path = TABLES / str(meta.get("csv_path", ""))
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            active_parents: dict[str, str] = {}
            for row_number, row in enumerate(csv.reader(handle)):
                if not row:
                    continue
                label = str(row[0]).strip()
                signature = numeric_signature(row)
                # Repeated/merged group rows such as ``Hội đồng Quản trị`` or
                # ``Ban Giám đốc`` carry the semantic role for all following
                # person rows.  Preserve that row hierarchy instead of
                # flattening every person into an unqualified label.
                if not signature:
                    for group, matches in facets(label).items():
                        if len(matches) == 1:
                            active_parents[group] = label
                    continue
                if len(fold(label)) < 4:
                    continue
                parent_labels = list(dict.fromkeys(active_parents.values()))
                result.append(
                    {
                        "table_ref": str(meta["table_ref"]),
                        "row": row_number,
                        "label": label,
                        "row_path": [*parent_labels, label],
                        "row_context": " | ".join(parent_labels),
                        "values": list(signature),
                    }
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--question-ids", help="optional comma-separated IDs")
    args = parser.parse_args()

    audit = json.loads(args.legacy_report.read_text(encoding="utf-8"))
    selected_ids = (
        {int(value) for value in args.question_ids.split(",") if value.strip()}
        if args.question_ids
        else None
    )
    catalog_by_report, catalog_by_ref = load_catalog()
    reports = report_index()
    rows_cache: dict[str, list[dict[str, Any]]] = {}
    context_cache: dict[str, str] = {}
    report_lines: dict[str, list[str]] = {}

    def context(table_ref: str) -> str:
        if table_ref in context_cache:
            return context_cache[table_ref]
        report, line_text = table_ref.rsplit("|", 1)
        path = reports.get(report)
        if path and report not in report_lines:
            report_lines[report] = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        local = (
            nearby_context(report_lines[report], int(line_text))
            if report in report_lines
            else ""
        )
        section = str(catalog_by_ref.get(table_ref, {}).get("section_title", ""))
        result = " | ".join(part for part in (section, local) if part)
        context_cache[table_ref] = result
        return result

    findings: list[dict[str, Any]] = []
    checked = 0
    for record in audit.get("records", []):
        qid = int(record["id"])
        if selected_ids is not None and qid not in selected_ids:
            continue
        question = str(record.get("question", ""))
        requested = requested_facets(question)
        if not requested:
            continue
        for read in record.get("terminal_reads", []):
            current_ref = str(read.get("source_table", ""))
            if "|" not in current_ref:
                continue
            report = current_ref.rsplit("|", 1)[0]
            if report not in rows_cache:
                rows_cache[report] = load_rows(report, catalog_by_report)
            current_label = str(read.get("source_label", ""))
            current_raw = re.sub(r"\s+", "", str(read.get("raw", "")))
            current_row_context = ""
            for candidate in rows_cache[report]:
                if (
                    candidate["table_ref"] == current_ref
                    and fold(candidate["label"]) == fold(current_label)
                    and current_raw in candidate["values"]
                ):
                    current_row_context = str(candidate.get("row_context", ""))
                    break
            current_context = " | ".join(
                part for part in (context(current_ref), current_row_context) if part
            )
            checked += 1

            for group, expected in requested.items():
                current_facets = effective_group_facets(
                    current_label, current_context, group
                )
                if expected in current_facets or not current_facets:
                    continue
                for alternative in rows_cache[report]:
                    alternative_ref = str(alternative["table_ref"])
                    if alternative_ref == current_ref:
                        continue
                    if current_raw in alternative["values"]:
                        continue
                    alternative_label = str(alternative["label"])
                    if not labels_share_metric(question, current_label, alternative_label):
                        continue
                    alternative_context = " | ".join(
                        part
                        for part in (
                            context(alternative_ref),
                            str(alternative.get("row_context", "")),
                        )
                        if part
                    )
                    alternative_facets = effective_group_facets(
                        alternative_label, alternative_context, group
                    )
                    if expected not in alternative_facets:
                        continue
                    findings.append(
                        {
                            "id": qid,
                            "rule": "explicit_facet_opposes_executed_source",
                            "severity": "high",
                            "facet_group": group,
                            "requested_facet": expected,
                            "current_facets": sorted(current_facets),
                            "alternative_facets": sorted(alternative_facets),
                            "question": question,
                            "answer": record.get("answer"),
                            "current": {
                                "table_ref": current_ref,
                                "label": current_label,
                                "raw": read.get("raw"),
                                "context": current_context,
                            },
                            "alternative": {
                                **alternative,
                                "context": alternative_context,
                            },
                        }
                    )

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for finding in findings:
        key = (
            finding["id"],
            finding["facet_group"],
            finding["current"]["table_ref"],
            finding["alternative"]["table_ref"],
            finding["alternative"]["row"],
        )
        unique[key] = finding
    findings = sorted(
        unique.values(),
        key=lambda row: (int(row["id"]), row["facet_group"], row["alternative"]["table_ref"]),
    )
    payload = {
        "kind": "field_aware_hard_negative_review_queue",
        "legacy_report": str(args.legacy_report.resolve()),
        "terminal_reads_checked": checked,
        "finding_count": len(findings),
        "question_count": len({row["id"] for row in findings}),
        "findings": findings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "findings"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
