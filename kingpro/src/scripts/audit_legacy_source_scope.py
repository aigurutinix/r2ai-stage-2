"""Flag legacy evidence whose disclosed scope is narrower than the question.

This is a read-only semantic audit.  It consumes the resolved terminal reads
from ``audit_legacy_query_sources.py`` and compares the question with:

* the source table's period headers;
* nearby report text before the table; and
* the exact row label/value read by the pandas program.

The detector is deliberately conservative.  It reports evidence for manual
review and never executes or rewrites a submission query.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "financial_statements"


def fold(text: object) -> str:
    value = unicodedata.normalize("NFD", str(text or "").lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value.replace("đ", "d")).strip()


ENDING_RE = re.compile(
    r"\b(cuoi nam|cuoi ky|so du cuoi|31/12|31 thang 12|tai ngay|den ngay)\b"
)
FLOW_HEADER_RE = re.compile(r"\b(nam nay|nam truoc|trong nam|ky nay|ky truoc)\b")
BALANCE_HEADER_RE = re.compile(
    r"\b(so cuoi nam|so dau nam|cuoi nam|dau nam|31[./-]12|01[./-]01)\b"
)
ACQUISITION_RE = re.compile(
    r"\b(tai ngay mua|gia tri hop ly.{0,35}ngay mua|hop nhat kinh doanh|mua cong ty)\b"
)
ACQUISITION_QUALIFIER_RE = re.compile(
    r"\b(ngay mua|gia tri hop ly|hop nhat kinh doanh|mua cong ty)\b"
)
COLLATERAL_RE = re.compile(r"\b(the chap|cam co|tai san bao dam|dam bao cho)\b")
COLLATERAL_NEGATION_RE = re.compile(
    r"\b(chua tinh den tai san the chap|khong co tai san bao dam|khong co tai san dam bao)\b"
)
RELATED_RE = re.compile(r"\b(ben lien quan|giao dich voi cac ben|giao dich voi ben)\b")
RELATED_QUALIFIER_RE = re.compile(r"\b(ben lien quan|giao dich voi)\b")
OPENING_QUALIFIER_RE = re.compile(r"\b(dau nam|dau ky|so du dau|01[./-]01|1 thang 1)\b")
OPENING_CONTEXT_YEAR_RE = re.compile(
    r"\bbang.{0,160}(?:tai ngay )?0?1(?:[./-]0?1[./-]| thang 0?1 nam )(20\d{2})\b"
)
QUESTION_YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class Finding:
    id: int
    rule: str
    severity: str
    question: str
    source_table: str
    source_label: str
    raw: str
    source_header: str
    context: str


def classify(
    *, question: str, source_table: str, source_label: str, raw: str,
    source_header: str, context: str,
) -> list[Finding]:
    """Classify one resolved read.  ID is filled by the caller."""

    q = fold(question)
    header = fold(source_header)
    nearby = fold(context)
    label = fold(source_label)
    result: list[Finding] = []

    def add(rule: str, severity: str) -> None:
        result.append(Finding(
            id=0, rule=rule, severity=severity, question=question,
            source_table=source_table, source_label=source_label, raw=str(raw),
            source_header=source_header, context=context,
        ))

    label_is_ending = bool(ENDING_RE.search(label))
    if (
        ENDING_RE.search(q)
        and FLOW_HEADER_RE.search(header)
        and not BALANCE_HEADER_RE.search(header)
        and not label_is_ending
    ):
        add("ending_question_flow_table", "high")

    # A common old-model failure is to use the comparative opening table for
    # a metric requested simply "năm YYYY".  Report filenames still contain
    # YYYY, so document-year gates cannot see it; the decisive date is in the
    # prose immediately before the serialized table.
    question_years = set(QUESTION_YEAR_RE.findall(q))
    opening_context_years = set(OPENING_CONTEXT_YEAR_RE.findall(nearby))
    if (
        question_years & opening_context_years
        and not OPENING_QUALIFIER_RE.search(q)
    ):
        add("current_year_question_opening_table", "high")

    # A generic "mua công ty con" movement column is not enough: it may sit
    # beside the requested balance or in the preceding table.  In the table
    # header itself require the explicit purchase-date fair-value phrase.
    acquisition_header = bool(re.search(r"gia tri hop ly.{0,45}ngay mua", header))
    acquisition_scope = bool(ACQUISITION_RE.search(nearby) or acquisition_header)
    if acquisition_scope and not ACQUISITION_QUALIFIER_RE.search(q):
        add("unqualified_question_acquisition_table", "high")

    # Collateral words in the exact target row/header are stronger than a
    # remote note paragraph, which may merely describe security for a total.
    collateral_target = bool(
        (COLLATERAL_RE.search(label) or COLLATERAL_RE.search(nearby))
        and not COLLATERAL_NEGATION_RE.search(nearby)
    )
    if collateral_target and not COLLATERAL_RE.search(q):
        add("unqualified_question_collateral_subset", "high")

    # Related-party context is useful but noisier: a question may name the
    # counterparty without literally saying "bên liên quan".  Keep it medium
    # unless the exact row is a generic metric with no named counterparty.
    if RELATED_RE.search(nearby) and not RELATED_QUALIFIER_RE.search(q):
        generic = not any(token in label for token in ("cong ty", "ngan hang", "tap doan"))
        if generic:
            add("unqualified_question_related_party_context", "medium")

    return result


def report_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in CORPUS.rglob("*_extracted.txt"):
        name = path.name.removesuffix("_extracted.txt")
        index.setdefault(name, path)
    return index


def nearby_context(path: Path, line_number: int, window: int = 24) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line_number - 1 - window)
    preceding = [line.strip() for line in lines[start : line_number - 1] if line.strip()]
    # Report mastheads repeat on every page and dilute the semantic heading;
    # the last eight non-empty lines are enough to retain section scope.
    preceding = preceding[-8:]
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
    # Tables are serialized on one line.  Content before the final closing
    # table tag belongs to the previous disclosure, not the target table's
    # section heading.
    if "</table>" in context:
        context = context.rsplit("</table>", 1)[-1]
    return context.strip(" |")


def csv_header(submission: Path, relative_csv: str, rows: int = 2) -> str:
    path = submission / Path(relative_csv)
    if not path.is_file():
        return ""
    return " | ".join(path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:rows])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("legacy_audit", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-high", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.legacy_audit.read_text(encoding="utf-8"))
    reports = report_index()
    findings: list[Finding] = []
    unresolved_reports: list[str] = []

    for record in audit.get("records", []):
        for read in record.get("terminal_reads", []):
            table_ref = str(read.get("source_table", ""))
            if "|" not in table_ref:
                continue
            document, line_text = table_ref.rsplit("|", 1)
            report = reports.get(document)
            if report is None:
                unresolved_reports.append(document)
                context = ""
            else:
                context = nearby_context(report, int(line_text))
            header = csv_header(args.submission_dir, str(read.get("csv", "")))
            matches = classify(
                question=str(record.get("question", "")),
                source_table=table_ref,
                source_label=str(read.get("source_label", "")),
                raw=str(read.get("raw", "")),
                source_header=header,
                context=context,
            )
            findings.extend(Finding(id=int(record["id"]), **{
                key: value for key, value in asdict(item).items() if key != "id"
            }) for item in matches)

    severity_order = {"high": 0, "medium": 1}
    findings.sort(key=lambda item: (severity_order.get(item.severity, 9), item.id, item.rule))
    payload = {
        "submission": str(args.submission_dir.resolve()),
        "legacy_audit": str(args.legacy_audit.resolve()),
        "records_checked": len(audit.get("records", [])),
        "unresolved_report_count": len(set(unresolved_reports)),
        "unresolved_reports": sorted(set(unresolved_reports)),
        "high_count": sum(item.severity == "high" for item in findings),
        "medium_count": sum(item.severity == "medium" for item in findings),
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.fail_on_high and payload["high_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
