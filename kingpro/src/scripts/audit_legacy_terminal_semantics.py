"""Compare legacy terminal source labels/scope with Vietnamese question intent.

This consumes the exact-cell report produced by ``audit_legacy_query_sources``
and finds high-signal oppositions such as receivable/payable, current/noncurrent,
opening/closing, separate/consolidated, gross/net, and domestic/foreign.

The output is a review queue only.  A lexical conflict can be legitimate when
the program combines several operands, so no automatic repair is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from audit_legacy_source_scope import fold, nearby_context, report_index


CONTRASTS = [
    ("phải thu", "phải trả"),
    ("ngắn hạn", "dài hạn"),
    ("đầu kỳ", "cuối kỳ"),
    ("đầu năm", "cuối năm"),
    ("công ty mẹ", "hợp nhất"),
    ("riêng", "hợp nhất"),
    ("trong nước", "nước ngoài"),
    ("nội địa", "xuất khẩu"),
    ("gộp", "thuần"),
    ("nguyên giá", "giá trị còn lại"),
    ("chi phí", "thu nhập"),
    ("lãi", "lỗ"),
    ("phát hành", "lưu hành"),
    ("tiền gốc", "tiền lãi"),
]

# These concepts usually live in the table/section title rather than in the
# selected row.  q118 is the canonical example: both the general-provision and
# specific-provision movement tables expose a row named ``Số dư cuối kỳ``.  A
# row-label-only audit therefore cannot distinguish them.
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


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_labels(record: dict) -> list[str]:
    result: list[str] = []
    for read in record.get("terminal_reads", []):
        if read.get("source_label") is not None:
            result.append(norm(read["source_label"]))
        result.extend(norm(item) for item in read.get("source_labels", []))
    return list(dict.fromkeys(item for item in result if item))


def context_conflicts(question: object, contexts: list[str]) -> list[str]:
    """Return high-signal section-scope conflicts for manual review."""

    q = fold(question)
    result: list[str] = []
    for context in contexts:
        scope = fold(context)
        for expected, opposite in CONTEXT_CONTRASTS:
            if expected in q and opposite in scope and expected not in scope:
                result.append(
                    f"question asks '{expected}' but table context only signals '{opposite}'"
                )
    return list(dict.fromkeys(result))


RISK_CLASSIFICATION_RE = re.compile(
    r"\b(no (?:du tieu chuan|can chu y|duoi tieu chuan|nghi ngo|co kha nang mat von)|"
    r"rui ro tin dung)\b"
)


def structural_context_conflicts(
    question: object,
    labels: list[str],
    contexts: list[str],
    table_refs: list[str],
) -> list[str]:
    """Catch generic labels whose surrounding table changes their meaning.

    These rules intentionally emit a review queue only.  They target two
    failures that exact-label retrieval cannot see:

    * an asset/bond ``sá»‘ dÆ°`` row nested inside a provision-movement table;
    * a current-report-year question reading the comparative ``nÄƒm trÆ°á»›c``
      movement table from the same report.
    """

    q = fold(question)
    label_text = " | ".join(fold(item) for item in labels)
    context_text = " | ".join(fold(item) for item in contexts)
    result: list[str] = []

    if (
        "so du" in q
        and "du phong" not in q
        and "du phong" not in label_text
        and "bien dong du phong" in context_text
        and not RISK_CLASSIFICATION_RE.search(q)
    ):
        result.append(
            "unqualified balance question reads a row inside a provision-movement table"
        )

    document_years = {
        match.group(1)
        for ref in table_refs
        if (match := re.search(r"_financial_statements_(20\d{2})_", ref))
    }
    question_years = set(re.findall(r"\b20\d{2}\b", q))
    if (
        question_years & document_years
        and "nam truoc" in context_text
        and "nam truoc" not in q
    ):
        result.append(
            "current report-year question reads a table explicitly scoped to the prior year"
        )

    return list(dict.fromkeys(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--legacy-report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    report_path = args.legacy_report or (
        submission.parents[0] / "build" / "release_gate" / submission.name / "legacy_sources.json"
    )
    if not report_path.exists():
        raise SystemExit(f"legacy report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reports = report_index()
    findings: list[dict] = []

    for record in report.get("records", []):
        question = norm(record.get("question", ""))
        labels = source_labels(record)
        label_text = " | ".join(labels)
        tables = [norm(item) for item in record.get("actual_tables", [])]
        contexts: list[str] = []
        for read in record.get("terminal_reads", []):
            table_ref = str(read.get("source_table", ""))
            if "|" not in table_ref:
                continue
            document, line_text = table_ref.rsplit("|", 1)
            path = reports.get(document)
            if path is not None:
                contexts.append(nearby_context(path, int(line_text)))
        reasons: list[str] = []
        score = 0

        for expected, opposite in CONTRASTS:
            if expected in question and opposite in label_text and expected not in label_text:
                score += 3
                reasons.append(f"question asks '{expected}' but terminal label only signals '{opposite}'")

        scope_reasons = context_conflicts(record.get("question", ""), contexts)
        score += 5 * len(scope_reasons)
        reasons.extend(scope_reasons)

        structural_reasons = structural_context_conflicts(
            record.get("question", ""),
            labels,
            list(contexts),
            list(record.get("actual_tables", [])),
        )
        score += 5 * len(structural_reasons)
        reasons.extend(structural_reasons)

        if "công ty mẹ" in question and tables and not all("_separate" in item for item in tables):
            score += 4
            reasons.append("company-parent question uses a table not marked separate")
        if "hợp nhất" in question and tables and not all("_consolidated" in item for item in tables):
            score += 4
            reasons.append("consolidated question uses a table not marked consolidated")

        if reasons:
            findings.append({
                "id": int(record["id"]),
                "score": score,
                "reasons": reasons,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "source_labels": labels,
                "source_contexts": contexts,
                "actual_tables": record.get("actual_tables", []),
            })

    findings.sort(key=lambda item: (-item["score"], item["id"]))
    payload = {
        "submission": str(submission),
        "legacy_report": str(report_path.resolve()),
        "records_checked": len(report.get("records", [])),
        "finding_count": len(findings),
        "policy": "Lexical triage only; inspect the full formula and BTC cells before any repair.",
        "findings": findings,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "findings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
