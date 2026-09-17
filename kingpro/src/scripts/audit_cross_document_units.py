"""Find cross-document computations whose report-level units are not normalized.

This is a triage audit, not an answer mutator.  It profiles each financial
report from explicit unit labels in its extracted tables, then compares those
profiles with per-source ``scale`` values in ``source_audit.json``.  Findings
still require inspection of the exact source table and surrounding note.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
UNIT_PATTERNS = {
    3: re.compile(r"\b(?:ngan|nghin)\s*(?:vnd|dong)\b", re.I),
    6: re.compile(r"\btrieu\s*(?:vnd|dong)\b", re.I),
    9: re.compile(r"\bty\s*(?:vnd|dong)\b", re.I),
}
BASE_UNIT_PATTERN = re.compile(r"\b(?:vnd|dong)\b", re.I)


def plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def build_profiles(catalog_path: Path, documents: set[str]) -> dict[str, dict]:
    """Read the sequential table catalog once instead of opening 146k CSVs."""

    records: dict[str, list[dict]] = {document: [] for document in documents}
    with catalog_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            document = record.get("report_id")
            if document in records:
                records[document].append(record)
    profiles = {}
    for document in sorted(documents):
        counts: Counter[int] = Counter()
        examples: dict[int, list[str]] = {3: [], 6: [], 9: []}
        for record in records[document]:
            text = plain(str(record.get("search_text", "")))
            residual = text
            for exponent, pattern in UNIT_PATTERNS.items():
                hits = len(pattern.findall(text))
                if hits:
                    counts[exponent] += hits
                    if len(examples[exponent]) < 3:
                        examples[exponent].append(str(record.get("csv_path", "")))
                residual = pattern.sub(" ", residual)
            base_hits = len(BASE_UNIT_PATTERN.findall(residual))
            if base_hits:
                counts[0] += base_hits
                if 0 not in examples:
                    examples[0] = []
                if len(examples[0]) < 3:
                    examples[0].append(str(record.get("csv_path", "")))
        exponent = None
        if counts:
            best, best_count = counts.most_common(1)[0]
            runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
            if best_count >= 2 and best_count >= runner_up * 2:
                exponent = best
        profiles[document] = {
            "document": document,
            "unit_exponent": exponent,
            "counts": {str(key): value for key, value in sorted(counts.items())},
            "examples": {str(key): value for key, value in examples.items() if value},
        }
    return profiles


def company(document: str) -> str:
    return document.split("_financial_statements_", 1)[0]


def numeric_magnitude(raw: object) -> float | None:
    text = str(raw).strip()
    if not text or text == "-" or "%" in text:
        return None
    if " " in text:
        text = text.split()[0]
    text = text.replace("(", "").replace(")", "").replace("$", "")
    if not re.fullmatch(r"[-+]?\d[\d.,]*", text):
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    elif "." in text:
        groups = text.split(".")
        if len(groups) > 2 or len(groups[-1]) == 3:
            text = "".join(groups)
    try:
        return abs(float(text))
    except ValueError:
        return None


def manifest_sources(candidate: Path, question_id: int) -> list[dict]:
    """Load source operands used by panel programs from their compact manifest."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "build" / "catalog.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-profile-hits", type=int, default=2)
    args = parser.parse_args()

    candidate = args.submission_dir.resolve()
    audit = []
    for audit_name in ("source_audit.json", "panel_source_audit.json"):
        audit_path = candidate / audit_name
        if not audit_path.is_file():
            continue
        for audit_row in json.loads(audit_path.read_text(encoding="utf-8")):
            copied = dict(audit_row)
            copied["_audit_source"] = audit_name
            if not copied.get("sources") and audit_name == "panel_source_audit.json":
                copied["sources"] = manifest_sources(candidate, int(copied["id"]))
            audit.append(copied)
    submissions = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    questions = {int(row["id"]): row.get("question", "") for row in submissions}

    documents = sorted(
        {
            source["table_ref"].split("|", 1)[0]
            for row in audit
            for source in row.get("sources", [])
            if "|" in source.get("table_ref", "")
        }
    )
    profiles = build_profiles(args.catalog, set(documents))
    findings = []
    magnitude_outliers = []
    absolute_scale_mismatches = []
    checked = 0

    for row in audit:
        sources = row.get("sources", [])
        # A mixed-unit comparison is the highest-risk case, but a single
        # report can be wrong too.  Record every confident report/profile
        # disagreement so reviewers do not have to rediscover it manually.
        # This remains triage-only because some legacy queries deliberately
        # apply a unit multiplier after reading the compact source row.
        for source in sources:
            ref = source.get("table_ref", "")
            if "|" not in ref:
                continue
            document = ref.split("|", 1)[0]
            profile = profiles[document]
            exponent = profile["unit_exponent"]
            if exponent is None:
                continue
            counts = profile["counts"]
            if int(counts.get(str(exponent), 0)) < args.min_profile_hits:
                continue
            actual_scale = float(source.get("scale", 1.0))
            expected_scale = 10.0 ** exponent
            if abs(actual_scale - expected_scale) <= max(1e-9, expected_scale * 1e-9):
                continue
            absolute_scale_mismatches.append(
                {
                    "id": int(row["id"]),
                    "question": questions.get(int(row["id"]), ""),
                    "answer": row.get("answer"),
                    "metric": source.get("metric", ""),
                    "table_ref": ref,
                    "raw": source.get("raw"),
                    "actual_scale": actual_scale,
                    "expected_profile_scale": expected_scale,
                    "unit_exponent": exponent,
                    "profile_counts": counts,
                    "profile_examples": profile["examples"],
                    "note": row.get("note", ""),
                    "audit_source": row.get("_audit_source", ""),
                }
            )
        companies = {
            company(source["table_ref"].split("|", 1)[0])
            for source in sources
            if "|" in source.get("table_ref", "")
        }
        if len(companies) < 2:
            continue
        by_metric: dict[str, list[dict]] = {}
        for source in sources:
            ref = source.get("table_ref", "")
            if "|" not in ref:
                continue
            value = numeric_magnitude(source.get("raw"))
            if value is None or value == 0:
                continue
            metric = str(source.get("metric", ""))
            by_metric.setdefault(metric, []).append(
                {
                    "table_ref": ref,
                    "company": company(ref.split("|", 1)[0]),
                    "raw": source.get("raw"),
                    "value_as_printed": value,
                    "scale": float(source.get("scale", 1.0)),
                    "typed_factor": float(source.get("typed_factor", 1.0)),
                }
            )
        for metric, operands in by_metric.items():
            if len({item["company"] for item in operands}) < 2:
                continue
            smallest = min(item["value_as_printed"] for item in operands)
            largest = max(item["value_as_printed"] for item in operands)
            ratio = largest / smallest
            if ratio >= 100.0:
                magnitude_outliers.append(
                    {
                        "id": int(row["id"]),
                        "question": questions.get(int(row["id"]), ""),
                        "answer": row.get("answer"),
                        "metric": metric,
                        "printed_magnitude_ratio": ratio,
                        "operands": operands,
                        "note": row.get("note", ""),
                        "audit_source": row.get("_audit_source", ""),
                    }
                )
        profiled = []
        for source in sources:
            ref = source.get("table_ref", "")
            if "|" not in ref:
                continue
            document = ref.split("|", 1)[0]
            profile = profiles[document]
            exponent = profile["unit_exponent"]
            if exponent is None:
                continue
            counts = profile["counts"]
            if int(counts.get(str(exponent), 0)) < args.min_profile_hits:
                continue
            scale = float(source.get("scale", 1.0))
            profiled.append(
                {
                    "table_ref": ref,
                    "document": document,
                    "company": company(document),
                    "raw": source.get("raw"),
                    "scale": scale,
                    "typed_factor": float(source.get("typed_factor", 1.0)),
                    "unit_exponent": exponent,
                    "profile_counts": counts,
                    "profile_examples": profile["examples"],
                }
            )
        exponents = {item["unit_exponent"] for item in profiled}
        if len(profiled) < 2 or len(exponents) < 2:
            continue
        checked += 1
        mismatched_pairs = []
        for left_index, left in enumerate(profiled):
            for right in profiled[left_index + 1 :]:
                if left["company"] == right["company"]:
                    continue
                expected = 10.0 ** (left["unit_exponent"] - right["unit_exponent"])
                actual = left["scale"] / right["scale"]
                ratio = actual / expected
                if ratio < 0.999 or ratio > 1.001:
                    mismatched_pairs.append(
                        {
                            "left": left["table_ref"],
                            "right": right["table_ref"],
                            "expected_scale_ratio": expected,
                            "actual_scale_ratio": actual,
                        }
                    )
        if mismatched_pairs:
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": questions.get(int(row["id"]), ""),
                    "answer": row.get("answer"),
                    "note": row.get("note", ""),
                    "sources": profiled,
                    "mismatched_pairs": mismatched_pairs,
                    "severity": "triage",
                    "audit_source": row.get("_audit_source", ""),
                }
            )

    report = {
        "submission": str(candidate),
        "audit_sources": sorted({row.get("_audit_source", "") for row in audit}),
        "documents_profiled": len(documents),
        "cross_company_mixed_unit_questions_checked": checked,
        "finding_count": len(findings),
        "finding_ids": [item["id"] for item in findings],
        "findings": findings,
        "magnitude_outlier_count": len(magnitude_outliers),
        "magnitude_outlier_ids": sorted({item["id"] for item in magnitude_outliers}),
        "magnitude_outliers": sorted(
            magnitude_outliers,
            key=lambda item: item["printed_magnitude_ratio"],
            reverse=True,
        ),
        "absolute_scale_mismatch_count": len(absolute_scale_mismatches),
        "absolute_scale_mismatch_ids": sorted(
            {item["id"] for item in absolute_scale_mismatches}
        ),
        "absolute_scale_mismatches": absolute_scale_mismatches,
        "claim_limit": (
            "Heuristic report-profile triage only. Exact source-table context and query "
            "semantics must be verified before changing an answer."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
