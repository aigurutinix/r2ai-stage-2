"""Audit whether every explicitly named company/year cohort reaches the program.

This is a review-queue generator, not a correctness oracle.  It targets a
failure mode that scalar missing-operand audits cannot see: one whole company
or year can be absent from ``pandas_query`` while every remaining
``_source_value`` call succeeds.

The question facets are the contract.  We compare them with four independent
layers: relevant documents, relevant tables, source-cell lineage, and literal
``_source_value(ticker, year, ...)`` calls in the program.  A finding must still
be verified against the original statement before changing a submission.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402


REPORT_RE = re.compile(
    r"^(?P<ticker>[A-Za-z0-9]+)_financial_statements_(?P<year>\d{4})_"
)


def _pair_from_report(value: Any) -> tuple[str, str] | None:
    match = REPORT_RE.match(str(value))
    if not match:
        return None
    return match.group("ticker").upper(), match.group("year")


def _program_calls(program: str) -> list[tuple[str, str, str]]:
    """Return literal ``_source_value`` calls; dynamic calls are ignored."""

    try:
        tree = ast.parse(program or "")
    except SyntaxError:
        return []
    calls: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_source_value":
            continue
        if len(node.args) < 3:
            continue
        values: list[Any] = []
        for arg in node.args[:3]:
            try:
                values.append(ast.literal_eval(arg))
            except (ValueError, TypeError):
                values.append(None)
        ticker, year, metric = values
        if ticker is None or year is None or metric is None:
            continue
        calls.append((str(ticker).upper(), str(year), str(metric)))
    return calls


def _manifest_pairs(path: Path) -> tuple[set[tuple[str, str]], Counter[tuple[str, str]]]:
    pairs: set[tuple[str, str]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    if not path.is_file():
        return pairs, counts
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ticker = str(row.get("ticker", "")).strip().upper()
            year = str(row.get("year", "")).strip()
            if ticker and year:
                pair = (ticker, year)
                pairs.add(pair)
                counts[pair] += 1
    return pairs, counts


def _as_lists(values: set[tuple[str, str]] | list[tuple[str, str]]) -> list[list[str]]:
    return [list(pair) for pair in sorted(values)]


def _year_asymmetry(
    tickers: list[str], program_pairs: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Find years used for most explicit tickers but absent for a minority."""

    if len(tickers) < 2 or not program_pairs:
        return []
    years = sorted({year for ticker, year in program_pairs if ticker in tickers})
    findings: list[dict[str, Any]] = []
    threshold = math.ceil(len(tickers) / 2)
    for year in years:
        present = sorted(ticker for ticker in tickers if (ticker, year) in program_pairs)
        missing = sorted(set(tickers) - set(present))
        if missing and len(present) >= threshold:
            findings.append(
                {
                    "year": year,
                    "present_tickers": present,
                    "missing_tickers": missing,
                    "coverage": len(present),
                    "cohort_size": len(tickers),
                }
            )
    return findings


def audit(submission_dir: Path) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    eligible = 0
    literal_programs = 0

    for row in rows:
        qid = int(row["id"])
        question = str(row.get("question", ""))
        facets = extract_all_facets(question)
        tickers = list(dict.fromkeys(str(x).upper() for x in facets.get("tickers", [])))
        years = list(dict.fromkeys(str(x) for x in facets.get("years", [])))
        if not tickers or not years:
            continue
        eligible += 1
        expected = {(ticker, year) for ticker in tickers for year in years}

        docs = {
            pair
            for value in row.get("relevant_docs", [])
            if (pair := _pair_from_report(value)) is not None
        }
        tables = {
            pair
            for value in row.get("relevant_tables", [])
            if (pair := _pair_from_report(str(value).split("|", 1)[0])) is not None
        }
        manifest, manifest_counts = _manifest_pairs(
            submission_dir / "data" / f"q{qid}_source_cells.csv"
        )
        calls = _program_calls(str(row.get("pandas_query", "")))
        program_pairs = {(ticker, year) for ticker, year, _ in calls}
        if calls:
            literal_programs += 1

        missing_program_tickers = (
            sorted(set(tickers) - {ticker for ticker, _ in program_pairs}) if calls else []
        )
        missing_program_pairs = sorted(expected - program_pairs) if calls else []
        missing_manifest_pairs = sorted(expected - manifest)
        missing_doc_pairs = sorted(expected - docs)
        missing_table_pairs = sorted(expected - tables)
        asymmetry = _year_asymmetry(tickers, program_pairs)

        # Exact manifest/program gaps and whole missing tickers are stronger
        # than report-year table gaps: a single report table can legally carry
        # comparative values from two fiscal years.
        strong = bool(
            missing_program_tickers
            or (set(missing_program_pairs) & set(missing_manifest_pairs))
        )
        review = bool(
            strong
            or asymmetry
            or missing_manifest_pairs
            or missing_doc_pairs
            or missing_table_pairs
        )
        if not review:
            continue

        findings.append(
            {
                "id": qid,
                "question": question,
                "explicit_tickers": tickers,
                "explicit_years": years,
                "missing_program_tickers": missing_program_tickers,
                "missing_program_pairs": _as_lists(missing_program_pairs),
                "missing_manifest_pairs": _as_lists(missing_manifest_pairs),
                "missing_document_pairs": _as_lists(missing_doc_pairs),
                "missing_table_report_pairs": _as_lists(missing_table_pairs),
                "program_year_asymmetry": asymmetry,
                "manifest_cell_counts": {
                    f"{ticker}:{year}": manifest_counts[(ticker, year)]
                    for ticker, year in sorted(expected & manifest)
                },
                "review_priority": "high" if strong else "medium",
            }
        )

    high = [item for item in findings if item["review_priority"] == "high"]
    return {
        "submission": str(submission_dir),
        "entries": len(rows),
        "eligible_explicit_cohort_questions": eligible,
        "literal_source_value_programs": literal_programs,
        "finding_count": len(findings),
        "high_priority_count": len(high),
        "high_priority_question_ids": sorted({item["id"] for item in high}),
        "findings": findings,
        "claim_limit": (
            "Every finding is a review signal. Comparative columns and legitimate "
            "missing disclosures can create asymmetry; mutate only after source verification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
