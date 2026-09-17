"""Audit question company names/tickers against submission source documents.

The older structural audit extracts uppercase tokens and can confuse a company
name embedded in another company's official name (for example FPT versus FTS
in ``CTCP Chứng khoán FPT``).  This checker uses the BTC stock-code catalog as
the authority: exact normalized official names and explicit ticker tokens in
the question must each have a matching source-document prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value).casefold())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    value = " ".join(re.findall(r"[a-z0-9]+", value))
    value = value.replace("cong ty co phan", "ctcp")
    value = value.replace("ngan hang thuong mai co phan", "ngan hang tmcp")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def load_catalog() -> list[tuple[str, str, str]]:
    path = ROOT / "data" / "code_stock.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    catalog = []
    for row in rows:
        ticker = str(row["Mã CK"]).strip().upper()
        company = str(row["Tên công ty"]).strip()
        catalog.append((ticker, company, normalize(company)))
    return catalog


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    submission_path = args.submission_dir.resolve() / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    catalog = load_catalog()
    tickers = {ticker for ticker, _, _ in catalog}

    findings = []
    unmatched = []
    matched_questions = 0
    for item in submission:
        question = str(item.get("question", ""))
        normalized_question = normalize(question)
        expected: dict[str, list[str]] = {}
        matched_companies: list[tuple[str, str]] = []

        for ticker, company, normalized_company in catalog:
            if normalized_company and normalized_company in normalized_question:
                matched_companies.append((ticker, normalized_company))
                # An investee named after "investment in" is the object being
                # measured, not the reporting entity that owns the source.
                if any(
                    marker in normalized_question
                    for marker in (
                        f"dau tu vao {normalized_company}",
                        f"voi {normalized_company}",
                    )
                ):
                    continue
                expected.setdefault(ticker, []).append(f"official-name:{company}")

        # Only original uppercase tokens count as explicit ticker mentions.
        # This avoids treating ordinary lower-case Vietnamese words as codes.
        for token in re.findall(r"(?<![A-Z0-9])[A-Z][A-Z0-9]{1,4}(?![A-Z0-9])", question):
            if token in tickers:
                # FPT is part of the official names of FTS and FOX.  Do not
                # reinterpret that embedded brand token as a separate issuer;
                # a parenthesized code remains an explicit mention.
                embedded_in_other_name = any(
                    owner_ticker != token and token.casefold() in company_name.split()
                    for owner_ticker, company_name in matched_companies
                )
                explicitly_parenthesized = re.search(
                    rf"\(\s*{re.escape(token)}\s*\)", question
                ) is not None
                if embedded_in_other_name and not explicitly_parenthesized:
                    continue
                expected.setdefault(token, []).append(f"ticker:{token}")

        if not expected:
            unmatched.append(
                {
                    "id": int(item["id"]),
                    "relevant_docs": [str(value) for value in item.get("relevant_docs", [])],
                    "question": question,
                }
            )
            continue
        matched_questions += 1

        docs = [str(value) for value in item.get("relevant_docs", [])]
        source_tickers = {
            match.group(1).upper()
            for doc in docs
            if (match := re.match(r"^([A-Za-z0-9]+)_financial_statements", doc))
        }
        missing = sorted(set(expected) - source_tickers)
        if missing:
            findings.append(
                {
                    "id": int(item["id"]),
                    "missing_tickers": missing,
                    "matched_by": {ticker: expected[ticker] for ticker in missing},
                    "source_tickers": sorted(source_tickers),
                    "relevant_docs": docs,
                    "question": question,
                    "answer": item.get("answer"),
                }
            )

    report = {
        "submission": args.submission_dir.name,
        "entries": len(submission),
        "questions_with_catalog_entity": matched_questions,
        "questions_without_catalog_entity": len(unmatched),
        "finding_count": len(findings),
        "findings": findings[: max(args.limit, 0)],
        "unmatched": unmatched[: max(args.limit, 0)],
        "truncated": len(findings) > max(args.limit, 0),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
