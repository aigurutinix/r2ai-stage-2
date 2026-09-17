"""Find high-signal structural risks in legacy submission programs.

This is an audit helper only.  It never edits a submission and deliberately
prefers false negatives over automatically "fixing" an uncertain answer.
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


YEAR_RE = re.compile(r"\b20(?:0\d|1\d|2\d|3\d)\b")
YEAR_RANGE_RE = re.compile(
    r"(?:giai\s+đoạn\s+)?từ\s+(?:năm\s+)?(20\d{2})\s+"
    r"(?:đến|tới)\s+(?:năm\s+)?(20\d{2})",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")
DOCUMENT_YEAR_RE = re.compile(r"financial_statements_(20\d{2})")
ENDING_DATE_RE = re.compile(
    r"(?:cuối\s+năm|31\s*/\s*12|31\s+tháng\s+12|"
    r"ngày\s+31\s+tháng\s+12)",
    re.IGNORECASE,
)
BEGINNING_DATE_RE = re.compile(
    r"(?:đầu\s+năm|0?1\s*/\s*0?1|ngày\s+0?1\s+tháng\s+0?1)",
    re.IGNORECASE,
)


def normalize_entity(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def load_company_catalog(path: Path) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return set(), {}
    likely_columns = [
        key for key in rows[0] if key.lower() in {"code", "ticker", "stock_code", "symbol"}
    ]
    ticker_column = likely_columns[0] if likely_columns else next(iter(rows[0]))
    other_columns = [column for column in rows[0] if column != ticker_column]
    name_column = other_columns[0] if other_columns else None
    tickers: set[str] = set()
    aliases: dict[str, tuple[str, ...]] = {}
    legal_prefix = re.compile(
        r"^(?:cong ty co phan|ctcp|ngan hang thuong mai co phan|ngan hang tmcp|"
        r"tong cong ty co phan|tong cong ty|tap doan)\s+"
    )
    for row in rows:
        ticker = str(row.get(ticker_column, "")).strip().upper()
        if not ticker:
            continue
        tickers.add(ticker)
        if name_column is None:
            continue
        full_name = normalize_entity(str(row.get(name_column, "")))
        if len(full_name) < 12:
            continue
        candidates = {full_name}
        core_name = legal_prefix.sub("", full_name)
        if len(core_name) >= 12:
            candidates.add(core_name)
        aliases[ticker] = tuple(sorted(candidates, key=len, reverse=True))
    return tickers, aliases


def extract_years(question: str) -> list[str]:
    years = set(YEAR_RE.findall(question))
    for start_text, end_text in YEAR_RANGE_RE.findall(question):
        start, end = int(start_text), int(end_text)
        if start <= end and end - start <= 15:
            years.update(str(year) for year in range(start, end + 1))
    return sorted(years)


def _selected_company_alias_matches(
    question: str, aliases: dict[str, tuple[str, ...]]
) -> list[tuple[int, int, str, str]]:
    """Return longest, non-overlapping catalog-name matches."""

    normalized = normalize_entity(question)
    matches: list[tuple[int, int, str, str]] = []
    for ticker, ticker_aliases in aliases.items():
        for alias in ticker_aliases:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
            for match in pattern.finditer(normalized):
                matches.append((match.start(), match.end(), ticker, alias))

    # Longest-first makes the owning company win over any issuer alias nested
    # inside its legal name.  Stable positional tie-breaks keep output fully
    # deterministic across Python versions and catalog orderings.
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2], item[3]))
    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, int, str, str]] = []
    for start, end, ticker, _alias in matches:
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, end, ticker, _alias))
    return selected


def company_names_in_question(question: str, aliases: dict[str, tuple[str, ...]]) -> set[str]:
    """Return catalog entities named in *question* without nested aliases.

    Several Vietnamese company names contain another issuer's complete short
    name.  For example, ``Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP``
    (POW) contains ``Khí Việt Nam - CTCP`` (GAS), while FTS/FOX contain the
    one-token FPT core name.  A plain substring scan therefore invents extra
    entities and hides the genuinely useful structural warnings in noise.

    Prefer the longest catalog alias at each character span and do not accept
    a shorter overlapping alias.  Separate occurrences are retained, so a
    real two-company comparison still produces two tickers.
    """

    return {
        ticker
        for _start, _end, ticker, _alias in _selected_company_alias_matches(
            question, aliases
        )
    }


def standalone_ticker_tokens(
    question: str, tickers: set[str], aliases: dict[str, tuple[str, ...]]
) -> set[str]:
    """Extract explicit ticker tokens not merely embedded in another name.

    ``FPT`` in ``CTCP Chứng khoán FPT`` describes FTS, not a second requested
    issuer.  Work on normalized spans so a separately written ``FPT`` outside
    the FTS/FOX legal name is still retained.
    """

    raw_tokens = set(TOKEN_RE.findall(question)) & tickers
    if not raw_tokens:
        return set()
    normalized = normalize_entity(question)
    company_matches = _selected_company_alias_matches(question, aliases)
    selected: set[str] = set()
    for token in raw_tokens:
        occurrences = list(re.finditer(rf"\b{re.escape(token.casefold())}\b", normalized))
        if not occurrences:
            selected.add(token)
            continue
        if any(
            not any(
                start <= occurrence.start()
                and occurrence.end() <= end
                and owner != token
                for start, end, owner, _alias in company_matches
            )
            for occurrence in occurrences
        ):
            selected.add(token)
    return selected


def audited_ids(submission_dir: Path) -> set[int]:
    ids: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids.update(int(row["id"]) for row in payload if "id" in row)
    return ids


def extraction_rhs_counts(query: str) -> Counter[str]:
    """Count repeated scalar-cell extractions, ignoring boilerplate aliases."""
    result: Counter[str] = Counter()
    for line in query.splitlines():
        match = ASSIGN_RE.match(line)
        if not match:
            continue
        rhs = re.sub(r"\s+", " ", match.group(2).strip())
        if ".values[0]" in rhs or re.search(r"\.iloc\[[^\]]+\]", rhs):
            result[rhs] += 1
    return result


def ending_year_source_mismatch(question: str, docs: list[str]) -> str | None:
    """Return the requested ending year when no source document has that year.

    Prior-year reports are legitimate for opening balances such as 01/01/2022,
    so this deliberately covers only explicit ending-date language.  It catches
    the inverse error where a 2024 ending column is relabeled as 31/12/2025.
    """

    years = extract_years(question)
    if (
        len(years) != 1
        or not ENDING_DATE_RE.search(question)
        or BEGINNING_DATE_RE.search(question)
    ):
        return None
    doc_years = {
        match.group(1)
        for doc in docs
        if (match := DOCUMENT_YEAR_RE.search(doc)) is not None
    }
    requested_year = years[0]
    if doc_years and requested_year not in doc_years:
        return requested_year
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    submission_path = args.submission_dir / "submission.json"
    rows = json.loads(submission_path.read_text(encoding="utf-8"))
    ticker_path = args.submission_dir / "code_stock.csv"
    if not ticker_path.exists():
        ticker_path = Path("data/code_stock.csv")
    tickers, company_aliases = load_company_catalog(ticker_path)
    known_audited = audited_ids(args.submission_dir)
    findings: list[dict] = []

    for row in rows:
        qid = int(row["id"])
        if qid in known_audited and not args.include_audited:
            continue
        question = str(row.get("question", ""))
        query = str(row.get("pandas_query", ""))
        docs = [str(item) for item in row.get("relevant_docs", [])]
        normalized_question = normalize_entity(question)
        years = extract_years(question)
        ticker_tokens = standalone_ticker_tokens(question, tickers, company_aliases)
        company_name_tickers = company_names_in_question(question, company_aliases)
        named_tickers = sorted(ticker_tokens | company_name_tickers)
        doc_tickers = {doc.split("_", 1)[0].upper() for doc in docs}
        missing_tickers = sorted(set(named_tickers) - doc_tickers)
        repeated = extraction_rhs_counts(query)
        repeated = {rhs: count for rhs, count in repeated.items() if count >= 2}
        mismatched_ending_year = ending_year_source_mismatch(question, docs)

        reasons: list[str] = []
        score = 0
        if len(years) >= 3 and repeated:
            reasons.append("3+ requested years with repeated scalar extraction")
            score += 5
        if len(named_tickers) >= 2 and missing_tickers:
            reasons.append("named ticker/company absent from relevant_docs")
            score += min(5, 2 + len(missing_tickers))
        if len(years) >= 3 and len(docs) <= 1:
            reasons.append("3+ requested years but <=1 source document")
            score += 2
        if len(named_tickers) >= 3 and len(doc_tickers) <= 1:
            reasons.append("3+ requested tickers but <=1 source ticker")
            score += 4
        if mismatched_ending_year is not None:
            reasons.append(
                f"ending date {mismatched_ending_year} absent from relevant_docs years"
            )
            score += 6
        doc_scopes = {
            "separate" if "_separate" in doc
            else "consolidated" if "_consolidated" in doc
            else "unknown"
            for doc in docs
        }
        requests_parent_scope = any(token in normalized_question for token in (
            "cong ty me", "bctc rieng", "bao cao tai chinh rieng",
        ))
        requests_consolidated_scope = any(token in normalized_question for token in (
            "bctc hop nhat", "bao cao tai chinh hop nhat",
        ))
        # Unsuffixed BTC documents are often standalone company statements
        # (HND and securities companies are common examples).  "unknown" is
        # not evidence of a scope mismatch; only flag a positive conflict.
        if (
            requests_parent_scope
            and "separate" not in doc_scopes
            and "consolidated" in doc_scopes
        ):
            reasons.append("parent/separate question without a separate source document")
            score += 5
        if (
            requests_consolidated_scope
            and "consolidated" not in doc_scopes
            and "separate" in doc_scopes
        ):
            reasons.append("consolidated question without a consolidated source document")
            score += 5
        if not reasons:
            continue

        findings.append(
            {
                "id": qid,
                "score": score,
                "years": years,
                "tickers": named_tickers,
                "missing_tickers": missing_tickers,
                "relevant_docs": docs,
                "repeated_extractions": repeated,
                "reasons": reasons,
                "question": question,
            }
        )

    findings.sort(key=lambda item: (-item["score"], item["id"]))
    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known_audited),
        "finding_count": len(findings),
        "findings": findings,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
