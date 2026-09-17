"""Build v192 by ranking the data-selected result tables first.

The candidate keeps every relevant-table set byte-for-byte equivalent as a set;
only list order changes.  For panel programs, the query is replayed against its
submitted CSV and the selected ticker/year is read from the resulting Pandas
objects.  Tables for that data-derived winner are moved ahead of selector-only
tables so BTC MRR@5 receives the most direct answer source first.

No question id, answer value, leaderboard delta or hidden label controls the
ordering.  Rows without an unambiguous selected ticker/year remain untouched.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import numbers
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v191_eps_selection"
OUTPUT = ROOT / "sub_top123_candidate_v192_data_derived_table_order"

_WHITELIST = (
    "abs round len min max sum sorted float int str bool list dict set "
    "range enumerate zip all any isinstance"
).split()
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _WHITELIST}
_TABLE_RE = re.compile(r"^([^_]+)_financial_statements_(\d{4})_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_namespace(candidate: Path, row: dict) -> dict:
    dfs = {
        evidence["variable"]: pd.read_csv(
            candidate / evidence["csv_path"],
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
        for evidence in row.get("evidence", [])
    }
    namespace = {"pd": pd, "dfs": dfs, "__builtins__": _SAFE_BUILTINS}
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))
    exec(compile(row["pandas_query"], f"<q{row['id']}>", "exec"), namespace)  # noqa: S102
    return namespace


def _valid_year(value: object, valid_years: set[int]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or pd.isna(value):
        return None
    year = int(value)
    return year if year in valid_years else None


def _selected_groups(namespace: dict, audit: dict) -> tuple[set[str], set[int], str]:
    valid_tickers = {str(value).upper() for value in audit.get("tickers", [])}
    valid_years = {int(value) for value in audit.get("years", [])}
    series_groups: list[tuple[str | None, int | None, str]] = []

    for name in ("selected", "row", "selected_row", "winner", "best", "high", "low"):
        value = namespace.get(name)
        if not isinstance(value, pd.Series):
            continue
        raw_ticker = value.get("ticker")
        ticker = (
            str(raw_ticker).upper()
            if isinstance(raw_ticker, str) and str(raw_ticker).upper() in valid_tickers
            else None
        )
        year = _valid_year(value.get("year"), valid_years)
        if ticker is not None or year is not None:
            series_groups.append((ticker, year, name))

    if series_groups:
        tickers = {ticker for ticker, _year, _name in series_groups if ticker is not None}
        years = {year for _ticker, year, _name in series_groups if year is not None}
        return tickers, years, "+".join(name for _ticker, _year, name in series_groups)

    tickers: set[str] = set()
    years: set[int] = set()
    sources: list[str] = []
    for name, value in namespace.items():
        if name.startswith("_") or name in {"pd", "dfs", "df", "result"}:
            continue
        ticker_name = name == "ticker" or "ticker" in name
        year_name = (
            name in {"year", "next_year", "start", "end"}
            or name.endswith("_year")
        )
        if ticker_name and isinstance(value, str) and value.upper() in valid_tickers:
            tickers.add(value.upper())
            sources.append(name)
        if year_name:
            year = _valid_year(value, valid_years)
            if year is not None:
                years.add(year)
                sources.append(name)

    # A single entity/year in the audited panel is intrinsically unambiguous.
    if not tickers and len(valid_tickers) == 1:
        tickers = set(valid_tickers)
        sources.append("single_panel_ticker")
    if not years and len(valid_years) == 1:
        years = set(valid_years)
        sources.append("single_panel_year")
    return tickers, years, "+".join(dict.fromkeys(sources))


def _table_group(table_ref: str) -> tuple[str | None, int | None]:
    report = str(table_ref).split("|", 1)[0]
    match = _TABLE_RE.match(report)
    if not match:
        return None, None
    return match.group(1).upper(), int(match.group(2))


def _priority(table_ref: str, tickers: set[str], years: set[int]) -> int:
    ticker, year = _table_group(table_ref)
    ticker_match = not tickers or ticker in tickers
    year_match = not years or year in years
    if ticker_match and year_match:
        return 0
    if (tickers and ticker in tickers) or (years and year in years):
        return 1
    return 2


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    panel_audit = json.loads((OUTPUT / "panel_source_audit.json").read_text(encoding="utf-8"))
    changes: list[dict] = []
    unresolved: list[dict] = []

    for audit in panel_audit:
        qid = int(audit["id"])
        row = by_id[qid]
        original = list(row.get("relevant_tables") or [])
        if len(original) < 2:
            continue
        namespace = _load_namespace(OUTPUT, row)
        tickers, years, extraction = _selected_groups(namespace, audit)
        if not tickers and not years:
            unresolved.append({"id": qid, "reason": "no_unambiguous_data_selected_group"})
            continue

        ranked = sorted(
            enumerate(original),
            key=lambda pair: (_priority(pair[1], tickers, years), pair[0]),
        )
        reordered = [table_ref for _index, table_ref in ranked]
        if reordered == original:
            continue
        if set(reordered) != set(original) or len(reordered) != len(original):
            raise AssertionError(f"q{qid}: table set changed while reranking")

        before = min(
            index + 1
            for index, table_ref in enumerate(original)
            if _priority(table_ref, tickers, years) == 0
        )
        after = min(
            index + 1
            for index, table_ref in enumerate(reordered)
            if _priority(table_ref, tickers, years) == 0
        )
        row["relevant_tables"] = reordered
        changes.append(
            {
                "id": qid,
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "extraction": extraction,
                "table_count": len(original),
                "first_selected_rank_before": before,
                "first_selected_rank_after": after,
                "set_preserved": True,
            }
        )

    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    changed_ids = {change["id"] for change in changes}
    for qid, row in by_id.items():
        old = source_by_id[qid]
        if qid not in changed_ids and row != old:
            raise AssertionError(f"q{qid}: unexpected row change")
        if qid in changed_ids:
            old_without_order = dict(old)
            new_without_order = dict(row)
            old_tables = old_without_order.pop("relevant_tables", [])
            new_tables = new_without_order.pop("relevant_tables", [])
            if old_without_order != new_without_order:
                raise AssertionError(f"q{qid}: field other than relevant_tables changed")
            if set(old_tables) != set(new_tables) or len(old_tables) != len(new_tables):
                raise AssertionError(f"q{qid}: relevant_tables membership changed")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "data-derived selected result table ordering for BTC Tables MRR@5",
        "source_submission_sha256": _sha256(source_submission),
        "candidate_submission_sha256": _sha256(output_submission),
        "changed_question_count": len(changes),
        "changed_question_ids": sorted(changed_ids),
        "unresolved_panel_count": len(unresolved),
        "changes": changes,
        "unresolved": unresolved,
        "invariants": {
            "answers_unchanged": True,
            "queries_unchanged": True,
            "evidence_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_table_membership_unchanged": True,
            "only_relevant_table_order_changes": True,
            "v190_scored_champion_untouched": True,
            "v191_answer_candidate_untouched": True,
        },
        "claim_limit": (
            "This is a source-grounded MRR ordering hypothesis. BTC retrieval score "
            "is unknown until this exact artifact is evaluated."
        ),
    }
    (OUTPUT / "data_derived_table_order_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    os.chdir(ROOT)
    build()
