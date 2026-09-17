"""Simulate conservative currency-unit corrections for compact manifests.

Some extracted statement tables omit the report's unit banner.  A scale audit
that looks only at the exact table therefore misses errors such as q782, where
the selected balance-sheet table has no banner but the surrounding report
repeatedly declares ``Ngàn VND``.  This read-only audit uses exact-table units
first, then a conservative document-level consensus (at least two tables and
one unique non-base unit), executes the submitted program, and reports only
answer-changing questions.

The default safe mode also requires the submitted program to divide a monetary
result by the multiplier named in the question.  That proves the program
expects every operand in VND.  Without this contract, ``scale=1`` can be
intentional: for example, a question asking for nghìn VND may read a nghìn-VND
table and return the raw number.  ``--include-unsafe`` is research-only and may
therefore produce false positives.  Every finding still needs source review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from grader_check import _SAFE_BUILTINS, _read_csv


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "build" / "tables"


def plain(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return (
        "".join(char for char in text if not unicodedata.combining(char))
        .replace("đ", "d")
        .replace("Đ", "D")
        .lower()
    )


def table_unit(path: Path, *, include_base: bool = True) -> float | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        cells: list[str] = []
        for index, row in enumerate(csv.reader(handle)):
            cells.extend(row)
            if index >= 9:
                break
    text = plain(" ".join(cells))
    patterns = (
        (1_000.0, r"(?:^|[^a-z])(?:nghin|ngan)\s*(?:vnd|dong)\b"),
        (1_000_000.0, r"(?:^|[^a-z])trieu\s*(?:vnd|dong)\b"),
        (1_000_000_000.0, r"(?:^|[^a-z])ty\s*(?:vnd|dong)\b"),
    )
    for scale, pattern in patterns:
        if re.search(pattern, text):
            return scale
    if include_base and re.search(r"\b(?:vnd|dong)\b", text):
        return 1.0
    return None


def source_path(table_ref: str, source_csv: str) -> Path | None:
    """Resolve copied source names and generated ``table_*_lineN`` names."""
    if "|" not in table_ref:
        return None
    document, line = table_ref.split("|", 1)
    folder = TABLES / document
    direct = folder / source_csv
    if direct.is_file():
        return direct
    matches = list(folder.glob(f"*_line{line}.csv")) if folder.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def output_multiplier(question: str) -> float | None:
    """Return the VND multiplier of an explicitly requested currency unit."""
    text = plain(question)
    patterns = (
        (1_000_000_000_000.0, r"\bnghin\s+ty\s+dong\b"),
        (100_000_000_000.0, r"\btram\s+ty\s+(?:vnd|dong)\b"),
        (1_000_000_000.0, r"\bty\s+(?:vnd|dong)\b"),
        (1_000_000.0, r"\btrieu\s+(?:vnd|dong)\b"),
        (1_000.0, r"\b(?:nghin|ngan)\s+(?:vnd|dong)\b"),
        (1.0, r"\b(?:vnd|dong)\b"),
    )
    for multiplier, pattern in patterns:
        if re.search(pattern, text):
            return multiplier
    return None


def query_normalizes_vnd(row: dict[str, Any]) -> bool:
    """Prove that a monetary answer is normalized from VND at the result."""
    multiplier = output_multiplier(str(row.get("question", "")))
    if multiplier is None or multiplier == 1.0:
        return False
    exponent = int(round(math.log10(multiplier)))
    aliases = {
        f"1e{exponent}",
        f"10**{exponent}",
        f"10 ** {exponent}",
        str(int(multiplier)),
        f"{int(multiplier):_}",
    }
    # Restrict the proof to an assignment to result.  This avoids treating a
    # mixed-unit max such as max(v0 / 1e9, v1 / 1e3) as a VND contract.
    for line in str(row.get("pandas_query", "")).splitlines():
        if not re.match(r"^\s*result\s*=", line):
            continue
        compact = re.sub(r"\s+", "", line.lower())
        if any(f"/{alias.replace(' ', '')}" in compact for alias in aliases):
            return True
    return False


@lru_cache(maxsize=1)
def unit_banner_index() -> dict[str, Counter[float]]:
    """Index all multiplier banners with one native filesystem traversal."""
    try:
        process = subprocess.run(
            ["rg", "-l", "-i", r"(?:vnd|vnđ|đồng|dong)", str(TABLES)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        paths = [Path(value) for value in process.stdout.splitlines() if value]
    except OSError:
        paths = list(TABLES.glob("*/*.csv"))
    index: dict[str, Counter[float]] = {}
    for path in paths:
        # Bare VND inside a note may label only one column, so document
        # inheritance uses multiplier banners only.
        scale = table_unit(path, include_base=False)
        if scale is not None:
            index.setdefault(path.parent.name, Counter())[scale] += 1
    return index


@lru_cache(maxsize=None)
def document_consensus(document: str) -> tuple[float | None, dict[str, int]]:
    votes = unit_banner_index().get(document, Counter())
    if len(votes) == 1 and sum(votes.values()) >= 2:
        return next(iter(votes)), {str(int(k)): v for k, v in votes.items()}
    return None, {str(int(k)): v for k, v in votes.items()}


def adjustments(
    candidate: Path,
    qid: int,
    *,
    allow_document_consensus: bool = False,
    standard_metrics_only: bool = True,
) -> list[dict[str, Any]]:
    path = candidate / "data" / f"q{qid}_source_cells.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    found = []
    for index, row in enumerate(rows):
        metric_key = str(row.get("metric_key", ""))
        if standard_metrics_only and not re.match(r"^(?:cdkt|kqkd|lctt):", metric_key):
            continue
        table_ref = str(row.get("source_table", ""))
        source_csv = str(row.get("source_csv", ""))
        if "|" not in table_ref or not source_csv:
            continue
        document = table_ref.split("|", 1)[0]
        resolved = source_path(table_ref, source_csv)
        # Bare ``VND`` can occur in a currency-composition row and is not a
        # reliable table banner.  Automatic corrections therefore require an
        # explicit non-base multiplier (nghìn/triệu/tỷ).
        exact = table_unit(resolved, include_base=False) if resolved is not None else None
        # Exact evidence is both stronger and dramatically cheaper.  Do not
        # scan every table in the document unless the cited table lacks a unit.
        if exact is not None:
            expected = exact
            votes: dict[str, int] = {}
        elif allow_document_consensus:
            expected, votes = document_consensus(document)
        else:
            continue
        if expected is None:
            continue
        try:
            current = float(row.get("scale", 1.0))
        except (TypeError, ValueError):
            continue
        if abs(current - expected) <= max(1e-9, expected * 1e-9):
            continue
        found.append(
            {
                "row_index": index,
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "metric_key": row.get("metric_key", ""),
                "raw": row.get("raw", ""),
                "table_ref": table_ref,
                "source_csv": source_csv,
                "old_scale": current,
                "new_scale": expected,
                "inference": "exact-table" if exact is not None else "document-consensus",
                "document_votes": votes,
            }
        )
    return found


def execute(candidate: Path, row: dict[str, Any], changes: list[dict[str, Any]]) -> Any:
    dfs = {
        str(item["variable"]): _read_csv(
            (candidate / str(item["csv_path"])).resolve(), typed=True
        )
        for item in row.get("evidence", [])
    }
    manifest_name = f"q{int(row['id'])}_source_cells.csv"
    for item in row.get("evidence", []):
        if Path(str(item.get("csv_path", ""))).name != manifest_name:
            continue
        frame = dfs[str(item["variable"])]
        if "scale" not in frame.columns:
            continue
        for change in changes:
            frame.loc[int(change["row_index"]), "scale"] = float(change["new_scale"])
    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": dfs,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))
    exec(compile(str(row["pandas_query"]), f"<q{row['id']}_doc_unit>", "exec"), namespace)  # noqa: S102
    value = namespace["result"]
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--include-unsafe",
        action="store_true",
        help="also simulate rows whose query does not prove VND normalization",
    )
    parser.add_argument(
        "--document-consensus",
        action="store_true",
        help="infer units for cited tables without banners (slower; source review required)",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="research-only: include semantic note metrics as well as statement codes",
    )
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    submissions = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))

    changed = []
    simulated = 0
    for row in submissions:
        qid = int(row["id"])
        if not args.include_unsafe and not query_normalizes_vnd(row):
            continue
        changes = adjustments(
            candidate,
            qid,
            allow_document_consensus=args.document_consensus,
            standard_metrics_only=not args.all_metrics,
        )
        if not changes:
            continue
        simulated += 1
        try:
            new_answer = execute(candidate, row, changes)
            same = abs(float(new_answer) - float(row.get("answer"))) <= 1e-9
            if not same:
                changed.append(
                    {
                        "id": qid,
                        "question": row.get("question", ""),
                        "old_answer": row.get("answer"),
                        "new_answer": new_answer,
                        "adjustments": changes,
                    }
                )
        except Exception as exc:
            changed.append(
                {
                    "id": qid,
                    "question": row.get("question", ""),
                    "old_answer": row.get("answer"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "adjustments": changes,
                }
            )

    report = {
        "candidate": candidate.name,
        "questions": len(submissions),
        "documents_scanned": document_consensus.cache_info().currsize,
        "simulated_questions": simulated,
        "safe_vnd_contract_only": not args.include_unsafe,
        "document_consensus_enabled": args.document_consensus,
        "standard_metrics_only": not args.all_metrics,
        "changed_count": len(changed),
        "changed_ids": [item["id"] for item in changed],
        "findings": changed,
        "claim_limit": (
            "Conservative document-unit simulation; every answer change requires "
            "exact source review before mutation."
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(args.out.resolve())
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
