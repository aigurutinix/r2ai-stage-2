"""Rebuild the address book on the organisers' own corpus, not on re-parsed HTML.

`data/official_corpus/` turns out to hold exactly what their pipeline consumes:

  <TICKER>/<YEAR>/<DOC>/<DOC>_extracted.txt          page markers + [table_N](…) anchors
  <TICKER>/<YEAR>/<DOC>/<DOC>_extracted_tables/table_N.csv   the normalised table

That matters for three reasons the earlier pass could not satisfy. The table ids are
theirs, so a `relevant_tables` reference of `doc|table_N` means what the gold means.
The CSV is the one the scorer binds, so a cell address is the cell the scorer reads.
And the unit line sits on the anchor's own page — `Đơn vị: VND` two lines above
`[table_0](…)` — which is how their `table_unit_snippets` finds it.

Everything else follows their `parse_statement_table`: rows keyed by `Mã số`, the
first value cell of a row is the current period and the second is the prior one,
scale from the header or the page's unit line, value in đồng, and a table is a
statement only if its codes classify as cân đối, kết quả kinh doanh or lưu chuyển
tiền tệ.

Also stamps each report with whether its ticker is a credit institution, taken from
`Ngành cấp 2 = "Tổ chức tín dụng"` in `data/file_filter.csv` — 21 of the 100
companies. Those use a different chart of accounts, and mixing them into the
enterprise identities is what dragged `kqkd: 20 = 10 - 11` down.

Usage:  python scripts/fresh/build_index2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402

PAGE_RE = re.compile(r"===== PAGE (\d+) =====")
ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
UNIT_PREFIXES = ("Đơn vị tính", "Đơn vị tiền tệ", "Đơn vị", "ĐVT")
DOC_RE = re.compile(r"^(.+?)_financial_statements_((?:19|20)\d{2})_(.+)$")


def banks() -> set[str]:
    rows = list(csv.DictReader(
        (ROOT / "data" / "file_filter.csv").open(encoding="utf-8-sig")))
    if not rows:
        return set()
    ticker_col = next(c for c in rows[0] if "CK" in c)
    level2 = next(c for c in rows[0] if "cấp 2" in c)
    return {r[ticker_col] for r in rows if "tín dụng" in (r[level2] or "")}


def page_units(text: str) -> tuple[dict[int, int], dict[int, tuple[str, ...]]]:
    """Which page each table anchor sits on, and that page's unit lines."""

    marks = list(PAGE_RE.finditer(text))
    pages: list[tuple[int, str]] = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        pages.append((int(mark.group(1)), text[mark.end():end]))
    if not pages:
        pages = [(1, text)]

    table_page: dict[int, int] = {}
    snippets: dict[int, tuple[str, ...]] = {}
    for page_no, body in pages:
        lines = tuple(
            line.strip() for line in body.splitlines()
            if any(line.strip().casefold().startswith(p.casefold())
                   for p in UNIT_PREFIXES))
        for match in ANCHOR_RE.finditer(body):
            table_id = int(match.group(1))
            table_page[table_id] = page_no
            snippets[table_id] = lines
    return table_page, snippets


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="artifacts/fresh/statements2.jsonl")
    args = parser.parse_args()

    bank_tickers = banks()
    print(f"{len(bank_tickers)} ma la to chuc tin dung")

    docs = sorted(p for p in (ROOT / "data" / "official_corpus").glob("*/*/*")
                  if p.is_dir())
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs)} thu muc bao cao", flush=True)

    handle = (ROOT / args.out).open("w", encoding="utf-8")
    stats: Counter[str] = Counter()
    started = time.time()

    for index, doc_dir in enumerate(docs, start=1):
        doc_name = doc_dir.name
        match = DOC_RE.match(doc_name)
        if not match:
            stats["ten tai lieu khong khop mau"] += 1
            continue
        ticker, year, scope = match.group(1), match.group(2), match.group(3)
        text_path = doc_dir / f"{doc_name}_extracted.txt"
        table_dir = doc_dir / f"{doc_name}_extracted_tables"
        if not text_path.exists() or not table_dir.is_dir():
            stats["thieu text hoac thu muc bang"] += 1
            continue

        table_page, snippets = page_units(
            text_path.read_text(encoding="utf-8", errors="replace"))
        for csv_path in sorted(table_dir.glob("table_*.csv")):
            table_id = int(csv_path.stem.split("_")[-1])
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                rows = [row for row in csv.reader(file)]
            if not rows:
                continue
            stats["bang"] += 1
            table = ps.Table(doc_name=doc_name, table_id=table_id,
                             page_no=table_page.get(table_id, 0),
                             header=rows[0], rows=rows[1:],
                             unit_snippets=snippets.get(table_id, ()))
            statement = ps.parse_statement(table)
            if statement is None:
                continue
            stats[f"bao cao {statement.kind}"] += 1
            handle.write(json.dumps({
                "doc": doc_name,
                "ticker": ticker,
                "year": year,
                "scope": scope,
                "table_id": table_id,
                "table_ref": f"{doc_name}|table_{table_id}",
                "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "kind": statement.kind,
                "scale": statement.scale,
                "bank": ticker in bank_tickers,
                "current": {code: [cell.value, cell.label, cell.row_idx, cell.col_idx,
                                   cell.raw]
                            for code, cell in statement.current.items()},
                "prior": {code: [cell.value, cell.label, cell.row_idx, cell.col_idx,
                                 cell.raw]
                          for code, cell in statement.prior.items()},
            }, ensure_ascii=False) + "\n")
        if index % 200 == 0:
            print(f"  {index}/{len(docs)}  {time.time() - started:.0f}s", flush=True)

    handle.close()
    for name, count in stats.most_common():
        print(f"  {name}: {count}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
