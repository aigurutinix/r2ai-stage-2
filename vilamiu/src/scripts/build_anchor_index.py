"""Index the lines immediately introducing each table, not the whole page.

`build_context_index.py` gives every table on a page the same prose, so it cannot
tell two tables on one page apart — and at the shallow cutoffs that decide F2 it
measured no better than the label index.

The organisers have a second encoder for exactly this. `anchor_snippet_text`
takes six lines before the table and two after; `table_context_before` takes the
span between the previous table and this one. In a financial statement that span
is the note heading that introduces the table — "5.2. Phải thu ngắn hạn của khách
hàng" — which is both per-table and the wording a question is most likely to use.

Usage:  PYTHONPATH=src python scripts/build_anchor_index.py
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from build_context_index import (
    MAX_LABEL_ROWS,
    PAGE_RE,
    detect_report_scope,
    label_text,
)

ROOT = Path(__file__).resolve().parents[1]

TABLE_BLOCK_RE = re.compile(r"<table.*?</table>", re.S)
# The heading that introduces a table sits just above it; anything earlier
# belongs to the previous table on the page.
DROP_PREFIX = __import__("os").environ.get("DROP_PREFIX") == "1"

MAX_BEFORE_CHARS = 700
MAX_AFTER_CHARS = 200


def anchor_context(text_path: Path) -> dict[int, str]:
    """Prose between the previous table and this one, keyed by table ordinal."""

    text = text_path.read_text(encoding="utf-8")
    marks = list(PAGE_RE.finditer(text))
    context: dict[int, str] = {}
    ordinal = 0
    for i, mark in enumerate(marks):
        start = mark.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        content = text[start:end]
        blocks = list(TABLE_BLOCK_RE.finditer(content))
        for position, block in enumerate(blocks):
            lower = blocks[position - 1].end() if position else 0
            before = " ".join(content[lower:block.start()].split())[-MAX_BEFORE_CHARS:]
            upper = blocks[position + 1].start() if position + 1 < len(blocks) else len(content)
            after = " ".join(content[block.end():upper].split())[:MAX_AFTER_CHARS]
            context[ordinal] = f"{before} {after}".strip()
            ordinal += 1
    return context


def main() -> None:
    frame = pd.read_parquet(ROOT / "artifacts" / "tables.parquet")
    stock = pd.read_csv(ROOT / "data" / "code_stock.csv")
    company = {
        str(row[0]).strip().upper(): str(row[1]).strip()
        for row in stock.itertuples(index=False)
    }

    eligible = frame[frame.eligible]
    # A variant must never land on the path the working index occupies. Running
    # this with DROP_PREFIX=1 once truncated the 130 MB index the whole retrieval
    # path reads, and there is no git history to restore it from.
    out_path = ROOT / "artifacts" / (
        "anchor_index_noprefix.jsonl" if DROP_PREFIX else "anchor_index.jsonl")
    started = time.time()
    written = docs_done = empty = 0
    lengths: list[int] = []

    with out_path.open("w", encoding="utf-8") as handle:
        for doc_name, group in eligible.groupby("doc_name", sort=False):
            first = group.iloc[0]
            ticker, year = str(first.ticker), str(first.year)
            doc_dir = ROOT / "data" / "financial_statements" / ticker / year / doc_name
            text_path = next(iter(doc_dir.glob("*_extracted.txt")), None)
            if text_path is None:
                continue
            context = anchor_context(text_path)
            docs_done += 1

            name = company.get(ticker.upper(), "")
            head = f"Công ty: {name} (mã {ticker})" if name else f"Mã: {ticker}"
            scope = detect_report_scope(doc_name)
            prefix = f"{head}, năm {year}" + (f". Phạm vi báo cáo: {scope}" if scope else "") + "."

            for row in group.itertuples(index=False):
                prose = context.get(int(row.table_id), "")
                if not prose:
                    empty += 1
                lengths.append(len(prose))
                grid = json.loads(row.rows_json)
                # The company name, ticker, year and report scope are identical for
                # every table in a document. Retrieval already filters on those by
                # metadata, so inside a document they carry zero information — and
                # DOCS_F2 is 0.97 while TABLES_F2 is 0.55, meaning picking the
                # document is solved and picking the table inside it is not. Set
                # DROP_PREFIX=1 to test whether the prefix is diluting the only
                # text that actually discriminates: the note heading.
                head = "" if DROP_PREFIX else f"{prefix} "
                text = unicodedata.normalize(
                    "NFC", f"{head}{label_text(grid)}. Ngữ cảnh: {prose}")
                handle.write(json.dumps(
                    {"doc_name": doc_name, "table_id": int(row.table_id), "text": text},
                    ensure_ascii=False) + "\n")
                written += 1

            if docs_done % 400 == 0:
                print(f"  {docs_done} docs, {written} tables, {time.time() - started:.0f}s")

    lengths.sort()
    print(f"\n{written} tables from {docs_done} documents in {time.time() - started:.0f}s")
    print(f"tables with no prose above them: {empty}")
    print(f"anchor prose length: median {lengths[len(lengths) // 2]}, "
          f"p90 {lengths[int(len(lengths) * 0.9)]} chars "
          f"(labels capped at {MAX_LABEL_ROWS} rows)")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
