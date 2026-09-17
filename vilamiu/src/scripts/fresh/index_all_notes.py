"""Index every table of every report, identified by its heading — not only the tied ones.

The measurement that forces this: the oracle's answer sits somewhere in the resolved
document for 66% of the questions that can be checked, and inside the current index for
21%. Forty-five points are in the report and outside the address space. Document
resolution is not the problem — ticker, year and scope land on the right report almost
always — and neither is reading a cell, which the identities check at 97–99.6%. The
index simply covers about twenty-six of a report's seventy-five tables.

What was dropped and why it was wrong to drop it: a note was kept only if its column
total tied to a statement line in BOTH periods. That was a quality filter, and quality
was the wrong thing to optimise while coverage was the binding constraint — the same
mistake as the refusal gates, made a second time. Reports whose statements never parsed
lost every note as collateral, because there was nothing left to tie against.

But this must not become "throw more candidates at the same ranker". That has been
measured four times and lost every time: a wider pool picked by token overlap is worse
than a narrow one. What makes a note navigable is its heading, which agrees with the
code it ties to 78.8% of the time — so the address becomes two levels, question to
heading and heading to row among about ten, which is the shape that scored 43% where
undifferentiated greedy scored 3.7%.

So every table is indexed with its anchor heading, and the tie is kept as a confidence
flag rather than an entry condition.

Usage:  python scripts/fresh/index_all_notes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_index2 import DOC_RE, banks, page_units  # noqa: E402
from fix_headings import headings_for  # noqa: E402

MIN_ROWS = 2
# A table with no figure at all is prose, not a source of answers.
MIN_NUMBERS = 2


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="artifacts/fresh/all_tables.jsonl")
    args = parser.parse_args()

    bank_tickers = banks()
    docs = sorted(p for p in (ROOT / "data" / "official_corpus").glob("*/*/*")
                  if p.is_dir())
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs)} bao cao", flush=True)

    handle = (ROOT / args.out).open("w", encoding="utf-8")
    stats: Counter[str] = Counter()
    started = time.time()

    for index, doc_dir in enumerate(docs, start=1):
        doc_name = doc_dir.name
        match = DOC_RE.match(doc_name)
        if not match:
            stats["ten tai lieu khong khop mau"] += 1
            continue
        ticker, year, raw_scope = match.group(1), match.group(2), match.group(3)
        scope = ("separate" if "separate" in raw_scope
                 else "consolidated" if "consolidated" in raw_scope else raw_scope)
        text_path = doc_dir / f"{doc_name}_extracted.txt"
        table_dir = doc_dir / f"{doc_name}_extracted_tables"
        if not text_path.exists() or not table_dir.is_dir():
            stats["thieu text hoac thu muc bang"] += 1
            continue

        text = text_path.read_text(encoding="utf-8", errors="replace")
        table_page, snippets = page_units(text)
        note_headings = headings_for(text)

        for csv_path in sorted(table_dir.glob("table_*.csv")):
            table_id = int(csv_path.stem.split("_")[-1])
            try:
                with csv_path.open(encoding="utf-8-sig", newline="") as file:
                    rows = [row for row in csv.reader(file)]
            except OSError:
                continue
            if len(rows) < MIN_ROWS + 1:
                continue
            stats["bang"] += 1

            table = ps.Table(doc_name=doc_name, table_id=table_id,
                             page_no=table_page.get(table_id, 0),
                             header=rows[0], rows=rows[1:],
                             unit_snippets=snippets.get(table_id, ()))
            statement = ps.parse_statement(table)
            if statement is not None:
                stats[f"bao cao chinh {statement.kind}"] += 1
                continue  # already in the Mã số index

            # Rows worth addressing: a label plus at least one figure.
            body = []
            numbers = 0
            for row_index, row in enumerate(table.rows):
                columns = []
                for position, cell in enumerate(row):
                    raw = str(cell).strip()
                    if not raw or ps.BARE_INT_RE.match(raw):
                        continue
                    value = ps.parse_vn_number(raw)
                    if value is not None:
                        columns.append([position, value])
                if not columns:
                    continue
                label = max((str(c).strip() for c in row
                             if not any(ch.isdigit() for ch in str(c))),
                            key=len, default="")
                if not label:
                    continue
                numbers += len(columns)
                body.append({"row": row_index, "label": label[:110],
                             "cols": columns[:4]})
            if numbers < MIN_NUMBERS or not body:
                stats["bang khong co dong dung duoc"] += 1
                continue

            scale = ps.scale_from_unit_text(",".join(table.header))
            if scale is None:
                for snippet in table.unit_snippets:
                    scale = ps.scale_from_unit_text(snippet)
                    if scale is not None:
                        break
            stats["thuyet minh co index" if scale is not None
                  else "thuyet minh KHONG ro don vi"] += 1
            handle.write(json.dumps({
                "doc": doc_name, "ticker": ticker, "year": year, "scope": scope,
                "table_id": table_id,
                "table_ref": f"{doc_name}|table_{table_id}",
                "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "heading": note_headings.get(table_id, "")[:110],
                "scale": scale,
                "bank": ticker in bank_tickers,
                "rows": body[:60],
            }, ensure_ascii=False) + "\n")

        if index % 200 == 0:
            print(f"  {index}/{len(docs)}  {time.time() - started:.0f}s", flush=True)

    handle.close()
    for name, count in stats.most_common():
        print(f"  {name}: {count}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
