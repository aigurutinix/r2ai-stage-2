"""Identify a note table by tying its total to the statement line it details.

250 of the questions the Mã số address book cannot reach ask for note-table items with
a specific qualifier — "cho vay khách hàng ngành Thương mại", "thù lao HĐQT Chu Thị
Bình", "doanh thu cho thuê khô tàu bay". None of those exist in the statutory chart,
and a note table carries no `Mã số`, so nothing in the address book addresses them.

What a note does carry is a total, and that total is a statement line. The note
breaking down cash sums to `cdkt/110`; the note on trade receivables sums to
`cdkt/131`. The statement address book already holds those figures, verified at
97–99.6% by the printed identities, so the tie is checkable arithmetic rather than a
label match.

That single test does three things at once:

  identifies    a table whose column total equals `cdkt/110` IS the cash note — no
                heading to parse, no wording to match
  verifies      the residual is a signed number over several independent cells, so
                two coincident errors cannot satisfy it
  recovers the unit  a note declares its own scale and often the page does not; only
                one of 1, 1e3, 1e6, 1e9 makes the total land on a statement line, and
                that one is the note's scale

Also captures the text between the previous `[table_N]` anchor and this one, which is
the note's own heading — the second addressing route, kept for comparison rather than
relied on.

Usage:  python scripts/fresh/tie_notes.py --limit 0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_index2 import DOC_RE, banks, page_units  # noqa: E402

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
PAGE_RE = re.compile(r"===== PAGE (\d+) =====")
SCALES = (1.0, 1e3, 1e6, 1e9)
# A note total must land on a statement line to within OCR noise, not exactly: the
# statement may round where the note does not.
REL_TOL = 1e-4
# Below this a coincidence is likely: small integers collide across a whole report.
MIN_MAGNITUDE = 1e6


def headings(text: str) -> dict[int, str]:
    """The text between the previous anchor and each anchor: the note's own heading."""

    out: dict[int, str] = {}
    marks = list(PAGE_RE.finditer(text))
    pages = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        pages.append(text[mark.end():end])
    if not pages:
        pages = [text]
    for body in pages:
        anchors = list(ANCHOR_RE.finditer(body))
        for index, anchor in enumerate(anchors):
            start = anchors[index - 1].end() if index else 0
            chunk = ANCHOR_RE.sub("", body[start:anchor.start()])
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            out[int(anchor.group(1))] = " | ".join(lines[-3:])[:180]
    return out


def column_totals(rows: list[list[str]]) -> dict[int, list[float]]:
    """Per column, the candidate totals: the last value, the largest, and the sum."""

    per_column: dict[int, list[float]] = defaultdict(list)
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for index, cell in enumerate(row):
            raw = str(cell).strip()
            if not raw or ps.BARE_INT_RE.match(raw):
                continue
            value = ps.parse_vn_number(raw)
            if value is not None:
                values[index].append(value)
    for index, column in values.items():
        if len(column) < 2:
            continue
        candidates = {column[-1], max(column, key=abs), sum(column)}
        # A statement often prints the note's total, and the note prints it too, so
        # the sum of every row double counts it. Offer the sum without the largest
        # term as well.
        without = sum(column) - max(column, key=abs)
        candidates.add(without)
        per_column[index] = [c for c in candidates if abs(c) >= MIN_MAGNITUDE / 1e3]
    return per_column


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/notes.jsonl")
    args = parser.parse_args()

    # (ticker, year, scope) -> [(value, kind, code, period)]
    book: dict[tuple, list[tuple[float, str, str, str]]] = defaultdict(list)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        key = (record["ticker"], record["year"], scope)
        for period in ("current", "prior"):
            for code, cell in record[period].items():
                if abs(cell[0]) >= MIN_MAGNITUDE:
                    book[key].append((cell[0], record["kind"], code, period))
    print(f"so bao cao co so dia chi: {len(book)}", flush=True)

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
            continue
        ticker, year, raw_scope = match.group(1), match.group(2), match.group(3)
        scope = ("separate" if "separate" in raw_scope
                 else "consolidated" if "consolidated" in raw_scope else raw_scope)
        lines = book.get((ticker, year, scope))
        if not lines:
            stats["bao cao khong co so dia chi"] += 1
            continue

        text_path = doc_dir / f"{doc_name}_extracted.txt"
        table_dir = doc_dir / f"{doc_name}_extracted_tables"
        if not text_path.exists() or not table_dir.is_dir():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        table_page, snippets = page_units(text)
        note_headings = headings(text)

        for csv_path in sorted(table_dir.glob("table_*.csv")):
            table_id = int(csv_path.stem.split("_")[-1])
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                rows = [row for row in csv.reader(file)]
            if len(rows) < 3:
                continue
            table = ps.Table(doc_name=doc_name, table_id=table_id,
                             page_no=table_page.get(table_id, 0),
                             header=rows[0], rows=rows[1:],
                             unit_snippets=snippets.get(table_id, ()))
            if ps.parse_statement(table) is not None:
                continue  # already addressable through the Mã số book
            stats["bang khong phai bao cao chinh"] += 1

            hits = []
            for column, candidates in column_totals(table.rows).items():
                for candidate in candidates:
                    for scale in SCALES:
                        total = candidate * scale
                        if abs(total) < MIN_MAGNITUDE:
                            continue
                        for value, kind, code, period in lines:
                            if abs(total - value) <= REL_TOL * max(abs(total),
                                                                   abs(value)):
                                hits.append({"col": column, "scale": scale,
                                             "kind": kind, "code": code,
                                             "period": period,
                                             "total": total})
                                break
            if not hits:
                continue
            stats["bang NOI DUOC voi mot dong bao cao chinh"] += 1
            # A single tie can be coincidence: four candidate totals times four
            # scales against a few hundred codes gives the search plenty of chances.
            # A tie that holds for BOTH periods of the same code is two independent
            # matches in two different columns, and the hand-checked samples split
            # exactly that way — the right notes tied on both periods, the doubtful
            # ones on only one.
            periods_by_code: dict[tuple[str, str], set[str]] = defaultdict(set)
            for hit in hits:
                periods_by_code[(hit["kind"], hit["code"])].add(hit["period"])
            paired = sorted(f"{kind}/{code}" for (kind, code), periods
                            in periods_by_code.items() if len(periods) >= 2)
            if paired:
                stats["  trong do NOI CA HAI KY cung mot ma"] += 1
            handle.write(json.dumps({
                "doc": doc_name, "ticker": ticker, "year": year, "scope": scope,
                "table_id": table_id,
                "table_ref": f"{doc_name}|table_{table_id}",
                "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "heading": note_headings.get(table_id, ""),
                "n_rows": len(table.rows),
                "ties": hits[:6],
                "paired": paired,
                "labels": [str(r[0])[:70] for r in table.rows[:14] if r],
            }, ensure_ascii=False) + "\n")

        if index % 200 == 0:
            print(f"  {index}/{len(docs)}  noi duoc="
                  f"{stats['bang NOI DUOC voi mot dong bao cao chinh']}  "
                  f"{time.time() - started:.0f}s", flush=True)

    handle.close()
    for name, count in stats.most_common():
        print(f"  {name}: {count}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
