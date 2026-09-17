"""Does the printed note pointer actually lead somewhere, and does the total tie back?

The design in PIPELINE.md rests on one claim: a statement line names the note that details
it, so the note can be reached by navigation instead of by matching text. 2,875 such
pointers were counted across 297 documents. Counting them is not the same as being able to
follow them, and following them is not the same as the arithmetic agreeing at the far end.

This measures all three, with no model and no submission:

  found     the pointer "5.2" resolves to a table whose printed heading starts with 5.2
  tied      that table has a column whose total equals the statement line's value
  broken    it resolves but the arithmetic disagrees — a parse error somewhere, and the
            question should be dropped rather than answered

`tied` is the number that matters. It is a check that shares no code with the retriever,
the renderer or the reader, so unlike every offline figure produced so far it cannot be
confirming itself.

Usage:
  python scripts/fresh/measure_pointers.py --docs 30
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from find_statements import locate_columns  # noqa: E402

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
# A note number as printed in a statement's Thuyết minh column: 6, 5.2, V.12, B7.33.
POINTER_RE = re.compile(r"^([VB]?)\.?\s?(\d{1,2})(?:\.(\d{1,2}))?$")
CONTEXT_CHARS = 400
TOLERANCE = 0.02  # relative; OCR rounds, and a note may exclude a rounding line


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()


def contexts_of(path: Path) -> dict[int, str]:
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous = {}, 0
    for match in ANCHOR_RE.finditer(body):
        out[int(match.group(1))] = re.sub(
            r"\s+", " ", body[previous:match.start()].strip())[-CONTEXT_CHARS:]
        previous = match.end()
    return out


def heading_numbers(context: str) -> set[str]:
    """Numbered headings printed just above a table: "5.2", "6.", "B7.33"."""

    found = set()
    for match in re.finditer(r"(?:^|\s)([VB]?\.?\s?\d{1,2}(?:\.\d{1,2})?)[.\s)]",
                             context[-200:]):
        found.add(re.sub(r"[\s.]", "", match.group(1)).upper())
    return found


def column_totals(grid: list[list[str]]) -> list[float]:
    """Every column's sum of parsed cells, and any cell that is itself a printed total."""

    out = []
    width = max((len(r) for r in grid), default=0)
    for column in range(width):
        values = []
        for row in grid[1:]:
            if column >= len(row):
                continue
            raw = str(row[column]).strip()
            if not raw or (ps.BARE_INT_RE.match(raw) and len(raw) <= 3):
                continue
            parsed = ps.parse_vn_number(raw)
            if parsed is not None:
                values.append(parsed)
        if values:
            out.append(sum(values))
            out.extend(values)          # a printed "TỔNG CỘNG" row is one of these
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=30)
    parser.add_argument("--show", type=int, default=6)
    # The control this project has needed three times and skipped three times. A note
    # table holds dozens of cells and the tie is allowed against any of them, so a
    # statement line could match almost any note by chance. Pairing each line with the
    # WRONG note measures that chance rate; the real rate only means something to the
    # extent it exceeds it.
    parser.add_argument("--permute", action="store_true")
    args = parser.parse_args()

    corpus = ROOT / "data" / "official_corpus"
    doc_dirs = sorted(p for p in corpus.glob("*/*/*") if p.is_dir())[:args.docs]

    counters: Counter[str] = Counter()
    samples = []

    for doc_dir in doc_dirs:
        tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
        if not tables_dir.is_dir():
            continue
        anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")

        grids: dict[int, list[list[str]]] = {}
        for path in sorted(tables_dir.glob("table_*.csv"),
                           key=lambda p: int(p.stem.split("_")[-1])):
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    grids[int(path.stem.split("_")[-1])] = list(csv_mod.reader(handle))
            except OSError:
                continue

        # Index every table by the numbered headings printed above it.
        by_number: dict[str, list[int]] = {}
        for table_id in grids:
            for number in heading_numbers(anchors.get(table_id, "")):
                by_number.setdefault(number, []).append(table_id)

        for table_id, grid in grids.items():
            if not grid:
                continue
            # Columns located by content when the header does not name them, which is
            # most of the corpus: demanding a "Mã số" header in row 0 hid 23% of
            # ordinary companies' statements.
            ma_i, tm_i = locate_columns(grid)
            if ma_i is None or tm_i is None:
                continue
            values_start = max(ma_i, tm_i) + 1

            for row in grid[1:]:
                if len(row) <= values_start:
                    continue
                code = str(row[ma_i]).strip()
                pointer = str(row[tm_i]).strip()
                if not re.fullmatch(r"\d{1,3}", code):
                    continue
                match = POINTER_RE.match(pointer)
                if not match:
                    continue
                counters["dong co con tro"] += 1
                key = re.sub(r"[\s.]", "", pointer).upper()
                targets = by_number.get(key) or by_number.get(
                    key.lstrip("VB")) or []
                targets = [t for t in targets if t != table_id]
                if args.permute:
                    # Every pointed-to table in this document EXCEPT the right one.
                    others = sorted({t for group in by_number.values()
                                     for t in group}
                                    - set(targets) - {table_id})
                    targets = others[:len(targets)] if others else []
                if not targets:
                    counters["con tro KHONG dan tới bang nao"] += 1
                    continue
                counters["con tro dan tới bang"] += 1

                # The statement line's own value, first value column.
                line = None
                for cell in row[values_start:]:
                    raw = str(cell).strip()
                    if not raw:
                        continue
                    parsed = ps.parse_vn_number(raw)
                    if parsed is not None and parsed:
                        line = abs(parsed)
                        break
                if line is None:
                    counters["  dong khong co gia tri"] += 1
                    continue

                hit = False
                for target in targets:
                    for total in column_totals(grids.get(target, [])):
                        if total and abs(abs(total) - line) <= TOLERANCE * line:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    counters["  KHOP tong thuyet minh"] += 1
                    if len(samples) < args.show:
                        samples.append(
                            f"  {doc_dir.name[:34]:34s} ma {code:>3s} -> tm {pointer:>5s}"
                            f"  {str(row[0]).strip()[:38]}")
                else:
                    counters["  khong khop (parse sai o dau do)"] += 1

    print(f"{len(doc_dirs)} tai lieu\n")
    for name, count in counters.most_common():
        print(f"  {count:6d}  {name}")
    pointed = counters["con tro dan tới bang"]
    tied = counters["  KHOP tong thuyet minh"]
    if pointed:
        print(f"\nti le khop tong / con tro dan duoc: {tied}/{pointed} = "
              f"{100 * tied / pointed:.0f}%")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
