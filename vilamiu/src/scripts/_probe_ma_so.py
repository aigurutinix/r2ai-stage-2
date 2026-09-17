"""Locate the row by its statutory code instead of by matching its label.

Reading the dataset as the organisers released it — `data/financial_statements`,
not the processed `data/official_corpus` — shows the statements carry their tables
inline as HTML, and every balance-sheet and income-statement row carries a `Mã số`:

    <tr><td>2. Tài sản cố định vô hình</td><td>227</td><td>5.8</td><td>41.156.786.677</td>…

The code is fixed by the Vietnamese accounting regime. 227 is intangible fixed
assets in every report of every company in every year, whatever the OCR did to the
words next to it. `artifacts/ma_so.json` already holds 344 of them with a canonical
label and the label variants mined from the corpus, each seen in 275–347 documents.

The label matcher this project relies on scores 42.8% and works the other way round:
it matches the question's wording against whatever the OCR produced for the row. That
is an open vocabulary against a noisy target. Matching the question against 344
canonical labels and then finding the row whose code column equals that code is a
closed vocabulary against an exact target.

Measured here against the organisers' own cell coordinates, so the row either is the
one the gold program read or it is not.

Usage:
  PYTHONPATH=src python scripts/_probe_ma_so.py --limit 1500
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

# A statutory code is three digits, sometimes with a dotted sub-code.
CODE_RE = re.compile(r"^\s*(\d{3})(\.\d+)?\s*$")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def load_codes() -> list[tuple[str, str, set]]:
    """(code, canonical, token set) for every statutory line, canonical first."""

    raw = json.loads((ROOT / "artifacts" / "ma_so.json").read_text(encoding="utf-8"))
    entries = []
    for key, value in raw.items():
        code = key.split(":")[-1]
        labels = [value.get("canonical", "")]
        labels += [v.get("label", "") for v in value.get("variants", [])]
        for label in labels:
            tokens = {t for t in fold(label).split() if len(t) > 2}
            if tokens:
                entries.append((code, label, tokens))
    return entries


def best_code(metric: str, entries) -> tuple[str, float, str] | None:
    """The statutory line whose wording is closest to the question's metric."""

    wanted = {t for t in fold(metric).split() if len(t) > 2}
    if not wanted:
        return None
    best = None
    for code, label, tokens in entries:
        shared = len(wanted & tokens)
        if not shared:
            continue
        # Harmonic mean of coverage and focus, the same shape `match_row` uses:
        # coverage alone rewards the longest line in the chart of accounts.
        coverage = shared / len(wanted)
        focus = shared / len(tokens)
        score = 2 * coverage * focus / (coverage + focus)
        if best is None or score > best[1]:
            best = (code, score, label)
    return best


def code_column(grid) -> int | None:
    """The column holding statutory codes, if the table has one.

    Recognised by content rather than header text: an OCR'd header reads "Mã số",
    "MÃ SỐ", "Ma so" or nothing at all, but a column of three-digit codes is
    unmistakable.
    """

    if not grid or len(grid) < 3:
        return None
    width = max(len(row) for row in grid)
    best, best_hits = None, 0
    for column in range(1, min(width, 4)):
        hits = 0
        for row in grid[1:]:
            if column < len(row) and CODE_RE.match(str(row[column])):
                hits += 1
        if hits > best_hits:
            best, best_hits = column, hits
    # Require the column to be mostly codes, not a stray numeric coincidence.
    return best if best is not None and best_hits >= max(3, len(grid) // 4) else None


def gold_row_of(query: str, grid) -> int | None:
    match = ILOC_NAME.search(query) or ILOC_RC.search(query)
    if match is None:
        return None
    index = int(match.group(1))
    row = len(grid) + index if index < 0 else index + 1
    return row if 0 <= row < len(grid) else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    entries = load_codes()
    print(f"{len(entries)} nhãn pháp định (gồm biến thể)\n")

    tally: collections.Counter[str] = collections.Counter()
    examples = []
    seen = 0

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        grid = store.rows(TableKey(doc, int(table_id)))
        if not grid or len(grid) < 2:
            continue
        gold_row = gold_row_of(record.get("pandas_query") or "", grid)
        if gold_row is None:
            continue
        seen += 1

        column = code_column(grid)
        if column is None:
            tally["bảng không có cột Mã số"] += 1
            continue
        metric = lookup_mod.extract_metric(record["question"])
        found = best_code(metric, entries)
        if found is None or found[1] < args.min_score:
            tally["không khớp được mã"] += 1
            continue
        code, score, label = found
        rows = [index for index, row in enumerate(grid)
                if column < len(row)
                and (CODE_RE.match(str(row[column])) or [None])[0]
                and CODE_RE.match(str(row[column])).group(1) == code]
        if not rows:
            tally["mã không có trong bảng"] += 1
            continue
        if len(rows) > 1:
            tally["mã trùng nhiều dòng"] += 1
        picked = rows[0]
        if picked == gold_row:
            tally["ĐÚNG dòng"] += 1
        else:
            tally["sai dòng"] += 1
            if len(examples) < args.show:
                examples.append((record.get("id"), metric[:38], label[:34], code,
                                 str(grid[gold_row][0])[:34],
                                 str(grid[picked][0])[:34]))

    decided = tally["ĐÚNG dòng"] + tally["sai dòng"]
    print(f"{seen} bản ghi gold")
    for name, count in tally.most_common():
        print(f"  {name:26s} {count:5d}  {count / max(seen, 1):6.1%}")
    if decided:
        print(f"\n  ĐÚNG | có quyết định   {tally['ĐÚNG dòng']}/{decided} = "
              f"{tally['ĐÚNG dòng'] / decided:.1%}")
    print(f"  ĐÚNG | toàn bộ         {tally['ĐÚNG dòng']}/{seen} = "
          f"{tally['ĐÚNG dòng'] / max(seen, 1):.1%}")
    print("  (so: bộ khớp nhãn qua find() 30,2% toàn bộ / 51,7% trên số cam kết)")
    for qid, metric, label, code, gold_label, our_label in examples:
        print(f"\n  id={qid} tra {metric!r}")
        print(f"     khớp mã {code} = {label!r}")
        print(f"     gold: {gold_label!r}")
        print(f"     ta  : {our_label!r}")


if __name__ == "__main__":
    main()
