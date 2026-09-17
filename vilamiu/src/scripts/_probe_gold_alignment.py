"""Does each gold question match the table and answer stored with it?

Five records read by hand after the unit correction showed something worse than a
unit bug. A question about hydroelectric plants carried a table captioned "21.
PHẢI TRẢ DÀI HẠN KHÁC"; a question about cost of goods sold carried one about
land-lease income; one answer was the literal cell "1", a row number from a "TT"
column. If that is common, every A/B measured against this set today is noise, and
the ideas eliminated on those measurements were not eliminated at all.

Three checks, all mechanical:

* **label overlap** -- do the content words of the question appear in any row
  label of the gold table? A question naming a line item whose words appear
  nowhere in the table it is scored against cannot be answerable from it.
* **answer present** -- is the stored answer actually a cell of that table?
* **degenerate answer** -- is it a tiny integer, the shape of a row number rather
  than a financial figure.

Usage:
  PYTHONPATH=src python scripts/_probe_gold_alignment.py --limit 300
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
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import parse_number  # noqa: E402

from vifin.store import TableKey, TableStore  # noqa: E402

# Words that appear in nearly every question and so carry no signal about which
# table is meant.
STOP = {
    "cua", "la", "bao", "nhieu", "nam", "cuoi", "dau", "vao", "tai", "cong", "ty",
    "ctcp", "tong", "cp", "va", "cac", "trong", "den", "ngay", "thang", "co",
    "phan", "tap", "doan", "viet", "nam", "dong", "ty dong", "trieu", "nghin",
    "so", "du", "gia", "tri", "muc", "khoan", "mot", "hay", "theo", "tinh",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", text)


def content_words(text: str) -> set:
    return {w for w in fold(text).split() if len(w) > 2 and w not in STOP}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-overlap", type=float, default=0.34,
                        help="share of the question's content words that must "
                             "appear among the table's row labels")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][: args.limit]

    stat: collections.Counter[str] = collections.Counter()
    bad_examples = []
    for record in records:
        refs = record.get("relevant_tables") or []
        answer = parse_number(record.get("answer"))
        if not refs or answer is None:
            stat["unusable record"] += 1
            continue
        doc, tid = refs[0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc, int(tid)))
        if not grid:
            stat["gold table missing"] += 1
            continue

        labels = " ".join(str(row[0]) for row in grid[1:] if row)
        meta = store.meta(TableKey(doc, int(tid)))
        haystack = content_words(labels + " " + str(getattr(meta, "caption", "")))
        wanted = content_words(record["question"])
        overlap = len(wanted & haystack) / max(len(wanted), 1)

        in_table = any(
            parse_number(cell) == answer
            for row in grid[1:] for cell in row if parse_number(cell) is not None
        )

        if abs(answer) < 1000 and float(answer).is_integer():
            stat["answer looks like a row number"] += 1
            if len(bad_examples) < 3:
                bad_examples.append(("row-number answer", record["question"][:88],
                                     record["answer"]))
            continue
        if not in_table:
            stat["answer is not a cell of its gold table"] += 1
            continue
        if overlap < args.min_overlap:
            stat["question words absent from gold table"] += 1
            if len(bad_examples) < 6:
                bad_examples.append((f"overlap {overlap:.0%}",
                                     record["question"][:88],
                                     str(getattr(meta, "caption", ""))[:60]))
            continue
        stat["aligned"] += 1

    total = sum(stat.values()) or 1
    print(f"{total} records\n")
    for name, count in stat.most_common():
        print(f"  {count:4d}  {100 * count / total:5.1f}%  {name}")
    print()
    for tag, question, extra in bad_examples:
        print(f"  [{tag}] {question}")
        print(f"        -> {extra}")


if __name__ == "__main__":
    main()
