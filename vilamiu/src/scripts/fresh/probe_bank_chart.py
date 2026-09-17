"""What chart of accounts do the 21 credit institutions use?

Their statements fail every enterprise identity — `kqkd: 20 = 10 - 11` holds for 0 of
70 bank cases — because a bank reports under a different statutory layout. Worse, most
of their tables never classify as a statement at all: the classifier needs five or
more three-digit codes, or a label containing "chuyển tiền", or three of
{01, 11, 20, 25, 26}, and a bank income statement has none of those.

So the bank reports are largely invisible to the address book, and 21 of the 100
companies is a fifth of the corpus.

This looks at what is actually there: for the tickers flagged `Tổ chức tín dụng` in
`data/file_filter.csv`, take every table carrying `Mã số`-looking codes and report
which codes appear with which labels. That is the raw material for a bank code set,
and once there is one, the bank's own printed identities can verify it the same way
Thông tư 200's verify the enterprise one.

Usage:  python scripts/fresh/probe_bank_chart.py --limit 40
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_index2 import banks, page_units  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40, help="so bao cao")
    parser.add_argument("--show", type=int, default=26)
    args = parser.parse_args()

    bank_tickers = banks()
    docs = sorted(p for p in (ROOT / "data" / "official_corpus").glob("*/*/*")
                  if p.is_dir() and p.parts[-3] in bank_tickers)
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(bank_tickers)} ma to chuc tin dung, xem {len(docs)} bao cao")

    labels: dict[str, Counter[str]] = defaultdict(Counter)
    per_table: Counter[str] = Counter()
    classified = 0
    for doc_dir in docs:
        doc_name = doc_dir.name
        text_path = doc_dir / f"{doc_name}_extracted.txt"
        table_dir = doc_dir / f"{doc_name}_extracted_tables"
        if not text_path.exists() or not table_dir.is_dir():
            continue
        table_page, snippets = page_units(
            text_path.read_text(encoding="utf-8", errors="replace"))
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
                classified += 1

            hint = ps._header_ma_so_index(table.header)
            found = 0
            for row in table.rows:
                ma_so_idx = ps._find_ma_so_index(row, hint)
                if ma_so_idx is None:
                    continue
                values = ps._value_cells(row, exclude=ma_so_idx)
                if not values:
                    continue
                code = str(row[ma_so_idx]).strip()
                label = ps._row_label(row, ma_so_idx,
                                      {i for i, _, _ in values})
                if label:
                    labels[code][" ".join(label.split())] += 1
                found += 1
            if found >= 5:
                per_table[f"bang co >=5 dong co ma"] += 1

    print(f"bang phan loai duoc thanh bao cao chinh: {classified}")
    for name, count in per_table.most_common():
        print(f"  {name}: {count}")

    print(f"\nma xuat hien nhieu nhat va nhan pho bien nhat:")
    ranked = sorted(labels.items(), key=lambda item: -sum(item[1].values()))
    for code, counter in ranked[:args.show]:
        total = sum(counter.values())
        top = counter.most_common(1)[0][0]
        print(f"  {code:>4s} ({total:5d} lan)  {top[:82]}")


if __name__ == "__main__":
    main()
