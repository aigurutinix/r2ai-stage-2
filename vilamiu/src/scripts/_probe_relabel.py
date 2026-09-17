"""Can the generator's positional queries be rewritten as label lookups?

62% of generated `pandas_query` values are pure positional reads — `df.iloc[2, 1]`.
Training on them teaches the model to count rows, which is the one thing our own
A/B says it is worst at: asking for `{table, row, column}` measured 13.3%, while
handing over `find_row()` lifted runnable programs to 76.9%. The other 36% are
already label lookups, so the generator proves the better form is reachable; it
just does not choose it consistently.

A positional read is mechanically convertible: with the grid in hand, row `r`'s
label is the first non-empty cell of that row and column `c`'s name is the header.
The question is what fraction converts *and still returns the same number*, which
is not something to assume — a table with repeated or blank row labels cannot be
addressed by label at all, and a conversion that silently picks the wrong
duplicate would teach a wrong cell while looking correct.

So every rewrite here is executed and compared against the original answer. Only
the agreeing ones count.

Nothing is written. This measures whether the change is worth making.

Usage:  PYTHONPATH=src python scripts/_probe_relabel.py [records.jsonl]
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "easy_full.jsonl"

# `df.iloc[2, 1]` and `df.iloc[2][1]`, the two shapes the generator emits.
ILOC_PAIR = re.compile(r"^\s*result\s*=\s*df\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$")
ILOC_CHAIN = re.compile(r"^\s*result\s*=\s*df\.iloc\[\s*(-?\d+)\s*\]\[\s*(-?\d+)\s*\]\s*$")


def load_grid(csv_path: Path) -> list[list[str]]:
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.reader(handle))


def main() -> None:
    import pandas as pd

    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    stats: Counter[str] = Counter()
    samples: list[tuple[str, str]] = []

    for record in rows:
        query = str(record.get("pandas_query") or "").strip()
        match = ILOC_PAIR.match(query) or ILOC_CHAIN.match(query)
        if match is None:
            stats["not_positional" if "iloc" not in query else "positional_other_shape"] += 1
            continue

        stats["positional_simple"] += 1
        row_index, col_index = int(match.group(1)), int(match.group(2))

        path = ROOT / str(record["csv_path"])
        if not path.exists():
            stats["csv_missing"] += 1
            continue
        grid = load_grid(path)
        if not grid:
            stats["csv_empty"] += 1
            continue

        header, body = grid[0], grid[1:]
        if not (-len(body) <= row_index < len(body)) or not (0 <= col_index < len(header)):
            stats["index_outside_grid"] += 1
            continue

        # The label column is not always column 0. In these statements column 0 is
        # routinely a serial number, a year, or the figure itself, and matching on
        # it produced rewrites like `df.loc[df['2021'] == '1.794.379.900', '2021']`
        # — a row addressed by its own value, which is a positional read wearing a
        # string costume and a worse training target than the `iloc` it replaced.
        # The organisers' own `table_index_text` takes the first *non-numeric* cell
        # of the leading columns, which is the thing a human would call the label.
        line = body[row_index]
        label_col = None
        for index in range(min(3, len(header))):
            if index >= len(line):
                continue
            cell = (line[index] or "").strip()
            name = (header[index] or "").strip()
            if not cell or not name or index == col_index:
                continue
            if re.fullmatch(r"[\d.,()%\-\s]+", cell):  # a number, not a label
                continue
            label_col = index
            break
        if label_col is None:
            stats["row_has_no_label"] += 1
            continue

        label = (line[label_col] or "").strip()
        column = (header[col_index] or "").strip()
        if not column:
            stats["column_has_no_name"] += 1
            continue
        # A duplicated label cannot identify one row, and picking the first would
        # teach a cell the question never meant.
        if sum(1 for other in body
               if label_col < len(other)
               and (other[label_col] or "").strip() == label) > 1:
            stats["label_not_unique"] += 1
            continue

        rewritten = (f"result = df.loc[df[{header[label_col]!r}] == {label!r}, "
                     f"{column!r}].values[0]")
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            scope: dict[str, object] = {"df": frame, "pd": pd}
            exec(compile(query, "<original>", "exec"), scope)
            original = scope["result"]
            scope = {"df": frame, "pd": pd}
            exec(compile(rewritten, "<rewritten>", "exec"), scope)
            if str(scope["result"]).strip() == str(original).strip():
                stats["CONVERTS_AND_AGREES"] += 1
                if len(samples) < 3:
                    samples.append((query, rewritten))
            else:
                stats["converts_but_differs"] += 1
        except Exception:
            stats["rewrite_failed_to_run"] += 1

    total = len(rows)
    print(f"{total} records from {SOURCE.name}\n")
    for key, count in stats.most_common():
        print(f"  {key:26s} {count:5d}   {count / total:5.1%}")

    simple = stats["positional_simple"] or 1
    print(f"\n  of the {simple} simple positional reads, "
          f"{stats['CONVERTS_AND_AGREES'] / simple:.1%} convert to a label lookup "
          f"that returns the identical value")

    for before, after in samples:
        print(f"\n    {before}\n  -> {after}")

    print("\n  The failures are not noise to be tuned away: a table whose rows"
          "\n  repeat a label, or whose first column is blank, genuinely cannot be"
          "\n  addressed by label. Those records keep their positional form.")


if __name__ == "__main__":
    main()
