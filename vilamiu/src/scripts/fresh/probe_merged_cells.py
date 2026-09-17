"""Does OCR merge the Mã số into the value cell, and can the label tell them apart?

9,889 tables hold rows with a `Mã số` and still fail to classify as a statement, and
another 3,631 classify but are discarded for want of a unit. Something is eating the
structure, and one candidate is that the OCR collapses two cells into one: the code
column and the figure beside it arrive as a single run of digits, `110123.456.789`,
with nothing to say where the code stops.

The split is guessable because a third signal is present. Take the leading one, two
or three digits as a candidate code, require the remainder to parse as a Vietnamese
number, and then ask whether the ROW LABEL matches what that code is called in the
consensus dictionary — a dictionary built only from statements whose printed
arithmetic already checks out. Where the label agrees, the split is almost certainly
right; where it does not, the row is left alone.

This only measures. It reports how often merged cells appear, how often a split is
unambiguous, and what the labels say about it — enough to decide whether the repair
is worth building.

Usage:  python scripts/fresh/probe_merged_cells.py --limit 300
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from refine_codes import RELIABLE_LEN, agrees, tokens  # noqa: E402

def merged_splits(cell: str) -> list[tuple[str, str]]:
    """Ways to read one cell as a `Mã số` glued to a figure.

    The test has to be that the cell is NOT a number on its own. A pattern like
    `^(\\d{1,3})[\\s.]?(\\d{1,3}(\\.\\d{3})+)$` also matches every ordinary
    Vietnamese figure — `1.234.567` splits into "1" and "234.567" — and firing on
    those produced 167,930 hits, almost all of them nonsense. A genuinely merged
    cell breaks the grouping instead: `110123.456.789` starts with six ungrouped
    digits, so it fails to parse, and only becomes a valid figure once the leading
    code is removed.
    """

    text = str(cell).strip()
    if not text or ps.parse_vn_number(text) is not None:
        return []
    negative = text.startswith("(")
    body = text[1:] if negative else text
    out = []
    for length in (3, 2, 1):
        if len(body) <= length or not body[:length].isdigit():
            continue
        head, rest = body[:length], body[length:].lstrip(" .")
        if not rest:
            continue
        candidate = f"({rest}" if negative and not rest.startswith("(") else rest
        if ps.parse_vn_number(candidate if negative else rest) is not None:
            out.append((head, rest))
    return out


def reference_dictionary() -> dict[tuple[str, str], frozenset[str]]:
    """Consensus labels per (kind, code), from statements the identities verified."""

    votes: dict[tuple[str, str], Counter[frozenset[str]]] = defaultdict(Counter)
    for line in (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ok = False
        for kind, target, plus, minus in IDENTITIES:
            if record["kind"] != kind:
                continue
            for period in ("current", "prior"):
                cells = record[period]
                if not all(c in cells for c in (target,) + plus + minus):
                    continue
                expected = cells[target][0]
                total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
                if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) == want:
                key = tokens(cell[1])
                if key:
                    votes[(record["kind"], code)][key] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    reference = reference_dictionary()
    print(f"tu dien tham chieu (chi tu bang qua dang thuc): {len(reference)} ma")

    files = sorted((ROOT / "data" / "financial_statements").glob("*/*/*/*.txt"))
    if args.limit:
        files = files[:args.limit]

    counters: Counter[str] = Counter()
    examples: list[str] = []
    for path in files:
        try:
            tables = ps.read_document(path)
        except Exception:  # noqa: BLE001
            continue
        for table in tables:
            hint = ps._header_ma_so_index(table.header)
            # Only tables the current parser cannot use: no row yields a code, or
            # the codes it finds do not classify.
            usable = []
            for row_idx, row in enumerate(table.rows):
                ma_so_idx = ps._find_ma_so_index(row, hint)
                if ma_so_idx is None:
                    continue
                values = ps._value_cells(row, exclude=ma_so_idx)
                if values:
                    usable.append((str(row[ma_so_idx]).strip(), row_idx,
                                   ps._row_label(row, ma_so_idx,
                                                 {i for i, _, _ in values})))
            kind = ps._classify({u[0] for u in usable},
                                [u[2] for u in usable]) if usable else None

            for row in table.rows:
                merged = []
                for cell in row:
                    merged += merged_splits(cell)
                if not merged:
                    continue
                counters["dong co o dang nghi bi dan"] += 1
                # The label is whatever text the row carries.
                label = max((str(c).strip() for c in row
                             if not any(ch.isdigit() for ch in str(c))),
                            key=len, default="")
                if not label:
                    counters["  khong co nhan de doi chieu"] += 1
                    continue
                row_tokens = tokens(label)
                hits = []
                for kinds in (("cdkt",), ("kqkd",), ("lctt",)) if kind is None else ((kind,),):
                    for candidate_kind in kinds:
                        for code, _value in merged:
                            for length in (3, 2, 1):
                                head = code[:length]
                                key = (candidate_kind, head)
                                if key in reference and agrees(row_tokens, reference[key]):
                                    hits.append((candidate_kind, head, code))
                if hits:
                    counters["  NHAN KHOP mot ma cat ra duoc"] += 1
                    if len(examples) < args.show:
                        k, head, whole = hits[0]
                        examples.append(
                            f"  {k} cat '{whole}' -> ma={head}  nhan='{label[:52]}'")
                else:
                    counters["  nhan khong khop ma nao"] += 1

    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
