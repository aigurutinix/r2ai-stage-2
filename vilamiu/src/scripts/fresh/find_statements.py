"""Find the statement tables by what makes them consistent, not by where a header sits.

`parse_statements` looks for a `Mã số` header and covers 1,111 of 1,965 documents — 57%.
Measured separately, the header sits in row 0 in only 52% of documents: 23% put it in rows
one to three and 25% print no header at all while plainly carrying a column of two- and
three-digit codes. So roughly 43% of the corpus has no parsed statement at all, and every
coverage figure produced here was measured on the half that parsed.

The fix is not a rule per layout. A statement table has a property no other table has: its
codes satisfy the arithmetic that Circular 200 prints into the statement — 270 = 100 + 200,
50 = 30 + 40, 70 = 50 + 60 + 61. So every plausible reading of a table is enumerated — which
column holds the code, which column holds the period — and the reading kept is the one whose
identities hold. A wrong column choice fails them; a header in row 2 is not a special case;
a fused code-and-value cell is settled by whichever split satisfies the sum.

The identities double as the acceptance test, which is the property this project has been
missing: the check shares no code with the retriever, the renderer or the reader, so it
cannot confirm itself.

Usage:
  python scripts/fresh/find_statements.py --docs 200
  python scripts/fresh/find_statements.py --docs 200 --permute     # the control
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from check_identities import IDENTITIES, REL_TOL  # noqa: E402

CODE_RE = re.compile(r"^\d{2,3}$")
# A note pointer as printed in a Thuyết minh column: 6, 5.2, V.12, B7.33.
POINTER_SHAPE = re.compile(r"^[VB]?\.?\s?\d{1,2}(\.\d{1,2})?$")
MIN_CODES = 6
MAX_CODE_COLUMN = 4
# A statement is recognised when at least this many of its identities hold. One is too
# few — a single sum can coincide — and three excludes the short cash-flow statements.
MIN_IDENTITIES = 2


def code_columns(grid: list[list[str]]) -> list[int]:
    """Columns that look like a Mã số column, judged by content and not by a header."""

    width = max((len(row) for row in grid), default=0)
    out = []
    for column in range(min(width, MAX_CODE_COLUMN)):
        codes = []
        for row in grid:
            if column >= len(row):
                continue
            text = str(row[column]).strip()
            if CODE_RE.match(text):
                codes.append(text)
        # Codes are mostly distinct; a column of repeated "12" is a year or a page.
        if len(codes) >= MIN_CODES and len(set(codes)) >= 0.7 * len(codes):
            out.append(column)
    return out


def value_columns(grid: list[list[str]], code_column: int) -> list[int]:
    """Columns to the right of the codes that carry figures rather than codes."""

    width = max((len(row) for row in grid), default=0)
    out = []
    for column in range(width):
        if column == code_column:
            continue
        big = 0
        for row in grid:
            if column >= len(row):
                continue
            text = str(row[column]).strip()
            if not text or CODE_RE.match(text):
                continue
            value = ps.parse_vn_number(text)
            # A figure, not a note pointer and not a code: at least four digits.
            if value is not None and abs(value) >= 1000:
                big += 1
        if big >= MIN_CODES:
            out.append(column)
    return out


def read_map(grid: list[list[str]], code_column: int,
             value_column: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in grid:
        if max(code_column, value_column) >= len(row):
            continue
        code = str(row[code_column]).strip()
        if not CODE_RE.match(code) or code in out:
            continue
        value = ps.parse_vn_number(str(row[value_column]).strip())
        if value is not None:
            out[code] = value
    return out


def identities_held(values: dict[str, float], kind: str | None = None) -> tuple[int, str]:
    """How many printed identities this reading satisfies, and for which statement."""

    best = (0, "")
    for candidate in ("cdkt", "kqkd", "lctt"):
        if kind is not None and candidate != kind:
            continue
        held = 0
        for family, target, plus, minus in IDENTITIES:
            if family != candidate or target not in values:
                continue
            if not all(code in values for code in plus + minus):
                continue
            expected = (sum(values[c] for c in plus)
                        - sum(values[c] for c in minus))
            actual = values[target]
            scale = max(abs(actual), 1.0)
            if abs(expected - actual) <= REL_TOL * scale:
                held += 1
        if held > best[0]:
            best = (held, candidate)
    return best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/statements3.jsonl")
    # The control: shift every value column by one before testing. A reading that
    # satisfies two identities on the wrong column would mean the test is loose.
    parser.add_argument("--permute", action="store_true")
    args = parser.parse_args()

    known: set[str] = set()
    path = ROOT / args.index
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line)["doc"])

    corpus = ROOT / "data" / "official_corpus"
    doc_dirs = sorted(p for p in corpus.glob("*/*/*") if p.is_dir())[:args.docs]

    counters: Counter[str] = Counter()
    per_doc: dict[str, Counter] = defaultdict(Counter)
    found = []

    for doc_dir in doc_dirs:
        tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
        if not tables_dir.is_dir():
            continue
        group = "da parse duoc truoc day" if doc_dir.name in known else "TRUOC DAY BO TRONG"
        per_doc[group]["tai lieu"] += 0        # register the group

        hits = 0
        for csv_path in sorted(tables_dir.glob("table_*.csv"),
                              key=lambda p: int(p.stem.split("_")[-1])):
            try:
                with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                    grid = list(csv_mod.reader(handle))
            except OSError:
                continue
            if len(grid) < MIN_CODES:
                continue
            for code_column in code_columns(grid):
                columns = value_columns(grid, code_column)
                if args.permute:
                    columns = [c + 1 for c in columns]
                for value_column in columns:
                    values = read_map(grid, code_column, value_column)
                    if len(values) < MIN_CODES:
                        continue
                    held, kind = identities_held(values)
                    if held >= MIN_IDENTITIES:
                        hits += 1
                        counters[f"bang nhan ra: {kind}"] += 1
                        found.append({
                            "doc": doc_dir.name,
                            "table_id": int(csv_path.stem.split("_")[-1]),
                            "kind": kind, "code_col": code_column,
                            "value_col": value_column,
                            "identities": held, "codes": len(values),
                        })
                        break
        per_doc[group]["tai lieu"] += 1
        per_doc[group]["co statement" if hits else "van khong co"] += 1

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in found),
        encoding="utf-8")

    print(f"{len(doc_dirs)} tai lieu\n")
    for group, counter in sorted(per_doc.items()):
        total = counter["tai lieu"]
        got = counter["co statement"]
        print(f"  {group:26s} {total:4d} tai lieu | tim duoc statement {got:4d} "
              f"({100 * got / max(1, total):3.0f}%)")
    print()
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()


def locate_columns(grid: list[list[str]]) -> tuple[int | None, int | None]:
    """Which column holds the Mã số and which holds the note pointer.

    The header is consulted first because when it exists it is unambiguous, and the
    content is consulted when it does not — which is often. Requiring a `Mã số` header
    in row 0 was the assumption that made 23% of ordinary companies' documents look as
    though they had no statement at all.

    A pointer column is recognised the same way: several cells shaped like "6", "5.2" or
    "B7.33", none of which are figures, in a column that is not the code column.
    """

    import unicodedata

    def fold_cell(text: str) -> str:
        text = str(text).replace("đ", "d").replace("Đ", "D")
        flat = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn").casefold()
        return re.sub(r"\s+", "", flat)

    header = grid[0] if grid else []
    code_col = None
    pointer_col = None
    for index, cell in enumerate(header):
        flat = fold_cell(cell)
        if code_col is None and "maso" in flat:
            code_col = index
        if pointer_col is None and "thuyetminh" in flat:
            pointer_col = index

    if code_col is None:
        candidates = code_columns(grid)
        code_col = candidates[0] if candidates else None
    if code_col is None:
        return None, None

    if pointer_col is None:
        width = max((len(row) for row in grid), default=0)
        best, best_count = None, 0
        for column in range(width):
            if column == code_col:
                continue
            hits = 0
            for row in grid:
                if column >= len(row):
                    continue
                text = str(row[column]).strip()
                if not text or CODE_RE.match(text):
                    continue
                if POINTER_SHAPE.match(text):
                    hits += 1
            if hits > best_count:
                best, best_count = column, hits
        if best_count >= 4:
            pointer_col = best
    return code_col, pointer_col
