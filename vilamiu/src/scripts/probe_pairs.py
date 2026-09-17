"""How many label paraphrase pairs can numeric agreement mine, without text?

Training a matcher on pairs the lexical matcher already produces teaches it to
imitate that matcher, and it would never pass the 42% ceiling that motivated the
work. The supervision has to come from somewhere text-independent.

It does: report(T, Y+1) restates year Y beside year Y+1. A row in doc(T, Y) whose
figure reappears in doc(T, Y+1) is the same account in two independently OCR'd
documents — whatever the two labels happen to say. When they say *different*
things, that difference is exactly the paraphrase a matcher needs to learn and a
token overlap cannot.

Coincidence is the obvious risk, so a pair only counts when the figure is large
and unique inside both documents. Period headings are then dropped: a closing
balance equals the next year's opening balance, so "Tại ngày 31/12/2015" links
perfectly to "Tại ngày 01/01/2016" and names no account at all.

Every pattern here is matched against the accent-stripped form. Comparing
accented literals failed silently once already — the corpus and this file
disagreed on Unicode normalisation (NFC vs NFD), so "tại" never equalled "tại"
and every period heading slipped through the filter.

Usage:  PYTHONPATH=src python scripts/probe_pairs.py [--limit-docs 400]
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.embed_match import clean_label  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# Below this the figure is too weak a fingerprint: statements are full of small
# repeated numbers (counts, note references, percentages).
MIN_AMOUNT = 1e8

PERIOD_LABEL_RE = re.compile(
    r"^(?:tai|vao|den|tu)?\s*(?:ngay|nam|quy|ky|thoi diem)\b|"
    r"^so\s+du\s+(?:dau|cuoi|tai)\b|"
    r"^(?:dau|cuoi)\s+(?:nam|ky)\b|"
    r"^[\d\s/.\-]+$"
)


def plain(text: str) -> str:
    """Lowercase, accent-free, and independent of Unicode normalisation."""

    decomposed = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def names_an_account(label: str) -> bool:
    """Whether the label reads like a line item rather than a period heading."""

    flat = plain(label)
    if len(flat) < 6 or PERIOD_LABEL_RE.search(flat):
        return False
    return len([w for w in re.split(r"[^a-z]+", flat) if len(w) > 1]) >= 2


def document_index(store: TableStore, doc: str, ids) -> dict[float, str]:
    """Every figure that occurs exactly once in the document, to its row label."""

    seen: dict[float, str] = {}
    duplicated: set[float] = set()
    for table_id in ids:
        grid = store.rows(TableKey(doc, int(table_id)))
        if len(grid) < 2:
            continue
        label_col = lookup_mod.label_column(grid)
        for column in lookup_mod.value_columns(grid, label_col):
            for row in grid[1:]:
                if column >= len(row) or label_col >= len(row):
                    continue
                value = lookup_mod._parse_cell(row[column])
                if value is None or abs(value) < MIN_AMOUNT:
                    continue
                label = str(row[label_col]).strip()
                if not label:
                    continue
                rounded = round(abs(value), 2)
                if rounded in seen:
                    duplicated.add(rounded)
                else:
                    seen[rounded] = label
    for value in duplicated:
        seen.pop(value, None)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-docs", type=int, default=300)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    frame = store.frame

    ids_by_doc: dict[str, list[int]] = collections.defaultdict(list)
    for doc, table_id in zip(frame.doc_name, frame.table_id):
        ids_by_doc[str(doc)].append(int(table_id))

    by_ticker_year: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for doc, ticker, year, scope in (
        frame[["doc_name", "ticker", "year", "scope"]].drop_duplicates().itertuples(index=False)
    ):
        by_ticker_year[(str(ticker), str(year))].append(str(doc))

    stats: collections.Counter = collections.Counter()
    repeats: dict = collections.defaultdict(set)
    surface: dict = {}
    samples: list[tuple[str, str]] = []
    cache: dict[str, dict[float, str]] = {}
    matched = 0
    done = 0

    for (ticker, year), docs in sorted(by_ticker_year.items()):
        if done >= args.limit_docs:
            break
        if not year.isdigit():
            continue
        following = by_ticker_year.get((ticker, str(int(year) + 1)), [])
        if not following:
            continue
        for doc in docs:
            for neighbour in following:
                if done >= args.limit_docs:
                    break
                done += 1
                for name in (doc, neighbour):
                    if name not in cache:
                        cache[name] = document_index(store, name, ids_by_doc.get(name, []))
                here, there = cache[doc], cache[neighbour]
                for value, label_a in here.items():
                    label_b = there.get(value)
                    if label_b is None:
                        continue
                    matched += 1
                    a, b = clean_label(label_a), clean_label(label_b)
                    if not (names_an_account(a) and names_an_account(b)):
                        stats["nhan la tieu de ky/ngay (loai)"] += 1
                        continue
                    if plain(a) == plain(b):
                        stats["nhan giong het (khong hoc duoc gi)"] += 1
                        continue
                    overlap = lookup_mod.match_row([[""], [a]], b)
                    if overlap and overlap[1] >= 0.75:
                        stats["nhan khac, F1 >= 0.75 (lexical lam duoc roi)"] += 1
                    else:
                        stats["nhan khac, F1 < 0.75 (TIN HIEU MOI)"] += 1
                        # Uniqueness inside one document is not enough: two
                        # different accounts collide often enough that a single
                        # sighting is unreliable. A pair seen through several
                        # independent company-year links is not a coincidence.
                        key = (plain(a), plain(b)) if plain(a) < plain(b) else (plain(b), plain(a))
                        repeats[key].add((ticker, year))
                        surface[key] = (a[:46], b[:46])

    print(f"{done} cap tai lieu (T, Y) - (T, Y+1)")
    print(f"{matched} cap dong khop gia tri\n")
    total = sum(stats.values()) or 1
    for key, count in stats.most_common():
        print(f"  {count:6d}  ({count / total:4.0%})  {key}")
    once = sum(1 for hits in repeats.values() if len(hits) == 1)
    repeated = sum(1 for hits in repeats.values() if len(hits) >= 2)
    print(f"\nCap tin hieu moi, gop trung lap: {len(repeats)} cap phan biet")
    print(f"  chi thay 1 lan (nghi trung ngau nhien): {once}")
    print(f"  thay o >= 2 cong ty/nam (DANG TIN):     {repeated}")

    strongest = sorted(repeats.items(), key=lambda item: -len(item[1]))[: args.show]
    print("\nCap lap lai nhieu nhat:")
    for key, hits in strongest:
        a, b = surface[key]
        print(f"  [{len(hits)}x] {a!r}\n        <-> {b!r}")


if __name__ == "__main__":
    main()
