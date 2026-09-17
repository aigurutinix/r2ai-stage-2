"""Mine label paraphrases from numeric agreement, then distil abbreviations.

Report(T, Y+1) restates year Y, so a figure that is unique in both documents and
appears in each links two row labels to the same account — independently of what
either label says. That is the only supervision available here that does not
come from the token matcher itself, which is what makes it worth having: pairs
harvested from the matcher would only teach a model to reproduce its 42% ceiling.

The reliable half of that yield turns out to be dominated by abbreviations and
OCR variants ("Thuế TNDN" / "Thuế thu nhập doanh nghiệp", "TÀI SẢN NGĂN HẠN" /
"TÀI SẢN NGẢN HẠN"). Those do not need a trained model. An abbreviation is
verifiable: the short token's letters must equal the initials of the span it
replaces, so the rules can be extracted and checked rather than guessed.

Usage:
  PYTHONPATH=src python scripts/mine_aliases.py [--limit-docs 0] [--min-repeats 2]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.embed_match import clean_label  # noqa: E402
from vifin.query.companies import strip_tones  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

MIN_AMOUNT = 1e8

# Period and balance headings. A closing balance equals the next year's opening
# balance, so these link perfectly and name no account at all — 'Số cuối năm' to
# 'Số đầu năm' alone appeared 30 times before this covered it.
PERIOD_LABEL_RE = re.compile(
    r"^(?:tai|vao|den|tu)?\s*(?:ngay|nam|quy|ky|thoi diem)\b|"
    r"^so\s+(?:du\s+)?(?:dau|cuoi|tai|phat sinh)\b|"
    r"^(?:dau|cuoi)\s+(?:nam|ky)\b|"
    r"\b(?:so\s+)?(?:dau|cuoi)\s+(?:nam|ky)\s*$|"
    r"^[\d\s/.\-]+$"
)


def rule_form(text: str) -> str:
    """The normalisation  uses, so mined rules land in its space.

     strips every accent;  keeps the circumflex, so "co"
    from one and "cô" from the other never matched and the whole table was inert.
    Trap #7 in the handover notes, hit for the third time.
    """

    import re as _re
    return _re.sub(r"[^\w\s]", " ", strip_tones(str(text)).casefold()).strip()


def plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def names_an_account(label: str) -> bool:
    flat = plain(label)
    if len(flat) < 6 or PERIOD_LABEL_RE.search(flat):
        return False
    return len([w for w in re.split(r"[^a-z]+", flat) if len(w) > 1]) >= 2


def document_index(store: TableStore, doc: str, ids) -> dict[float, str]:
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


def abbreviation_rules(short: str, long: str) -> list[tuple[str, str]]:
    """Short tokens in one label whose letters are the initials of a span in the other.

    "Thuế TNDN" against "Thuế thu nhập doanh nghiệp" yields TNDN -> thu nhập
    doanh nghiệp, because T-N-D-N are exactly the initials of that span. The
    initials test is what makes this safe to extract automatically: a pair that
    is merely similar produces no rule at all.
    """

    short_tokens = [t for t in re.split(r"[^A-Za-zÀ-ỹ]+", short) if t]
    long_tokens = [t for t in re.split(r"[^A-Za-zÀ-ỹ]+", long) if t]
    if not short_tokens or not long_tokens:
        return []

    rules: list[tuple[str, str]] = []
    for token in short_tokens:
        letters = plain(token)
        # An abbreviation is short, and written in capitals in the source.
        if not (2 <= len(letters) <= 6) or not token.isupper():
            continue
        for start in range(len(long_tokens) - len(letters) + 1):
            span = long_tokens[start:start + len(letters)]
            initials = "".join(plain(w)[:1] for w in span)
            if initials == letters:
                rules.append((rule_form(token), " ".join(rule_form(w) for w in span)))
                break
    return rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--show", type=int, default=25)
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

    repeats: dict[tuple[str, str], set] = collections.defaultdict(set)
    surface: dict[tuple[str, str], tuple[str, str]] = {}
    stats: collections.Counter = collections.Counter()
    cache: dict[str, dict[float, str]] = {}
    done = 0
    started = time.time()

    for (ticker, year), docs in sorted(by_ticker_year.items()):
        if args.limit_docs and done >= args.limit_docs:
            break
        if not year.isdigit():
            continue
        following = by_ticker_year.get((ticker, str(int(year) + 1)), [])
        if not following:
            continue
        for doc in docs:
            for neighbour in following:
                if args.limit_docs and done >= args.limit_docs:
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
                    a, b = clean_label(label_a), clean_label(label_b)
                    if not (names_an_account(a) and names_an_account(b)):
                        stats["tieu de ky/ngay"] += 1
                        continue
                    if plain(a) == plain(b):
                        stats["nhan giong het"] += 1
                        continue
                    overlap = lookup_mod.match_row([[""], [a]], b)
                    if overlap and overlap[1] >= 0.75:
                        stats["F1 >= 0.75 (lexical lam duoc)"] += 1
                        continue
                    stats["tin hieu moi"] += 1
                    key = (plain(a), plain(b)) if plain(a) < plain(b) else (plain(b), plain(a))
                    repeats[key].add((ticker, year))
                    surface.setdefault(key, (a, b))
        if done % 400 == 0 and done:
            print(f"  {done} cap tai lieu, {len(repeats)} cap nhan, {time.time() - started:.0f}s")

    reliable = {k: v for k, v in repeats.items() if len(v) >= args.min_repeats}
    print(f"\n{done} cap tai lieu, {time.time() - started:.0f}s")
    for key, count in stats.most_common():
        print(f"  {count:7d}  {key}")
    print(f"\ncap nhan phan biet: {len(repeats)}   dang tin (>= {args.min_repeats}x): {len(reliable)}")

    out = root / "artifacts" / "label_pairs.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for key, hits in sorted(reliable.items(), key=lambda item: -len(item[1])):
            a, b = surface[key]
            handle.write(json.dumps(
                {"a": a, "b": b, "repeats": len(hits)}, ensure_ascii=False) + "\n")
    print(f"da ghi {out}")

    votes: collections.Counter = collections.Counter()
    for key, hits in reliable.items():
        a, b = surface[key]
        for short, long in abbreviation_rules(a, b) + abbreviation_rules(b, a):
            votes[(short, long)] += len(hits)
    print(f"\nquy tac viet tat rut duoc: {len(votes)}")
    for (short, long), weight in votes.most_common(args.show):
        print(f"  [{weight:3d}]  {short}  ->  {long}")

    rules_path = root / "artifacts" / "abbreviations.json"
    # Highest-weight expansion wins. Built naively from  the last
    # entry overwrote the first, so "tndn" kept a weight-2 OCR merge over the
    # weight-31 correct expansion.
    best: dict = {}
    for (short, long), weight in votes.most_common():
        if short not in best or weight > best[short][1]:
            best[short] = (long, weight)
    rules_path.write_text(json.dumps(
        {short: long for short, (long, _) in best.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"da ghi {rules_path}")


if __name__ == "__main__":
    main()
