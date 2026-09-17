"""One unit scale per document, read from anywhere in its text — validated first.

37,111 indexed tables carry no scale because the scale was only ever taken from tables
that parsed as a primary statement, and 807 documents have none: banks and securities
firms use a different chart, so `parse_statement` rejects their statements and every
note in the report goes dark with them. Measured ceiling on fixing that: 197 of the 511
measurable questions, or 39 points of reachability.

The assumption — one unit for the whole document — is already established: the
arithmetic ties recover a note's true scale independently, and it equals the document's
modal statement scale for 22,103 of 22,309 notes (99.1%).

What is NOT established is that a text scan finds the right scale, so this validates
before it applies. The 1,111 documents with parseable statements already have a scale
derived from the tables themselves; the scanner is run against those and its agreement
reported. Only then is it worth applying to the 807 dark ones.

Three failure modes handled here rather than discovered later:

  narrative "đơn vị"   "đầu tư vào đơn vị khác" is not a unit declaration, so a line
                       must be SHORT and read like one
  OCR damage           "Dom vi: VND" — matched on the diacritic-folded form
  several units        take the modal, which is what the tie check validated

Usage:  python scripts/fresh/document_scale.py --limit 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A declaration is short. Anything longer is prose that happens to contain the word.
MAX_LINE = 60
MARKER_RE = re.compile(r"don\s*vi|dvt|dom\s*vi")
DONG_RE = re.compile(r"\bdong\b|\bvnd\b|\bvn[dđ]\b")
SCALE_WORDS = ((r"\btrieu\b", 1e6), (r"\bty\b", 1e9), (r"\bnghin\b", 1e3))


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s:.,()/-]", " ", flat)).strip()


# How far past the marker a declaration reaches. "Đơn vị tính: triệu đồng" is about
# twenty characters; forty leaves room for OCR noise without swallowing the sentence
# that follows.
WINDOW = 40


def scale_of_line(line: str) -> float | None:
    """The multiplier a unit declaration implies, or None if the line is not one.

    Measured on 120 documents the scanner was silent on: NO short line carried a
    marker, while 403 lines carried one and ran past sixty characters — OCR glues the
    declaration to whatever follows it on the page. So the length test moves from the
    whole line to the span just after the marker, which keeps a 471-character narrative
    that happens to contain "đơn vị" from qualifying.
    """

    flat = fold(line)
    match = MARKER_RE.search(flat)
    if match is None:
        return None
    span = flat[match.start():match.start() + WINDOW]
    if not DONG_RE.search(span):
        return None
    for pattern, scale in SCALE_WORDS:
        if re.search(pattern, span):
            return scale
    return 1.0


def document_scale(text: str,
                   headers: list[str] | None = None) -> tuple[float | None, int]:
    """The modal scale across the document's unit declarations, and how many were seen.

    `headers` are the table header rows, joined per table. The statement path always
    read those, but a document whose statements never parsed had its headers read by
    nothing — which is exactly the bank and securities reports this is meant to reach.
    """

    votes: Counter[float] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        scale = scale_of_line(stripped)
        if scale is not None:
            votes[scale] += 1
    for header in headers or ():
        scale = scale_of_line(header)
        if scale is not None:
            votes[scale] += 1
    if not votes:
        return None, 0
    return votes.most_common(1)[0][0], sum(votes.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/doc_scale.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Ground truth for validation: the modal scale of each document's statements.
    truth: dict[str, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            truth[record["doc"]][record["scale"]] += 1
    known = {doc: counter.most_common(1)[0][0] for doc, counter in truth.items()}

    # Which documents matter: those with indexed tables.
    where: dict[str, tuple[str, str]] = {}
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            where.setdefault(record["doc"], (record["ticker"], record["year"]))
    print(f"{len(where)} tai lieu co bang duoc index, "
          f"{len(known)} trong so do da biet he so", flush=True)

    docs = sorted(where)
    if args.limit:
        docs = docs[:args.limit]

    scanned: dict[str, float] = {}
    counters: Counter[str] = Counter()
    started = time.time()
    for index, doc in enumerate(docs, start=1):
        ticker, year = where[doc]
        path = (ROOT / "data" / "official_corpus" / ticker / year / doc /
                f"{doc}_extracted.txt")
        if not path.exists():
            counters["khong co file text"] += 1
            continue
        headers = []
        table_dir = path.parent / f"{doc}_extracted_tables"
        if table_dir.is_dir():
            import csv as csv_mod

            for csv_path in sorted(table_dir.glob("table_*.csv")):
                try:
                    with csv_path.open(encoding="utf-8-sig", newline="") as file:
                        first = next(csv_mod.reader(file), None)
                except OSError:
                    continue
                if first:
                    headers.append(",".join(str(c) for c in first))
        scale, votes = document_scale(
            path.read_text(encoding="utf-8", errors="replace"), headers)
        if scale is None:
            counters["khong tim thay dong khai don vi nao"] += 1
            continue
        scanned[doc] = scale
        counters["quet ra he so"] += 1
        counters["  chi 1 dong khai" if votes == 1 else "  nhieu dong khai"] += 1
        if index % 400 == 0:
            print(f"  {index}/{len(docs)}  {time.time() - started:.0f}s", flush=True)

    # Validation: on documents whose scale is already known from their statements.
    agree = disagree = 0
    kinds: Counter[str] = Counter()
    for doc, scale in scanned.items():
        if doc not in known:
            continue
        if abs(scale - known[doc]) < 1e-9:
            agree += 1
        else:
            disagree += 1
            kinds[f"quet {scale:g} vs bao cao {known[doc]:g}"] += 1

    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    total = agree + disagree
    print(f"\nKIEM tren {total} tai lieu da biet he so: "
          f"khop {100 * agree / max(1, total):.1f}%")
    for name, count in kinds.most_common(6):
        print(f"  lech — {name}: {count}")

    dark = [doc for doc in scanned if doc not in known]
    print(f"\ntai lieu TOI ma quet ra he so: {len(dark)}")
    still = counters["khong tim thay dong khai don vi nao"]
    print(f"tai lieu van khong co he so nao: {still}")

    payload = {doc: scale for doc, scale in scanned.items()}
    (ROOT / args.out).write_text(json.dumps(payload, indent=0), encoding="utf-8")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
