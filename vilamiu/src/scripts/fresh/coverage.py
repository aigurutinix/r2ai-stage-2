"""How much of the exam can the Mã số address book reach at all?

The index says where the organisers' own parser could see a figure. A question about
one of the three primary statements must resolve to a (document, kind, ma_so,
current|prior) address inside it, so before writing any answering logic the useful
number is what fraction of the 1,012 questions even names a company-year-scope whose
statements parsed.

Scope comes from the generator's own rule, which is a hard contract rather than a
guess: the easy prompt says consolidated is the default and must never be mentioned,
and that `công ty mẹ` must be stated whenever the scope is the parent company. So the
absence of that phrase means consolidated.

Also reports the reliable slice of the dictionary. Inside a `cdkt` table only the
three-digit codes are real: the parser scans the first three columns for a one-to-
three digit integer, so an ordinal in a note row is picked up as a code, which is
why `cdkt` code 10 collects labels like "Quỹ khen thưởng, phúc lợi". In `kqkd` and
`lctt` the two-digit codes are the statutory ones.

Usage:  python scripts/fresh/coverage.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PARENT_RE = re.compile(r"công ty mẹ|c[ôo]ng ty me", re.I)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tickers: dict[str, str] = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')

    available: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    codes_seen: dict[str, Counter[str]] = defaultdict(Counter)
    for line in (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = "separate" if "separate" in record["scope"] else (
            "consolidated" if "consolidated" in record["scope"] else record["scope"])
        available[(record["ticker"], record["year"], scope)].add(record["kind"])
        for code in record["current"]:
            codes_seen[record["kind"]][code] += 1

    print(f"{len(available)} (ma, nam, pham vi) co bao cao chinh doc duoc")
    kinds = Counter()
    for keys in available.values():
        kinds[" + ".join(sorted(keys))] += 1
    for name, count in kinds.most_common(8):
        print(f"  {name}: {count}")

    print("\nma so DANG TIN CAY:")
    three = [c for c in codes_seen["cdkt"] if len(c) == 3]
    print(f"  cdkt, ma 3 chu so: {len(three)} (tren tong {len(codes_seen['cdkt'])})")
    for kind in ("kqkd", "lctt"):
        two = [c for c in codes_seen[kind] if len(c) == 2]
        print(f"  {kind}, ma 2 chu so: {len(two)} (tren tong {len(codes_seen[kind])})")

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    # Longest company name first, so "CTCP Tập đoàn Hòa Phát" is not matched by a
    # shorter name that happens to be a prefix of it.
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))
    counters: Counter[str] = Counter()
    reachable = []
    for question in questions:
        text = question["question"]
        found = {code for code in tickers if re.search(rf"\b{re.escape(code)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = {y for y in YEAR_RE.findall(text)}
        scope = "separate" if PARENT_RE.search(text) else "consolidated"

        if not found:
            counters["khong nhan ra ma nao"] += 1
            continue
        if not years:
            counters["khong nhan ra nam nao"] += 1
            continue
        if len(found) > 1:
            counters["nhieu ma (cau nhom)"] += 1
            continue
        ticker = next(iter(found))
        hits = {y for y in years if (ticker, y, scope) in available}
        if hits:
            counters["co bao cao chinh doc duoc"] += 1
            reachable.append(question["id"])
        elif any((ticker, y, "consolidated") in available or
                 (ticker, y, "separate") in available for y in years):
            counters["co bao cao nhung SAI pham vi"] += 1
        else:
            counters["khong co bao cao chinh nao cho ma-nam do"] += 1

    print(f"\ntren {len(questions)} cau hoi:")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    (ROOT / "artifacts" / "fresh" / "reachable_ids.json").write_text(
        json.dumps(sorted(reachable)), encoding="utf-8")


if __name__ == "__main__":
    main()
