"""What is the most that a document-level unit scale could win? Measure before building.

37,111 indexed tables carry no usable scale, because they sit in documents where no
table parsed as a primary statement and the scale was only ever taken from those. The
proposed fix is to read the unit from anywhere in the document's text and apply one
scale per document — an assumption already confirmed at 99.1% by the arithmetic ties.

But the assumption being sound says nothing about the payoff. The 511 measurable
questions are what reachability is computed over, and a question only benefits if BOTH
hold: its document has no parseable statement (so its notes are currently dark), and the
answer is not already reachable. This counts exactly that set, which is the ceiling on
the fix.

If the count is small, the fix is not worth writing however correct it is.

Usage:  python scripts/fresh/bound_scale_fix.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

UNIT_LINE_RE = re.compile(r"đơn\s*vị|đvt", re.I)
SCALE_WORDS = ((r"\btriệu\b", 1e6), (r"\btỷ\b", 1e9), (r"\bnghìn\b", 1e3))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    args = parser.parse_args()

    with_statements = set()
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            with_statements.add(json.loads(line)["doc"])

    # Documents that have indexed note tables but no statements — the dark ones.
    dark_docs: dict[str, int] = Counter()
    docs_by_report: dict[tuple, set[str]] = defaultdict(set)
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        docs_by_report[(record["ticker"], record["year"],
                        record["scope"])].add(record["doc"])
        if record["doc"] not in with_statements:
            dark_docs[record["doc"]] += 1
    print(f"{len(with_statements)} tai lieu co bao cao chinh; "
          f"{len(dark_docs)} tai lieu chi co thuyet minh (dang toi)")

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    for question in questions:
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        _name, unit = unit_of(text)
        # The same scope as reachability: one company, a year, a money unit, and an
        # oracle answer that is a number. Anything else is not in the 511.
        if len(found) != 1 or not years or not unit:
            continue
        try:
            target = abs(float(oracle.get(question["id"])))
        except (TypeError, ValueError):
            continue
        if not target:
            continue

        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        docs = set()
        for key in ((ticker, year, scope),
                    (ticker, year, "consolidated" if scope == "separate"
                     else "separate")):
            docs |= docs_by_report.get(key, set())
        if not docs:
            counters["khong co bang nao duoc index cho ma-nam do"] += 1
        elif docs & with_statements:
            counters["tai lieu CO bao cao chinh — he so da suy duoc"] += 1
        else:
            counters["tai lieu KHONG co bao cao chinh — fix nay moi giup"] += 1

    total = sum(counters.values())
    print(f"\ntren {total} cau trong pham vi do duoc:\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / max(1, total):.0f}%)")
    gain = counters["tai lieu KHONG co bao cao chinh — fix nay moi giup"]
    print(f"\nCHAN TREN cua fix he so theo tai lieu: {gain} cau "
          f"= {100 * gain / max(1, total):.0f} diem do cham")
    print("va do la chan tren tuyet doi: no gia dinh moi cau trong nhom do se cham duoc,")
    print("ma thuc te con phai khop dung dong nua.")


if __name__ == "__main__":
    main()
