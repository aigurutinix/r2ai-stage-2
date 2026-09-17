"""Is the answer anywhere in the right DOCUMENT — not just in the indexed part of it?

Reachability inside the index came out at 21% against an oracle that is 39% correct, so
roughly half the answers are not in the address space at all. Two explanations fit, and
they call for opposite work:

  the index is too narrow   the answer is in the report, in a table the index skipped —
                            a note that tied on only one period, a note in a report
                            whose statements never parsed, a table of a bank. Then the
                            work is indexing, and there is a lot to win.
  the document is wrong     the answer is not in that report at all, because the company
                            or the year resolved wrongly, or the figure lives in another
                            report. Then indexing wins nothing.

This searches EVERY csv of the resolved document, at every plausible unit scale, and
reports which of the two it is.

The oracle is 39% correct, so this understates: where it is wrong, no cell matches.
Read it as a lower bound and compare it against the 21% the index reached.

Usage:  python scripts/fresh/reach_document.py --limit 400
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9)


def close(a: float, b: float, tol: float = 2e-4) -> bool:
    if not a or not b:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b))


def numbers_in(doc_dir: Path) -> list[float]:
    out: list[float] = []
    table_dir = next(doc_dir.glob("*_extracted_tables"), None)
    if table_dir is None or not table_dir.is_dir():
        return out
    for path in table_dir.glob("table_*.csv"):
        try:
            with path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.reader(file):
                    for cell in row:
                        raw = str(cell).strip()
                        if not raw or ps.BARE_INT_RE.match(raw):
                            continue
                        value = ps.parse_vn_number(raw)
                        if value is not None:
                            out.append(abs(value))
        except OSError:
            continue
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    corpus = ROOT / "data" / "official_corpus"
    cache: dict[Path, list[float]] = {}
    counters: Counter[str] = Counter()
    started = time.time()

    for index, question in enumerate(questions, start=1):
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        _name, unit = unit_of(text)
        if len(found) != 1 or not years or not unit:
            counters["ngoai pham vi"] += 1
            continue
        try:
            target = abs(float(oracle.get(question["id"]))) * unit
        except (TypeError, ValueError):
            counters["oracle khong phai so"] += 1
            continue
        if not target:
            counters["oracle tra 0"] += 1
            continue

        ticker = next(iter(found))
        year = max(years)
        want_separate = bool(PARENT_RE.search(text))
        base = corpus / ticker / year
        if not base.is_dir():
            counters["khong co thu muc ma-nam"] += 1
            continue

        # The scope the question implies first, then the other one.
        docs = sorted(p for p in base.iterdir() if p.is_dir())
        ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                   + [d for d in docs if ("separate" in d.name) != want_separate])
        hit = None
        for doc_dir in ordered:
            if doc_dir not in cache:
                cache[doc_dir] = numbers_in(doc_dir)
            values = cache[doc_dir]
            if not values:
                continue
            for scale in SCALES:
                if any(close(value * scale, target) for value in values):
                    hit = ("dung pham vi" if ("separate" in doc_dir.name)
                           == want_separate else "pham vi KHAC")
                    break
            if hit:
                break
        counters[f"CO trong tai lieu — {hit}" if hit
                 else "KHONG co trong tai lieu nao cua ma-nam do"] += 1
        if index % 200 == 0:
            print(f"  {index}/{len(questions)}  {time.time() - started:.0f}s",
                  flush=True)

    total = sum(counters.values())
    print(f"\n{total} cau\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / total:.0f}%)")
    reachable = sum(v for k, v in counters.items() if k.startswith("CO trong"))
    scoped = reachable + counters["KHONG co trong tai lieu nao cua ma-nam do"]
    if scoped:
        print(f"\ntrong {scoped} cau do duoc: dap an co trong TAI LIEU "
              f"{100 * reachable / scoped:.0f}%")
        print("so voi 21% chi so hien tai cham duoc — hieu so la thu indexing co the win.")


if __name__ == "__main__":
    main()
