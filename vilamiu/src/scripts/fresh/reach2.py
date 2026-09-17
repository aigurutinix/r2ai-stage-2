"""Reachability against the widened index. Does 21% move?

The old index covered a report's three statements plus the notes whose column total tied
to a statement line in both periods — about 26 of 75 tables. The oracle's answer sat
inside it for 21% of measurable questions while sitting somewhere in the resolved
document for 66%, so 45 points were in the report and outside the address space.

The widened index keeps every table with a labelled row and a figure: 111,761 note
tables against 22,365 tied ones. Two thirds of them declare no unit, because a
Vietnamese report states it once at the head of the notes and the rest inherit it — so
the scale is inferred from the document's modal statement scale, which the arithmetic
ties confirm at 99.1% and the notes' own declarations at 91.9%.

This is the number that decides whether indexing was the right thing to attack. It
compares the same oracle against the same questions, so it is directly comparable to
the 21%.

Usage:  python scripts/fresh/reach2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def close(a: float, b: float, tol: float = 2e-4) -> bool:
    if not a or not b:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    # One scale per document, read from its text and table headers. 807 documents have
    # no parseable statement to infer from — banks and securities firms, whose charts
    # `parse_statement` rejects — and this reaches 424 of them. Validated against the
    # documents whose scale is known: 98.1% on the set the earlier scanner also covered
    # (it scored 97.7% there) and 95.4% on the 350 it could not score at all.
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    args = parser.parse_args()

    # Statement cells, and the modal scale per document for inferring note scales.
    values: dict[tuple, list[float]] = defaultdict(list)
    doc_scales: dict[str, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        doc_scales[record["doc"]][record["scale"]] += 1
        for period in ("current", "prior"):
            for cell in record[period].values():
                values[(record["ticker"], record["year"], scope)].append(
                    abs(cell[0]))
    modal = {doc: counter.most_common(1)[0][0]
             for doc, counter in doc_scales.items()}
    print(f"{len(values)} bao cao co o bao cao chinh", flush=True)

    scale_path = ROOT / args.doc_scale
    scanned: dict[str, float] = {}
    if scale_path.exists():
        scanned = {doc: float(scale) for doc, scale
                   in json.loads(scale_path.read_text(encoding="utf-8")).items()}
    print(f"{len(scanned)} tai lieu co he so quet tu van ban", flush=True)

    # Note cells, scaled — declared where stated, inferred from the document otherwise.
    note_values: dict[tuple, list[float]] = defaultdict(list)
    inferred = declared = dark = 0
    started = time.time()
    for count, line in enumerate(
            (ROOT / args.tables).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        scale = record["scale"]
        if scale is None:
            # The document's own statements first — that inference is confirmed at
            # 99.1% by the arithmetic ties — then the document-wide text scan.
            scale = modal.get(record["doc"]) or scanned.get(record["doc"])
            if scale is None:
                dark += 1
                continue
            inferred += 1
        else:
            declared += 1
        key = (record["ticker"], record["year"], record["scope"])
        for row in record["rows"]:
            for _column, value in row["cols"]:
                note_values[key].append(abs(value) * scale)
        if count % 30000 == 0:
            print(f"  nap {count} bang  {time.time() - started:.0f}s", flush=True)
    print(f"he so: khai san {declared}, suy tu tai lieu {inferred}, "
          f"khong suy duoc {dark}")

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
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        keys = ((ticker, year, scope),
                (ticker, year, "consolidated" if scope == "separate" else "separate"))

        where = None
        for key in keys:
            if any(close(v, target) for v in values.get(key, ())):
                where = "o BAO CAO CHINH"
                break
        if where is None:
            for key in keys:
                if any(close(v, target) for v in note_values.get(key, ())):
                    where = "o THUYET MINH (chi so rong)"
                    break
        counters[where or "KHONG O DAU"] += 1

    total = sum(counters.values())
    print(f"\n{total} cau\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / total:.0f}%)")
    reachable = sum(v for k, v in counters.items() if k.startswith("o "))
    scoped = reachable + counters["KHONG O DAU"]
    if scoped:
        print(f"\ntrong {scoped} cau do duoc: CHAM DUOC "
              f"{100 * reachable / scoped:.0f}%  (chi so cu: 21%, tai lieu: 66%)")


if __name__ == "__main__":
    main()
