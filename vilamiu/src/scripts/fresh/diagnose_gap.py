"""For the questions still unreachable, which of my own filters dropped the answer?

Reachability has gone 21% -> 36% -> 43% against a document ceiling of 66%, so 23 points
remain. Three of my own index decisions could be holding them, and they need different
fixes, so this measures which before anything is built:

  no scale for the document   448 documents declare no unit anywhere, in text or header,
                              and have no parseable statement to infer from — 17,335
                              tables stay unusable
  table dropped               17,337 tables were skipped at index time for having no row
                              that carries both a label and a figure
  column dropped              only the first four figures of a row are kept, so a table
                              with many periods or currencies loses the rest

For each still-unreachable question, the target value is searched in the document's RAW
csv files. Where it is present, the filter that lost it is identified. Where it is
absent, the oracle is simply wrong there and nothing in the index can be blamed — which
is expected, since the oracle is only 39% correct.

Usage:  python scripts/fresh/diagnose_gap.py --limit 0
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

KEPT_COLUMNS = 4


def close(a: float, b: float, tol: float = 2e-4) -> bool:
    if not a or not b:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    doc_modal: dict[str, Counter] = defaultdict(Counter)
    statement_values: dict[tuple, list[float]] = defaultdict(list)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        doc_modal[record["doc"]][record["scale"]] += 1
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        for period in ("current", "prior"):
            for cell in record[period].values():
                statement_values[(record["ticker"], record["year"], scope)].append(
                    abs(cell[0]))
    modal = {doc: c.most_common(1)[0][0] for doc, c in doc_modal.items()}
    scanned = json.loads((ROOT / args.doc_scale).read_text(encoding="utf-8"))
    scanned = {doc: float(scale) for doc, scale in scanned.items()}

    # The indexed note values, exactly as reach2 builds them.
    indexed: dict[tuple, list[float]] = defaultdict(list)
    docs_of: dict[tuple, set[str]] = defaultdict(set)
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["ticker"], record["year"], record["scope"])
        docs_of[key].add(record["doc"])
        scale = record["scale"] or modal.get(record["doc"]) or scanned.get(
            record["doc"])
        if scale is None:
            continue
        for row in record["rows"]:
            for _column, value in row["cols"]:
                indexed[key].append(abs(value) * scale)

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    counters: Counter[str] = Counter()
    started = time.time()
    for number, question in enumerate(questions, start=1):
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        _name, unit = unit_of(text)
        if len(found) != 1 or not years or not unit:
            continue
        try:
            target = abs(float(oracle.get(question["id"]))) * unit
        except (TypeError, ValueError):
            continue
        if not target:
            continue

        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        keys = ((ticker, year, scope),
                (ticker, year, "consolidated" if scope == "separate" else "separate"))
        if any(close(v, target) for key in keys
               for v in statement_values.get(key, ())):
            counters["da cham — bao cao chinh"] += 1
            continue
        if any(close(v, target) for key in keys for v in indexed.get(key, ())):
            counters["da cham — thuyet minh"] += 1
            continue

        # Not reachable. Look in the raw csvs and find which filter lost it.
        verdict = "oracle sai — gia tri khong co trong csv nao"
        for key in keys:
            for doc in docs_of.get(key, set()):
                base = (ROOT / "data" / "official_corpus" / key[0] / key[1] / doc)
                table_dir = base / f"{doc}_extracted_tables"
                if not table_dir.is_dir():
                    continue
                scale = modal.get(doc) or scanned.get(doc)
                for csv_path in table_dir.glob("table_*.csv"):
                    try:
                        with csv_path.open(encoding="utf-8-sig", newline="") as file:
                            rows = [row for row in csv_mod.reader(file)]
                    except OSError:
                        continue
                    for row in rows[1:]:
                        figures = []
                        for position, cell in enumerate(row):
                            raw = str(cell).strip()
                            if not raw or ps.BARE_INT_RE.match(raw):
                                continue
                            value = ps.parse_vn_number(raw)
                            if value is not None:
                                figures.append((position, abs(value)))
                        if not figures:
                            continue
                        label = max((str(c).strip() for c in row
                                     if not any(ch.isdigit() for ch in str(c))),
                                    key=len, default="")
                        for order, (_position, value) in enumerate(figures):
                            for candidate in ((scale,) if scale
                                              else (1.0, 1e3, 1e6, 1e9)):
                                if not close(value * candidate, target):
                                    continue
                                if scale is None:
                                    verdict = "MAT vi tai lieu khong co he so"
                                elif not label:
                                    verdict = "MAT vi dong khong co nhan"
                                elif order >= KEPT_COLUMNS:
                                    verdict = "MAT vi o nam ngoai 4 cot dau"
                                else:
                                    verdict = "MAT vi ly do khac (bang bi loai)"
                                break
                            if verdict.startswith("MAT"):
                                break
                        if verdict.startswith("MAT"):
                            break
                    if verdict.startswith("MAT"):
                        break
                if verdict.startswith("MAT"):
                    break
            if verdict.startswith("MAT"):
                break
        counters[verdict] += 1
        if number % 200 == 0:
            print(f"  {number}/{len(questions)}  {time.time() - started:.0f}s",
                  flush=True)

    total = sum(counters.values())
    print(f"\n{total} cau trong pham vi do duoc\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / max(1, total):.0f}%)")
    lost = sum(v for k, v in counters.items() if k.startswith("MAT"))
    print(f"\nMAT boi bo loc cua chinh toi: {lost} cau "
          f"= {100 * lost / max(1, total):.0f} diem do cham co the lay lai")


if __name__ == "__main__":
    main()
