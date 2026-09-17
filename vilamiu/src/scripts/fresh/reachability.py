"""Is the answer anywhere in the indexed address space at all?

The shortlist measurement came back with a 3-point lift from top-1 to top-5, which means
the ranking already puts the right line first about three times in four WHEN the right
line is on the list. So choosing better among candidates cannot be the problem. The
problem is that the right line is usually not a candidate.

That turns the important question into a ceiling: for how many questions does a cell
holding the answer exist anywhere in the index — any statement line, any tied note row,
either period, that company and year? No matching, no ranking, no labels. Just: is it
reachable.

Whatever that number is, it caps the fresh pipeline. If it is 40%, no amount of better
line choice gets past 0.40. If it is 85%, then choosing is the whole remaining problem
and it is worth attacking again.

The oracle is the best shipped submission at 39% correct, so this UNDERSTATES
reachability: where the oracle is wrong, a matching cell will not be found even if the
right cell is present. Read the result as a lower bound.

Usage:  python scripts/fresh/reachability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def close(a: float, b: float, tol: float = 2e-4) -> bool:
    if not a or not b:
        return False
    scale = max(abs(a), abs(b))
    return abs(a - b) <= tol * scale


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    values: dict[tuple, list[float]] = defaultdict(list)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        for period in ("current", "prior"):
            for cell in record[period].values():
                values[(record["ticker"], record["year"], scope)].append(cell[0])
    print(f"{len(values)} bao cao trong so dia chi", flush=True)

    notes_by: dict[tuple, list[dict]] = defaultdict(list)
    for line in (ROOT / args.notes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        note = json.loads(line)
        if note.get("paired"):
            notes_by[(note["ticker"], note["year"], note["scope"])].append(note)

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    note_cache: dict[str, list[float]] = {}
    counters: Counter[str] = Counter()
    for question in questions:
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        _name, unit = unit_of(text)
        if len(found) != 1 or not years or not unit:
            counters["ngoai pham vi (nhieu ma / khong nam / don vi khac)"] += 1
            continue
        try:
            target = abs(float(oracle.get(question["id"]))) * unit
        except (TypeError, ValueError):
            counters["oracle khong co dap an so"] += 1
            continue
        if not target:
            counters["oracle tra 0"] += 1
            continue

        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)

        where = None
        for key in ((ticker, year, scope),
                    (ticker, year, "consolidated" if scope == "separate"
                     else "separate")):
            if any(close(abs(v), target) for v in values.get(key, ())):
                where = "o BAO CAO CHINH"
                break
        if where is None:
            for key in ((ticker, year, scope),
                        (ticker, year, "consolidated" if scope == "separate"
                         else "separate")):
                for note in notes_by.get(key, ()):
                    path = note["csv"]
                    if path not in note_cache:
                        numbers = []
                        try:
                            with (ROOT / path).open(encoding="utf-8-sig",
                                                    newline="") as file:
                                for row in csv.reader(file):
                                    for cell in row:
                                        raw = str(cell).strip()
                                        if not raw or ps.BARE_INT_RE.match(raw):
                                            continue
                                        parsed = ps.parse_vn_number(raw)
                                        if parsed is not None:
                                            numbers.append(parsed)
                        except OSError:
                            numbers = []
                        note_cache[path] = numbers
                    scale = Counter(t["scale"] for t in note["ties"]).most_common(
                        1)[0][0]
                    if any(close(abs(v) * scale, target)
                           for v in note_cache[path]):
                        where = "o THUYET MINH"
                        break
                if where:
                    break
        counters[where or "KHONG O DAU trong chi so"] += 1

    total = sum(counters.values())
    print(f"\n{total} cau\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / total:.0f}%)")
    reachable = counters["o BAO CAO CHINH"] + counters["o THUYET MINH"]
    scoped = reachable + counters["KHONG O DAU trong chi so"]
    if scoped:
        print(f"\ntrong {scoped} cau do duoc: dap an CHAM DUOC "
              f"{100 * reachable / scoped:.0f}%")
    print("day la CHAN TREN cua duong lam lai — va no la chan duoi that su,")
    print("vi oracle chi dung 39% nen cho no sai thi phep do nay khong tim thay.")


if __name__ == "__main__":
    main()
