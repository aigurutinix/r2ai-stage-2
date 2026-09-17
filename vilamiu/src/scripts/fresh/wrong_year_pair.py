"""How often did the incumbent difference the wrong pair of years?

Found by reading, not by inference. Question 628 asks for the change between end-2018 and
end-2017; the incumbent answered 32.96 tỷ, and 59.991.150.644 − 27.028.312.486 =
32.962.838.158 is exactly the two columns of the SAME 2017 report — end-2017 against
end-2016. A statement's two columns are the current and prior period of one report, so
using them answers "year N against year N−1" and nothing else.

That is a mechanical fingerprint rather than a statistical hint: an answer that matches a
specific wrong computation to the cent did that computation. So the class can be counted
without knowing any correct answer.

For each question naming two years Y1 < Y2, the incumbent's answer is compared against
|a − b| for every row of the report, where a and b are two period columns of one table.
A match is only evidence of an error when the question does NOT ask for the report year
against the year before it — when it does, the two columns are the right cells.

Usage:
  python scripts/fresh/wrong_year_pair.py --limit 40
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submissions/aimed.zip")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="artifacts/fresh/wrong_year_ids.json")
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.submission) as archive:
        incumbent = {r["id"]: r.get("answer")
                     for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    flagged, samples = [], []
    corpus = ROOT / "data" / "official_corpus"
    considered = 0

    for question in questions:
        text = question["question"]
        years = sorted({int(y) for y in YEAR_RE.findall(text)})
        if len(years) < 2:
            continue
        # Only difference-shaped questions: a ratio or an argmax over years does not
        # reduce to one subtraction, so a match there would not mean what this counts.
        if not re.search(r"trừ đi|chênh lệch|tăng|giảm|thay đổi|so với", text, re.I):
            counters["nhieu nam nhung khong phai phep tru"] += 1
            continue
        _name, unit = unit_of(text)
        if not unit:
            counters["khong ro don vi"] += 1
            continue
        try:
            answer = abs(float(incumbent.get(question["id"]))) * unit
        except (TypeError, ValueError):
            continue
        if not answer:
            continue
        found = sorted(resolver.resolve(text))
        if not found:
            continue
        considered += 1
        if args.limit and considered > args.limit:
            break

        ticker = found[0]
        high, low = years[-1], years[-2]
        # The two columns of the report for `high` answer "high against high-1".
        legitimate = (low == high - 1)
        want_separate = bool(PARENT_RE.search(text))

        matched_year = None
        for candidate in (high, low):
            base = corpus / ticker / str(candidate)
            if not base.is_dir():
                continue
            docs = sorted(p for p in base.iterdir() if p.is_dir())
            ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                       + [d for d in docs if ("separate" in d.name) != want_separate])
            for doc_dir in ordered[:1]:
                tables = doc_dir / f"{doc_dir.name}_extracted_tables"
                if not tables.is_dir():
                    continue
                for path in sorted(tables.glob("table_*.csv")):
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as handle:
                            grid = list(csv_mod.reader(handle))
                    except OSError:
                        continue
                    for row in grid:
                        values = []
                        for cell in row:
                            raw = str(cell).strip()
                            if not raw or (ps.BARE_INT_RE.match(raw)
                                           and len(raw) <= 3):
                                continue
                            value = ps.parse_vn_number(raw)
                            if value is not None:
                                values.append(value)
                        for i in range(len(values)):
                            for j in range(i + 1, len(values)):
                                if abs(abs(values[i] - values[j]) - answer) <= 0.01:
                                    matched_year = candidate
                                    break
                            if matched_year:
                                break
                        if matched_year:
                            break
                    if matched_year:
                        break
                if matched_year:
                    break
            if matched_year:
                break

        if matched_year is None:
            counters["khong khop hieu trong cung mot bao cao"] += 1
        elif legitimate:
            counters["khop, va dung vi cau hoi hoi N vs N-1"] += 1
        else:
            counters["KHOP nhung SAI cap nam — dap an cu chac sai"] += 1
            flagged.append(question["id"])
            if len(samples) < 5:
                samples.append(f"  id={question['id']} hoi {low} vs {high}, "
                               f"khop trong bao cao {matched_year}: {text[:80]}")

    (ROOT / args.out).write_text(json.dumps(sorted(flagged)), encoding="utf-8")
    print(f"da xet {considered} cau hoi hieu nhieu nam\n")
    for name, count in counters.most_common():
        print(f"  {count:4d}  {name}")
    print(f"\nso cau dap an cu chac sai cap nam: {len(flagged)}")
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
