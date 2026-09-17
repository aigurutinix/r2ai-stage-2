"""Which answers in a submission correspond to no cell in the document at all.

The best zip answers 1007 of 1012 questions, so there is no room left for coverage — a
new reader can only gain by REPLACING answers, and replacing is what lost 0.32 to 0.40
points per row on three earlier occasions. Those attempts replaced blindly. This picks
the rows where the incumbent answer is least defensible.

The test is simple and gold-free: a money question asks for one figure that is printed
in the report. Take the incumbent answer, multiply by the unit the question names, and
look for a cell equal to it in any table of that document — at any of the four scales a
Vietnamese report is denominated in, because the incumbent may have converted. If no
cell anywhere in the document holds that figure, the incumbent did not read it off the
report: it was derived, mis-scaled, or invented.

That is not proof the incumbent is wrong. A legitimately derived answer — a sum, a
difference — matches no single cell either, which is why only money questions are
scored, and why the output is a ranking of doubt rather than a verdict.

Usage:
  python scripts/fresh/locatable.py --submission submissions/aimed.zip
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
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

# A question whose answer is a computed quantity is not expected to sit in a cell.
DERIVED_RE = re.compile(
    r"tỷ lệ|tỉ lệ|phần trăm|\bROA\b|\bROE\b|biên|tăng trưởng|chênh lệch|"
    r"bao nhiêu lần|trung bình|bình quân|tổng cộng của|so với", re.I)
SCALES = (1.0, 1e3, 1e6, 1e9)
TOLERANCE = 0.005  # the scorer's own 0.01 absolute, halved for the round trip


def cells_of(doc_dir: Path) -> set[int]:
    """Every cell of the document at every plausible scale, as cents.

    Membership, not a scan: the caller asks about one figure and a large report holds
    tens of thousands of cells, so the four scaled forms are precomputed once per
    document and looked up. The key is the value in cents because the scorer's
    tolerance is an absolute 0.01.
    """

    values: set[int] = set()
    for path in sorted(doc_dir.glob("table_*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv_mod.reader(handle):
                    for cell in row:
                        raw = str(cell).strip()
                        # Bare integers are skipped only up to three digits, which is
                        # the width of a Mã số. Skipping every separator-less integer
                        # discarded figures whose thousands dots the OCR dropped, and
                        # put 86 rows on the replacement list where the incumbent was
                        # in fact reading a real cell — a 39% false-positive rate.
                        if not raw or (ps.BARE_INT_RE.match(raw) and len(raw) <= 3):
                            continue
                        value = ps.parse_vn_number(raw)
                        if value is None:
                            continue
                        for scale in SCALES:
                            values.add(round(abs(value) * scale * 100))
        except OSError:
            continue
    return values


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submissions/aimed.zip")
    parser.add_argument("--out", default="artifacts/fresh/not_a_cell.json")
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.submission) as archive:
        incumbent = {r["id"]: r.get("answer")
                     for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    corpus = ROOT / "data" / "official_corpus"
    # One document's cells at a time. Holding every document's set at once is tens of
    # gigabytes — four scaled copies of a hundred thousand cells per report — and the
    # first attempt at this was killed for it. Questions are visited in document order
    # so each set is built once and dropped when the document changes.
    cache: dict[Path, set[int]] = {}
    CACHE_LIMIT = 3
    counters: Counter[str] = Counter()
    doubtful, solid = [], []
    started = time.time()

    def document_key(question: dict) -> tuple[str, str]:
        found = sorted(resolver.resolve(question["question"]))
        years = YEAR_RE.findall(question["question"])
        return (found[0] if found else "", max(years) if years else "")

    for number, question in enumerate(sorted(questions, key=document_key), start=1):
        text = question["question"]
        _name, unit = unit_of(text)
        if not unit or DERIVED_RE.search(text):
            counters["khong phai cau hoi mot o tien"] += 1
            continue
        try:
            target = abs(float(incumbent.get(question["id"]))) * unit
        except (TypeError, ValueError):
            counters["dap an cu khong phai so"] += 1
            continue
        if not target:
            counters["dap an cu bang 0"] += 1
            continue
        found = sorted(resolver.resolve(text))
        years = YEAR_RE.findall(text)
        if not found or not years:
            counters["khong xac dinh duoc tai lieu"] += 1
            continue
        ticker, year = found[0], max(years)
        want_separate = bool(PARENT_RE.search(text))
        # The next year's report too. "Cuối năm N" is printed twice — as the current
        # column of the report for N and as the prior column of the report for N+1 —
        # so a figure absent from one is not absent from the document set, and calling
        # the incumbent wrong on that basis would put correct answers on the
        # replacement list.
        ordered = []
        for candidate in (year, str(int(year) + 1)):
            base = corpus / ticker / candidate
            if not base.is_dir():
                continue
            docs = sorted(p for p in base.iterdir() if p.is_dir())
            ordered += [d for d in docs if ("separate" in d.name) == want_separate]
            ordered += [d for d in docs if ("separate" in d.name) != want_separate]
        if not ordered:
            counters["khong xac dinh duoc tai lieu"] += 1
            continue

        hit = False
        key = round(target * 100)
        for doc_dir in ordered:
            tables = doc_dir / f"{doc_dir.name}_extracted_tables"
            if not tables.is_dir():
                continue
            if tables not in cache:
                if len(cache) >= CACHE_LIMIT:
                    cache.clear()
                cache[tables] = cells_of(tables)
            found_cells = cache[tables]
            if any(k in found_cells for k in (key - 1, key, key + 1)):
                hit = True
                break
        if hit:
            counters["dap an cu LA mot o"] += 1
            solid.append(question["id"])
        else:
            counters["dap an cu KHONG phai o nao — cho phep thay"] += 1
            doubtful.append(question["id"])
        if number % 100 == 0:
            print(f"  {number}/{len(questions)}  {time.time() - started:.0f}s",
                  flush=True)

    (ROOT / args.out).write_text(json.dumps(
        {"submission": args.submission, "doubtful": doubtful, "solid": solid},
        ensure_ascii=False), encoding="utf-8")
    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\nngan sach thay the: {len(doubtful)} cau")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
