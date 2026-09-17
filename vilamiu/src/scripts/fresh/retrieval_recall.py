"""Is the answer's cell even among the tables the model is shown?

Five mechanisms were built on top of this retrieval without ever measuring it. Every one
of them could only pick from the nine candidate tables the retrieval offered, so if the
right table is usually absent, then the quote checks, the gates, the prompt rules and the
splices were all tuning a stage that was never the bottleneck.

The measurement needs a set of questions whose answer is known, and there is no gold. So
it uses consensus: the questions where the incumbent and at least one independent reader
produced the SAME value. Two mechanisms that fail differently do not land on the same
twelve-digit figure by chance, so those values are near-certainly right — and crucially
they were not produced by this retrieval, so asking whether the retrieval covers them is
not circular.

For each such question it asks one thing: does any cell of the offered tables equal that
value, at any of the four scales a Vietnamese report is denominated in?

  high recall   the offered tables contain the answer, so the loss is in choosing the row
                and column, and prompt work is the right work
  low recall    the answer is not there to be found, and every downstream stage was
                noise. Retrieval has to be rebuilt, not tuned.

Usage:
  python scripts/fresh/retrieval_recall.py
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)


def answers_of(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            records = json.loads(archive.read("submission.json"))
    else:
        records = [json.loads(line) for line
                   in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        value = record.get("answer")
        if value in (None, "", 0.0):
            continue
        try:
            out[record["id"]] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def cells_of(csv_path: Path, rows: int = 0, cols: int = 0) -> list[float]:
    """Every cell of one table, as an absolute number in đồng-equivalents.

    Kept as values rather than a set of cents because the comparison cannot be exact:
    the answer being checked was rounded to two decimals IN THE QUESTION'S UNIT, so
    145.731.366.146 đồng ships as 145.73 tỷ. Comparing those in cents of đồng differs
    by 1.37 million and never matches — the first version of this measurement reported
    16% recall for exactly that reason, and a table that was in fact offered first.
    """

    found: list[float] = []
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            grid = list(csv_mod.reader(handle))
            if rows:
                # Row 0 is the header the renderer prints separately, then `rows` more.
                grid = grid[:rows + 1]
            for row in grid:
                if cols:
                    row = row[:cols]
                for cell in row:
                    raw = str(cell).strip()
                    if not raw:
                        continue
                    value = ps.parse_vn_number(raw)
                    if value is None:
                        continue
                    found.append(abs(value))
    except OSError:
        pass
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    # The control. Nine tables hold on the order of two thousand cells, and a hit is
    # allowed at five scales within an absolute 0.01, so a table set can match almost
    # any figure by chance. Testing each question against ANOTHER question's answer
    # measures that chance rate. A real recall only means something to the extent it
    # exceeds the permuted one — this is the check whose absence produced a "66%
    # document reachability" figure that turned out to be coincidence.
    parser.add_argument("--permute", action="store_true")
    # Only the cells the prompt actually PRINTS. The renderer stops at 26 data rows and
    # 6 columns, so a table can contain the answer while the prompt does not show it —
    # in which case no model could pick it and the loss is mine, not the model's.
    parser.add_argument("--rendered", action="store_true")
    parser.add_argument("--rows", type=int, default=26)
    parser.add_argument("--cols", type=int, default=6)
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    incumbent = answers_of(ROOT / "submissions" / "aimed.zip")
    readers = {
        "bang": answers_of(ROOT / "submissions" / "fresh_tab.zip"),
        "tho": answers_of(ROOT / "submissions" / "fresh_raw.zip"),
        "prog": answers_of(ROOT / "artifacts" / "fresh" / "prog_results.jsonl"),
    }

    # Consensus set: the incumbent and at least one independent reader on the same value.
    consensus: dict[int, float] = {}
    for qid, value in incumbent.items():
        for reader in readers.values():
            other = reader.get(qid)
            if other is not None and abs(other - value) <= 0.01:
                consensus[qid] = value
                break

    offered = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            offered[record["id"]] = record["meta"]["refs"]

    counters: Counter[str] = Counter()
    misses = []
    cache: dict[Path, list[float]] = {}
    ordered = sorted(consensus)
    if args.limit:
        ordered = ordered[:args.limit]

    for qid in ordered:
        refs = offered.get(qid)
        if not refs:
            counters["khong co prompt cho cau nay"] += 1
            continue
        _name, unit = unit_of(questions[qid])
        if not unit:
            counters["khong ro don vi — bo qua"] += 1
            continue
        target = abs(consensus[qid])
        if args.permute:
            others = [v for other, v in consensus.items() if other != qid]
            target = abs(others[qid % len(others)])

        hit = False
        for ref in refs:
            path = (ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                    ref["doc"] / f"{ref['doc']}_extracted_tables"
                    / f"table_{ref['table_id']}.csv")
            if path not in cache:
                if len(cache) > 400:
                    cache.clear()
                cache[path] = cells_of(
                    path, args.rows if args.rendered else 0,
                    args.cols if args.rendered else 0)
            # A cell counts as the answer if converting it to the unit the question
            # asks for lands within the rounding the answer itself carries.
            for value in cache[path]:
                if any(abs(value * scale / unit - target) <= 0.01
                       for scale in SCALES):
                    hit = True
                    break
            if hit:
                break
        counters["CO trong 9 bang duoc dua" if hit
                 else "KHONG co trong bang nao duoc dua"] += 1
        if not hit and len(misses) < 6:
            misses.append(f"  id={qid} dap an={consensus[qid]} — "
                          f"{questions[qid][:88]}")

    measured = (counters["CO trong 9 bang duoc dua"]
                + counters["KHONG co trong bang nao duoc dua"])
    print(f"bo dong thuan: {len(consensus)} cau, do duoc {measured}\n")
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    if measured:
        recall = 100 * counters["CO trong 9 bang duoc dua"] / measured
        print(f"\nRECALL TRUY HOI: {recall:.0f}%")
        print("  cao  -> nghen o buoc chon dong/cot, prompt la viec dung")
        print("  thap -> moi tang phia sau la vo nghia, phai dung lai truy hoi")
    for line in misses:
        print(line)


if __name__ == "__main__":
    main()
