"""Measure a new answer source against the incumbent, then choose the diff to ship.

There is no gold here, so the only offline signal is agreement between mechanisms that
fail differently. Two readers that both produce a figure in the billions and produce the
SAME figure did not do that by chance, so agreement is evidence for both; disagreement
says one of them is wrong without saying which.

The incumbent answers 1007 of 1012 questions, so there is nothing to add by coverage and
a new source can only gain by replacing. Replacing blindly has cost 0.32 to 0.40 points
per row three times. So the diff is chosen, in order of how little it risks:

  gate      the incumbent's answer is the wrong KIND for the question — a money figure
            where a year was asked for. It cannot be right, so replacing it is free.
  not-cell  the incumbent's answer for a single-figure money question matches no cell in
            the document at any scale. It was not read off the report.
  agree     the new source and the incumbent produce the same value; shipping it changes
            nothing and is only listed to size the overlap.
  block     everything the new source answered inside the blocks a single computation can
            address, which is the wide diff — worth shipping only if agreement is high
            enough to believe the source outranks the incumbent.

Usage:
  python scripts/fresh/decide_splice.py --results artifacts/fresh/prog_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocks import block_of  # noqa: E402

# The blocks one computation over the offered tables can plausibly settle.
REACHABLE = ("tien — MOT O", "khac — don gian", "ty le — mot nam", "nam nao",
             "tien — nhieu nam", "ty le — nhieu nam", "khac — nhieu nam",
             "tong hop nhieu o", "dem cong ty")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--base", default="submissions/aimed.zip")
    parser.add_argument("--blocks", default="artifacts/fresh/blocks.json")
    parser.add_argument("--not-a-cell", default="artifacts/fresh/not_a_cell.json")
    parser.add_argument("--gate-reject", default="artifacts/fresh/gate_reject_aimed.json")
    parser.add_argument("--second", default="",
                        help="another plan or results file, for cross-agreement")
    parser.add_argument("--out-prefix", default="artifacts/fresh/splice")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    with zipfile.ZipFile(ROOT / args.base) as archive:
        incumbent = {r["id"]: r.get("answer")
                     for r in json.loads(archive.read("submission.json"))}

    # Either a results jsonl or a built zip, so a plan that goes through
    # build_submission can be compared on the same footing as a program run.
    new = {}
    source = ROOT / args.results
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for record in json.loads(archive.read("submission.json")):
                value = record.get("answer")
                if value not in (None, "", 0.0):
                    new[record["id"]] = value
    else:
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                new[record["id"]] = record["answer"]

    second = {}
    if args.second:
        path = ROOT / args.second
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if "answer" in record:
                        second[record["id"]] = record["answer"]

    blocks = json.loads((ROOT / args.blocks).read_text(encoding="utf-8"))
    of_block = {qid: name for name, ids in blocks.items() for qid in ids}
    doubtful = set()
    path = ROOT / args.not_a_cell
    if path.exists():
        doubtful = set(json.loads(path.read_text(encoding="utf-8"))["doubtful"])
    gate_rejected = set()
    path = ROOT / args.gate_reject
    if path.exists():
        gate_rejected = set(json.loads(path.read_text(encoding="utf-8")))

    per_block: dict[str, Counter] = {}
    agreeing = set()
    for qid, value in new.items():
        name = of_block.get(qid, block_of(questions.get(qid, "")))
        counter = per_block.setdefault(name, Counter())
        counter["tra loi"] += 1
        try:
            old = float(incumbent.get(qid))
        except (TypeError, ValueError):
            counter["incumbent khong so"] += 1
            continue
        if abs(float(value) - old) <= 0.01:
            counter["trung incumbent"] += 1
            agreeing.add(qid)

    print(f"nguon moi tra loi {len(new)} cau\n")
    order = sorted(per_block.items(), key=lambda item: -item[1]["tra loi"])
    total_answered = total_agree = 0
    for name, counter in order:
        answered, agree = counter["tra loi"], counter["trung incumbent"]
        share = 100 * agree / answered if answered else 0
        mark = "" if name in REACHABLE else "   (khoi sang loc)"
        print(f"  {answered:5d} tra loi | trung incumbent {agree:5d} "
              f"({share:3.0f}%)  {name}{mark}")
        if name in REACHABLE:
            total_answered += answered
            total_agree += agree
    if total_answered:
        rate = 100 * total_agree / total_answered
        print(f"\ntrong cac khoi voi tay tra loi duoc: {total_agree}/"
              f"{total_answered} = {rate:.0f}% trung incumbent")

    reachable_ids = {qid for qid in new
                     if of_block.get(qid, block_of(questions.get(qid, "")))
                     in REACHABLE}
    variants = {
        "gate": sorted(set(new) & gate_rejected),
        "notcell": sorted((set(new) & doubtful) - agreeing),
        "block": sorted(reachable_ids - agreeing),
    }
    if second:
        cross = {qid for qid, value in new.items()
                 if qid in second and abs(float(value) - float(second[qid])) <= 0.01}
        variants["hai_nguon_trung"] = sorted(
            (cross & reachable_ids) - agreeing)

    print()
    for name, ids in variants.items():
        target = ROOT / f"{args.out_prefix}_{name}.json"
        target.write_text(json.dumps(ids), encoding="utf-8")
        print(f"  {len(ids):5d} dong  ->  {target.name}")


if __name__ == "__main__":
    main()
