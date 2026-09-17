"""Lay out a random sample of questions so the answer can be established by reading.

Every accuracy figure produced today rests on treating agreement between two mechanisms
as truth. Agreement is not truth: the incumbent is wrong on three fifths of the exam, so
a set built from "the incumbent agreed with a reader" carries an unknown error rate, and
calling it gold made every number computed against it uninterpretable.

There is exactly one source of ground truth available without spending a submission:
reading the report. So this prints, for a RANDOM sample — not one conditioned on
agreement, which would reproduce the same bias — the question, the candidate tables with
row and column indices, and what each mechanism answered, in a form compact enough to
adjudicate by hand.

The sample is random over the whole exam so the resulting accuracy estimate is unbiased.
Twelve questions gives roughly ±14 points at 95% confidence, which is enough to tell a
39% mechanism from a 66% one, and not enough to rank two mechanisms five points apart —
that is the honest resolution of a hand sample this size.

Usage:
  python scripts/fresh/dump_for_hand.py --n 12 --seed 3 --out artifacts/fresh/hand.txt
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_TABLES = 3
MAX_ROWS = 30
LABEL_CHARS = 60
CELL_CHARS = 26


def render(ref: str, context: str, grid: list[list[str]]) -> str:
    lines = [f"  [{ref}]"]
    if context:
        lines.append(f"    (tren bang: ...{context[-200:]})")
    width = min(6, max((len(r) for r in grid), default=0))
    header = grid[0] if grid else []
    lines.append("    " + "  ".join(
        f"c{i}={str(header[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]}"
        if i < len(header) else f"c{i}=" for i in range(width)))
    kept = 0
    for index, row in enumerate(grid[1:]):
        if kept >= MAX_ROWS:
            lines.append("    ... (con dong)")
            break
        if not any(str(c).strip() for c in row):
            continue
        lines.append(f"    r{index} | " + " | ".join(
            str(row[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]
            if i < len(row) else "" for i in range(width)))
        kept += 1
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/hand.txt")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    with zipfile.ZipFile(ROOT / "submissions" / "aimed.zip") as archive:
        incumbent = {r["id"]: r.get("answer")
                     for r in json.loads(archive.read("submission.json"))}
    programs = {}
    path = ROOT / "artifacts" / "fresh" / "prog_think_both.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                programs[record["id"]] = record

    prompts = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            prompts[record["id"]] = record["meta"]

    # Random over the whole exam, restricted only to questions we have candidates for,
    # since a question with no candidates cannot be adjudicated from this dump.
    pool = sorted(set(questions) & set(prompts))
    random.seed(args.seed)
    chosen = random.sample(pool, min(args.n, len(pool)))

    out = io.StringIO()
    out.write(f"mau ngau nhien {len(chosen)} cau, seed={args.seed}, "
              f"tu {len(pool)} cau co ung vien\n\n")
    for qid in sorted(chosen):
        meta = prompts[qid]
        out.write("=" * 96 + f"\nid={qid}\n")
        out.write(f"HOI   : {questions[qid]}\n")
        out.write(f"aimed : {incumbent.get(qid)}\n")
        entry = programs.get(qid)
        out.write(f"prog  : {entry['answer'] if entry else '(khong tra loi)'}\n")
        if entry:
            code = entry["pandas_query"].strip().splitlines()
            out.write("        " + " / ".join(l.strip() for l in code[:4])[:300] + "\n")
        out.write("\n")
        for ref in meta["refs"][:MAX_TABLES]:
            csv_path = (ROOT / "data" / "official_corpus" / ref["ticker"] /
                        ref["year"] / ref["doc"] /
                        f"{ref['doc']}_extracted_tables"
                        / f"table_{ref['table_id']}.csv")
            try:
                with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                    grid = list(csv_mod.reader(handle))
            except OSError:
                continue
            out.write(render(ref["ref"], "", grid) + "\n\n")

    (ROOT / args.out).write_text(out.getvalue(), encoding="utf-8")
    print(f"{len(chosen)} cau -> {ROOT / args.out}")
    print(f"  {len(out.getvalue())} ky tu")


if __name__ == "__main__":
    main()
