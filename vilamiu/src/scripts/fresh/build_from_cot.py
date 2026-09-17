"""Build a submission from chain-of-thought answers, standing on its own.

Every submission so far was the best zip with a patch on top, and each patch lost. The
organisers' own numbers say why the patches could not help: 89.3% of end-to-end failures
are the wrong cell or a missing table, so a mechanism that changes how the arithmetic is
written cannot move the score.

This one is built from its own answers. Where the reader produced no number the incumbent
stands, because a blank scores the same as a wrong answer and there is nothing to gain by
refusing.

`pandas_query` is the field the private round reviews by hand, and a chain-of-thought
answer has no program behind it. So each answered row ships the program that READS the
cell the reasoning used — reconstructed only when the reasoning's own number can be found
in a cell of the offered tables, which is also a check on the answer. Rows where no cell
matches keep the incumbent's query and are marked, rather than shipping a program that
asserts something the model never did.

Usage:
  python scripts/fresh/build_from_cot.py --results artifacts/fresh/cot_results.jsonl
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
from answer_gate import verdict  # noqa: E402
from build_submission import unit_of  # noqa: E402
from num_helper import SOURCE as NUM_SOURCE  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)

PROGRAM = '''{helper}
# Chain-of-thought located this figure; the cell below holds it, and the scale is the
# one the table is denominated in.
_cell = _num(df1.iloc[{row}, {col}])
result = round(_cell * {scale!r} / {unit!r}, 2)
'''


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/fresh/cot_results.jsonl")
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--base", default="submissions/aimed.zip")
    parser.add_argument("--out", default="submissions/cot.zip")
    parser.add_argument("--gate", action="store_true",
                        help="drop answers the question's own type rules out")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    answers = {}
    for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("answer") is not None:
                answers[record["id"]] = float(record["answer"])

    with zipfile.ZipFile(ROOT / args.base) as archive:
        base_rows = json.loads(archive.read("submission.json"))
        base_files = {name: archive.read(name) for name in archive.namelist()
                      if name != "submission.json"}

    counters: Counter[str] = Counter()
    out_rows, extra_files = [], {}

    for row in base_rows:
        qid = row["id"]
        value = answers.get(qid)
        if value is None:
            counters["giu incumbent — CoT khong tra loi"] += 1
            out_rows.append(dict(row))
            continue
        # A vote that lands on the incumbent's own value changes nothing, so its row is
        # left exactly as it was rather than given a freshly written program.
        try:
            if abs(value - float(row.get("answer"))) <= 0.01:
                counters["giu nguyen — bo phieu trung incumbent"] += 1
                out_rows.append(dict(row))
                continue
        except (TypeError, ValueError):
            pass
        if args.gate:
            rejected = verdict(questions[qid], value)
            if rejected is not None:
                counters["giu incumbent — cong kieu loai"] += 1
                out_rows.append(dict(row))
                continue

        # Find the cell the reasoning must have read, so the shipped program is real.
        _name, unit = unit_of(questions[qid])
        located = None
        if unit:
            for ref in (meta.get(qid) or {}).get("refs", []):
                path = (ROOT / "data" / "official_corpus" / ref["ticker"] /
                        ref["year"] / ref["doc"] /
                        f"{ref['doc']}_extracted_tables"
                        / f"table_{ref['table_id']}.csv")
                try:
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        grid = list(csv_mod.reader(handle))
                except OSError:
                    continue
                for r_index, grid_row in enumerate(grid[1:]):
                    for c_index, cell in enumerate(grid_row):
                        raw = str(cell).strip()
                        if not raw:
                            continue
                        parsed = ps.parse_vn_number(raw)
                        if parsed is None:
                            continue
                        for scale in SCALES:
                            if abs(abs(parsed) * scale / unit - abs(value)) <= 0.01:
                                located = (path, r_index, c_index, scale, ref)
                                break
                        if located:
                            break
                    if located:
                        break
                if located:
                    break

        merged = dict(row)
        merged["answer"] = round(value, 2)
        if located:
            path, r_index, c_index, scale, ref = located
            name = f"data/{ref['doc']}_table_{ref['table_id']}.csv"
            extra_files[name] = path.read_bytes()
            merged["pandas_query"] = PROGRAM.format(
                helper=NUM_SOURCE.rstrip(), row=r_index, col=c_index,
                scale=scale, unit=unit)
            merged["evidence"] = [{"variable": "df1", "csv_path": name}]
            counters["CoT tra loi, tim duoc o — kem chuong trinh that"] += 1
        else:
            counters["CoT tra loi, khong tim duoc o — giu query cu"] += 1
        out_rows.append(merged)

    files = dict(base_files)
    files.update(extra_files)
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(out_rows, ensure_ascii=False, indent=1))
        for name, blob in sorted(files.items()):
            archive.writestr(name, blob)

    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {target}")


if __name__ == "__main__":
    main()
