"""Let the model choose the cell and let code choose the unit.

Classifying every wrong chain-of-thought answer against the questions whose value is
known: 22% picked the wrong cell, 11% computed something that is in no cell at all, and
**11% were out by an exact power of ten** — nine of them by exactly 1000. An answer that
is 1000× the truth is not a reading failure. The model found the right figure and then
mis-scaled it, usually by treating a "Triệu VND" column as though it held đồng.

That half of the job does not need a model. The scale of a table is written above it or in
its header, and a scanner for that was measured at 98% on the documents it covers. So this
recovers the cell the model must have read — the one whose value, at some scale, produces
the answer it gave — and re-derives the answer using the scale the table actually
declares.

A rewrite only happens when the model's implied scale and the declared scale disagree AND
the cell is unambiguous. Where several cells could explain the answer, nothing is touched:
guessing which one the model meant would be inventing a reading.

Usage:
  python scripts/fresh/fix_units.py --results artifacts/fresh/cot_results.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from document_scale import scale_of_line  # noqa: E402

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
CONTEXT_CHARS = 300


def contexts_of(path: Path) -> dict[int, str]:
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous = {}, 0
    for match in ANCHOR_RE.finditer(body):
        out[int(match.group(1))] = re.sub(
            r"\s+", " ", body[previous:match.start()].strip())[-CONTEXT_CHARS:]
        previous = match.end()
    return out


def declared_scale(grid: list[list[str]], context: str) -> float | None:
    """The scale the table itself states: header row first, then the prose above it."""

    if grid:
        found = ps.scale_from_unit_text(",".join(grid[0]))
        if found is not None:
            return found
    for piece in reversed(re.split(r"(?<=[:.)])\s+", context)):
        found = scale_of_line(piece)
        if found is not None:
            return found
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/fresh/cot_results.jsonl")
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/cot_unitfixed.jsonl")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    refs_of = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            refs_of[record["id"]] = record["meta"]["refs"]

    counters: Counter[str] = Counter()
    grids: dict[Path, list[list[str]]] = {}
    contexts: dict[str, dict[int, str]] = {}
    out = []

    for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get("answer")
        if value is None:
            out.append(record)
            counters["khong co dap an"] += 1
            continue
        value = float(value)
        _name, unit = unit_of(questions.get(record["id"], ""))
        if not unit or not value:
            out.append(record)
            counters["khong ro don vi cau hoi"] += 1
            continue

        # Which (cell, scale) pairs would produce exactly this answer?
        matches = []
        for ref in refs_of.get(record["id"], []):
            path = (ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                    ref["doc"] / f"{ref['doc']}_extracted_tables"
                    / f"table_{ref['table_id']}.csv")
            if path not in grids:
                if len(grids) > 300:
                    grids.clear()
                try:
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        grids[path] = list(csv_mod.reader(handle))
                except OSError:
                    grids[path] = []
            grid = grids[path]
            if not grid:
                continue
            if ref["doc"] not in contexts:
                contexts[ref["doc"]] = contexts_of(
                    ROOT / "data" / "official_corpus" / ref["ticker"] /
                    ref["year"] / ref["doc"] / f"{ref['doc']}_extracted.txt")
            stated = declared_scale(grid, contexts[ref["doc"]].get(
                ref["table_id"], ""))
            if stated is None:
                continue
            for row in grid[1:]:
                for cell in row:
                    raw = str(cell).strip()
                    if not raw:
                        continue
                    parsed = ps.parse_vn_number(raw)
                    if parsed is None or not parsed:
                        continue
                    for scale in SCALES:
                        if abs(abs(parsed) * scale / unit - abs(value)) <= 0.01:
                            matches.append((abs(parsed), scale, stated))
                            break

        implied = {(scale, stated) for _cell, scale, stated in matches}
        if not matches:
            counters["dap an khong khop o nao — de nguyen"] += 1
            out.append(record)
            continue
        if len({m[0] for m in matches}) > 1 or len(implied) > 1:
            counters["nhieu o giai thich duoc — de nguyen"] += 1
            out.append(record)
            continue
        cell, scale, stated = matches[0]
        if abs(scale - stated) < 1e-9:
            counters["he so model dung khop bang — de nguyen"] += 1
            out.append(record)
            continue

        fixed = round(cell * stated / unit, 2)
        counters[f"SUA: model dung {scale:g}, bang khai {stated:g}"] += 1
        record = dict(record)
        record["answer"] = fixed
        record["unit_fixed_from"] = value
        out.append(record)

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
