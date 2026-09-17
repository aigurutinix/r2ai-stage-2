"""Repair "năm nào" answers inside an existing submission zip.

The best submission is `direct14b.zip` at 0.3794, and everything built today sits
below it, so a fix belongs on that artefact rather than on a pipeline that has
regressed. Twenty-two of its fifty-three "năm nào" questions answer with a figure
where a year was asked for, and those are wrong with certainty.

This rewrites only those, leaving every other prediction untouched, and emits a
real program: the model's own extraction replayed against each year's table with a
comparison chain returning the winner.

Usage:
  PYTHONPATH=src python scripts/patch_year_answers.py \
      --base direct14b.zip --out direct14b_year.zip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import year_answer  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame

    base = ROOT / "submissions" / args.base
    out = ROOT / "submissions" / args.out
    shutil.copy(base, out)

    with zipfile.ZipFile(base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])

    def grid_for(ref: str):
        if "|" not in ref:
            return None, None
        doc, line = ref.rsplit("|", 1)
        try:
            selected = frame[(frame.doc_name == doc) & (frame.start_line == int(line))]
        except ValueError:
            return None, None
        if not len(selected):
            return None, None
        key = TableKey(doc, int(selected.iloc[0].table_id))
        return key, store.rows(key)

    counts = {"asks": 0, "already": 0, "repaired": 0, "declined": 0}
    for row in rows:
        if not year_answer.asks_for_a_year(row.get("question", "")):
            continue
        counts["asks"] += 1
        if year_answer.looks_like_a_year(row.get("answer")):
            counts["already"] += 1
            continue

        dated = []
        for ref in row.get("relevant_tables") or []:
            year = year_answer.year_of_document(ref.split("|")[0])
            key, grid = grid_for(ref)
            if year is not None and grid:
                dated.append((year, key, grid))
        # One table per year, in year order: a document appearing twice would
        # otherwise compare a year against itself.
        seen = set()
        unique = []
        for year, key, grid in sorted(dated, key=lambda item: item[0]):
            if year in seen:
                continue
            seen.add(year)
            unique.append((year, key, grid))

        frames = [(year, grid) for year, _, grid in unique]
        want_max = not re.search(
            r"thấp nhất|nhỏ nhất|ít nhất", row.get("question", ""), re.I)
        # Prefer locating the (wrong) figure's label across years — replay of a
        # broken extractor often returns a non-None tuple that then fails probe
        # and blocks the repair path entirely.
        fixed = year_answer.repair_across_documents(
            frames, row.get("answer"), want_max=want_max)
        if fixed is None:
            fixed = year_answer.replay_across_years(
                row.get("pandas_query") or "", frames, want_max=want_max)
        if fixed is None:
            # Last resort: single multi-year table with year columns in the header.
            for _, key, grid in unique:
                one = year_answer.repair(grid, row.get("answer"), want_max=want_max)
                if one is not None:
                    code, year = one
                    probe = run_query(
                        code if "def num(" in code else PRELUDE + "\n" + code,
                        {"df": grid},
                    )
                    if probe.ok and year_answer.looks_like_a_year(probe.value):
                        row["answer"] = probe.value
                        row["pandas_query"] = (
                            code if "def num(" in code else PRELUDE + "\n" + code)
                        row["evidence"] = [{
                            "variable": "df",
                            "csv_path": (
                                f"data/{key.doc_name}_table_{key.table_id}.csv"),
                        }]
                        counts["repaired"] += 1
                        fixed = "done"
                        break
            if fixed == "done":
                continue
            counts["declined"] += 1
            continue

        code, _, used = fixed
        # Sandbox binds df1..dfN by insertion order, so consecutive names must
        # match the order of `used` frame positions.
        bound = {f"df{index + 1}": frames[position][1]
                 for index, position in enumerate(used)}
        probe = run_query(code if "def num(" in code else PRELUDE + "\n" + code, bound)
        if not (probe.ok and year_answer.looks_like_a_year(probe.value)):
            counts["declined"] += 1
            continue

        row["answer"] = probe.value
        row["pandas_query"] = code if "def num(" in code else PRELUDE + "\n" + code
        row["evidence"] = [
            {"variable": f"df{index + 1}",
             "csv_path": f"data/{unique[position][1].doc_name}_table_"
                         f"{unique[position][1].table_id}.csv"}
            for index, position in enumerate(used)
        ]
        counts["repaired"] += 1

    with zipfile.ZipFile(base) as archive:
        names = [n for n in archive.namelist() if n != "submission.json"]
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as fresh:
            for name in names:
                fresh.writestr(name, archive.read(name))
            fresh.writestr("submission.json",
                           json.dumps(payload, ensure_ascii=False))

    print(f"{args.base} -> {args.out}")
    for name, value in counts.items():
        print(f"  {name:10s} {value}")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
