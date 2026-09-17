"""Splice deterministic panel answers into a submission zip.

Only `answer`, `pandas_query`, and `evidence` move. `relevant_tables` /
`relevant_docs` stay as in the base so TABLES_F2 is unchanged. Each swapped
row ships `data/metric_panel_q{id}.csv` — the frame its program reads.

By default overwrites rows where the panel answer passes `impossible()` and
re-executes cleanly. Use `--where defect` to only replace base rows that are
already impossible on the question's own terms.

Usage:
  PYTHONPATH=src python scripts/splice_panel_det.py \
      --base aimed_fill.zip --cache artifacts/panel_det.jsonl --out aimed_fill.zip
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402

spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts" / "audit_submission.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)


def impossible(question: str, answer) -> bool:
    value = aud.as_float(answer)
    if value is None:
        return True
    if aud.YEAR_Q.search(question) and not (
            float(value).is_integer() and 1990 <= value <= 2100):
        return True
    if aud.SHARE_Q.search(question) and not aud.TIMES_Q.search(question) and \
            not aud.GROWTH_Q.search(question) and (value > 100 or value < 0):
        return True
    if aud.PERCENT_Q.search(question) and not aud.TIMES_Q.search(question) and \
            abs(value) > 1e5:
        return True
    if aud.TIMES_Q.search(question) and (value > 500 or value < 0):
        return True
    if aud.COUNT_Q.search(question) and (
            not float(value).is_integer() or value < 0 or value > 60):
        return True
    return value == 0


def csv_text(rows: list[list[str]]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aimed_fill.zip")
    parser.add_argument("--cache", default="artifacts/panel_det.jsonl")
    parser.add_argument("--out", default="aimed_fill.zip")
    parser.add_argument("--where", choices=("all", "defect"), default="all")
    parser.add_argument("--backup", default="aimed_fill_pre_panel.zip",
                        help="Copy of --base before overwrite (empty to skip)")
    args = parser.parse_args()

    base_path = ROOT / "submissions" / args.base
    out_path = ROOT / "submissions" / args.out
    cache_path = ROOT / args.cache

    if args.backup and base_path.resolve() == out_path.resolve():
        backup_path = ROOT / "submissions" / args.backup
        if not backup_path.exists():
            shutil.copy2(base_path, backup_path)
            print(f"backup -> {backup_path.name}")
        else:
            print(f"backup already exists: {backup_path.name}")

    with zipfile.ZipFile(base_path) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    panel: dict[int, dict] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ok"):
                panel[record["id"]] = record

    by_id = {row["id"]: index for index, row in enumerate(rows)}
    added: dict[str, str] = {}
    swapped = 0
    same = 0
    skipped = {"no_panel": 0, "not_defect": 0, "failed": 0, "impossible": 0}

    for qid, record in panel.items():
        index = by_id.get(qid)
        if index is None:
            skipped["no_panel"] += 1
            continue
        row = rows[index]
        if args.where == "defect" and not impossible(row["question"], row.get("answer")):
            skipped["not_defect"] += 1
            continue

        outcome = run_query(record["code"], {"df": record["panel_rows"]})
        if not outcome.ok:
            skipped["failed"] += 1
            continue
        if impossible(row["question"], outcome.value):
            skipped["impossible"] += 1
            continue

        name = f"metric_panel_q{qid}.csv"
        added[name] = csv_text(record["panel_rows"])
        old = aud.as_float(row.get("answer"))
        if old is not None and abs(old - outcome.value) < 0.011:
            same += 1
        rows[index] = {
            **row,
            "answer": outcome.value,
            "pandas_query": record["code"],
            "evidence": [{"variable": "df", "csv_path": f"data/{name}"}],
        }
        swapped += 1

    # If writing over the same path, stage to a temp zip then replace.
    stage = out_path if out_path != base_path else out_path.with_suffix(".tmp.zip")
    refresh = {f"data/{name}" for name in added}
    with zipfile.ZipFile(base_path) as src, \
            zipfile.ZipFile(stage, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            if name == "submission.json" or name in refresh:
                continue
            dst.writestr(name, src.read(name))
        for name, text in added.items():
            dst.writestr(f"data/{name}", text)
        dst.writestr("submission.json", json.dumps(rows, ensure_ascii=False, indent=1))

    if stage != out_path:
        stage.replace(out_path)

    print(f"doi {swapped} dong (trung dap an {same}), them {len(added)} panel csv "
          f"-> {args.out}")
    print("  bo qua: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v))


if __name__ == "__main__":
    main()
