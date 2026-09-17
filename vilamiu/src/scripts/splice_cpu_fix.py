"""Splice CPU-fix cache into a submission zip (default: aimed_fill).

Panel rows become `data/metric_panel_q{id}.csv`. Table keys pull CSV from the
store like `splice_reader.py`. `relevant_tables` / `relevant_docs` stay put.

Usage:
  PYTHONPATH=src python scripts/splice_cpu_fix.py \
      --base aimed_fill.zip --cache artifacts/cpu_fix.jsonl --out aimed_fill.zip
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
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

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
    parser.add_argument("--cache", default="artifacts/cpu_fix.jsonl")
    parser.add_argument("--out", default="aimed_fill.zip")
    parser.add_argument("--where", choices=("all", "defect"), default="all")
    parser.add_argument("--backup", default="aimed_fill_pre_cpu.zip")
    args = parser.parse_args()

    base_path = ROOT / "submissions" / args.base
    out_path = ROOT / "submissions" / args.out

    if args.backup and base_path.resolve() == out_path.resolve():
        backup = ROOT / "submissions" / args.backup
        if not backup.exists():
            shutil.copy2(base_path, backup)
            print(f"backup -> {backup.name}")

    with zipfile.ZipFile(base_path) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    cache: dict[int, dict] = {}
    for line in (ROOT / args.cache).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ok"):
                cache[record["id"]] = record

    from collections import Counter

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    by_id = {row["id"]: index for index, row in enumerate(rows)}
    added: dict[str, str] = {}
    swapped = same = 0
    skipped: Counter[str] = Counter()

    for qid, record in cache.items():
        index = by_id.get(qid)
        if index is None:
            skipped["missing"] += 1
            continue
        row = rows[index]
        if args.where == "defect" and not impossible(row["question"], row.get("answer")):
            skipped["not_defect"] += 1
            continue

        if record["mode"] == "panel":
            frames = {"df": record["panel_rows"]}
            outcome = run_query(record["code"], frames)
            if not outcome.ok or reads_no_frame(record["code"]):
                skipped["failed"] += 1
                continue
            if impossible(row["question"], outcome.value):
                skipped["impossible"] += 1
                continue
            name = f"metric_panel_q{qid}.csv"
            added[name] = csv_text(record["panel_rows"])
            evidence = [{"variable": "df", "csv_path": f"data/{name}"}]
        else:
            keys = [TableKey(doc, int(tid)) for doc, tid in record["keys"]]
            frames = {
                name: store.rows(key)
                for name, key in zip(record["variables"], keys)
            }
            outcome = run_query(record["code"], frames)
            if not outcome.ok or reads_no_frame(record["code"]):
                skipped["failed"] += 1
                continue
            if impossible(row["question"], outcome.value):
                skipped["impossible"] += 1
                continue
            evidence = []
            single = len(keys) == 1
            for position, key in enumerate(keys, start=1):
                name = f"{key.doc_name}_table_{key.table_id}.csv"
                if name not in added:
                    added[name] = csv_text(store.rows(key))
                variable = "df" if single else f"df{position}"
                evidence.append({"variable": variable, "csv_path": f"data/{name}"})

        old = aud.as_float(row.get("answer"))
        if old is not None and abs(old - outcome.value) < 0.011:
            same += 1
        rows[index] = {
            **row,
            "answer": outcome.value,
            "pandas_query": record["code"],
            "evidence": evidence,
        }
        swapped += 1

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

    print(f"doi {swapped} dong (trung {same}), them {len(added)} csv -> {args.out}")
    print("  bo qua:", ", ".join(f"{k}={v}" for k, v in skipped.items() if v) or "(khong)")


if __name__ == "__main__":
    main()
