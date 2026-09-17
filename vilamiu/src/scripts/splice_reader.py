"""Put the two-stage reader's programs into the best shipped submission.

The code on disk no longer reproduces `direct14b` — a rebuild from it differs on
518 of 1012 answers and declares a flat 5 refs where that build declared 3 to 11 —
so a fresh `run_submit` cannot be compared against `aimed.zip` at all. The reader's
output therefore ships as a splice onto `aimed`, which is not answer-rearrangement:
every spliced row carries a program the 14B model wrote this morning against a
table it was pinned to, and its value was executed, not copied.

Only three fields move: `answer`, `pandas_query`, `evidence` (plus the csv the new
evidence cites). `relevant_tables` and `relevant_docs` stay exactly as `aimed` has
them, which holds TABLES_F2 at 0.5603 and makes any change in EXECUTION_ACCURACY
attributable to the reader alone. The two fields are scored independently — a team
on the public board declares no tables at all and still scores EXEC 0.6067.

  --where defect   only rows `aimed` cannot be right about (unanswered, zero,
                   exactly 100.0, or contradicting the question) — a narrow diff,
                   the band that has won before. Measured: 1 row, so not useful.
  --where all      every row the reader ran — the head-on test of reader vs the
                   label matcher, at the wide-diff size that has lost before

Usage:
  PYTHONPATH=src python scripts/splice_reader.py \
      --base aimed.zip --cache artifacts/two_stage.jsonl --out tsall.zip --where all
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts" / "audit_submission.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)


def impossible(question: str, answer) -> bool:
    """True only where the answer contradicts the question, with no gold needed."""

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
    parser.add_argument("--base", default="aimed.zip")
    parser.add_argument("--cache", default="artifacts/two_stage.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--where", choices=("defect", "all"), default="all")
    parser.add_argument("--spec", default="")
    parser.add_argument("--min-score", type=float, default=0.0)
    # `num(df, r, c) * scale` is one cell and one factor. On the 440 one-cell
    # questions it reached, its self-evident-defect rate is 1.4% — the shipped
    # build's is 2.3%. On the other 172 it is 88%, because a share question
    # answered with a raw money figure is wrong by construction. Confining the
    # splice to the shape the emitter was built for is the difference between
    # testing the reader and testing a category error.
    parser.add_argument("--shape", default="", choices=("", "mot o", "ty le", "nam"))
    # The surgical path: `_label_agree.py` names the questions where the reader's
    # row label matches the question better than the shipped build's does. That is
    # 47 rows, which is the diff size that has twice won on the board, where a
    # 285-row swap is the size that has lost four times out of four.
    parser.add_argument("--ids-file", default="")
    args = parser.parse_args()

    wanted: set[int] | None = None
    if args.ids_file:
        text = (ROOT / args.ids_file).read_text(encoding="utf-8")
        wanted = {int(t) for t in text.replace("\n", ",").split(",") if t.strip()}

    base_path = ROOT / "submissions" / args.base
    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(base_path) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    reader: dict[int, dict] = {}
    for line in (ROOT / args.cache).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ok"):
                reader[record["id"]] = record

    scores: dict[int, float] = {}
    if args.spec:
        for line in (ROOT / args.spec).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                scores[record["id"]] = record.get("score", 0.0)

    shape = None
    if args.shape:
        shape_spec = importlib.util.spec_from_file_location(
            "qs", ROOT / "scripts" / "_question_shape.py")
        shape_mod = importlib.util.module_from_spec(shape_spec)
        shape_spec.loader.exec_module(shape_mod)
        shape = shape_mod.shape

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    added: dict[str, str] = {}
    swapped = 0
    skipped = {"no_reader": 0, "low_score": 0, "wrong_shape": 0,
               "not_defect": 0, "failed": 0, "impossible": 0}

    for index, row in enumerate(rows):
        qid = row["id"]
        if wanted is not None and qid not in wanted:
            skipped["not_listed"] = skipped.get("not_listed", 0) + 1
            continue
        record = reader.get(qid)
        if record is None:
            skipped["no_reader"] += 1
            continue
        if args.min_score and scores.get(qid, 0.0) < args.min_score:
            skipped["low_score"] += 1
            continue
        if shape is not None and shape(row["question"]) != args.shape:
            skipped["wrong_shape"] += 1
            continue
        if args.where == "defect" and not impossible(row["question"], row.get("answer")):
            skipped["not_defect"] += 1
            continue

        # Re-execute here rather than trusting the cached value: the cache was
        # produced on the rented box against the same parquet, but a value that
        # does not reproduce locally would ship a query the scorer cannot honour.
        keys = [TableKey(doc, int(tid)) for doc, tid in record["keys"]]
        frames = {name: store.rows(key) for name, key in zip(record["variables"], keys)}
        outcome = run_query(record["code"], frames)
        if not outcome.ok:
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

        rows[index] = {
            **row,
            "answer": outcome.value,
            "pandas_query": record["code"],
            "evidence": evidence,
        }
        swapped += 1

    with zipfile.ZipFile(base_path) as src:
        present = set(src.namelist())
    missing = {f"data/{name}": text for name, text in added.items()
               if f"data/{name}" not in present}

    with zipfile.ZipFile(base_path) as src, \
            zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            if name != "submission.json":
                dst.writestr(name, src.read(name))
        for name, text in missing.items():
            dst.writestr(name, text)
        dst.writestr("submission.json", json.dumps(rows, ensure_ascii=False, indent=1))

    print(f"doi {swapped} dong, them {len(missing)} csv -> {args.out}")
    print("  bo qua: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v))


if __name__ == "__main__":
    main()
