"""Splice verified raw-text cell reads into an existing submission.

Only rows where:
  - raw_plan relocated the model's figure to a CSV cell with label agreement
  - the question is a money one-cell (has a named unit)
  - the new answer differs from the incumbent
  - the incumbent is on the doubtful (not-a-cell) list when that file exists
  - the program re-executes and passes basic impossible() gates

Usage:
  PYTHONPATH=src python scripts/splice_raw_plan.py \
      --base aimed_fill.zip --plan artifacts/fresh/raw_plan.jsonl \
      --doubtful artifacts/fresh/not_a_cell.json --out aimed_fill.zip
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from build_submission import UNIT_SCALES, unit_of  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableStore  # noqa: E402

spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts" / "audit_submission.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)

PROGRAM = '''def _num(raw):
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    if not text or text in ("-", "--"):
        return 0.0
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value

_cell = _num(df.iloc[{row}, {col}])
result = round(abs(_cell) * {scale!r} / {unit!r}, 2)
'''

DERIVED_RE = re.compile(
    r"tỷ lệ|tỉ lệ|phần trăm|\bROA\b|\bROE\b|biên|tăng trưởng|chênh lệch|"
    r"bao nhiêu lần|trung bình|bình quân|bao nhiêu (doanh nghiệp|công ty)|"
    r"năm nào",
    re.I,
)


def load_grid(path: Path) -> list[list[str]] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle)]


def impossible(question: str, value: float) -> bool:
    if value != value or value in (float("inf"), float("-inf")):
        return True
    if aud.YEAR_Q.search(question) and not (
            float(value).is_integer() and 1990 <= value <= 2100):
        return True
    if aud.COUNT_Q.search(question) and (
            not float(value).is_integer() or value < 0 or value > 60):
        return True
    if aud.PERCENT_Q.search(question) and not aud.TIMES_Q.search(question) and \
            abs(value) > 1e5:
        return True
    if aud.TIMES_Q.search(question) and (value > 500 or value < 0):
        return True
    return value == 0


def start_line_ref(store: TableStore, doc: str, table_id: int) -> str | None:
    frame = store.frame
    hit = frame[(frame.doc_name == doc) & (frame.table_id == table_id)]
    if not len(hit):
        return None
    return f"{doc}|{int(hit.iloc[0].start_line)}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aimed_fill.zip")
    parser.add_argument("--plan", default="artifacts/fresh/raw_plan.jsonl")
    parser.add_argument("--doubtful", default="artifacts/fresh/not_a_cell.json")
    parser.add_argument("--out", default="aimed_fill.zip")
    parser.add_argument("--backup", default="aimed_fill_pre_raw.zip")
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--allow-all", action="store_true",
                        help="splice even when not on doubtful list")
    args = parser.parse_args()

    base = ROOT / "submissions" / args.base
    out = ROOT / "submissions" / args.out
    shutil.copy(base, ROOT / "submissions" / args.backup)

    doubtful: set[int] = set()
    dpath = ROOT / args.doubtful
    if dpath.exists():
        payload = json.loads(dpath.read_text(encoding="utf-8"))
        doubtful = set(payload.get("doubtful") or [])

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions[row["id"]] = row["question"]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    with zipfile.ZipFile(base) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))
        files = {n: archive.read(n) for n in archive.namelist()
                 if n != "submission.json"}

    by_id = {r["id"]: r for r in rows}
    spliced = 0
    skipped = {"score": 0, "derived": 0, "unit": 0, "same": 0,
               "doubt": 0, "grid": 0, "exec": 0, "impossible": 0}

    for line in (ROOT / args.plan).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        plan = json.loads(line)
        qid = plan["id"]
        if float(plan.get("score") or 0) < args.min_score:
            skipped["score"] += 1
            continue
        text = questions.get(qid, "")
        if DERIVED_RE.search(text):
            skipped["derived"] += 1
            continue
        _name, unit = unit_of(text)
        if not unit:
            skipped["unit"] += 1
            continue
        if not args.allow_all and doubtful and qid not in doubtful:
            skipped["doubt"] += 1
            continue

        csv_path = Path(plan["csv"])
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        grid = load_grid(csv_path)
        if not grid:
            skipped["grid"] += 1
            continue
        row_i, col_i = int(plan["row"]), int(plan["col"])
        scale = float(plan["scale"])
        code = PROGRAM.format(row=row_i, col=col_i, scale=scale, unit=unit)
        outcome = run_query(code, {"df": grid})
        if not outcome.ok:
            skipped["exec"] += 1
            continue
        value = float(outcome.value)
        if impossible(text, value):
            skipped["impossible"] += 1
            continue

        old = aud.as_float(by_id[qid].get("answer"))
        if old is not None and abs(old - value) < 0.011:
            skipped["same"] += 1
            continue

        # Ship CSV under the scorer's usual data/ name when possible.
        doc = plan["doc"]
        tid = int(plan["table_id"])
        ship_name = f"data/{doc}_table_{tid}.csv"
        if ship_name not in files:
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(grid)
            files[ship_name] = buf.getvalue().encode("utf-8")

        ref = start_line_ref(store, doc, tid)
        by_id[qid]["answer"] = value
        by_id[qid]["pandas_query"] = code
        by_id[qid]["evidence"] = [{"variable": "df", "csv_path": ship_name}]
        if ref:
            by_id[qid]["relevant_tables"] = [ref]
        spliced += 1

    payload = [by_id[i] for i in sorted(by_id)]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("submission.json",
                         json.dumps(payload, ensure_ascii=False))

    print(f"backup -> {args.backup}")
    print(f"spliced {spliced} -> {out}")
    print("skipped:", {k: v for k, v in skipped.items() if v})


if __name__ == "__main__":
    main()
