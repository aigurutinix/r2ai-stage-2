"""Take a row wholesale from a fallback submission where the base ships an
answer that cannot be right on the question's own terms.

The regenerated build fixed 47 self-evident defects and introduced 18, all of one
shape: a count/year/screen question answered with a raw money figure, which is the
unguarded column scan taking over when the program failed. The previous build
answered those sensibly, so the cheapest repair is to keep its row.

The whole row moves — question, answer, pandas_query, relevant_tables, evidence —
never the answer alone: the private round reads the query, and an answer that its
own query does not produce is worse than a wrong one. Every csv the moved row
cites is copied across too, or the scorer cannot bind its frames.

Usage:
  PYTHONPATH=src python scripts/splice_rows.py \
      --base nounit_clean.zip --fallback planfirst_direct_clean.zip \
      --out nounit_hybrid.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util  # noqa: E402

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
    # A count of companies is a small whole number; the screens never name 60.
    if aud.COUNT_Q.search(question) and (
            not float(value).is_integer() or value < 0 or value > 60):
        return True
    return value == 0


def read(path: Path):
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])
    return payload, rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_path = ROOT / "submissions" / args.base
    fall_path = ROOT / "submissions" / args.fallback
    out_path = ROOT / "submissions" / args.out

    payload, rows = read(base_path)
    _, fall_rows = read(fall_path)
    fallback = {row["question"]: row for row in fall_rows}

    swapped, needed = [], set()
    for index, row in enumerate(rows):
        question = row["question"]
        other = fallback.get(question)
        if other is None:
            continue
        if impossible(question, row.get("answer")) and \
                not impossible(question, other.get("answer")):
            rows[index] = other
            swapped.append((question, row.get("answer"), other.get("answer")))
            for item in other.get("evidence") or []:
                needed.add(item["csv_path"])

    if isinstance(payload, list):
        payload = rows
    elif "predictions" in payload:
        payload["predictions"] = rows
    else:
        payload[list(payload.keys())[0]] = rows

    shutil.copy(base_path, out_path)
    with zipfile.ZipFile(base_path) as archive:
        present = set(archive.namelist())
    missing = sorted(needed - present)

    # Rewriting submission.json in place is not possible in a zip, so the archive
    # is rebuilt: every original entry except the manifest, plus the csvs the
    # moved rows need, plus the new manifest.
    with zipfile.ZipFile(base_path) as src, \
            zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            if name == "submission.json":
                continue
            dst.writestr(name, src.read(name))
        if missing:
            with zipfile.ZipFile(fall_path) as extra:
                for name in missing:
                    dst.writestr(name, extra.read(name))
        dst.writestr("submission.json",
                     json.dumps(payload, ensure_ascii=False, indent=1))

    print(f"đổi {len(swapped)} dòng, thêm {len(missing)} csv -> {args.out}")
    for question, before, after in swapped:
        print(f"  {question[:88]}")
        print(f"     {before}  ->  {after}")


if __name__ == "__main__":
    main()
