"""Gate a submission zip before a slot is spent on it.

Every check here has caught a real defect at least once: a zip with 1011 rows, an
`evidence` entry pointing at a csv that was never written, a program that assigns a
constant (which the private round rejects on manual review), an empty
`relevant_tables`. Prints one line per zip and a verdict.

Usage:  PYTHONPATH=src python scripts/_validate_zip.py a.zip b.zip ...
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTANT = re.compile(r"^\s*result\s*=\s*-?[\d_.,eE+\-]+\s*$", re.MULTILINE)


def check(name: str) -> bool:
    path = ROOT / "submissions" / name
    if not path.exists():
        print(f"{name}: KHONG CO FILE")
        return False
    problems = []
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            problems.append("zip hong")
        names = set(archive.namelist())
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    if len(rows) != 1012:
        problems.append(f"{len(rows)} dong")
    ids = sorted(r["id"] for r in rows)
    if ids != list(range(1, 1013)):
        problems.append("id khong du 1..1012")

    missing = set()
    empty_answer = empty_query = empty_tables = constants = 0
    for row in rows:
        for item in row.get("evidence") or []:
            if item["csv_path"] not in names:
                missing.add(item["csv_path"])
        if row.get("answer") is None or row.get("answer") == "":
            empty_answer += 1
        query = row.get("pandas_query") or ""
        if not query.strip():
            empty_query += 1
        elif CONSTANT.search(query) and "num(" not in query and "df" not in query:
            constants += 1
        if not (row.get("relevant_tables") or []):
            empty_tables += 1

    if missing:
        problems.append(f"{len(missing)} csv thieu")
    if empty_answer:
        problems.append(f"{empty_answer} dong khong co answer")
    if empty_query:
        problems.append(f"{empty_query} dong khong co pandas_query")
    if constants:
        problems.append(f"{constants} chuong trinh gan hang")

    csvs = sum(1 for n in names if n.startswith("data/"))
    verdict = "OK" if not problems else "LOI: " + ", ".join(problems)
    print(f"{name:20s} {len(rows)} dong, {csvs} csv, "
          f"{empty_tables} dong khong khai bang  -> {verdict}")
    return not problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = all(check(name) for name in sys.argv[1:])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
