"""Compare the two-stage reader against the shipped build without any gold.

The only offline instruments this project has left are the leaderboard and things
that need no gold at all. `impossible()` is the second kind: it flags an answer
that contradicts its own question — a year that is not a year, a share above 100,
a count of companies at 4000, a zero. It cannot say an answer is right, but every
answer it flags is certainly wrong, so the flag rate is a lower bound on the error
rate and it is comparable across builds on the same questions.

Run before spending a submission slot on a wide splice.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("sr", ROOT / "scripts" / "splice_reader.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def load_zip(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as archive:
        return {r["id"]: r for r in json.loads(archive.read("submission.json"))}


def load_cache(path: str) -> dict[int, dict]:
    rows = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ok"):
                rows[record["id"]] = record
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = load_zip("aimed.zip")
    for name, path in (("ghim theo note", "artifacts/two_stage.jsonl"),
                       ("bang hang 1", "artifacts/rank1.jsonl")):
        cache = load_cache(path)
        ids = sorted(cache)
        bad_reader = sum(1 for i in ids
                         if sr.impossible(base[i]["question"], cache[i].get("value")))
        bad_base = sum(1 for i in ids
                       if sr.impossible(base[i]["question"], base[i].get("answer")))
        print(f"{name}: {len(ids)} cau  ->  may doc sai hien nhien {bad_reader} "
              f"({100 * bad_reader / len(ids):.1f}%)   aimed {bad_base} "
              f"({100 * bad_base / len(ids):.1f}%)")
    total = sum(1 for r in base.values() if sr.impossible(r["question"], r.get("answer")))
    print(f"aimed tren ca 1012: {total} ({100 * total / 1012:.1f}%)")


if __name__ == "__main__":
    main()
