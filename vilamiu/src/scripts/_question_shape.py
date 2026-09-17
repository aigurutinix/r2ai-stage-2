"""Which questions can a one-cell program answer at all?

The two-stage reader emits `num(df, r, c) * scale` — one cell, one factor. That
shape can answer "how much was X", and it cannot answer "what share", "what
percentage", "how many times", "which year", "how many companies": those need two
cells and an operation, or a header read. Its answers on those questions are raw
money figures, which is exactly what the self-evident-defect audit found — 153 of
its 612 answers contradict their own question, against 14 for the shipped build.

So the reader is not simply worse; it is being asked questions outside its range.
This splits the 1012 by shape so the splice can be confined to the questions the
emitter is built for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("sr", ROOT / "scripts" / "splice_reader.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
aud = sr.aud


def shape(question: str) -> str:
    """The coarsest split that matters: can one cell be the whole answer?"""

    if aud.YEAR_Q.search(question):
        return "nam"
    if aud.COUNT_Q.search(question):
        return "dem"
    if aud.TIMES_Q.search(question):
        return "so lan"
    if aud.SHARE_Q.search(question) or aud.PERCENT_Q.search(question) or \
            aud.GROWTH_Q.search(question):
        return "ty le"
    return "mot o"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with zipfile.ZipFile(ROOT / "submissions" / "aimed.zip") as archive:
        base = {r["id"]: r for r in json.loads(archive.read("submission.json"))}

    reader = {}
    for line in (ROOT / "artifacts" / "two_stage.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("ok"):
                reader[record["id"]] = record

    everything = Counter(shape(r["question"]) for r in base.values())
    covered = Counter(shape(base[i]["question"]) for i in reader)
    print("hinh dang cau hoi trong 1012 / trong 612 cau may doc tra loi:")
    for name, count in everything.most_common():
        print(f"  {name:8s} {count:4d}  ->  may doc phu {covered.get(name, 0):4d}")

    plain = [i for i in reader if shape(base[i]["question"]) == "mot o"]
    flagged = sum(1 for i in plain
                  if sr.impossible(base[i]["question"], reader[i].get("value")))
    print(f"\ntren {len(plain)} cau mot-o: may doc sai hien nhien {flagged} "
          f"({100 * flagged / max(1, len(plain)):.1f}%)")
    agree = sum(1 for i in plain
                if abs(float(reader[i].get("value") or 0)
                       - float(base[i].get("answer") or 0)) <= 0.01)
    print(f"trung dap an voi aimed: {agree}/{len(plain)} "
          f"({100 * agree / max(1, len(plain)):.1f}%)")


if __name__ == "__main__":
    main()
