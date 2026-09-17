"""Check shipped answers against the answering contract BTC's own generator used.

`prompts/answering/program_system.txt` states two conventions that produced the
gold answers: a difference with no stated direction is a non-negative magnitude,
and the final value is converted to the unit the question asks for. Answers that
break either are wrong by construction, independent of which table was read.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIFF = re.compile(r"chênh lệch|hiệu số|biến động|thay đổi|cao hơn|thấp hơn|nhiều hơn|ít hơn", re.I)
DIRECTED = re.compile(r"tăng hay giảm|tăng/giảm|âm hay dương|dấu của", re.I)
RATIO_UNIT = re.compile(r"bao nhiêu\s*%|phần trăm|tỷ lệ|tỉ lệ|\blần\b|\bvòng\b|biên lợi nhuận", re.I)


def load_zip(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as zf:
        target = next(n for n in zf.namelist() if n.endswith("submission.json"))
        rows = json.loads(zf.read(target).decode("utf-8"))
    return {int(r["id"]): r for r in rows}


def load_questions() -> dict[int, str]:
    path = ROOT / "data" / "questions" / "questions.jsonl"
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["id"])] = rec["question"]
    return out


def num(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = sys.argv[1] if len(sys.argv) > 1 else "retry14b.zip"
    rows = load_zip(name)
    questions = load_questions()

    neg_diff, neg_other, ratio_big, ratio_ok, zeros = [], [], [], [], []
    for qid, row in rows.items():
        q = questions.get(qid, "")
        value = num(row.get("answer"))
        if value is None:
            continue
        if value == 0.0:
            zeros.append(qid)
        if value < 0:
            (neg_diff if DIFF.search(q) and not DIRECTED.search(q) else neg_other).append(qid)
        if RATIO_UNIT.search(q):
            (ratio_big if abs(value) > 1000 else ratio_ok).append(qid)

    print(f"== {name}")
    print(f"negative on an undirected difference : {len(neg_diff)}   (contract says magnitude)")
    print(f"negative elsewhere                   : {len(neg_other)}")
    print(f"ratio-unit question, |answer| > 1000 : {len(ratio_big)} of {len(ratio_big) + len(ratio_ok)}")
    print(f"answer exactly zero                  : {len(zeros)}")
    print("\nsample undirected-difference negatives:")
    for qid in neg_diff[:6]:
        print(f"  id={qid}  {rows[qid]['answer']}  |  {questions.get(qid, '')[:110]}")
    print("\nsample oversized ratios:")
    for qid in ratio_big[:6]:
        print(f"  id={qid}  {rows[qid]['answer']}  |  {questions.get(qid, '')[:110]}")


if __name__ == "__main__":
    main()
