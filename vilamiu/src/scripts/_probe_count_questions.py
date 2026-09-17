"""Counting questions wearing a currency unit taken from their own threshold.

`_target_unit` scans the whole sentence for the first unit word. "Có bao nhiêu
công ty có doanh thu trên 1.000 tỷ đồng?" contains "tỷ đồng", so the question is
tagged `ty` and every branch downstream believes the answer is money — it divides
by a billion, it gates on `unit_scale`, and `impossible()` measures the wrong
ceiling. The answer is a count of companies.

The unit word in those sentences belongs to the *filter*, not to the thing being
asked for. This measures the family and what it currently ships, so the size of
the class is known before any regex is touched.

Usage:  PYTHONPATH=src python scripts/_probe_count_questions.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"

# "Có bao nhiêu công ty …", "… có bao nhiêu doanh nghiệp …", "bao nhiêu năm …".
COUNT_RE = re.compile(
    r"\bcó\s+bao\s+nhiêu\s+"
    r"(công ty|doanh nghiệp|ngân hàng|đơn vị|năm|quý|khoản|mã|cổ đông|thành viên)\b",
    re.I,
)


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}

    def answer(i: int) -> float:
        try:
            return float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            return 0.0

    lines: list[str] = []
    p = lines.append

    hits = [q for q in parsed.values() if COUNT_RE.search(q.question)]
    p(f"counting questions: {len(hits)}")
    p(f"  units they currently carry: "
      f"{dict(Counter(q.target_unit or '(none)' for q in hits))}")

    mistagged = [q for q in hits if q.target_unit]
    p(f"\ncarrying a unit that belongs to their own filter: {len(mistagged)}")
    # A count of companies named in the question cannot exceed the roster it
    # ranges over, so anything large is proof the unit was applied.
    suspect = [q for q in mistagged if abs(answer(q.id)) > 60]
    p(f"  of which ship a figure too large to be a count: {len(suspect)}")
    for q in mistagged:
        mark = "  WRONG" if abs(answer(q.id)) > 60 else "       "
        p(f" {mark} id={q.id:4d} unit={q.target_unit:15s} ships {answer(q.id):.6g}")
        p(f"          {q.question[:145]}")

    p(f"\ncounting questions already untagged ({len(hits) - len(mistagged)}), "
      f"what they ship:")
    for q in hits:
        if q.target_unit:
            continue
        p(f"   id={q.id:4d} ships {answer(q.id):.6g}   {q.question[:110]}")

    out = ROOT / "artifacts" / "_probe_count_questions.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
