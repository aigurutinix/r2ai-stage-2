"""Currency answers whose magnitude is impossible once the asked-for unit is applied.

`impossible()` rejects a value when `answer * unit_scale` exceeds 1e16 đồng — VCB's
total assets are about 2e15, so nothing in this corpus is larger. It is checked on
the screen, locate, llm and plan branches, but **not** on the best-effort fallback
or the column scan, which are the last two things the cascade tries and together
serve 122 questions.

An answer that fails that test is a scale error: the program returned đồng where
the question asked for tỷ. It scores zero, which makes it free to replace — the
same free-roll property the rate-unit pool has.

Usage:  PYTHONPATH=src python scripts/_probe_dong_ceiling.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"
DONG_CEILING = 1e16


def main() -> None:
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    def answer(i: int) -> float:
        try:
            return float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            return 0.0

    lines: list[str] = []
    p = lines.append

    over: list[tuple[int, float, float]] = []
    per_unit: Counter[str] = Counter()
    for qid, question in parsed.items():
        scale = UNIT_SCALE.get(question.target_unit)
        if scale is None:
            continue
        per_unit[question.target_unit] += 1
        value = answer(qid)
        if abs(value) * scale > DONG_CEILING:
            over.append((qid, value, abs(value) * scale))

    p(f"currency questions: {sum(per_unit.values())}  {dict(per_unit)}")
    p(f"shipping a figure that exceeds 1e16 đồng once scaled: {len(over)}")
    for qid, value, implied in sorted(over, key=lambda r: -r[2])[:30]:
        q = parsed[qid]
        p(f"  id={qid:4d} unit={q.target_unit:9s} ships {value:.4g} "
          f"-> {implied:.3g} đồng")
        p(f"        {q.question[:130]}")

    # The same defect one order of magnitude down: an answer that is plausible as
    # đồng but implausible in the unit asked for is very likely unconverted.
    p("\nsuspicious but not impossible (answer looks like raw đồng):")
    soft = 0
    for qid, question in parsed.items():
        scale = UNIT_SCALE.get(question.target_unit)
        if scale is None or scale <= 1.0:
            continue
        value = abs(answer(qid))
        # A figure at least as large as the scale itself means the conversion
        # probably never happened: "bao nhiêu tỷ đồng" answered with 1.5e12.
        if value >= scale:
            soft += 1
    p(f"  answer >= its own unit scale: {soft} questions")

    out = ROOT / "artifacts" / "_probe_dong_ceiling.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
