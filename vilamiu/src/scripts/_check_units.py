"""Is the generated answer expressed in the unit its question asks for?

Hand-reading sixty records turned up a pattern no counter was looking for:

    "... cuối năm 2017 là bao nhiêu (tỷ đồng)?"   answer "6.831.894.847.293"
    "... đến ngày 31/12/2025 là bao nhiêu triệu đồng?"  answer "82.497.905.724"

Those are raw đồng, not tỷ and not triệu. The question names a unit and the gold
answer ignores it.

This matters more than any defect found so far. Unit handling is our single
largest error class — 82 questions ship a đồng amount for a "%" question — and
training on pairs whose gold answer ignores the asked unit teaches the model to
ignore it too, on purpose and thoroughly.

The test needs no gold: parse the unit the question asks for with our own parser,
then ask whether the recorded answer's magnitude is consistent with it. A figure
at least a thousand times its own unit scale did not get converted.

Usage:  PYTHONPATH=src python scripts/_check_units.py artifacts/easy_416.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import UNIT_SCALE, _target_unit  # noqa: E402

RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "easy_416.jsonl"


def to_float(text) -> float | None:
    raw = str(text).strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(".", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    rows = [
        json.loads(line.replace(": NaN", ": null"))
        for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"{RECORDS.name}: {len(rows)} records\n")

    units: Counter[str] = Counter()
    verdict: Counter[str] = Counter()
    offenders: dict[str, list[int]] = {}

    for record in rows:
        question = record.get("question", "")
        unit = _target_unit(question)
        units[unit or "(none)"] += 1
        scale = UNIT_SCALE.get(unit)
        value = to_float(record.get("answer"))
        if scale is None or scale <= 1.0 or value is None or value == 0:
            verdict["not_checkable"] += 1
            continue
        # A figure expressed in its own unit is small; the raw đồng amount is
        # `scale` times larger. Anything at or above the scale itself never got
        # converted — "bao nhiêu tỷ đồng" answered with 6.8e12 rather than 6831.
        if abs(value) >= scale:
            verdict["UNCONVERTED"] += 1
            offenders.setdefault(unit, []).append(record["id"])
        else:
            verdict["consistent"] += 1

    print("units the questions ask for:")
    for unit, count in units.most_common():
        print(f"  {unit:16s} {count:4d}")

    checkable = verdict["UNCONVERTED"] + verdict["consistent"]
    print(f"\nof the {checkable} records asking for a scaled currency unit:")
    print(f"  answer already in that unit   : {verdict['consistent']:4d}")
    print(f"  answer left as raw đồng       : {verdict['UNCONVERTED']:4d}"
          f"  ({verdict['UNCONVERTED'] / max(checkable, 1):.1%})")
    print(f"  not checkable (no scale)      : {verdict['not_checkable']:4d}")

    for unit, ids in sorted(offenders.items()):
        print(f"\n  {unit}: {len(ids)} unconverted, ids {ids[:12]}")

    print("\n  Training on the unconverted ones teaches the model that the unit")
    print("  named in the question does not change the answer — which is the")
    print("  error class we lose the most points to.")


if __name__ == "__main__":
    main()
