"""Compare shipped submissions: how many answers moved, and what shape the code has.

Reads the zips directly so it works even when `run_submit.py` on disk cannot
rebuild the best submission.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "submissions"


def load(name: str) -> dict[int, dict]:
    path = SUBS / name
    with zipfile.ZipFile(path) as zf:
        target = next(n for n in zf.namelist() if n.endswith("submission.json"))
        rows = json.loads(zf.read(target).decode("utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("predictions") or rows.get("results") or list(rows.values())
    return {int(r["id"]): r for r in rows}


CONST = re.compile(r"^\s*result\s*=\s*-?[\d_.]+\s*$")


def shape(code: str) -> str:
    if not code or CONST.match(code.strip()):
        return "constant"
    frames = len(set(re.findall(r"\bdfs?\d*\b", code)))
    if "dfs[" in code:
        frames = max(frames, len(set(re.findall(r"dfs\[[^\]]+\]", code))))
    return f"frames={min(frames, 4)}"


def num(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def profile(name: str, rows: dict[int, dict]) -> None:
    shapes = Counter(shape(r.get("pandas_query", "")) for r in rows.values())
    vals = [num(r.get("answer")) for r in rows.values()]
    zeros = sum(1 for v in vals if v == 0.0)
    huge = sum(1 for v in vals if v is not None and abs(v) > 1e16)
    neg = sum(1 for v in vals if v is not None and v < 0)
    refs = Counter(len(r.get("relevant_tables") or []) for r in rows.values())
    print(f"\n== {name}  n={len(rows)}")
    print("  code shapes:", dict(shapes.most_common()))
    print(f"  zero answers: {zeros}   |ans|>1e16: {huge}   negative: {neg}")
    print("  declared-table counts:", dict(sorted(refs.items())))


def main() -> None:
    names = sys.argv[1:] or ["screen_ratio_gated.zip", "retry14b.zip"]
    loaded = {n: load(n) for n in names}
    for n, rows in loaded.items():
        profile(n, rows)

    if len(names) == 2:
        a, b = (loaded[n] for n in names)
        ids = sorted(set(a) & set(b))
        changed = [i for i in ids if not close(num(a[i].get("answer")), num(b[i].get("answer")))]
        print(f"\n== diff {names[0]} -> {names[1]}: {len(changed)} answers moved")
        code_changed = [i for i in ids if a[i].get("pandas_query") != b[i].get("pandas_query")]
        print(f"   programs rewritten: {len(code_changed)}")
        for i in changed[:8]:
            print(f"   id={i}: {a[i].get('answer')!r} -> {b[i].get('answer')!r}")


def close(x, y) -> bool:
    if x is None or y is None:
        return x is y
    return abs(x - y) <= 0.01


if __name__ == "__main__":
    main()
