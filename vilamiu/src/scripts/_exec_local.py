"""Run every shipped program against the zip's own CSVs: local EXEC, and the
consistency between the two scored fields.

`noconst.zip` scored identically to `retry14b.zip` on all ten leaderboard columns
after 26 programs were rewritten to read cells instead of embedding literals. That
is the harness below being right about the grader, so it can now be trusted to
predict EXEC offline instead of spending a submission to learn one number.

Two things get counted, and the second is the point:

* crash rate — a program that raises scores zero on EXEC however good its logic.
* `answer` vs the program's own output. These are *separately scored* fields, and
  nothing forces them to agree. Where they disagree we are asserting two different
  numbers for one question, so at most one of the two metrics can be earned. That
  is not a wrong answer; it is a wasted one, and it is invisible on the leaderboard
  because both columns simply look low.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering.sandbox import SAFE_BUILTINS, portability_problems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-2


def run(code: str, frames: dict) -> tuple[float | None, str]:
    import pandas as pd

    if not code.strip():
        return None, "empty"
    problems = portability_problems(code)
    if problems:
        return None, problems[0].split(":")[0]
    scope: dict = {"pd": pd, "dfs": dict(frames)}
    if len(frames) == 1:
        scope["df"] = next(iter(frames.values()))
    for position, frame in enumerate(frames.values(), start=1):
        scope.setdefault(f"df{position}", frame)
    scope.update(frames)
    try:
        exec(compile(code, "<q>", "exec"), {"__builtins__": SAFE_BUILTINS}, scope)
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__
    value = scope.get("result")
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"not-scalar:{type(value).__name__}"
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None, "not-finite"
    return number, ""


def main() -> None:
    import pandas as pd

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "noconst.zip"

    stats: Counter = Counter()
    mismatches: list[tuple[int, float, float]] = []

    with zipfile.ZipFile(ROOT / "submissions" / base) as zf:
        rows = json.loads(zf.read("submission.json"))
        cache: dict[str, object] = {}
        for record in rows:
            frames = {}
            for item in record.get("evidence") or []:
                path = item["csv_path"]
                if path not in cache:
                    cache[path] = pd.read_csv(io.BytesIO(zf.read(path)))
                frames[item["variable"]] = cache[path]

            value, error = run(record.get("pandas_query") or "", frames)
            shipped = float(record.get("answer") or 0.0)
            if value is None:
                stats["CRASH: " + error] += 1
                continue
            stats["ran"] += 1
            if abs(value - shipped) <= TOL:
                stats["ran, matches `answer`"] += 1
            else:
                stats["ran, DIFFERS from `answer`"] += 1
                mismatches.append((record["id"], shipped, value))

    total = len(rows)
    crashes = sum(v for k, v in stats.items() if k.startswith("CRASH"))
    print(f"base={base}  n={total}")
    print(f"  ran without crashing            : {stats['ran']}  ({stats['ran'] / total:.1%})")
    print(f"  crashed (scores 0 on EXEC)      : {crashes}  ({crashes / total:.1%})")
    print(f"  ran and agrees with `answer`    : {stats['ran, matches `answer`']}")
    print(f"  ran but DISAGREES with `answer` : {stats['ran, DIFFERS from `answer`']}")
    for name, count in sorted(stats.items()):
        if name.startswith("CRASH"):
            print(f"      {count:4d}  {name}")
    for qid, shipped, value in mismatches[:15]:
        print(f"    id={qid:4d}  answer={shipped!r}  program returns {value!r}")

    out = ROOT / "artifacts" / "_field_mismatch_ids.json"
    out.write_text(json.dumps([q for q, _, _ in mismatches]), encoding="utf-8")
    print(f"\n{len(mismatches)} ids where the two scored fields disagree -> artifacts/{out.name}")


if __name__ == "__main__":
    main()
