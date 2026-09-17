"""Does the generator's gold program reproduce its gold answer under OUR reader?

The generated `pandas_query` is `result = df.iloc[2, 0]`. That is only correct
under one specific reading of the CSV: whether `read_csv` consumed the first row
as a header, whether the label column is present, and how the number is parsed
all shift what `iloc[2, 0]` means. If their convention differs from the one the
scorer uses — and from the one our own submissions use — then every training pair
teaches a cell offset that is wrong at inference time, and the model would learn
the mistake perfectly.

So no record enters the training set on trust. Each one is executed here the way
`vifin.answering.sandbox` executes a submission, and kept only if the value it
returns matches the `answer` the generator recorded.

A high pass rate says the two conventions agree and the data is usable as-is. A
low one is not a reason to discard the data — it is a reason to find the offset
and correct it once, before generating thousands more.

Usage:
  PYTHONPATH=src python scripts/validate_qarecords.py runs/vilamiu-gen-easy/per_question.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402

RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "runs" / "vilamiu-gen-easy" / "per_question.jsonl")
TOL = 2e-4

# Injected so the gold program's trailing `df.iloc[r, c]` can be turned into the
# float our sandbox requires, without editing the program's cell choice.
_VN_HELPER = '''
def _vn(text):
    raw = str(text).strip().replace("%", "").replace(" ", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")
    value = float(raw)
    if negative:
        value = -value
    return value
'''


def to_float(text) -> float | None:
    """Vietnamese numeral to float: '.' groups thousands, ',' is the decimal."""

    raw = str(text).strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", raw):
        raw = raw.replace(".", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def load_grid(csv_path: Path) -> list[list[str]]:
    import csv as csv_mod  # noqa: PLC0415

    with csv_path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv_mod.reader(handle)]


def main() -> None:
    rows = [
        json.loads(line)
        for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"{RECORDS}: {len(rows)} records")

    outcome: Counter[str] = Counter()
    failures: list[tuple[int, str, object, object]] = []
    for record in rows:
        paths = record["csv_path"]
        paths = [paths] if isinstance(paths, str) else list(paths)
        try:
            grids = [load_grid(ROOT / p if not Path(p).is_absolute() else Path(p))
                     for p in paths]
        except FileNotFoundError as exc:
            outcome["csv_missing"] += 1
            failures.append((record["id"], f"missing csv: {exc}", None, None))
            continue

        names = ["df"] if len(grids) == 1 else [f"df{i+1}" for i in range(len(grids))]
        tables = dict(zip(names, grids))
        # Their gold program ends in a raw `df.iloc[r, c]`, which yields the cell
        # *string* — "1.498.203.140.705" — and their gold `answer` is that same
        # string. Our sandbox refuses a non-numeric result because EXECUTION is
        # scored by numeric comparison, so running the gold program verbatim
        # fails here by construction. Execute it as-is and parse both sides:
        # what matters for training is whether the cell they chose is the right
        # cell, not whether their formatting convention is ours.
        source = record["pandas_query"].rstrip() + "\nresult = float(_vn(result))"
        raw = run_query(_VN_HELPER + "\n" + source, tables)
        if not raw.ok:
            outcome["program_failed"] += 1
            failures.append((record["id"], str(raw.error)[:90], None, None))
            continue
        result = raw

        want = to_float(record["answer"])
        got = to_float(result.value) if not isinstance(result.value, (int, float)) \
            else float(result.value)
        if want is None or got is None:
            outcome["unparseable"] += 1
            failures.append((record["id"], "unparseable", record["answer"], result.value))
            continue
        if abs(got - want) <= TOL * max(abs(got), abs(want), 1e-9):
            outcome["MATCH"] += 1
        else:
            outcome["mismatch"] += 1
            failures.append((record["id"], "value mismatch", want, got))

    total = sum(outcome.values()) or 1
    print()
    for key, count in outcome.most_common():
        print(f"  {key:16s} {count:5d}  ({count / total:5.1%})")

    if failures:
        print(f"\n  first {min(8, len(failures))} failures:")
        for qid, why, want, got in failures[:8]:
            print(f"    id={qid:5d} {why}")
            if want is not None or got is not None:
                print(f"           recorded {want!r}   executed {got!r}")

    print(f"\n  usable for training: {outcome['MATCH']}/{total} = "
          f"{outcome['MATCH'] / total:.1%}")


if __name__ == "__main__":
    main()
