"""Which shipped answers cannot exist in the tables we shipped with them.

ANSWER is scored on the `answer` field alone, with no execution step, so it can be
checked by any means at all — including means a pandas program cannot use. This is
the cheapest one, and it needs neither a label nor a GPU.

For a plain lookup ("Lãi tiền gửi năm 2018 ... là bao nhiêu triệu đồng?") the answer
*is* a cell. So the number we shipped, converted back to đồng, has to equal some cell
of some table in that question's own evidence, times that column's unit multiplier.
If it equals nothing, the answer is not a reading of those tables — it is wrong, and
we know it without being told.

Only single-lookup questions are tested. A ratio or a difference is derived, so no
cell holds it and absence proves nothing.

The column multiplier is unknown per cell, so every plausible one is tried. That
makes the test *conservative*: trying more multipliers can only turn a "wrong" into a
"grounded", never the reverse, so every question it flags is flagged on the weakest
possible assumption.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Vietnamese statements print in đồng, nghìn, triệu, tỷ; "trăm tỷ" appears in
# questions rather than column headers but costs nothing to allow.
SCALES = (1.0, 1e3, 1e6, 1e9, 1e11, 1e12)


def cell_values(zf: zipfile.ZipFile, record: dict) -> list[float]:
    values: list[float] = []
    for item in record.get("evidence") or []:
        try:
            text = zf.read(item["csv_path"]).decode("utf-8", errors="replace")
        except KeyError:
            continue
        for row in csv.reader(io.StringIO(text)):
            for cell in row:
                parsed = lookup_mod._parse_cell(cell)
                if parsed is not None:
                    values.append(float(parsed))
    return values


def grounded(target: float, values: list[float], unit_scale: float) -> bool:
    """Is `target` (in đồng) some cell times a power of ten?

    The window is set by how the answer was rounded, not by its magnitude. The
    shipped number is `round(x, 2)` where `x` is in the question's unit, so the cell
    behind it is only pinned to ±0.005 × unit_scale đồng — five million đồng for a
    question asked in tỷ. A relative window instead of that one was this script's
    first answer and it over-flagged badly: id=4 reads a single cell through
    `find_row` and still failed, because 0.005 tỷ is ten times wider than any
    relative slack at that magnitude.
    """

    window = max(1e-2, 0.005 * abs(unit_scale) * 1.02, abs(target) * 1e-9)
    if target == 0.0:
        return any(abs(v) * s <= window for v in values for s in SCALES) or any(
            v == 0.0 for v in values
        )
    for value in values:
        for scale in SCALES:
            if abs(value * scale - target) <= window:
                return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "noconst.zip"

    questions = {
        q.id: q
        for q in parse_all(
            ROOT / "data" / "questions" / "questions.jsonl",
            ROOT / "data" / "code_stock.csv",
        )
    }

    tested = ok = bad = 0
    flagged: list[int] = []
    examples: list[str] = []

    with zipfile.ZipFile(ROOT / "submissions" / base) as zf:
        rows = json.loads(zf.read("submission.json"))
        for record in rows:
            question = questions.get(record["id"])
            if question is None or question.unit_scale is None:
                continue
            if not lookup_mod.is_single_lookup(question.question):
                continue
            tested += 1
            answer = float(record.get("answer") or 0.0)
            unit_scale = float(question.unit_scale)
            target = answer * unit_scale
            if grounded(target, cell_values(zf, record), unit_scale):
                ok += 1
            else:
                bad += 1
                flagged.append(record["id"])
                if len(examples) < 12:
                    examples.append(
                        f"    id={record['id']:4d}  answer={answer!r} "
                        f"(={target:.4g} đồng)  {question.question[:64]}"
                    )

    print(f"base={base}")
    print(f"  single-lookup questions tested   : {tested}")
    print(f"  answer IS a cell in own evidence : {ok}")
    print(f"  answer is in NO cell -> wrong    : {bad}  ({bad / max(1, tested):.1%})")
    for line in examples:
        print(line)

    out = ROOT / "artifacts" / "_ungrounded_ids.json"
    out.write_text(json.dumps(flagged), encoding="utf-8")
    print(f"\n{len(flagged)} ids -> artifacts/{out.name}")


if __name__ == "__main__":
    main()
