"""Grade generated QARecords by the failure modes execution validation cannot see.

The pipeline discards any program that does not run and reproduce its answer.
That gate is real, but it is blind to a program that reproduces the answer
*because the answer is written inside it*:

    result = df.loc[df['31/12/2017...VND'] == '6.831.894.847.293',
                    '31/12/2017...VND'].values[0]

This executes, returns exactly the recorded answer, and passes every check —
while teaching a model to write a filter containing the number it was asked to
find. At inference that number is precisely what is unknown. It is the same
defect as a hard-coded constant, which `reads_no_frame` already rejects, wearing
a DataFrame as a disguise.

Three other shapes survive validation and are equally useless as supervision:
an answer of `NaN`, an answer that is a row *label* rather than a figure, and an
answer that is an empty-cell dash.

Usage:  PYTHONPATH=src python scripts/_audit_generated.py progress.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "progress.jsonl"

# A run of digits with Vietnamese thousands separators, long enough that it is a
# figure rather than a row index or a statutory code.
BIG_LITERAL_RE = re.compile(r"\d{1,3}(?:\.\d{3}){2,}|\d{7,}")
LABELISH_RE = re.compile(r"^[^\d]*[A-Za-zÀ-ỹ][^\d]*$")


def normalise(text: str) -> str:
    return re.sub(r"[^\d]", "", str(text))


def classify(record: dict) -> str:
    """`USABLE`, or the reason this record must not become a training pair.

    Exposed so `build_sft.py` runs the same gate. Auditing by hand and then
    building the dataset with a different filter is how the two drift apart.
    """

    answer = record.get("answer")
    query = str(record.get("pandas_query") or "")

    if answer is None:
        return "answer_is_nan"
    text = str(answer).strip()
    if text in ("-", "--", "", "nan"):
        return "answer_is_empty_cell"
    digits = normalise(text)
    if not digits:
        return "answer_is_a_label"

    # `df['Năm nay'].sum()` over a *string* column concatenates instead of adding.
    # The result reproduces itself perfectly and is nonsense:
    #   "Triệu VND4.198.071117.727.9198.028.5311.351.989553.980811.247132.671.737"
    # Nothing in this corpus is a 20-digit figure — VCB's total assets are 16.
    if len(digits) > 20:
        return "answer_is_concatenated_strings"

    # `.values[0][0]` indexes the first *character* of the cell string, so a
    # question asking for tỷ đồng gets back "1". A one-digit answer to a question
    # that names a money unit is a truncation, not a figure.
    if len(digits) <= 2 and re.search(r"\]\s*\[\s*0\s*\]|\.values\[0\]\[", query):
        return "answer_is_one_character"

    # A question that asks for a rate must not come back with a balance-sheet
    # amount. The `paper` engine produced "Tỷ lệ dự phòng rủi ro cho vay khách
    # hàng ... là bao nhiêu phần trăm?" answered 168969.0 — the query read the
    # provision itself instead of dividing it by anything. Both gates upstream
    # pass it: the program runs, and the answer is a number.
    #
    # The bound is deliberately loose. Ratios in "lần" and "vòng" reach the low
    # hundreds in this corpus (inventory turnover, coverage), and a percentage can
    # exceed 100 legitimately, so only figures a rate cannot reach are rejected.
    # A sum of two real financial figures is not zero. The caption-grouped pool
    # produced "Tổng thu nhập từ hoạt động dịch vụ của NAB 2023 và SHB 2016" with
    # answer 0.0 — the query matched no row in one of the two tables and summed an
    # empty selection. Both gates upstream pass it: the program runs, and zero is
    # a number.
    try:
        if abs(float(digits[:18])) == 0.0:
            return "answer_is_zero"
    except ValueError:
        pass

    question = str(record.get("question") or "")
    asks_rate = re.search(r"phần trăm|bao nhiêu %|tỷ lệ|bao nhiêu lần|vòng",
                          question, re.I)
    if asks_rate:
        try:
            magnitude = abs(float(digits[:18]))
        except ValueError:
            magnitude = 0.0
        if magnitude > 10_000:
            return "rate_question_absolute_answer"

    literals = BIG_LITERAL_RE.findall(query)
    if any(normalise(found) == digits for found in literals):
        return "query_contains_its_own_answer"
    if len(digits) >= 7 and digits in normalise(query):
        return "query_contains_its_own_answer"
    return "USABLE"


def main() -> None:
    rows = []
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # The generator writes bare NaN, which is not valid JSON.
        rows.append(json.loads(line.replace(": NaN", ": null")))

    verdict: Counter[str] = Counter()
    examples: dict[str, list[int]] = {}

    def flag(key: str, qid: int) -> None:
        verdict[key] += 1
        examples.setdefault(key, []).append(qid)

    for record in rows:
        flag(classify(record), record["id"])

    total = len(rows) or 1
    print(f"{RECORDS.name}: {len(rows)} records\n")
    for key, count in verdict.most_common():
        ids = ", ".join(str(i) for i in examples[key][:8])
        print(f"  {key:32s} {count:4d}  ({count / total:5.1%})   ids: {ids}")

    print(f"\n  usable as supervision: {verdict['USABLE']}/{total} = "
          f"{verdict['USABLE'] / total:.1%}")
    print("\n  `query_contains_its_own_answer` is the one that matters: those")
    print("  records pass the pipeline's execution check by construction, so no")
    print("  amount of validating downstream will remove them. They have to be")
    print("  filtered on the shape of the program.")


if __name__ == "__main__":
    main()
