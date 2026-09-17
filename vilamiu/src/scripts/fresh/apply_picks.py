"""Read the cell the model picked and convert it. The model chose; code reads.

The division of labour is the whole point. The model decided which row and which column,
because those are questions about meaning — whether `phải thu` or `phải trả` is what was
asked, whether `31/12/2020` or `1/1/2020` is "cuối năm". Everything after that is
arithmetic, and the organisers measured arithmetic errors at 0.9% of their failures while
a scale decision made in the wrong place cost this project a submission.

So here: parse the cell, take the scale the table states, divide by the unit the question
names, round once. The one semantic input carried over is `do_lon` — whether a figure
printed in brackets should be reported as a magnitude, which is a fact about the question
and not about the table.

Usage:
  python scripts/fresh/apply_picks.py --replies artifacts/fresh/pick_replies.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from fix_units import contexts_of, declared_scale  # noqa: E402
from label_match import score as label_score  # noqa: E402
from measure_unique import variants_from  # noqa: E402
from score_model import parse as parse_json  # noqa: E402
from table_reading import fold as fold_cell  # noqa: E402

_MILLION = re.compile(r"trieu")
_BILLION = re.compile(r"\bty\b")
_THOUSAND = re.compile(r"nghin|ngan")
_MONEY = re.compile(r"vnd|dong")


def scale_of_column(grid: list[list[str]], column: int) -> float | None:
    """The unit stated by the chosen column's own header, and nothing else.

    Reading any cell of the first two rows was too greedy: question 227's answer came
    from a row whose label says "(VND)" and was multiplied by a billion because some
    other header cell in the table mentioned tỷ. The unit that governs a cell is the one
    printed above that cell.
    """

    for row in grid[:2]:
        if column >= len(row):
            continue
        flat = fold_cell(row[column])
        if not _MONEY.search(flat):
            continue
        if _MILLION.search(flat):
            return 1e6
        if _BILLION.search(flat):
            return 1e9
        if _THOUSAND.search(flat):
            return 1e3
        return 1.0
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", default="artifacts/fresh/pick_replies.jsonl")
    parser.add_argument("--plan", default="artifacts/fresh/pick_plan.json")
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    parser.add_argument("--out", default="artifacts/fresh/pick_answers.jsonl")
    parser.add_argument("--specs", default="artifacts/fresh/spec_all.jsonl")
    args = parser.parse_args()

    variants = variants_from(ROOT / args.specs)

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    plan = {int(k): v for k, v in
            json.loads((ROOT / args.plan).read_text(encoding="utf-8")).items()}
    doc_scale: dict[str, float] = {}
    path = ROOT / args.doc_scale
    if path.exists():
        doc_scale = {k: float(v) for k, v
                     in json.loads(path.read_text(encoding="utf-8")).items()}

    counters: Counter[str] = Counter()
    answers = []

    for line in (ROOT / args.replies).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = record["id"]
        spec = parse_json(record.get("reply", ""))
        candidates = plan.get(qid)
        if not spec or not candidates:
            counters["khong doc duoc lua chon"] += 1
            continue
        try:
            index = int(spec.get("ung_vien"))
        except (TypeError, ValueError):
            counters["ung_vien khong phai so"] += 1
            continue
        if index < 0:
            counters["model noi khong ung vien nao dung"] += 1
            continue
        if index >= len(candidates):
            counters["ung_vien ngoai danh sach"] += 1
            continue
        item = candidates[index]

        column = spec.get("cot")
        if isinstance(column, str):
            match = re.search(r"\d+", column)
            column = int(match.group()) if match else None
        try:
            column = int(column)
        except (TypeError, ValueError):
            counters["cot khong doc duoc"] += 1
            continue
        offered = {c for c, _h, _r in item["cells"]}
        if column not in offered:
            counters["cot khong nam trong o duoc dua"] += 1
            continue

        csv_path = ROOT / item["csv"]
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                grid = list(csv_mod.reader(handle))
        except OSError:
            counters["khong doc duoc csv"] += 1
            continue
        row_index = item["row"] + 1
        if row_index >= len(grid) or column >= len(grid[row_index]):
            counters["dia chi ngoai bang"] += 1
            continue
        raw = str(grid[row_index][column]).strip()
        value = ps.parse_vn_number(raw)
        if value is None:
            counters["o khong phai so"] += 1
            continue

        _name, unit = unit_of(questions[qid])
        if not unit:
            counters["khong ro don vi cau hoi"] += 1
            continue
        # The chosen column's own header first; only then anything wider.
        scale = scale_of_column(grid, column)
        source = "tieu de cua chinh cot do"
        if scale is None:
            scale = item.get("scale")
            source = "tieu de bang"
        if scale is None:
            scale = declared_scale(grid, contexts_of(
                csv_path.parent.parent / f"{csv_path.parent.parent.name}"
                                         "_extracted.txt").get(item["table_id"], ""))
            source = "van ban tren bang"
        if scale is None:
            scale = doc_scale.get(csv_path.parent.parent.name)
            source = "ca tai lieu"
        if scale is None:
            counters["chua biet he so — bo qua"] += 1
            continue

        if spec.get("do_lon") is True:
            value = abs(value)
        result = round(value * scale / unit, 2)
        if not result:
            counters["ket qua bang 0 — bo qua"] += 1
            continue
        # How well the label the model chose actually matches the indicator. Read by
        # hand, this is what separated the right picks from the wrong ones: the model
        # was right on `TỔNG CỘNG NGUỒN VỐN` and `TỔNG TÀI SẢN`, and wrong on `Tiền`,
        # `- Nguyên giá` and `Từ Bảo hiểm Bảo Việt` — labels that match almost nothing.
        strength = label_score(variants.get(qid) or [], item["label"])
        answers.append({
            "id": qid, "answer": result, "csv": item["csv"],
            "row": item["row"], "col": column, "scale": scale, "unit": unit,
            "label": item["label"], "raw": raw, "strength": strength,
            "scale_from": source,
        })
        counters[f"TRA LOI (he so tu {source})"] += 1

    (ROOT / args.out).write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
