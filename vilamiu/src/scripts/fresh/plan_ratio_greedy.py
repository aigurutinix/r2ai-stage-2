"""Give the rate questions a denominator, because a blank scores zero either way.

248 questions ask for a %, a multiple or a turnover and end up blank: the greedy plan
finds a cell for them, but a single cell cannot be converted into a rate and the
money-unit path skips them.

The numerator is already located. What is missing is a base, and the explicit-pair
regex only parses a minority of the phrasings. So where the pair cannot be read, the
base falls back to the total of the numerator's own statement:

  cân đối       270 TỔNG CỘNG TÀI SẢN, or 440 TỔNG CỘNG NGUỒN VỐN
  kết quả       10  Doanh thu thuần
  lưu chuyển    50  Lưu chuyển tiền thuần trong năm

That is a guess, and it is a defensible one — an unqualified "tỷ trọng X" in these
questions is usually a share of total assets or of net revenue. More to the point, the
alternative is not a better answer, it is no answer: on EXECUTION_ACCURACY a wrong
rate and a blank both score zero, so the only thing a refusal protects is the illusion
of precision.

Where the question does name a base explicitly, `plan_ratio.py` has already claimed it
and this leaves it alone.

Usage:  python scripts/fresh/plan_ratio_greedy.py --show 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402

PERCENT_RE = re.compile(r"%|phần trăm", re.I)
TIMES_RE = re.compile(r"bao nhiêu lần|vòng quay|hệ số", re.I)

# The base each statement offers when the question does not name one.
DEFAULT_BASE = {
    "cdkt": ("270", "440", "400"),
    "kqkd": ("10", "01"),
    "lctt": ("50", "20"),
}
# Wording that names a base outright, even where the pair regex could not split it.
NAMED_BASE = (
    (re.compile(r"tổng tài sản|tổng cộng tài sản", re.I), "cdkt", ("270",)),
    (re.compile(r"tổng nguồn vốn", re.I), "cdkt", ("440",)),
    (re.compile(r"vốn chủ sở hữu", re.I), "cdkt", ("400",)),
    (re.compile(r"nợ phải trả", re.I), "cdkt", ("300",)),
    (re.compile(r"doanh thu thuần|doanh thu", re.I), "kqkd", ("10", "01")),
    (re.compile(r"tổng nợ ngắn hạn|nợ ngắn hạn", re.I), "cdkt", ("310",)),
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--greedy", default="artifacts/fresh/greedy_plan.jsonl")
    parser.add_argument("--taken", default="artifacts/fresh/ratio_plan.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/ratio_greedy_plan.jsonl")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    # (ticker, year, scope, kind, period) -> code -> cell
    cells: dict[tuple, dict] = defaultdict(dict)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        for period in ("current", "prior"):
            key = (record["ticker"], record["year"], scope, record["kind"], period)
            for code, cell in record[period].items():
                cells[key].setdefault(code, {
                    "value": cell[0], "label": cell[1], "row": cell[2],
                    "col": cell[3], "csv": record["csv"], "doc": record["doc"],
                    "table_id": record["table_id"],
                    "table_ref": record["table_ref"], "scale": record["scale"],
                })

    taken = set()
    taken_path = ROOT / args.taken
    if taken_path.exists():
        for line in taken_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                taken.add(json.loads(line)["id"])

    greedy = {}
    for line in (ROOT / args.greedy).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            greedy[record["id"]] = record

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    resolver_path = Path(__file__).resolve().parent
    sys.path.insert(0, str(resolver_path))
    from plan_answers import OPENING_RE, PARENT_RE, YEAR_RE  # noqa: E402
    from resolve_ticker import TickerResolver  # noqa: E402

    resolver = TickerResolver()
    counters: Counter[str] = Counter()
    plan, samples = [], []

    for qid, text in questions.items():
        if qid in taken:
            continue
        _name, unit = unit_of(text)
        if unit:
            continue  # a money question; the single-cell path handles it
        if not (PERCENT_RE.search(text) or TIMES_RE.search(text)):
            counters["khong phai cau ty le"] += 1
            continue
        entry = greedy.get(qid)
        if entry is None:
            counters["khong co tu so"] += 1
            continue
        counters["cau ty le co tu so"] += 1

        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        if not found or not years:
            counters["khong nhan ra ma hoac nam"] += 1
            continue
        ticker = sorted(found)[0]
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        period = "prior" if OPENING_RE.search(text) else "current"

        base = None
        for pattern, kind, codes in NAMED_BASE:
            if not pattern.search(text):
                continue
            for code in codes:
                candidate = cells.get((ticker, year, scope, kind, period), {}).get(code)
                if candidate:
                    base = (kind, code, candidate)
                    break
            if base:
                counters["mau so lay tu cach noi trong cau"] += 1
                break
        if base is None:
            kind = entry["kind"] if entry["kind"] in DEFAULT_BASE else "cdkt"
            for code in DEFAULT_BASE[kind]:
                candidate = cells.get((ticker, year, scope, kind, period), {}).get(code)
                if candidate:
                    base = (kind, code, candidate)
                    break
            if base:
                counters["mau so lay MAC DINH (dong tong)"] += 1
        if base is None:
            counters["khong tim duoc mau so nao"] += 1
            continue

        base_kind, base_code, base_cell = base
        if (entry["kind"], entry["code"]) == (base_kind, base_code):
            counters["tu va mau trung nhau"] += 1
            continue
        if not base_cell["value"]:
            counters["mau bang 0"] += 1
            continue
        as_percent = bool(PERCENT_RE.search(text))
        value = (abs(entry.get("value", 0.0) or 0.0) if "value" in entry else None)
        counters["CO CAP DIA CHI (doan mau so)"] += 1
        plan.append({
            "id": qid, "op": "pct" if as_percent else "times",
            "num": {"kind": entry["kind"], "code": entry["code"],
                    "label": entry["label"], "row": entry["row"],
                    "col": entry["col"], "csv": entry["csv"], "doc": entry["doc"],
                    "table_id": entry["table_id"],
                    "table_ref": entry["table_ref"], "scale": entry["scale"]},
            "den": {"kind": base_kind, "code": base_code,
                    "label": base_cell["label"], "row": base_cell["row"],
                    "col": base_cell["col"], "csv": base_cell["csv"],
                    "doc": base_cell["doc"], "table_id": base_cell["table_id"],
                    "table_ref": base_cell["table_ref"],
                    "scale": base_cell["scale"]},
        })
        if len(samples) < args.show:
            samples.append(
                f"  id={qid:<5d} {entry['kind']}/{entry['code']} / "
                f"{base_kind}/{base_code}\n"
                f"     hoi : {text[:96]}\n"
                f"     tu  : {str(entry['label'])[:70]}\n"
                f"     mau : {str(base_cell['label'])[:70]}")
        del value

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print()
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
