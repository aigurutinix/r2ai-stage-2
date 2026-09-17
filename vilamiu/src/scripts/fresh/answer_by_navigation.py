"""Answer the questions the navigation resolves to exactly one cell, and only those.

The chain: the spec names the indicator, the ranking picks the row, the note pointer picks
the table, the column comes from the period the question asks for, the scale comes from the
table, and the arithmetic is one division. No model chooses a cell and no model converts a
unit — the first because the ranking already left one candidate, the second because letting
a scale decision live in the wrong place is what cost `vote7` twenty-four questions.

Deliberately narrow. A question is answered only when all of these hold:

  * exactly one candidate row survives the ranking;
  * the operation is `single` — a difference or a ratio needs a second operand and that
    is a separate piece of work;
  * the table's own scale is known, from its header or the prose above it. Where it is
    not, the question is left alone rather than answered in the wrong unit;
  * the period maps to a column.

Everything else keeps whatever the incumbent said. Coverage is the thing that suffers from
being strict here, and coverage is recoverable; a wrong unit is not.

Usage:
  python scripts/fresh/answer_by_navigation.py --specs artifacts/fresh/spec_replies.jsonl
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
from find_statements import locate_columns  # noqa: E402
from fix_units import contexts_of, declared_scale  # noqa: E402
from label_match import score  # noqa: E402
from table_reading import (money_columns, row_label, scale_from_headers,
                           section_label)  # noqa: E402
from measure_pointers import POINTER_RE, heading_numbers  # noqa: E402
from measure_unique import variants_from  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

CODE_RE = re.compile(r"^\d{1,3}$")
# Which value column a period wants. A statement prints the current period first and the
# prior one second, which is the convention the whole corpus follows.
PERIOD_SLOT = {"cuoi_nam": 0, "nam_nay": 0, "dau_nam": 1, "nam_truoc": 1}


def spec_fields(path: Path) -> dict[int, dict]:
    """Operation and period per question, beside the label variants."""

    from score_model import parse as parse_json

    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        spec = parse_json(record.get("reply", ""))
        if spec:
            out[record["id"]] = spec
    return out


def value_slots(row: list[str], skip: set[int]) -> list[tuple[int, float]]:
    """The figures in a row, left to right, ignoring the code and pointer columns."""

    out = []
    for index, cell in enumerate(row):
        if index in skip:
            continue
        raw = str(cell).strip()
        if not raw or CODE_RE.match(raw):
            continue
        value = ps.parse_vn_number(raw)
        if value is not None:
            out.append((index, value))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="artifacts/fresh/spec_replies.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/nav_answers.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    # The document-wide unit scan, as a third source after the table header and the
    # prose above it. Skipping it left 38 of 120 questions unanswered for want of a
    # unit that had already been measured elsewhere at 98% on the documents it covers.
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    args = parser.parse_args()

    doc_scale: dict[str, float] = {}
    scale_path = ROOT / args.doc_scale
    if scale_path.exists():
        doc_scale = {k: float(v) for k, v
                     in json.loads(scale_path.read_text(encoding="utf-8")).items()}

    variants = variants_from(ROOT / args.specs)
    specs = spec_fields(ROOT / args.specs)
    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    counters: Counter[str] = Counter()
    answers = []

    for question in questions:
        qid = question["id"]
        names = variants.get(qid)
        spec = specs.get(qid) or {}
        if not names:
            counters["khong co spec"] += 1
            continue
        if (spec.get("phep_tinh") or "").strip() != "single":
            counters["khong phai phep tinh don"] += 1
            continue
        text = question["question"]
        _name, unit = unit_of(text)
        if not unit:
            counters["khong ro don vi cau hoi"] += 1
            continue
        tickers = sorted(resolver.resolve(text))
        years = sorted({y for y in YEAR_RE.findall(text)})
        if not tickers or not years:
            counters["khong xac dinh duoc ma/nam"] += 1
            continue
        base = ROOT / "data" / "official_corpus" / tickers[0] / years[-1]
        if not base.is_dir():
            counters["khong co tai lieu"] += 1
            continue

        want_separate = bool(PARENT_RE.search(text))
        docs = sorted(p for p in base.iterdir() if p.is_dir())
        docs = ([d for d in docs if ("separate" in d.name) == want_separate]
                + [d for d in docs if ("separate" in d.name) != want_separate])

        # candidate = (score, pointed, table_id, row_index, csv_path, grid, skip cols)
        found = []
        for doc_dir in docs:
            tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
            if not tables_dir.is_dir():
                continue
            grids, paths = {}, {}
            for csv_path in sorted(tables_dir.glob("table_*.csv")):
                table_id = int(csv_path.stem.split("_")[-1])
                try:
                    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                        grids[table_id] = list(csv_mod.reader(handle))
                    paths[table_id] = csv_path
                except OSError:
                    continue
            anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")
            by_number: dict[str, list[int]] = {}
            for table_id in grids:
                for number in heading_numbers(anchors.get(table_id, "")):
                    by_number.setdefault(number, []).append(table_id)

            pointed: dict[int, str] = {}
            for table_id, grid in grids.items():
                if not grid:
                    continue
                code_col, pointer_col = locate_columns(grid)
                if code_col is None:
                    continue
                skip = {code_col} | ({pointer_col} if pointer_col is not None else set())
                for index, row in enumerate(grid[1:]):
                    if len(row) <= code_col:
                        continue
                    if not CODE_RE.match(str(row[code_col]).strip()):
                        continue
                    pointer = ""
                    if pointer_col is not None and len(row) > pointer_col:
                        candidate = str(row[pointer_col]).strip()
                        if POINTER_RE.match(candidate):
                            pointer = re.sub(r"[\s.]", "", candidate).upper()
                    value = score(names, row_label(row, skip))
                    if value:
                        found.append((value, bool(pointer), table_id, index,
                                      paths[table_id], grid, skip,
                                      anchors.get(table_id, "")))
                    if pointer:
                        for target in by_number.get(pointer, []):
                            pointed.setdefault(target, pointer)

            for table_id in pointed:
                grid = grids.get(table_id) or []
                for index, row in enumerate(grid[1:]):
                    if not row:
                        continue
                    value = score(names, section_label(grid, index, set()))
                    if value:
                        found.append((value, True, table_id, index, paths[table_id],
                                      grid, set(), anchors.get(table_id, "")))
            if found:
                break

        if not found:
            counters["khong co ung vien"] += 1
            continue
        top = max(item[0] for item in found)
        best = [item for item in found if item[0] == top]
        pointed_best = [item for item in best if item[1]]
        best = pointed_best or best
        seen = {(item[2], item[3]) for item in best}
        if len(seen) != 1:
            counters[f"con {min(len(seen), 4)} ung vien — bo qua"] += 1
            continue

        _s, _p, table_id, row_index, csv_path, grid, skip, context = best[0]
        scale = scale_from_headers(grid)
        scale_from = "tieu de cot"
        if scale is None:
            scale = declared_scale(grid, context)
            scale_from = "van ban tren bang"
        if scale is None:
            scale = doc_scale.get(csv_path.parent.parent.name)
            scale_from = "ca tai lieu"
        if scale is None:
            counters["chua biet he so — bo qua"] += 1
            continue
        allowed = money_columns(grid)
        slots = value_slots(grid[row_index + 1], skip)
        if allowed is not None:
            money_slots = [(c, v) for c, v in slots if c in allowed]
            if money_slots:
                slots = money_slots
        if not slots:
            counters["dong khong co so"] += 1
            continue
        wanted = PERIOD_SLOT.get((spec.get("ky") or "").strip(), 0)
        if wanted >= len(slots):
            wanted = 0
        column, raw_value = slots[wanted]
        # The sign is kept. Taking the magnitude was a decision copied from an older
        # mechanism and it is wrong here: a negative cash flow is a fact, and the
        # organisers' own rule takes an absolute value only for a difference whose
        # direction the question left unstated. `single` states no difference at all.
        value = raw_value * scale / unit
        if not value:
            # A zero came from an empty cell in this column, not from a statement that
            # the figure is zero. A blank and a wrong answer score the same, so there is
            # nothing to gain by shipping it.
            counters["ket qua bang 0 — bo qua"] += 1
            continue
        answers.append({
            "id": qid,
            "answer": round(value, 2),
            "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
            "row": row_index, "col": column, "scale": scale, "unit": unit,
        })
        counters[f"TRA LOI (he so tu {scale_from})"] += 1

    (ROOT / args.out).write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
