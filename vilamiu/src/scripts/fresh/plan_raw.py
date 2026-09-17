"""Find the cell the model read, so a hallucinated figure cannot be shipped.

The model reads html text and returns the figure as printed plus the row label. Its
position in that text cannot be mapped to the organisers' csv numbering, so the figure is
located again: every csv of the document is scanned for a row whose label matches and a
cell whose parsed value equals what the model reported.

That relocation is the guarantee. A figure the model invented sits in no cell, matches
nothing, and the question is dropped rather than answered — the same property every path
here has kept, and the one the first cell-picking experiment lacked.

The unit is then read off the cell's own table, not inferred: the model was told to copy
the figure as printed, so the conversion to the question's unit is
`raw × table scale / question unit` with the scale taken where the cell actually lives.

Usage:
  python scripts/fresh/plan_raw.py --replies artifacts/fresh/replies_raw.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from score_model import parse  # noqa: E402

# Where a cell's unit comes from: its own table header, the page's unit line, the
# document's statements, the document-wide text scan. First hit wins.
PERIOD_HINT = re.compile(r"đầu năm|đầu kỳ|1/1/", re.I)


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def label_matches(reported: str, actual: str) -> bool:
    """The model was told to copy the label, so this is forgiving of OCR only."""

    left, right = fold(reported), fold(actual)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    a, b = set(left.split()), set(right.split())
    return bool(a & b) and len(a & b) / min(len(a), len(b)) >= 0.6


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", required=True)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_raw.jsonl")
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    parser.add_argument("--out", default="artifacts/fresh/raw_plan.jsonl")
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    doc_modal: dict[str, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            doc_modal[record["doc"]][record["scale"]] += 1
    modal = {doc: c.most_common(1)[0][0] for doc, c in doc_modal.items()}
    scanned = {}
    scale_path = ROOT / args.doc_scale
    if scale_path.exists():
        scanned = {doc: float(scale) for doc, scale
                   in json.loads(scale_path.read_text(encoding="utf-8")).items()}

    counters: Counter[str] = Counter()
    plan, samples = [], []
    started = time.time()
    tables_cache: dict[str, list[tuple[int, list[list[str]]]]] = {}

    for number, line in enumerate(
            (ROOT / args.replies).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        info = meta.get(record["id"])
        if info is None:
            continue
        spec = parse(record["reply"])
        if not spec:
            counters["khong doc duoc JSON"] += 1
            continue
        printed = str(spec.get("so", "")).strip()
        row_label = str(spec.get("nhan_dong", "")).strip()
        if not printed:
            counters["model noi khong tim thay"] += 1
            continue
        target = ps.parse_vn_number(printed)
        if target is None:
            counters["con so khong parse duoc"] += 1
            continue

        doc = info["doc"]
        if doc not in tables_cache:
            # A few documents at a time: a report is hundreds of csv files and holding
            # every one parsed at once exhausts memory.
            if len(tables_cache) >= 4:
                tables_cache.clear()
            base = (ROOT / "data" / "official_corpus" / info["ticker"] /
                    info["year"] / doc / f"{doc}_extracted_tables")
            rows = []
            if base.is_dir():
                for path in sorted(base.glob("table_*.csv")):
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as file:
                            rows.append((int(path.stem.split("_")[-1]),
                                         [r for r in csv_mod.reader(file)]))
                    except OSError:
                        continue
            tables_cache[doc] = rows
        tables = tables_cache[doc]
        if not tables:
            counters["khong co csv cho tai lieu"] += 1
            continue

        # Find the cell: same value, and a label the model's label agrees with.
        best = None
        for table_id, grid in tables:
            if not grid:
                continue
            scale = ps.scale_from_unit_text(",".join(grid[0]))
            for row_index, row in enumerate(grid[1:]):
                label = max((str(c).strip() for c in row
                             if not any(ch.isdigit() for ch in str(c))),
                            key=len, default="")
                for column, cell in enumerate(row):
                    raw = str(cell).strip()
                    if not raw or ps.BARE_INT_RE.match(raw):
                        continue
                    value = ps.parse_vn_number(raw)
                    if value is None or abs(abs(value) - abs(target)) > 0.01:
                        continue
                    agrees = label_matches(row_label, label) if row_label else False
                    rank = (0 if agrees else 1)
                    if best is None or rank < best[0]:
                        best = (rank, table_id, row_index, column, label,
                                scale, grid)
                    if rank == 0:
                        break
                if best is not None and best[0] == 0:
                    break
            if best is not None and best[0] == 0:
                break
        if best is None:
            counters["KHONG tim lai duoc o — bo cau nay"] += 1
            continue
        rank, table_id, row_index, column, label, scale, _grid = best
        counters["tim lai duoc, nhan KHOP" if rank == 0
                 else "tim lai duoc, nhan khong khop"] += 1
        if scale is None:
            scale = modal.get(doc) or scanned.get(doc)
        if scale is None:
            counters["  nhung khong biet he so — bo"] += 1
            continue

        plan.append({
            "id": record["id"], "source": "maso", "kind": "van ban tho",
            "code": f"t{table_id}", "period":
                "prior" if PERIOD_HINT.search(info["question"]) else "current",
            "label": label, "row": row_index, "col": column,
            "csv": str((Path("data") / "official_corpus" / info["ticker"] /
                        info["year"] / doc /
                        f"{doc}_extracted_tables" /
                        f"table_{table_id}.csv")).replace("\\", "/"),
            "doc": doc, "table_id": table_id,
            "table_ref": f"{doc}|table_{table_id}", "scale": scale,
            "score": 1.0 if rank == 0 else 0.5, "picked_by": "raw_text",
            "printed": printed,
        })
        if len(samples) < args.show:
            samples.append(
                f"  id={record['id']:<5d} t{table_id} dong={row_index} cot={column} "
                f"{'nhan khop' if rank == 0 else 'chi khop gia tri'}\n"
                f"     hoi   : {info['question'][:88]}\n"
                f"     model : {printed}  |  {row_label[:52]}\n"
                f"     csv   : {label[:60]}")
        if number % 200 == 0:
            print(f"  {number}  {time.time() - started:.0f}s", flush=True)

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\ntong dia chi: {len(plan)}")
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
