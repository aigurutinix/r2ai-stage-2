"""Turn a table-reading reply into an address, keeping only the ones that check out.

The reply names a table, a row, a column AND copies the cell. Those four facts are
redundant, and the redundancy is the whole point: code opens that csv, reads that cell,
and compares it with what the model claims is there. A model that guessed an address
quotes a figure the cell does not hold, and the question is dropped instead of answered.

That is the check every earlier model pass here lacked. Those passes asked for a row
label or a Mã số, which cannot be verified against anything — a plausible-looking wrong
answer was indistinguishable from a right one, and three of them cost 0.32 to 0.40
points per row spliced.

The scale is taken where the cell lives, nearest first: the table's own header row, then
the prose printed above the table in the report, then the document. A cell whose scale
cannot be established is dropped rather than guessed, because the unit multiplies into
the answer and a wrong one is wrong by three orders of magnitude.

Usage:
  python scripts/fresh/plan_tab.py --replies artifacts/fresh/replies_tab.jsonl
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
from document_scale import scale_of_line  # noqa: E402
from score_model import parse  # noqa: E402

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
CONTEXT_CHARS = 420


def contexts_of(text_path: Path) -> dict[int, str]:
    try:
        body = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous_end = {}, 0
    for match in ANCHOR_RE.finditer(body):
        lead = re.sub(r"\s+", " ", body[previous_end:match.start()].strip())
        out[int(match.group(1))] = lead[-CONTEXT_CHARS:]
        previous_end = match.end()
    return out


def scale_of_context(context: str) -> float | None:
    """The nearest declared unit in the prose above the table, if any."""

    for piece in reversed(re.split(r"(?<=[:.)])\s+", context)):
        scale = scale_of_line(piece)
        if scale is not None:
            return scale
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", required=True)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_tab.jsonl")
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    parser.add_argument("--out", default="artifacts/fresh/tab_plan.jsonl")
    parser.add_argument("--allow-unquoted", action="store_true",
                        help="keep addresses whose quoted figure does not match")
    args = parser.parse_args()

    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    doc_scale = {}
    path = ROOT / args.doc_scale
    if path.exists():
        doc_scale = {k: float(v) for k, v
                     in json.loads(path.read_text(encoding="utf-8")).items()}

    counters: Counter[str] = Counter()
    grid_cache: dict[Path, list[list[str]]] = {}
    context_cache: dict[str, dict[int, str]] = {}
    plan = []

    for line in (ROOT / args.replies).read_text(encoding="utf-8").splitlines():
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
        try:
            table_id = int(spec.get("bang"))
            row = int(spec.get("dong"))
            column = int(spec.get("cot"))
        except (TypeError, ValueError):
            counters["dia chi khong phai so"] += 1
            continue
        quoted = str(spec.get("so", "")).strip()
        if table_id < 0 or row < 0 or column < 0 or not quoted:
            counters["model noi khong co o nao"] += 1
            continue

        offered = {t["table_id"]: t["doc"] for t in info["tables"]}
        if table_id not in offered:
            counters["bang khong nam trong so duoc dua"] += 1
            continue
        doc = offered[table_id]
        csv_rel = (f"data/official_corpus/{info['ticker']}/{info['year']}/{doc}/"
                   f"{doc}_extracted_tables/table_{table_id}.csv")
        csv_path = ROOT / csv_rel
        if csv_path not in grid_cache:
            try:
                with csv_path.open(encoding="utf-8-sig", newline="") as file:
                    grid_cache[csv_path] = list(csv_mod.reader(file))
            except OSError:
                grid_cache[csv_path] = []
        grid = grid_cache[csv_path]
        # The model was shown rows starting after the header, so row r of the reply is
        # grid row r+1 — the same offset the scorer's frame uses.
        if row + 1 >= len(grid) or column >= len(grid[row + 1]):
            counters["dia chi nam ngoai bang"] += 1
            continue
        actual = str(grid[row + 1][column]).strip()

        claimed = ps.parse_vn_number(quoted)
        present = ps.parse_vn_number(actual)
        if claimed is None or present is None or abs(claimed - present) > 0.01:
            counters["TRICH DAN SAI — bo cau nay"] += 1
            if not args.allow_unquoted:
                continue
        else:
            counters["trich dan dung"] += 1

        scale = ps.scale_from_unit_text(",".join(grid[0])) if grid else None
        source = "tieu de bang"
        if scale is None:
            if doc not in context_cache:
                context_cache[doc] = contexts_of(
                    ROOT / "data" / "official_corpus" / info["ticker"] /
                    info["year"] / doc / f"{doc}_extracted.txt")
            scale = scale_of_context(context_cache[doc].get(table_id, ""))
            source = "van ban tren bang"
        if scale is None:
            scale = doc_scale.get(doc)
            source = "ca tai lieu"
        if scale is None:
            counters["  khong biet he so — bo"] += 1
            continue
        counters[f"  he so tu {source}"] += 1

        label = str(grid[row + 1][0]).strip() if grid[row + 1] else ""
        plan.append({
            "id": record["id"], "source": "maso", "kind": "bang",
            "code": f"t{table_id}", "label": label,
            "row": row, "col": column, "csv": csv_rel, "doc": doc,
            "table_id": table_id, "table_ref": f"{doc}|table_{table_id}",
            "scale": scale, "score": 1.0, "picked_by": "table_reader",
            "quoted": quoted,
        })

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\ntong dia chi da kiem chung: {len(plan)}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
