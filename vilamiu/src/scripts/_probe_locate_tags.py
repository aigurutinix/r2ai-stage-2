"""Can the model locate a cell through the path a submission actually takes?

This is the measurement the last two days lacked. Three probes reported large gains
and three submissions then lost, every time because the probe handed the model the
one table holding the answer. Retrieval is most of the difficulty, so an oracle
table makes the task a different task.

Here the model gets our own shortlist — the same `search_balanced` order
`run_generate` uses — renders it the way the column-tag experiment renders a table,
and returns `{table, row, col}`. Nothing else changes. The cell it names is then
converted to the unit the question asks for and compared with the gold answer, using
the organisers' own comparison: `abs_tol=0.01`, no relative slack
(`common/numeric/parsing.py:63`).

Why locate rather than generate a program: the pipeline currently asks one model call
to do two hard things at once — find the cell and write correct pandas. The isolated
probe that scored 69.1% only did the first. If localisation survives the real path at
anything like that rate, the pandas can be emitted deterministically from the
coordinates and will always run.

The two branches that already work this way are measured at 13.3% (`locate`) and ~21%
(`plan`), and neither was ever given the tagged rendering. That gap is what this
measures.

Usage:
  VIFIN_LOC_TAGS=1 PYTHONPATH=src python scripts/_probe_locate_tags.py \
      --model Qwen/Qwen3-14B --local-url http://127.0.0.1:18000/v1 --limit 300
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import parse_number  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

TAGS = os.environ.get("VIFIN_LOC_TAGS", "1") != "0"

SYSTEM = """Bạn định vị MỘT Ô trong báo cáo tài chính Việt Nam.

Bạn được đưa nhiều bảng, đánh số t0, t1, t2, … Trong mỗi bảng, dòng đánh r0, r1, …
(r0 là dòng tiêu đề) và cột đánh c0, c1, … ngay trên dòng tiêu đề.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"table": <chỉ số bảng>, "row": <chỉ số dòng>, "col": <chỉ số cột>}

Quy tắc:
- Chọn bảng chứa chỉ tiêu câu hỏi hỏi tới. Nhiều bảng cùng công ty cùng năm; chỉ một
  bảng có đúng chỉ tiêu đó.
- Chọn dòng có nhãn khớp chỉ tiêu, không phải dòng có số lớn nhất. Nhãn thật thường
  có tiền tố đánh số ("1. ", "V. ") mà câu hỏi không có.
- Dòng cộng có thể để trống ô nhãn; nếu câu hỏi hỏi tổng thì đó là dòng cần.
- Bỏ qua cột "Mã số" và "Thuyết minh" — chúng không chứa giá trị.
- Nếu không bảng nào có chỉ tiêu đó, trả {"table": -1, "row": -1, "col": -1}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{tables}

Ô nào chứa số câu hỏi cần?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
MAX_ROWS = 60
MAX_CELL = 200
MAX_HEADER_CELL = 120


def render_one(index: int, key: TableKey, grid, caption: str) -> str:
    """One table, with the column index bound to its header text.

    The binding lives on the header line rather than in a separate ruler: a ruler's
    commas do not line up with the body, and the isolated probe showed the model
    then counts data columns and skips the label column — 49 of 58 column errors
    were exactly -1.
    """

    lines = [f"### t{index} — {caption[:90]}"]
    if TAGS:
        lines.append(",".join(f"c{c}={str(cell)[:MAX_HEADER_CELL]}"
                              for c, cell in enumerate(grid[0])))
        body = grid[1:MAX_ROWS]
        start = 1
    else:
        body = grid[:MAX_ROWS]
        start = 0
    for offset, row in enumerate(body):
        cells = ",".join(str(cell)[:MAX_CELL] for cell in row)
        lines.append(f"r{offset + start}: {cells}")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--out", default="artifacts/_locate_tags.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)
    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))

    work = []
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(work) >= args.limit:
            continue
        record = json.loads(line)
        gold = parse_number(record.get("answer"))
        if gold is None:
            continue
        question = parse_question(record.get("id", 0), record["question"], roster)
        if not question.unit_scale:
            # A ratio or a count is not one cell, so it cannot score this way.
            continue
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        if not keys:
            continue
        work.append((record, question, keys, gold))

    print(f"{len(work)} câu gold, shortlist={args.shortlist}, "
          f"tags={'bật' if TAGS else 'tắt'}, model={args.model}", flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        record, question, keys, gold = item
        blocks, grids, contexts = [], [], []
        for index, key in enumerate(keys):
            grid = store.rows(key)
            if not grid:
                continue
            meta = store.meta(key)
            caption = str(getattr(meta, "caption", ""))
            blocks.append(render_one(len(grids), key, grid, caption))
            grids.append(grid)
            contexts.append(f"{getattr(meta, 'unit_page', '')} "
                            f"{getattr(meta, 'unit_doc', '')} {caption}")
        if not grids:
            return
        user = USER.format(question=record["question"], tables="\n\n".join(blocks))
        try:
            reply = client.complete(SYSTEM, user)
        except RuntimeError:
            with lock:
                tally["lỗi truyền"] += 1
            return

        match = JSON_RE.search(reply or "")
        table = row = column = None
        if match:
            try:
                data = json.loads(match.group(0))
                table = int(data.get("table", -1))
                row = int(data.get("row", -1))
                column = int(data.get("col", -1))
            except (ValueError, TypeError):
                table = None

        verdict = "không đọc được trả lời"
        value = None
        if table is not None:
            if table < 0 or row < 0 or column < 0:
                verdict = "model nói không có"
            elif table >= len(grids):
                verdict = "chỉ số bảng ngoài phạm vi"
            else:
                grid = grids[table]
                if row >= len(grid) or column >= len(grid[row]):
                    verdict = "ô ngoài phạm vi"
                else:
                    raw = parse_number(grid[row][column])
                    if raw is None:
                        verdict = "ô không phải số"
                    else:
                        scale = lookup_mod.column_scale(grid, column, contexts[table])
                        value = abs(raw) * scale / question.unit_scale
                        # The organisers compare with abs_tol=0.01 and no relative
                        # slack, so that is the bar. The looser band is reported
                        # alongside to separate "wrong cell" from "right cell,
                        # scale off".
                        if abs(value - abs(gold)) <= 0.01:
                            verdict = "ĐÚNG (chuẩn BTC)"
                        elif abs(gold) and abs(value - abs(gold)) / abs(gold) <= 5e-3:
                            verdict = "gần đúng 0,5%"
                        else:
                            verdict = "SAI ô"

        with lock:
            tally["n"] += 1
            tally[verdict] += 1
            rows_out.append({"id": record.get("id"), "verdict": verdict,
                             "gold": gold, "value": value,
                             "table": table, "row": row, "col": column})
            if tally["n"] % 50 == 0:
                n = tally["n"]
                hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
                print(f"  {n}/{len(work)}  đúng={hit / n:.1%}  "
                      f"{time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} câu, shortlist {args.shortlist} bảng, "
          f"tags={'bật' if TAGS else 'tắt'}")
    for name, count in tally.most_common():
        if name == "n":
            continue
        print(f"  {name:26s} {count:4d}  {count / n:6.1%}")
    hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
    print(f"\n  ĐỘ ĐÚNG QUA ĐƯỜNG THẬT: {hit}/{tally['n']} = {hit / n:.1%}")
    print("  (so: locate hiện tại 13,3% | plan ~21% | lookup 42,8% | "
          "probe oracle một bảng 69,1%)")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")


if __name__ == "__main__":
    main()
