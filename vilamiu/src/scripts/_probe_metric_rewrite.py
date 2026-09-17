"""Let the model rewrite the metric; let the label matcher find the row.

Two numbers decide the division of labour, and both are measured on the same corpus
through the same path:

* asked to locate a cell among our eight retrieved tables, the model is right 21.7%
  of the time — better than the current `locate` branch at 13.3%, but half the
  lexical label matcher's 42.8%. Choosing among tables is what defeats it.
* the label matcher, when it commits, takes the right cell 74.2% of the time. It only
  commits on 29.1% of questions: on the other 40.5% no label scores above the 0.75
  floor at all. Coverage, not accuracy, is its binding constraint.

So the model is asked for the thing it is actually good at — language. The organisers'
generator explicitly rejects questions that copy a row label verbatim
(`generation/common.py:436-439`), so the gap between "chi phí trả trước ngắn hạn" as a
question phrases it and as a balance sheet writes it is a translation problem, not a
navigation problem. The model proposes several candidate labels in statement wording;
the matcher scores each against the real tables and keeps the best.

What this measures is whether coverage rises without accuracy collapsing.

Usage:
  PYTHONPATH=src python scripts/_probe_metric_rewrite.py \
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

# 0 disables the rewrite and scores the matcher on the question's own phrasing, which
# is the baseline this has to beat.
REWRITE = os.environ.get("VIFIN_REWRITE", "1") != "0"

SYSTEM = """Bạn viết lại tên chỉ tiêu tài chính theo đúng cách BÁO CÁO TÀI CHÍNH
Việt Nam đặt tên dòng, để tra bảng.

Câu hỏi dùng lời tự nhiên; báo cáo dùng thuật ngữ chuẩn mực. Ví dụ:
- "tiền mặt cuối kỳ" -> "Tiền và các khoản tương đương tiền"
- "lãi ròng" -> "Lợi nhuận sau thuế thu nhập doanh nghiệp"
- "nợ ngắn hạn" -> "Nợ ngắn hạn"
- "vốn góp của chủ sở hữu" -> "Vốn góp của chủ sở hữu"

Trả về ĐÚNG một mảng JSON gồm 3 nhãn, không kèm gì khác, xếp theo độ tin cậy giảm:
["nhãn 1", "nhãn 2", "nhãn 3"]

Quy tắc:
- Chỉ viết TÊN CHỈ TIÊU. Không kèm tên công ty, năm, đơn vị, hay chữ "của".
- Giữ đúng các cặp phân biệt: hữu hình/vô hình, ngắn hạn/dài hạn, trước thuế/sau
  thuế, phải thu/phải trả. Sai một chữ là sai dòng khác hẳn.
- Nhãn 1 sát nghĩa nhất; nhãn 2 và 3 là cách viết khác hoặc ngắn hơn để dự phòng."""

USER = """Câu hỏi: {question}

Chỉ tiêu mà câu hỏi hỏi tới, viết theo cách báo cáo tài chính đặt tên dòng?"""

ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def candidates(client, question_text: str) -> list[str]:
    try:
        reply = client.complete(SYSTEM, USER.format(question=question_text))
    except RuntimeError:
        return []
    match = ARRAY_RE.search(THINK_RE.sub("", reply or ""))
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    return [str(item).strip() for item in items
            if isinstance(item, str) and item.strip()][:3]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--out", default="artifacts/_rewrite.jsonl")
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
            continue
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        if keys:
            work.append((record, question, keys, gold))

    print(f"{len(work)} câu gold, shortlist={args.shortlist}, "
          f"rewrite={'bật' if REWRITE else 'tắt'}, ngưỡng={lookup_mod.MIN_LABEL_SCORE}",
          flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        record, question, keys, gold = item
        phrases = [lookup_mod.extract_metric(record["question"])]
        if REWRITE:
            # The question's own phrasing stays first: where it already matches, the
            # rewrite must not be allowed to displace it.
            phrases += [p for p in candidates(client, record["question"])
                        if p not in phrases]

        best = None  # (score, value)
        for key in keys:
            grid = store.rows(key)
            if not grid:
                continue
            meta = store.meta(key)
            context = (f"{getattr(meta, 'unit_page', '')} "
                       f"{getattr(meta, 'unit_doc', '')} "
                       f"{getattr(meta, 'caption', '')}")
            for phrase in phrases:
                found = lookup_mod.match_row(grid, phrase)
                if found is None:
                    continue
                row, score, _ = found
                if best is not None and score <= best[0]:
                    continue
                # Take the first numeric cell to the right of the label, scaled the
                # way the shipped pipeline scales it.
                for column in range(1, len(grid[row])):
                    raw = parse_number(grid[row][column])
                    if raw is None or raw == 0:
                        continue
                    scale = lookup_mod.column_scale(grid, column, context)
                    best = (score, abs(raw) * scale / question.unit_scale)
                    break

        with lock:
            tally["n"] += 1
            if best is None:
                tally["không có ứng viên"] += 1
            else:
                tally["có cam kết"] += 1
                if abs(best[1] - abs(gold)) <= 0.01:
                    tally["ĐÚNG (chuẩn BTC)"] += 1
                elif abs(gold) and abs(best[1] - abs(gold)) / abs(gold) <= 5e-3:
                    tally["gần đúng 0,5%"] += 1
                else:
                    tally["sai ô"] += 1
            rows_out.append({"id": record.get("id"),
                             "value": best[1] if best else None, "gold": gold,
                             "phrases": phrases})
            if tally["n"] % 50 == 0:
                n = tally["n"]
                hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
                print(f"  {n}/{len(work)}  đúng={hit / n:.1%}  "
                      f"phủ={tally['có cam kết'] / n:.1%}  "
                      f"{time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    commit = max(tally["có cam kết"], 1)
    hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
    print(f"\n== {tally['n']} câu, rewrite={'bật' if REWRITE else 'tắt'}")
    for name, count in tally.most_common():
        if name == "n":
            continue
        print(f"  {name:24s} {count:4d}  {count / n:6.1%}")
    print(f"\n  ĐỘ PHỦ    {tally['có cam kết']}/{tally['n']} = {commit / n:.1%}")
    print(f"  ĐÚNG|phủ  {hit}/{commit} = {hit / commit:.1%}")
    print(f"  ĐÚNG toàn {hit}/{tally['n']} = {hit / n:.1%}")
    print("  (so: model tự định vị 21,7% | lookup trên bảng gold 42,8%)")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")


if __name__ == "__main__":
    main()
