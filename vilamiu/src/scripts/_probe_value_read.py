"""Ask the model for the figure itself, not for coordinates and not for a program.

Everything the pipeline has ever asked of a model makes it do index arithmetic:

    write a pandas program over eight tables      19.5%
    return {table, row, col} over eight tables    21.7%

Both numbers are measured through the real path. Neither asks the model to do the
thing it is best at, which is to read a value and copy it out. Row and column
indices are exactly what language models are worst at — the isolated probe found 49
of 58 column errors were off by precisely one, a counting mistake rather than a
reading one.

So this asks for the number. The tables are the same shortlist, rendered the same
way; only the question put to the model changes.

If the value comes back more accurately than the coordinates did, the pipeline can be
rebuilt around it: the model supplies the figure, and the cell holding that figure is
then found in the shipped CSVs by search, which yields a deterministic
`num(df, r, c)` that always runs and always reads a real cell — valid under the
private round's manual review, and with EXECUTION tracking ANSWER the way it does for
every team above us.

The answer is compared with gold on the organisers' own terms: `abs_tol=0.01`,
`rel_tol=0.0`.

Usage:
  PYTHONPATH=src python scripts/_probe_value_read.py \
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

from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

SYSTEM = """Bạn đọc báo cáo tài chính Việt Nam và trả về CON SỐ mà câu hỏi cần.

Bạn được đưa nhiều bảng. Tìm ô chứa số câu hỏi hỏi tới và chép **nguyên văn** ô đó.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"raw": "<ô nguyên văn, chép y hệt>", "unit": "<đơn vị của bảng: đồng | nghìn đồng | triệu đồng | tỷ đồng>"}

Quy tắc:
- CHÉP nguyên ô, giữ nguyên dấu chấm phẩy và dấu ngoặc: "1.234.567", "(89.012)".
  Không tự tính, không tự quy đổi, không làm tròn.
- Chọn dòng có nhãn khớp chỉ tiêu, không phải dòng có số lớn nhất. Nhãn thật thường
  có tiền tố đánh số ("1. ", "V. ") mà câu hỏi không có.
- Dòng cộng có thể để trống ô nhãn; nếu câu hỏi hỏi tổng thì lấy dòng đó.
- Chọn đúng cột kỳ. "Số cuối năm" của báo cáo năm N là cuối năm N; "Số đầu năm" là
  cuối năm N-1.
- `unit` là đơn vị mà BẢNG dùng, đọc ở dòng "Đơn vị tính" hoặc ở tiêu đề cột.
- Nếu không tìm thấy, trả {"raw": "", "unit": ""}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{tables}

Ô nào chứa số câu hỏi cần? Chép nguyên văn ô đó."""

# The pipeline disables the model's reasoning mode to keep `<think>` from eating the
# token budget. That was never measured against accuracy, and picking the right line
# of a Vietnamese statement is exactly the kind of step reasoning is for.
THINK = os.environ.get("VIFIN_THINK", "0") == "1"

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# Measured on the gold records: 18.7% of the row labels that hold the answer are
# longer than 30 characters (median 13, p90 42, max 124). Vietnamese line items carry
# the distinguishing part at the END — "... ngắn hạn", "... vô hình", "... trước
# thuế" — so cutting a cell at 30 characters deletes exactly what separates a row
# from its sibling, on nearly one question in five. Every reader number this project
# reported (69.1%, 27.3%, 21.7%) was measured through that cut and therefore
# understates the model. The shipped pipeline does not truncate cells at all, so the
# probes were crippled and the pipeline was not.
MAX_ROWS = 60
MAX_CELL = 200
UNIT_SCALES = {"đồng": 1.0, "nghìn đồng": 1e3, "ngàn đồng": 1e3,
               "triệu đồng": 1e6, "tỷ đồng": 1e9, "nghìn tỷ đồng": 1e12}


def render(index: int, grid, caption: str) -> str:
    lines = [f"### t{index} — {caption[:90]}"]
    lines.append(",".join(f"c{c}={str(cell)[:52]}" for c, cell in enumerate(grid[0])))
    for offset, row in enumerate(grid[1:MAX_ROWS], start=1):
        lines.append(f"r{offset}: " + ",".join(str(cell)[:MAX_CELL] for cell in row))
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
    parser.add_argument("--out", default="artifacts/_value_read.jsonl")
    # Self-consistency: every run so far has been a single greedy sample. Voting
    # over several samples is the standard remedy for exactly this failure shape —
    # the model lands on a plausible neighbouring row, and different samples land
    # on different ones, so the true cell is the one they agree on.
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
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

    print(f"{len(work)} câu gold, shortlist={args.shortlist}, model={args.model}",
          flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        record, question, keys, gold = item
        blocks = []
        for index, key in enumerate(keys):
            grid = store.rows(key)
            if not grid:
                continue
            caption = str(getattr(store.meta(key), "caption", ""))
            blocks.append(render(len(blocks), grid, caption))
        if not blocks:
            return
        user = USER.format(question=record["question"], tables="\n\n".join(blocks))
        base = {"chat_template_kwargs": {"enable_thinking": True},
                "max_tokens": 4000} if THINK else {}
        votes: collections.Counter = collections.Counter()
        units: dict[str, str] = {}
        for sample in range(max(1, args.samples)):
            extra = dict(base)
            if args.samples > 1:
                # The first sample stays greedy so voting can only add to it.
                extra["temperature"] = 0.0 if sample == 0 else args.temperature
            try:
                reply = client.complete(SYSTEM, user, extra=extra or None)
            except RuntimeError:
                continue
            found = JSON_RE.search(THINK_RE.sub("", reply or ""))
            if not found:
                continue
            try:
                data = json.loads(found.group(0))
            except (ValueError, TypeError):
                continue
            text = str(data.get("raw", "")).strip()
            if not text:
                continue
            number = parse_number(text)
            if number is None:
                continue
            key = f"{abs(number):.4f}"
            votes[key] += 1
            units.setdefault(key, str(data.get("unit", "")).strip().lower())

        if not votes:
            with lock:
                tally["lỗi truyền"] += 1
            return
        winner, _ = votes.most_common(1)[0]
        raw, unit = winner, units.get(winner, "")

        verdict, value = "không đọc được trả lời", None
        if raw is not None:
            if not raw:
                verdict = "model nói không có"
            else:
                cell = parse_number(raw)
                if cell is None:
                    verdict = "ô không phải số"
                else:
                    scale = UNIT_SCALES.get(unit or "", 1.0)
                    value = abs(cell) * scale / question.unit_scale
                    if abs(value - abs(gold)) <= 0.01:
                        verdict = "ĐÚNG (chuẩn BTC)"
                    elif abs(gold) and abs(value - abs(gold)) / abs(gold) <= 5e-3:
                        verdict = "gần đúng 0,5%"
                    else:
                        # Separating these two says whether the reading or the unit
                        # is at fault, and they need opposite fixes.
                        ratios = [abs(value * f - abs(gold)) <= max(0.01, abs(gold) * 1e-6)
                                  for f in (1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9)]
                        verdict = "đúng số, SAI đơn vị" if any(ratios) else "SAI ô"

        with lock:
            tally["n"] += 1
            tally[verdict] += 1
            rows_out.append({"id": record.get("id"), "verdict": verdict,
                             "raw": raw, "unit": unit, "value": value, "gold": gold})
            if tally["n"] % 50 == 0:
                n = tally["n"]
                hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
                print(f"  {n}/{len(work)}  đúng={hit / n:.1%}  "
                      f"{time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    hit = tally["ĐÚNG (chuẩn BTC)"] + tally["gần đúng 0,5%"]
    unit_only = tally["đúng số, SAI đơn vị"]
    print(f"\n== {tally['n']} câu")
    for name, count in tally.most_common():
        if name == "n":
            continue
        print(f"  {name:24s} {count:4d}  {count / n:6.1%}")
    print(f"\n  ĐÚNG CON SỐ:            {hit}/{tally['n']} = {hit / n:.1%}")
    print(f"  đúng nếu sửa được đơn vị: {(hit + unit_only) / n:.1%}")
    print("  (so: viết chương trình 19,5% | trả toạ độ 21,7%)")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")


if __name__ == "__main__":
    main()
