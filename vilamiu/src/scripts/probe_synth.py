"""Would synthetic training data look like the real questions, or like the easy half?

Supervision is free here: pick any cell in the corpus and the answer and the
program that reads it are both constructible without a model. The missing piece
is the *question*, and that is where the whole idea lives or dies.

The organisers' generator rejects any question that copies a row or column label
(`judge_and_maybe_rewrite`), which is why our label matcher tops out at 42%. If
we write questions straight from the labels, we manufacture exactly the easy half
the matcher already solves, and a model trained on them learns nothing about the
hard half. So the questions have to be paraphrased under the same constraint.

This measures whether that worked, by the only test that matters: run our own
matcher on the synthetic questions. Real questions score ~42%.

    ~90%  the paraphrase stayed too close to the label -> data is useless, stop
    ~40%  the distribution matches the real one -> the idea is worth the compute

Usage:
  PYTHONPATH=src python scripts/probe_synth.py --count 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SYSTEM = """Bạn viết câu hỏi tài chính tiếng Việt cho một bộ dữ liệu kiểm thử.

Cho một dòng trong báo cáo tài chính, hãy viết MỘT câu hỏi tự nhiên mà đáp án
đúng là con số ở dòng đó.

RÀNG BUỘC BẮT BUỘC:
- KHÔNG được chép nguyên văn nhãn dòng. Diễn đạt lại bằng ngôn ngữ tài chính mà
  một nhà phân tích sẽ dùng khi nói, không phải bằng chữ in trong bảng.
- Nêu tên công ty và năm.
- Nêu đơn vị mong muốn (tỷ đồng / triệu đồng / %).
- Chỉ trả về đúng một câu hỏi, không giải thích, không dấu ngoặc kép."""

USER = """<công_ty>{company} ({ticker})</công_ty>
<năm>{year}</năm>
<nhãn_dòng>{label}</nhãn_dòng>
<ngữ_cảnh_bảng>{caption}</ngữ_cảnh_bảng>

Viết câu hỏi, nhớ KHÔNG chép nhãn dòng."""

NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")


def usable_rows(grid: list[list[str]]) -> list[int]:
    """Rows with a wordy label and at least one large figure beside it."""

    out = []
    for index, row in enumerate(grid[1:], start=1):
        if not row:
            continue
        label = str(row[0]).strip()
        if len(label) < 12 or len(label.split()) < 3:
            continue
        if not any(NUMERIC_RE.match(str(c).strip()) for c in row[1:]):
            continue
        out.append(index)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument("--out", default="artifacts/synth_probe.jsonl")
    args = parser.parse_args()

    random.seed(42)
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame
    frame = frame[frame.eligible]
    stock = {}
    import pandas as pd
    for row in pd.read_csv(ROOT / "data" / "code_stock.csv").itertuples(index=False):
        stock[str(row[0]).strip().upper()] = str(row[1]).strip()

    picks = []
    order = list(range(len(frame)))
    random.shuffle(order)
    for position in order:
        if len(picks) >= args.count:
            break
        meta = frame.iloc[position]
        grid = json.loads(meta.rows_json)
        rows = usable_rows(grid)
        if not rows:
            continue
        chosen = random.choice(rows)
        picks.append({
            "key": TableKey(str(meta.doc_name), int(meta.table_id)),
            "row": chosen,
            "label": str(grid[chosen][0]).strip(),
            "ticker": str(meta.ticker),
            "year": str(meta.year),
            "caption": str(meta.caption)[:110],
        })
    print(f"{len(picks)} ô mẫu từ corpus")

    client = ChatClient.from_env(ROOT, model=args.model, max_tokens=160)
    questions = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl", ROOT / "data" / "code_stock.csv")
    template = questions[0]

    def ask(item):
        user = USER.format(
            company=stock.get(item["ticker"].upper(), item["ticker"]),
            ticker=item["ticker"], year=item["year"],
            label=item["label"], caption=item["caption"])
        try:
            reply = client.complete(SYSTEM, user)
        except RuntimeError:
            return None
        text = reply.strip().splitlines()
        text = [t for t in text if t.strip() and not t.strip().startswith("<")]
        if not text:
            return None
        return {**item, "question": text[-1].strip().strip('"')}

    started = time.time()
    made = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for position, result in enumerate(pool.map(ask, picks), start=1):
            if result:
                made.append(result)
            if position % 25 == 0:
                print(f"  {position}/{len(picks)}  {time.time() - started:.0f}s")
    print(f"{len(made)} câu hỏi sinh được\n")

    # The decisive measurement: does our own matcher solve them?
    solved = copied = 0
    out_path = ROOT / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for item in made:
            probe = dataclasses.replace(
                template, question=item["question"],
                tickers=[item["ticker"]], years=[int(item["year"])])
            grid = store.rows(item["key"])
            found = lookup_mod.find(grid, probe)
            hit = (found is not None and found.row == item["row"]
                   and found.score >= lookup_mod.MIN_LABEL_SCORE)
            solved += hit
            # Did the paraphrase simply reuse the label?
            overlap = lookup_mod.match_row([[""], [item["label"]]], item["question"])
            copied += bool(overlap and overlap[1] >= 0.75)
            handle.write(json.dumps({**item, "key": [item["key"].doc_name, item["key"].table_id],
                                     "solved": bool(hit)}, ensure_ascii=False) + "\n")

    n = len(made) or 1
    print(f"bộ khớp nhãn giải đúng ô: {solved}/{n} = {solved / n:.1%}")
    print(f"câu hỏi vẫn chép nhãn   : {copied}/{n} = {copied / n:.1%}")
    print()
    print("  ~90% giải được -> dữ liệu quá dễ, train vô ích")
    print("  ~40% giải được -> phân bố giống câu thật, đáng đầu tư")
    print(f"\nđã ghi {out_path}")


if __name__ == "__main__":
    main()
