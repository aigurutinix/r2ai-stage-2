"""Ask the model for the number, not for a program.

The leaderboard says retrieval is not what is costing us. We put the gold table in
the declared set for **72%** of questions and answer only **37%** correctly — a
conversion rate of 51%. Trần Đình Minh Vương declares the gold table 53% of the
time and answers 66%, which is only possible if what a team *reads* is not what it
*declares*. Either way the constraint is the same for us: turning a table we
already hold into the right number.

Our own evidence says why. Stripping the six clauses off the program prompt made
106 of 170 replies hard-code a constant instead of writing code — and 52.8% of
those constants traced back to a real cell in the tables the model was given. The
model **finds** the figure; it fails at **expressing the lookup as pandas**. The
whole pipeline forces it to do the thing it is bad at.

`direct_answer` is a first-class strategy in the organisers' own code
(`configs/answering/*.yaml`), and `report.py` notes it "has no execution step" —
program generation carries a failure mode that reading does not.

So this asks for the value. A separate pass locates that value in the tables and
emits `num(df_k, r, c)`, which is a legitimate program that reads a real frame —
that is what keeps EXECUTION scorable.

Usage:
  PYTHONPATH=src python scripts/run_direct.py --limit 100 --cache artifacts/direct.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.generate import render_tables, variable_names  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# The organisers' direct prompt, plus the one clause it omits that their *program*
# prompt states explicitly: the answer must be in the unit the question asks for.
# Without it the model reports the raw cell, which is the single largest error
# class we have — 82 questions ship a đồng amount for a "%" question.
SYSTEM = """You are an assistant for analyzing Vietnamese financial reports. Based on the CSV data table or tables provided below, answer the question with EXACTLY ONE numeric value. Do not include a unit or symbol, an explanation, or Markdown; output only the number.

Numerical values in the tables are raw Vietnamese-formatted strings: "." is the thousands separator, "," is the decimal separator, "(...)" denotes a negative value, and the values must be converted before answering.

Determine the unit of each table independently from its column headers (a header such as "Triệu VND" means the figures are already in millions), then convert the final value to the unit the question asks for.

Semantic conventions:
- Always compute a difference without a specified direction as a non-negative absolute magnitude.
- When a direction is explicitly specified, preserve the sign of the difference.

Do not round during intermediate calculations. Round only the final result to 2 decimal places."""

USER = """<câu_hỏi>{question}</câu_hỏi>

{tables}

Trả về đúng một con số, không đơn vị, không giải thích."""

NUMBER_RE = re.compile(r"-?\d[\d.,]*")


def parse_reply(text: str) -> float | None:
    """The last number in the reply, read as a plain float or a Vietnamese one."""

    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    matches = NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    raw = matches[-1].rstrip(".,")
    negative = raw.startswith("-")
    raw = raw.lstrip("-")
    # The model is told to answer in plain form, so "1234.56" is a decimal. Only
    # treat "." as a thousands separator when the grouping is unambiguous.
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
        raw = raw.replace(".", "")
    elif "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/direct.jsonl")
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", default="")
    # The shortlist decides what the model can possibly read, and the anchor
    # ranking puts the gold table in the pool 92.6% of the time against the
    # lexical retriever's 85.6%. The 45.0% this path measured on the gold set was
    # measured through anchor; running it through the weaker shortlist would ship
    # something other than what was measured.
    parser.add_argument("--rank", default="artifacts/anchor_rank.jsonl",
                        help="ranking file to draw candidates from; falls back to "
                             "the lexical retriever per question when absent")
    args = parser.parse_args()

    questions = parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    ranking: dict[int, list] = {}
    rank_path = ROOT / args.rank
    if rank_path.exists():
        for line in rank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                ranking[row["id"]] = [
                    TableKey(r["doc_name"], int(r["table_id"]))
                    for r in row.get("refs", [])
                ]
        print(f"anchor ranking for {len(ranking)} questions")

    cache_path = Path(args.cache)
    done: set[int] = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in questions if q.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} cached, {len(todo)} to ask, model={args.model}")

    client = (ChatClient.local(args.model, args.base_url, max_tokens=2048)
              if args.base_url
              else ChatClient.from_env(ROOT, model=args.model, max_tokens=2048))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    handle = cache_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    counts = {"ok": 0, "unparsed": 0, "transport": 0}
    started = time.time()

    def ask(question) -> None:
        try:
            keys = ranking.get(question.id, [])[: args.tables]
            if not keys:
                groups = max(1, len(question.tickers)) * max(1, len(question.years))
                per_group = max(2, -(-args.tables // groups))
                keys = [
                    hit.key for hit in retriever.search_balanced(
                        question, per_group=per_group, cap=args.tables)
                ]
            if not keys:
                return
            names = variable_names(len(keys))
            grids = {n: store.rows(k) for n, k in zip(names, keys)}
            refs = {
                n: f"{k.doc_name}|{int(store.meta(k).start_line)}"
                for n, k in zip(names, keys)
            }
            user = USER.format(
                question=question.question,
                tables=render_tables(grids, refs))
            reply = client.complete(SYSTEM, user)
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
            with lock:
                counts["transport"] += 1
            print(f"  id={question.id} dropped: {type(exc).__name__}"[:120])
            return

        value = parse_reply(reply)
        row = {
            "id": question.id,
            "value": value,
            "reply": reply[-400:],
            "keys": [[k.doc_name, k.table_id] for k in keys],
            "variables": names,
        }
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            counts["ok" if value is not None else "unparsed"] += 1
            total = counts["ok"] + counts["unparsed"]
            if total % 10 == 0:
                print(f"  {total}/{len(todo)}  parsed={counts['ok'] / total:.0%}  "
                      f"{time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(ask, todo))
    handle.close()

    total = counts["ok"] + counts["unparsed"] or 1
    print(f"\nanswered {counts['ok']}/{total} ({counts['ok'] / total:.1%}) "
          f"in {time.time() - started:.0f}s")
    if counts["transport"]:
        print(f"transport failures, not cached: {counts['transport']}")


if __name__ == "__main__":
    main()
