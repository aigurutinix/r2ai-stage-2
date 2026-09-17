"""Answer first, then write the program that reproduces the answer.

The same model, on the same retrieved tables, answers 44.2% correctly in prose and
about 35% when made to emit pandas in one pass. It finds the figure and fails at
expressing the lookup, so this splits the two jobs.

Stage one asks for the number, with the direct prompt.
Stage two shows the same tables, states that number, and asks for a program that
lands on it. That is a transcription with a known target rather than a search, and
unlike `patch_direct.py` -- which traces the value back to a single cell -- it also
serves the 70% of questions whose answer sits in no cell because it is a ratio or
a difference.

What it is not: a way to submit an answer the query does not compute. The point of
conditioning stage two on the answer is that the two fields agree, which is what
keeps the pair admissible under the private round's manual review.

Usage:
  PYTHONPATH=src python scripts/_probe_two_stage.py --limit 120 --workers 8
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_direct import SYSTEM as DIRECT_SYSTEM  # noqa: E402
from run_direct import USER as DIRECT_USER  # noqa: E402
from run_direct import parse_reply  # noqa: E402

from vifin.answering.generate import (  # noqa: E402
    build_prompts, extract_code, render_tables, variable_names,
)
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# Stage two is told the target. The wording forbids the one shortcut that would
# make the whole exercise worthless -- writing a bare constant -- because a
# constant is rejected by hand in the private round, and because it would make
# this measurement say nothing about whether the model can locate the cells.
STAGE2_EXTRA = """

<known_answer>
An independent reading of these same tables gives the answer: {answer}

Write the program that derives this figure from the tables. It must read the
DataFrames; assigning the number directly, as `result = {answer}`, is invalid and
will be rejected. If your reading of the tables contradicts the figure above,
trust your own reading and return the program for what the tables actually say.
</known_answer>"""


def close(a, b, tol: float = 2e-3) -> bool:
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def gold_number(text):
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip().replace("%", "").replace(" ", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    if not re.fullmatch(r"-?[\d.,]+", raw or ""):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--out", default="artifacts/_two_stage.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    client = ChatClient.from_env(ROOT, model=args.model)
    prog_system, prog_user = build_prompts(ROOT)

    ranking = {}
    for line in (ROOT / args.rank).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranking[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"])) for r in row.get("refs", [])
            ]

    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    work = []
    for index, record in enumerate(records):
        gold = gold_number(record.get("answer"))
        if gold is None:
            continue
        keys = ranking.get(index, [])[: args.tables]
        if not keys:
            continue
        work.append((index, record, gold, keys))
        if len(work) >= args.limit:
            break
    print(f"{len(work)} questions, model={args.model}", flush=True)

    tally = {"n": 0, "direct_ok": 0, "one_pass_ok": 0, "two_stage_ok": 0,
             "two_stage_exec": 0, "one_pass_exec": 0, "constant": 0}
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        index, record, gold, keys = item
        names = variable_names(len(keys))
        tables = {n: store.rows(k) for n, k in zip(names, keys)}
        refs = {n: f"{k.doc_name}|table_{k.table_id}" for n, k in zip(names, keys)}
        rendered = render_tables(tables, refs)

        try:
            reply = client.complete(
                DIRECT_SYSTEM,
                DIRECT_USER.format(question=record["question"], tables=rendered))
        except RuntimeError:
            return
        direct = parse_reply(reply)

        base_user = (prog_user.replace("{{QUESTION}}", record["question"])
                     .replace("{{VAR_HINT}}", ", ".join(names))
                     .replace("{{TABLES}}", rendered))

        def program(extra: str):
            try:
                out = client.complete(prog_system, base_user + extra)
            except RuntimeError:
                return False, None, ""
            code = extract_code(out)
            if code and "def num(" not in code:
                code = PRELUDE + "\n" + code
            result = run_query(code, tables)
            return bool(result.ok), (result.value if result.ok else None), code

        one_ok, one_value, _ = program("")
        two_ok, two_value, two_code = program(
            STAGE2_EXTRA.format(answer=direct) if direct is not None else "")

        # A stage-two program that merely restates the number proves nothing and
        # would be thrown out by hand, so it is counted apart rather than scored.
        constant = "num(" not in re.sub(r"\s+", "", two_code)

        with lock:
            tally["n"] += 1
            tally["direct_ok"] += int(close(direct, gold))
            tally["one_pass_exec"] += int(one_ok)
            tally["two_stage_exec"] += int(two_ok)
            tally["one_pass_ok"] += int(one_ok and close(one_value, gold))
            tally["two_stage_ok"] += int(two_ok and close(two_value, gold) and not constant)
            tally["constant"] += int(constant)
            rows_out.append({
                "id": record.get("id", index), "gold": gold, "direct": direct,
                "one_pass": one_value, "two_stage": two_value,
                "constant": constant, "code": two_code[:600],
            })
            if tally["n"] % 20 == 0:
                n = tally["n"]
                print(f"  {n}/{len(work)}  direct={tally['direct_ok']/n:.1%}"
                      f"  one_pass={tally['one_pass_ok']/n:.1%}"
                      f"  two_stage={tally['two_stage_ok']/n:.1%}"
                      f"  {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} questions, same tables for every arm")
    print(f"  direct answer correct    : {tally['direct_ok'] / n:6.1%}")
    print(f"  one-pass program exec    : {tally['one_pass_exec'] / n:6.1%}")
    print(f"  one-pass program CORRECT : {tally['one_pass_ok'] / n:6.1%}")
    print(f"  two-stage program exec   : {tally['two_stage_exec'] / n:6.1%}")
    print(f"  two-stage program CORRECT: {tally['two_stage_ok'] / n:6.1%}")
    print(f"  two-stage wrote a constant (excluded): {tally['constant']}")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out), encoding="utf-8")
    print(f"  per-question -> {args.out}")


if __name__ == "__main__":
    main()
