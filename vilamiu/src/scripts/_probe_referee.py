"""Let the directly-read answer choose between candidate programs.

Rescored against gold in the unit each question asks for, the same model on the
same tables answers 58.3% correctly in prose and 50.0% when it writes a program.
Telling it the answer first does not close that (48.3%), so the reading is worth
more as a *judge* than as a hint.

This generates several programs, asks separately for the number, and keeps the
program whose own output agrees with that number. What ships is still a program
that reads the DataFrames, and the answer it produces is the answer declared, so
the pair stays consistent under the private round's manual review — which is the
reason for refereeing rather than simply submitting the prose answer.

Three arms come out of one run, on identical questions and identical samples:

* `first`    -- one program, today's pipeline
* `majority` -- the largest cluster among the samples, needing no second opinion
* `referee`  -- the sample nearest the directly-read answer

`oracle` bounds all of them: whether any sample was right at all.

Usage:
  PYTHONPATH=src python scripts/_probe_referee.py --limit 120 --samples 3
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import close, parse_number  # noqa: E402
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


def majority(values):
    clusters = []
    for value in values:
        for cluster in clusters:
            if close(cluster[0], value):
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return max(clusters, key=len)[0] if clusters else None


def nearest(values, target):
    """The sample closest to the target in relative terms, or None."""

    if target is None or not values:
        return None
    best = None
    best_gap = None
    for value in values:
        scale = max(abs(value), abs(target), 1e-9)
        gap = abs(value - target) / scale
        if best_gap is None or gap < best_gap:
            best, best_gap = value, gap
    return best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--out", default="artifacts/_referee.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    client = ChatClient.from_env(ROOT, model=args.model, temperature=args.temperature)
    cold = ChatClient.from_env(ROOT, model=args.model, temperature=0.0)
    system, user_template = build_prompts(ROOT)

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
        answer = parse_number(record.get("answer"))
        keys = ranking.get(index, [])[: args.tables]
        if answer is None or not keys:
            continue
        work.append((index, record, answer, keys))
        if len(work) >= args.limit:
            break
    print(f"{len(work)} questions x {args.samples} programs + 1 direct read",
          flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        index, record, gold, keys = item
        names = variable_names(len(keys))
        tables = {n: store.rows(k) for n, k in zip(names, keys)}
        refs = {n: f"{k.doc_name}|table_{k.table_id}" for n, k in zip(names, keys)}
        notes = {}
        for n, k in zip(names, keys):
            meta = store.meta(k)
            caption = str(getattr(meta, "caption", "") or "").strip()
            unit = str(getattr(meta, "unit_line", "") or "").strip()
            parts = [caption] + ([unit] if unit and unit not in caption else [])
            notes[n] = " | ".join(p for p in parts if p)[:300]
        rendered = render_tables(tables, refs, notes=notes)

        try:
            reply = cold.complete(
                DIRECT_SYSTEM,
                DIRECT_USER.format(question=record["question"], tables=rendered))
        except RuntimeError:
            return
        direct = parse_reply(reply)

        user = (user_template.replace("{{QUESTION}}", record["question"])
                .replace("{{VAR_HINT}}", ", ".join(names))
                .replace("{{TABLES}}", rendered))
        values = []
        for _ in range(args.samples):
            try:
                out = client.complete(system, user)
            except RuntimeError:
                continue
            code = extract_code(out)
            if code and "def num(" not in code:
                code = PRELUDE + "\n" + code
            outcome = run_query(code, tables)
            if outcome.ok and outcome.value is not None:
                values.append(outcome.value)

        first = values[0] if values else None
        voted = majority(values)
        judged = nearest(values, direct)

        with lock:
            tally["n"] += 1
            tally["direct"] += int(close(direct, gold))
            tally["first"] += int(close(first, gold))
            tally["majority"] += int(close(voted, gold))
            tally["referee"] += int(close(judged, gold))
            tally["oracle"] += int(any(close(v, gold) for v in values))
            rows_out.append({
                "id": record.get("id", index), "gold": gold, "direct": direct,
                "values": values, "voted": voted, "judged": judged,
            })
            n = tally["n"]
            if n % 20 == 0:
                print(f"  {n}/{len(work)}  first={tally['first']/n:.1%}"
                      f"  majority={tally['majority']/n:.1%}"
                      f"  referee={tally['referee']/n:.1%}"
                      f"  direct={tally['direct']/n:.1%}"
                      f"  oracle={tally['oracle']/n:.1%}"
                      f"  {time.time()-started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} questions, {args.samples} programs each")
    for name in ("first", "majority", "referee", "direct", "oracle"):
        print(f"  {name:9s} {tally[name] / n:6.1%}")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")
    print(f"  per-question -> {args.out}")


if __name__ == "__main__":
    main()
