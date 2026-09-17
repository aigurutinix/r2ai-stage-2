"""Sample several programs per question and let them vote.

Two runs of the one-pass program branch over the same 60 questions, at
temperature 0, disagreed on 8 of them and scored 30.0% and 38.3%. That spread is
not only a measurement nuisance: it says roughly one question in seven sits on a
knife edge, where the model sometimes reads the right cell and sometimes does not.
Voting is the standard way to collect that.

Three numbers come out, and the third is the one that decides whether to continue:

* `single`   -- mean accuracy of one sample, the current pipeline
* `majority` -- accuracy after clustering the samples by value and taking the
                largest cluster, which is what a submission could actually ship
* `oracle`   -- accuracy if a perfect chooser picked the right sample whenever one
                exists. This is the ceiling of every voting or reranking scheme.

If `oracle` sits near `single`, the model is simply wrong on those questions and no
selection mechanism can help; that closes the direction rather than inviting a
better voter. If `oracle` is far above and `majority` is not, the answers are there
and the chooser is what needs work.

Usage:
  PYTHONPATH=src python scripts/_probe_self_consistency.py --limit 200 --samples 5
"""

from __future__ import annotations

import argparse
import collections
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

from vifin.answering.generate import (  # noqa: E402
    build_prompts, extract_code, render_tables, variable_names,
)
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


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


def majority(values):
    """The largest cluster of mutually-close values, and its size.

    Exact equality would split 3.14 from 3.140001 and hand the vote to a
    singleton, so clustering uses the same tolerance the scorer does.
    """

    clusters = []
    for value in values:
        for cluster in clusters:
            if close(cluster[0], value):
                cluster.append(value)
                break
        else:
            clusters.append([value])
    if not clusters:
        return None, 0
    best = max(clusters, key=len)
    return best[0], len(best)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="0 still varies by provider, but not enough to vote on")
    parser.add_argument("--bare", action="store_true",
                        help="the organisers' prompt alone, without our six clauses "
                             "(3503 of 6427 characters) and without the prelude")
    parser.add_argument("--captions", action="store_true",
                        help="show each table caption and unit line, which the "
                             "rule branch already reads but the model never saw")
    parser.add_argument("--local-url", default="",
                        help="serve address for a self-hosted model; "
                             "empty means OpenRouter")
    parser.add_argument("--out", default="artifacts/_self_consistency.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    client = (
        ChatClient.local(args.model, args.local_url, temperature=args.temperature)
        if args.local_url else
        ChatClient.from_env(ROOT, model=args.model, temperature=args.temperature)
    )
    system, user_template = build_prompts(ROOT, bare=args.bare)

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
    print(f"{len(work)} questions x {args.samples} samples, model={args.model}, "
          f"temperature={args.temperature}", flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        index, record, gold, keys = item
        names = variable_names(len(keys))
        tables = {n: store.rows(k) for n, k in zip(names, keys)}
        refs = {n: f"{k.doc_name}|table_{k.table_id}" for n, k in zip(names, keys)}
        notes = None
        if args.captions:
            notes = {}
            for n, k in zip(names, keys):
                meta = store.meta(k)
                caption = str(getattr(meta, "caption", "") or "").strip()
                unit = str(getattr(meta, "unit_line", "") or "").strip()
                # The caption usually ends with the unit line already, so joining
                # them blindly produced "Đơn vị: VND | Đơn vị: VND".
                parts = [caption] + ([unit] if unit and unit not in caption else [])
                notes[n] = " | ".join(parts)[:300]
        user = (user_template.replace("{{QUESTION}}", record["question"])
                .replace("{{VAR_HINT}}", ", ".join(names))
                .replace("{{TABLES}}", render_tables(tables, refs, notes=notes)))

        values = []
        for _ in range(args.samples):
            try:
                reply = client.complete(system, user)
            except RuntimeError:
                continue
            code = extract_code(reply)
            # The prelude is the other half of what `--bare` removes: with `num`
            # and `find_row` pre-defined, the model never has to say which cell it
            # means. Buying that convenience cost 147 executable programs for two
            # correct ones, and the trade has never been measured the other way.
            if code and not args.bare and "def num(" not in code:
                code = PRELUDE + "\n" + code
            outcome = run_query(code, tables)
            if outcome.ok and outcome.value is not None:
                values.append(outcome.value)

        voted, support = majority(values)
        right = [v for v in values if close(v, gold)]

        with lock:
            tally["n"] += 1
            tally["single_right"] += len(right)
            tally["single_total"] += max(len(values), 1)
            tally["majority_right"] += int(close(voted, gold))
            tally["oracle_right"] += int(bool(right))
            tally["unanimous"] += int(support == len(values) and len(values) > 1)
            rows_out.append({
                "id": record.get("id", index), "gold": gold,
                "values": values, "voted": voted, "support": support,
            })
            n = tally["n"]
            if n % 20 == 0:
                print(f"  {n}/{len(work)}  single={tally['single_right']/max(tally['single_total'],1):.1%}"
                      f"  majority={tally['majority_right']/n:.1%}"
                      f"  oracle={tally['oracle_right']/n:.1%}"
                      f"  {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} questions, {args.samples} samples each")
    print(f"  single sample (current) : {tally['single_right'] / max(tally['single_total'], 1):6.1%}")
    print(f"  majority vote           : {tally['majority_right'] / n:6.1%}")
    print(f"  oracle (any sample right): {tally['oracle_right'] / n:6.1%}   <-- ceiling")
    print(f"  unanimous questions     : {tally['unanimous']} of {tally['n']}")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out), encoding="utf-8")
    print(f"  per-question -> {args.out}")


if __name__ == "__main__":
    main()
