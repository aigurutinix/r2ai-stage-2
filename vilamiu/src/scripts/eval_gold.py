"""Score an answering configuration against gold records, offline.

Until now every change shipped blind: the project has no labelled slice of the
1,012, so the leaderboard was the only instrument and it costs a submission per
reading. The pool in `runs/easy_pool.jsonl` closes that gap — it comes from the
organisers' own generator, on the same corpus, with an executed `pandas_query`
behind every answer.

Two modes, and the difference between them is the point:

  --tables oracle     hand the model exactly the tables the gold program read
  --tables retrieved  hand it our own shortlist, as inference would

Oracle accuracy answers "can the model read a table it has been given". The gap
to retrieved accuracy is what table selection is costing. Nothing in this project
has separated those two, and they call for opposite fixes.

Usage:
  PYTHONPATH=src python scripts/eval_gold.py --tables oracle --limit 40 \
      --model Qwen/Qwen3-14B-AWQ --local-url http://127.0.0.1:18000/v1
  PYTHONPATH=src python scripts/eval_gold.py --tables oracle --no-llm   # rule floor
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

from vifin.answering import compose  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.answering.generate import build_prompts, generate_query, variable_names  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.corpus.numeric import is_correct  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

RATIO_UNITS = ("phan_tram", "lan", "vong")

REF_RE = re.compile(r"^(?P<doc>.+)\|table_(?P<tid>\d+)$")


def gold_keys(record: dict) -> list[TableKey] | None:
    """The gold `relevant_tables`, as keys into our own store.

    The organisers' corpus was rebuilt from our HTML tables, so `table_N` is our
    `table_id` N. A ref that does not parse means the two indexes have drifted
    and the record cannot be scored — better to drop it loudly than to score it
    against the wrong table.
    """

    keys = []
    for ref in record.get("relevant_tables") or []:
        match = REF_RE.match(ref)
        if match is None:
            return None
        keys.append(TableKey(match.group("doc"), int(match.group("tid"))))
    return keys or None


def rule_reading(
    store: TableStore, parsed, keys: list[TableKey]
) -> tuple[float | None, float | None, TableKey | None]:
    """What the label matcher reads, in the unit the question asks for.

    Returning `Lookup.value` raw was this harness's first version and it made the
    floor look like 10%: the cell is in the column's own unit (usually triệu or tỷ
    đồng) while the gold answer is in the question's unit, so nearly every
    comparison failed by a factor of a million. `column_scale` is the same
    converter the submission pipeline uses, so the floor measured here is now the
    floor that actually ships.

    Both signs come back because the project could not settle the convention:
    Vietnamese statements bracket costs, so "chi phí lãi vay" parses negative
    while the question asks for its size. `lookup.synthesize` says outright that
    which one the gold uses is "unmeasurable without labels" — with labels it is
    one column of this report.
    """

    best = None
    best_key = None
    for key in keys:
        found = lookup_mod.find(store.rows(key), parsed)
        if found is not None and (best is None or found.score > best.score):
            best, best_key = found, key
    if best is None or best_key is None:
        return None, None, None

    grid = store.rows(best_key)
    meta = store.meta(best_key)
    scale_in = lookup_mod.column_scale(
        grid, best.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
    )
    scale_out = float(parsed.unit_scale or 1.0)
    signed = float(best.value) * scale_in / scale_out
    return signed, abs(signed), best_key


def shipped_chain(
    question,
    store: TableStore,
    retriever: LexicalRetriever,
    corroborator: Corroborator,
    *,
    search_k: int,
    magnitude: bool,
) -> tuple[float | None, str]:
    """The deterministic priority order `run_submit.py` ships, in that order.

    Order is the point, not the parts. The project's own notes call fixing this
    ordering its single largest win — larger than two GPU rentals and three model
    mechanisms combined — because a later branch overwriting an earlier one loses
    questions silently. So this mirrors the sequence rather than picking whichever
    branch answers.

    Omitted on purpose: the `located`, `panel`, `generated` and `planned` branches
    read caches keyed by the organisers' question ids, which do not exist for gold
    records. Everything omitted is a *model* branch, so what this measures is the
    deterministic floor the model branches are then layered onto.

    `search_k` and `magnitude` are arguments because both are open questions the
    gold set can now settle: how wide the candidate list should be, and whether a
    single figure should be reported as a magnitude.
    """

    searched = [hit.key for hit in retriever.search(question, top_k=search_k)]
    if not searched:
        return None, "no candidates"

    divided = ratio_mod.resolve(question, store, retriever)
    if divided is not None:
        tables = {n: store.rows(k) for n, k in zip(divided.variables, divided.keys)}
        outcome = run_query(divided.code, tables)
        if outcome.ok and not reads_no_frame(divided.code):
            return outcome.value, "ratio_divide"

    screened = compose.resolve_screen(question, store, retriever)
    if screened is None:
        screened = compose.resolve_screen_ratio(question, store, retriever)
    if screened is None:
        screened = compose.resolve_screen_ratio_threshold(question, store, retriever)
    if screened is not None:
        if screened.operand_keys:
            names = (["df"] if len(screened.operand_keys) == 1
                     else [f"df{i + 1}" for i in range(len(screened.operand_keys))])
            tables = {n: store.rows(k) for n, k in zip(names, screened.operand_keys)}
        else:
            tables = {"df": store.rows(screened.key)}
        outcome = run_query(screened.code, tables)
        if outcome.ok and not reads_no_frame(screened.code):
            return outcome.value, "screen"

    composed = compose.resolve(question, store, retriever)
    if composed is not None:
        tables = {n: store.rows(k) for n, k in zip(composed.variables, composed.keys)}
        outcome = run_query(composed.code, tables)
        if outcome.ok and not reads_no_frame(composed.code):
            return outcome.value, "compose"

    if question.unit_scale:
        picked = corroborator.choose(question, searched)
        if picked is not None:
            grid = store.rows(picked.key)
            meta = store.meta(picked.key)
            found = lookup_mod.find(grid, question)
            if found is not None:
                scale_in = lookup_mod.column_scale(
                    grid, found.column,
                    f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                )
                code = lookup_mod.synthesize(
                    found, scale_in, question.unit_scale, magnitude=magnitude
                )
                outcome = run_query(code, {"df": grid})
                if outcome.ok:
                    return outcome.value, "lookup"

    if question.target_unit in RATIO_UNITS:
        best = None
        for hit in retriever.search(question, top_k=8):
            found = lookup_mod.find_ratio(store.rows(hit.key), question)
            if found is not None and (best is None or found.score > best[1].score):
                best = (hit.key, found)
        if best is not None:
            key, found = best
            code = lookup_mod.synthesize_ratio_cell(
                found, as_percent=question.target_unit == "phan_tram"
            )
            outcome = run_query(code, {"df": store.rows(key)})
            if outcome.ok and not reads_no_frame(code):
                return outcome.value, "ratio_lookup"

    picked = corroborator.choose_best_effort(question, searched)
    if picked is not None:
        grid = store.rows(picked.key)
        meta = store.meta(picked.key)
        found = lookup_mod.find_best_effort(grid, question)
        if found is not None:
            scale_in = lookup_mod.column_scale(
                grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            if question.unit_scale is None and question.target_unit in RATIO_UNITS:
                code = lookup_mod.synthesize_ratio(
                    found, as_percent=question.target_unit == "phan_tram"
                )
            else:
                code = lookup_mod.synthesize(
                    found, scale_in, question.unit_scale or 1.0, magnitude=magnitude
                )
            outcome = run_query(code, {"df": grid})
            if outcome.ok and not reads_no_frame(code):
                return outcome.value, "fallback"

    key = searched[0]
    grid = store.rows(key)
    columns = lookup_mod.value_columns(grid) or [min(1, max(0, len(grid[0]) - 1))]
    code = lookup_mod.synthesize_scan(
        columns[0],
        as_ratio=question.target_unit in RATIO_UNITS,
        as_percent=question.target_unit == "phan_tram",
    )
    outcome = run_query(code, {"df": grid})
    if outcome.ok:
        return outcome.value, "scan"
    return None, "none"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_416.jsonl")
    parser.add_argument("--tables", choices=("oracle", "retrieved"), default="oracle")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0, help="drop the first N records, to score a held-out tail")
    parser.add_argument("--shortlist", type=int, default=6, help="tables per question in retrieved mode")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--bare-prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-llm", action="store_true", help="score the label matcher alone")
    parser.add_argument("--chain", action="store_true",
                        help="score the shipped deterministic priority order, not lookup.find alone")
    parser.add_argument("--search-k", type=int, default=20,
                        help="[--chain] candidate tables the chain ranks over")
    parser.add_argument("--signed", action="store_true",
                        help="[--chain] report a single figure signed instead of as a magnitude")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = records[args.skip:]
    if args.limit:
        records = records[: args.limit]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    usable, dropped = [], 0
    for record in records:
        keys = gold_keys(record)
        if keys is None or any(key not in store for key in keys):
            dropped += 1
            continue
        usable.append((record, keys))
    print(f"{len(usable)} scorable records ({dropped} dropped: gold table absent from our store)")
    if not usable:
        return

    client = None
    if not args.no_llm:
        client = ChatClient.local(args.model, args.local_url, temperature=args.temperature)
    system, user_template = build_prompts(ROOT, bare=args.bare_prompt)

    lock = threading.Lock()
    corroborator = Corroborator(store) if args.chain else None
    sources: dict[str, int] = {}
    sources_right: dict[str, int] = {}
    tally = {"n": 0, "llm_ok": 0, "llm_right": 0, "rule_right": 0,
             "rule_right_abs": 0, "rule_answered": 0, "gold_in_shortlist": 0,
             "read_gold_table": 0, "gold_table_wrong_cell": 0,
             "chain_answered": 0, "chain_right": 0}
    rows_out: list[dict] = []
    started = time.time()

    def score(item) -> None:
        record, keys = item
        question_text = record["question"]
        question = parse_question(record["id"], question_text, roster)

        if args.tables == "oracle":
            shown = keys
            covered = True
        else:
            groups = max(1, len(question.tickers)) * max(1, len(question.years))
            per_group = max(2, -(-args.shortlist // groups))
            shown = [
                hit.key
                for hit in retriever.search_balanced(
                    question, per_group=per_group, cap=args.shortlist)
            ]
            covered = any(key in shown for key in keys)

        if not shown:
            return
        names = variable_names(len(shown))
        tables = {name: store.rows(key) for name, key in zip(names, shown)}

        rule_value, rule_abs, rule_key = rule_reading(store, question, shown)

        chain_value, chain_source = (None, "")
        if corroborator is not None:
            chain_value, chain_source = shipped_chain(
                question, store, retriever, corroborator,
                search_k=args.search_k, magnitude=not args.signed,
            )
        llm_value, llm_ok, code = None, False, ""
        if client is not None:
            refs = {
                name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
                for name, key in zip(names, shown)
            }
            generated = generate_query(
                client, system, user_template, question_text, tables, refs,
                max_attempts=args.attempts, prelude=not args.bare_prompt,
            )
            llm_ok = generated.result.ok and generated.result.value is not None
            llm_value = generated.result.value if llm_ok else None
            code = generated.code

        gold = record["answer"]
        with lock:
            tally["n"] += 1
            tally["gold_in_shortlist"] += int(covered)
            right = False
            if rule_value is not None:
                tally["rule_answered"] += 1
                right = is_correct(gold, rule_value)
                tally["rule_right"] += int(right)
                right_abs = is_correct(gold, rule_abs)
                tally["rule_right_abs"] += int(right_abs)
                # Splitting "wrong table" from "wrong cell" is the whole reason
                # this harness exists: they need opposite fixes, and the
                # leaderboard shows their sum.
                if rule_key in keys:
                    tally["read_gold_table"] += 1
                    if not (right or right_abs):
                        tally["gold_table_wrong_cell"] += 1
            if corroborator is not None:
                sources[chain_source] = sources.get(chain_source, 0) + 1
                if chain_value is not None:
                    tally["chain_answered"] += 1
                    if is_correct(gold, chain_value):
                        tally["chain_right"] += 1
                        sources_right[chain_source] = sources_right.get(chain_source, 0) + 1
            tally["llm_ok"] += int(llm_ok)
            tally["llm_right"] += int(llm_ok and is_correct(gold, llm_value))
            rows_out.append({
                "id": record["id"], "gold": gold, "llm": llm_value,
                "rule": rule_value, "rule_abs": rule_abs, "covered": covered,
                "rule_table": None if rule_key is None else
                f"{rule_key.doc_name}|{rule_key.table_id}",
                "gold_tables": [f"{k.doc_name}|{k.table_id}" for k in keys],
                "chain": chain_value, "chain_source": chain_source,
                "code": code,
            })
            if tally["n"] % 20 == 0:
                print(f"  {tally['n']}/{len(usable)}  "
                      f"llm={tally['llm_right'] / tally['n']:.1%}  "
                      f"{time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(score, usable))

    n = tally["n"] or 1
    print(f"\n== {args.tables} tables, model={'none' if args.no_llm else args.model}, n={tally['n']}")
    print(f"  gold table in shortlist : {tally['gold_in_shortlist'] / n:.1%}")
    print(f"  label matcher answered  : {tally['rule_answered'] / n:.1%}")
    print(f"  label matcher CORRECT   : {tally['rule_right'] / n:.1%}  (signed)")
    print(f"                          : {tally['rule_right_abs'] / n:.1%}  (magnitude)")
    read = tally["read_gold_table"]
    print(f"  read out of a gold table: {read} of {tally['rule_answered']} answered")
    if read:
        print(f"    ... right table, WRONG CELL : {tally['gold_table_wrong_cell']} "
              f"({tally['gold_table_wrong_cell'] / read:.1%} of those)")
    if args.chain:
        print(f"\n  -- shipped chain, search_k={args.search_k}, "
              f"{'signed' if args.signed else 'magnitude'}")
        print(f"  chain answered          : {tally['chain_answered'] / n:.1%}")
        print(f"  chain CORRECT           : {tally['chain_right'] / n:.1%}"
              f"   ({tally['chain_right']} of {tally['n']})")
        for name in sorted(sources, key=lambda s: -sources[s]):
            right = sources_right.get(name, 0)
            print(f"      {sources[name]:4d}  {name:14s} right {right:3d} "
                  f"({right / max(1, sources[name]):.0%})")
    if not args.no_llm:
        print(f"  program executed        : {tally['llm_ok'] / n:.1%}")
        print(f"  program CORRECT         : {tally['llm_right'] / n:.1%}")
    print(f"  elapsed {time.time() - started:.0f}s")

    if args.out:
        path = ROOT / args.out
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
                        encoding="utf-8")
        print(f"  per-question detail -> {path}")


if __name__ == "__main__":
    main()
