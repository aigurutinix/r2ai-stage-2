"""Generate pandas for the questions the deterministic lookup cannot decide.

550 of the 1,012 questions compare entities, span years, or derive a ratio, and
the lookup path scores zero on every one of them. This runs the LLM over that
remainder and keeps only programs that execute here, in the grader's contract,
and return a finite number.

Results are cached per question so a re-run costs nothing for work already done,
and a crash mid-way loses nothing.

Usage:
    python scripts/run_generate.py --limit 30            # sample first
    python scripts/run_generate.py --workers 12          # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.generate import build_prompts, generate_query, variable_names  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402




def load_cache(CACHE: Path) -> dict[int, dict]:
    if not CACHE.exists():
        return {}
    done = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["id"]] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument("--tables", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--cache", default="artifacts/generated.jsonl")
    parser.add_argument("--reranked", default="artifacts/reranked_metric.jsonl")
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="generate for all 1012, including single lookups")
    parser.add_argument("--only", default="",
                        help="JSON file of question ids to generate for, ignoring "
                             "every other filter; for retrying the questions a "
                             "previous pass could not produce a program for")
    parser.add_argument("--bare-prompt", action="store_true",
                        help="drop our six clauses and the num/find_row prelude, "
                             "leaving the organisers' prompt; tests whether the "
                             "helpers that lifted executability 52%%->77%% for only "
                             "+2 correct answers were crowding out the reasoning")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature; 0 makes every run identical, "
                             "which is why running this twice has never produced "
                             "two opinions to compare")
    parser.add_argument("--captions", action="store_true",
                        help="show each table caption and unit line")
    parser.add_argument("--local-url", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    # Cross-encoder ordering, kept entity-balanced: a five-company comparison
    # needs one table per company, and a flat relevance ranking collapses onto
    # whichever company's wording scores best.
    reranked: dict[int, list[TableKey]] = {}
    rerank_path = root / args.reranked
    if rerank_path.exists():
        for line in rerank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reranked[row["id"]] = [TableKey(d, int(t)) for d, t in row["keys"]]
    print(f"{len(reranked)} questions with cross-encoder ordering")

    def balanced(question, cap: int) -> list[TableKey]:
        """Tables for one question, entity-balanced.

        MEASURED: feeding this branch the cross-encoder ordering instead dropped
        executable programs from 49.9% to 26.4%. The reranked list comes from a
        flat top-30 relevance search, which for a five-company comparison need
        not contain all five companies at all — and a program missing an operand
        cannot run. Reranking helps where the problem is *relevance* (finding one
        line item) and hurts where it is *coverage* (assembling a comparison).
        """

        order = reranked.get(question.id) if args.use_rerank else None
        if not order:
            # `search_balanced` reserves `per_group` slots per (ticker, year) and
            # defaults to 2. For a five-company comparison that fills the budget,
            # but a question about one company in one year has a single group and
            # therefore receives **two** tables however large `--tables` is — and
            # those are exactly the questions this branch was just opened to.
            # Picking the right table inside the right document is our weakest
            # step (TABLES precision 0.336 against DOCS precision 0.971), so two
            # candidates is close to a coin flip on whether the model is even
            # shown the row it needs. Spread the budget over however many groups
            # the question actually has.
            groups = max(1, len(question.tickers)) * max(1, len(question.years))
            per_group = max(2, -(-cap // groups))
            return [
                hit.key
                for hit in retriever.search_balanced(
                    question, per_group=per_group, cap=cap)
            ]
        groups: dict[tuple[str, str], list[TableKey]] = {}
        for key in order:
            meta = store.meta(key)
            groups.setdefault((meta.ticker, meta.year), []).append(key)
        picked: list[TableKey] = []
        for rank in range(cap):
            for members in groups.values():
                if rank < len(members) and len(picked) < cap:
                    picked.append(members[rank])
            if len(picked) >= cap:
                break
        return picked[:cap]

    client = (ChatClient.local(args.model, args.local_url, temperature=args.temperature)
              if args.local_url
              else ChatClient.from_env(root, model=args.model,
                                       temperature=args.temperature))
    system, user_template = build_prompts(root, bare=args.bare_prompt)

    CACHE = Path(args.cache)
    done = load_cache(CACHE)
    # The filter below withholds a generation from every question that names a
    # currency unit and reads as a single lookup — 411 of the 1,012 — on the
    # premise that the lexical matcher owns them. It does, at 42.8%, and the LLM
    # branch sits *after* the matcher in the cascade, so a program generated here
    # can only fire where the matcher already declined. Withholding it therefore
    # buys nothing and costs the questions that fall through to the best-effort
    # branch at 5.9%. `--all` generates for the full set.
    if args.only:
        wanted = set(json.loads(Path(args.only).read_text(encoding="utf-8")))
        todo = [q for q in parsed if q.id in wanted and q.id not in done]
        print(f"restricted to {len(wanted)} ids, {len(todo)} not already cached")
    else:
        todo = [
            q for q in parsed
            if q.id not in done
            and (args.all
                 or q.unit_scale is None
                 or not lookup_mod.is_single_lookup(q.question))
        ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"model={args.model}  {len(todo)} questions to generate ({len(done)} cached)")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    handle = CACHE.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "transport": 0}
    started = time.time()

    def work(question) -> None:
        # `pool.map` re-raises on the first exception the moment its results are
        # iterated, so one flaky response kills the whole batch: an
        # `http.client.IncompleteRead` from a truncated chunked reply ended a run
        # 471 questions in, with 540 still to do. A transport failure is already
        # treated as "say nothing, cache nothing" below; escaping the thread is
        # the only part that was ever fatal.
        try:
            _work(question)
        except Exception as exc:  # noqa: BLE001 - a whole run must not die here
            with lock:
                counters["transport"] += 1
                counters.setdefault("last_error", 0)
            print(f"  id={question.id} dropped: {type(exc).__name__}: {exc}"[:160])

    def _work(question) -> None:
        keys = balanced(question, args.tables)
        if not keys:
            return
        names = variable_names(len(keys))
        tables = {name: store.rows(key) for name, key in zip(names, keys)}
        refs = {
            name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
            for name, key in zip(names, keys)
        }
        notes = None
        if args.captions:
            # The caption states the statement and, where the table declares one,
            # its unit. The rule branch has always scaled by it; the model branch
            # was left to infer the unit from column headers that often omit it.
            notes = {}
            for name, key in zip(names, keys):
                meta = store.meta(key)
                caption = str(getattr(meta, "caption", "") or "").strip()
                unit = str(getattr(meta, "unit_line", "") or "").strip()
                parts = [caption] + ([unit] if unit and unit not in caption else [])
                notes[name] = " | ".join(p for p in parts if p)[:300]
        generated = generate_query(
            client, system, user_template, question.question, tables, refs,
            max_attempts=args.attempts, prelude=not args.bare_prompt, notes=notes,
        )
        row = {
            "id": question.id,
            "ok": generated.result.ok,
            "value": generated.result.value,
            "error": generated.result.error[:300],
            "attempts": generated.attempts,
            "code": generated.code,
            "variables": list(tables),
            "refs": refs,
            "keys": [[k.doc_name, k.table_id] for k in keys],
        }
        # A transport or configuration failure says nothing about the question.
        # Caching it would permanently skip work that never actually ran — an
        # invalid model id once wrote 40 dead rows that looked like real results.
        if generated.result.error.startswith(("llm error", "network error", "malformed")):
            with lock:
                counters["transport"] += 1
                # Counting these without showing one hides the reason a run lost
                # most of its questions: two A/B arms finished at 60 of 150 and
                # the log said only "transport/config failures: 90". Print the
                # first few so the cause is visible while the run is still going.
                if counters["transport"] <= 3:
                    print(f"  id={question.id} bỏ qua: "
                          f"{generated.result.error[:200]}", flush=True)
            return
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            counters["ok" if generated.result.ok else "fail"] += 1
            total = counters["ok"] + counters["fail"]
            if total % 10 == 0 or total == len(todo):
                rate = counters["ok"] / total
                print(f"  {total}/{len(todo)}  executable={rate:.1%}  {time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()

    total = counters["ok"] + counters["fail"]
    if counters["transport"]:
        print(f"transport/config failures, not cached: {counters['transport']}")
    if total:
        print(f"\nexecutable programs: {counters['ok']}/{total} ({counters['ok'] / total:.1%})")
        print(f"elapsed {time.time() - started:.0f}s -> {(time.time() - started) / total:.1f}s per question")


if __name__ == "__main__":
    main()
