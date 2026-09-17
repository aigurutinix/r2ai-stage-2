"""Pick the tables first, then read only those.

Two measured quantities set this up. Coverage -- the share of questions where some
shown table holds the answer -- rises with the number of tables shown: 76.7% at
three, 89.3% at eight. Reading accuracy falls: 57.9% of covered questions at
three, 45.2% at eight. The product peaks somewhere in between and the best seen so
far is 44.4%.

Neither number has to be accepted. The 89.3% coverage of eight tables and the
57.9% reading of three would multiply to 0.52 if a question's few relevant tables
could be identified before the reading starts.

So pass one shows all eight tables as schemas only -- caption, columns, row labels,
no CSV body -- and asks which could answer the question. That is recognition, and
it is a far easier job than computing. Pass two shows only the chosen tables in
full and asks for the program.

The measurement reports what fraction of the coverage survives pass one, because
that is what bounds the whole idea: a selector that discards the holder cannot be
rescued by good reading.

Usage:
  PYTHONPATH=src python scripts/_probe_coarse_to_fine.py --limit 150
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

from _rescore_unit_fixed import parse_number  # noqa: E402

from vifin.answering.generate import (  # noqa: E402
    build_prompts, describe_schema, extract_code, render_tables, variable_names,
)
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SELECT_SYSTEM = """You choose which financial tables can answer a question.

You are shown several table schemas: the statement caption, the column headers, and
the row labels of each. You are NOT shown the numbers, and you do not need them.

Reply with only the variable names of the tables needed, separated by commas, most
relevant first. Name at most three. Reply with nothing else.

A table is needed when its row labels contain the line item the question asks
about and its columns cover the period the question asks for. Prefer the statement
the item belongs to over a note that merely mentions it."""

SELECT_USER = """<câu_hỏi>{question}</câu_hỏi>

{schemas}

Which tables are needed? Reply with variable names only, at most three."""

NAME_RE = re.compile(r"\bdf\d*\b")


def close(a, b, tol: float = 2e-3) -> bool:
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def holds(grid, answer, context: str = "") -> bool:
    """Does any cell equal the answer, raw or after its column scale?

    The gold is stored in the unit the question asks for; the cells are raw. A
    search that only compares raw values reported 61.3% coverage where the same
    tables really cover 89.3%.
    """

    from vifin.answering import lookup as lookup_mod

    for row in grid[1:]:
        for column, cell in enumerate(row):
            value = parse_number(cell)
            if value is None or value == 0:
                continue
            if close(value, answer, 5e-4):
                return True
            scaled = value * lookup_mod.column_scale(grid, column, context)
            if close(scaled, answer, 5e-4):
                return True
            for unit in (1e9, 1e6, 1e3):
                if close(value / unit, answer, 5e-4) or close(scaled / unit, answer, 5e-4):
                    return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--shown", type=int, default=8, help="tables offered to pass one")
    parser.add_argument("--keep", type=int, default=3, help="cap on what pass one may keep")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--out", default="artifacts/_coarse_to_fine.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    client = ChatClient.from_env(ROOT, model=args.model, temperature=0.0)
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
        if answer is None:
            continue
        keys = ranking.get(index, [])[: args.shown]
        if not keys:
            continue
        work.append((index, record, answer, keys))
        if len(work) >= args.limit:
            break
    print(f"{len(work)} questions, {args.shown} offered, at most {args.keep} kept",
          flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        index, record, answer, keys = item
        names = variable_names(len(keys))
        grids = {n: store.rows(k) for n, k in zip(names, keys)}
        notes = {}
        for n, k in zip(names, keys):
            meta = store.meta(k)
            caption = str(getattr(meta, "caption", "") or "").strip()
            unit = str(getattr(meta, "unit_line", "") or "").strip()
            parts = [caption] + ([unit] if unit and unit not in caption else [])
            notes[n] = " | ".join(p for p in parts if p)[:300]
        refs = {n: f"{k.doc_name}|table_{k.table_id}" for n, k in zip(names, keys)}

        covered_before = any(holds(grids[n], answer, notes[n]) for n in names)

        schemas = "\n".join(
            describe_schema(n, refs[n], grids[n], notes[n]) for n in names)
        try:
            reply = client.complete(
                SELECT_SYSTEM,
                SELECT_USER.format(question=record["question"], schemas=schemas))
        except RuntimeError:
            return
        picked = []
        for token in NAME_RE.findall(reply):
            if token in grids and token not in picked:
                picked.append(token)
        picked = picked[: args.keep] or names[: args.keep]

        covered_after = any(holds(grids[n], answer, notes[n]) for n in picked)

        # Pass two must show the kept tables under the names the sandbox will
        # bind, not the names pass one used to choose them. Showing `df3` while
        # binding `df1` is a NameError on every program that reads it, which is
        # what made the first run of this probe score 18.7%.
        fresh = variable_names(len(picked))
        chosen = dict(zip(fresh, (grids[n] for n in picked)))
        chosen_refs = dict(zip(fresh, (refs[n] for n in picked)))
        chosen_notes = dict(zip(fresh, (notes[n] for n in picked)))
        user = (user_template.replace("{{QUESTION}}", record["question"])
                .replace("{{VAR_HINT}}", ", ".join(fresh))
                .replace("{{TABLES}}", render_tables(
                    chosen, chosen_refs, notes=chosen_notes)))
        try:
            out = client.complete(system, user)
        except RuntimeError:
            return
        code = extract_code(out)
        if code and "def num(" not in code:
            code = PRELUDE + "\n" + code
        outcome = run_query(code, chosen)
        value = outcome.value if outcome.ok else None

        with lock:
            tally["n"] += 1
            tally["covered_before"] += int(covered_before)
            tally["covered_after"] += int(covered_after)
            tally["kept"] += len(picked)
            tally["right"] += int(close(value, answer))
            rows_out.append({
                "id": record.get("id", index), "gold": answer, "values": [value],
                "picked": picked, "covered_before": covered_before,
                "covered_after": covered_after,
            })
            n = tally["n"]
            if n % 25 == 0:
                print(f"  {n}/{len(work)}  correct={tally['right']/n:.1%}"
                      f"  coverage {tally['covered_before']/n:.1%}->"
                      f"{tally['covered_after']/n:.1%}"
                      f"  {time.time()-started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} questions")
    print(f"  coverage offered to pass one : {tally['covered_before'] / n:6.1%}")
    print(f"  coverage surviving pass one  : {tally['covered_after'] / n:6.1%}")
    print(f"  tables kept, mean            : {tally['kept'] / n:6.2f}")
    print(f"  CORRECT                      : {tally['right'] / n:6.1%}")
    if tally["covered_after"]:
        print(f"  reading accuracy on what survived: "
              f"{tally['right'] / tally['covered_after']:6.1%}")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")
    print(f"  per-question -> {args.out}")


if __name__ == "__main__":
    main()
