"""The rate-unit sanity guard is dead code. What would switching it on be worth?

`impossible()` in `run_submit.py` starts with

    scale = question.unit_scale
    if scale is None:
        return False

and `unit_scale` is None for every question answered in %, lần or vòng, because
`UNIT_SCALE` maps currency words only. So the one guard that runs on every branch
never fires on the 298 rate questions. The guard written for them, `implausible()`,
sits behind `USE_RATIO`, which is off — measured worse in an experiment that
changed two things at once: it *vetoed* the impossible answer AND *substituted* a
share-of-column-total guess. The note blames the substitution ("plausible is not
correct"). The veto alone was never measured on its own.

A veto is only worth anything if some later branch has a better answer waiting.
The branches that could serve these questions all sit behind
`query is PLACEHOLDER_QUERY`, so an early branch shipping an impossible figure
locks them out. This counts, per question that currently ships an impossible rate,
how many of those later branches produce a value that is at least in range.

Usage:  PYTHONPATH=src python scripts/_probe_rate_veto.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering import ratio as R  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"
GEN = ROOT / "artifacts" / "gen_helpers.jsonl"
RATE_UNITS = frozenset({"phan_tram", "lan", "vong"})
LIMIT = 1000.0


def in_range(value: float) -> bool:
    return abs(value) <= LIMIT


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}
    generated = {}
    for line in GEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                generated[row["id"]] = row

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ret = LexicalRetriever(store.frame)

    def answer(i: int) -> float:
        try:
            return float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            return 0.0

    def branch(i: int) -> str:
        c = (preds[i].get("pandas_query") or "").strip()
        if c in ("", "result = 0.0"):
            return "zero"
        if "def num(" in c or "find_row(" in c:
            return "llm"
        if c.count("df") >= 2:
            return "multi"
        if "iloc" in c:
            return "lookup"
        return "other"

    pool = [
        q for q in parsed.values()
        if q.target_unit in RATE_UNITS and abs(answer(q.id)) > LIMIT
    ]

    lines: list[str] = []
    p = lines.append
    p(f"questions shipping an impossible rate: {len(pool)}")
    p(f"  by branch that shipped it: {dict(Counter(branch(q.id) for q in pool))}")
    p(f"  already served by the llm branch: "
      f"{sum(1 for q in pool if branch(q.id) == 'llm')}  <- veto gains nothing here")

    rescue: Counter[str] = Counter()
    detail: list[str] = []
    for q in pool:
        if branch(q.id) == "llm":
            rescue["already_llm"] += 1
            continue
        got: list[tuple[str, float]] = []

        # 1. The cached LLM program, which never ran because an earlier branch
        #    had already claimed the question.
        row = generated.get(q.id)
        if row is not None and not reads_no_frame(row["code"]):
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            try:
                tables = {n: store.rows(k) for n, k in zip(row["variables"], keys)}
                out = run_query(row["code"], tables)
            except Exception:
                out = None
            if out is not None and out.ok and out.value != 0.0 and in_range(out.value):
                got.append(("llm", float(out.value)))

        # 2. `ratio_lookup` — a cell already printed as a percentage.
        best = None
        for variant in lookup_mod.metric_variants(q.question):
            probe = dataclasses.replace(q, question=variant)
            for hit in ret.search(probe, top_k=8):
                found = lookup_mod.find_ratio(store.rows(hit.key), q)
                if found is not None and (best is None or found.score > best[1].score):
                    best = (hit.key, found)
        if best is not None:
            code = lookup_mod.synthesize_ratio_cell(
                best[1], as_percent=q.target_unit == "phan_tram")
            out = run_query(code, {"df": store.rows(best[0])})
            if out.ok and in_range(out.value) and out.value != 0.0:
                got.append(("ratio_cell", float(out.value)))

        # 3. The quotient the question names.
        try:
            res = (R.resolve_compound(q, store, ret, 10)
                   if R.compound_shape(q) else
                   (R.resolve(q, store, ret, 10) if R.eligible(q) else None))
        except Exception:
            res = None
        if res is not None and in_range(res.value):
            got.append(("ratio_divide", float(res.value)))

        if not got:
            rescue["no_alternative"] += 1
            continue
        rescue["|".join(name for name, _ in got)] += 1
        alt = ", ".join(f"{n}={v:.4g}" for n, v in got)
        detail.append(
            f"  id={q.id:4d} {branch(q.id):6s} ships {answer(q.id):.4g} -> {alt}")
        detail.append(f"        {q.question[:135]}")

    p("\nwhat a veto would fall through to:")
    for name, count in rescue.most_common():
        p(f"  {name:34s} {count:3d}")

    rescuable = sum(c for n, c in rescue.items()
                    if n not in ("already_llm", "no_alternative"))
    p(f"\nquestions a veto would hand to a branch with an in-range answer: {rescuable}")
    p("  every one of them scores zero today, so the change cannot lose EXEC or")
    p("  ANSWER on this pool; only the declared tables move.")
    p("")
    lines.extend(detail)

    out_path = ROOT / "artifacts" / "_probe_rate_veto.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
