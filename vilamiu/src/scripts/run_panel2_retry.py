"""Retry panel2 failures with richer schema (count, screen-ratio) + signed metrics.

Re-injects Circular-200 figures with SIGN (panels previously stored abs), then
re-asks Qwen3-14B only on ids that failed the first pass.

Usage:
  PYTHONPATH=src python scripts/run_panel2_retry.py --local-url http://127.0.0.1:18000/v1
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

from vifin.answering import plan_json  # noqa: E402
from vifin.answering.panel_det import load_panel  # noqa: E402
from vifin.answering.sandbox import portability_problems, run_query  # noqa: E402
from vifin.corpus.metrics import METRICS  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

import importlib.util

spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_panel2.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

METRIC_LABEL = {m.name: m.aliases[0] for m in METRICS}
LABEL_TO_NAME = {m.aliases[0]: m.name for m in METRICS}
LABEL_TO_NAME.update({m.name: m.name for m in METRICS})


def resign_panel(rows: list[list[str]], question, cube: dict) -> list[list[str]]:
    """Overwrite panel cells with signed metrics.parquet values when known."""

    if not question.tickers or not question.years:
        return rows
    scope = question.scope or "consolidated"
    by_key: dict[tuple[str, str, str], list] = {}
    header = rows[0]
    out = [list(header)]
    for row in rows[1:]:
        key = (str(row[0]), str(row[1]), str(row[2]))
        by_key[key] = [str(c) for c in row]

    for ticker in question.tickers:
        for year in question.years:
            y = str(year)
            blob = None
            for sc in (scope, "separate" if scope == "consolidated" else "consolidated",
                       "unspecified"):
                blob = cube.get((ticker, y, sc))
                if blob is not None:
                    break
            if blob is None:
                continue
            for name, label in METRIC_LABEL.items():
                value = blob.get(name)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number != number:
                    continue
                for chi in (label, name):
                    by_key[(ticker, y, chi)] = [
                        ticker, y, chi, label, repr(number),  # signed
                    ]
    out.extend(by_key.values())
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--prev", default="artifacts/panel2_scale.jsonl")
    parser.add_argument("--build", default="artifacts/panel2_build_scale.jsonl")
    parser.add_argument("--cache", default="artifacts/panel2_retry.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ids", default="",
                        help="Comma-separated ids to force (skip fail discovery)")
    args = parser.parse_args()

    parsed = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv")}
    panels = {}
    for line in (ROOT / args.build).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            panels[row["id"]] = row

    if args.ids:
        todo = [int(x) for x in args.ids.replace("\n", ",").split(",") if x.strip()]
        todo = [i for i in todo if i in panels]
    else:
        prev_ok = set()
        fail = []
        for line in (ROOT / args.prev).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                prev_ok.add(row["id"])
            elif row["id"] in panels:
                fail.append(row["id"])
        seen = set()
        todo = []
        for i in fail:
            if i not in seen and i not in prev_ok:
                seen.add(i)
                todo.append(i)
    print(f"retry {len(todo)} failures; model={args.model}")

    cube = load_panel(ROOT / "artifacts" / "metrics.parquet")
    client = ChatClient.local(args.model, args.local_url, max_tokens=500)
    cache = ROOT / args.cache
    done = set()
    if cache.exists():
        done = {json.loads(l)["id"]
                for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in todo if i not in done]
    print(f"todo after cache skip: {len(todo)}")

    handle = cache.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters: collections.Counter = collections.Counter()
    started = time.time()

    def work(qid: int) -> None:
        question = parsed[qid]
        raw_panel = panels[qid]["rows"]
        rows = resign_panel(raw_panel, question, cube)
        try:
            reply = client.complete(rp.SYSTEM, rp.build_user(question, rows))
        except RuntimeError:
            with lock:
                counters["transport"] += 1
            return
        metrics = rp.panel_metrics(rows)
        plan = plan_json.parse(reply, metrics=metrics) or plan_json.parse(reply)
        row = {"id": qid, "ok": False, "source": "panel2_retry"}
        if plan is None:
            row["error"] = "no usable plan"
            row["reply"] = reply[-300:]
            with lock:
                counters["no plan"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return
        row["plan"] = {
            "op": plan.op, "metric": plan.metric, "axis": plan.axis,
            "denominator": plan.denominator, "filter_metric": plan.filter_metric,
            "filter_take": plan.filter_take,
        }
        done_pair = plan_json.execute(plan, rows, question.target_unit or "")
        if done_pair is None:
            row["error"] = "unresolved"
            with lock:
                counters["unresolved"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return
        expected, used = done_pair
        code = plan_json.compile_query(plan, question.target_unit or "")
        problems = portability_problems(code)
        outcome = run_query(code, {"df": rows})
        agrees = (outcome.ok and not problems and not reads_no_frame(code)
                  and abs(outcome.value - expected) <= 0.011)
        row.update(
            ok=bool(agrees), value=outcome.value if outcome.ok else None,
            expected=expected, code=code, used=used,
            panel_rows=rows, keys=panels[qid].get("keys") or [],
            mode="panel", shape=plan.op, variables=["df"],
            error=("; ".join(problems) or outcome.error or "")[:160],
        )
        with lock:
            counters["ok" if agrees else "fail"] += 1
            if agrees:
                counters["op:" + plan.op] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            total = counters["ok"] + counters["fail"] + counters["no plan"] + counters["unresolved"]
            if total % 20 == 0:
                print(f"  {total}/{len(todo)} ok={counters['ok']} {time.time()-started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()
    print(f"done ok={counters['ok']} fail={counters['fail']} "
          f"no_plan={counters['no plan']} unresolved={counters['unresolved']}")
    for k, v in counters.most_common(12):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
