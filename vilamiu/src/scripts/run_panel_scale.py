"""Scale panel answering on CPU: BCTC lookup + Circular-200 inject + rule plans.

High-impact path without training:
  1. Build a per-question evidence panel from OCR tables AND/OR metrics.parquet
  2. Derive a plan_json Plan from regex (screen / ratio / compose) — no LLM
  3. Execute with plan_json; emit py37-safe pandas
  4. Cache everything that re-executes and passes impossible()

Leftovers where rules cannot name a plan are listed for a 14B `run_panel2` batch.

Usage:
  PYTHONPATH=src python scripts/run_panel_scale.py
  PYTHONPATH=src python scripts/run_panel_scale.py --limit 100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import compose, lookup as lookup_mod, ratio as ratio_mod  # noqa: E402
from vifin.answering import panel2, plan_json  # noqa: E402
from vifin.answering.panel_context import expand_operands  # noqa: E402
from vifin.answering.panel_det import load_panel, metrics_mentioned  # noqa: E402
from vifin.answering.plan_json import Plan  # noqa: E402
from vifin.answering.sandbox import portability_problems, run_query  # noqa: E402
from vifin.corpus.metrics import METRICS, _fold  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts" / "audit_submission.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)

# Circular-200 column → Vietnamese label used as panel chi_tieu (matches FORMULAS).
METRIC_LABEL = {m.name: m.aliases[0] for m in METRICS}
# Also keep English names so screen filters that say "cfo" still resolve.
METRIC_LABEL_EN = {m.name: m.name for m in METRICS}

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}


def impossible(question: str, answer) -> bool:
    value = aud.as_float(answer)
    if value is None:
        return True
    if aud.YEAR_Q.search(question) and not (
            float(value).is_integer() and 1990 <= value <= 2100):
        return True
    if aud.SHARE_Q.search(question) and not aud.TIMES_Q.search(question) and \
            not aud.GROWTH_Q.search(question) and (value > 100 or value < 0):
        return True
    if aud.PERCENT_Q.search(question) and not aud.TIMES_Q.search(question) and \
            abs(value) > 1e5:
        return True
    if aud.TIMES_Q.search(question) and (value > 500 or value < 0):
        return True
    if aud.COUNT_Q.search(question) and (
            not float(value).is_integer() or value < 0 or value > 60):
        return True
    return value == 0


def referenced_metrics(question: str) -> list[str]:
    folded = _fold(question)
    found: list[str] = []
    for alias, name in ALIAS_TO_METRIC.items():
        if alias in folded and name not in found:
            found.append(name)
    return expand_operands(question, found)


def inject_metrics(question, cube: dict) -> list[list[str]]:
    """Rows from metrics.parquet with Vietnamese (+ English) chi_tieu labels."""

    names = referenced_metrics(question.question)
    if not names or not question.tickers or not question.years:
        return [list(panel2.PANEL_HEADER)]
    scope = question.scope or "consolidated"
    rows = [list(panel2.PANEL_HEADER)]
    for ticker in question.tickers:
        for year in sorted(question.years):
            y = str(year)
            row = None
            for sc in (scope, "separate" if scope == "consolidated" else "consolidated",
                       "unspecified"):
                row = cube.get((ticker, y, sc))
                if row is not None:
                    break
            if row is None:
                continue
            for name in names:
                if name.startswith("_"):
                    continue
                value = row.get(name)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number != number:
                    continue
                label_vi = METRIC_LABEL.get(name, name)
                rows.append([ticker, y, label_vi, label_vi, repr(abs(number))])
                # English twin helps when the rule plan uses METRICS.name.
                if name != label_vi:
                    rows.append([ticker, y, name, label_vi, repr(abs(number))])
    return rows


def merge_panels(*panels: list[list[str]]) -> list[list[str]]:
    """Union of panel bodies; first occurrence of (ma, nam, chi_tieu) wins."""

    out = [list(panel2.PANEL_HEADER)]
    seen: set[tuple[str, str, str]] = set()
    for panel in panels:
        if not panel or len(panel) < 2:
            continue
        for row in panel[1:]:
            key = (str(row[0]), str(row[1]), str(row[2]))
            if key in seen:
                continue
            seen.add(key)
            out.append([str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])])
    return out


def panel_metric_names(rows: list[list[str]]) -> list[str]:
    seen: list[str] = []
    for row in rows[1:]:
        if row[2] not in seen:
            seen.append(row[2])
    return seen


def snap(phrase: str | None, metrics: list[str]) -> str | None:
    return plan_json._snap(phrase or "", metrics)


def rule_plan(question, metrics: list[str]) -> Plan | None:
    """Derive a plan_json Plan without an LLM."""

    text = question.question
    axis_default = "nam" if len(question.years) >= 2 else "ma"

    shape = compose.screen_shape(question)
    if shape is not None:
        axis_name, filter_phrase, want_max = shape
        axis = "nam" if axis_name == "year" else "ma"
        pattern = (compose.SCREEN_YEAR_RE if axis_name == "year"
                   else compose.SCREEN_TICKER_RE)
        match = pattern.search(text)
        asked_text = text[: match.start()] if match else text
        asked_phrase = lookup_mod.extract_metric(asked_text) if match else None
        for pattern_f, num, den in ratio_mod.FORMULAS:
            if pattern_f.search(asked_text):
                metric = snap(num, metrics)
                denom = snap(den, metrics)
                filt = snap(filter_phrase, metrics)
                if metric and denom and filt and metric != filt:
                    # Asked figure is a ratio of the winner — plan_json screen
                    # only returns one metric. Prefer unresolved → LLM over a
                    # wrong one-sided screen.
                    if denom in metrics:
                        return None
        metric = snap(asked_phrase or lookup_mod.extract_metric(asked_text), metrics)
        filt = snap(filter_phrase, metrics)
        if metric and filt and metric != filt:
            return Plan(
                op="screen", metric=metric, axis=axis,
                filter_metric=filt, filter_take="max" if want_max else "min",
            )
        return None

    # Named / explicit ratio, one company.
    if len(question.tickers) == 1 and question.years:
        quot = ratio_mod.shape(question)
        if quot is not None:
            num, den = snap(quot[0], metrics), snap(quot[1], metrics)
            if num and den:
                return Plan(op="ratio", metric=num, denominator=den, axis="nam")

    elig = compose.eligible(question)
    if elig is not None:
        op, axis_name = elig
        axis = "nam" if axis_name == "year" else "ma"
        phrase = lookup_mod.extract_metric(text)
        metric = snap(phrase, metrics)
        if metric:
            return Plan(op=op, metric=metric, axis=axis)

    # Single named ratio already covered; last resort: value of one metric.
    if len(question.tickers) == 1 and len(question.years) == 1:
        phrase = lookup_mod.extract_metric(text)
        metric = snap(phrase, metrics)
        if metric and not compose.SCREEN_RE.search(text):
            return Plan(op="value", metric=metric, axis="nam")
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", default="artifacts/panel_scale.jsonl")
    parser.add_argument("--todo", default="artifacts/panel_scale_todo.jsonl",
                        help="Questions that need the 14B planner")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="Only Circular-200 inject (faster)")
    args = parser.parse_args()

    parsed = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    cube = load_panel(ROOT / "artifacts" / "metrics.parquet")
    print(f"questions={len(parsed)}  metrics_keys={len(cube)}")

    cache_path = ROOT / args.cache
    done = set()
    if cache_path.exists():
        done = {json.loads(l)["id"]
                for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo_q = []
    for q in parsed:
        if q.id in done:
            continue
        # Skip pure one-cell — lookup already owns those.
        if q.unit_scale is not None and lookup_mod.is_single_lookup(q.question):
            continue
        if not q.tickers or not q.years:
            continue
        todo_q.append(q)
    if args.limit:
        todo_q = todo_q[: args.limit]
    print(f"to process: {len(todo_q)} (cached {len(done)})")

    sources: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    written = 0
    need_llm: list[dict] = []

    with cache_path.open("a", encoding="utf-8") as handle:
        for index, question in enumerate(todo_q, start=1):
            ocr_rows = None
            keys: list = []
            phrases = panel2.expand_formulas(panel2.metric_phrases(question))
            for name in referenced_metrics(question.question):
                label = METRIC_LABEL.get(name)
                if label and label not in phrases:
                    phrases.append(label)
            if not args.skip_ocr:
                built = panel2.build(question, store, retriever, phrases=phrases or None)
                if built is not None:
                    ocr_rows = built.rows
                    keys = list(built.keys)
            metric_rows = inject_metrics(question, cube)
            rows = merge_panels(metric_rows, ocr_rows or [])
            if len(rows) < 3:  # header + <2 data
                need_llm.append({"id": question.id, "reason": "thin_panel",
                                 "phrases": phrases[:12]})
                continue

            # Persist panel for the 14B planner even when rules fail.
            panel_build = {
                "id": question.id,
                "rows": rows,
                "keys": [[k.doc_name, k.table_id] for k in keys],
                "phrases": phrases,
                "filled": len(rows) - 1,
                "wanted": max(len(phrases), 1) * max(len(question.tickers), 1)
                          * max(len(question.years), 1),
            }

            metrics = panel_metric_names(rows)
            plan = rule_plan(question, metrics)
            if plan is None:
                need_llm.append({
                    "id": question.id,
                    "reason": "no_rule_plan",
                    "metrics": metrics[:20],
                    "n_rows": len(rows) - 1,
                    "panel": panel_build,
                })
                continue

            done_pair = plan_json.execute(plan, rows, question.target_unit or "")
            if done_pair is None:
                need_llm.append({
                    "id": question.id,
                    "reason": "unresolved",
                    "plan": {
                        "op": plan.op, "metric": plan.metric, "axis": plan.axis,
                        "denominator": plan.denominator,
                        "filter_metric": plan.filter_metric,
                        "filter_take": plan.filter_take,
                    },
                    "metrics": metrics[:20],
                    "panel": panel_build,
                })
                continue
            expected, _used = done_pair
            code = plan_json.compile_query(plan, question.target_unit or "")
            if portability_problems(code):
                need_llm.append({"id": question.id, "reason": "unsafe_code",
                                 "panel": panel_build})
                continue
            outcome = run_query(code, {"df": rows})
            if (not outcome.ok or reads_no_frame(code)
                    or abs(outcome.value - expected) > 0.011):
                need_llm.append({"id": question.id, "reason": "exec_fail",
                                 "panel": panel_build})
                continue
            if impossible(question.question, outcome.value):
                need_llm.append({"id": question.id, "reason": "impossible",
                                 "panel": panel_build})
                continue

            record = {
                "id": question.id,
                "ok": True,
                "value": outcome.value,
                "code": code,
                "source": "panel_scale_rules",
                "shape": plan.op,
                "mode": "panel",
                "panel_rows": rows,
                "keys": [[k.doc_name, k.table_id] for k in keys],
                "variables": ["df"],
                "plan": {
                    "op": plan.op, "metric": plan.metric, "axis": plan.axis,
                    "denominator": plan.denominator,
                    "filter_metric": plan.filter_metric,
                    "filter_take": plan.filter_take,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            sources["panel_scale_rules"] += 1
            ops[plan.op] += 1
            if written % 20 == 0 or index == len(todo_q):
                print(f"  {index}/{len(todo_q)} ok={written} todo_llm={len(need_llm)}")

    # Materialise panels for 14B into panel2_build format (run_panel2 reads this).
    build_path = ROOT / "artifacts" / "panel2_build_scale.jsonl"
    with build_path.open("w", encoding="utf-8") as bhandle:
        n_build = 0
        for row in need_llm:
            panel = row.pop("panel", None)
            if panel is None:
                continue
            bhandle.write(json.dumps(panel, ensure_ascii=False) + "\n")
            n_build += 1
        # Also include successful ones so a full GPU re-plan can overwrite.
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("ok"):
                    continue
                bhandle.write(json.dumps({
                    "id": rec["id"],
                    "rows": rec["panel_rows"],
                    "keys": rec.get("keys") or [],
                    "phrases": [],
                    "filled": len(rec["panel_rows"]) - 1,
                    "wanted": len(rec["panel_rows"]) - 1,
                }, ensure_ascii=False) + "\n")
                n_build += 1

    todo_path = ROOT / args.todo
    with todo_path.open("w", encoding="utf-8") as handle:
        for row in need_llm:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"ok={written}  need_14B={len(need_llm)}  panels_for_gpu={n_build}")
    print(f"  cache -> {args.cache}")
    print(f"  build -> {build_path}")
    print(f"  todo  -> {args.todo}")
    print("ops:", dict(ops))
    print("todo reasons:", dict(Counter(r["reason"] for r in need_llm)))


if __name__ == "__main__":
    main()
