"""Probe: regex rule_plan vs LLM plan_json on nested questions regex botched.

Usage (vLLM tunneled to localhost:18000):
  PYTHONPATH=src python scripts/probe_plan_vs_regex.py --local-url http://127.0.0.1:18000/v1 \
      --model Qwen/Qwen3-14B
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import plan_json  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

spec_rp = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_panel2.py")
rp = importlib.util.module_from_spec(spec_rp)
spec_rp.loader.exec_module(rp)

spec_sc = importlib.util.spec_from_file_location("scale", ROOT / "scripts" / "run_panel_scale.py")
scale = importlib.util.module_from_spec(spec_sc)
spec_sc.loader.exec_module(scale)

HARD_IDS = [449, 387, 520, 552, 512, 371, 495, 503, 667, 457]


def fmt_plan(plan) -> str:
    if plan is None:
        return "None"
    return (
        f"op={plan.op} metric={plan.metric!r} filter={plan.filter_metric!r} "
        f"den={plan.denominator!r} axis={plan.axis} take={plan.filter_take}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--build", default="artifacts/panel2_build_scale.jsonl")
    args = parser.parse_args()

    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data" / "questions" / "questions.jsonl",
            ROOT / "data" / "code_stock.csv",
        )
    }
    panels = {}
    for line in (ROOT / args.build).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            panels[row["id"]] = row

    client = ChatClient.local(args.model, args.local_url, max_tokens=400)
    print(f"model={args.model}")

    for qid in HARD_IDS:
        q = parsed.get(qid)
        panel = panels.get(qid)
        if q is None:
            print(f"q{qid}: missing question")
            continue
        metrics = scale.panel_metric_names(panel["rows"]) if panel else []
        regex_plan = scale.rule_plan(q, metrics) if metrics else None
        print("=" * 72)
        print(f"q{qid}: {q.question[:150]}")
        print(f"  regex: {fmt_plan(regex_plan)}")
        if panel is None:
            print("  model: no prebuilt panel")
            continue
        try:
            reply = client.complete(rp.SYSTEM, rp.build_user(q, panel["rows"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  model ERROR: {exc}")
            continue
        plan = plan_json.parse(reply, metrics=metrics)
        print(f"  raw: {reply[-180:].replace(chr(10), ' | ')}")
        print(f"  model: {fmt_plan(plan)}")
        if plan is not None:
            done = plan_json.execute(plan, panel["rows"], q.target_unit or "")
            print(f"  exec: {done[0] if done else None}")


if __name__ == "__main__":
    main()
