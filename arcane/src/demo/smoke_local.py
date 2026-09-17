"""Run the whole demo pipeline on real data with the two model calls stubbed.

This is the pre-demo check that needs no GPU: it proves the corpus loads, the
gate resolves, retrieval and selection run, and a cited answer executes from the
source cells. Usage:

    VIFINQA_DATA_ROOT=../data/raw/vifinqa PYTHONPATH=src:. python demo/smoke_local.py
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("PLANNER_URL", "http://stub")
os.environ.setdefault("RERANK_URL", "http://stub")

from vifinqa.demo_pipeline import DemoPipeline, Plan, PlanFact  # noqa: E402


QUESTION = "Lãi tiền gửi của VJC năm 2018 trên báo cáo riêng là bao nhiêu tỷ đồng?"


def stub(pipeline: DemoPipeline) -> None:
    """Stand in for the planner and the reranker without changing the real path."""
    plan = {"facts": [{"id": "f1", "query": "Lãi tiền gửi", "entity": "VJC", "period": 2018}], "steps": []}

    async def chat(url, prompt, model):
        if "expression" in prompt:
            row, column = _first_numeric_cell(prompt)
            return {"expression": f"result = float(df0.iloc[{row}, {column}])"}
        return plan

    async def rerank(question, plan_obj, record):
        return {table["table_id"]: 1.0 / (1 + rank)
                for rank, table in enumerate(record["candidates"])}

    pipeline._chat = chat
    pipeline._rerank = rerank


def _first_numeric_cell(prompt: str) -> tuple[int, int]:
    """Pick a masked numeric cell out of the prompt the way a model would pick one."""
    for line in prompt.splitlines():
        if not line.startswith("ROW "):
            continue
        parts = [part.strip() for part in line.split(" | ")]
        row = int(parts[0].split()[1])
        for column, value in enumerate(parts[1:], 0):
            if value == "<NUM>":
                return row, column
    raise SystemExit("no masked numeric cell in the evidence prompt")


async def main() -> int:
    pipeline = DemoPipeline.from_env()
    stub(pipeline)
    result = None
    async for event in pipeline.run(QUESTION, "smoke"):
        print(f"[{event['stage']:<9}] {event['status']:<6} {event['message']}")
        if event["type"] == "error":
            return 1
        if event["type"] == "result":
            result = event["data"]
    if result is None:
        return 1
    print("answer   :", result["answer"])
    print("code     :", result["code"])
    for citation in result["citations"]:
        print("cite     :", citation["id"], citation["row_label"], "|",
              citation["column_label"], "=", citation["raw_value"])
        assert pipeline.evidence("smoke", citation["id"])["highlight"]["row"] == citation["row"]
    print("latency  :", result["latency_seconds"], "s (no model time)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
