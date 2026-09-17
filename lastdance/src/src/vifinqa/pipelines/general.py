"""General pipeline for ViFinQA (handling new questions without ID lock)."""

from typing import Optional
from pathlib import Path
import json
import sys
import os
import requests

from ..semantic_parser import parse_question
from ..retrieval_context import RetrievalContextBuilder
from ..auto_planner import build_prompt
from ..financial_ir import FinancialPlanV2, compile_pandas, evaluate_plan
from ..grounded_plans import GroundedBinder
from ..plan_verifier import llm_plan_to_execution_item

def run_general(
    questions_file: Optional[Path],
    output_dir: Path,
    database_path: Path,
) -> None:
    questions = []
    interactive = False
    
    if questions_file and questions_file.exists():
        with open(questions_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    questions.append(json.loads(line)["question"])
                else:
                    questions.append(line)
    else:
        interactive = True
        print("🚀 Bắt đầu Interactive Mode (Nhánh General) - Gõ 'exit' hoặc 'quit' để thoát")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    modal_url = os.environ.get("MODAL_API_URL")
    
    if interactive and modal_url:
        print(f"🔗 Đã kết nối Modal LLM Endpoint: {modal_url}")
    elif interactive:
        print("⚠️ Không tìm thấy MODAL_API_URL. Sẽ chạy chế độ MVP Stub (Không gọi LLM).")
        
    schema_path = Path("analysis/financial_plan.schema.json")
    if not schema_path.exists():
        # Fallback empty schema if not strictly running from project root
        schema = {}
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

    with RetrievalContextBuilder(database_path) as builder:
        i = 0
        while True:
            q_list = questions
            if interactive:
                try:
                    q = input("\n[❓] Câu hỏi: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in ("exit", "quit"):
                    break
                if not q:
                    continue
                q_list = [q]
            else:
                if i >= len(questions):
                    break
                
            for q in q_list:
                spec = parse_question(q)
                tickers = {ent.ticker for ent in spec.entities if ent.ticker}
                try:
                    ctx = builder.build(
                        question_id=i+1,
                        question=q,
                        tickers=tickers,
                        question_spec=spec,
                    )
                except Exception as e:
                    print(f"❌ Retrieval Error: {e}")
                    ctx = {"tables": []}
                
                print(f"\n🔍 [SEMANTIC PARSER]")
                print(f"   Entities: {spec.entities}")
                print(f"   Periods:  {spec.periods}")
                print(f"   Scope:    {spec.scope}")
                print(f"   Unit:     {spec.output_unit}")
                
                print(f"\n📄 [RETRIEVAL]")
                tables = ctx.get("tables", [])
                if tables:
                    print(f"   Tìm thấy {len(tables)} bảng liên quan.")
                    for t in tables[:2]:  # Show top 2
                        print(f"   - {t['table_id']} (Score: {t.get('score', 0):.2f})")
                else:
                    print("   Không tìm thấy bảng dữ liệu.")

                if modal_url:
                    prompt = build_prompt(
                        question=q,
                        tickers=sorted(tickers),
                        schema=schema,
                        context=ctx,
                    )
                    print(f"\n🧠 [LLM PLANNER]")
                    print("   Đang gọi API Modal...")
                    try:
                        resp = requests.post(modal_url, json={"prompts": [prompt], "max_tokens": 2400})
                        resp.raise_for_status()
                        raw_json = resp.json()["responses"][0]
                        # Trim any markdown code blocks
                        if raw_json.startswith("```json"):
                            raw_json = raw_json[7:-3]
                        
                        plan = FinancialPlanV2.from_dict(json.loads(raw_json))
                        print(f"   [Thành công] Generated Plan: {plan.facts[0].id if plan.facts else 'No facts'}")
                        
                        print(f"\n⚙️ [GROUNDING & EXECUTION]")
                        try:
                            with GroundedBinder(database_path) as binder:
                                bound_plan = binder.bind_plan(plan)
                                execution = compile_pandas(bound_plan)
                                
                                # Dummy dict mapping variables to dummy values for evaluating the DAG
                                # In reality, we need to execute the pandas script to get `fact_values`
                                print(f"   Pandas Code:\n{execution.pandas_query}")
                        except Exception as exec_err:
                            print(f"   [Execution Error]: {exec_err}")

                    except Exception as e:
                        print(f"   [Lỗi gọi LLM API]: {e}")
                else:
                    # MVP: Single-fact lookup plan stub
                    if len(spec.entities) == 1 and len(spec.periods) == 1:
                        from ..financial_ir import FactRequest, PlanNode
                        
                        ticker = spec.entities[0].ticker or "UNKNOWN"
                        year = spec.periods[0].year or 2023
                        
                        plan = FinancialPlanV2(
                            question=q,
                            facts=(
                                FactRequest(
                                    id="f1", 
                                    ticker=ticker, 
                                    year=year, 
                                    metric="UNKNOWN_METRIC", 
                                    scope=spec.scope, 
                                    period="end_or_flow", 
                                    unit=spec.output_unit
                                ),
                            ),
                            nodes=(PlanNode(id="n1", op="identity", inputs=("f1",)),),
                            output="n1",
                            output_unit=spec.output_unit,
                            generator="general_single_fact_lookup"
                        )
                        print(f"\n⚙️ [PLANNER STUB]")
                        print(f"   Plan JSON: {plan.to_json()}")
                print("-" * 50)
                
            if not interactive:
                i += 1
            else:
                pass
