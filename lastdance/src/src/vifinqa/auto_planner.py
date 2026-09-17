"""Stage 0: LLM-based question understanding and FinancialPlan generation.

For each question, a 14B open-weight LLM reads the Vietnamese text plus
number-masked retrieval context and proposes a FinancialPlan JSON specifying:

- Which financial facts to retrieve (ticker, year, metric, scope, unit)
- What computation to perform (operator DAG: ratio, growth, aggregate, ...)
- Output unit and accounting conventions

The LLM never sees source numbers, answers or teacher formulas.  Its role is
semantic: understand the question, identify the required data dimensions and
select the right operators.

This module reuses the existing FinancialPlan schema, prompt templates and
validation infrastructure.  It can be run locally (if GPU available) or
remotely via Modal.

Usage:
    # Generate plans for all 1012 questions
    PYTHONPATH=src python -m vifinqa.auto_planner \\
        --questions ViFinQA/questions/questions.jsonl \\
        --companies ViFinQA/code_stock.csv \\
        --database artifacts/vifinqa.db \\
        --schema analysis/financial_plan.schema.json \\
        --output outputs/auto_plans/auto_plans.jsonl

    # Generate plans for specific questions
    PYTHONPATH=src python -m vifinqa.auto_planner \\
        --questions ViFinQA/questions/questions.jsonl \\
        --companies ViFinQA/code_stock.csv \\
        --database artifacts/vifinqa.db \\
        --schema analysis/financial_plan.schema.json \\
        --ids 1,2,3,100 \\
        --output outputs/auto_plans/auto_plans.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .financial_ir import FinancialPlanV2, financial_plan_v2_json_schema
from .l1_fact_layer import load_companies, resolve_company
from .retrieval_context import RetrievalContextBuilder, question_years, question_scope
from .shadow_ir import extract_json_object


SYSTEM_PROMPT = (
    "Bạn là semantic planner cho truy vấn tài chính tiếng Việt. "
    "Bạn chỉ sinh FinancialPlan JSON; không tính và không bịa số liệu. "
    "Không viết Python, SQL, markdown hoặc văn bản ngoài JSON."
)


def _parse_ids(value: str) -> set[int]:
    result = set()
    for token in value.split(","):
        token = token.strip().lower().removeprefix("q")
        if token:
            result.add(int(token))
    return result


def build_prompt(
    *,
    question: str,
    tickers: list[str],
    schema: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> str:
    """Build the LLM prompt for a single question.

    Reuses the proven prompt structure from the shadow IR pilot.
    """

    lines = [
        f"QUESTION: {question}",
        f"ALLOWED_TICKERS: {json.dumps(tickers, ensure_ascii=False)}",
        "Yêu cầu:",
        "- Giữ nguyên chính xác question.",
        "- Mỗi fact là một scalar cần lấy từ BCTC; không điền value/answer.",
        "- facts chỉ được dùng ticker trong ALLOWED_TICKERS.",
        "- nodes phải theo thứ tự topo và chỉ dùng operator trong schema.",
        "- Fact trực tiếp: nodes=[] và output chính là fact id; literal chỉ chứa số.",
        "- Tỷ lệ % dùng ratio_percent với 2 inputs; nhân hệ số dùng scale={factor}.",
        "- Tăng trưởng dùng percent_change inputs=[new,old], denominator absolute/reported.",
        "- 'công ty mẹ' bắt buộc scope=separate; BCTC hợp nhất dùng consolidated.",
        "- Dùng abs explicit khi câu hỏi yêu cầu độ lớn hoặc convention cần trị tuyệt đối.",
        "- output_unit phải đúng đơn vị mà câu hỏi yêu cầu. Ví dụ hỏi tỷ đồng thì "
        "output_unit=VND_1e9; fact tiền tệ trong phép cộng/trừ cũng dùng VND_1e9 "
        "hoặc phải có node scale tường minh.",
        "- unit in trên bảng chỉ mô tả nguồn; fact.unit là đơn vị tính chuẩn hóa mà "
        "binder sẽ chuyển đổi sang.",
        "- conventions không thay thế DAG: expense_sign=absolute vẫn phải dùng abs "
        "nếu ô chi phí được báo cáo âm.",
        "- generator ghi đúng tên model của bạn; confidence phản ánh độ chắc chắn.",
        "Mọi fact bắt buộc có source_preference=auto|primary_statement|note_table.",
        "Operator params: literal={value}; scale={factor}; "
        "percent_change={denominator}; vector={labels}; "
        "filter/filter_by/count_if={comparator,threshold}; "
        "top_k/bottom_k={k}; round={digits}; operator khác params={}.",
        "Trả về đúng một JSON object tuân thủ schema sau:",
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
    ]
    if context is not None:
        lines.extend(
            [
                "RETRIEVAL_CONTEXT dưới đây đã che mọi numeric cell:",
                "- Mỗi fact phải chọn row_ref có thật và metric phải sao chép label tương ứng.",
                "- Ghi assumption dạng grounding:<fact_id>=<row_ref> cho từng fact.",
                "- table_kind/unit/header giúp chọn scope, kỳ và đơn vị; không tự bịa row_ref.",
                "- Không đồng nhất tên dòng với nghĩa câu hỏi: kiểm tra đủ mọi thành phần "
                "(ví dụ 'ngoại bảng', 'liên quan', 'ròng') trước khi chốt facts.",
                json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    return "\n".join(lines)


def validate_response(
    *,
    raw_response: str,
    question: str,
    model: str,
) -> dict[str, Any]:
    """Parse and validate an LLM response into a structured result row."""

    row: dict[str, Any] = {
        "model": model,
        "status": "INVALID",
        "raw_response": raw_response,
    }
    try:
        payload = extract_json_object(raw_response)
        plan = FinancialPlanV2.from_dict(payload)
        row["status"] = "VALID"
        row["plan"] = plan.to_dict()
        row["fact_count"] = len(plan.facts)
        row["node_count"] = len(plan.nodes)
    except Exception as exc:
        row["error"] = str(exc)[:500]
    return row


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load the BTC question set."""

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def build_retrieval_context(
    *,
    builder: RetrievalContextBuilder,
    question_id: int,
    question: str,
    tickers: set[str],
) -> dict[str, Any] | None:
    """Build number-masked retrieval context for one question.

    Returns None if no tables are found (e.g., missing data).
    """

    try:
        context = builder.build(
            question_id=question_id,
            question=question,
            tickers=tickers,
            table_limit=8,
            rows_per_table=12,
        )
        if not context.get("tables"):
            return None
        return context
    except Exception:
        return None


def generate_plans_local(
    prompts: list[str],
    model_id: str = "Qwen/Qwen2.5-14B-Instruct",
    max_new_tokens: int = 2400,
) -> list[str]:
    """Generate plans using a local GPU.  Requires torch + transformers."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    responses = []
    for index, prompt in enumerate(prompts, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
        print(f"  [{index}/{len(prompts)}] q done", flush=True)
    return responses


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated question IDs; empty = all")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--backend", choices=["local", "precomputed", "generate-prompts"],
                        default="precomputed",
                        help="'local' runs LLM on GPU; 'precomputed' reads "
                             "from a pre-generated JSONL; 'generate-prompts' dumps prompts to JSON.")
    parser.add_argument("--precomputed-plans", type=Path, default=None,
                        help="Path to precomputed raw responses (JSON array of strings) from Modal run")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    companies = load_companies(args.companies)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    selected_ids = _parse_ids(args.ids) if args.ids else None

    if selected_ids:
        questions = [q for q in questions if int(q["id"]) in selected_ids]

    print(f"Processing {len(questions)} questions", flush=True)

    if args.backend == "precomputed" and args.precomputed_plans:
        # Read pre-generated structured responses from Modal run
        raw_responses = json.loads(args.precomputed_plans.read_text(encoding="utf-8"))
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for raw in raw_responses:
            # We look up the question text
            q_text = next((q["question"] for q in questions if int(q["id"]) == raw["question_id"]), "")
            row = validate_response(
                raw_response=raw["raw_response"],
                question=q_text,
                model=raw["model"],
            )
            rows.append(row)

        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        valid = [r for r in rows if r["status"] == "VALID"]
        print(json.dumps({
            "output": str(args.output),
            "total": len(rows),
            "valid": len(valid),
            "invalid": len(rows) - len(valid),
        }, ensure_ascii=False))
        return

    # Build retrieval contexts and prompts
    prompts = []
    prompt_questions = []

    with RetrievalContextBuilder(args.database) as builder:
        for question in questions:
            qid = int(question["id"])
            question_text = question["question"]

            # Resolve tickers for this question
            tickers = set()
            for field in ("ticker", "company_name"):
                if field in question and question[field]:
                    if isinstance(question[field], list):
                        for value in question[field]:
                            resolved = resolve_company(companies, value)
                            if resolved:
                                tickers.add(resolved)
                    else:
                        resolved = resolve_company(companies, str(question[field]))
                        if resolved:
                            tickers.add(resolved)

            if not tickers:
                # Fallback: try to extract ticker from question text
                for company in companies:
                    if company.ticker in question_text:
                        tickers.add(company.ticker)

            context = build_retrieval_context(
                builder=builder,
                question_id=qid,
                question=question_text,
                tickers=tickers,
            )

            prompt = build_prompt(
                question=question_text,
                tickers=sorted(tickers),
                schema=schema,
                context=context,
            )
            prompts.append({
                "question_id": qid,
                "prompt": prompt,
            })
            prompt_questions.append(question)

    print(f"Built {len(prompts)} prompts...", flush=True)

    if args.backend == "generate-prompts":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(prompts)} prompts to {args.output}")
        return

    print("Calling LLM locally (Qwen only)...", flush=True)
    local_prompt_texts = [p["prompt"] for p in prompts]
    responses = generate_plans_local(
        local_prompt_texts, model_id=args.model, max_new_tokens=args.max_new_tokens
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for question, raw in zip(prompt_questions, responses, strict=True):
        row = validate_response(
            raw_response=raw,
            question=question["question"],
            model=args.model,
        )
        rows.append(row)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid = [r for r in rows if r["status"] == "VALID"]
    print(json.dumps({
        "output": str(args.output),
        "total": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
