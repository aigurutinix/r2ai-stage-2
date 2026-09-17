"""Independently review and re-execute generated panel-answer programs."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.llm_client import chat
from kingpro.financial.panel_metrics import PanelMetricEngine
from kingpro.financial.statement_cube import FinancialCube
from scripts.solve_panel_questions import (
    CUBE_PATH,
    QUESTIONS_PATH,
    SYSTEM,
    execute_code,
    extract_tickers,
    extract_years,
    load_env,
    panel_csv,
    parse_response,
    relevant_metrics,
)

REVIEW_SYSTEM = SYSTEM + """

Bạn đang REVIEW code của một người khác. Code cũ có thể chạy nhưng sai logic.
Hãy tự giải lại từ câu hỏi và CSV, kiểm tra kỹ tập lọc, năm, mẫu số, chỉ tiêu ở mệnh đề cuối.
Viết code mới độc lập; đừng mặc định code cũ đúng.
"""


def latest_rows(path: Path) -> dict[int, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["id"])] = row
    return rows


def questions() -> dict[int, str]:
    rows = {}
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[int(row["id"])] = row["question"]
    return rows


def review_one(qid, original, question, engine, known, base_url, model, api_key):
    tickers = extract_tickers(question, known)
    available = {
        int(year)
        for ticker in tickers
        for year in engine.cube.data.get(ticker, {})
        if engine.scope in engine.cube.data[ticker][year]
    }
    years = extract_years(question, available)
    metrics = relevant_metrics(question)
    context = panel_csv(engine, tickers, years, metrics)
    frame = pd.read_csv(io.StringIO(context))
    user = (
        f"CÂU HỎI:\n{question}\n\nDỮ LIỆU PANEL:\n{context}\n\n"
        f"CODE CŨ CẦN REVIEW:\n{original.get('code', '')}\n"
        f"Kết quả code cũ: {original.get('answer')}"
    )
    last_error = ""
    for attempt in range(1, 3):
        repair = f"\n\nCode review trước lỗi: {last_error}. Viết lại." if last_error else ""
        try:
            raw = chat(
                REVIEW_SYSTEM,
                user + repair,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=0,
                max_tokens=900,
                timeout=180,
            )
            parsed = parse_response(raw)
            answer = execute_code(parsed["code"], frame)
            return {
                "id": qid,
                "ok": True,
                "answer": answer,
                "original_answer": original.get("answer"),
                "agrees": answer == original.get("answer"),
                "attempts": attempt,
                "code": parsed["code"],
            }
        except Exception as exc:
            last_error = str(exc)[:300]
    return {"id": qid, "ok": False, "error": last_error, "original_answer": original.get("answer")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "build" / "panel_answers_v4.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "panel_review_v4.jsonl")
    parser.add_argument("--start", type=int, default=362)
    parser.add_argument("--end", type=int, default=420)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    load_env()
    cube = FinancialCube.read_jsonl(CUBE_PATH)
    engine = PanelMetricEngine(cube)
    known = set(cube.data)
    source = latest_rows(args.input)
    existing = latest_rows(args.output)
    qs = questions()
    ids = [qid for qid in range(args.start, args.end + 1) if source.get(qid, {}).get("ok") and qid not in existing]
    base_url = os.environ.get("BASE_CODER") or os.environ.get("KINGPRO_LLM_BASE_URL", "")
    model = os.environ.get("MODEL_CODER") or os.environ.get("KINGPRO_LLM_MODEL", "")
    api_key = os.environ.get("KINGPRO_LLM_API_KEY", "EMPTY")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(review_one, qid, source[qid], qs[qid], engine, known, base_url, model, api_key): qid
                for qid in ids
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(ids)}] Q{row['id']} ok={row.get('ok')} "
                    f"answer={row.get('answer')} agree={row.get('agrees')} error={row.get('error', '')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
