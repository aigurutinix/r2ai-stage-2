"""Solve generated panel questions against the normalized statement cube.

The LLM only interprets the Vietnamese operation chain.  Every financial value
and derived ratio in its prompt is computed deterministically by local code.
Results are cached as JSONL and include the selected entities/years for audit.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import builtins
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.llm_client import chat
from kingpro.financial.panel_metrics import PanelMetricEngine
from kingpro.financial.statement_cube import FinancialCube
from kingpro.retrieval.bm25_index import fold

CUBE_PATH = ROOT / "build" / "statement_cube.jsonl"
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.jsonl"
DEFAULT_CACHE = ROOT / "build" / "panel_answers_v4.jsonl"

ALIASES = {
    "hoa phat": "HPG",
    "hoa sen": "HSG",
    "nam kim": "NKG",
    "dam phu my": "DPM",
    "dam ca mau": "DCM",
    "vincom retail": "VRE",
    "vingroup": "VIC",
    "kinh bac": "KBC",
    "masan": "MSN",
    "vinamilk": "VNM",
    "dabaco": "DBC",
    "sao mai": "ASM",
    "minh phu": "MPC",
    "dai duong": "OGC",
    "duong quang ngai": "QNS",
    "binh son": "BSR",
    "pvtrans": "PVT",
    "the gioi di dong": "MWG",
    "dau khi ca mau": "DCM",
    "vicem ha tien": "HT1",
    "van phu": "VPI",
    "hai phat": "HPX",
    "mobiworld": "MWG",
    "fpt": "FPT",
    "phat trien do thi kinh bac": "KBC",
}

SYSTEM = """Bạn là bộ biên dịch câu hỏi tài chính sang pandas code chính xác.
Dữ liệu CSV đã được máy chuẩn hóa từ BCTC hợp nhất; không tự bịa hay thay số.
Các cột kết thúc bằng _pct hoặc _pp đã là phần trăm/điểm phần trăm (không nhân 100 lần nữa).
Median là trung vị chuẩn; điều kiện 'cao hơn/thấp hơn' là bất đẳng thức nghiêm ngặt.
Thực hiện đúng thứ tự lọc -> chọn max/min -> lấy chỉ tiêu đích. Làm tròn 2 chữ số thập phân.
Chỉ tiêu được hỏi ở mệnh đề CUỐI là kết quả; các chỉ tiêu trước đó chỉ dùng để lọc/chọn.
Ví dụ: "năm D/E cao nhất, hệ số thanh toán lãi vay là bao nhiêu" phải trả interest_coverage, không trả D/E.
"Năm ngay sau năm đầu tiên X" nghĩa là tìm năm đầu tiên thỏa X rồi cộng đúng 1 năm.
QUY TẮC CODE BẮT BUỘC:
- Median/trung vị phải tính trên đúng danh sách ticker và ĐÚNG năm được nêu, không lấy lẫn năm khác.
- "Tỷ trọng tổng X của nhóm lọc" = sum(X của nhóm lọc) / sum(X của TOÀN nhóm) * 100; không chia từng dòng cho tổng nhóm lọc.
- Điều kiện duy trì ở tất cả các năm: groupby ticker và chỉ giữ ticker có all(condition) cùng đủ số năm.
- CAGR từ năm A đến B dùng số khoảng thời gian B-A: (end/start)**(1/(B-A))-1.
- Các cột revenue_growth_pct và gross_margin_change_pp tại năm Y đã là thay đổi từ Y-1 sang Y; dùng trực tiếp, không merge/diff lại.
- Khi so sánh hai năm, lọc từng năm theo ticker hoặc pivot; không so hai Series mang index khác nhau.
- Không dùng groupby.apply nếu có thể dùng sort/drop_duplicates/groupby trực tiếp.
Không tự tính nhẩm đáp án. Viết code chạy trên DataFrame tên df và GÁN kết quả số cuối cùng vào biến result.
Chỉ dùng các cột có trong CSV; không import, không đọc file, không mạng.
Chỉ trả code nằm giữa đúng hai marker sau, không giải thích, không markdown:
CODE_START
filtered = ...
result = ...
CODE_END
"""

# Override the legacy prompt above, which was copied through a non-UTF-8
# terminal and became mojibake.  Keeping the instructions readable materially
# improves operation-order fidelity on long Vietnamese panel questions.
SYSTEM = """Bạn là bộ biên dịch câu hỏi tài chính tiếng Việt sang pandas code chính xác.
Dữ liệu CSV đã được máy chuẩn hóa từ BCTC hợp nhất; không tự bịa hoặc thay số.
Các cột kết thúc bằng _pct hoặc _pp đã là phần trăm/điểm phần trăm, không nhân 100 lần nữa.
Median/trung vị là trung vị chuẩn. Điều kiện cao hơn/thấp hơn là bất đẳng thức nghiêm ngặt.

Thực hiện đúng thứ tự: lọc năm và ticker -> tính điều kiện -> chọn max/min -> lấy chỉ tiêu đích.
Chỉ tiêu được hỏi ở MỆNH ĐỀ CUỐI là kết quả; các chỉ tiêu trước chỉ dùng để lọc hoặc chọn.
Ví dụ: “năm D/E cao nhất, hệ số thanh toán lãi vay là bao nhiêu” phải trả interest_coverage, không trả D/E.
“Năm liền sau năm đầu tiên X” nghĩa là tìm năm đầu tiên thỏa X rồi cộng đúng 1 năm.

QUY TẮC BẮT BUỘC:
- Median phải tính trên đúng danh sách ticker và đúng năm được nêu, không trộn năm khác.
- “Tỷ trọng tổng X của nhóm lọc” = sum(X nhóm lọc) / sum(X toàn nhóm) * 100.
- Điều kiện đúng trong mọi năm: groupby ticker, giữ ticker đủ số năm và all(condition).
- CAGR từ A đến B có B-A khoảng thời gian: (end/start)**(1/(B-A))-1.
- revenue_growth_pct và gross_margin_change_pp ở năm Y đã là thay đổi từ Y-1 sang Y.
- Khi so sánh hai năm, pivot hoặc merge theo ticker; không so hai Series có index lệch nhau.
- “Từ A đến B thay đổi bao nhiêu điểm phần trăm” = giá trị tại B trừ giá trị tại A.
- “Bình quân” là mean của đúng tập sau khi lọc; “cao nhất/thấp nhất” phải chọn đúng dòng rồi mới lấy metric đích.
- Không dùng groupby.apply nếu có thể dùng sort, pivot, merge, transform hoặc agg.

Viết code chạy trên DataFrame tên df và GÁN đúng một kết quả số cuối cùng vào biến result.
Chỉ dùng các cột trong CSV; không import, không đọc file, không mạng.
Làm tròn 2 chữ số thập phân.
Chỉ trả code giữa đúng hai marker, không giải thích, không markdown:
CODE_START
filtered = ...
result = ...
CODE_END
"""


METRIC_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("bien loi nhuan gop", "loi nhuan gop tren doanh thu"), ("gross_margin_pct",)),
    (("bien loi nhuan rong",), ("net_margin_pct",)),
    (("bien loi nhuan hoat dong",), ("operating_margin_pct",)),
    (("d/e", "no phai tra tren von chu", "no phai tra chia cho von chu", "don bay tai chinh"), ("liabilities_to_equity",)),
    (("no phai tra tren tong tai san", "no tren tong tai san"), ("liabilities_to_assets_pct",)),
    (("thanh toan hien hanh",), ("current_ratio",)),
    (("thanh toan nhanh",), ("quick_ratio",)),
    (("thanh toan lai vay", "ebit"), ("interest_coverage", "pbt", "interest_expense")),
    (("hang ton kho/no ngan han", "hang ton kho tren no ngan han"), ("inventory_to_current_liabilities",)),
    (("cfo tren no ngan han", "dong tien hoat dong tren no ngan han"), ("operating_cash_flow_ratio",)),
    (("cfo margin", "cfo tren doanh thu", "dong tien hoat dong tren doanh thu", "luu chuyen tien thuan tu hoat dong kinh doanh tren doanh thu"), ("cfo_margin_pct",)),
    (("cfo tren lnst", "cfo/lnst", "chuyen doi loi nhuan"), ("cfo_to_npat",)),
    (("ty trong hang ton kho", "hang ton kho tren tong tai san"), ("inventory_to_assets_pct",)),
    (("tai san dai han tren tong tai san",), ("long_term_assets_share_pct",)),
    (("chi phi ban hang va chi phi quan ly", "sg&a"), ("sga_expense", "sga_intensity_pct")),
    (("tang truong doanh thu", "doanh thu thuan tang", "doanh thu thuan giam", "cagr doanh thu"), ("revenue_growth_pct",)),
    (("vong quay tong tai san",), ("asset_turnover_avg",)),
    (("roe", "loi nhuan sau thue chiem bao nhieu phan tram von chu"), ("roe_pct",)),
    (("roa",), ("roa_pct",)),
    (("so ngay ton kho", "doh"), ("inventory_days",)),
    (("von luu dong rong",), ("net_working_capital",)),
    (("don tich",), ("operating_accruals_ratio_pct",)),
    (("muc thay doi bien loi nhuan gop", "chenh lech bien loi nhuan gop"), ("gross_margin_change_pp",)),
    (("luu chuyen tien thuan tu hoat dong kinh doanh", "cfo duong", "cfo am"), ("cfo",)),
    (("loi nhuan sau thue", "lnst"), ("npat",)),
    (("loi nhuan truoc thue", "lntt"), ("pbt",)),
    (("chi phi lai vay",), ("interest_expense",)),
    (("loi nhuan thuan tu hoat dong kinh doanh",), ("operating_profit",)),
    (("hang ton kho",), ("inventory",)),
    (("no ngan han",), ("current_liabilities",)),
    (("tai san ngan han",), ("current_assets",)),
    (("tai san dai han",), ("long_term_assets",)),
    (("tong tai san",), ("total_assets",)),
    (("von chu so huu",), ("equity",)),
    (("doanh thu thuan",), ("revenue",)),
)


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def load_questions() -> dict[int, str]:
    out = {}
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[int(row["id"])] = row["question"]
    return out


def extract_tickers(question: str, known: set[str]) -> list[str]:
    explicit = [token for token in re.findall(r"\b[A-Z]{3}\b", question) if token in known]
    normalized = fold(question)
    aliases = [ticker for alias, ticker in ALIASES.items() if alias in normalized and ticker in known]
    return list(dict.fromkeys([*explicit, *aliases]))


def extract_years(question: str, available: set[int]) -> list[int]:
    mentioned = sorted({int(value) for value in re.findall(r"20\d{2}", question)})
    if not mentioned:
        return sorted(available)
    if len(mentioned) >= 2:
        years = list(range(min(mentioned), max(mentioned) + 1))
    else:
        years = mentioned
    return [year for year in dict.fromkeys(years) if year in available]


def relevant_metrics(question: str) -> list[str]:
    normalized = fold(question)
    selected: list[str] = []
    for phrases, metrics in METRIC_HINTS:
        if any(phrase in normalized for phrase in phrases):
            selected.extend(metrics)
    return list(dict.fromkeys(selected))


def panel_csv(engine: PanelMetricEngine, tickers: list[str], years: list[int], metrics: list[str]) -> str:
    rows = [
        {"ticker": ticker, "year": str(year), **{metric: engine.value(ticker, year, metric) for metric in metrics}}
        for ticker in tickers
        for year in years
    ]
    if not rows:
        return ""
    fields = list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        compact = {}
        for key, value in row.items():
            if isinstance(value, float):
                compact[key] = "" if not math.isfinite(value) else f"{value:.10g}"
            else:
                compact[key] = "" if value is None else value
        writer.writerow(compact)
    return stream.getvalue()


def parse_response(text: str) -> dict:
    marker = re.search(r"CODE_START\s*(.*?)\s*CODE_END", text, flags=re.DOTALL | re.IGNORECASE)
    if marker:
        code = marker.group(1).strip()
        if "result" not in code:
            raise ValueError("model response has no result code")
        return {"code": code}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model response has no JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload.get("code"), str) or "result" not in payload["code"]:
        raise ValueError("model response has no result code")
    return payload


def execute_code(code: str, frame: pd.DataFrame) -> float:
    code = "\n".join(
        line for line in code.splitlines()
        if line.strip() not in ("import pandas as pd", "import numpy as np")
    )
    lowered = code.casefold()
    forbidden = ("import ", "open(", "exec(", "eval(", "compile(", "__", "subprocess", "socket", "requests", "urllib", "pathlib", "os.", "sys.")
    if any(token in lowered for token in forbidden):
        raise ValueError("unsafe generated code")
    allowed_builtins = {
        name: getattr(builtins, name)
        for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")
    }
    globals_dict = {"__builtins__": allowed_builtins, "pd": pd, "np": np}
    locals_dict = {"df": frame.copy()}
    exec(code, globals_dict, locals_dict)
    answer = float(locals_dict["result"])
    if not math.isfinite(answer):
        raise ValueError("non-finite answer")
    return round(answer, 2)


def solve_one(
    qid: int,
    question: str,
    engine: PanelMetricEngine,
    known: set[str],
    model: str,
    base_url: str,
    api_key: str,
) -> dict:
    tickers = extract_tickers(question, known)
    if not tickers:
        return {"id": qid, "ok": False, "error": "no_ticker", "question": question}
    available = {
        int(year)
        for ticker in tickers
        for year in engine.cube.data.get(ticker, {})
        if engine.scope in engine.cube.data[ticker][year]
    }
    years = extract_years(question, available)
    metrics = relevant_metrics(question)
    if not metrics:
        return {"id": qid, "ok": False, "error": "no_metric", "question": question, "tickers": tickers, "years": years}
    context = panel_csv(engine, tickers, years, metrics)
    if not context:
        return {"id": qid, "ok": False, "error": "no_panel", "question": question}
    user = f"CÂU HỎI:\n{question}\n\nDỮ LIỆU PANEL:\n{context}"
    frame = pd.read_csv(io.StringIO(context))
    previous_raw = ""
    previous_error = ""
    for attempt in range(1, 3):
        repair = ""
        if previous_error:
            repair = (
                "\n\nLẦN TRƯỚC LỖI. Hãy viết lại code đơn giản hơn. "
                f"Lỗi chạy: {previous_error}\nPhản hồi trước:\n{previous_raw[:2500]}"
            )
        try:
            raw = chat(
                SYSTEM,
                user + repair,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=0,
                max_tokens=850,
                timeout=180,
            )
            parsed = parse_response(raw)
            answer = execute_code(parsed["code"], frame)
            return {
                "id": qid,
                "ok": True,
                "question": question,
                "tickers": tickers,
                "years": years,
                "metrics": metrics,
                "attempts": attempt,
                **parsed,
                "answer": answer,
            }
        except Exception as exc:
            previous_raw = locals().get("raw", "")
            previous_error = str(exc)[:300]
    return {
        "id": qid,
        "ok": False,
        "error": previous_error,
        "question": question,
        "tickers": tickers,
        "years": years,
        "metrics": metrics,
        "raw_response": previous_raw[:3000],
    }


def read_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[int(row["id"])] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=362)
    parser.add_argument("--end", type=int, default=506)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    cube = FinancialCube.read_jsonl(CUBE_PATH)
    engine = PanelMetricEngine(cube)
    questions = load_questions()
    known = set(cube.data)
    todo_ids = [qid for qid in range(args.start, args.end + 1) if qid in questions]

    if args.dry_run:
        for qid in todo_ids:
            tickers = extract_tickers(questions[qid], known)
            available = {int(year) for ticker in tickers for year in cube.data.get(ticker, {})}
            print(qid, tickers, extract_years(questions[qid], available))
        return

    base_url = os.environ.get("BASE_CODER") or os.environ.get("KINGPRO_LLM_BASE_URL", "")
    model = os.environ.get("MODEL_CODER") or os.environ.get("KINGPRO_LLM_MODEL", "")
    api_key = os.environ.get("KINGPRO_LLM_API_KEY", "EMPTY")
    if not base_url or not model:
        raise SystemExit("Missing BASE_CODER/MODEL_CODER configuration")

    done = read_cache(args.cache)
    todo_ids = [qid for qid in todo_ids if not done.get(qid, {}).get("ok")]
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a", encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(solve_one, qid, questions[qid], engine, known, model, base_url, api_key): qid
                for qid in todo_ids
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                status = f"answer={row['answer']}" if row.get("ok") else f"error={row.get('error')}"
                print(f"[{index}/{len(todo_ids)}] Q{row['id']} {status}", flush=True)


if __name__ == "__main__":
    main()
