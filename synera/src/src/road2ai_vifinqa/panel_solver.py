"""LLM-assisted, execution-checked compiler for grounded panel questions."""

from __future__ import annotations

import ast
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .local_llm import chat, extract_json
from .panel import (
    PANEL_COLUMN_GUIDE,
    RAW_COLUMNS,
    FinancialPanel,
    infer_panel_tickers,
    infer_panel_years,
)
from .text import fold_text


SYSTEM_PROMPT = """Bạn là compiler tài chính deterministic. Chuyển câu hỏi tiếng Việt thành
MỘT biểu thức pandas chạy trên DataFrame `df`. Chỉ trả về đúng một JSON object, không markdown:
{"pandas_query":"<expression>","required_raw_columns":["..."],"note":"..."}

Quy tắc bắt buộc:
- `df` có một dòng cho mỗi ticker-year. Tiền tệ luôn là VND. Các cột margin/ROA/ROE/growth
  đã là phần trăm; ratio/lần giữ dạng số lần.
- Lọc, median, argmin/argmax phải dùng full precision. Không round trung gian.
- Khoảng năm là bao gồm cả hai đầu. So sánh "trên/cao hơn/lớn hơn" dùng >; "dưới/thấp hơn" dùng <.
- `Series.median()` là trung vị pandas. Trung bình dùng `.mean()`.
- Nếu hỏi tỷ trọng tổng của nhóm lọc, tính sum(filtered)/sum(all)*100, không lấy mean tỷ lệ.
- Nếu hỏi năm, trả về `int`; số doanh nghiệp/năm trả về `int`; còn lại trả scalar float.
- Không import, không đọc file, không dùng eval/exec, không tạo code block.
- `required_raw_columns` chỉ gồm tên raw operands thật sự từ danh mục, không gồm derived columns.
- Trước khi tính median, min, max, mean hoặc sum phải lọc đúng tập ticker và đúng giai đoạn được hỏi.
- `idxmax()`/`idxmin()` trả về index của DataFrame, KHÔNG phải đáp án. Khi câu hỏi chọn theo chỉ tiêu A
  rồi hỏi chỉ tiêu B, hãy sort theo A và lấy B ở cùng dòng, ví dụ:
  `float(df[(df.ticker == 'KBC') & df.year.between(2016, 2020)].sort_values('liabilities_to_equity', ascending=False).iloc[0]['interest_coverage'])`.
- Cụm "năm sau", "năm ngay sau", "năm kế tiếp" nghĩa là lấy dòng có year bằng năm được chọn cộng 1.
- "nợ ngắn hạn" là `current_liabilities`; "nợ phải trả" là `liabilities`; không thay thế hai khái niệm này.
- "EBIT proxy" hoặc "lợi nhuận trước lãi vay và thuế" được tính bằng `pbt + interest_expense`.
  Hệ số khả năng thanh toán lãi vay chính là cột full-precision `interest_coverage`, không dùng `operating_profit` thay EBIT.
  Nếu dùng `interest_coverage`, lấy trực tiếp scalar của cột này; TUYỆT ĐỐI không chia nó cho `interest_expense` lần nữa.
- Tỷ trọng tổng của nhóm lọc luôn dùng đúng đại lượng được hỏi ở tử và mẫu. Ví dụ tỷ trọng nợ ngắn hạn:
  `df[SCOPE & (df.inventory_to_current_liabilities > df[SCOPE].inventory_to_current_liabilities.median())].current_liabilities.sum() / df[SCOPE].current_liabilities.sum() * 100`,
  trong đó phải lặp lại nguyên điều kiện SCOPE thật (không được lập biến tạm).
- "CFO trên doanh thu", "CFO margin", "dòng tiền hoạt động trên doanh thu" là `cfo_margin`;
  `operating_cash_flow_ratio` chỉ là CFO chia nợ ngắn hạn.
- Luôn trả một scalar đích cuối cùng, không trả index, vector, mảng hay nhiều cột.
- Với điều kiện nhiều ticker, dùng `.isin([...])`; đặt ngoặc đầy đủ quanh mọi điều kiện boolean.

Danh mục cột:
""" + PANEL_COLUMN_GUIDE


@dataclass(frozen=True, slots=True)
class PanelSolution:
    answer: float | int
    pandas_query: str
    required_raw_columns: tuple[str, ...]
    tickers: tuple[str, ...]
    years: tuple[int, ...]
    attempts: int
    model_note: str
    elapsed_seconds: float


def _validate_ast(expression: str) -> None:
    tree = ast.parse(expression, mode="eval")
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.NamedExpr,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"df", "pd", "np", "int", "float", "abs", "min", "max"}:
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute is forbidden")


def execute_panel_query(expression: str, frame: pd.DataFrame) -> float | int:
    _validate_ast(expression)
    result = eval(  # noqa: S307 - AST and namespace are deliberately constrained above.
        compile(ast.parse(expression, mode="eval"), "<panel_query>", "eval"),
        {"__builtins__": {}, "pd": pd, "np": np, "int": int, "float": float, "abs": abs, "min": min, "max": max},
        {"df": frame.copy()},
    )
    if isinstance(result, pd.Series):
        if len(result) != 1:
            raise ValueError(f"Expression returned Series of length {len(result)}")
        result = result.iloc[0]
    if isinstance(result, np.ndarray):
        if result.size != 1:
            raise ValueError(f"Expression returned ndarray of size {result.size}")
        result = result.item()
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"Expression returned non-numeric scalar: {result!r}")
    if not math.isfinite(float(result)):
        raise ValueError(f"Expression returned non-finite scalar: {result!r}")
    return result


def _validate_financial_semantics(question: str, expression: str) -> None:
    """Reject high-impact accounting substitutions before accepting code."""

    folded = fold_text(question)
    compact = expression.replace(" ", "")
    if any(
        marker in folded
        for marker in (
            "cfo tren doanh thu",
            "cfo margin",
            "dong tien hoat dong tren doanh thu",
            "dong tien tu hoat dong kinh doanh tren doanh thu",
            "luu chuyen tien thuan tu hoat dong kinh doanh tren doanh thu",
        )
    ):
        if "operating_cash_flow_ratio" in expression:
            raise ValueError("Semantic error: CFO/revenue must use cfo_margin, not operating_cash_flow_ratio")
        if "cfo_margin" not in expression and not ("cfo" in expression and "net_revenue" in expression):
            raise ValueError("Semantic error: missing CFO/revenue metric")
    asks_interest_coverage = "he so kha nang thanh toan lai vay" in folded
    asks_ebit_over_interest = (
        "loi nhuan truoc lai vay va thue" in folded and "chi phi lai vay" in folded
    )
    if asks_interest_coverage or asks_ebit_over_interest:
        valid_derived = "interest_coverage" in expression and "interest_expense" not in expression
        valid_formula = "pbt" in expression and "interest_expense" in expression and "+" in compact
        if not (valid_derived or valid_formula):
            raise ValueError(
                "Semantic error: return interest_coverage directly (do not divide it again), or compute "
                "(pbt + interest_expense) / interest_expense"
            )


def _repair_prompt(question: str, rows: list[tuple[str, int]], prior: str, error: str) -> str:
    return f"""/no_think
Câu hỏi: {question}
Các dòng ticker-year có sẵn: {rows}
Biểu thức lần trước: {prior}
Lỗi thực thi: {error}
Hãy sửa và trả JSON đúng contract. Không giải thích ngoài JSON."""


def solve_panel_question(
    question: str,
    panel: FinancialPanel,
    *,
    max_attempts: int = 3,
    log_path: Path | None = None,
) -> PanelSolution:
    tickers = infer_panel_tickers(question, panel.tickers)
    years = infer_panel_years(question)
    if not tickers:
        raise ValueError("No panel ticker resolved")
    if not years:
        raise ValueError("No year resolved")
    # Derived rolling metrics are already computed on the complete canonical
    # panel.  We expose adjacent years as read-only context for questions that
    # explicitly ask for the year before/after a selected year.
    context_years = set(years)
    context_years.update(year - 1 for year in years)
    context_years.update(year + 1 for year in years)
    frame = panel.subset(tickers, context_years, include_prior=False)
    if frame.empty:
        raise ValueError("No panel rows for resolved entities/years")
    rows = [(str(ticker), int(year)) for ticker, year in frame[["ticker", "year"]].itertuples(index=False)]
    prompt = f"""/no_think
Câu hỏi: {question}
Các dòng ticker-year có sẵn trong df: {rows}
Hãy biên dịch chính xác theo contract. Chỉ trả JSON."""
    started = time.time()
    logs: list[dict[str, object]] = []
    prior = ""
    error = ""

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            prompt = _repair_prompt(question, rows, prior, error)
        completion = chat(system=SYSTEM_PROMPT, user=prompt, max_tokens=768, temperature=0.0)
        entry: dict[str, object] = {"attempt": attempt, "response": completion.content, **asdict(completion)}
        try:
            payload = extract_json(completion.content)
            expression = str(payload["pandas_query"]).strip()
            prior = expression
            required = tuple(str(value) for value in payload.get("required_raw_columns", []))
            # The model sometimes lists a derived column here even though the
            # expression itself is sound.  Provenance is reconstructed from the
            # executed expression later, so this advisory field must not reject
            # an otherwise valid program.
            required = tuple(column for column in required if column in RAW_COLUMNS)
            _validate_financial_semantics(question, expression)
            answer = execute_panel_query(expression, frame)
            entry["answer"] = answer
            logs.append(entry)
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return PanelSolution(
                answer=answer,
                pandas_query=expression,
                required_raw_columns=required,
                tickers=tuple(tickers),
                years=tuple(years),
                attempts=attempt,
                model_note=str(payload.get("note", "")),
                elapsed_seconds=time.time() - started,
            )
        except Exception as exc:  # the next attempt receives exact validator feedback
            error = f"{type(exc).__name__}: {exc}"
            entry["error"] = error
            logs.append(entry)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(f"Panel compilation failed after {max_attempts} attempts: {error}")
