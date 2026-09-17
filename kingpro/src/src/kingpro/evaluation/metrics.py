"""Chấm điểm — BÊ NGUYÊN công thức baseline BTC để khớp 100%.

Nguồn: vendor/vifinqa-baseline evaluation/retrieval_metrics.py + answer_match.py + constants.py.
Khác baseline đúng 1 chỗ: table_ref của ta là "report_id|dòng" (không phải "doc|table_N"),
nên doc = phần trước dấu '|' cuối cùng.
"""

from __future__ import annotations

import math
import re

ANSWER_ABS_TOL = 1e-2  # constants.py: sai số answer TUYỆT ĐỐI 0.01, rel_tol=0

_NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$|^\(?-?\d+(\.\d+)?\)?%?$")


def doc_of(table_ref: str) -> str:
    return table_ref.rpartition("|")[0] or table_ref


def is_numeric_cell(text: str) -> bool:
    return bool(_NUMERIC_RE.match(text.strip()))


def coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if is_numeric_cell(text):
        cleaned = text
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        if negative:
            cleaned = cleaned[1:-1]
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]  # bỏ %, KHÔNG chia 100
        cleaned = cleaned.replace(".", "").replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            return None
        return -num if negative else num
    return None


def is_correct(expected, actual, *, abs_tol: float = ANSWER_ABS_TOL) -> bool:
    exp, act = coerce_number(expected), coerce_number(actual)
    if exp is None or act is None:
        return False
    return math.isclose(exp, act, rel_tol=0.0, abs_tol=abs_tol)


# ---- Truy hồi (giao tập CHUỖI chính xác, y baseline) ----

def _fbeta(tp: int, retrieved: int, gold: int, beta: float = 2.0) -> float:
    p = tp / retrieved if retrieved else 0.0
    r = tp / gold if gold else 0.0
    if p == 0.0 and r == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def table_metrics(retrieved_refs: list[str], gold_refs, k: int) -> dict:
    gold = set(gold_refs)
    top = retrieved_refs[:k]
    tp = len(set(top) & gold)
    return {
        "tp": tp,
        "precision": tp / len(top) if top else 0.0,
        "recall": tp / len(gold) if gold else 0.0,
        "f2": _fbeta(tp, len(top), len(gold)),
    }


def doc_metrics(retrieved_refs: list[str], gold_refs, k: int) -> dict:
    gold_docs = {doc_of(r) for r in gold_refs}
    ret_docs = list(dict.fromkeys(doc_of(r) for r in retrieved_refs[:k]))  # khử trùng, giữ thứ tự
    tp = len(set(ret_docs) & gold_docs)
    return {
        "tp": tp,
        "precision": tp / len(ret_docs) if ret_docs else 0.0,
        "recall": tp / len(gold_docs) if gold_docs else 0.0,
        "f2": _fbeta(tp, len(ret_docs), len(gold_docs)),
    }


def macro_f2(per_query: list[dict], key: str = "f2") -> float:
    """macro-average: trung bình f2 theo từng câu."""
    return sum(m[key] for m in per_query) / len(per_query) if per_query else 0.0


if __name__ == "__main__":
    # kiểm nhanh cho khớp kỳ vọng
    assert is_correct(1234.56, "1234.56")
    assert is_correct(1234.56, 1234.565)          # trong 0.01
    assert not is_correct(1234.56, 1234.60)       # ngoài 0.01
    assert coerce_number("(1.234,5)") == -1234.5  # ngoặc âm, dấu VN
    assert coerce_number("50%") == 50.0           # % KHÔNG chia 100
    m = table_metrics(["A|10", "A|20", "B|5"], {"A|10"}, k=3)
    assert abs(m["f2"] - 1.0) < 1e-9 or m["recall"] == 1.0
    print("metrics OK | table_metrics(1 gold, 3 retrieved):", table_metrics(["A|10", "A|20", "B|5"], {"A|10"}, k=3))
    print("nhồi 5 bảng (1 gold):", table_metrics([f"A|{i}" for i in range(5)], {"A|0"}, k=5))
