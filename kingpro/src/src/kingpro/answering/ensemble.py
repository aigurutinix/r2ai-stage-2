"""Ensemble 2 model: chạy Coder-14B + Qwen3-14B, đồng thuận = tin cao.

- Cả 2 ra số VÀ khớp (sai lệch tương đối < rel_tol) -> answer đó, confidence=high (rất khả năng đúng).
- Cả 2 ra số nhưng LỆCH -> lấy model mạnh hơn (Qwen3) làm mặc định, confidence=low.
- Chỉ 1 ra số -> lấy nó, confidence=med.
- Cả 2 fail -> không có đáp án (từ chối), confidence=none.

Đồng thuận dùng làm CHỈ SỐ TIN CẬY: chỗ high để yên tâm, chỗ low/none là nơi cần cải thiện.
"""

from __future__ import annotations

import json
import re

from kingpro.answering.pandas_answer import answer_question
from kingpro.evaluation.metrics import coerce_number


def _num(res: dict):
    return coerce_number(res.get("answer")) if res.get("ok") else None


def combine(res_a: dict, res_b: dict, rel_tol: float = 0.01) -> dict:
    """res_a = Coder, res_b = Qwen3 (model mạnh hơn -> ưu tiên khi lệch)."""
    a, b = _num(res_a), _num(res_b)
    if a is not None and b is not None:
        if abs(a - b) / max(abs(a), abs(b), 1.0) < rel_tol:
            return {"answer": a, "confidence": "high", "source": "agree", "pandas_query": res_a["pandas_query"]}
        return {"answer": b, "confidence": "low", "source": "disagree->qwen3", "pandas_query": res_b["pandas_query"]}
    if b is not None:
        return {"answer": b, "confidence": "med", "source": "qwen3-only", "pandas_query": res_b["pandas_query"]}
    if a is not None:
        return {"answer": a, "confidence": "med", "source": "coder-only", "pandas_query": res_a["pandas_query"]}
    return {"answer": None, "confidence": "none", "source": "both-fail", "pandas_query": ""}


def self_consistent(question, tables, base, key, model, n=10, temp=0.6, **kw) -> dict:
    """MỘT model, sinh n bản (temp>0), VOTE theo giá trị kết quả -> cụm đông nhất. 0 fill.

    Đây là công thức thắng VLSP 2025 (Innovation-LLM/MoFin: 1 model + majority voting, n=10 tối ưu EA,
    +8-11% EA). consensus = tỉ lệ đồng thuận. KHÔNG ensemble 2 model (thừa + tốn gấp đôi).
    """
    from kingpro.answering.llm_client import chat

    vals, results = [], []
    llm = lambda s, u: chat(s, u, base_url=base, api_key=key, model=model, temperature=temp, max_tokens=1200, timeout=300)
    for _ in range(n):
        try:
            r = answer_question(question, tables, llm, **kw)
        except Exception:
            continue  # timeout/mạng: bỏ mẫu này, không sập cả câu
        v = _num(r)
        if v is not None:
            vals.append(v)
            results.append(r)
    if not vals:
        return {"answer": None, "consensus": 0.0, "n_ok": 0, "pandas_query": "", "evidence": []}
    # Zero is often a failed lookup, but it is also a valid financial answer.
    # Drop it only when at least one non-zero executable sample exists; retain
    # an all-zero quorum instead of turning a valid result into a refusal.
    non_zero = [(value, result) for value, result in zip(vals, results) if abs(value) > 1e-9]
    if non_zero:
        vals = [value for value, _ in non_zero]
        results = [result for _, result in non_zero]
    # gom theo cụm (sai số tuyệt đối 0.01 HOẶC tương đối 1%), lấy cụm đông nhất
    def close(a, b):
        return abs(a - b) <= 0.01 or abs(a - b) / max(abs(a), abs(b), 1.0) < 0.01
    # #13: chọn cụm ĐÔNG nhất; HÒA PHIẾU -> chọn code NGẮN hơn (đơn giản = đúng hơn, OpenSearch-SQL).
    best_i, best_key = 0, (-1, 0)
    for i, a in enumerate(vals):
        cnt = sum(1 for b in vals if close(a, b))
        key_i = (cnt, -len(results[i].get("pandas_query", "")))
        if key_i > best_key:
            best_key, best_i = key_i, i
    best_cnt = best_key[0]
    win = results[best_i]
    return {"answer": vals[best_i], "consensus": best_cnt / len(vals), "n_ok": len(vals),
            "pandas_query": win.get("pandas_query", ""), "evidence": win.get("evidence", [])}


DIVERSE_HINTS = [
    "",
    "\nCHIẾN LƯỢC A: lọc đúng DÒNG bằng nhãn chỉ tiêu (df1[df1['0'].str.contains('<từ khoá>', na=False)]), rồi lấy đúng CỘT kỳ/năm bằng tên cột chuỗi.",
    "\nCHIẾN LƯỢC B: xác định vị trí (số hàng, số cột) của ô cần lấy rồi dùng df1.iloc[hàng, cột]. Làm sạch số kiểu Việt trước khi tính.",
]


def diverse_vote(question, tables, llms, hints=DIVERSE_HINTS, max_fix=2, timeout=5.0) -> dict:
    """CHASE-SQL-lite: sinh nhiều ứng viên ĐA DẠNG (model × chiến lược) -> chạy -> BỎ PHIẾU theo kết quả.

    llms = [(tên, llm_fn), ...]. Đa dạng ứng viên -> lỗi đa dạng -> đa số lọc ra đáp án đúng.
    """
    cands = []
    for name, llm in llms:
        for h in hints:
            try:
                r = answer_question(question, tables, llm, max_fix=max_fix, timeout=timeout, hint=h)
            except Exception:
                continue
            v = _num(r)
            if v is not None:
                cands.append({"val": v, "code": r["pandas_query"], "src": name})
    if not cands:
        return {"answer": None, "votes": 0, "total": 0, "pandas_query": "", "confidence": "none"}
    # BẪY 0.0: ứng viên trả đúng 0.0 gần như luôn là tra hụt/ô rỗng, không phải số tài chính thật.
    # Loại chúng khỏi vote (trừ khi TẤT CẢ đều 0.0 -> có thể đúng là 0).
    nz = [c for c in cands if abs(c["val"]) > 1e-9]
    cands = nz if nz else cands
    best_i, best_cnt = 0, 0
    for i, c in enumerate(cands):
        cnt = sum(1 for d in cands if abs(c["val"] - d["val"]) / max(abs(c["val"]), abs(d["val"]), 1.0) < 0.01)
        if cnt > best_cnt:
            best_cnt, best_i = cnt, i
    frac = best_cnt / len(cands)
    conf = "high" if frac >= 0.6 else "med" if frac >= 0.4 else "low"
    return {"answer": cands[best_i]["val"], "votes": best_cnt, "total": len(cands),
            "pandas_query": cands[best_i]["code"], "confidence": conf}


def smart_answer(question, tables, models, key, escalate=True, **kw) -> dict:
    """PHÂN TẦNG: Tier1 ensemble rẻ (2 model temp0). Đồng thuận + khác 0 -> chốt (tier1).
    Nghi ngờ (lệch / chỉ 1 / degenerate) -> Tier2 diverse-vote (đắt), chỉ khi cần.

    models = [(tên, base_url, model_id), ...]. Trả answer + confidence + tier + pandas_query + evidence.
    """
    from kingpro.answering.llm_client import chat

    results = []
    for name, base, model in models:
        llm = lambda s, u, b=base, m=model: chat(s, u, base_url=b, api_key=key, model=m, temperature=0, max_tokens=1200, timeout=300)
        try:
            r = answer_question(question, tables, llm, **kw)
        except Exception:
            r = {"ok": False, "answer": None, "pandas_query": "", "evidence": []}
        results.append(r)

    valid = [(coerce_number(r["answer"]), r) for r in results if r.get("ok") and coerce_number(r.get("answer")) is not None]
    valid = [(v, r) for v, r in valid if abs(v) > 1e-9]        # bỏ degenerate 0.0
    ev = next((r.get("evidence", []) for r in results if r.get("evidence")), [])

    if len(valid) >= 2:                                        # Tier 1: đồng thuận -> chốt
        a = valid[0][0]
        if all(abs(a - v) / max(abs(a), abs(v), 1.0) < 0.01 for v, _ in valid):
            return {"answer": a, "confidence": "high", "tier": 1, "pandas_query": valid[0][1]["pandas_query"], "evidence": ev}

    if escalate:                                               # Tier 2: nghi ngờ -> diverse-vote
        llms = [(name, (lambda s, u, b=base, m=model: chat(s, u, base_url=b, api_key=key, model=m, temperature=0.3, max_tokens=1200, timeout=300)))
                for name, base, model in models]
        dv = diverse_vote(question, tables, llms, **kw)
        dv["tier"] = 2
        dv["evidence"] = ev
        return dv

    if valid:
        return {"answer": valid[0][0], "confidence": "med", "tier": 1, "pandas_query": valid[0][1]["pandas_query"], "evidence": ev}
    return {"answer": None, "confidence": "none", "tier": 1, "pandas_query": "", "evidence": ev}


def ensemble_answer(question, tables, llm_coder, llm_qwen3, **kw) -> dict:
    ra = answer_question(question, tables, llm_coder, **kw)
    rb = answer_question(question, tables, llm_qwen3, **kw)
    out = combine(ra, rb)
    out["evidence"] = (rb if out["source"].startswith(("disagree", "qwen3")) else ra).get("evidence", [])
    return out


# ---- MPR-Agent (Innovation-LLM, VLSP 2025): phân rã -> trích số -> synth + vote ----
_DECOMPOSE_SYS = (
    "Bạn là chuyên gia chia nhỏ câu hỏi tài chính thành 3-5 CÂU HỎI CON chỉ để TRÍCH SỐ THÔ.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- KHÔNG tạo câu SO SÁNH (cái nào cao/thấp nhất, hơn kém).\n"
    "- KHÔNG tạo câu TÍNH TOÁN (tổng, trung bình, tỷ lệ, tăng trưởng là bao nhiêu).\n"
    "- KHÔNG hỏi đáp án cuối.\n"
    "- Mỗi câu con hỏi ĐÚNG MỘT điểm dữ liệu: một chỉ tiêu, một công ty, một năm.\n"
    'Chỉ trả JSON: {"subqueries": ["...", "..."]}'
)


def _plan(question: str, llm) -> list[str]:
    """1 lời gọi -> danh sách câu hỏi con 'chỉ trích số' (Innovation-LLM Fig B.1). Lỗi -> []."""
    try:
        txt = llm(_DECOMPOSE_SYS, f"Câu hỏi: {question}\n\nChỉ trả JSON.")
    except Exception:
        return []
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return []
    try:
        subs = json.loads(m.group(0)).get("subqueries", [])
    except Exception:
        return []
    return [s for s in subs if isinstance(s, str) and len(s) > 8][:6]


def aggregate_facts(question: str, facts: list) -> float | None:
    """#3 argmax TẤT ĐỊNH: từ các số đã trích (facts=[(nhãn,số)]), tự tính max/min/tổng/trung bình
    thay vì tin model so sánh. Câu 'cao nhất/thấp nhất/tổng/trung bình' rất dễ verify bằng code."""
    ql = question.lower()
    vals = [v for _, v in facts if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    # A nested selector has two semantic roles: one metric chooses the entity
    # or year, another metric is returned. Blind max/min over every extracted
    # fact mixes those roles and can be confidently wrong. Let the program
    # synthesizer handle these until facts carry explicit role tags.
    if re.search(
        r"(?:tại|trong) năm có|của công ty có|năm nào|ứng với|"
        r"tại thời điểm có|trong số.+có",
        ql,
    ):
        return None
    if re.search(r"cao nhất|lớn nhất|nhiều nhất|đỉnh", ql):
        return round(max(vals), 4)
    if re.search(r"thấp nhất|nhỏ nhất|ít nhất", ql):
        return round(min(vals), 4)
    if re.search(r"tổng cộng|tổng của|tổng ", ql):
        return round(sum(vals), 4)
    if re.search(r"trung bình|bình quân", ql):
        return round(sum(vals) / len(vals), 4)
    if re.search(r"chênh lệch|hiệu số", ql) and len(vals) == 2:
        delta = vals[0] - vals[1]
        if re.search(r"độ chênh lệch|khoảng chênh lệch|chênh lệch tuyệt đối", ql):
            delta = abs(delta)
        return round(delta, 4)
    return None


def mpr_answer(question, tables, base, key, model, n_vote=5, max_fix=2, analytic=False, **kw) -> dict:
    """Câu PHÂN TÍCH: phân rã -> trích từng số (grounding) -> đưa các số đã trích làm gợi ý
    cho bước sinh code cuối + self-consistency vote. Câu ĐƠN: chạy thẳng self_consistent.

    Công thức thắng VLSP 2025 (Innovation-LLM MPR-Agent, EA 79% không train). Fallback an toàn.
    """
    from kingpro.answering.llm_client import chat

    if not analytic:                                       # câu đơn: khỏi phân rã, trực tiếp
        return self_consistent(question, tables, base, key, model, n=n_vote, temp=0.6, max_fix=max_fix)

    plan_llm = lambda s, u: chat(s, u, base_url=base, api_key=key, model=model, temperature=0.3, max_tokens=700, timeout=300)
    subs = _plan(question, plan_llm)
    facts = []
    for sq in subs:                                        # trích từng số (self-consistency nhỏ n=3)
        r = self_consistent(sq, tables, base, key, model, n=3, temp=0.6, max_fix=1)
        if r.get("answer") is not None:
            facts.append((sq, r["answer"]))
    agg = aggregate_facts(question, facts)                 # #3: gộp tất định (max/min/tổng/tb) nếu được
    if agg is not None:
        return {"answer": agg, "confidence": "high", "pandas_query": "# argmax tất định từ facts",
                "evidence": [], "facts": facts, "consensus": 1.0}
    hint = ""
    if facts:                                              # đưa số đã trích làm GROUNDING cho bước cuối
        hint = ("\n\nCÁC SỐ ĐÃ TRÍCH (dùng để tính; nếu lệch với bảng thì tin bảng):\n"
                + "\n".join(f"- {sq} = {v}" for sq, v in facts))
    out = self_consistent(question, tables, base, key, model, n=n_vote, temp=0.6, max_fix=max_fix, hint=hint)
    out["facts"] = facts
    return out
