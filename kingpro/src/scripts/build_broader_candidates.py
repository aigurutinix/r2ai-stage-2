"""Mẻ BROADER (GPT): 25-35 câu 'họ hàng' 3 câu đã thắng. Lọc CỨNG:
single-entity + <=2 operand + op LOOKUP/RATIO/GROWTH/DIFF + KHÔNG đa-tầng + base chưa fix
+ retrieval mơ hồ (top1/top2 bảng <1.5) + base-sai (% & |base|>1e4).
Chạy: PYTHONUTF8=1 python scripts/build_broader_candidates.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.answering.operand_pipeline import classify_operation
from kingpro.evaluation.metrics import doc_of, coerce_number

QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
wb = json.load(open("build/winning_buckets.json"))
fill_ids = set(i for i, e in base.items() if not (e.get("pandas_query") or "").strip())
fixed = set(wb["unit"]) | set(wb.get("cross", [])) | {644, 680, 710}
done = {c["id"] for c in json.load(open("build/ctx_candidates.json", encoding="utf-8"))} | \
       {c["id"] for c in json.load(open("build/absurd_candidates.json", encoding="utf-8"))}
MULTI = re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất|trung vị|trung bình|"
                   r"tổng.{0,6}(của|các)|trong nhóm|trong số|giữa các|bao nhiêu (công ty|doanh nghiệp)|"
                   r"các (công ty|doanh nghiệp|năm|ngân hàng)|mã cổ phiếu", re.I)
PCT = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ|tỷ trọng|tỉ trọng|tốc độ tăng|tăng trưởng|gấp.{0,8}lần|biên|hệ số|tỷ số", re.I)

retrieve_decomposed("warm")
cand = []
for qid, q in QMAP.items():
    if qid in fixed or qid in fill_ids or qid in done:
        continue
    e = base.get(qid, {})
    if not (e.get("pandas_query") or "").strip():
        continue
    if MULTI.search(q):
        continue
    try:
        if len(extract_all_facets(q).get("tickers", [])) != 1:
            continue
    except Exception:
        continue
    op, req = classify_operation(q)
    if req > 2:
        continue
    ba = coerce_number(e.get("answer"))
    # base-sai: % question & |base|>1e4
    base_wrong = ba is not None and PCT.search(q) and abs(ba) > 1e4
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        at = tables_in_reports(q, rel, n=6)
    except Exception:
        continue
    if len(at) < 2:
        continue
    ratio = at[0]["score"] / max(at[1]["score"], 0.01)
    ambiguous = ratio < 1.5
    if not (base_wrong or ambiguous):
        continue
    cand.append({"id": qid, "base_ans": ba, "op": op, "ratio": round(ratio, 3),
                 "base_wrong": bool(base_wrong), "q": q[:55], "tables": [h["table_ref"] for h in at]})

# ưu tiên base-wrong trước, rồi retrieval mơ hồ nhất
cand.sort(key=lambda x: (not x["base_wrong"], x["ratio"]))
top = cand[:35]
json.dump(top, open("build/broader_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"BROADER candidates = {len(cand)}, lấy top {len(top)} (base_wrong={sum(1 for c in top if c['base_wrong'])})")
for c in top[:20]:
    print(f"  Q{c['id']} op={c['op']} ratio={c['ratio']} bw={c['base_wrong']} base={c['base_ans']} | {c['q']}")
