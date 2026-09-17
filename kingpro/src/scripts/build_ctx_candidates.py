"""GPU #3 Bước 1 (GPT): chọn ~40 câu retrieval MƠ HỒ để rerun context-conditioned.
Tiêu chí: base KHÔNG rỗng + KHÔNG thuộc fix cũ (fill/unit/cross); đơn-ticker; KHÔNG superlative/agg;
base đọc 1 bảng; BM25 table top1/top2 < 1.3; >=2 bảng plausible cùng entity+năm.
Rank theo ambiguity. Chạy: PYTHONUTF8=1 python scripts/build_ctx_candidates.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.evaluation.metrics import doc_of

QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
wb = json.load(open("build/winning_buckets.json"))
fixed = set(wb["unit"]) | set(wb.get("cross", []))
# fill ids = câu base RỖNG (đã lấp)
fill_ids = set(i for i, e in base.items() if not (e.get("pandas_query") or "").strip())
BAD = re.compile(r"trong nhóm|trong số|cao nhất|thấp nhất|trung bình|trung vị|bao nhiêu công ty|các công ty|các doanh nghiệp", re.I)

retrieve_decomposed("warm")
cand = []
for qid, q in QMAP.items():
    e = base.get(qid, {})
    if not (e.get("pandas_query") or "").strip():
        continue                     # rỗng -> thuộc fill, bỏ
    if qid in fixed or qid in fill_ids or BAD.search(q):
        continue
    try:
        if len(extract_all_facets(q).get("tickers", [])) != 1:
            continue
    except Exception:
        continue
    # base đọc mấy bảng?
    nev = len(e.get("evidence", []))
    used1 = ("_dfvals[1]" not in (e.get("pandas_query") or "")) and ("df2" not in (e.get("pandas_query") or ""))
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        at = tables_in_reports(q, rel, n=6)
    except Exception:
        continue
    if len(at) < 2:
        continue
    t1, t2 = at[0]["score"], at[1]["score"]
    ratio = t1 / max(t2, 0.01)
    plausible = sum(1 for h in at if h["score"] >= 0.75 * t1)
    amb = 0
    if ratio < 1.15:
        amb += 3
    elif ratio < 1.30:
        amb += 2
    if plausible >= 3:
        amb += 1
    if used1:
        amb += 1
    if amb < 2:
        continue
    cand.append({"id": qid, "amb": amb, "ratio": round(ratio, 3), "plausible": plausible,
                 "base_ans": e.get("answer"), "q": q[:60],
                 "tables": [h["table_ref"] for h in at]})

cand.sort(key=lambda x: (-x["amb"], x["ratio"]))
top = cand[:40]
json.dump(top, open("build/ctx_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"AMBIGUOUS-retrieval candidates = {len(cand)}, lấy top {len(top)}")
for c in top[:15]:
    print(f"  Q{c['id']} amb={c['amb']} ratio={c['ratio']} plaus={c['plausible']} base={c['base_ans']} | {c['q']}")
