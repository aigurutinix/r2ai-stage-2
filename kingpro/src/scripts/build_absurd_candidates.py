"""MỞ RỘNG đòn thắng: quét câu base VÔ LÝ (câu %/tỷ-lệ/tỷ-trọng/tăng-trưởng/gấp-lần mà |base|>1e6
= chắc chắn 0 điểm) -> ứng viên rerun Context B (risk-free override). Bỏ câu đã thuộc fix cũ.
Chạy: PYTHONUTF8=1 python scripts/build_absurd_candidates.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.evaluation.metrics import doc_of, coerce_number

QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
wb = json.load(open("build/winning_buckets.json"))
fixed = set(wb["unit"]) | set(wb.get("cross", [])) | {644, 680, 710}
done_ctx = {c["id"] for c in json.load(open("build/ctx_candidates.json", encoding="utf-8"))}
# câu PHẦN TRĂM / tỷ lệ / tỷ số / gấp lần -> đáp án hợp lý |x| < ~1e5
PCT = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ|tỷ trọng|tỉ trọng|tốc độ tăng|tăng trưởng|gấp.{0,8}lần|biên (lợi nhuận|lãi)|hệ số|tỷ số", re.I)

retrieve_decomposed("warm")
cand = []
for qid, q in QMAP.items():
    if qid in fixed or qid in done_ctx:
        continue
    e = base.get(qid, {})
    if not (e.get("pandas_query") or "").strip():
        continue
    ba = coerce_number(e.get("answer"))
    if ba is None:
        continue
    if not PCT.search(q):
        continue
    if abs(ba) <= 1e6:          # trong ngưỡng hợp lý -> bỏ
        continue
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        at = tables_in_reports(q, rel, n=6)
    except Exception:
        continue
    cand.append({"id": qid, "base_ans": ba, "q": q[:60], "tables": [h["table_ref"] for h in at]})

json.dump(cand, open("build/absurd_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"BASE-ABSURD (câu % mà |base|>1e6) = {len(cand)}")
for c in cand[:30]:
    print(f"  Q{c['id']} base={c['base_ans']:.3g} | {c['q']}")
