"""Đo TABLE-RECALL của retrieval hiện tại bằng clean gold (có table_ref THẬT). MIỄN PHÍ.
Xác nhận lỗ hổng evidence trước khi tiêu GPU build dense+reranker.
Chạy: PYTHONUTF8=1 python scripts/test_table_recall.py
"""
import json
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.evaluation.metrics import doc_of

gold = [json.loads(l) for l in open("build/clean_gold.jsonl", encoding="utf-8")]
gold = [g for g in gold if g.get("table_ref")]
retrieve_decomposed("warm")

ks = [4, 6, 10, 20, 50]
doc_hit = 0
tab_hit = {k: 0 for k in ks}
n = 0
misses = []
for g in gold:
    n += 1
    gref = g["table_ref"]
    gdoc = doc_of(gref)
    hits = retrieve_decomposed(g["question"])
    rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    if gdoc in rel_docs:
        doc_hit += 1
    atabs = tables_in_reports(g["question"], rel_docs, n=max(ks))
    refs = [h["table_ref"] for h in atabs]
    for k in ks:
        if gref in refs[:k]:
            tab_hit[k] += 1
    if gref not in refs[:10]:
        misses.append((g["id"], gdoc in rel_docs, gref.split("|")[0], g["question"][:45]))

print(f"CLEAN GOLD N={n}")
print(f"DOC recall (đúng báo cáo có trong rel_docs) = {doc_hit}/{n} = {doc_hit/n:.3f}")
for k in ks:
    print(f"TABLE recall@{k} (đúng bảng vào top-{k}) = {tab_hit[k]}/{n} = {tab_hit[k]/n:.3f}")
print("--- miss@10 (bảng đúng KHÔNG vào top-10) ---")
for qid, docok, tk, q in misses[:10]:
    print(f"Q{qid} doc_ok={docok} tk={tk} | {q}")
