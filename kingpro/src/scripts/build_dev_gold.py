"""Dựng dev-gold BẠC (không cần gold thật) để đo offline candidate TRƯỚC khi nộp.

Nhãn bạc = câu nào có >=2 trong 3 phương pháp ĐỘC LẬP ra CÙNG số (rel tol 0.5%):
  (1) program (sub_program), (2) base (sub_base_fix), (3) maso tất định (pandas thuần).
2 phương pháp độc lập trùng số lớn tài chính -> gần như chắc đúng. Ghi build/dev_gold.jsonl.

Chạy (py3.14, có bm25s): PYTHONUTF8=1 python scripts/build_dev_gold.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.pandas_answer import maso_answer, ratio_answer
from kingpro.evaluation.metrics import doc_of, coerce_number

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def csv_full(tref):
    r = CAT.get(tref)
    return "build/tables/" + r["csv_path"] if r else None


def maso_for(q):
    try:
        hits = retrieve_decomposed(q)
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel_docs, n=12)
        tables = [{"csv_path": csv_full(h["table_ref"]), "table_ref": h["table_ref"]}
                  for h in atabs if csv_full(h["table_ref"])]
        res = ratio_answer(q, tables) or maso_answer(q, tables)
        if res and res.get("ok") and res.get("answer") is not None:
            return coerce_number(res["answer"])
    except Exception:
        return None
    return None


def real_ans(row):
    if not row:
        return None
    if not (row.get("pandas_query") or "").strip():
        return None
    return coerce_number(row.get("answer"))


def agree(a, b, tol=0.005):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


def main():
    prog = {r["id"]: r for r in json.load(open("sub_program/submission.json", encoding="utf-8"))}
    base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
    qs = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")]
    retrieve_decomposed("khoi dong")
    gold, n = [], 0
    for i, q in enumerate(qs):
        qid = q["id"]
        cands = []
        pa = real_ans(prog.get(qid))
        if pa is not None:
            cands.append(("prog", pa))
        ba = real_ans(base.get(qid))
        if ba is not None:
            cands.append(("base", ba))
        ma = maso_for(q["question"])
        if ma is not None:
            cands.append(("maso", ma))
        # tìm cặp/nhóm >=2 đồng thuận
        best = None
        for j in range(len(cands)):
            grp = [cands[j]]
            for k in range(len(cands)):
                if k != j and agree(cands[j][1], cands[k][1]):
                    grp.append(cands[k])
            if len(grp) >= 2 and (best is None or len(grp) > len(best)):
                best = grp
        if best and abs(best[0][1]) > 1e-9:            # bỏ 0.0 suy biến
            gold.append({"id": qid, "question": q["question"],
                         "gold": float(best[0][1]),
                         "sources": [s for s, _ in best], "n_agree": len(best)})
            n += 1
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(qs)} -> dev-gold {n}", flush=True)
    with open("build/dev_gold.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    n3 = sum(1 for g in gold if g["n_agree"] >= 3)
    print(f"DONE dev-gold: {n} câu ({n3} câu 3/3 đồng thuận, {n-n3} câu 2/3) -> build/dev_gold.jsonl")


if __name__ == "__main__":
    main()
