"""Đo PRECISION của select_maso_table offline trên val (biết bảng gold).
Given gold report, selector có chọn đúng bảng gold không?"""
import sys, json, re, io
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from build_full_submission import select_maso_table, csv_full
from kingpro.answering.ma_so_tt200 import maso_of

rows = [json.loads(l) for l in open("build/sft_chatml_val.jsonl", encoding="utf-8")]
out = io.StringIO()
hit = miss = nomaso = noreport = 0
by_type = {}
for r in rows:
    trefs = r.get("table_refs") or []
    if not trefs:
        continue
    gold = trefs[0]
    report = gold.split("|")[0]
    # trích câu hỏi từ user message
    user = r["conversations"][1]["content"]
    m = re.search(r"Câu hỏi:\s*(.+)", user)
    q = m.group(1).strip() if m else ""
    mm = maso_of(q)
    typ = r.get("type", "?")
    d = by_type.setdefault(typ, [0, 0])
    if not mm:
        nomaso += 1
        continue
    sel = select_maso_table(q, [report], n_cand=12)
    if sel and sel[0]["table_ref"] == gold:
        hit += 1; d[0] += 1
    else:
        miss += 1; d[1] += 1
tot = hit + miss
out.write(f"maso-detectable: {tot} | no-maso(fallback): {nomaso}\n")
out.write(f"PRECISION (chon dung bang gold | co mã): {hit}/{tot} = {hit*100//max(1,tot)}%\n")
out.write("theo type (hit/miss):\n")
for t, (h, m) in sorted(by_type.items()):
    out.write(f"  {t}: {h}/{h+m}\n")
open("selector_test.txt", "w", encoding="utf-8").write(out.getvalue())
print("wrote selector_test.txt")
