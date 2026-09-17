import sys, json, re, io
sys.path.insert(0, "src")
from kingpro.answering.pandas_answer import maso_answer, ratio_answer

rows = [json.loads(l) for l in open("build/sft_chatml_val.jsonl", encoding="utf-8")]
CAT = {r["table_ref"]: "build/tables/" + r["csv_path"]
       for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
out = io.StringIO()
hit = miss = none = 0
by_type = {}
for r in rows:
    trefs = r.get("table_refs") or []
    if not trefs:
        continue
    gold = trefs[0]
    try:
        ans = float(r["answer"])
    except Exception:
        continue
    user = r["conversations"][1]["content"]
    m = re.search(r"Câu hỏi:\s*(.+)", user)
    q = m.group(1).strip() if m else ""
    typ = r.get("type", "?")
    d = by_type.setdefault(typ, [0, 0, 0])
    tables = [{"csv_path": CAT[gold], "table_ref": gold}]
    res = ratio_answer(q, tables) or maso_answer(q, tables)
    if not res or not res.get("ok") or res.get("answer") is None:
        none += 1; d[2] += 1; continue
    if abs(float(res["answer"]) - ans) <= 0.01:
        hit += 1; d[0] += 1
    else:
        miss += 1; d[1] += 1
tot = hit + miss
out.write(f"maso tra loi: {tot}/{len(rows)} (none={none}) | DUNG {hit}/{tot} = {hit*100//max(1,tot)}%\n")
for t, (h, m, n) in sorted(by_type.items()):
    out.write(f"  {t}: dung {h} sai {m} none {n}\n")
open("maso_test.txt", "w", encoding="utf-8").write(out.getvalue())
print("wrote")
