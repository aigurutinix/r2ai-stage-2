"""GPU-FREE: đè maso nâng cấp (lookup+ratio, 95% dung) lên base+fix. KHONG goi LLM."""
import sys, json, zipfile, os, re, shutil
sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, retrieve
from kingpro.answering.pandas_answer import maso_answer, ratio_answer
from kingpro.evaluation.metrics import doc_of

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def csv_full(tref):
    r = CAT.get(tref)
    return "build/tables/" + r["csv_path"] if r else None


def safe(tref):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tref) + ".csv"


SRC, OUT = "sub_base_fix", "sub_maso"
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
retrieve("khoi dong", k=1)
existing = set(os.listdir(f"{OUT}/data"))
n_over = 0
for i, e in enumerate(rows):
    q = e["question"]
    try:
        hits = retrieve_decomposed(q)
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel_docs, n=50)
        tables = [{"csv_path": csv_full(h["table_ref"]), "table_ref": h["table_ref"]}
                  for h in atabs if csv_full(h["table_ref"])]
        res = ratio_answer(q, tables) or maso_answer(q, tables)
    except Exception:
        res = None
    if res and res.get("ok") and res.get("answer") is not None and res.get("evidence"):
        ev = res["evidence"][0]
        tref = ev["table_ref"]
        nm = safe(tref)
        if nm not in existing:
            shutil.copyfile(csv_full(tref), f"{OUT}/data/{nm}")
            existing.add(nm)
        e["answer"] = float(res["answer"])
        e["pandas_query"] = res["pandas_query"]
        e["evidence"] = [{"variable": ev["variable"], "csv_path": f"data/{nm}"}]
        n_over += 1
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(rows)} override={n_over}", flush=True)
print("TOTAL override", n_over)
json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = "sub_maso.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print("ZIP", zp, os.path.getsize(zp), "bytes")
