"""PIVOT (GPT + user chọn GPU): LLM program-search + MBR-Exec cho ĐỘ ĐÚNG ~960 câu.
Kiến trúc: deterministic = bộ GỢI Ý evidence (maso-table precision + top-k report) ->
Qwen-Coder sinh pandas -> chạy N mẫu grader sandbox -> vote theo KẾT-QUẢ (MBR-Exec) -> chọn cụm đông.

Chạy:
  PYTHONUTF8=1 python -u scripts/run_llm_mbr.py                    # full 1012 (resume qua cache)
  PYTHONUTF8=1 python -u scripts/run_llm_mbr.py --build sub_mbr    # ráp submission tươi từ cache
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
sys.path.insert(0, ".")
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ["KINGPRO_LLM_BASE_URL"] = os.environ.get("BASE_CODER", "")
os.environ["KINGPRO_LLM_MODEL"] = os.environ.get("MODEL_CODER", "Qwen/Qwen2.5-Coder-14B-Instruct")

from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.program_engine import run_program
from kingpro.answering.llm_client import chat
from kingpro.evaluation.metrics import doc_of, coerce_number
from scripts.build_full_submission import csv_full, safe_name, select_maso_table

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QS = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")]
CACHE = "build/llm_mbr_cache.jsonl"
_lock = threading.Lock()


def llm(system, user):
    return chat(system, user, temperature=0.6, max_tokens=1300, timeout=180)


def evidence_for(q, frags=6):
    """GỢI Ý deterministic: maso-table (statement chứa mã số câu hỏi) TRƯỚC + top-k report (recall). Dedup, giữ thứ tự.
    frags = số mảnh feed (bảng chẻ ~70 mảnh/report -> nhiều mảnh = dòng đáp án dễ lọt context)."""
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    refs = []
    try:
        for c in select_maso_table(q, rel, n_cand=12):
            refs.append(c["table_ref"])
    except Exception:
        pass
    for c in tables_in_reports(q, rel, n=frags):
        if c["table_ref"] not in refs:
            refs.append(c["table_ref"])
    refs = refs[:frags]
    tabs = [{"table_ref": r, "csv_path": csv_full(r)} for r in refs if csv_full(r)]
    return tabs, rel


def work(qrec, frags=6):
    qid, q = qrec["id"], qrec["question"]
    try:
        tabs, rel = evidence_for(q, frags=frags)
        if not tabs:
            return {"id": qid, "ok": False, "reason": "no_tables", "table_refs": [], "rel_docs": []}
        r = run_program(q, tabs, llm, max_fix=2, n_vote=5, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        return {"id": qid, "ok": False, "reason": f"err:{str(ex)[:80]}", "table_refs": [], "rel_docs": []}
    ans = coerce_number(r.get("answer")) if r.get("ok") else None
    return {"id": qid, "ok": bool(r.get("ok") and ans is not None), "answer": ans,
            "pandas_query": r.get("pandas_query", ""), "agree": r.get("agree", 0),
            "attempts": r.get("attempts", 0),
            "table_refs": [t["table_ref"] for t in tabs], "rel_docs": rel}


def load_done():
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                done[r["id"]] = r
    return done


def run(workers, frags=6):
    done = load_done()
    todo = [q for q in QS if q["id"] not in done]
    print(f"total={len(QS)} done={len(done)} todo={len(todo)} workers={workers} frags={frags} cache={CACHE}", flush=True)
    retrieve_decomposed("warm")
    fh = open(CACHE, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, q, frags): q["id"] for q in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if n % 20 == 0 or r.get("ok"):
                print(f"[{n}/{len(todo)}] Q{r['id']} ok={r.get('ok')} ans={r.get('answer')} agree={r.get('agree')}/{r.get('attempts')}", flush=True)
    fh.close()
    print("LLM-MBR DONE", flush=True)


def build(out_dir):
    """Ráp submission TƯƠI: mọi câu = kết quả pipeline mới; câu fail -> answer=0 + rỗng (không có gì để mất)."""
    done = load_done()
    data_dir = os.path.join(out_dir, "data")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(data_dir, exist_ok=True)
    copied = {}
    rows = []
    n_ok = 0
    for q in QS:
        r = done.get(q["id"])
        evidence, rel_tables = [], []
        answer, pq = 0.0, ""
        if r and r.get("table_refs"):
            rel_tables = r["table_refs"]
            for j, tref in enumerate(r["table_refs"], 1):
                if tref not in copied:
                    src = csv_full(tref)
                    if not src or not os.path.exists(src):
                        continue
                    dst = safe_name(tref)
                    shutil.copyfile(src, os.path.join(data_dir, dst))
                    copied[tref] = f"data/{dst}"
                evidence.append({"variable": f"df{j}", "csv_path": copied[tref]})
            if r.get("ok") and r.get("answer") is not None:
                answer, pq = float(r["answer"]), r.get("pandas_query", "") or ""
                n_ok += 1
        rows.append({"id": q["id"], "question": q["question"],
                     "relevant_docs": r.get("rel_docs", []) if r else [],
                     "relevant_tables": rel_tables, "evidence": evidence,
                     "answer": answer, "pandas_query": pq})
    json.dump(rows, open(os.path.join(out_dir, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if not any(f.endswith(".csv") for f in os.listdir(data_dir)):
        open(os.path.join(data_dir, "_placeholder.csv"), "w").write("0\n0\n")
    zip_path = f"{out_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out_dir, "submission.json"), "submission.json")
        for f in os.listdir(data_dir):
            z.write(os.path.join(data_dir, f), f"data/{f}")
    print(f"ZIP -> {zip_path} | {n_ok}/{len(QS)} câu có đáp án, {len(copied)} CSV", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--frags", type=int, default=6, help="số mảnh bảng feed vào evidence (6=config A; 14=config B nhiều-mảnh)")
    ap.add_argument("--cache", type=str, default="build/llm_mbr_cache.jsonl", help="file cache (config B dùng file khác để không trộn A)")
    ap.add_argument("--build", type=str, default="")
    a = ap.parse_args()
    CACHE = a.cache
    if a.build:
        build(a.build)
    else:
        run(a.workers, a.frags)
