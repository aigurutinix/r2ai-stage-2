"""⭐ MỎ VÀNG 24%: lấp 240 câu base BỎ TRỐNG (rỗng code+evidence -> auto 0 điểm).
Nguyên nhân: base's run_program fail -> evidence=[] -> câu trống. Nhưng retrieval TÌM ĐÚNG bảng.
Cách: tự retrieve (maso-table precision + top-k report) -> LUÔN bundle evidence -> sinh code program_engine.
Kể cả không đồng thuận, vẫn ghi best code + evidence (đoán > chắc chắn 0).

Chạy:
  PYTHONUTF8=1 python -u scripts/run_fill_empties.py                 # chạy fill (resume qua cache)
  PYTHONUTF8=1 python -u scripts/run_fill_empties.py --build sub_fill # ráp: base 772 + 240 lấp
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
sys.path.insert(0, ".")  # để 'scripts.build_full_submission' import được khi chạy python scripts/x.py
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

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
BASE_DIR = "sub_base_fix"
CACHE = "build/fill_empties_cache.jsonl"
_lock = threading.Lock()

# select_maso_table từ builder gốc (chọn bảng chứa mã số câu hỏi, precision cao)
sys.argv = sys.argv[:1]  # tránh builder parse argv
from scripts.build_full_submission import csv_full, safe_name, select_maso_table  # noqa: E402


def llm(system, user):
    return chat(system, user, temperature=0.4, max_tokens=1200, timeout=180)


def evidence_for(q):
    """Bảng đưa vào evidence: maso-table (precision) + top-k trong đúng báo cáo (recall). Giữ thứ tự, dedup."""
    hits = retrieve_decomposed(q)
    rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    refs = []
    try:
        for c in select_maso_table(q, rel_docs, n_cand=12):
            refs.append(c["table_ref"])
    except Exception:
        pass
    for c in tables_in_reports(q, rel_docs, n=6):
        if c["table_ref"] not in refs:
            refs.append(c["table_ref"])
    refs = refs[:6]
    tabs = [{"table_ref": r, "csv_path": csv_full(r)} for r in refs if csv_full(r)]
    return tabs, rel_docs


def work(entry):
    qid, q = entry["id"], entry["question"]
    try:
        tabs, rel_docs = evidence_for(q)
        if not tabs:
            return {"id": qid, "ok": False, "reason": "no_tables", "table_refs": [], "rel_docs": []}
        r = run_program(q, tabs, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        return {"id": qid, "ok": False, "reason": f"err:{str(ex)[:80]}", "table_refs": [], "rel_docs": []}
    ans = coerce_number(r.get("answer")) if r.get("ok") else None
    return {"id": qid, "ok": bool(r.get("ok") and ans is not None),
            "answer": ans, "pandas_query": r.get("pandas_query", ""),
            "agree": r.get("agree", 0), "attempts": r.get("attempts", 0),
            "table_refs": [t["table_ref"] for t in tabs], "rel_docs": rel_docs}


def load_done():
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                done[r["id"]] = r
    return done


def run(workers):
    base = json.load(open(f"{BASE_DIR}/submission.json", encoding="utf-8"))
    empty = [e for e in base if not (e.get("pandas_query") or "").strip()]
    done = load_done()
    todo = [e for e in empty if e["id"] not in done]
    print(f"empty={len(empty)} done={len(done)} todo={len(todo)} workers={workers}", flush=True)
    retrieve_decomposed("warm")
    fh = open(CACHE, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, e): e["id"] for e in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            tag = "OK" if r.get("ok") else "fail"
            print(f"[{n}/{len(todo)}] Q{r['id']} {tag} ans={r.get('answer')} agree={r.get('agree')}/{r.get('attempts')}", flush=True)
    fh.close()
    print("FILL DONE", flush=True)


def build(out_dir):
    """Ráp: base (772 câu có code giữ nguyên) + 240 câu lấp từ cache. Copy CSV evidence vào data/."""
    base = json.load(open(f"{BASE_DIR}/submission.json", encoding="utf-8"))
    done = load_done()
    data_dir = os.path.join(out_dir, "data")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(data_dir, exist_ok=True)
    # copy sẵn CSV data của base (cho 772 câu có evidence)
    for f in os.listdir(os.path.join(BASE_DIR, "data")):
        shutil.copyfile(os.path.join(BASE_DIR, "data", f), os.path.join(data_dir, f))
    copied = {}
    n_fill, n_ok = 0, 0
    out = []
    for e in base:
        e2 = dict(e)
        if not (e.get("pandas_query") or "").strip():   # câu rỗng -> lấp từ cache
            r = done.get(e["id"])
            if r and r.get("table_refs"):
                evidence = []
                for i, tref in enumerate(r["table_refs"], 1):
                    if tref not in copied:
                        src = csv_full(tref)
                        if not src or not os.path.exists(src):
                            continue
                        dst = safe_name(tref)
                        shutil.copyfile(src, os.path.join(data_dir, dst))
                        copied[tref] = f"data/{dst}"
                    evidence.append({"variable": f"df{i}", "csv_path": copied[tref]})
                e2["evidence"] = evidence
                e2["relevant_tables"] = r["table_refs"]
                e2["relevant_docs"] = r.get("rel_docs", [])
                e2["pandas_query"] = r.get("pandas_query", "") or ""
                av = r.get("answer")
                e2["answer"] = float(av) if av is not None else 0.0
                n_fill += 1
                if r.get("ok"):
                    n_ok += 1
        out.append(e2)
    json.dump(out, open(os.path.join(out_dir, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"BUILT {out_dir}: lấp {n_fill}/240 câu (ok={n_ok}), copy thêm {len(copied)} CSV", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--build", type=str, default="")
    a = ap.parse_args()
    if a.build:
        build(a.build)
    else:
        run(a.workers)
