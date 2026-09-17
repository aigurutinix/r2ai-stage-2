"""ĐÒN #1 dùng ĐÚNG CÁCH: LLM đọc STATEMENT GHÉP (merge = evidence, KHÔNG phải deterministic override).
Kết hợp: merge (phá phân-mảnh, mọi toán hạng cùng 1 df) + Qwen (đọc statement giỏi hơn deterministic).
evidence = merged_statements(report) [fallback mảnh top-k cho ngân hàng/CK/report lỗi] -> run_program vote5.

Chạy:
  PYTHONUTF8=1 python -u scripts/run_llm_merged.py --workers 14
  PYTHONUTF8=1 python -u scripts/run_llm_merged.py --build sub_llmmerged
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
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ["KINGPRO_LLM_BASE_URL"] = os.environ.get("BASE_CODER", "")
os.environ["KINGPRO_LLM_MODEL"] = os.environ.get("MODEL_CODER", "Qwen/Qwen2.5-Coder-14B-Instruct")

from kingpro.tables.merge_statements import merged_statements
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.program_engine import run_program
from kingpro.answering.llm_client import chat
from kingpro.evaluation.metrics import doc_of, coerce_number

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QS = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")]
BASE = {e["id"]: e for e in json.load(open("sub_ctxB/submission.json", encoding="utf-8"))}
CACHE = "build/llm_merged_cache.jsonl"
MERGED_DIR = "build/merged"
os.makedirs(MERGED_DIR, exist_ok=True)
_lock = threading.Lock()
_mcache = {}


def csv_full(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def merged_paths(report_id):
    if report_id in _mcache:
        return _mcache[report_id]
    out = []
    try:
        m = merged_statements(report_id)
    except Exception:
        m = {}
    for typ, df in m.items():
        p = os.path.join(MERGED_DIR, f"{report_id}__{typ}.csv")
        if not os.path.exists(p):
            df.to_csv(p, index=False, encoding="utf-8-sig")
        out.append({"ref": f"{report_id}|{typ}", "path": p})
    _mcache[report_id] = out
    return out


def llm(system, user):
    return chat(system, user, temperature=0.6, max_tokens=1200, timeout=300)


def evidence_for(qid, q):
    """CUNG CẤP ĐỦ evidence (BTC dặn: câu 5 df thì đủ 5). STATEMENT GHÉP của MỌI report liên quan
    (đa-năm/đa-công-ty -> mỗi năm/công-ty 1 report); fallback mảnh top-k cho ngân hàng/CK."""
    docs = (BASE.get(qid, {}) or {}).get("relevant_docs", []) or []
    hits = retrieve_decomposed(q)
    rdocs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    # gộp docs base + retrieval (đủ mọi năm/công-ty), khử trùng, giữ thứ tự
    for d in rdocs:
        if d not in docs:
            docs.append(d)
    # câu đa-thực-thể (nhiều năm/mã) -> cần nhiều report; câu đơn -> 1-2 report
    n_years = len(set(re.findall(r"20\d{2}", q)))
    n_rep = min(6, n_years + 1) if n_years >= 2 else 2   # gọn: đủ năm nhưng KHÔNG over-feed (chống timeout)
    tabs = []
    for d in docs[:n_rep]:
        for mp in merged_paths(d):        # CDKT/KQKD/LCTT của report đó
            tabs.append({"table_ref": mp["ref"], "csv_path": mp["path"]})
    if not tabs:                          # ngân hàng/CK/report lỗi -> fallback mảnh
        for c in tables_in_reports(q, rdocs, n=8):
            p = csv_full(c["table_ref"])
            if p:
                tabs.append({"table_ref": c["table_ref"], "csv_path": p})
    return tabs[:6], docs                 # cap 6 df: prompt gọn -> nhanh, không timeout, model đọc kỹ hơn


def work(qrec):
    qid, q = qrec["id"], qrec["question"]
    try:
        tabs, docs = evidence_for(qid, q)
        if not tabs:
            return {"id": qid, "ok": False, "reason": "no_ev", "ev": [], "rel_docs": []}
        r = run_program(q, tabs, llm, max_fix=1, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        return {"id": qid, "ok": False, "reason": f"err:{str(ex)[:80]}", "ev": [], "rel_docs": []}
    ans = coerce_number(r.get("answer")) if r.get("ok") else None
    return {"id": qid, "ok": bool(r.get("ok") and ans is not None), "answer": ans,
            "pandas_query": r.get("pandas_query", ""), "agree": r.get("agree", 0), "attempts": r.get("attempts", 0),
            "ev": [{"ref": t["table_ref"], "path": t["csv_path"]} for t in tabs], "rel_docs": docs}


def load_done():
    d = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                d[r["id"]] = r
    return d


def run(workers):
    done = load_done()
    todo = [q for q in QS if q["id"] not in done]
    print(f"total={len(QS)} done={len(done)} todo={len(todo)} workers={workers}", flush=True)
    retrieve_decomposed("warm")
    fh = open(CACHE, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, q): q["id"] for q in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if n % 25 == 0 or r.get("ok"):
                print(f"[{n}/{len(todo)}] Q{r['id']} ok={r.get('ok')} ans={r.get('answer')} agree={r.get('agree')}/{r.get('attempts')}", flush=True)
    fh.close()
    print("LLM-MERGED DONE", flush=True)


def build(out_dir):
    done = load_done()
    ddir = os.path.join(out_dir, "data")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(ddir, exist_ok=True)
    copied = {}
    rows = []
    n_ok = 0
    for q in QS:
        qid = q["id"]
        r = done.get(qid)
        base_e = BASE.get(qid, {})
        # mặc định GIỮ base; chỉ thay khi LLM-merged ok (đây là submission tươi, so base độc lập)
        e2 = {"id": qid, "question": q["question"],
              "relevant_docs": (r.get("rel_docs") if r else None) or base_e.get("relevant_docs", []),
              "relevant_tables": base_e.get("relevant_tables", []),
              "evidence": [], "answer": 0.0, "pandas_query": ""}
        if r and r.get("ok") and r.get("answer") is not None and r.get("ev"):
            evidence = []
            for j, ev in enumerate(r["ev"], 1):
                ref, src = ev["ref"], ev["path"]
                if not os.path.exists(src):
                    continue
                if ref not in copied:
                    dst = re.sub(r"[^0-9A-Za-z_]+", "_", ref) + ".csv"
                    shutil.copyfile(src, os.path.join(ddir, dst))
                    copied[ref] = f"data/{dst}"
                evidence.append({"variable": f"df{j}", "csv_path": copied[ref]})
            if evidence:
                e2["evidence"] = evidence
                e2["answer"] = float(r["answer"])
                e2["pandas_query"] = r.get("pandas_query", "") or ""
                n_ok += 1
        rows.append(e2)
    json.dump(rows, open(os.path.join(out_dir, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if not any(f.endswith(".csv") for f in os.listdir(ddir)):
        open(os.path.join(ddir, "_placeholder.csv"), "w").write("0\n0\n")
    zp = f"{out_dir}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out_dir, "submission.json"), "submission.json")
        for f in os.listdir(ddir):
            z.write(os.path.join(ddir, f), f"data/{f}")
    print(f"ZIP -> {zp} | {n_ok}/{len(QS)} câu có đáp án LLM-merged, {len(copied)} CSV", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--build", type=str, default="")
    a = ap.parse_args()
    if a.build:
        build(a.build)
    else:
        run(a.workers)
