"""PIVOT: thay CODE deterministic bằng Qwen-Coder sinh pandas, GIỮ NGUYÊN evidence của base.
Cô lập đúng biến 'chất lượng code'. Nếu exec bật lên => code là nút thắt (không phải retrieval).

Mỗi câu: run_program(question, base_evidence_tables, n_vote=3, self-debug=2) trên grader sandbox.
- ok + đồng thuận => dùng answer+query MỚI.
- fail/không đồng thuận => GIỮ base (không regress).
Resumable qua build/full_program_cache.jsonl. Song song bằng thread.

Chạy:
  PYTHONUTF8=1 python -u scripts/run_full_program.py --limit 60          # smoke test
  PYTHONUTF8=1 python -u scripts/run_full_program.py                     # full 1012
  PYTHONUTF8=1 python -u scripts/run_full_program.py --build sub_llm     # ráp submission từ cache
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ["KINGPRO_LLM_BASE_URL"] = os.environ.get("BASE_CODER", "")
os.environ["KINGPRO_LLM_MODEL"] = os.environ.get("MODEL_CODER", "Qwen/Qwen2.5-Coder-14B-Instruct")

from kingpro.answering.llm_client import chat
from kingpro.answering.program_engine import run_program

GRADER_PY = ".venv-grader/Scripts/python.exe"
BASE_DIR = "sub_base_fix"
CACHE = "build/full_program_cache.jsonl"
_lock = threading.Lock()


def llm(system, user):
    return chat(system, user, temperature=0.3, max_tokens=1100, timeout=150)


def load_base():
    return json.load(open(f"{BASE_DIR}/submission.json", encoding="utf-8"))


def evidence_tables(entry):
    """Chuyển evidence của base -> list {table_ref, csv_path abs} theo ĐÚNG thứ tự."""
    out = []
    for ev in entry.get("evidence", []):
        p = ev.get("csv_path", "")
        # csv_path kiểu 'data/XXX.csv' -> file thật ở sub_base_fix/data/XXX.csv
        real = os.path.join(BASE_DIR, p)
        if os.path.exists(real):
            out.append({"table_ref": ev.get("variable", ""), "csv_path": real})
    return out


def load_done():
    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                done[r["id"]] = r
    return done


def work(entry):
    qid = entry["id"]
    q = entry["question"]
    tabs = evidence_tables(entry)
    if not tabs:
        return {"id": qid, "ok": False, "reason": "no_evidence"}
    try:
        r = run_program(q, tabs, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        return {"id": qid, "ok": False, "reason": f"err:{str(ex)[:80]}"}
    return {"id": qid, "ok": bool(r.get("ok")), "answer": r.get("answer"),
            "pandas_query": r.get("pandas_query", ""), "agree": r.get("agree", 0),
            "attempts": r.get("attempts", 0)}


def run(limit, start, workers):
    base = load_base()
    done = load_done()
    todo = [e for e in base[start:start + limit] if e["id"] not in done and e.get("evidence")]
    print(f"base={len(base)} done={len(done)} todo(this slice)={len(todo)} workers={workers}", flush=True)
    n = 0
    fh = open(CACHE, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, e): e["id"] for e in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if r.get("ok"):
                print(f"[{n}/{len(todo)}] Q{r['id']} OK ans={r.get('answer')} agree={r.get('agree')}/{r.get('attempts')}", flush=True)
            else:
                print(f"[{n}/{len(todo)}] Q{r['id']} FAIL {r.get('reason','')}", flush=True)
    fh.close()
    print("DONE slice", flush=True)


def build(out_dir):
    """Ráp submission: LLM answer khi ok, else giữ base. Giữ nguyên evidence/relevant_*."""
    import shutil
    base = load_base()
    done = load_done()
    n_new = 0
    out = []
    for e in base:
        e2 = dict(e)
        r = done.get(e["id"])
        if r and r.get("ok") and r.get("answer") is not None:
            e2["answer"] = r["answer"]
            e2["pandas_query"] = r["pandas_query"]
            n_new += 1
        out.append(e2)
    os.makedirs(out_dir, exist_ok=True)
    # copy data dir (evidence CSV giữ nguyên)
    if os.path.exists(f"{out_dir}/data"):
        shutil.rmtree(f"{out_dir}/data")
    shutil.copytree(f"{BASE_DIR}/data", f"{out_dir}/data")
    json.dump(out, open(f"{out_dir}/submission.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"BUILT {out_dir}: {n_new}/{len(out)} câu dùng code LLM (còn lại giữ base)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--build", type=str, default="")
    a = ap.parse_args()
    if a.build:
        build(a.build)
    else:
        run(a.limit, a.start, a.workers)
