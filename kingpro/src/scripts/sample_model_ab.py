"""A/B test MODEL ≤14.9B trên MẪU STRATIFIED cố định (~120 câu, đủ bucket) — KHÔNG cần full 1024.
Quy trình: deploy model X trên RunPod (đổi MODEL_CODER/.env) -> chạy script tag=X -> lặp cho model Y
-> so consensus theo bucket (consensus cao ở bucket khó = model giỏi hơn). Chỉ full-run kẻ thắng.

Pipeline = run_llm_merged (evidence statement GHÉP + đủ df). Model lấy từ env hiện tại của endpoint.

Chạy:
  # sau khi deploy model + set BASE_CODER/MODEL_CODER trong .env:
  PYTHONUTF8=1 python -u scripts/sample_model_ab.py --tag qwen_coder     # baseline hiện tại
  PYTHONUTF8=1 python -u scripts/sample_model_ab.py --tag qwen_instruct  # sau khi swap model
  PYTHONUTF8=1 python -u scripts/sample_model_ab.py --compare qwen_coder qwen_instruct phi4
"""
import argparse
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import importlib.util

_spec = importlib.util.spec_from_file_location("rm", "scripts/run_llm_merged.py")
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)
from kingpro.retrieval.bm25_index import retrieve_decomposed

QS = {q["id"]: q["question"] for q in rm.QS}
_lock = threading.Lock()

YEARS = re.compile(r"20\d{2}")
SEL = re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất|trung vị", re.I)
GROUP = re.compile(r"nhóm|các công ty|các doanh nghiệp|các ngân hàng", re.I)
LIST = re.compile(r"các năm|những năm|giai đoạn", re.I)
PCT = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ|tỷ trọng|biên|tỷ suất|ROE|ROA", re.I)
LAN = re.compile(r"bao nhiêu lần|số lần|hệ số|vòng quay|D/E", re.I)


def bucket(q):
    yrs = set(YEARS.findall(q))
    if GROUP.search(q) and SEL.search(q):
        return "H cross-company"
    if len(yrs) >= 2 and SEL.search(q) and (LIST.search(q) or len(yrs) >= 3):
        return "M year-selector"
    if LAN.search(q):
        return "R hệ số/lần"
    if PCT.search(q):
        return "F/S tỷ số/%"
    return "L lookup"


def stratified_sample(n_per=30):
    byb = defaultdict(list)
    for qid, q in QS.items():
        byb[bucket(q)].append(qid)
    samp = []
    for b in sorted(byb):
        samp += sorted(byb[b])[:n_per]     # deterministic -> mọi model chạy CÙNG câu
    return sorted(set(samp))


def run(tag, workers):
    cache = f"build/ab_{tag}.jsonl"
    done = set()
    if os.path.exists(cache):
        for l in open(cache, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    samp = stratified_sample()
    todo = [q for q in samp if q not in done]
    print(f"[{tag}] model={os.environ.get('MODEL_CODER','?')} | mẫu={len(samp)} done={len(done)} todo={len(todo)}", flush=True)
    retrieve_decomposed("warm")
    fh = open(cache, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rm.work, {"id": q, "question": QS[q]}): q for q in todo}
        for fut in as_completed(futs):
            r = fut.result()
            r["bucket"] = bucket(QS[r["id"]])
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if n % 10 == 0:
                print(f"  [{tag}] {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"[{tag}] XONG", flush=True)


def compare(tags):
    from kingpro.evaluation.metrics import coerce_number
    data = {}
    for t in tags:
        p = f"build/ab_{t}.jsonl"
        if not os.path.exists(p):
            print(f"  (thiếu {p})")
            continue
        data[t] = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
    if len(data) < 1:
        return
    # consensus mạnh theo bucket cho từng tag
    print(f"\n{'bucket':<18}" + "".join(f"{t[:12]:>16}" for t in data))
    buckets = sorted(set(b for d in data.values() for r in d.values() for b in [r.get('bucket', '?')]))
    for b in buckets:
        row = f"{b:<18}"
        for t, d in data.items():
            rs = [r for r in d.values() if r.get("bucket") == b]
            strong = sum(1 for r in rs if r.get("attempts", 0) >= 4 and r.get("agree", 0) >= r.get("attempts", 0))
            ok = sum(1 for r in rs if r.get("ok"))
            row += f"{f'{strong}/{ok}/{len(rs)}':>16}"
        print(row)
    print("(mạnh5/5 / có-đáp-án / tổng)  — bucket khó mà consensus mạnh cao hơn = model giỏi hơn")
    # head-to-head: câu 2 model KHÁC nhau (dấu hiệu 1 con sai)
    if len(data) == 2:
        t1, t2 = list(data)
        d1, d2 = data[t1], data[t2]
        both = set(d1) & set(d2)
        agree = diff = 0
        for q in both:
            a, b = coerce_number(d1[q].get("answer")), coerce_number(d2[q].get("answer"))
            if a is not None and b is not None:
                if abs(a - b) <= max(0.01, abs(b) * 0.005):
                    agree += 1
                else:
                    diff += 1
        print(f"\nHead-to-head {t1} vs {t2}: khớp {agree}, KHÁC {diff} (câu khác = ít nhất 1 con sai -> soi tay)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--compare", nargs="+", default=[])
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    elif a.tag:
        run(a.tag, a.workers)
    else:
        print(__doc__)
