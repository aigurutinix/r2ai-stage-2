"""Smoke-test LLM-pandas trên N câu hard-conditional đa dạng — in question->pandas->eval
để KIEM TAY (khong co ground-truth). Chay sau khi Modal deploy xong:
  USE_LLM=1 LLM_N_SAMPLES=6 python smoke_llm.py
Chon cau da dang: multi-company max, temporal (nam thoa dieu kien), ratio-conditional."""
import os, re, json
os.environ.setdefault("USE_LLM", "1")
import pipeline as P
import build_submission as B
import llm_engine as LLM

OUT = os.path.join(os.path.dirname(__file__), "submission_out")
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)

def pick(n=15):
    """Chon n cau AGG/COND, UU TIEN cau 'simple max' (1 buoc, 1 nam) = phan lon 350 + model de thang;
    kem vai cau kho (temporal/nested) de biet tran."""
    TEMPORAL = ["năm nào", "năm ngay sau", "ghi nhận", "lần đầu", "đầu tiên", "năm sau năm", "sang năm"]
    simple, hard = [], []
    for q in P.Q:
        qt = q["question"]; l = qt.lower()
        if not any(k in l for k in P.AGG + P.COND):
            continue
        ntk = len(set(m for m in re.findall(r"\b([A-Z0-9]{2,4})\b", qt) if m in P.tickers))
        nyr = len(set(re.findall(r"\b20\d{2}\b", qt)))
        is_max = any(w in l for w in ["cao nhất", "thấp nhất", "lớn nhất", "nhỏ nhất", "nhiều nhất"])
        if ntk >= 2 and nyr <= 1 and is_max and not any(w in l for w in TEMPORAL):
            simple.append(q)      # 'trong nhom A,B,C ... cao nhat nam X' — 1 buoc
        else:
            hard.append(q)
    k_simple = max(1, n * 2 // 3)
    return (simple[:k_simple] + hard[:n - k_simple])[:n]

def main():
    qs = pick(int(os.environ.get("SMOKE_N", "15")))
    written = set()
    for i, q in enumerate(qs, 1):
        qt = q["question"]
        mc = LLM.build_multi_report_csv(qt, DATA, written)
        if not mc:
            print(f"\n[{i}] id{q['id']} CSV=None\n   Q: {qt[:100]}")
            continue
        fname, meta = mc
        ev = [{"variable": "df1", "csv_path": f"data/{fname}"}]
        prompt = LLM.make_prompt(qt, meta, B.q_unit(qt))
        pq = LLM.gen_pandas_voted(prompt, ev, OUT)
        res = LLM.eval_pandas(pq, ev, OUT)
        print(f"\n[{i}] id{q['id']}  companies={meta['companies']} years={meta['years']}")
        print(f"   Q: {qt}")
        print(f"   PANDAS: {pq[:240]}")
        print(f"   -> {res}")

if __name__ == "__main__":
    main()
