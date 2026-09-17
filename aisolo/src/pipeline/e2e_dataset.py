"""E2E trên dataset ViFinQA thật (hướng khuyến nghị: deterministic + THỰC THI pandas thật).
Mỗi câu direct-lookup: ingest report → ghi long-CSV → sinh pandas_query self-contained →
EXEC thật → verify result khớp answer deterministic (isclose abs_tol=0.01). Phần nghiên cứu, mở rộng sau."""
import pipeline as P
import pandas as pd
import csv, re, math, json, os, tempfile

OUT = tempfile.mkdtemp(prefix="vifinqa_csv_")
def pyq(s): return json.dumps(str(s))

def write_long_csv(path, ticker, ry, rows):
    """Ghi long-CSV: 1 dòng/chỉ tiêu/kỳ. value = số VND đã scale (nguyên trơn, KHÔNG dấu chấm)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "year", "table_id", "statement", "ma_so", "label", "value"])
        for r in rows:
            if r["cur"] is not None:
                w.writerow([ticker, ry, r["tid"], r["st"], r["ma"], r["label"], int(round(r["cur"]))])
            if r["prev"] is not None:
                w.writerow([ticker, ry - 1, r["tid"], r["st"], r["ma"], r["label"], int(round(r["prev"]))])

def gen_pandas(csv_path, tid, ma, label, year):
    key = f'(df1["ma_so"] == {pyq(ma)})' if ma else f'(df1["label"] == {pyq(label)})'
    return (
        "import pandas as pd\n"
        f"df1 = pd.read_csv({pyq(csv_path)}, dtype=str, keep_default_na=False)\n"
        f'result = pd.to_numeric(\n'
        f'    df1.loc[(df1["table_id"] == {pyq(str(tid))}) & (df1["year"] == {pyq(str(year))}) & {key}, "value"],\n'
        f'    errors="coerce",\n'
        f").iloc[0]"
    )

def run_pandas(code):
    ns = {}
    exec(code, ns)   # self-contained: đọc CSV + gán result (giống BTC re-run pandas_query)
    return ns["result"]

dev = P.sample
tot = ok = executed = matched = 0
mism = []; examples = []
for q in dev:
    text = q["question"]
    tk = P.resolve(text); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", text)))
    if not tk or not yrs:
        continue
    ry = int(yrs[-1])
    fr = P.find_report(tk, str(ry), P.doctype(text))
    if not fr:
        continue
    tot += 1
    rep, docname = fr
    rows, uf, src = P.ingest(rep.read_text(encoding="utf-8", errors="replace"))
    a = P.extract(text)
    if a.get("val") is None:
        continue
    ok += 1
    year = ry - 1 if re.search(r"đầu (năm|kỳ)|01/01", text.lower()) else ry
    csv_path = os.path.join(OUT, f"{docname}_bctc.csv")
    write_long_csv(csv_path, tk, ry, rows)
    code = gen_pandas(csv_path, a["tid"], a["maso"], a["flabel"], year)
    try:
        result = run_pandas(code)
        executed += 1
    except Exception as e:
        mism.append((q["id"], f"EXEC FAIL: {type(e).__name__}: {e}"))
        continue
    if result is not None and not (isinstance(result, float) and math.isnan(result)) and \
       math.isclose(float(result), float(a["val"]), rel_tol=0, abs_tol=0.01):
        matched += 1
        if len(examples) < 3:
            examples.append((q["id"], text[:60], code, result))
    else:
        mism.append((q["id"], f"MISMATCH pandas={result} vs answer={a['val']} ({text[:50]})"))

print(f"=== E2E TRÊN DATASET ViFinQA (dev direct-lookup) ===")
print(f"Có report+answer: {ok}/{tot}")
print(f"Pandas THỰC THI được (không crash): {executed}/{ok} ({executed*100//max(1,ok)}%)")
print(f"Kết quả pandas KHỚP answer deterministic (isclose 0.01): {matched}/{ok} ({matched*100//max(1,ok)}%)")
print(f"→ Execution Accuracy proxy trên slice này: {matched*100//max(1,ok)}%")
print(f"\nCSV sinh ra ở: {OUT}")
if examples:
    print(f"\n--- Ví dụ pandas_query CHẠY THẬT + khớp ---")
    eid, eq, ecode, eres = examples[0]
    print(f"[id{eid}] {eq}")
    print(ecode)
    print(f"# result = {eres}  ✓ khớp answer")
if mism:
    print(f"\n--- Ca lệch/lỗi ({len(mism)}) ---")
    for i, m in mism[:8]:
        print(f"  id{i}: {m}")
