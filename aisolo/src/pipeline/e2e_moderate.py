"""E2E slice MODERATE v2 (anchor-and-propagate + clean_metric).
Khớp metric ở 1 report (anchor) → tìm ĐÚNG cùng line-item ở report kia (locate_like, ưu tiên cùng Mã số)
→ đồng bộ giữa report, hết lỗi 'khớp lệch line-item'. Sinh pandas self-contained → THỰC THI thật → verify.
Đo: coverage (tìm được cùng metric ở cả 2) + execution + eyeball anchor (correctness ~ mức direct)."""
import pipeline as P
import pandas as pd
import csv, re, math, json, os, tempfile

OUT = tempfile.mkdtemp(prefix="vifinqa_mod2_")
def pyq(s): return json.dumps(str(s))
def keycond(ma, label): return f'(df1["ma_so"] == {pyq(ma)})' if ma else f'(df1["label"] == {pyq(label)})'

def load_rows(tk, year, dtype):
    fr = P.find_report(tk, str(year), dtype)
    if not fr: return None
    rep, _ = fr
    return P.ingest(rep.read_text(encoding="utf-8", errors="replace"))[0]

def run(code):
    ns = {}; exec(code, ns); return ns["result"]
def check(det, code):
    try: r = run(code)
    except Exception as e: return "EXECFAIL", f"{type(e).__name__}: {e}"
    if r is None or (isinstance(r, float) and math.isnan(r)): return "NAN", r
    return ("MATCH" if math.isclose(float(r), float(det), rel_tol=0, abs_tol=0.01) else "MISMATCH"), r

def do_multiyear(q):
    tk = P.resolve(q); years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    if not tk or len(years) < 2: return None
    y1, y2 = years[0], years[-1]; dt = P.doctype(q)
    target = P.clean_metric(P.target_label(q))
    r2 = load_rows(tk, y2, dt); r1 = load_rows(tk, y1, dt)
    if r2 is None or r1 is None: return {"cover": False}
    a = P.locate(r2, target, None)                       # anchor ở năm mới
    if not a or (a["ov"] is not None and a["ov"] < 0.5): return {"cover": False}  # guardrail: ov thấp = không chắc metric → refuse
    b = P.locate_like(r1, a["maso"], a["label"])         # propagate cùng line-item sang năm cũ
    if not b or not b["cur"]: return {"cover": False}
    low = q.lower()
    diff_mode = any(k in low for k in ["chênh lệch", "hiệu số"]) and "%" not in q and "phần trăm" not in low
    det = (a["cur"] - b["cur"]) if diff_mode else round((a["cur"] - b["cur"]) / b["cur"] * 100, 2)
    path = os.path.join(OUT, f"{tk}_{y1}_{y2}_bctc.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["ticker", "year", "ma_so", "label", "value"])
        w.writerow([tk, y2, a["maso"], a["label"], int(round(a["cur"]))])
        w.writerow([tk, y1, b["maso"], b["label"], int(round(b["cur"]))])
    code = ("import pandas as pd\n"
            f"df1 = pd.read_csv({pyq(path)}, dtype=str, keep_default_na=False)\n"
            f'v2 = pd.to_numeric(df1.loc[(df1["year"] == {pyq(str(y2))}) & {keycond(a["maso"], a["label"])}, "value"], errors="coerce").iloc[0]\n'
            f'v1 = pd.to_numeric(df1.loc[(df1["year"] == {pyq(str(y1))}) & {keycond(b["maso"], b["label"])}, "value"], errors="coerce").iloc[0]\n'
            + ("result = v2 - v1" if diff_mode else "result = round((v2 - v1) / v1 * 100, 2)"))
    st, r = check(det, code)
    return {"cover": True, "status": st, "det": det, "res": r, "code": code, "anchor": a["label"], "ov": a["ov"]}

def do_multicompany(q, ticks):
    a_tk, b_tk = ticks[0], ticks[1]; years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    if not years: return {"cover": False}
    yr = years[-1]; dt = P.doctype(q); target = P.clean_metric(P.target_label(q))
    ra = load_rows(a_tk, yr, dt); rb = load_rows(b_tk, yr, dt)
    if ra is None or rb is None: return {"cover": False}
    a = P.locate(ra, target, None)
    if not a or (a["ov"] is not None and a["ov"] < 0.5): return {"cover": False}  # guardrail: refuse khi không chắc
    b = P.locate_like(rb, a["maso"], a["label"])
    if not b or b["cur"] is None: return {"cover": False}
    det = a["cur"] - b["cur"]
    path = os.path.join(OUT, f"{a_tk}_{b_tk}_{yr}_bctc.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["ticker", "year", "ma_so", "label", "value"])
        w.writerow([a_tk, yr, a["maso"], a["label"], int(round(a["cur"]))])
        w.writerow([b_tk, yr, b["maso"], b["label"], int(round(b["cur"]))])
    code = ("import pandas as pd\n"
            f"df1 = pd.read_csv({pyq(path)}, dtype=str, keep_default_na=False)\n"
            f'a = pd.to_numeric(df1.loc[(df1["ticker"] == {pyq(a_tk)}) & {keycond(a["maso"], a["label"])}, "value"], errors="coerce").iloc[0]\n'
            f'b = pd.to_numeric(df1.loc[(df1["ticker"] == {pyq(b_tk)}) & {keycond(b["maso"], b["label"])}, "value"], errors="coerce").iloc[0]\n'
            "result = a - b")
    st, r = check(det, code)
    return {"cover": True, "status": st, "det": det, "res": r, "code": code, "anchor": a["label"], "ov": a["ov"]}

Q = P.Q
def ntk(q): return [m for m in dict.fromkeys(re.findall(r"\b([A-Z0-9]{2,4})\b", q)) if m in P.tickers]
AGG = P.AGG; COND = P.COND; OWN = P.OWN
def is_multiyear(q):
    low = q.lower()
    if any(k in low for k in AGG + COND + OWN): return False
    return len(set(re.findall(r"\b20\d{2}\b", q))) >= 2 and len(set(ntk(q))) == 1
def is_multicompany(q):
    low = q.lower()
    if any(k in low for k in AGG + COND + OWN): return False
    return len(set(ntk(q))) == 2 and any(k in low for k in ["chênh lệch", "so sánh", "hơn", "kém"])

MY = [q for q in Q if is_multiyear(q["question"])]
MC = [q for q in Q if is_multicompany(q["question"])]

def summarize(name, items, fn):
    n = len(items); samp = items[:: max(1, n // 15)][:15]
    cov = execmatch = 0; anchors = []
    for q in samp:
        r = fn(q["question"]) if fn.__code__.co_argcount == 1 else fn(q["question"], list(dict.fromkeys(ntk(q["question"]))))
        if not r or not r.get("cover"): continue
        cov += 1
        if r["status"] == "MATCH": execmatch += 1
        anchors.append((q["id"], round(r.get("ov") or 1.0, 2), r.get("anchor", "")[:34], q["question"][:46]))
    print(f"\n=== {name}: tổng {n}, thử {len(samp)} ===")
    print(f"Cover (anchor + propagate CÙNG line-item ở cả 2 report): {cov}/{len(samp)}")
    print(f"Pandas execute + khớp deterministic: {execmatch}/{cov} (plumbing OK)")
    print(f"→ Anchor để eyeball (ov=độ khớp metric hỏi; thấp = nghi sai):")
    for a in sorted(anchors, key=lambda x: x[1])[:8]:
        print(f"   id{a[0]} ov={a[1]} anchor='{a[2]}'  | {a[3]}")

summarize("MULTI-YEAR", MY, do_multiyear)
summarize("MULTI-COMPANY", MC, do_multicompany)
print(f"\nCSV: {OUT}")
