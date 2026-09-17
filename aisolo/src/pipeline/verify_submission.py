"""Verify submission NHU BTC re-run: load df1 tu csv, chay pandas_query, khop answer?
+ kiem tra du 1012 id, csv ton tai, ZIP structure."""
import sys as _sys
# Console Windows mac dinh la cp1252, khong ma hoa noi tieng Viet co dau: chay dung lenh trong
# README tren mot may sach thi script chet ngay o dong in dau tien bang UnicodeEncodeError,
# truoc khi lam bat cu viec gi. Ep stdout/stderr ve utf-8 ngay tu day de nguoi cham khong phai
# biet meo dat PYTHONIOENCODING moi chay duoc.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # stream bi thay the / khong ho tro
        pass
import json, os, math, zipfile
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "submission_out")
entries = json.load(open(os.path.join(OUT, "submission.json"), encoding="utf-8"))

ids = [e["id"] for e in entries]
print(f"Tong entry: {len(entries)} | id unique: {len(set(ids))} | du 1..1012: {set(ids)==set(range(1,1013))}")

# kiem field bat buoc
bad_fmt = [e["id"] for e in entries if not all(k in e for k in
           ["id","question","answer","relevant_docs","relevant_tables","evidence","pandas_query"])]
print(f"Entry thieu field: {len(bad_fmt)}")

# mo phong BTC re-run pandas (dtype=str — case chat nhat)
have_pq = run = match = 0; execfail = []; mismatch = []; missing_csv = []
for e in entries:
    pq = e["pandas_query"]
    if not pq: continue
    have_pq += 1
    ns = {"float": float, "pd": pd}; ok_load = True   # load HET evidence (df1, df2...) nhu BTC — dtype MẶC ĐỊNH
    for ev in e["evidence"]:
        cp = os.path.join(OUT, ev["csv_path"])
        if not os.path.exists(cp): missing_csv.append(e["id"]); ok_load = False; break
        ns[ev["variable"]] = pd.read_csv(cp)
    if not ok_load: continue
    try:
        result = eval(pq, ns)
        run += 1
        if math.isclose(float(result), float(e["answer"]), rel_tol=0, abs_tol=0.01):
            match += 1
        else:
            if len(mismatch) < 5: mismatch.append((e["id"], result, e["answer"]))
    except Exception as ex:
        if len(execfail) < 5: execfail.append((e["id"], f"{type(ex).__name__}: {ex}"))

print(f"\nCau co pandas_query: {have_pq}")
print(f"Pandas CHAY duoc (dtype=str): {run}/{have_pq}")
print(f"Ket qua KHOP answer (isclose 0.01): {match}/{have_pq} ({match*100//max(1,have_pq)}%)")
print(f"CSV thieu: {len(missing_csv)} | exec fail: {len(execfail)} | mismatch: {len(mismatch)}")
for i in execfail: print("   EXECFAIL", i)
for i in mismatch: print("   MISMATCH", i)

# ZIP structure: submission.json + data/ o cap ngoai cung
zp = os.path.join(OUT, "submission.zip")
with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    top = set(n.split("/")[0] for n in names)
    has_json = "submission.json" in names
    n_json = sum(1 for n in names if n.endswith(".json"))
    all_data = all(n=="submission.json" or n.startswith("data/") for n in names)
    print(f"\nZIP: {len(names)} file | co submission.json o cap ngoai: {has_json} | so .json: {n_json}")
    print(f"     moi file la submission.json HOAC data/*: {all_data} | top-level: {top}")
