"""DO: model doc BANG GOC co nhac duoc dong dung tu hang 2-3 len hang 1 khong?

BOI CANH — vi sao dung 41 ca nay chu khong phai tap khac (diag_pool.py / row-selection-exhausted):
tren 104 cau sai cua dev set, 87.5% co dong dung NAM SAN trong `cands`, trung vi hang 4, va 41 ca
o hang 2-3. Tuc thong tin da o trong ro; thieu duy nhat mot tin hieu du tot de nhac len hang 1.
Bon tin hieu tu vung/cap bang deu da bi chia-doi bac bo. Cau hoi con lai: TIN HIEU NGU NGHIA
(model doc bang goc) co lam duoc khong? Neu KHONG thi fine-tune/chung cat gan nhu chac chan cung
khong cuu duoc, va ta tiet kiem duoc vai ngay.

*** BAY DO LUONG PHAI TRANH ***
Gold cua dev set (`llm_val`) do MOT LLM DOC BANG THO sinh ra. Cham thang "model co trung gold khong"
la do su DONG THUAN GIUA HAI LLM CUNG PHUONG PHAP -> thien vi len, dung cai bay da lam hong ket
luan router self-consistency (dev +12, nop that -0.002).
Vi vay o day KHONG cham dung/sai theo gold. Ta do model CHON NHAN NAO trong ba kha nang:
  A = nhan pipeline dang chon (hang 1 lexical, tuc dang SAI theo dev set)
  B = nhan o hang 2-3 (dev set coi la dung)
  C = nhan khac / tu che
Ket qua chi co gia tri khi kem SOI TAY. Script in san bang doi chieu de nguoi doc tu kiem.

Chay: python exp_rank23_llm.py            (can Ollama chay o :11434)
      LIMIT=5 python exp_rank23_llm.py    (chay thu vai ca)
"""
import json, os, re, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AGENT_MODEL", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
LIMIT = int(os.environ.get("LIMIT", "0"))
OUT = os.path.join(HERE, "exp_rank23_llm.json")

D = [d for d in json.load(open(os.path.join(HERE, "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]

_C = {}
def load(p):
    if p not in _C:
        txt = open(p, encoding="utf-8", errors="replace").read()
        _C[p] = (txt, P.ingest(txt), list(re.finditer(r"<table>(.*?)</table>", txt, re.S)))
    return _C[p]

def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)

def as_loc(r):
    return {"maso": "", "label": r["label"], "tid": r["tid"], "cur": r["cur"], "prev": r["prev"],
            "ov": 0.0, "hdrs": r.get("hdrs") or [], "vals": r.get("vals") or []}

def raw_table(ms, tid, cap=6000):
    """Text THO cua bang (bo the HTML cho de doc), cat bot neu qua dai."""
    if tid >= len(ms): return ""
    t = ms[tid].group(1)
    t = re.sub(r"</t[dh]>\s*<t[dh]>", " | ", t)
    t = re.sub(r"</?tr>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(l.strip() for l in t.split("\n") if l.strip())
    return t[:cap]

SYS = ("Ban doc bang bao cao tai chinh Viet Nam. Nguoi dung hoi mot chi tieu. "
       "Nhiem vu: chon DUNG MOT dong trong bang khop voi cau hoi. "
       "Tra ve DUY NHAT ten dong do, sao chep NGUYEN VAN tu bang, khong giai thich, khong them chu.")

def ask(question, tables):
    body = json.dumps({
        "model": MODEL, "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Cau hoi: {question}\n\nCac bang:\n{tables}\n\nTen dong:"},
        ],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        msg = json.load(r)["message"]
        return (msg.get("content") or "").strip().split("\n")[0].strip(' "\'*')

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", P.strip_vn(s or "")).strip()

# ---- gom 41 ca: dong dung nam o hang 2-3 cua cands ----
CASES = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr: continue
    txt, (rows, uf, src), ms = load(str(fr[0]))
    tgt = P.target_label(d["question"]); qdir = P.qdir_of(d["question"]); gold = d["llm_val"]
    loc = P.locate(rows, tgt, qdir)
    if loc and close(P.pick_value(loc, qdir), gold): continue          # dang dung -> bo qua
    cands = P.score_cands(rows, tgt, qdir)
    if not cands: continue
    good = [(i, r) for i, (ov, r) in enumerate(cands) if close(P.pick_value(as_loc(r), qdir), gold)]
    if not good or good[0][0] == 0 or good[0][0] + 1 > 3: continue     # chi lay hang 2-3
    CASES.append((d["id"], d["question"], cands[0][1], good[0][1], ms))
print(f"So ca hang 2-3: {len(CASES)}")
if LIMIT: CASES = CASES[:LIMIT]

# TIEP TUC duoc: nap ket qua da co, bo qua id da chay. Lan truoc chay 41 ca roi bi kill va MAT
# SACH vi script chi ghi JSON o CUOI — dung lop loi da ghi trong repo (viec chay dai phai ghi
# tang dan, xem AGENTS.md §7).
recs = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
done = {r["id"] for r in recs if "ket" in r}
if done:
    print(f"Da co {len(done)} ca truoc do -> chay tiep phan con lai")
CASES = [c for c in CASES if c[0] not in done]

tally = {"B_dung_hang23": 0, "A_giu_hang1": 0, "C_khac": 0, "X_trung_nhan": 0, "X_nhap_nhang_tien_to": 0, "loi": 0}
for k, (qid, q, rowA, rowB, ms) in enumerate(CASES, 1):
    tids = list(dict.fromkeys([rowA["tid"], rowB["tid"]]))              # ghep ca hai bang lien quan
    tables = "\n\n---\n\n".join(f"[Bang {i+1}]\n{raw_table(ms, t)}" for i, t in enumerate(tids))
    try:
        pick = ask(q, tables)
    except Exception as e:
        tally["loi"] += 1; recs.append({"id": qid, "err": str(e)[:80]})
        json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1); continue
    nA, nB, nP = norm(rowA["label"]), norm(rowB["label"]), norm(pick)
    # PHAI loai truoc cac ca KHONG THE phan biet bang NHAN, neu khong ket qua bi thoi phong:
    #  - nA == nB: hai dong TRUNG NHAN, khac nhau o BANG => hoi ten dong khong tra loi duoc
    #  - nA la con/cha cua nB (vd "Chi phi hoat dong" vs "VIII. Chi phi hoat dong"): model tra
    #    dung chu cua A van bi tinh cho B neu kiem B truoc bang phep chua hai chieu. Do chinh la
    #    bug o ban dau, thoi phong ty le B tu 45% len 54%.
    if nA == nB: key = "X_trung_nhan"
    elif nA in nB or nB in nA: key = "X_nhap_nhang_tien_to"
    else:
        hitB, hitA = (nP in nB or nB in nP), (nP in nA or nA in nP)
        key = "B_dung_hang23" if (hitB and not hitA) else "A_giu_hang1" if (hitA and not hitB) else "C_khac"
    tally[key] += 1
    recs.append({"id": qid, "q": q[:110], "A_pipeline": rowA["label"], "B_hang23": rowB["label"],
                 "model_chon": pick, "ket": key})
    print(f"  [{k}/{len(CASES)}] id{qid} -> {key}", flush=True)
    json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tally = {"B_dung_hang23": 0, "A_giu_hang1": 0, "C_khac": 0, "X_trung_nhan": 0, "X_nhap_nhang_tien_to": 0, "loi": 0}
for r in recs: tally["loi" if "err" in r else r["ket"]] += 1
# Mau so = so ca PHAN BIET DUOC, khong phai tong so ca.
n = max(1, tally["B_dung_hang23"] + tally["A_giu_hang1"] + tally["C_khac"])
print(f"\n{'ket qua':<22}{'so ca':>7}{'ty le':>9}")
print("-" * 38)
for kk in ("B_dung_hang23", "A_giu_hang1", "C_khac", "X_trung_nhan", "X_nhap_nhang_tien_to", "loi"):
    print(f"{kk:<22}{tally[kk]:>7}{100*tally[kk]/n:>8.0f}%")
print(f"\nChi tiet -> {OUT}  (PHAI soi tay truoc khi tin: gold dev set cung do LLM sinh)")
