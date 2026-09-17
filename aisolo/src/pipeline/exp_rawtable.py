"""KET QUA — DA DONG HUONG 22/08. Hai dinh dang, ca hai AM. Dung thu lai voi model <=4B.

    dinh dang        model dung   KET HOP voi pipeline   bat dong: model thang
    pipeline              -        192/280 = 68,6%              -
    rows (nguyen van)   51,8%      174/280 = 62,1%  (-18)     27,6%
    coord (toa do)      42,5%      156/280 = 55,7%  (-36)     19,8%

Bien the "coord" duoc dung theo dung chi dan slide 8 cua tai lieu mentor — moi o mot dong kem
(RowPath, ColPath, Value) — de go lop loi "lay nham cot nam". No lam TE HON dinh dang nguyen van.
Gia thuyet "loi la do cot xe dich" SAI: sua can le xong thi model doc con kem hon truoc. Co the vi
dang toa do dai gap doi nen it bang lot vao context hon, va vi bang dang dong-cot giong du lieu
huan luyen hon la danh sach toa do lap lai.

=== Ghi chep goc, giu nguyen ===
"""
_GHI_CHEP_GOC = """KET QUA — DA CHAY XONG 22/08. Voi model 4B: AM RO RANG.

Gia thuyet: khau chon dong cua ta lam PHANG bang truoc khi tra (do duoc: chi 5,4% dong bang chinh
con giu header cot, 11,7% con ngu canh dong cha), dung dieu tai lieu mentor canh bao — bang phan cap
phai giu toa do (RowPath, ColPath, Value). Neu dua model BANG GOC chua parse thi no tu doc duoc
cau truc, va co the thang khop tu vung.

Boi canh: hai thi nghiem LLM truoc deu cho model an du lieu DA PARSE (rows do ingest() dung, hoac
ro ung vien loc qua target_label) — ma chinh hai khau do moi la cho hong. Lan nay chi doi DAU VAO:
dua nguyen van noi dung <table> de model tu doc header, cot ky va nhan.

DO DUOC tren 305 cau dev set, Qwen3.5-4B (hop quy che <=14B):
    pipeline hien tai                            208/305 = 68,2%
    model doc bang goc                           158/305 = 51,8%
    ket hop (model khi cam ket, pipeline khi tu choi) 188/305 = 61,6%  -> KEM 20 cau
    ca bat dong dut khoat: model thang 34/118 = 28,8%   [tieu chi chot truoc: >= 60%]
    tran oracle (chon duoc ben dung)             242/305 = 79,3%

Ba dieu khien ket qua nay DANG TIN:
  1. Trong tai dev set la gemma4:31b DOC BANG THO — thien vi dung huong LLM. Vay ma LLM van thua.
  2. Thi nghiem CONG BANG: bang da duoc dua vao (3 bang o 255/256 ca tra loi), khong phai model bi
     bo doi du lieu.
  3. Loi cua model la loi NANG LUC DOC, khong phai loi setup: chon dong TONG thay vi dong cu the
     (GAS: "Tong doanh thu thuan" thay vi "Doanh thu thuan ve ban hang"); lay NHAM COT NAM
     (Bluemarq: 6.117 ty thay vi 7.224 ty); mat SAC THAI ("Chi phi thue TNDN" tong thay vi "hien
     hanh"); doi khi tra rac (mot ca ra so 4).

CANH BAO VE MAU NHO: o 37 cau dau, so lieu cho thay model THANG pipeline khi no chiu tra loi
(77,4% vs 71,0%) — va ket luan do SAI. Tren 305 cau thi nguoc lai (61,7% vs 69,5%). Dung ket luan
tu vai chuc cau.

CON MO: tran oracle 79,3% nghia la model DUNG 34 cau ma pipeline sai — tin hieu bu nhau CO THAT,
nhung khong co cach nao biet tin ben nao o tung cau. Va day moi la 4B; nhom dau bang xep hang dung
14B. Muon tra loi dut diem phai deploy lai endpoint 14B (hien 404).

Chay:  VIFINQA_ROOT=<duong dan> python pipeline/exp_rawtable.py
Env: RAW_N (so cau) - RAW_TOPK (so bang, 3) - RAW_MAXCH (ky tu, 9000) - RAW_CTX (16384)
     RAW_MODEL - RAW_FMT (rows | coord)
"""
import os, re, sys, json, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devset", "devset.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devset",
                   f"rawtable_{os.environ.get('RAW_FMT','rows')}.json")
MODEL = os.environ.get("RAW_MODEL", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
TOPK = int(os.environ.get("RAW_TOPK", "3"))
MAXCH = int(os.environ.get("RAW_MAXCH", "9000"))
CTX = int(os.environ.get("RAW_CTX", "16384"))
N = int(os.environ.get("RAW_N", "0"))          # 0 = hết


def table_texts(txt, tids):
    """Nội dung NGUYÊN VĂN của các bảng theo tid → text dạng dòng, ô ngăn bởi ' | '.

    Chỉ bỏ thẻ HTML, KHÔNG dò cột mã số, KHÔNG ép nhãn/giá trị như ingest(). Giữ nguyên
    header ('Số cuối năm'/'Số đầu năm'), cột thuyết minh và các dòng tiêu đề nhóm — đây
    chính là thông tin mà rows đã parse làm mất."""
    ms = list(re.finditer(r"<table>(.*?)</table>", txt, re.S))
    out = []
    for tid in tids:
        if tid >= len(ms):
            continue
        body = ms[tid].group(1)
        lines = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
            cs = [re.sub(r"<[^>]*>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            cs = [c for c in cs if c]
            if cs:
                lines.append(" | ".join(cs))
        if lines:
            out.append(f"=== BẢNG {tid} ===\n" + "\n".join(lines))
    return out



# ---------- BIEN THE 2: tuyen tinh hoa THEO TOA DO (slide 8 cua tai lieu mentor) ----------
NUM = re.compile(r"^[\d.,()\-]+$")


def _cells(tr):
    return [re.sub(r"<[^>]*>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]


def table_coords(txt, tids):
    """Moi O mot dong, kem TOA DO: [<ten cot> | <duong dan dong>] = <gia tri>.

    Vi sao can bien the nay: table_texts() noi cac o bang " | " VA BO O RONG, nen cot xe dich theo
    tung dong — dong co o "Thuyet minh" thi gia tri nam o vi tri 4, dong khong co thi nam o vi tri
    3. Do tren mot bang can doi that: vut 13/106 o (12%), du de lech phan lon dong, va header mat o
    dau nen lech mot nhip so voi du lieu. Model buoc phai tu suy can le tung dong — dung lop loi
    "lay nham cot nam" da thay khi soi ca hong.

    Cach nay bo han viec can le: moi gia tri di kem TEN COT va DUONG DAN DONG (dong tieu de nhom >
    nhan dong), tuc dung dang (RowPath, ColPath, Value) ma slide 8 chi dinh.
    """
    ms = list(re.finditer(r"<table>(.*?)</table>", txt, re.S))
    out = []
    for tid in tids:
        if tid >= len(ms):
            continue
        rows = [c for c in (_cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>", ms[tid].group(1), re.S)) if any(c)]
        if not rows:
            continue
        w = max(len(r) for r in rows)
        rows = [r + [""] * (w - len(r)) for r in rows]       # GIU o rong -> cot thang hang
        # header = dong dau tien co >=2 o chu (khong phai so dai)
        hdr = next((r for r in rows[:4]
                    if len([c for c in r if c]) >= 2
                    and not any(NUM.match(c) and len(re.sub(r"\D", "", c)) >= 4 for c in r if c)), None)
        lines, ctx = [], ""
        for r in rows:
            if r is hdr:
                continue
            label = next((c for c in r if c), "")
            nums = [(j, c) for j, c in enumerate(r)
                    if c and NUM.match(c) and len(re.sub(r"\D", "", c)) >= 4]
            if not nums:
                if label:
                    ctx = label                               # dong tieu de nhom -> RowPath cha
                continue
            path = f"{ctx} > {label}" if ctx and ctx != label else label
            for j, v in nums:
                col = hdr[j] if (hdr and j < len(hdr) and hdr[j]) else f"cot {j}"
                lines.append(f"[{col} | {path}] = {v}")
        if lines:
            out.append(f"=== BANG {tid} ===\n" + "\n".join(lines))
    return out


FMT = os.environ.get("RAW_FMT", "rows")     # rows = nguyen van (ban cu) | coord = theo toa do
render = table_coords if FMT == "coord" else table_texts

def pick_tables(txt, question, k=TOPK):
    """Chọn bảng bằng BM25 trên CẢ CÂU HỎI (không qua target_label — khâu đang hỏng)."""
    tt = P.table_tokens(txt)
    if not tt:
        return []
    sc = P.bm25_scores(P.toks_seg(question), [t["toks"] for t in tt])
    order = sorted(zip(tt, sc), key=lambda x: -x[1])
    return [t["tid"] for t, s in order[:k]]


SYS = (
    "Bạn đọc bảng trích nguyên văn từ báo cáo tài chính và tìm ĐÚNG con số trả lời câu hỏi.\n"
    "Chú ý sắc thái nghiệp vụ — sai sắc thái là sai hoàn toàn:\n"
    "  'phải nộp' ≠ 'đã nộp' · 'nguyên giá' ≠ 'giá trị còn lại' · 'ngắn hạn' ≠ 'dài hạn'\n"
    "  'giá trị gộp' ≠ 'dự phòng' · dòng TỔNG ≠ dòng chi tiết · số CUỐI kỳ ≠ số ĐẦU kỳ.\n"
    "Chép lại con số ĐÚNG NHƯ IN trong bảng (bỏ dấu chấm phân cách, số âm giữ dấu trừ). "
    "KHÔNG tự quy đổi đơn vị, KHÔNG tự tính toán."
)


def ask(question, tabs):
    prompt = (
        f"{SYS}\n\n" + "\n\n".join(tabs) + f"\n\nCÂU HỎI: {question}\n\n"
        'Trả về JSON DUY NHẤT: {"value": <số nguyên như in trong bảng>, "nhan": "<nhãn dòng đã chọn>"}. '
        'Nếu bảng không có dòng nào trả lời được, trả {"value": null, "nhan": null}. Chỉ trả JSON.'
    )
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "think": False, "stream": False, "format": "json",
                       "options": {"num_ctx": CTX, "temperature": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        c = json.loads(r.read().decode())["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c).strip()
    return json.loads(c)


def as_int(v):
    """Model có thể trả '1.234.567' hoặc '(1.234)' hoặc số thực → số nguyên, hoặc None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d\-]", "", s)
    if not s or s == "-":
        return None
    n = int(s)
    return -n if (neg and n > 0) else n


def eq(a, b):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


def main():
    dev = [d for d in json.load(open(DEV, encoding="utf-8")) if d["llm_val"] is not None]
    if N:
        dev = dev[:N]
    done = {}
    if os.path.exists(OUT):
        done = {d["id"]: d for d in json.load(open(OUT, encoding="utf-8"))}

    cache = {}
    def read(p):
        if p not in cache:
            cache[p] = open(p, encoding="utf-8", errors="replace").read()
        return cache[p]

    t0 = time.time()
    for k, d in enumerate(dev):
        qid = d["id"]
        if qid in done:
            continue
        qt = d["question"]
        fr = P.find_report(d["tk"], d["nam"], d["doctype"])
        if not fr:
            print(f"  id{qid}: khong tim thay report", flush=True)
            continue
        txt = read(str(fr[0]))
        rows, uf, _ = P.ingest(txt)

        # pipeline HIỆN TẠI (không dùng pipe_val đóng băng trong devset.json)
        qdir = P.qdir_of(qt)
        loc = P.locate(rows, P.target_label(qt), qdir)
        pipe_val = P.pick_value(loc, qdir) if loc else None
        ov = loc.get("ov") if loc else None
        maso = bool(loc and loc.get("maso"))

        tids = pick_tables(txt, qt)
        tabs = render(txt, tids)
        blob = "\n\n".join(tabs)
        if len(blob) > MAXCH:                      # cắt bớt bảng cuối cho vừa context
            while tabs and len("\n\n".join(tabs)) > MAXCH:
                tabs.pop()
            if not tabs:
                tabs = [blob[:MAXCH]]
        try:
            a = ask(qt, tabs)
        except Exception as e:
            print(f"  id{qid}: LOI goi model: {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        raw = as_int(a.get("value"))
        qwen_val = raw * uf if raw is not None else None

        gold = d["llm_val"]
        rec = {"id": qid, "question": qt, "gold": gold, "gold_nhan": d["llm_nhan"],
               "pipe_val": pipe_val, "pipe_label": loc["label"] if loc else None,
               "ov": ov, "maso": maso, "qwen_val": qwen_val, "qwen_nhan": a.get("nhan"),
               "uf": uf, "tids": tids, "ntab": len(tabs),
               "pipe_ok": eq(pipe_val, gold), "qwen_ok": eq(qwen_val, gold)}
        done[qid] = rec
        json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  [{k+1}/{len(dev)}] id{qid} qwen={'OK ' if rec['qwen_ok'] else 'sai'} "
              f"pipe={'OK ' if rec['pipe_ok'] else 'sai'} ov={ov} ({int(time.time()-t0)}s)", flush=True)

    report(list(done.values()))


def report(d):
    n = len(d)
    if not n:
        print("khong co ket qua")
        return
    p_ok = sum(1 for x in d if x["pipe_ok"])
    q_ok = sum(1 for x in d if x["qwen_ok"])
    both = sum(1 for x in d if x["pipe_ok"] and x["qwen_ok"])
    only_p = sum(1 for x in d if x["pipe_ok"] and not x["qwen_ok"])
    only_q = sum(1 for x in d if x["qwen_ok"] and not x["pipe_ok"])
    none = sum(1 for x in d if not x["pipe_ok"] and not x["qwen_ok"])
    absta = sum(1 for x in d if x["qwen_val"] is None)
    print(f"\n=== Qwen3.5-4B doc BANG GOC vs pipeline — {n} cau (gold: trong tai gemma4:31b) ===")
    print(f"  pipeline dung : {p_ok}/{n} = {100*p_ok/n:.1f}%")
    print(f"  qwen dung     : {q_ok}/{n} = {100*q_ok/n:.1f}%")
    print(f"  ca hai dung {both} | chi pipe {only_p} | CHI QWEN {only_q} | ca hai sai {none}")
    print(f"  qwen tu choi tra loi (value=null): {absta}")
    print(f"  tran oracle (chon duoc cai dung): {p_ok+only_q}/{n} = {100*(p_ok+only_q)/n:.1f}%")

    def band(x):
        if x["maso"]:
            return "MA_SO"
        o = x["ov"]
        if o is None:
            return "khac"
        return ">=0.80" if o >= 0.8 else "0.60-0.79" if o >= 0.6 else "<0.60"
    print(f"\n  {'tang ov':<12}{'n':>5}{'pipe':>7}{'qwen':>7}{'chi qwen':>10}{'chi pipe':>10}")
    for b in ["MA_SO", ">=0.80", "0.60-0.79", "<0.60", "khac"]:
        g = [x for x in d if band(x) == b]
        if not g:
            continue
        print(f"  {b:<12}{len(g):>5}{sum(1 for x in g if x['pipe_ok']):>7}"
              f"{sum(1 for x in g if x['qwen_ok']):>7}"
              f"{sum(1 for x in g if x['qwen_ok'] and not x['pipe_ok']):>10}"
              f"{sum(1 for x in g if x['pipe_ok'] and not x['qwen_ok']):>10}")

    # CHIEN LUOC GOP — nguong CHOT TRUOC khi nhin so: dung qwen khi ov < 0.6 (va khong phai ma so)
    def merged(thr):
        ok = 0
        for x in d:
            use_q = (not x["maso"]) and (x["ov"] is not None) and x["ov"] < thr and x["qwen_val"] is not None
            ok += x["qwen_ok"] if use_q else x["pipe_ok"]
        return ok
    m = merged(0.6)
    print(f"\n  GOP tai nguong ov<0.60 (chot TRUOC): {m}/{n} = {100*m/n:.1f}%  "
          f"net {m-p_ok:+d} cau  [tieu chi: >= +10]")
    print("  (tham khao, KHONG dung de quyet dinh — quet nguong la exploratory)")
    for t in [0.4, 0.5, 0.7, 0.8, 1.01]:
        mm = merged(t)
        print(f"    ov<{t:<5} -> {mm}/{n} net {mm-p_ok:+d}")


if __name__ == "__main__":
    if os.environ.get("RAW_REPORT_ONLY"):
        report(json.load(open(OUT, encoding="utf-8")))
    else:
        main()
