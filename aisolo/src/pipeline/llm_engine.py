"""Engine LLM-sinh-pandas cho câu hard-conditional (AGG/COND) — nhóm deterministic trả SAI.
LLM (Qwen ≤14B qua Modal endpoint, OpenAI-compatible) sinh MỘT biểu thức pandas trên df1
(= CSV gộp nhiều report); answer = eval(pandas) qua GUARD → KHÔNG để LLM tự khai số (chống bịa).

Config qua env:
  LLM_BASE_URL  (default http://localhost:11434/v1 — Ollama local để test plumbing)
  LLM_MODEL     (default Qwen3.5-4B local; production: Qwen2.5-14B-Instruct trên Modal)
  LLM_API_KEY   (default 'ollama')
Cache đĩa (llm_cache/) → rerun không tốn tiền + tái lập bit-for-bit."""
import os, re, json, csv, hashlib, math, ssl, urllib.request
import pipeline as P

def _ssl_ctx():   # máy user CA bundle hết hạn → dùng certifi; hoặc LLM_INSECURE_SSL=1 (endpoint của chính mình)
    if os.environ.get("LLM_INSECURE_SSL", "0") == "1":
        return ssl._create_unverified_context()
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _load_env():   # nạp pipeline/.env (KEY=VALUE) vào os.environ nếu chưa set — cho tiện đặt creds Modal
    envf = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(envf):
        return
    for line in open(envf, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env()

LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
LLM_KEY = os.environ.get("LLM_API_KEY", "ollama")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "llm_cache")

# Mã số line-item CHÍNH luôn giữ (để LLM tính ratio/so sánh dù label metric không khớp)
MAIN_CODES = {"10", "11", "20", "21", "50", "60", "100", "200", "270", "300", "400", "500", "110", "140"}

# ---------- CSV gộp nhiều report ----------
def build_multi_report_csv(qt, DATA, written):
    """Gộp mọi report liên quan (rel_pairs) → 1 CSV long, lọc theo metric để bound size.
    Trả (fname, meta{companies,years,labels}) hoặc None."""
    import build_submission as B   # lazy: tránh circular import (B import llm_engine ở top)
    dt = P.doctype(qt)
    tt = P.toks(P.clean_metric(P.target_label(qt)))
    rows_out = []; companies = set(); years = set()
    for tk, y in B.rel_pairs(qt):
        fr = P.find_report(tk, y, dt)
        if not fr:
            continue
        rows = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
        note_kept = 0
        for r in rows:
            if r["cur"] is None or not r["label"]:
                continue
            st = r["st"]
            ov = len(tt & P.toks(r["label"])) / max(1, len(tt))
            if st in ("BS", "PL", "CF"):
                pass                       # LUÔN giữ row báo cáo CHÍNH (BS/PL/CF) — xương sống tính toán (CFO, ma_so)
            elif r["ma"] or ov > 0:        # NOTE: chỉ giữ nếu có Mã số HOẶC khớp metric, cap riêng
                if note_kept >= 40:
                    continue
                note_kept += 1
            else:
                continue
            rows_out.append([tk, int(y), st, r["ma"], r["label"], int(round(r["cur"]))])
            companies.add(tk); years.add(str(y))
    if not rows_out:
        return None
    fname = f"multi_{hashlib.md5(qt.encode()).hexdigest()[:10]}.csv"
    if fname not in written:
        with open(os.path.join(DATA, fname), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["company", "year", "statement", "ma_so", "label", "value"])
            w.writerows(rows_out)
        written.add(fname)
    # Label THẬT (r[4]), ưu tiên label khớp metric (giúp LLM lọc đúng dòng NOTE không có ma_so)
    all_lab = list(dict.fromkeys(r[4] for r in rows_out if r[4]))
    rel = [l for l in all_lab if tt & P.toks(l)]
    labels = list(dict.fromkeys(rel + all_lab))
    return fname, {"companies": sorted(companies), "years": sorted(years), "labels": labels}

# ---------- Prompt ----------
def make_prompt(qt, meta, unit):
    labels = " | ".join(meta["labels"][:50])
    return (
        "Bạn là chuyên gia pandas. Có sẵn DataFrame df1 với các cột:\n"
        "  company (mã CK, CHUỖI), year (năm, SỐ NGUYÊN), statement (loại bảng: BS=cân đối, PL=kết quả KD, CF=lưu chuyển tiền, NOTE=thuyết minh),\n"
        "  ma_so (Mã số TT200, SỐ – vd 270.0), label (tên chỉ tiêu, CHUỖI), value (giá trị VND, SỐ NGUYÊN).\n\n"
        f"Công ty: {', '.join(meta['companies'])}\n"
        f"Năm: {', '.join(meta['years'])}\n"
        f"Một số label: {labels}\n\n"
        f"Câu hỏi: {qt}\n"
        f"Đơn vị: value là VND → CHIA cho {unit} (nếu hỏi triệu/tỷ). Hỏi %/tỷ-lệ → nhân 100.\n\n"
        "QUY TẮC:\n"
        "- Lọc ma_so là SỐ: df1[df1['ma_so']==270]  (KHÔNG =='270').\n"
        "- QUAN TRỌNG: cùng ma_so khác nghĩa theo statement (Mã20 = LN gộp ở PL, = tiền thuần HĐKD ở CF). Luôn kèm statement khi cần: df1[(df1['ma_so']==20)&(df1['statement']=='PL')].\n"
        "- Lọc company/label là CHUỖI, .astype(str): df1[df1['company'].astype(str)=='HPG'].\n"
        "- 'cao nhất/thấp nhất/công ty nào': trả GIÁ TRỊ SỐ (không trả tên).\n"
        "- Câu đa-công-ty/đa-năm: DÙNG VÒNG LẶP Python — vd max(float(df1[(df1['company'].astype(str)==c)&...]['value'].sum()) for c in df1['company'].astype(str).unique()). TUYỆT ĐỐI TRÁNH .groupby() trên series đã lọc (lệch index → lỗi).\n\n"
        "MÃ SỐ TT200:\n"
        "  PL: 10 doanh thu thuần, 11 giá vốn, 20 LN gộp, 30 LN từ HĐKD, 50 LN trước thuế, 60 LNST.\n"
        "  BS: 100 TS ngắn hạn, 140 hàng tồn kho, 270 tổng tài sản, 300 nợ phải trả, 310 nợ ngắn hạn, 400 vốn chủ sở hữu.\n"
        "  CF: 20 lưu chuyển tiền thuần từ HĐKD (CFO).\n"
        "CÔNG THỨC phái sinh (statement kèm theo):\n"
        "  Biên LN gộp % = Mã20(PL)/Mã10(PL)*100 ; Biên LN ròng % = Mã60(PL)/Mã10(PL)*100\n"
        "  ROE % = Mã60(PL)/Mã400(BS)*100 ; ROA % = Mã60(PL)/Mã270(BS)*100\n"
        "  D/E = Mã300(BS)/Mã400(BS) ; Tỷ lệ nợ = Mã300(BS)/Mã270(BS)\n"
        "  Thanh toán hiện hành = Mã100(BS)/Mã310(BS) ; Thanh toán nhanh = (Mã100(BS)-Mã140(BS))/Mã310(BS)\n\n"
        "Câu 'công ty CÓ [A] cao nhất/thấp nhất thì [B] là bao nhiêu' — A (chọn) và B (trả lời) thường KHÁC:\n"
        "  chọn công ty rồi tính B: (lambda c: <B theo c>)(max(['MA1','MA2'], key=lambda c: <A theo c>)); 'trung vị' = pd.Series([...]).median().\n\n"
        "HÃY SUY LUẬN NGẮN GỌN trước (2-4 dòng):\n"
        "  1. Câu hỏi hỏi chỉ tiêu/hệ số CHÍNH XÁC nào? Công thức theo ma_so + statement? (nếu là chỉ số lạ như số ngày tồn kho, vòng quay, hệ số dòng tiền — tự suy công thức chuẩn kế toán).\n"
        "  2. Lọc gì (công ty/năm/điều kiện phụ như 'CFO dương','cao hơn trung vị')? Phép gộp gì (max/min/median/so sánh/năm nào)?\n"
        "Rồi KẾT THÚC bằng ĐÚNG một dòng bắt đầu bằng 'PANDAS:' chứa MỘT biểu thức Python trả về MỘT SỐ.\n"
        "  (chỉ dùng df1, pd, lambda, float, int, round, abs, min, max, sum, len, sorted, range, list; ma_so lọc bằng SỐ; label/company .astype(str); KHÔNG gán biến, KHÔNG hardcode số).\n\n"
        "Ví dụ:\n"
        "1. Câu hỏi hỏi biên LN gộp = LN gộp / doanh thu thuần = Mã20(PL)/Mã10(PL).\n"
        "2. Lọc nhóm công ty, năm 2024; gộp: lấy max qua các công ty.\n"
        "PANDAS: float(max(float(df1[(df1['company'].astype(str)==c)&(df1['ma_so']==20)&(df1['statement']=='PL')&(df1['year']==2024)]['value'].sum())/float(df1[(df1['company'].astype(str)==c)&(df1['ma_so']==10)&(df1['statement']=='PL')&(df1['year']==2024)]['value'].sum())*100 for c in df1['company'].astype(str).unique()))\n"
    )

# ---------- Gọi LLM (OpenAI-compatible) + cache ----------
def _post_chat(prompt):
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1200")),
    }
    if os.environ.get("LLM_NO_THINK", "0") == "1":   # =1 CHỈ cho model thinking (Qwen3.x); Qwen2.5-Instruct để 0
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(LLM_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"})
    msg = json.load(urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()))["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or ""   # fallback reasoning nếu content rỗng

def _post_chat_n(prompt, n, temperature):
    """Sinh N mẫu trong 1 request (vLLM param 'n' → prefill chung, rẻ). Trả list[str]."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "n": n,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1000")),
    }
    if os.environ.get("LLM_NO_THINK", "0") == "1":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(LLM_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"})
    choices = json.load(urllib.request.urlopen(req, timeout=300, context=_ssl_ctx()))["choices"]
    return [(c["message"].get("content") or c["message"].get("reasoning") or "") for c in choices]

def _post_chat_conv(messages):
    """1 completion từ hội thoại nhiều lượt (cho self-debug)."""
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2,
               "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1000"))}
    if os.environ.get("LLM_NO_THINK", "0") == "1":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(LLM_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"})
    msg = json.load(urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()))["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning") or ""

def _balance_expr(s):
    """Cắt MỘT biểu thức Python hoàn chỉnh từ đầu chuỗi bằng cân bằng ()[]{}, bỏ prose phía sau.
    Model hay xuất pandas rồi thêm 'Lưu ý:...' → chỉ giữ tới khi ngoặc cân bằng (hết biểu thức)."""
    starts = [s.find(t) for t in ("float(", "max(", "min(", "sum(", "sorted(", "round(", "abs(", "len(", "df1")]
    starts = [p for p in starts if p >= 0]
    if not starts:
        return s.split("\n")[0].strip()
    s = s[min(starts):]
    depth = 0; opened = False; out = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            depth += 1; opened = True
        elif ch in ")]}":
            depth -= 1
        if opened and depth == 0 and ch == "\n":   # ngoặc đã cân bằng + xuống dòng → hết biểu thức
            break
        if ch == "\n" and not opened:
            break
        out.append(ch)
        if opened and depth == 0 and ch in ")]}":
            # đã đóng hết; nếu ký tự kế không phải toán tử/tiếp nối thì dừng
            pass
    expr = "".join(out).strip()
    # cắt prose đuôi: giữ tới ký tự hợp lệ cuối của biểu thức (đóng ngoặc/số/chữ/dấu chấm)
    expr = re.sub(r"\s+(Lưu ý|Ghi chú|Note|Giải thích|Trong đó|Ở đây)[\s\S]*$", "", expr, flags=re.I)
    return expr.strip()

def _extract_expr(text):
    """Lấy biểu thức pandas (MỘT expression) từ output LLM, bỏ prose/giải thích phía sau."""
    s = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    if "PANDAS:" in s:                    # reason-then-code: lấy phần sau marker cuối
        s = s.rsplit("PANDAS:", 1)[1].strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()                          # có fence: lấy block bên trong
    s = re.sub(r"^\s*(?:result|ans|answer|res)\s*=\s*", "", s.strip())   # bỏ 'result =' ở đầu
    return _balance_expr(s.strip())

def gen_pandas(prompt):
    """Sinh pandas expr (có cache đĩa theo model+prompt)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cf = os.path.join(CACHE_DIR, hashlib.md5((LLM_MODEL + "|" + prompt).encode()).hexdigest() + ".txt")
    if os.path.exists(cf):
        return open(cf, encoding="utf-8").read()
    try:
        expr = _extract_expr(_post_chat(prompt))
    except Exception:
        expr = ""
    with open(cf, "w", encoding="utf-8") as f:
        f.write(expr)
    return expr

def gen_pandas_voted(prompt, evidence, OUT):
    """SELF-CONSISTENCY: sinh N pandas (temperature>0) → eval từng cái → VOTE theo giá trị answer.
    Trả pandas THẮNG (chương trình tính thật từ CSV, KHÔNG hardcode → hợp lệ private). Cache đĩa."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    n = int(os.environ.get("LLM_N_SAMPLES", "12"))
    cf = os.path.join(CACHE_DIR, hashlib.md5((LLM_MODEL + "|voted|" + str(n) + "|" + prompt).encode()).hexdigest() + ".txt")
    if os.path.exists(cf):
        return open(cf, encoding="utf-8").read()
    from collections import Counter
    cands = []   # (pandas, giá trị eval)
    try:
        for raw in _post_chat_n(prompt, n, float(os.environ.get("LLM_TEMP", "0.7"))):
            pq = _extract_expr(raw)
            res = eval_pandas(pq, evidence, OUT)
            if res is not None:
                cands.append((pq, round(res, 4)))
    except Exception:
        cands = []
    if cands:
        best_val = Counter(r for _, r in cands).most_common(1)[0][0]   # vote giá trị phổ biến nhất
        pq = next(p for p, r in cands if r == best_val)                # chương trình sinh ra giá trị đó
    else:
        pq = ""   # tất cả N mẫu fail → fallback deterministic
    with open(cf, "w", encoding="utf-8") as f:
        f.write(pq)
    return pq

# ---------- Guard: chạy thử pandas (giống BTC re-run) ----------
def _eval_ns(pd):
    """Namespace hạn chế: pd + builtins mà model hay dùng (khớp môi trường BTC re-run eval(pandas)
    với builtins chuẩn).

    ĐỪNG GỌI ĐÂY LÀ SANDBOX. Bỏ `__import__`/`open`/`exec` khỏi namespace **không** chặn được truy
    cập hệ thống: chỉ cần `pd` được phép là cả cây phụ thuộc đi theo, và còn đường vòng qua thuộc
    tính nội bộ — `pandas → numpy → numpy.ma.core → inspect → sys → sys.modules → os`. Đây là giới
    hạn của mọi lớp cách ly Ở TẦNG NGÔN NGỮ, không phải thiếu sót cấu hình.

    Ranh giới an toàn thật phải ở tầng hệ điều hành (container/microVM). Ở phạm vi hiện tại thì
    KHÔNG cần: biểu thức chỉ do model của chính ta sinh, chạy ngoại tuyến trên máy ta, không phục
    vụ người dùng bên ngoài. `_reject` bên dưới là lớp lọc rẻ cho đường vòng đã biết, không phải
    ranh giới."""
    return {"pd": pd, "float": float, "int": int, "round": round, "abs": abs, "str": str,
            "min": min, "max": max, "sum": sum, "len": len, "sorted": sorted, "range": range,
            "list": list, "set": set, "dict": dict, "tuple": tuple, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "bool": bool}

# Nguyên thuỷ dùng để thoát khỏi namespace hạn chế. Đo trên 1009 truy vấn thật của bài nộp: KHÔNG
# truy vấn nào chứa bất kỳ mục nào dưới đây, nên cổng chặn này có 0 dương tính giả — chặn được thì
# chặn, không mất gì. Nó KHÔNG thay được cách ly tầng hệ điều hành, chỉ đóng đường vòng đã biết.
_REJECT = re.compile(r"__|import|open|exec|eval|compile"
                     r"|getattr|setattr|delattr|globals|locals|vars|subclasses|mro")


def eval_pandas(pq, evidence, OUT):
    """eval pandas trên CSV thật; trả số hữu hạn | None. answer = giá trị này → execution=answer."""
    import pandas as pd
    if not pq:
        return None
    if _REJECT.search(pq):
        print(f"[CHAN] tu choi chay bieu thuc chua nguyen thuy thoat hiem: {pq[:70]}")
        return None
    ns = _eval_ns(pd)
    try:
        for ev in evidence:
            ns[ev["variable"]] = pd.read_csv(os.path.join(OUT, ev["csv_path"]))
        ns["__builtins__"] = {}
        res = eval(pq, ns)   # ns là GLOBALS → generator/comprehension thấy được float/df1 (scope quirk)
        res = float(res)
        if math.isnan(res) or math.isinf(res) or abs(res) > 1e14:   # >1e14 = blowup scaling (đáp án AGG/COND là %/lần/tỷ → nhỏ)
            return None
        return round(res, 6)
    except Exception:
        return None

def eval_pandas_err(pq, evidence, OUT):
    """Như eval_pandas nhưng trả (result|None, error_str) — cho self-debug."""
    import pandas as pd
    if not pq:
        return None, "rỗng"
    ns = _eval_ns(pd)
    try:
        for ev in evidence:
            ns[ev["variable"]] = pd.read_csv(os.path.join(OUT, ev["csv_path"]))
        ns["__builtins__"] = {}
        res = float(eval(pq, ns))   # ns là GLOBALS → generator/comprehension thấy được float/df1
        if math.isnan(res) or math.isinf(res):
            return None, "kết quả NaN/inf"
        return round(res, 6), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"

def gen_pandas_debug(prompt, evidence, OUT, rounds=2):
    """PoT + SELF-DEBUG: sinh pandas → chạy → lỗi thì feed lỗi lại cho model sửa (≤rounds vòng). Cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cf = os.path.join(CACHE_DIR, hashlib.md5((LLM_MODEL + "|debug|" + prompt).encode()).hexdigest() + ".txt")
    if os.path.exists(cf):
        return open(cf, encoding="utf-8").read()
    pq = ""; res = None
    try:
        raw = _post_chat(prompt)
        pq = _extract_expr(raw)
        res, err = eval_pandas_err(pq, evidence, OUT)
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}]
        for _ in range(rounds):
            if res is not None:
                break
            conv.append({"role": "user", "content":
                f"Biểu thức pandas trên bị lỗi: {err}. Sửa cho CHẠY ĐƯỢC. "
                "Nhớ: TRÁNH .groupby() trên series đã lọc → dùng vòng lặp Python qua công ty với .sum(). "
                "Kết thúc bằng đúng một dòng 'PANDAS: <biểu thức>'."})
            raw = _post_chat_conv(conv)
            conv.append({"role": "assistant", "content": raw})
            pq = _extract_expr(raw)
            res, err = eval_pandas_err(pq, evidence, OUT)
    except Exception:
        res = None
    pq = pq if res is not None else ""
    with open(cf, "w", encoding="utf-8") as f:
        f.write(pq)
    return pq
