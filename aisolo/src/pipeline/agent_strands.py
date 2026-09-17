"""Agentic decompose bằng STRANDS-py (local Qwen3.5-4B, Ollama, THINKING OFF).
Orchestration lo phần chia-bước → model chỉ quyết tool-call đơn giản → không cần think → nhanh hơn.
Chạy: python agent_strands.py   (env AGENT_IDS=371,392,...)
"""
import os, re, json
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P
import build_submission as B
from strands import Agent, tool
from strands.models.ollama import OllamaModel

MODEL_ID = os.environ.get("AGENT_MODEL", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
_SCOPE = {}   # (company, year) -> rows ; set per câu hỏi trước khi gọi agent

class _ToolBudgetExceeded(BaseException):
    """Model kẹt vòng lặp gọi tool liên tục không dừng (đo được: 31 lần search_label không hội tụ,
    id515 — vẫn 37 lần dù đã raise Exception, vì strands/tools/executors/_executor.py bắt
    'except Exception' và biến thành lỗi trả lại cho model thay vì crash — model bỏ qua lỗi, gọi
    tiếp). Cảnh báo nhúng trong JSON KHÔNG đủ.
    Kế thừa BaseException (KHÔNG phải Exception) để XUYÊN qua mọi 'except Exception' trong
    strands (đã kiểm: tools/executors/_executor.py, models/ollama.py, event_loop/event_loop.py —
    không nơi nào bắt BaseException) → thoát thẳng agent() bất kể model có 'nghe lời' hay không."""

_TOOL_CALL_COUNT = 0
TOOL_BUDGET = 20   # tổng số lần gọi TOOL (mọi loại) cho phép trong 1 câu hỏi

def _tick_tool_budget():
    global _TOOL_CALL_COUNT
    _TOOL_CALL_COUNT += 1
    if _TOOL_CALL_COUNT > TOOL_BUDGET:
        raise _ToolBudgetExceeded(f"vượt {TOOL_BUDGET} lần gọi tool trong 1 câu — dừng, coi như fail")

# ---------- Truy xuất giá trị grounded ----------
def _get_raw(company, year, statement, ma_so):
    rows = _SCOPE.get((str(company), int(year)))
    if rows is None:
        rows = _load2(str(company), int(year))   # on-demand (vd năm t-1 cho chỉ tiêu bình quân) — không có trong _SCOPE
    if not rows:
        return None
    ma = str(ma_so).strip()
    if len(ma) >= 3:   # mã 3 chữ số = BS aggregate; st-tagging không đáng tin cho BS → match bỏ qua st
        cands = [r for r in rows if str(r["ma"]) == ma and r["cur"] is not None]
        return max(cands, key=lambda r: abs(r["cur"]))["cur"] if cands else None
    for r in rows:   # mã 2 chữ số PL/CF → dùng statement disambiguate (ma20 = LN gộp PL / CFO CF)
        if str(r["ma"]) == ma and r["st"] == statement and r["cur"] is not None:
            return r["cur"]
    return None

@tool
def get_value(company: str, year: int, statement: str, ma_so: str) -> str:
    """Lấy giá trị (VND) của một chỉ tiêu theo Mã số TT200 trong 1 công ty-năm-báo cáo.
    statement: 'BS' (cân đối), 'PL' (kết quả KD), 'CF' (lưu chuyển tiền). Trả 'null' nếu không có dòng đó.
    """
    _tick_tool_budget()
    v = _get_raw(company, year, statement, ma_so)
    return json.dumps(v) if v is not None else "null"

@tool
def get_values(companies: str, year: int, statement: str, ma_so: str) -> str:
    """Lấy CÙNG một chỉ tiêu (ma_so, statement, year) cho NHIỀU công ty trong 1 lần — DÙNG khi cần so sánh
    nhóm công ty (tiết kiệm bước, tránh lỗi). companies = mã CK cách nhau dấu phẩy, vd 'ASM,DBC,MPC'.
    Trả JSON {mã CK: giá trị VND (hoặc null)}."""
    _tick_tool_budget()
    out = {}
    for c in companies.replace(" ", "").split(","):
        if c:
            out[c] = _get_raw(c, year, statement, ma_so)
    return json.dumps(out, ensure_ascii=False)

_SEARCH_COUNT = 0   # đếm số lần gọi search_label TRONG 1 câu hỏi — reset ở load_scope() (đầu mỗi câu)

@tool
def search_label(company: str, year: int, keyword: str) -> str:
    """KHÔNG BIẾT Mã số của một chỉ tiêu (đặc biệt ngân hàng/bảo hiểm — dùng hệ Mã số KHÁC công ty
    thường, đừng đoán) → tra bằng TỪ KHOÁ tiếng Việt, đọc thẳng nhãn thật trong báo cáo thay vì đoán mù.
    Trả JSON {"candidates":[{ma_so,statement,label,value},...]} top-6 dòng khớp gần nhất, sắp theo độ
    khớp giảm dần. Chỉ trả dòng CÓ Mã số (dòng không có Mã số không dùng được cho get_value/EXPR).
    Đọc kỹ 'label' để chọn đúng dòng khớp Ý NGHĨA câu hỏi — nhiều dòng gần giống nhau (vd tổng vs.
    thành phần con) — CHỌN NGAY rồi gọi get_value với đúng (statement, ma_so) của dòng đã chọn.
    ĐỪNG gọi lại nhiều lần cho CÙNG một chỉ tiêu — nếu candidates đã có ứng viên hợp lý, dùng luôn."""
    _tick_tool_budget()
    global _SEARCH_COUNT
    _SEARCH_COUNT += 1
    rows = _load2(str(company), int(year))
    out = []
    if rows:
        seen = set()
        for ov, r in P.score_cands(rows, keyword):
            if not r["ma"]:
                continue
            key = (r["ma"], r["st"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"ma_so": r["ma"], "statement": r["st"], "label": r["label"],
                        "value": r["cur"]})
            if len(out) >= 6:
                break
    result = {"candidates": out}
    if _SEARCH_COUNT >= 4:   # chặn vòng lặp tìm-mãi-không-chốt — ép quyết định thay vì tìm thêm
        result["CANH_BAO"] = ("Đã tra cứu search_label nhiều lần trong câu này. DỪNG tìm thêm — chọn "
            "ứng viên khớp nhất trong 'candidates' hiện tại (hoặc kết quả các lần tra trước) và gọi "
            "get_value NGAY. Nếu không có ứng viên nào hợp lý, kết luận thiếu dữ liệu và trả 'EXPR: None'.")
    return json.dumps(result, ensure_ascii=False)

# ---------- ①: CODE lo compositional logic (median/filter/argmax) — model chỉ gọi tên metric ----------
_DT = "consolidated"   # doctype hiện tại (set per câu)
_RC = {}
def _load2(tk, y):
    k = (str(tk), int(y), _DT)
    if k not in _RC:
        fr = P.find_report(tk, str(y), _DT)
        _RC[k] = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0] if fr else None
    return _RC[k]
def _v2(tk, y, st, ma):
    rows = _load2(tk, y)
    if rows is None: return None
    ma = str(ma)
    if len(ma) >= 3:
        c = [r["cur"] for r in rows if str(r["ma"]) == ma and r["cur"] is not None]; return max(c, key=abs) if c else None
    return next((r["cur"] for r in rows if str(r["ma"]) == ma and r["st"] == st and r["cur"] is not None), None)
def _DD(a, b): return a/b if (a is not None and b) else None
def _avg2(t, y, st, ma):
    """Số dư bình quân đầu-cuối kỳ = 0.5·(kỳ trước) + 0.5·(kỳ này) — convention BTC cho ROA/ROE/accrual."""
    cur, prev = _v2(t, y, st, ma), _v2(t, y-1, st, ma)
    return 0.5*prev + 0.5*cur if (cur is not None and prev is not None) else None
def _qk(t, y):
    a,h,n=_v2(t,y,'BS',100),_v2(t,y,'BS',140),_v2(t,y,'BS',310); return (a-h)/n if (a is not None and h is not None and n) else None
def _ic(t, y):
    m50,m23=_v2(t,y,'PL',50),_v2(t,y,'PL',23); return (m50+m23)/m23 if (m50 is not None and m23) else None
def _accrual(t, y):
    l,c,avg=_v2(t,y,'PL',60),_v2(t,y,'CF',20),_avg2(t,y,'BS',270); return (l-c)/avg if (l is not None and c is not None and avg) else None
def _gnd(t, y):
    g,n,r=_v2(t,y,'PL',20),_v2(t,y,'PL',60),_v2(t,y,'PL',10); return (g-n)/r if (g is not None and n is not None and r) else None
def _sga(t, y):
    s,a,r=_v2(t,y,'PL',25),_v2(t,y,'PL',26),_v2(t,y,'PL',10); return ((s or 0)+(a or 0))/r if (r and (s is not None or a is not None)) else None
def _invd(t, y):
    hp,hc,cg=_v2(t,y-1,'BS',140),_v2(t,y,'BS',140),_v2(t,y,'PL',11); return (0.5*hp+0.5*hc)/abs(cg)*365 if (hp is not None and hc is not None and cg) else None
METRICS = {
 # ---- biên lợi nhuận (percentage → EXPR phải ×100) ----
 "bien_gop": lambda t,y: _DD(_v2(t,y,'PL',20),_v2(t,y,'PL',10)),        # gross_margin
 "bien_rong": lambda t,y: _DD(_v2(t,y,'PL',60),_v2(t,y,'PL',10)),       # net_margin
 "npm": lambda t,y: _DD(_v2(t,y,'PL',60),_v2(t,y,'PL',10)),
 "bien_hdkd": lambda t,y: _DD(_v2(t,y,'PL',30),_v2(t,y,'PL',10)),       # operating_margin
 "cfo_margin": lambda t,y: _DD(_v2(t,y,'CF',20),_v2(t,y,'PL',10)),
 "sga_intensity": _sga,                                                 # (Mã25+Mã26)/Mã10
 "gross_net_diff": _gnd,                                                # PRO_04 target: (biên gộp − biên ròng), điểm %
 # ---- sinh lời BÌNH QUÂN (percentage) — mẫu số là số dư bình quân, KHÔNG phải cuối kỳ ----
 "roe": lambda t,y: _DD(_v2(t,y,'PL',60), _avg2(t,y,'BS',400)),         # ROE = LNST / VCSH bình quân
 "roa": lambda t,y: _DD(_v2(t,y,'PL',60), _avg2(t,y,'BS',270)),         # ROA = LNST / TS bình quân
 "roe_end": lambda t,y: _DD(_v2(t,y,'PL',60),_v2(t,y,'BS',400)),        # npat_to_ending_equity (khi hỏi 'cuối kỳ')
 "roa_end": lambda t,y: _DD(_v2(t,y,'PL',60),_v2(t,y,'BS',270)),        # npat_to_ending_assets
 "accrual": _accrual,                                                   # EQ_01 target: (LNST − CFO)/TS bình quân, %
 # ---- thanh khoản / đòn bẩy (number — GIỮ tỉ số, KHÔNG ×100) ----
 "quick": _qk,                                                          # (Mã100−Mã140)/Mã310
 "current": lambda t,y: _DD(_v2(t,y,'BS',100),_v2(t,y,'BS',310)),
 "de": lambda t,y: _DD(_v2(t,y,'BS',300),_v2(t,y,'BS',400)),           # Nợ/VCSH
 "debt_asset": lambda t,y: _DD(_v2(t,y,'BS',300),_v2(t,y,'BS',270)),   # percentage
 "icover": _ic,                                                        # (Mã50+Mã23)/Mã23
 "cfo_debt": lambda t,y: _DD(_v2(t,y,'CF',20),_v2(t,y,'BS',310)),      # WCA_10 target: CFO/nợ ngắn hạn
 "htk_nnh": lambda t,y: _DD(_v2(t,y,'BS',140),_v2(t,y,'BS',310)),      # LIQ_02 target: HTK/nợ ngắn hạn
 "equity_mult": lambda t,y: _DD(_avg2(t,y,'BS',270),_avg2(t,y,'BS',400)),
 "asset_turn": lambda t,y: _DD(_v2(t,y,'PL',10),_avg2(t,y,'BS',270)),  # vòng quay TS theo TS bình quân
 "inv_days": _invd,                                                    # số ngày tồn kho
 # ---- cơ cấu / dòng tiền / tăng trưởng ----
 "tsdh_ratio": lambda t,y: _DD(_v2(t,y,'BS',200),_v2(t,y,'BS',270)),   # long_term_assets_share (%)
 "htk_ratio": lambda t,y: _DD(_v2(t,y,'BS',140),_v2(t,y,'BS',270)),    # inventory_to_assets (%)
 "cfo_lnst": lambda t,y: _DD(_v2(t,y,'CF',20),_v2(t,y,'PL',60)),
 "cfo_lnthuan": lambda t,y: _DD(_v2(t,y,'CF',20),_v2(t,y,'PL',30)),
 "growth_rev": lambda t,y: _DD((_v2(t,y,'PL',10) or 0)-(_v2(t,y-1,'PL',10) or 0), _v2(t,y-1,'PL',10)),  # tương đối
 "cfo": lambda t,y: _v2(t,y,'CF',20),
 "lnst": lambda t,y: _v2(t,y,'PL',60),
 "rev": lambda t,y: _v2(t,y,'PL',10),
 "wcap": lambda t,y: (lambda a,n: (a-n) if (a is not None and n is not None) else None)(_v2(t,y,'BS',100),_v2(t,y,'BS',310)),  # vốn lưu động ròng
 "cfo_lnst_rev": lambda t,y: (lambda cf,l,r: (cf-l)/r if (cf is not None and l is not None and r) else None)(_v2(t,y,'CF',20),_v2(t,y,'PL',60),_v2(t,y,'PL',10)),  # (CFO−LNST)/DTT
}

# ---------- Executor: mỗi metric → biểu thức V() (để eval ra answer + sinh pandas/refs cho submission) ----------
# MIRROR chính xác METRICS ở trên. Metric cần năm t-1 (bình quân/growth) tham chiếu {y1}=y-1.
METRIC_EXPR = {
 "bien_gop": lambda c,y: f"V('{c}',{y},'PL','20')/V('{c}',{y},'PL','10')",
 "bien_rong": lambda c,y: f"V('{c}',{y},'PL','60')/V('{c}',{y},'PL','10')",
 "npm": lambda c,y: f"V('{c}',{y},'PL','60')/V('{c}',{y},'PL','10')",
 "bien_hdkd": lambda c,y: f"V('{c}',{y},'PL','30')/V('{c}',{y},'PL','10')",
 "cfo_margin": lambda c,y: f"V('{c}',{y},'CF','20')/V('{c}',{y},'PL','10')",
 "sga_intensity": lambda c,y: f"(V('{c}',{y},'PL','25')+V('{c}',{y},'PL','26'))/V('{c}',{y},'PL','10')",
 "gross_net_diff": lambda c,y: f"(V('{c}',{y},'PL','20')-V('{c}',{y},'PL','60'))/V('{c}',{y},'PL','10')",
 "roe": lambda c,y: f"V('{c}',{y},'PL','60')/(0.5*V('{c}',{y-1},'BS','400')+0.5*V('{c}',{y},'BS','400'))",
 "roa": lambda c,y: f"V('{c}',{y},'PL','60')/(0.5*V('{c}',{y-1},'BS','270')+0.5*V('{c}',{y},'BS','270'))",
 "roe_end": lambda c,y: f"V('{c}',{y},'PL','60')/V('{c}',{y},'BS','400')",
 "roa_end": lambda c,y: f"V('{c}',{y},'PL','60')/V('{c}',{y},'BS','270')",
 "accrual": lambda c,y: f"(V('{c}',{y},'PL','60')-V('{c}',{y},'CF','20'))/(0.5*V('{c}',{y-1},'BS','270')+0.5*V('{c}',{y},'BS','270'))",
 "quick": lambda c,y: f"(V('{c}',{y},'BS','100')-V('{c}',{y},'BS','140'))/V('{c}',{y},'BS','310')",
 "current": lambda c,y: f"V('{c}',{y},'BS','100')/V('{c}',{y},'BS','310')",
 "de": lambda c,y: f"V('{c}',{y},'BS','300')/V('{c}',{y},'BS','400')",
 "debt_asset": lambda c,y: f"V('{c}',{y},'BS','300')/V('{c}',{y},'BS','270')",
 "icover": lambda c,y: f"(V('{c}',{y},'PL','50')+V('{c}',{y},'PL','23'))/V('{c}',{y},'PL','23')",
 "cfo_debt": lambda c,y: f"V('{c}',{y},'CF','20')/V('{c}',{y},'BS','310')",
 "htk_nnh": lambda c,y: f"V('{c}',{y},'BS','140')/V('{c}',{y},'BS','310')",
 "equity_mult": lambda c,y: f"(0.5*V('{c}',{y-1},'BS','270')+0.5*V('{c}',{y},'BS','270'))/(0.5*V('{c}',{y-1},'BS','400')+0.5*V('{c}',{y},'BS','400'))",
 "asset_turn": lambda c,y: f"V('{c}',{y},'PL','10')/(0.5*V('{c}',{y-1},'BS','270')+0.5*V('{c}',{y},'BS','270'))",
 "inv_days": lambda c,y: f"(0.5*V('{c}',{y-1},'BS','140')+0.5*V('{c}',{y},'BS','140'))/abs(V('{c}',{y},'PL','11'))*365",
 "tsdh_ratio": lambda c,y: f"V('{c}',{y},'BS','200')/V('{c}',{y},'BS','270')",
 "htk_ratio": lambda c,y: f"V('{c}',{y},'BS','140')/V('{c}',{y},'BS','270')",
 "cfo_lnst": lambda c,y: f"V('{c}',{y},'CF','20')/V('{c}',{y},'PL','60')",
 "cfo_lnthuan": lambda c,y: f"V('{c}',{y},'CF','20')/V('{c}',{y},'PL','30')",
 "growth_rev": lambda c,y: f"(V('{c}',{y},'PL','10')-V('{c}',{y-1},'PL','10'))/V('{c}',{y-1},'PL','10')",
 "cfo": lambda c,y: f"V('{c}',{y},'CF','20')",
 "lnst": lambda c,y: f"V('{c}',{y},'PL','60')",
 "rev": lambda c,y: f"V('{c}',{y},'PL','10')",
 "wcap": lambda c,y: f"(V('{c}',{y},'BS','100')-V('{c}',{y},'BS','310'))",
 "cfo_lnst_rev": lambda c,y: f"(V('{c}',{y},'CF','20')-V('{c}',{y},'PL','60'))/V('{c}',{y},'PL','10')",
}
# value_kind BTC: percent/point → ×100; ratio(number) → giữ; money → chia theo đơn vị câu hỏi
_PCT_METRICS = {"bien_gop","bien_rong","npm","bien_hdkd","cfo_margin","sga_intensity","roe","roa",
                "roe_end","roa_end","accrual","debt_asset","tsdh_ratio","htk_ratio","growth_rev","gross_net_diff","cfo_lnst_rev"}
_MONEY_METRICS = {"cfo","lnst","rev","wcap"}
_UNIT_DIV = {"trieu": "1e6", "ty": "1e9", "nghinty": "1e12"}

def _scaled_expr(metric, expr, unit):
    """Bọc biểu thức theo value_kind của metric (khớp compiler.py gold): % ×100, số lần giữ, tiền /đơn vị."""
    if metric in _PCT_METRICS:
        return f"({expr})*100"
    if metric in _MONEY_METRICS:
        div = _UNIT_DIV.get(unit)
        return f"({expr})/{div}" if div else expr
    return expr

# ---------- Executor deterministic (tổng quát hoá verify_gt.gt_maxb/gt_temporal) ----------
def _apply_filter(cs, y, fmetric, fop):
    """Lọc companies theo (fmetric, fop). fop: '>median','<median','>0','<0','>1.5','<1'..."""
    import statistics, re as _re
    ff = METRICS.get(fmetric)
    if not ff or not fop:
        return cs
    vals = {c: ff(c, y) for c in cs}
    oks = [v for v in vals.values() if v is not None]
    if not oks:
        return []
    thr = statistics.median(oks) if "median" in fop else float(_re.sub(r"[^0-9.\-]", "", fop) or 0)
    gt = ">" in fop
    return [c for c in cs if vals[c] is not None and (vals[c] > thr if gt else vals[c] < thr)]

def _thr_of(cs, y, fmetric, fop):
    """(ngưỡng, có phải '>' không) của một điều kiện lọc. None nếu không dựng được."""
    import statistics, re as _re
    ff = METRICS.get(fmetric)
    if not ff or not fop:
        return None, None
    if "median" in fop:
        oks = [v for v in (ff(c, y) for c in cs) if v is not None]
        if not oks:
            return None, None
        thr = statistics.median(oks)
    else:
        m = _re.sub(r"[^0-9.\-]", "", fop)
        if m in ("", "-", "."):
            return None, None
        thr = float(m)
    return thr, (">" in fop)

def _argsel(cs, y, smetric, direction):
    sf = METRICS.get(smetric)
    if not sf:
        return None
    sv = {c: sf(c, y) for c in cs}; sv = {c: v for c, v in sv.items() if v is not None}
    if not sv:
        return None
    return max(sv, key=sv.get) if direction == "max" else min(sv, key=sv.get)

def _argsel_year(c, yrs, metric, direction):
    f = METRICS.get(metric)
    if not f:
        return None
    vals = {yy: f(c, yy) for yy in yrs}; vals = {yy: v for yy, v in vals.items() if v is not None}
    if not vals:
        return None
    return max(vals, key=vals.get) if direction == "max" else min(vals, key=vals.get)

def exec_plan(plan):
    """Thực thi PLAN deterministic → {answer, pandas, refs, expr}. Trả None nếu không giải được (→ fallback)."""
    try:
        arch = plan.get("archetype")
        cs = [str(c).strip().upper() for c in (plan.get("companies") or []) if str(c).strip()]
        ys = [int(x) for x in (plan.get("years") or [])]
        tm = plan.get("target_metric")
        if not cs:
            return None
        if arch == "group_count":
            # ĐẾM công ty thoả điều kiện. Trước đây agent tự đếm trong đầu rồi xuất HẰNG SỐ, không có
            # dấu vết bằng chứng nên build_submission từ chối (đúng). Ở đây phép đếm được viết thành
            # TỔNG các biểu thức so sánh trên chính ô dữ liệu → có refs thật, pandas chấm lại được.
            # Chỉ dùng phép so sánh + nhân + cộng, KHÔNG dùng hàm dựng sẵn nào ngoài float().
            if not ys:
                return None
            y = ys[0]
            conds = [(fm, fo) for fm, fo in ((plan.get("filter_metric"), plan.get("filter_op")),
                                             (plan.get("filter2_metric"), plan.get("filter2_op")))
                     if fm in METRIC_EXPR and fo]
            if not conds:
                return None
            terms = []
            for c in cs:
                # công ty thiếu dữ liệu cho BẤT KỲ điều kiện nào → không đưa vào phép đếm
                if any(METRICS[fm](c, y) is None for fm, _ in conds):
                    continue
                parts = []
                for fm, fo in conds:
                    thr, gt = _thr_of(cs, y, fm, fo)
                    if thr is None:
                        return None
                    parts.append(f"(({METRIC_EXPR[fm](c, y)}) {'>' if gt else '<'} {thr})")
                terms.append("*".join(parts) if len(parts) > 1 else parts[0])
            if not terms:
                return None
            expr = "(" + "+".join(terms) + ")"
            ans, refs = eval_expr(expr)
            return {"answer": ans, "pandas": expr_to_pandas(expr), "refs": refs, "expr": expr,
                    "winner": ",".join(cs), "year": y}
        if tm not in METRIC_EXPR:
            return None
        if arch == "lookup":
            if not ys:
                return None
            winner, year = cs[0], ys[0]
        elif arch == "group_maxb":
            if not ys:
                return None
            y = ys[0]
            pool = (_apply_filter(cs, y, plan.get("filter_metric"), plan.get("filter_op"))
                    if plan.get("filter_metric") and plan.get("filter_op") else cs)
            if not pool:
                return None
            winner = _argsel(pool, y, plan.get("select_metric"), plan.get("direction", "max"))
            year = y
        elif arch == "temporal":
            c = cs[0]; yrs = sorted(ys)
            if not yrs:
                return None
            sm = plan.get("select_metric")
            if plan.get("first_neg"):
                sf = METRICS.get(sm)
                base = next((yy for yy in yrs if sf and (sf(c, yy) or 0) < 0), None)
            else:
                base = _argsel_year(c, yrs, sm, plan.get("direction", "max"))
            if base is None:
                return None
            winner = c; year = base + int(plan.get("target_offset_year", 0) or 0)
        else:
            return None
        if winner is None:
            return None
        expr = _scaled_expr(tm, METRIC_EXPR[tm](winner, year), plan.get("unit"))
        ans, refs = eval_expr(expr)
        pq = expr_to_pandas(expr)
        return {"answer": ans, "pandas": pq, "refs": refs, "expr": expr, "winner": winner, "year": year}
    except Exception:
        return None

@tool
def compute_metric(companies: str, year: int, metric: str) -> str:
    """Tính 1 CHỈ TIÊU PHÁI SINH cho nhiều công ty cùng lúc (code tính, khỏi tự nhân chia).
    metric ∈ bien_gop, bien_rong, bien_hdkd, cfo_margin, gross_net_diff, roe, roa (bình quân), roe_end, roa_end,
    accrual, quick, current, de, debt_asset, icover, cfo_debt, htk_nnh, equity_mult, asset_turn, inv_days,
    tsdh_ratio, htk_ratio, sga_intensity, cfo_lnst, cfo_lnthuan, growth_rev, cfo, lnst, rev. Trả JSON {mã CK: giá trị (tỉ số thô, CHƯA ×100)}."""
    _tick_tool_budget()
    f = METRICS.get(metric)
    if not f: return f"metric không hợp lệ. Chọn: {list(METRICS)}"
    return json.dumps({c: f(c, int(year)) for c in companies.replace(" ", "").split(",") if c}, ensure_ascii=False)

@tool
def pick_company(companies: str, year: int, select_metric: str, direction: str, filter_metric: str = "", filter_op: str = "") -> str:
    """CHỌN công ty theo logic tổ hợp — CODE lo lọc + trung vị + argmax/min (đừng tự tính trong đầu).
    select_metric/filter_metric: tên chỉ tiêu (như compute_metric). direction: 'max'/'min'.
    filter_op: '>median','<median','>0','<0','>1.5','<1'... (rỗng = không lọc). Trả JSON {winner, values}."""
    _tick_tool_budget()
    import statistics, re as _re
    cs = [c for c in companies.replace(" ", "").split(",") if c]; y = int(year)
    sf = METRICS.get(select_metric)
    if not sf: return f"select_metric không hợp lệ: {list(METRICS)}"
    if filter_metric and filter_op:
        ff = METRICS.get(filter_metric)
        if ff:
            vals = {c: ff(c, y) for c in cs}; oks = [v for v in vals.values() if v is not None]
            thr = statistics.median(oks) if ("median" in filter_op and oks) else (float(_re.sub(r"[^0-9.\-]", "", filter_op) or 0))
            gt = ">" in filter_op
            cs = [c for c in cs if vals[c] is not None and (vals[c] > thr if gt else vals[c] < thr)]
    sv = {c: sf(c, y) for c in cs}; sv = {c: v for c, v in sv.items() if v is not None}
    if not sv: return json.dumps({"winner": None})
    winner = max(sv, key=sv.get) if direction == "max" else min(sv, key=sv.get)
    return json.dumps({"winner": winner, "values": {c: round(v, 4) for c, v in sv.items()}}, ensure_ascii=False)

@tool
def pick_year(company: str, years: str, metric: str, direction: str = "max", offset: int = 0, first_neg: bool = False) -> str:
    """CHỌN NĂM cho 1 công ty theo điều kiện — CODE lo (đừng tự dò/đếm năm, dễ sai off-by-one).
    years = danh sách năm cách phẩy, vd '2016,2017,2018,2019,2020,2021'. metric = tên chỉ tiêu (như compute_metric).
    first_neg=True → năm ĐẦU TIÊN metric < 0. Ngược lại → năm có metric cao nhất/thấp nhất (direction max/min).
    offset=1 → trả về NĂM SAU năm được chọn ('năm ngay sau năm...'). Trả JSON {year}."""
    _tick_tool_budget()
    ys = sorted(int(y) for y in str(years).replace(" ", "").split(",") if y)
    f = METRICS.get(metric)
    if not f: return f"metric không hợp lệ: {list(METRICS)}"
    if first_neg:
        yr = next((y for y in ys if (f(company, y) or 0) < 0), None)
    else:
        vals = {y: f(company, y) for y in ys}; vals = {y: v for y, v in vals.items() if v is not None}
        yr = (max(vals, key=vals.get) if direction == "max" else min(vals, key=vals.get)) if vals else None
    return json.dumps({"year": (yr + int(offset)) if yr is not None else None})

# ---------- EXPR (agent xuất) → answer (eval) + pandas_query + evidence ----------
_REFS = []
def _V(company, year, statement, ma_so):
    v = _get_raw(company, year, statement, ma_so)
    if v is None:
        raise ValueError(f"V miss {company} {year} {statement} {ma_so}")
    _REFS.append((str(company), int(year), str(statement).upper(), int(ma_so), int(round(v))))   # ma_so int → khớp pandas ==N
    return float(v)

_EVAL_NS = {"float": float, "int": int, "round": round, "abs": abs, "min": min, "max": max, "sum": sum}
def eval_expr(expr):
    global _REFS
    _REFS = []
    val = round(float(eval(expr, {"__builtins__": {}}, {**_EVAL_NS, "V": _V})), 2)   # gold luôn round 2 chữ số
    return val, list(dict.fromkeys(_REFS))   # dedupe giữ thứ tự

_VPAT = re.compile(r"V\(\s*'([^']+)'\s*,\s*(\d+)\s*,\s*'([^']+)'\s*,\s*'?(\d+)'?\s*\)")
def expr_to_pandas(expr):
    def repl(mt):
        c, y, s, m = mt.group(1), mt.group(2), mt.group(3).upper(), int(mt.group(4))
        # astype(int/str) → robust cả khi BTC load CSV dtype=str LẪN inference (không lệ thuộc kiểu đọc)
        return (f"float(df1[(df1['company'].astype(str)=='{c}')&(df1['year'].astype(int)=={y})"
                f"&(df1['statement'].astype(str)=='{s}')&(df1['ma_so'].astype(int)=={m})]['value'].values[0])")
    return _VPAT.sub(repl, expr)

SYS = (
    "Bạn là trợ lý phân tích BCTC. Giải câu hỏi bằng cách GỌI TOOL — KHÔNG bịa số.\n"
    "So sánh NHÓM công ty: DÙNG get_values (lấy 1 chỉ tiêu cho nhiều công ty 1 lần, vd get_values('ASM,DBC,MPC',2024,'PL','20')) "
    "thay vì gọi get_value nhiều lần — nhanh + ít lỗi. get_value chỉ cho 1 công ty lẻ.\n"
    "Mã số TT200: PL: 10 doanh thu thuần, 11 giá vốn, 20 LN gộp, 23 chi phí lãi vay, 30 LN thuần HĐKD, 50 LN trước thuế, 60 LNST. "
    "BS: 100 TS ngắn hạn, 140 hàng tồn kho, 270 tổng tài sản, 300 nợ phải trả, 310 nợ ngắn hạn, 400 vốn chủ sở hữu. "
    "CF: 20 lưu chuyển tiền thuần HĐKD (CFO).\n"
    "NGÂN HÀNG/BẢO HIỂM/tổ chức tài chính dùng HỆ MÃ SỐ KHÁC hoàn toàn danh sách trên — TUYỆT ĐỐI đừng đoán Mã số cho "
    "chỉ tiêu ngoài danh sách (tiền gửi khách hàng, dự phòng rủi ro cho vay, chi phí hoạt động, lợi nhuận trước thuế "
    "của ngân hàng...). Với chỉ tiêu KHÔNG có trong danh sách Mã số ở trên, GỌI search_label(company, year, keyword) "
    "trước để tìm đúng (statement, ma_so) bằng từ khoá, rồi mới get_value.\n"
    "Công thức: biên gộp %=Mã20(PL)/Mã10(PL)×100; biên ròng %=Mã60(PL)/Mã10(PL)×100; "
    "ROE %=Mã60(PL)/((Mã400(BS) đầu kỳ+cuối kỳ)/2)×100 — mẫu số là VCSH BÌNH QUÂN 2 năm, KHÔNG phải cuối kỳ; "
    "ROA %=Mã60(PL)/((Mã270(BS) đầu+cuối)/2)×100; thanh toán nhanh=(Mã100(BS)-Mã140(BS))/Mã310(BS); "
    "D/E=Mã300(BS)/Mã400(BS); hệ số thanh toán lãi vay=(Mã50(PL)+Mã23(PL))/Mã23(PL); tỷ lệ nợ=Mã300(BS)/Mã270(BS)×100; "
    "HTK/nợ ngắn hạn=Mã140(BS)/Mã310(BS); CFO/nợ ngắn hạn=Mã20(CF)/Mã310(BS); "
    "chênh biên gộp−biên ròng (điểm %)=(Mã20(PL)-Mã60(PL))/Mã10(PL)×100; "
    "tỷ lệ dồn tích %=(Mã60(PL)-Mã20(CF))/((Mã270(BS) đầu+cuối)/2)×100.\n"
    "Câu 'trong nhóm, công ty CÓ [A] cao nhất/thấp nhất [+ điều kiện lọc] thì [B]': ĐỪNG tự tính trung vị/so sánh trong đầu (dễ sai). "
    "DÙNG pick_company(companies, year, select_metric=A, direction='max'/'min', filter_metric=điều-kiện, filter_op='>median'/'<median'/'>1.5'/'>0'...) "
    "→ CODE trả về ĐÚNG công ty thắng. Sau đó chỉ tính [B] cho công ty thắng đó rồi ra EXPR. "
    "select_metric/filter_metric phải là TÊN chỉ tiêu: bien_gop, bien_rong, bien_hdkd, cfo_margin, gross_net_diff, "
    "roe, roa (BÌNH QUÂN), roe_end, roa_end, accrual, quick, current, de, debt_asset, icover, cfo_debt, htk_nnh, "
    "equity_mult, asset_turn, inv_days, tsdh_ratio, htk_ratio, sga_intensity, cfo_lnst, cfo_lnthuan, growth_rev, cfo, lnst, rev. "
    "Chọn tên KHỚP đúng ý câu hỏi (vd 'hàng tồn kho so với nợ ngắn hạn'→htk_nnh; 'biên gộp cao hơn biên ròng ... điểm %'→gross_net_diff; "
    "'tỷ lệ dồn tích'→accrual; ROE/ROA mặc định dùng bản BÌNH QUÂN, chỉ dùng roe_end/roa_end khi câu ghi rõ 'cuối kỳ').\n"
    "Cần tỉ số phái sinh cho nhiều công ty: compute_metric(companies, year, metric) trả cả nhóm 1 lần.\n"
    "Câu 1 công ty CHỌN NĂM ('năm mà X cao nhất', 'năm ĐẦU TIÊN CFO âm', 'năm NGAY SAU năm...'): ĐỪNG tự dò năm (dễ off-by-one). "
    "DÙNG pick_year(company, years='2016,2017,...', metric, direction='max'/'min', offset=0/1, first_neg=True/False) "
    "→ CODE trả về ĐÚNG năm (offset=1 = năm SAU; first_neg=True = năm đầu tiên metric<0). Rồi tính chỉ tiêu cho năm đó → EXPR.\n"
    "QUAN TRỌNG: sau MỖI kết quả tool, GỌI get_value tiếp theo NGAY — KHÔNG viết đoạn giải thích dài giữa các bước.\n"
    "Khi đã có đủ MỌI số cần thiết, kết thúc bằng ĐÚNG một dòng bắt đầu 'EXPR:' chứa MỘT biểu thức Python tính đáp án, "
    "trong đó MỖI số phải viết dạng V('MÃ_CK', NĂM, 'PL'/'BS'/'CF', 'MÃ_SỐ') — KHÔNG viết số thật, KHÔNG gọi get_value nữa. "
    "ĐƠN VỊ ĐÁP ÁN (bắt buộc đúng, nếu không sẽ lệch >0.01):\n"
    " • Câu hỏi '%'/'phần trăm'/'điểm phần trăm' (biên gộp, biên ròng, ROE, ROA, tỷ lệ nợ, dồn tích, chênh biên gộp−ròng): "
    "EXPR NHÂN ×100 → vd (V(gộp)/V(rev))*100.\n"
    " • Câu hỏi 'bao nhiêu LẦN'/'hệ số'/'vòng' (current, quick, D/E, thanh toán lãi vay, HTK/nợ ngắn hạn, CFO/nợ ngắn hạn, "
    "vòng quay, hệ số nhân VCSH): GIỮ nguyên tỉ số, KHÔNG ×100.\n"
    " • Câu hỏi tiền tuyệt đối: triệu→/1e6, tỷ→/1e9, NGHÌN TỶ→/1e12. Số ngày tồn kho: giữ nguyên (đã ×365).\n"
    "VÍ DỤ: EXPR: (V('PVT',2024,'PL','50')+V('PVT',2024,'PL','23'))/V('PVT',2024,'PL','23')\n"
    "Chỉ dùng các công ty/dòng ĐÃ tra. Biểu thức phải trả về MỘT SỐ.\n"
    "NẾU một số cần thiết là null, hoặc phải chia cho 0, hoặc thiếu dữ liệu để tính → trả NGAY 'EXPR: None' và DỪNG. "
    "TUYỆT ĐỐI KHÔNG lặp lại phân tích nhiều lần — mỗi bước phải TIẾN (gọi tool mới hoặc ra EXPR)."
)

def load_scope(qt):
    global _SEARCH_COUNT, _TOOL_CALL_COUNT
    _SEARCH_COUNT = 0   # reset đầu mỗi câu — ngưỡng cảnh báo trong search_label tính theo TỪNG câu
    _TOOL_CALL_COUNT = 0   # reset đầu mỗi câu — ngân sách tool CỨNG tính theo TỪNG câu
    d = {}
    for tk, y in B.rel_pairs(qt):
        fr = P.find_report(tk, y, P.doctype(qt))
        if fr:
            d[(tk, int(y))] = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
    return d

def build_model():
    return OllamaModel(
        host="http://localhost:11434", model_id=MODEL_ID,
        additional_args={"think": False},          # TẮT thinking → nhanh, agentic lo reasoning
        keep_alive="30m", options={"temperature": 0.2},
    )

def _extract_expr_line(text):
    """Lấy dòng 'EXPR:' CUỐI CÙNG, chỉ nhận nếu là biểu thức thật (không phải prose trích dẫn lệnh)."""
    if "EXPR:" not in text:
        return None
    e = text.rsplit("EXPR:", 1)[1].strip().split("\n")[0].strip("` *")
    if not e:
        return None
    if e == "None" or e[:2] == "V(" or e[:1] in "(-" or e[:1].isdigit() or e.startswith(("float", "min", "max", "sum", "abs", "round")):
        return e
    return None

def run_question(qt, model=None, max_turns=18):
    global _SCOPE
    _SCOPE = load_scope(qt)
    companies = sorted(set(c for c, _ in _SCOPE)); years = sorted(set(y for _, y in _SCOPE))
    global _DT
    _DT = P.doctype(qt)
    agent = Agent(model=model or build_model(), tools=[get_value, get_values, compute_metric, pick_company, pick_year, search_label], system_prompt=SYS)
    prompt = f"Câu hỏi: {qt}\nCông ty trong phạm vi: {companies}\nNăm: {years}\nGiải từng bước bằng get_value."
    text = ""; stuck = 0; expr = None
    for turn in range(max_turns):   # nudge tới khi model xuất EXPR (strands kết thúc mỗi lần model narrate)
        try:
            text = str(agent(prompt))   # 4B đôi khi sinh tool-call JSON hỏng → Ollama 500 → bỏ run này
        except _ToolBudgetExceeded:
            return {"answer": None, "pandas": None, "refs": [], "expr": None, "text": "[TOOL BUDGET EXCEEDED]"}
        except Exception as e:
            return {"answer": None, "pandas": None, "refs": [], "expr": None, "text": f"[AGENT ERR {type(e).__name__}]"}
        expr = _extract_expr_line(text)   # lấy EXPR CUỐI + chỉ nhận biểu thức thật (bỏ prose trích lệnh)
        if expr:
            break
        stuck += 1
        if stuck >= 4:   # narrate 4 lượt không tiến (kẹt lặp / null) → ép kết thúc rồi bỏ
            break
        # Đo được (id498): model hay VIẾT "tôi sẽ gọi get_value..." rồi KHÔNG gọi trong cùng lượt →
        # hết lượt mà vẫn chưa có EXPR dù đã biết đủ số. Nudge phải chặn rõ kiểu "nói-không-làm" này.
        prompt = ("Bạn vừa MÔ TẢ sẽ làm gì đó nhưng CHƯA THỰC HIỆN. GỌI TOOL NGAY trong lượt này — "
                  "đừng viết thêm câu giải thích nào trước khi gọi. Nếu đã đủ số, xuất NGAY dòng "
                  "'EXPR: <biểu thức V()>' — đừng mô tả ý định, HÀNH ĐỘNG luôn."
                  if stuck >= 2 else
                  "GỌI get_value/get_values tiếp, hoặc trả 'EXPR: <biểu thức V()>' nếu đủ số.")
    if not expr or expr == "None":
        return {"answer": None, "pandas": None, "refs": [], "expr": expr, "text": text}
    try:
        ans, refs = eval_expr(expr)
        pq = expr_to_pandas(expr)
    except Exception as e:
        return {"answer": None, "pandas": None, "refs": [], "expr": expr, "text": f"{text} [EXPR ERR {e}]"}
    return {"answer": ans, "pandas": pq, "refs": refs, "expr": expr, "text": text}

# ================== V2: Planner (LLM prompt NHỎ) → PLAN JSON → Executor deterministic ==================
PLAN_SYS = (
    "Bạn là bộ PHÂN TÍCH câu hỏi BCTC. Đọc câu hỏi + danh sách MÃ CK và NĂM có sẵn, xuất DUY NHẤT một JSON "
    "mô tả CÁCH giải. KHÔNG giải, KHÔNG tính số, KHÔNG viết gì ngoài JSON.\n\n"
    "ARCHETYPE:\n"
    "- group_maxb: so sánh NHIỀU công ty trong 1 năm → chọn 1 công ty theo điều kiện → tính 1 chỉ tiêu.\n"
    "- temporal: MỘT công ty → chọn 1 NĂM theo điều kiện → tính 1 chỉ tiêu.\n"
    "- lookup: lấy thẳng 1 chỉ tiêu của 1 công ty-năm.\n"
    "- group_count: ĐẾM xem trong nhóm có BAO NHIÊU công ty thoả điều kiện ('có bao nhiêu doanh nghiệp...'). "
    "Dùng filter_metric+filter_op; nếu câu hỏi có HAI điều kiện ('đồng thời', 'và') thì thêm filter2_metric+filter2_op. "
    "KHÔNG cần target_metric.\n"
    "- other: không khớp 3 loại trên (để hệ khác xử lý).\n\n"
    "METRIC (tên: nghĩa):\n"
    "bien_gop=biên LN gộp; bien_rong/npm=biên LN ròng(LNST/DTT); bien_hdkd=biên LN thuần HĐKD; cfo_margin=CFO/DTT; "
    "sga_intensity=(CP bán hàng+quản lý)/DTT; gross_net_diff=chênh biên gộp−biên ròng(điểm %); "
    "roe=ROE(VCSH bình quân); roa=ROA(TS bình quân); roe_end/roa_end=ROE/ROA cuối kỳ; accrual=tỷ lệ dồn tích; "
    "quick=thanh toán nhanh; current=thanh toán hiện hành; de=nợ/VCSH; debt_asset=nợ/tổng TS; "
    "icover=thanh toán lãi vay; cfo_debt=CFO/nợ ngắn hạn; htk_nnh=HTK/nợ ngắn hạn; equity_mult=hệ số nhân VCSH; "
    "asset_turn=vòng quay tổng TS; inv_days=số ngày tồn kho; tsdh_ratio=TSDH/tổng TS; htk_ratio=HTK/tổng TS; "
    "cfo_lnst=CFO/LNST; cfo_lnthuan=CFO/LN thuần HĐKD; cfo_lnst_rev=(CFO−LNST)/doanh thu thuần; growth_rev=tăng trưởng doanh thu; "
    "cfo=dòng tiền HĐKD(tiền); lnst=LNST(tiền); rev=doanh thu thuần(tiền); wcap=vốn lưu động ròng(TSNH−nợ NH).\n"
    "PHÂN BIỆT QUAN TRỌNG: 'lợi nhuận sau thuế TRÊN DOANH THU THUẦN' = npm (KHÔNG phải roe!). "
    "roe = LNST trên VỐN CHỦ SỞ HỮU. 'giá trị lưu chuyển tiền thuần từ HĐKD' = cfo (metric tiền, không phải tỉ số).\n\n"
    "SCHEMA JSON (chỉ điền field cần):\n"
    '{"archetype","companies":[mã CK],"years":[năm],"filter_metric","filter_op",'
    '"filter2_metric","filter2_op",'
    '"select_metric","direction","target_metric","target_offset_year","first_neg","unit"}\n\n'
    "QUY TẮC:\n"
    "- companies/years CHỈ lấy từ danh sách cho sẵn (đã là mã CK).\n"
    "- select_metric = chỉ tiêu để CHỌN công ty/năm ('có [X] cao nhất/thấp nhất'). direction: cao/lớn nhất=max, thấp/nhỏ nhất=min.\n"
    "- filter_metric+filter_op = điều kiện lọc phụ. filter_op ∈ '>0','<0','>median','<median','>1.5','<1'... Không có lọc → bỏ 2 field này.\n"
    "- target_metric = chỉ tiêu CẦN TÍNH ('... là bao nhiêu'), thường ở đầu/cuối câu.\n"
    "- CẤU TRÚC 'A của doanh nghiệp/công ty có B cao nhất/thấp nhất': A=target_metric, B=select_metric. A đứng TRƯỚC chữ 'của', "
    "B đứng SAU chữ 'có' và gắn 'cao/thấp nhất'. TUYỆT ĐỐI không đảo A↔B (dù A,B đều là chỉ số thanh khoản gần giống nhau).\n"
    "- temporal: 'năm mà X cao nhất'→select_metric=X+direction; 'năm ĐẦU TIÊN X âm'→first_neg=true,select_metric=X; "
    "'năm NGAY SAU năm...'→target_offset_year=1.\n"
    "- unit: 'phần trăm'/'%'→percent; 'điểm phần trăm'→point; 'bao nhiêu lần'/'hệ số'→ratio; 'tỷ đồng'→ty; 'triệu'→trieu; 'nghìn tỷ'→nghinty.\n"
    "- ROE/ROA không ghi 'cuối kỳ'→roe/roa(bình quân). 'vốn lưu động ròng âm'→filter_metric=wcap,filter_op='<0'.\n"
    "- Không chắc khớp archetype→ {\"archetype\":\"other\"}.\n\n"
    "VÍ DỤ:\n"
    "Q: Năm 2024, có bao nhiêu doanh nghiệp trong nhóm HPX, NVL, SCR, VIC và VRE đồng thời có lưu chuyển tiền thuần từ "
    "hoạt động kinh doanh dương và hệ số nợ trên vốn chủ sở hữu dưới 1.5? [MÃ:HPX,NVL,SCR,VIC,VRE NĂM:2024]\n"
    'JSON: {"archetype":"group_count","companies":["HPX","NVL","SCR","VIC","VRE"],"years":[2024],"filter_metric":"cfo","filter_op":">0","filter2_metric":"de","filter2_op":"<1.5"}\n'
    "Q: Năm 2024, trong nhóm BSR, PLX và PVT, công ty có biên lợi nhuận gộp cao nhất trong số các công ty có lưu chuyển "
    "tiền thuần từ hoạt động kinh doanh dương có hệ số khả năng thanh toán lãi vay là bao nhiêu lần? [MÃ:BSR,PLX,PVT NĂM:2024]\n"
    'JSON: {"archetype":"group_maxb","companies":["BSR","PLX","PVT"],"years":[2024],"filter_metric":"cfo","filter_op":">0","select_metric":"bien_gop","direction":"max","target_metric":"icover","unit":"ratio"}\n'
    "Q: Trong nhóm MCH, QNS và OGC năm 2024, hệ số thanh toán nhanh của doanh nghiệp có tỷ lệ CFO trên lợi nhuận sau thuế "
    "cao nhất là bao nhiêu lần? [MÃ:MCH,QNS,OGC NĂM:2024]\n"
    'JSON: {"archetype":"group_maxb","companies":["MCH","QNS","OGC"],"years":[2024],"select_metric":"cfo_lnst","direction":"max","target_metric":"quick","unit":"ratio"}\n'
    "Q: Trong giai đoạn 2016-2021, biên lợi nhuận gộp của năm ngay sau năm đầu tiên công ty KBC ghi nhận CFO âm là bao "
    "nhiêu phần trăm? [MÃ:KBC NĂM:2016,2017,2018,2019,2020,2021]\n"
    'JSON: {"archetype":"temporal","companies":["KBC"],"years":[2016,2017,2018,2019,2020,2021],"first_neg":true,"select_metric":"cfo","target_metric":"bien_gop","target_offset_year":1,"unit":"percent"}\n'
    "Q: Năm 2024, đối với các công ty HPX, KBC, NVL, SCR, VIC, VPI, VRE có tỉ số thanh toán hiện hành lớn hơn 1.5, tỉ trọng "
    "hàng tồn kho trên tổng tài sản của doanh nghiệp có tỉ số thanh toán nhanh thấp nhất là bao nhiêu phần trăm? "
    "[MÃ:HPX,KBC,NVL,SCR,VIC,VPI,VRE NĂM:2024]\n"
    'JSON: {"archetype":"group_maxb","companies":["HPX","KBC","NVL","SCR","VIC","VPI","VRE"],"years":[2024],"filter_metric":"current","filter_op":">1.5","select_metric":"quick","direction":"min","target_metric":"htk_ratio","unit":"percent"}\n'
    "Q: Trong nhóm AAA, BBB, CCC năm 2023 có hệ số nợ trên vốn chủ sở hữu lớn hơn 1, hệ số dòng tiền hoạt động trên nợ ngắn "
    "hạn của doanh nghiệp có hệ số thanh toán nhanh thấp nhất là bao nhiêu lần? [MÃ:AAA,BBB,CCC NĂM:2023]  "
    "(target=cfo_debt đứng TRƯỚC 'của'; select=quick SAU 'có ... thấp nhất'; đừng đảo hai chỉ số thanh khoản)\n"
    'JSON: {"archetype":"group_maxb","companies":["AAA","BBB","CCC"],"years":[2023],"filter_metric":"de","filter_op":">1","select_metric":"quick","direction":"min","target_metric":"cfo_debt","unit":"ratio"}'
)

def _extract_json(text):
    """Lấy khối JSON {...} (từ '{' đầu đến '}' cuối). None nếu parse lỗi."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j+1])
    except Exception:
        return None

def plan_question(qt, companies, years, model=None):
    """Planner: câu hỏi + mã/năm cho sẵn → PLAN dict. None nếu không parse được."""
    agent = Agent(model=model or build_model(), tools=[], system_prompt=PLAN_SYS)
    tag = f" [MÃ:{','.join(companies)} NĂM:{','.join(str(y) for y in years)}]"
    prompt = f"Q: {qt}{tag}\nJSON:"
    for attempt in range(2):
        try:
            text = str(agent(prompt))
        except Exception:
            return None
        p = _extract_json(text)
        if p and isinstance(p, dict) and p.get("archetype"):
            return p
        prompt = "Chỉ xuất DUY NHẤT một dòng JSON đúng schema, không giải thích. JSON:"
    return None

def run_question_v2(qt, model=None):
    """Planner → Executor deterministic; fallback về tool-loop cũ nếu không lập/giải được plan."""
    global _SCOPE, _DT
    _SCOPE = load_scope(qt)
    _DT = P.doctype(qt)
    companies = sorted(set(c for c, _ in _SCOPE)); years = sorted(set(y for _, y in _SCOPE))
    if companies:
        plan = plan_question(qt, companies, years, model)
        if plan and plan.get("archetype") in ("group_maxb", "temporal", "lookup", "group_count"):
            r = exec_plan(plan)
            if r and r.get("answer") is not None:
                return {"answer": r["answer"], "pandas": r["pandas"], "refs": r["refs"],
                        "expr": r["expr"], "text": f"[V2 {plan['archetype']} winner={r.get('winner')} y={r.get('year')}]"}
    return run_question(qt, model)   # fallback: agent tool-loop (grounded như cũ, không mất gì)

def _sig(x, n=4):
    from math import log10, floor
    if not x:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def run_voted(qt, model=None, n=3, runner=None):
    """Self-consistency: chạy n lần → vote đáp án (theo 4 chữ số ý nghĩa) → giữ EXPR/pandas của phe thắng.
    runner = run_question (tool-loop) hoặc run_question_v2 (planner→executor)."""
    from collections import Counter
    runner = runner or run_question
    res = []
    for _ in range(n):
        try: res.append(runner(qt, model))
        except Exception: pass
    ok = [r for r in res if r["answer"] is not None]
    if not ok:
        return {"answer": None, "pandas": None, "refs": [], "expr": None, "votes": 0, "n": 0}
    cnt = Counter(_sig(r["answer"]) for r in ok)
    win = cnt.most_common(1)[0][0]
    same = [r for r in ok if _sig(r["answer"]) == win]
    # Trong phe thắng, ƯU TIÊN lượt CÓ BẰNG CHỨNG (pandas + refs). Trước đây lấy lượt ĐẦU TIÊN, nên
    # khi tool-loop đoán trúng cùng số với planner thì ta giữ bản KHÔNG có refs — cùng đáp án nhưng
    # build_submission buộc phải vứt đi. Đo được: nhóm câu đếm chỉ còn 1/10 câu dùng được vì lỗi này.
    r = next((x for x in same if x.get("pandas") and x.get("refs")), same[0])
    r["votes"] = cnt[win]; r["n"] = len(ok); r["all"] = [round(x["answer"], 4) for x in ok]
    return r

def hard_ids():
    """350 câu AGG/COND (deterministic ~0%) — nhóm agent xử lý."""
    return [q["id"] for q in P.Q if any(k in q["question"].lower() for k in P.AGG + P.COND)]

if __name__ == "__main__":
    import time, json as _json
    model = build_model()
    N = int(os.environ.get("AGENT_VOTES", "3"))
    RUNNER = run_question_v2 if os.environ.get("AGENT_V2") == "1" else run_question   # V2 = planner→executor
    OUT_F = os.environ.get("AGENT_OUT", "/tmp/agent_batch.json")
    ids_env = os.environ.get("AGENT_IDS", "")
    ids = [int(x) for x in ids_env.split(",")] if ids_env else hard_ids()
    out, done = [], set()
    if os.environ.get("AGENT_RESUME") == "1" and os.path.exists(OUT_F):   # RESUME: bỏ qua câu đã xong
        out = _json.load(open(OUT_F, encoding="utf-8")); done = {e["id"] for e in out if e.get("answer") is not None}
        print(f"RESUME: đã có {len(done)} câu, chạy tiếp {len(ids)-len(done)}")
    for qid in ids:
        if qid in done:
            continue
        q = [x for x in P.Q if x["id"] == qid][0]
        t = time.time()
        try:
            r = run_voted(q["question"], model, N, runner=RUNNER)
        except Exception as e:
            r = {"answer": None, "pandas": None, "refs": [], "expr": f"BATCH ERR {type(e).__name__}", "votes": 0}
        dt = time.time() - t
        print(f"id{qid} [{dt:.0f}s] answer={r['answer']} votes={r['votes']}/{r.get('n',0)} all={r.get('all')}")
        print(f"    EXPR: {r['expr']}")
        out.append({"id": qid, "question": q["question"], "answer": r["answer"],
                    "pandas": r["pandas"], "refs": r["refs"], "votes": r["votes"], "n": r.get("n", 0)})
        with open(os.environ.get("AGENT_OUT", "/tmp/agent_batch.json"), "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False)   # lưu tăng dần → không mất khi bị cắt
    print("SAVED", len(out))
