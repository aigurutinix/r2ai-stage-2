"""Prototype AGENTIC decompose (local Qwen3.5-4B qua Ollama tool-calling).
Giả thuyết: model nhỏ làm nhiều bước ĐƠN GIẢN (gọi get_value từng chỉ tiêu) sẽ giải được
câu multi-step ('công ty có A cao nhất thì B') mà one-shot-pandas fail.
Chạy: python agent_proto.py   (in trace tool-call để soi tay đáp án vs report).
"""
import os, json, re, urllib.request
import pipeline as P
import build_submission as B

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("AGENT_MODEL", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "20"))

# ---------- Data access (grounded) ----------
def load_scope(qt):
    data = {}
    for tk, y in B.rel_pairs(qt):
        fr = P.find_report(tk, y, P.doctype(qt))
        if not fr:
            continue
        rows = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
        data[(tk, int(y))] = rows
    return data

def get_value(data, company, year, statement, ma_so):
    rows = data.get((str(company), int(year)))
    if not rows:
        return None
    ma = str(ma_so).strip()
    if len(ma) >= 3:   # mã 3 chữ số = BS aggregate (100/270/300/400...) — st-tagging KHÔNG đáng tin cho BS,
        cands = [r for r in rows if str(r["ma"]) == ma and r["cur"] is not None]   # match ma_so bỏ qua st (3-digit không trùng PL/CF)
        return max(cands, key=lambda r: abs(r["cur"]))["cur"] if cands else None
    for r in rows:   # mã 2 chữ số PL/CF: dùng statement để disambiguate (ma20 = LN gộp PL / CFO CF)
        if str(r["ma"]) == ma and r["st"] == statement and r["cur"] is not None:
            return r["cur"]
    return None

TOOLS = [
    {"type": "function", "function": {
        "name": "get_value",
        "description": "Giá trị (VND) của chỉ tiêu theo Mã số TT200 trong 1 công ty-năm-báo cáo. statement: BS/PL/CF. Trả null nếu không có dòng đó.",
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string"}, "year": {"type": "integer"},
            "statement": {"type": "string", "enum": ["BS", "PL", "CF"]}, "ma_so": {"type": "string"}},
            "required": ["company", "year", "statement", "ma_so"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Nộp đáp án số cuối cùng (đã đổi đúng đơn vị câu hỏi).",
        "parameters": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}}},
]

SYS = (
    "Bạn là trợ lý phân tích BCTC. Trả lời bằng cách GỌI TOOL từng bước, TUYỆT ĐỐI KHÔNG tự bịa số — mọi con số phải lấy từ get_value.\n"
    "Mã số TT200: PL: 10 doanh thu thuần, 11 giá vốn, 20 LN gộp, 30 LN thuần HĐKD, 50 LN trước thuế, 60 LNST. "
    "BS: 100 TS ngắn hạn, 140 hàng tồn kho, 270 tổng tài sản, 300 nợ phải trả, 310 nợ ngắn hạn, 400 vốn chủ sở hữu. "
    "CF: 20 lưu chuyển tiền thuần HĐKD (CFO). PL: 22 chi phí tài chính, 23 chi phí lãi vay, 25 chi phí bán hàng, 26 chi phí QLDN.\n"
    "Công thức: biên gộp % = Mã20(PL)/Mã10(PL)×100; ROE % = Mã60(PL)/Mã400(BS)×100; ROA % = Mã60(PL)/Mã270(BS)×100; "
    "thanh toán nhanh = (Mã100(BS)-Mã140(BS))/Mã310(BS); thanh toán hiện hành = Mã100(BS)/Mã310(BS); D/E = Mã300(BS)/Mã400(BS); "
    "hệ số thanh toán lãi vay = EBIT/lãi vay = (Mã50(PL)+Mã23(PL))/Mã23(PL); tỷ lệ nợ = Mã300(BS)/Mã270(BS).\n"
    "Câu 'công ty CÓ [A] cao nhất/thấp nhất thì [B]': (1) get_value lấy A của TỪNG công ty, (2) so sánh tìm công ty max/min, "
    "(3) get_value lấy B của ĐÚNG công ty đó, (4) tính, (5) final_answer. A và B thường KHÁC nhau.\n"
    "Đơn vị: get_value trả VND. Đổi theo câu hỏi: triệu→/1e6, tỷ→/1e9, %/lần→giữ tỉ số (×100 nếu %). "
    "Chỉ gọi final_answer khi đã có đủ số thật."
)

def _post(messages):
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0.2}
    req = urllib.request.Request(OLLAMA + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]

def run_question(qt, verbose=True):
    data = load_scope(qt)
    companies = sorted(set(c for c, _ in data)); years = sorted(set(y for _, y in data))
    ctx = f"Câu hỏi: {qt}\nCông ty trong phạm vi: {companies}\nNăm: {years}\nHãy giải từng bước bằng get_value."
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": ctx}]
    nval = 0; seen = set()
    for step in range(MAX_STEPS):
        msg = _post(messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if verbose: print("   [model text]", (msg.get("content") or "")[:120])
            messages.append({"role": "user", "content": "GỌI TOOL: get_value để lấy thêm số, hoặc final_answer nếu đã đủ để tính."})
            continue
        for c in calls:
            fn = c["function"]["name"]; args = c["function"]["arguments"]
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            if fn == "final_answer":
                if verbose: print(f"   [FINAL] {args.get('value')}")
                return args.get("value")
            if fn == "get_value":
                nval += 1
                key = (args.get("company"), args.get("year"), args.get("statement"), str(args.get("ma_so")))
                v = get_value(data, args.get("company"), args.get("year"), args.get("statement"), args.get("ma_so"))
                if verbose: print(f"   get_value({key[0]},{key[1]},{key[2]},ma={key[3]}) -> {v}")
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(v)})
                if key in seen:   # lặp lại call đã gọi → ép hội tụ
                    messages.append({"role": "user", "content": "Bạn đang lặp lại. DÙNG các số ĐÃ CÓ, tính công thức, gọi final_answer NGAY."})
                seen.add(key)
        if nval >= 14:   # đủ số → ép tính + final
            messages.append({"role": "user", "content": "Đã đủ dữ liệu. TÍNH kết quả theo công thức (nhớ đổi đơn vị câu hỏi) và gọi final_answer NGAY, KHÔNG lấy thêm."})
    if verbose: print("   [het step, khong final]")
    return None

if __name__ == "__main__":
    ids = [int(x) for x in os.environ.get("AGENT_IDS", "371,392,398,368,364").split(",")]
    for qid in ids:
        q = [x for x in P.Q if x["id"] == qid][0]
        print(f"\n===== id{qid} =====\n{q['question'][:150]}")
        run_question(q["question"])
