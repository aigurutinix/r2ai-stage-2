"""Rerank bảng bằng Qwen3-Reranker-0.6B (NGOẠI TUYẾN) → rerank_cache.json.

Vì sao tách khỏi build_submission: giữ bài nộp THUẦN DETERMINISTIC. Chạy file này một lần để
đóng băng thứ tự bảng; `build_submission.py` chỉ đọc JSON, không import torch, vẫn chạy CPU vài
phút — đúng cách `agent_full_v2.json` đang làm.

Model: Qwen/Qwen3-Reranker-0.6B, phát hành 05/06/2025 (trước mốc 31/05/2026), dùng nguyên trọng
số gốc, KHÔNG huấn luyện thêm.

Đo trên 200 câu (bảng đúng nằm trong BM25 top-10):
    BM25 thuần : top-1 44.5% · top-3 82.5% · top-5 90.5%
    + rerank    : top-1 47.0% · top-3 84.0% · top-5 93.5%
Cải thiện dương ở cả ba mức nhưng mỏng — kỳ vọng ~+0.01 TABLES, cần leaderboard xác nhận.

Chạy:  python rerank_offline.py        (~40-50 phút trên RTX 3050 6GB)
"""
import json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pipeline as P
import build_submission as B

MODEL = "Qwen/Qwen3-Reranker-0.6B"
TOPK = 10                       # số bảng BM25 đưa vào rerank
OUT = os.path.join(os.path.dirname(__file__), "rerank_cache.json")

PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. '
       'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTR = "Tìm bảng trong báo cáo tài chính chứa chỉ tiêu mà câu hỏi nhắc tới"

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
YES, NO = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


@torch.no_grad()
def rerank(query, docs, bs=4):
    """Điểm liên quan = P("yes") ở token cuối — cách chấm chuẩn của Qwen3-Reranker."""
    out = []
    for i in range(0, len(docs), bs):
        batch = [f"{PRE}<Instruct>: {INSTR}\n<Query>: {query}\n<Document>: {d}{SUF}" for d in docs[i:i + bs]]
        enc = tok(batch, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(model.device)
        logits = model(**enc).logits[:, -1, :]
        out.extend(torch.stack([logits[:, NO], logits[:, YES]], 1).float().softmax(1)[:, 1].tolist())
    return out


def table_text(txt, line, maxrows=30):
    """Nội dung bảng dạng text thô cho reranker đọc (không parse, giữ nguyên như trong báo cáo)."""
    for m in re.finditer(r"<table>(.*?)</table>", txt, re.S):
        if txt.count("\n", 0, m.start()) + 1 == line:
            rows = []
            for tr in re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)[:maxrows]:
                c = [re.sub(r"<[^>]*>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                if any(c): rows.append(" | ".join(c))
            return "\n".join(rows)[:1500]
    return ""


def bm25_ranked(txt, query, k):
    tt = P.table_tokens(txt)
    if not tt: return []
    sc = P.bm25_scores(P.toks_seg(query), [t["toks"] for t in tt])
    return [t["line"] for t, _ in sorted(zip(tt, sc), key=lambda x: -x[1])][:k]


cache, t0, n, skip = {}, time.time(), 0, 0
for q in P.Q:
    qt = q["question"]
    tk = P.resolve(qt); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", qt)))
    fr = P.find_report(tk, yrs[-1], P.doctype(qt)) if (tk and yrs) else None
    if not fr:
        skip += 1; continue
    txt = fr[0].read_text(encoding="utf-8", errors="replace")
    query = B.table_query(qt)
    key = f"{fr[1]}|{query}"                     # khoá theo report + cụm chỉ tiêu (không theo id câu)
    if key in cache:
        continue                                  # nhiều câu chung một report + chỉ tiêu → tính một lần
    cand = bm25_ranked(txt, query, TOPK)
    docs = [(ln, table_text(txt, ln)) for ln in cand]
    docs = [(ln, d) for ln, d in docs if d]
    if len(docs) < 2:
        continue
    sc = rerank(f"{qt} — chỉ tiêu: {P.target_label(qt)}", [d for _, d in docs])
    cache[key] = [ln for ln, _ in sorted(zip([d[0] for d in docs], sc), key=lambda x: -x[1])]
    n += 1
    if n % 50 == 0:
        print(f"  {n} khoá · {time.time()-t0:.0f}s", flush=True)

json.dump(cache, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\n{len(cache)} khoá đã ghi vào {OUT} · {time.time()-t0:.0f}s · bỏ qua {skip} câu không tìm được báo cáo")
