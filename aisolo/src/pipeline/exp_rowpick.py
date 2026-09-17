"""THÍ NGHIỆM: LLM chọn DÒNG chỉ tiêu có hơn so khớp từ vựng không?

Bối cảnh: đo theo tầng cho thấy nghẽn ANSWER nằm ở khâu chọn dòng (~46% đúng), trong khi
chọn báo cáo 89% / parse OCR 97.3% / gán cột 87.2% đều tốt. Qwen3.5-4B đã thử và KHÔNG hơn
lexical (8/15 trùng, thắng-thua xen kẽ). Câu hỏi còn lại: model mạnh hơn (9B) có hơn không?

Đo bằng CONTAINMENT — tín hiệu ĐỘC LẬP với hàm điểm lexical: nhãn chọn ra có chứa trọn cụm
đích (hoặc ngược lại) không. Đây chính là thước đo đã dùng khi sửa locate (60.3% -> 68.6%)
và nó đã tương quan đúng với cải thiện thật trên leaderboard.

Mẫu phân tầng: một nửa lấy từ nhóm lexical khớp KÉM (jaccard < 0.5 — nơi lexical gần chắc
sai, LLM có cơ hội cứu), một nửa lấy ngẫu nhiên (để phát hiện LLM có làm HỎNG câu đang đúng).

Chạy:
  # baseline 4B local
  LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  LLM_API_KEY=ollama python exp_rowpick.py

  # 9B trên Modal
  LLM_BASE_URL=https://<workspace>--r2ai-llm-serve.modal.run/v1 \
  LLM_MODEL=QuantTrio/Qwen3.5-9B-AWQ LLM_API_KEY=$LLM_API_KEY LLM_NO_THINK=1 \
  python exp_rowpick.py

Env phụ: EXP_N (số câu, mặc định 40), EXP_K (số ứng viên đưa cho LLM, mặc định 30).
"""
import os, re, json, random
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
# Qwen3.5 là model THINKING: nếu để nó suy luận thì hết max_tokens trước khi ra nội dung
# (smoke test 4B: 3/6 lần gọi trả rỗng). Tác vụ này chỉ cần MỘT số nguyên → tắt thinking + cắt ngắn.
os.environ.setdefault("LLM_NO_THINK", "1")
os.environ.setdefault("LLM_MAX_TOKENS", "64")
import pipeline as P
import build_submission as B
import llm_engine as L

N = int(os.environ.get("EXP_N", "40"))
K = int(os.environ.get("EXP_K", "30"))

SYS = (
    "Bạn chọn ĐÚNG MỘT dòng chỉ tiêu trong báo cáo tài chính khớp với câu hỏi.\n"
    "Chú ý sắc thái nghiệp vụ — chọn sai sắc thái là sai hoàn toàn:\n"
    "  'phải nộp' ≠ 'đã nộp' · 'nguyên giá' ≠ 'giá trị còn lại' · 'ngắn hạn' ≠ 'dài hạn'\n"
    "  'giá trị ghi sổ/gộp' ≠ 'dự phòng' · 'số dư' ≠ 'phát sinh trong kỳ'\n"
    "Trả về DUY NHẤT một số nguyên là chỉ số dòng. KHÔNG giải thích, KHÔNG chữ nào khác."
)

def candidates(rows, target, question, k=K):
    """Ứng viên = HỢP của top-k theo target_label VÀ top-k theo CẢ CÂU HỎI.
    Lý do: target_label đôi khi trích hỏng (ra nguyên câu hỏi) → nếu chỉ dựng rổ bằng nó thì
    dòng đúng có thể KHÔNG nằm trong rổ, và ta sẽ đo nhầm 'LLM cứu rổ tồi' thay vì 'LLM chọn giỏi'."""
    uniq, seen = [], set()
    for r in rows:
        if r["cur"] is None or not r["label"]:
            continue
        key = (r["label"], r["cur"])
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
    if not uniq:
        return []
    labs = [P.toks_list(r["label"]) for r in uniq]
    out, got = [], set()
    for q in (target, question):
        sc = P.bm25_scores(P.toks_list(q), labs)
        for r, _ in sorted(zip(uniq, sc), key=lambda x: -x[1])[:k]:
            key = (r["label"], r["cur"])
            if key not in got:
                got.add(key); out.append(r)
    return out[:k * 2]

def llm_pick(qt, cands):
    lines = "\n".join(f"[{i}] {r['label'][:70]} = {int(r['cur']):,}" for i, r in enumerate(cands))
    prompt = f"{SYS}\n\nCâu hỏi: {qt}\n\nCác dòng:\n{lines}\n\nChỉ số:"
    try:
        out = L._post_chat(prompt)
    except Exception as e:
        return None, f"ERR {type(e).__name__}: {str(e)[:70]}"
    txt = (out if isinstance(out, str) else str(out)).strip()
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if not nums:
        return None, f"no-index (model trả: {txt[:50]!r})"
    # Lấy số HỢP LỆ ĐẦU TIÊN trong [0, len). Trước đây lấy số CUỐI → nhặt trúng NĂM khi model
    # trả kèm ngữ cảnh ("dòng 12 ... năm 2024") → 4/10 câu bị vứt oan thành 'lỗi model'.
    for i in nums:
        if 0 <= i < len(cands):
            return cands[i], ""
    return None, f"khong co chi so hop le trong {nums[:4]} (rổ {len(cands)} dòng)"

def contain(t, l):
    a, b = P.strip_vn(t), P.strip_vn(l)
    return bool(a) and bool(b) and (a in b or b in a)

def jac(t, l):
    tt, lt = P.toks(t), P.toks(l)
    return len(tt & lt) / max(1, len(tt | lt))

def n_tick(q):
    return len(set(m for m in re.findall(r"\b([A-Z0-9]{2,4})\b", q) if m in P.tickers))

def main():
    money = [q for q in P.Q
             if re.search(r"tỷ đồng|triệu đồng|nghìn tỷ|trăm tỷ", q["question"], re.I)
             and n_tick(q["question"]) <= 1 and not B.GROUPY_RE.search(q["question"])
             and not any(k in q["question"].lower() for k in P.AGG + P.COND)]
    # PHÂN TẦNG THẬT: nửa mẫu lấy từ nhóm lexical khớp KÉM (jaccard < 0.5 — nơi lexical gần chắc
    # sai, LLM có cơ hội cứu), nửa còn lại lấy từ nhóm lexical khớp TỐT (để phát hiện LLM có phá
    # câu đang đúng không). Chỉ đo một nửa "kém" sẽ thổi phồng lợi ích của LLM.
    random.seed(13); random.shuffle(money)
    weak, strong = [], []
    for q in money:
        if len(weak) >= N // 2 and len(strong) >= N - N // 2:
            break
        qt = q["question"]; tk = P.resolve(qt); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", qt)))
        fr = P.find_report(tk, yrs[-1], P.doctype(qt)) if (tk and yrs) else None
        if not fr:
            continue
        rows = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
        loc = P.locate(rows, P.target_label(qt), P.qdir_of(qt))
        if not loc:
            continue
        bucket = weak if jac(P.target_label(qt), loc["label"]) < 0.5 else strong
        cap = N // 2 if bucket is weak else N - N // 2
        if len(bucket) < cap:
            bucket.append((q, fr))
    print(f"mẫu: {len(weak)} câu lexical KÉM (jac<0.5) + {len(strong)} câu lexical TỐT")
    weak = weak + strong

    picked, errs, stats = [], [], {"lex_contain": 0, "llm_contain": 0, "same": 0, "n": 0,
                         "llm_fix": 0, "llm_break": 0, "err": 0}
    for q, fr in weak:
        if stats["n"] >= N:
            break
        qt = q["question"]
        rows = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
        tgt = P.target_label(qt)
        lex = P.locate(rows, tgt, P.qdir_of(qt))
        if not lex:
            continue
        cands = candidates(rows, tgt, qt)
        if len(cands) < 2:
            continue
        pick, err = llm_pick(qt, cands)
        if err:
            stats["err"] += 1
            if len(errs) < 5:
                errs.append((q["id"], err))
        stats["n"] += 1
        cl = contain(tgt, lex["label"])
        cp = contain(tgt, pick["label"]) if pick else False
        stats["lex_contain"] += cl; stats["llm_contain"] += cp
        same = pick and pick["label"] == lex["label"]
        stats["same"] += bool(same)
        if not same:
            if cp and not cl: stats["llm_fix"] += 1
            if cl and not cp: stats["llm_break"] += 1
            picked.append((q["id"], tgt, lex["label"], pick["label"] if pick else None,
                           round(jac(tgt, lex["label"]), 2),
                           round(jac(tgt, pick["label"]), 2) if pick else None, cl, cp))

    n = max(1, stats["n"])
    print(f"\n=== LLM chọn dòng vs LEXICAL — {stats['n']} câu ===")
    print(f"  model: {L.LLM_MODEL}  @ {L.LLM_BASE}")
    print(f"  trùng nhau           : {stats['same']}/{n} ({stats['same']/n*100:.0f}%)")
    print(f"  containment LEXICAL  : {stats['lex_contain']}/{n} = {stats['lex_contain']/n*100:.1f}%")
    print(f"  containment LLM      : {stats['llm_contain']}/{n} = {stats['llm_contain']/n*100:.1f}%  <== tín hiệu chính")
    print(f"  LLM CỨU được         : {stats['llm_fix']}   |  LLM LÀM HỎNG: {stats['llm_break']}")
    print(f"  lỗi gọi model        : {stats['err']}")
    for i, e in errs:
        print(f"      id{i}: {e}")
    print("\n  ví dụ khác nhau (id | target | lexical | llm):")
    for i, t, l, p, jl, jp, cl, cp in picked[:12]:
        mark = "LLM tốt hơn" if (cp and not cl) else ("LEX tốt hơn" if (cl and not cp) else "~")
        print(f"    id{i} [{mark}]\n       target = {t[:52]!r}\n       lex    = {l[:52]!r} (jac {jl})\n       llm    = {(p or '')[:52]!r} (jac {jp})")

if __name__ == "__main__":
    main()
