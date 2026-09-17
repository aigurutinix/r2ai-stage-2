"""VÒNG PLANNER → EXECUTOR → AUDITOR: chạy lại những câu Auditor đã CHỨNG MINH là sai.

Ý tưởng lấy từ Model Collaboration của BTC (slide 41): tách vai trò, và thêm một Auditor kiểm lại
trước khi chốt. Khác BTC ở chỗ Auditor của ta **thuần deterministic** — nó suy miền giá trị hợp lệ
từ chính câu hỏi (xem audit_wrong.py) nên không cần model lớn và không bao giờ tự lừa mình.

Vòng lặp: đáp án hiện tại bị Auditor bác → nạp lại cho Planner KÈM RÀNG BUỘC miền giá trị →
Executor chạy → Auditor kiểm lần nữa → chỉ nhận nếu đạt.

Cận dưới bằng 0: chỉ đụng câu ĐÃ chứng minh sai, nên thay bằng gì cũng không thể tệ hơn.

Chạy: python agent_audit.py            (resume được; ghi agent_audit.json)
      AUDIT_IDS=1,2,3 python agent_audit.py    (chạy riêng vài câu để thử)
"""
import os, re, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
os.environ.setdefault("AGENT_V2", "1")
import pipeline as P
import agent_strands as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("AUDIT_OUT", os.path.join(HERE, "agent_audit.json"))
NVOTE = int(os.environ.get("AGENT_VOTES", "3"))

# vá cùng bug với audit_wrong.py: `t[ỉi] tr[ọo]ng` khớp "tỉ trọng" nhưng KHÔNG khớp "tỷ trọng"
PCT = re.compile(r"phần trăm|\bt[ỷyỉi] (?:l[ệe]|tr[ọo]ng)\b|\b%", re.I)
MON = re.compile(r"tỷ đồng|triệu đồng|nghìn đồng|ngàn đồng|\bvnđ\b|\bđồng\b", re.I)


def domain(qt):
    """Miền giá trị hợp lệ suy từ câu hỏi → (mô tả cho Planner, hàm kiểm tra).

    Mô tả CỐ TÌNH viết bằng chữ, tránh chữ số 4 ký tự — chuỗi này được nối vào câu hỏi trước khi
    đưa Planner, mà `load_scope`/`doctype` vẫn dò năm và loại báo cáo bằng regex trên chuỗi đó.
    """
    l = qt.lower()
    if re.search(r"\bnăm nào\b", l):
        return ("một con số NĂM (bốn chữ số, trong khoảng hai nghìn mười lăm tới hai nghìn hai lăm)",
                lambda v: 2014 <= v <= 2026)
    if re.search(r"(bao nhiêu|có mấy)\s+(doanh nghiệp|công ty|ngân hàng|mã)", l):
        return ("một SỐ NGUYÊN đếm số doanh nghiệp, không âm và không quá ba mươi",
                lambda v: 0 <= v <= 30 and abs(v - round(v)) < 1e-6)
    if PCT.search(qt):
        return ("một TỶ LỆ PHẦN TRĂM, giá trị tuyệt đối không quá ba trăm",
                lambda v: -300 <= v <= 300)
    if re.search(r"bao nhiêu lần\b", l):
        return ("một HỆ SỐ tính bằng lần, giá trị tuyệt đối không quá hai trăm",
                lambda v: -200 <= v <= 200)
    if MON.search(qt):
        import build_submission as B
        qf = B.q_unit(qt)
        return ("một SỐ TIỀN đã quy về đúng đơn vị câu hỏi yêu cầu (chú ý chia đúng hệ số đơn vị)",
                lambda v: not (abs(v) * qf > 2e16 or 0 < abs(v) * qf < 1e5))
    return (None, None)


def main():
    ids = ([int(x) for x in os.environ["AUDIT_IDS"].split(",")] if os.environ.get("AUDIT_IDS")
           else json.load(open(os.path.join(HERE, "audit_retry_ids.json"))))
    QT = {q["id"]: q["question"] for q in P.Q}
    done = {}
    if os.path.exists(OUT):
        done = {d["id"]: d for d in json.load(open(OUT, encoding="utf-8"))}
    model = A.build_model()
    t0, n = time.time(), 0
    for qid in ids:
        if qid in done:
            continue
        qt = QT[qid]
        desc, ok = domain(qt)
        if desc is None:                       # Auditor không phát biểu được ràng buộc → bỏ qua
            done[qid] = {"id": qid, "answer": None, "pandas": None, "refs": [], "why": "khong co rang buoc"}
            continue
        # RÀNG BUỘC nối vào câu hỏi: đây chính là phản hồi của Auditor cho Planner
        hint = f"{qt} [RÀNG BUỘC: đáp án cuối cùng phải là {desc}. Lần trước hệ thống trả sai miền này.]"
        try:
            r = A.run_voted(hint, model, NVOTE, runner=A.run_question_v2)
        except Exception as e:
            r = {"answer": None, "pandas": None, "refs": [], "votes": 0, "expr": f"ERR {type(e).__name__}"}
        v = r.get("answer")
        passed = False
        if v is not None:
            try:
                passed = bool(ok(float(v)))
            except Exception:
                passed = False
        done[qid] = {"id": qid, "question": qt, "answer": (v if passed else None),
                     "pandas": (r.get("pandas") if passed else None),
                     "refs": (r.get("refs") if passed else []),
                     "votes": r.get("votes", 0), "n": r.get("n", 0),
                     "raw_answer": v, "why": ("dat" if passed else "auditor van bac")}
        json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        n += 1
        okn = sum(1 for d in done.values() if d["answer"] is not None)
        print(f"id{qid} [{time.time()-t0:.0f}s] raw={v} -> {'NHAN' if passed else 'bac'} | tong nhan {okn}", flush=True)
    okn = sum(1 for d in done.values() if d["answer"] is not None)
    print(f"XONG {len(done)}/{len(ids)} | Auditor CHAP NHAN {okn} cau | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
