"""LẤY MẪU CÁC CA PIPELINE VÀ TRỌNG TÀI ĐỒNG Ý → để người đọc báo cáo gốc phân xử.

Vì sao cần: dev set chỉ đo "pipeline khớp trọng tài", KHÔNG đo "pipeline đúng". Trọng tài
gemma4:31b và pipeline có thể cùng bị hút về một nhãn nổi bật rồi cùng sai. Trước nay mới kiểm tay
2 ca BẤT ĐỒNG, chưa bao giờ kiểm ca ĐỒNG Ý — nên con số 68% chưa được xác nhận là thật.

In ra: câu hỏi, dòng pipeline chọn, và TOÀN BỘ dòng thô quanh đó trong bảng gốc để đối chiếu.
"""
import os, re, sys, json, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

N = int(os.environ.get("AGREE_N", "12"))
HERE = os.path.dirname(os.path.abspath(__file__))
D = [d for d in json.load(open(os.path.join(HERE, "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None and d["agree"]]
random.seed(7)
sample = random.sample(D, min(N, len(D)))
print(f"Tong ca DONG Y: {len(D)} | lay mau {len(sample)}\n")

for d in sample:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    txt = open(str(fr[0]), encoding="utf-8", errors="replace").read()
    rows, uf, src = P.ingest(txt)
    loc = P.locate(rows, P.target_label(d["question"]), P.qdir_of(d["question"]))
    if not loc:
        continue
    print("=" * 100)
    print(f"id{d['id']} | {d['tk']} {d['nam']} {d['doctype']} | don vi bao cao x{int(uf)} ({src})")
    print(f"HOI    : {d['question']}")
    print(f"target : {P.target_label(d['question'])!r}   qdir={P.qdir_of(d['question'])}")
    print(f"CHON   : nhan={loc['label']!r} ma={loc['maso']!r} tid={loc['tid']} ov={loc['ov']}")
    print(f"GIA TRI: cur={loc['cur']} prev={loc['prev']} -> nop {P.pick_value(loc, P.qdir_of(d['question']))}")
    # bảng gốc: header + dòng được chọn + lân cận
    tabs = [m.group(1) for m in re.finditer(r"<table>(.*?)</table>", txt, re.S)]
    if loc["tid"] < len(tabs):
        trows = [P.cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>", tabs[loc["tid"]], re.S)]
        hit = next((j for j, r in enumerate(trows) if r and any(c == loc["label"] for c in r)), None)
        print(f"--- BANG GOC tid={loc['tid']} ({len(trows)} dong), dong khop = {hit} ---")
        rng = range(max(0, (hit or 0) - 4), min(len(trows), (hit or 0) + 5)) if hit is not None else range(min(8, len(trows)))
        for j in rng:
            r = trows[j]
            if not r:
                continue
            mark = " <<<< PIPELINE CHON" if j == hit else ""
            print(f"   r{j:<3} {[c[:34] for c in r[:7]]}{mark}")
    print()
