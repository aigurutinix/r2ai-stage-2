"""KET QUA AM — GIU LAI DE KHONG AI THU LAI HUONG NAY.

Gia thuyet (22/08): khau chon dong cua pipeline dung 67,6%, lop loi lon nhat la "nhan trung chu,
sai BANG" (67% du dia). Ky thuat trong build_corpus.py go duoc lop do bang hai thu locate() khong
dung — khop CA ma so LAN ten, va lay DANG THUC KE TOAN lam bo chon. Neu ap vao bai nop thi co the
sua duoc mot phan.

DO DUOC — gia thuyet SAI:
  78/1012 cau co dich thuoc 29 chi tieu chuan TT200 (6,1%). Cam bang tra vao build_submission,
  no chot lai dong cho 59 cau. Ket qua tren bai nop: 0 dap an doi, 0 pandas_query doi, 0 evidence
  doi, 0 relevant_tables doi. Tren dev set 312 cau: 209 dung TRUOC va 209 dung SAU, 0 cau doi.
  Tuc voi chi tieu CHUAN TT200, locate() VON DA chon dung san — ky thuat moi dong y 100%.

  Va khong the mo rong: 934 cau con lai co 785 dich KHAC NHAU, 724 trong so do chi xuat hien DUNG
  MOT LAN ("Chi phi xay dung co ban do dang", "Tien tra truoc cho nguoi ban ngan han"). Do la duoi
  dai khoan muc thuyet minh, khong bang tra nao phu noi.

KET LUAN: nghen 67,6% nam HOAN TOAN o khoan muc phi chuan, khong o chi tieu chuan. Moi huong dua
tren bang tra Ma so TT200 deu se cho ket qua nhu the nay. Day la tin hieu thu 10 duoc do va la tin
hieu thu 10 khong dich chuyen diem.

Script van chay duoc de tai lap ket qua tren. Dau ra tt200_rowpick.json KHONG commit (2,1 MB,
khong ai dung).

Chay:  python pipeline/build_rowpick.py
"""
import os, sys, json, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Mac dinh tro vao data_vifinqa/ CANH repo, khong phai duong dan may ca nhan — nguoi khac
# clone ve va lam theo README la chay duoc ngay. Dat VIFINQA_ROOT de tro noi khac.
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P
import build_corpus as B          # doc registry.ts + logic khop hai khoa; CHI dung luc sinh tep

ROOT = os.path.join(os.environ["VIFINQA_ROOT"], "financial_statements")
OUT = os.path.join(HERE, "tt200_rowpick.json")


def main():
    tks = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    t0, out = time.time(), {}
    stat = collections.Counter()
    for tk in tks:
        for year in sorted(os.listdir(os.path.join(ROOT, tk))):
            for dt in ("consolidated", "separate"):
                fr = P.find_report(tk, year, dt)
                if not fr or not fr[1].endswith(dt):
                    continue
                rows, _uf, _src = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))
                cells, _how = B.collect(rows)
                stat["bao_cao"] += 1
                for ma, c in cells.items():
                    # tid + gia tri la du de build_submission dinh vi lai dung dong trong `rows`
                    out[f"{tk}|{year}|{dt}|{ma}"] = {"tid": c["tid"], "v": c["v"]}
                    stat["o"] += 1
    # Anh xa CAU HOI -> Ma so, tinh luon o day. Nho vay build_submission chi tra theo id, khong
    # phai mang theo logic chuan hoa ten (boc tien to danh so, hau to cong thuc, bo dau) va khong
    # phu thuoc vao r2ai-app/ luc dung bai nop.
    spec = {r["ma"]: r for r in B.REGISTRY}
    qcode = {}
    for q in P.Q:
        key = B.label_key(P.target_label(q["question"]))
        ma = next((m for m, sp in spec.items() if key in sp["keys"]), None)
        if ma:
            qcode[str(q["id"])] = ma
    stat["cau_co_ma"] = len(qcode)
    json.dump({"pick": out, "qcode": qcode},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    mb = os.path.getsize(OUT) / 1e6
    print(f"{stat['bao_cao']} bao cao | {stat['o']} o | {stat['cau_co_ma']} cau co ma so | "
          f"{time.time()-t0:.0f}s | {mb:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
