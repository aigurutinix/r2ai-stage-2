"""Tra CUU TRUC TIEP mot chi tieu tren bao cao OCR goc — phuc vu nhanh "chua kiem chung" cua app.

Vi sao ton tai song song voi build_corpus.py:
  build_corpus tien tinh 30 chi tieu registry, co DANG THUC KE TOAN xac nhan => dang tin, nhung
  hep. Bao cao goc co ~650 dong/bao cao; cau hoi that hay roi vao phan con lai ("Lai tien gui",
  "So du cho vay nganh Thuong mai", "Chi phi xay dung co ban do dang").
  Script nay tra THANG tren bao cao goc nen phu HET, nhung dung khau chon dong lexical cua
  pipeline — do duoc 67,6% moi lan tra. KHONG co gi xac nhan.

=> App phai NOI RO hai tang do khac nhau. Ket qua tra o day luon mang cо `verified=false`.

Chay:
  python pipeline/lookup_live.py --ticker HPG --year 2024 --query "Lai tien gui"
  python pipeline/lookup_live.py --ticker VJC --year 2018 --doctype separate --query "..."
"""
import os, sys, json, argparse

# Console Windows mac dinh la cp1252, khong ma hoa noi tieng Viet co dau: chay dung lenh ghi o
# docstring tren mot console sach thi script chet o dong in ket qua bang UnicodeEncodeError.
# App goi qua spawn co dat PYTHONIOENCODING=utf-8 nen duong do khong sao — day la de chay tay.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # stream bi thay the / khong ho tro
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P


def lookup(ticker, year, query, doctype="consolidated"):
    """Tra mot chi tieu. Tra dict co `ok`; loi thi kem `error` doc duoc, khong nem exception."""
    fr = P.find_report(ticker, str(year), doctype)
    if not fr:
        fr = P.find_report(ticker, str(year), "separate" if doctype == "consolidated" else "consolidated")
    if not fr:
        return {"ok": False, "error": f"Không có báo cáo {ticker} năm {year}"}
    rep, doc = fr
    rows, unit, _src = P.ingest(rep.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        return {"ok": False, "error": f"Không đọc được bảng nào trong {doc}"}

    qdir = P.qdir_of(query)
    loc = P.locate(rows, P.clean_metric(query), qdir)
    if not loc or loc.get("cur") is None or not loc.get("label"):
        return {"ok": False, "error": f"Không tìm thấy chỉ tiêu «{query}» trong {doc}"}

    # CONG CHAN: locate() KHONG BAO GIO noi "khong tim thay" — no luon tra dong khop nhat, ke ca khi
    # cau hoi chang lien quan gi. Do duoc tren HPG 2024: truy van RAC van ra ket qua tu tin —
    # "Khong ton tai gi ca" -> "Hang ton kho" (ov 0,25), "thoi tiet hom nay" -> "- LNST nam nay"
    # (0,29), "gia vang SJC" -> "Nguyen gia" (0,40). Trong khi truy van THAT hoac khop qua Ma so
    # (ov=None, maso co gia tri) hoac co ov >= 0,91. Nguong 0,6 nam giua hai vung.
    # Day la nguong dat tu 9 vi du soi tay, khong phai tu mot phep do lon — nhung tha bo qua con
    # hon tra loi tu tin mot con so bia ra.
    if not loc.get("maso") and (loc.get("ov") is None or float(loc["ov"]) < 0.6):
        return {"ok": False,
                "error": f"Không đủ tin cậy để trả lời «{query}» — chỉ tiêu gần nhất tìm được là "
                         f"«{loc['label']}», độ khớp quá thấp."}

    val = P.pick_value(loc, qdir)
    try:
        val = None if val is None else float(val)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"Giá trị đọc được không phải số: {val!r}"}

    # locate() tra ve dict co cac khoa: cur/prev/vals/hdrs/label/maso/ov/tid — KHONG co so dong.
    # Lay so dong tu chinh `rows` theo tid, giong cach build_submission.py lam, de trich dan
    # duoc "bao cao nao, dong bao nhieu".
    tid = str(loc.get("tid"))
    line = next((r["line"] for r in rows if str(r["tid"]) == tid), None)
    return {
        "ok": True,
        "value": val,
        "label": loc["label"],
        "maSo": loc.get("maso") or None,
        "doc": doc,
        "table": tid,
        "line": line,
        "unitFactor": unit,
        # Khong co dang thuc nao xac nhan gia tri nay — app PHAI hien thi dung nhu vay.
        "verified": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--year", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--doctype", default="consolidated", choices=["consolidated", "separate"])
    a = ap.parse_args()
    try:
        out = lookup(a.ticker, a.year, a.query, a.doctype)
    except Exception as e:                       # route goi qua stdout => khong duoc nem stack
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
