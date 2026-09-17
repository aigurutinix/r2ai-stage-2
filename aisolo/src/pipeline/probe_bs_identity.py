"""THAM DO: trich xuat toan kho theo Ma so TT200 va kiem bang DANG THUC KE TOAN.

Cau hoi can tra loi TRUOC khi hop nhat app voi dataset (khong phai de "xac nhan" mot huong da chon):
  A. DO PHU  — moi cap (cong ty, nam) lay duoc bao nhieu trong so cac ma so cot loi?
  B. DANG THUC — co bao nhieu cap thoa 270 = 300 + 400 (Tong tai san = No + Von chu)?
  C. TINH DUY NHAT — khi mot ma so co NHIEU ung vien o cac bang khac nhau, dang thuc co chon ra
     DUNG MOT to hop khong? Day la cau quan trong nhat: neu co, dang thuc la bo chon dong khong
     can nhan, danh thang vao lop loi "nhan trung chu, sai BANG" (67% du dia).

Chay:  python pipeline/probe_bs_identity.py [so_cong_ty]     (mac dinh 10)
Ket qua ghi ra pipeline/probe_bs_identity.json de khong mat khi phien xoay vong.
"""
import os, sys, json, time, itertools, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Mac dinh TUONG DOI theo vi tri script, khong theo o dia may nao ca. Ban truoc ghi cung
# duong dan tuyet doi cua mot may nen tren may khac no tro vao thu muc khong ton tai.
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ["VIFINQA_ROOT"], "financial_statements")

# Ma so cot loi cua Bang can doi ke toan theo TT200 (doanh nghiep phi tai chinh).
CORE = {
    "100": "Tài sản ngắn hạn", "200": "Tài sản dài hạn", "270": "Tổng cộng tài sản",
    "300": "Nợ phải trả", "310": "Nợ ngắn hạn", "330": "Nợ dài hạn",
    "400": "Vốn chủ sở hữu", "440": "Tổng cộng nguồn vốn",
}
# Dang thuc kiem duoc: (ve trai, [cac ve phai cong lai])
IDENTITIES = [("270", ["300", "400"]), ("270", ["100", "200"]), ("300", ["310", "330"])]
REL_TOL = 1e-3          # OCR co the lam tron; 0.1% la du chat de bat sai that


def close(a, b):
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1.0)


def candidates(rows):
    """ma_so -> tap gia tri ung vien (chi lay dong thuoc bang BS, bo trung lap)."""
    out = collections.defaultdict(set)
    for r in rows:
        if r["st"] == "BS" and r["ma"] in CORE and r["cur"] is not None:
            out[r["ma"]].add(round(r["cur"], 2))
    return out


def check(cand):
    """Tra (so_dang_thuc_kiem_duoc, so_dang_thuc_thoa, co_to_hop_duy_nhat_cho_270)."""
    testable = passed = 0
    solutions_270 = set()
    for lhs, rhs in IDENTITIES:
        if lhs not in cand or any(k not in cand for k in rhs):
            continue
        testable += 1
        ok = False
        for lv in cand[lhs]:
            for combo in itertools.product(*(cand[k] for k in rhs)):
                if close(lv, sum(combo)):
                    ok = True
                    if lhs == "270":
                        solutions_270.add(lv)
        passed += ok
    return testable, passed, solutions_270



def patched_stmt_type(tabtext):
    """stmt_type() nhung so khop tren ban DA BO DAU.

    OCR doc "NỢ" thanh "NỘ" o 27% lan xuat hien cum "nợ phải trả" (do tren 10 doanh nghiep:
    454/1654 lan). Ban goc so khop chuoi co dau nen truot, va bang NGUON VON roi xuong nhanh PL
    vi no chua "Lợi nhuận sau thuế chưa phân phối" (ma 421). Bo dau thi mien nhiem ca 5 bien the.
    """
    l = P.strip_vn(tabtext).lower()
    if "luu chuyen tien" in l: return "CF"
    if "tong cong tai san" in l or "tong tai san" in l or "no phai tra" in l: return "BS"
    if "loi nhuan sau thue" in l or "doanh thu thuan" in l or "gia von" in l: return "PL"
    return "NOTE"


def main():
    n_co = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if os.environ.get("PATCH_STMT", "0") == "1":
        P.stmt_type = patched_stmt_type
        print("[CHE DO] stmt_type mien nhiem loi dau OCR")
    tickers = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))[:n_co]
    t0 = time.time()
    recs, n_rep = [], 0
    for tk in tickers:
        for year in sorted(os.listdir(os.path.join(ROOT, tk))):
            for dt in ("consolidated", "separate"):
                fr = P.find_report(tk, year, dt)
                if not fr or fr[1].split("_")[-1] != dt:
                    continue
                txt = fr[0].read_text(encoding="utf-8", errors="replace")
                rows, _uf, _src = P.ingest(txt)
                n_rep += 1
                cand = candidates(rows)
                testable, passed, sol = check(cand)
                # so ma so co NHIEU hon 1 ung vien = do mo ho ma dang thuc phai go
                ambiguous = sum(1 for k in cand if len(cand[k]) > 1)
                recs.append({
                    "ticker": tk, "year": year, "doctype": dt, "doc": fr[1],
                    "coverage": len(cand), "core_total": len(CORE),
                    "ambiguous_codes": ambiguous,
                    "cand_270": len(cand.get("270", ())),
                    "testable": testable, "passed": passed,
                    "unique_270": (len(sol) == 1),
                })
    dt_s = time.time() - t0

    # ---- tong hop ----
    N = len(recs)
    full = [r for r in recs if r["coverage"] == r["core_total"]]
    has3 = [r for r in recs if r["coverage"] >= 3]
    testable = [r for r in recs if r["testable"] > 0]
    allpass = [r for r in testable if r["passed"] == r["testable"]]
    anypass = [r for r in testable if r["passed"] > 0]
    amb = [r for r in testable if r["cand_270"] > 1]
    amb_solved = [r for r in amb if r["unique_270"]]

    pc = lambda a, b: f"{100*len(a)/b:5.1f}%" if b else "  n/a"
    print(f"=== {len(tickers)} doanh nghiep | {n_rep} bao cao | {dt_s:.1f}s "
          f"({dt_s/max(n_rep,1):.2f}s/bao cao) ===\n")
    print(f"A. DO PHU (tren {N} cap cong ty-nam-loai)")
    print(f"   lay du ca {len(CORE)} ma so cot loi : {len(full):5d}  {pc(full, N)}")
    print(f"   lay duoc >= 3 ma so              : {len(has3):5d}  {pc(has3, N)}")
    print(f"\nB. DANG THUC")
    print(f"   kiem duoc it nhat 1 dang thuc    : {len(testable):5d}  {pc(testable, N)}")
    print(f"   THOA het moi dang thuc kiem duoc : {len(allpass):5d}  {pc(allpass, N)}"
          f"   (tren so kiem duoc: {pc(allpass, len(testable))})")
    print(f"   thoa it nhat 1 dang thuc         : {len(anypass):5d}  {pc(anypass, len(testable))}")
    print(f"\nC. DANG THUC CO GO DUOC MO HO KHONG")
    print(f"   ca co >1 ung vien cho ma 270     : {len(amb):5d}  {pc(amb, len(testable))}")
    print(f"   trong do dang thuc chot DUY NHAT : {len(amb_solved):5d}  {pc(amb_solved, len(amb))}")

    tag = "_patched" if os.environ.get("PATCH_STMT", "0") == "1" else "_base"
    out = os.path.join(HERE, f"probe_bs_identity{tag}.json")
    json.dump({"n_tickers": len(tickers), "n_reports": n_rep, "seconds": dt_s, "records": recs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nChi tiet: {out}")


if __name__ == "__main__":
    main()
