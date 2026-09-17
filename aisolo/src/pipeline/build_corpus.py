"""Trich xuat TOAN KHO thanh bang phang theo Ma so TT200, co kiem dang thuc ke toan.

Vi sao lam offline thay vi tra luc hoi: tra luc hoi = thua ke khau chon dong 67.6% cua pipeline
thi, tuc demo sai mot phan ba so cau. Tien tinh mot lan thi sai sot van con nhung xay ra NOI TA DO
VA SUA DUOC, con luc demo chi la mot phep tra khoa chinh.

Dau ra: <out>/<TICKER>.json — mot tep moi doanh nghiep, hinh dang anh xa thang sang TidyTable cua
app (byTicker[ticker][maSo].values[period]). Kem xuat xu tung o va co "da qua kiem can doi".

Chay:  python pipeline/build_corpus.py [--out THU_MUC] [--limit N]
"""
import os, re, sys, csv, json, time, argparse, collections, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Mac dinh tro vao data_vifinqa/ CANH repo, khong phai duong dan may ca nhan — nguoi khac
# clone ve va lam theo README la chay duoc ngay. Dat VIFINQA_ROOT de tro noi khac.
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

ROOT = os.path.join(os.environ["VIFINQA_ROOT"], "financial_statements")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r2ai-app", "public", "corpus")

# Dang thuc kiem duoc tu chinh Bang can doi (TT200) — khong can nhan, khong can model.
IDENTITIES = [("270", ["300", "400"]), ("270", ["100", "200"]), ("300", ["310", "330"])]
REL_TOL = 1e-3

# Vi sao KHONG do "co phai ngan hang khong" bang tu khoa: da thu va no gan nham 5 doanh nghiep
# (FTS, GVR, MBS, SSI, VIC) chi vi bao cao co nhac "cho vay khach hang" — trong khi ca 5 dung TT200
# va qua kiem can doi 9-11/11 nam. To chuc tin dung that (dung he tai khoan TT49/210) tu bieu lo
# bang viec KHONG dang thuc nao thoa. Nen pham vi ap dung suy TU KET QUA DO, khong tu phong doan.

REGISTRY_TS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "r2ai-app", "lib", "financial", "registry.ts")


def norm_key(x):
    """Ban Python cua normKey() trong registry.ts — PHAI khop tung buoc voi ban TS."""
    x = unicodedata.normalize("NFD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.replace("đ", "d").replace("Đ", "d").lower()
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


ORDINAL = re.compile(r"^\s*(?:[0-9]{1,2}|[IVXivx]{1,4}|[A-Ea-e])\s*[.)]\s*")
# Hau to cong thuc bao cao hay in kem: "TONG CONG TAI SAN (270=100+200)". Bat tu dau "(NN=" den
# HET chuoi, vi OCR hay cat cut ngoac: "Loi nhuan thuan tu HDKD (30=20+(21-22)-(" — dang khong can
# nay lam mau "\(...\)$" truot.
FORMULA_TAIL = re.compile(r"\s*\(\s*\d{1,3}\s*=.*$")
# Chu thich phan trang cua ban in, khong thuoc ten chi tieu: "Loi nhuan sau thue TNDN (mang sang
# trang sau)". Boc SAU khi chuan hoa nen khong con dau/ngoac de khop.
PAGE_NOTE = re.compile(r"\s*mang (?:sang trang sau|tu trang truoc sang)\s*$")
# Tien to danh so lan hai, lam SAU khi chuan hoa. Boc trên chuoi tho doi dau phan cach ngay sau
# ky tu ("A." / "1)"), ma OCR khong phai luc nao cung giu — "A TAI SAN NGAN HAN" thi truot. Sau
# chuan hoa thi moi dau phan cach da thanh khoang trang nen mien nhiem het. Do duoc: mat 30 dong
# ma 100, 39 dong ma 200, 25 dong ma 300, 20 dong ma 400 chi vi mot chu cai con sot.
ORDINAL_NORM = re.compile(r"^(?:[a-e]|[ivx]{1,4}|\d{1,2})\s+")


def label_key(lab):
    """norm_key() sau khi BOC TIEN TO DANH SO cua bao cao ("1.", "I.", "C.").

    Nhan trong bao cao that gan nhu luon co tien to: "1. Doanh thu ban hang...", "I. No ngan han",
    "C. NO PHAI TRA". Registry thi ghi ten tran. Khong boc thi khop-bang-nhau truot gan het (do
    duoc: AAA chi con 13/29 chi tieu). Day van la khop BANG NHAU — chi bo mot phan dinh dang da
    biet truoc, khong phai noi thanh khop mo.
    """
    lab = FORMULA_TAIL.sub("", lab)
    prev = None
    while prev != lab:
        prev, lab = lab, ORDINAL.sub("", lab)
    k = PAGE_NOTE.sub("", norm_key(lab)).strip()
    prev = None
    while prev != k:
        prev, k = k, ORDINAL_NORM.sub("", k)
    return k


def load_registry():
    """Doc chi tieu tu registry.ts — MOT nguon su that duy nhat voi app.

    Hardcode lai 29 ma so ben Python nghia la hai ban se troi khoi nhau lang le. Doc thang tu file
    TS thi sua registry mot cho la ca hai noi doi theo.
    """
    src = open(REGISTRY_TS, encoding="utf-8").read()
    out = []
    for m in re.finditer(r'\{[^{}]*?maSo:\s*"(\d+)"[^{}]*?\}', src):
        blk, ma = m.group(0), m.group(1)
        lab = re.search(r'label:\s*"([^"]*)"', blk)
        st = re.search(r'statementType:\s*"(BS|PL|CF)"', blk)
        al = re.search(r'aliases:\s*\[([^\]]*)\]', blk)
        if not (lab and st):
            continue
        names = [lab.group(1)] + (re.findall(r'"([^"]*)"', al.group(1)) if al else [])
        out.append({"ma": ma, "st": st.group(1), "label": lab.group(1),
                    "keys": {norm_key(n) for n in names if norm_key(n)}})
    return out


REGISTRY = load_registry()


def close(a, b):
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1.0)


def matched_rows(rows, spec):
    """Moi dong khop CA ma so LAN ten cho mot chi tieu registry.

    Chi khop ma so thoi la khong du — TT200 cho bang luu chuyen tien dung lai dai 01-70 voi nghia
    khac han (ma 01 = "Doanh thu ban hang" o KQKD nhung = "Loi nhuan truoc thue" o LCTT), va bang
    bien dong von chu dung lai 400/411/421. Da do: gop phang chi theo ma so lam AAA 2016 ra 166 ty
    thay vi 2.145 ty.

    CHI khop BANG NHAU tuyet doi (sau khi boc tien to/hau to dinh dang da biet). Ke hoach truoc do
    da do hai chien luoc tren 38 ten that: "chua/dai nhat thang" -> 22 dung nhung 9 SAI IM LANG;
    "bang nhau tuyet doi" -> 20 dung, 0 sai. Mot anh xa sai la mot con so sai tren san khau ma
    khong ai biet, nen tha bo qua con hon doan.
    """
    return [r for r in rows if r["ma"] == spec["ma"] and r["cur"] is not None and r["label"]
            and label_key(r["label"]) in spec["keys"]]


# Dang thuc KQKD. Bao cao in khoan KHAU TRU trong ngoac (= so am) khong nhat quan giua cac cong
# ty: AAA ghi gia von -8.215 ty, HPG ghi +... Nen voi khoan KHONG THE am that (gia von, chi phi
# thue hien hanh) ta lay DO LON; con thue TNDN hoan lai thi GIU DAU vi no co the la khoan LOI
# (HPG 2019: -84,9 ty) va dau o do mang nghia. Cung quy uoc voi build_submission.py.
PL_IDENTITIES = [("20", ["10"], ["11"], []), ("60", ["50"], ["51"], ["52"])]


def _score(vals):
    """So dang thuc ke toan THOA duoc voi mot bo gia tri. Dung lam ham chon, khong phai de bao cao."""
    n = 0
    for lhs, rhs in IDENTITIES:
        if lhs in vals and all(k in vals for k in rhs):
            n += close(vals[lhs], sum(vals[k] for k in rhs))
    for lhs, base, deduct_abs, deduct_signed in PL_IDENTITIES:
        if lhs not in vals or any(k not in vals for k in base + deduct_abs):
            continue
        v = sum(vals[k] for k in base)
        v -= sum(abs(vals[k]) for k in deduct_abs)
        v -= sum(vals[k] for k in deduct_signed if k in vals)
        n += close(vals[lhs], v)
    return n


def collect(rows):
    """ma_so -> cell, lay tu MOT ban bao cao nhat quan chu khong ghep tu nhieu ban.

    Mot tep bao cao thuong chua CA ban rieng lan ban hop nhat — VGC 2018 co ma 01 o hai bang:
    8.816 ty va 9.204 ty. Chon tung ma so doc lap thi co the ghep doanh thu ban nay voi gia von
    ban kia, ra mot bo so KHONG ton tai trong bat ky bao cao nao. Nhung "chon mot bang duy nhat"
    cung sai, vi bang can doi thuong bi tach lam hai bang (tai san | nguon von).

    Nen: lay tung bang lam MOC, uu tien doc o moc roi lan sang cac bang GAN NHAT, va cham diem
    bang so dang thuc ke toan thoa duoc. Moc nao thoa nhieu nhat thi thang. Dang thuc la thu duy
    nhat o day khong can nhan va khong the chieu long ai — no dung hoac sai.
    """
    hits = {sp["ma"]: matched_rows(rows, sp) for sp in REGISTRY}
    hits = {ma: rs for ma, rs in hits.items() if rs}
    if not hits:
        return {}, {}
    anchors = sorted({r["tid"] for rs in hits.values() for r in rs})

    def build(anchor):
        out = {}
        for ma, rs in hits.items():
            out[ma] = min(rs, key=lambda r: (abs(r["tid"] - anchor), r["tid"]))
        return out

    best = max(anchors, key=lambda t: (_score({ma: r["cur"] for ma, r in build(t).items()}), -t))
    chosen = build(best)
    out, how = {}, {}
    for ma, r in chosen.items():
        vals = {round(x["cur"], 2) for x in hits[ma]}
        out[ma] = {"v": round(r["cur"], 2), "tid": r["tid"], "line": r["line"], "amb": len(vals) > 1}
        how[ma] = "main" if r["tid"] == best else "other"
    return out, how


def verify_identities(items):
    """Tra (so_kiem_duoc, so_thoa) tren cac dang thuc bang can doi."""
    get = lambda ma: (items.get(ma) or {}).get("v")
    testable = passed = 0
    for lhs, rhs in IDENTITIES:
        lv, rv = get(lhs), [get(k) for k in rhs]
        if lv is None or any(v is None for v in rv):
            continue
        testable += 1
        passed += close(lv, sum(rv))
    return testable, passed


def build_ticker(tk):
    """Gop moi nam cua mot doanh nghiep. Uu tien bao cao HOP NHAT, thieu thi dung bao cao RIENG."""
    items = collections.defaultdict(dict)      # ma_so -> {nam -> cell}
    spec_of = {r["ma"]: r for r in REGISTRY}
    periods, verified, srcdoc = [], {}, {}
    n_main = n_other = 0
    for year in sorted(os.listdir(os.path.join(ROOT, tk))):
        fr = P.find_report(tk, year, "consolidated") or P.find_report(tk, year, "separate")
        if not fr:
            continue
        rows, _uf, _src = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))
        cells, how = collect(rows)
        if not cells:
            continue
        periods.append(year)
        srcdoc[year] = fr[1]
        t, p = verify_identities(cells)
        verified[year] = {"testable": t, "passed": p, "ok": bool(t) and t == p}
        for ma, c in cells.items():
            items[ma][year] = c
            n_main += how[ma] == "main"
            n_other += how[ma] == "other"
    if not periods:
        return None
    return {
        "ticker": tk, "periods": periods, "unit": "VND",
        "sourceDoc": srcdoc, "verified": verified,
        # bao nhieu o lay tu BANG CHINH so voi phai muon tu bang khac (dau hieu mo ho)
        "match": {"main": n_main, "other": n_other},
        "items": {ma: {"label": spec_of[ma]["label"], "statement": spec_of[ma]["st"], "values": v}
                  for ma, v in sorted(items.items())},
    }


def write_wide_csv(path, rec):
    """Bang RONG mot doanh nghiep: ma_so, chi_tieu, <moi nam mot cot>.

    Vi sao can CSV ben canh JSON: app sinh ma Pandas doc file GOC de nguoi xem copy ra chay lai
    (agent.ts:pandasHeader/pandasLookup). Khong co file that thi ma do chi la trang tri. Dang RONG
    de khop dung hop dong meta.periodKeys (moi ky mot cot) ma app von dung cho file nguoi dung nap.
    """
    # Cot ma_ck: thieu no thi normalize() dat ten cong ty la "(toan bo)" (normalize.ts:74) va giao
    # dien hien chuoi do thay vi ma chung khoan. Them mot cot re hon nhieu so voi sua app.
    # Cot ky xep MOI NHAT TRUOC. Bao cao tai chinh that luon in ky hien tai o cot so dau, va app
    # dua vao dung quy uoc do: computeKpis() lay periods[0] lam ky hien tai, periods[1] lam ky
    # truoc (kpi.ts:50). Xep cu-truoc thi the KPI hien so cua nam DAU TIEN — da thay that: HPG
    # hien 27.453 ty (nam 2015) kem "-17,5%" trong khi nam 2025 la 156.116 ty.
    years = list(reversed(rec["periods"]))
    tk = rec["ticker"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ma_ck", "ma_so", "chi_tieu"] + years)
        for ma, it in rec["items"].items():
            w.writerow([tk, ma, it["label"]] +
                       [(int(round(it["values"][y]["v"])) if y in it["values"] else "") for y in years])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="chi lam N doanh nghiep dau (de thu)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    # DON thu muc truoc khi ghi. Khong don thi tep cua lan chay TRUOC con nam lai: doanh nghiep
    # nay khong con trich duoc nua (vd ngan hang sau khi siet khop ten) van hien dien voi du lieu
    # cu, va khong ai nhan ra. Da xay ra that trong lan chay dau.
    for fn in os.listdir(out):
        if fn.endswith((".json", ".csv")):
            os.remove(os.path.join(out, fn))

    tks = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    if a.limit:
        tks = tks[:a.limit]
    t0, index, n_cell = time.time(), [], 0
    for tk in tks:
        rec = build_ticker(tk)
        if not rec:
            # Van GHI VAO danh muc kem ly do, khong im lang bo qua. Khong chi tieu nao khop
            # registry TT200 nghia la doanh nghiep dung he tai khoan khac (to chuc tin dung theo
            # TT49/210). Nguoi xem can biet pham vi that cua he thong, khong phai mot danh sach
            # da loc cho dep. App hien so nay kem ly do o duoi bo chon.
            n_year = len([y for y in os.listdir(os.path.join(ROOT, tk))])
            index.append({"ticker": tk, "periods": n_year, "items": 0, "cells": 0,
                          "verifiedYears": 0, "inScope": False,
                          "reason": "khong chi tieu nao khop he Ma so TT200 "
                                    "(to chuc tin dung dung he tai khoan rieng)"})
            continue
        json.dump(rec, open(os.path.join(out, f"{tk}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, separators=(",", ":"))
        write_wide_csv(os.path.join(out, f"{tk}.csv"), rec)
        cells = sum(len(v["values"]) for v in rec["items"].values())
        n_cell += cells
        vy = sum(1 for y in rec["verified"].values() if y["ok"])
        index.append({"ticker": tk, "periods": len(rec["periods"]), "items": len(rec["items"]),
                      "cells": cells, "verifiedYears": vy, "inScope": vy >= 3})
    json.dump({"tickers": index, "generatedFrom": "ViFinQA (AIGuruTinix/ViFinQA)"},
              open(os.path.join(out, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)

    okco = [i for i in index if i["inScope"]]
    out_of = [i for i in index if not i["inScope"]]
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"\n=== {len(index)} doanh nghiep | {n_cell} o | {time.time()-t0:.0f}s | {mb:.1f} MB ===")
    print(f"  TRONG pham vi (>=3 nam qua kiem can doi) : {len(okco)}")
    print(f"  NGOAI pham vi (<3 nam qua kiem)          : {len(out_of)}"
          f"  -> {sorted(i['ticker'] for i in out_of)}")
    print(f"\nDanh sach demo an toan ({len(okco)} doanh nghiep, xep theo so nam da kiem):")
    for i in sorted(okco, key=lambda x: -x["verifiedYears"])[:15]:
        print(f"    {i['ticker']:5s} {i['verifiedYears']:2d}/{i['periods']:2d} nam  "
              f"{i['items']:3d} chi tieu  {i['cells']:4d} o")
    print(f"\nDau ra: {out}")


if __name__ == "__main__":
    main()
