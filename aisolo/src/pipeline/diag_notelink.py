"""Do DO PHU cua LIEN KET THUYET MINH do chinh BCTC khai (cot so TM canh cot Ma so).
Neu do phu qua thap thi khong dang lam, giong het bai hoc 'tien to phan cap' (ctx chi 20,3%)."""
import os, sys, glob, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

files = sorted(glob.glob(os.path.join(P.ROOT, "financial_statements", "*", "*", "*", "*_extracted.txt")))
random.seed(0); files = random.sample(files, 120)

tot = has_note = 0
main_tot = main_note = 0
resolved = 0
for f in files:
    txt = open(f, encoding="utf-8", errors="replace").read()
    rows, _, _ = P.ingest(txt)
    for r in rows:
        tot += 1
        if r["note"]:
            has_note += 1
            if P.note_table_line(txt, r["note"]): resolved += 1
        if r["st"] in ("BS", "PL", "CF"):
            main_tot += 1
            if r["note"]: main_note += 1

print("Tong dong                       : %6d" % tot)
print("  co so Thuyet minh khai san    : %6d  (%.1f%%)" % (has_note, 100*has_note/tot))
print("  trong do GIAI RA duoc bang TM : %6d  (%.1f%% cua tong)" % (resolved, 100*resolved/tot))
print()
print("Dong thuoc bao cao CHINH (BS/PL/CF): %6d" % main_tot)
print("  co so Thuyet minh             : %6d  (%.1f%%)  <- do phu thuc te cua tin hieu" % (main_note, 100*main_note/max(1,main_tot)))
