"""CAN TREN cua HUONG 2 — gold-free, chi doc du lieu.

Cau hoi: khi mot NHAN xuat hien o NHIEU bang trong cung mot bao cao (o "cung nhan khac BANG"
chiem 59/88 dư dia), thi cac bang do co KHAC LOAI (BS/PL/CF/NOTE) khong?
  - Neu KHAC loai nhieu  -> cong loc theo stmt_type co cho de bam.
  - Neu CUNG loai la chinh -> stmt_type khong phan biet duoc, huong 2 chet truoc khi bat dau.
"""
import os, sys, glob, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

files = sorted(glob.glob(os.path.join(P.ROOT, "financial_statements", "*", "*", "*", "*_extracted.txt")))
random.seed(0); files = random.sample(files, 120)

dup_same = dup_diff = 0
pair = collections.Counter()
st_all = collections.Counter()
for f in files:
    txt = open(f, encoding="utf-8", errors="replace").read()
    rows, _, _ = P.ingest(txt)
    by = collections.defaultdict(set)          # nhan -> tap (tid, st)
    for r in rows:
        if r["label"] and r["cur"] is not None:
            by[P.strip_vn(r["label"]).lower()].add((r["tid"], r["st"]))
        st_all[r["st"]] += 1
    for lab, s in by.items():
        if len(s) < 2: continue                 # nhan chi o 1 bang -> khong mo ho
        sts = {st for _, st in s}
        if len(sts) == 1: dup_same += 1
        else:
            dup_diff += 1
            for a in sorted(sts):
                for b in sorted(sts):
                    if a < b: pair[(a, b)] += 1

tot = dup_same + dup_diff
print("Bao cao lay mau           :", len(files))
print("Phan bo dong theo loai bang:", dict(st_all))
print()
print("NHAN xuat hien o >=2 bang  :", tot)
print("  cac bang CUNG loai       : %5d  (%.1f%%)" % (dup_same, 100*dup_same/max(1,tot)))
print("  cac bang KHAC loai       : %5d  (%.1f%%)  <- vung cong stmt_type co the loc" % (dup_diff, 100*dup_diff/max(1,tot)))
print()
print("Cap loai hay lan nhau nhat:")
for (a, b), n in pair.most_common(8):
    print("   %-5s vs %-5s : %d" % (a, b, n))
