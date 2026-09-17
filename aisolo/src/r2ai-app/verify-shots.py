"""KIEM ANH DEMO sau khi chup — bat anh trang, anh chup hut, anh trung nhau, anh cu sot lai.

Vi sao can: `capture-demo-shots.mjs` chi biet no DA GHI file, khong biet trong file co gi. Loi that
da gap trong cac phien truoc deu la loai "chup dung luc chua xong": bieu do chua ve xong, trace cua
cau TRUOC con trong DOM nen cho-selector khop nham. Nhung anh do van la PNG hop le, van dung kich
thuoc — chi nhin pixel moi biet.

Bon phep kiem:
  1. DU     — moi ten trong _shots.json phai co file, va khong con file la sot tu lan chup cu
  2. TUOI   — mtime phai thuoc lan chay nay (khong phai anh cu chua bi de)
  3. KHONG TRANG — do lech chuan pixel; anh trang/loading gan nhu don sac
  4. KHONG TRUNG — hai anh lien tiep giong het nhau nghia la trang chua kip doi trang thai

Chay: python verify-shots.py [thu_muc]      (mac dinh <repo>/demo-shots)
"""
import hashlib
import json
import os
import sys
import time

from PIL import Image, ImageStat

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "demo-shots")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".demo-shots.json")
STDEV_MIN = 12.0       # duoi nguong nay coi nhu anh gan don sac (trang / dang loading)
FRESH_SEC = 3600       # anh phai duoc ghi trong 1 gio qua

names = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else None
if names is None:
    print(f"!! Thieu {MANIFEST} — chay capture-demo-shots.mjs truoc")
    sys.exit(2)

now = time.time()
rows, problems = [], []
seen_hash = {}
prev_hash = None

for n in names:
    p = os.path.join(DIR, n + ".png")
    if not os.path.exists(p):
        problems.append(f"THIEU FILE: {n}.png")
        continue
    st = os.stat(p)
    age = now - st.st_mtime
    with Image.open(p) as im:
        w, h = im.size
        stdev = sum(ImageStat.Stat(im.convert("RGB")).stddev) / 3.0
    digest = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]

    flags = []
    if age > FRESH_SEC:
        flags.append(f"CU({age/60:.0f}ph)")
    if stdev < STDEV_MIN:
        flags.append(f"GAN-DON-SAC({stdev:.1f})")
    if st.st_size < 20_000:
        flags.append(f"NHE({st.st_size//1024}KB)")
    if digest == prev_hash:
        flags.append("TRUNG-ANH-TRUOC")
    if digest in seen_hash and seen_hash[digest] != n:
        flags.append(f"TRUNG:{seen_hash[digest]}")
    seen_hash.setdefault(digest, n)
    prev_hash = digest

    rows.append((n, w, h, st.st_size // 1024, stdev, age, flags))
    if flags:
        problems.append(f"{n}: {', '.join(flags)}")

extra = sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".png") and f[:-4] not in names)
for e in extra:
    problems.append(f"ANH LA (sot tu lan chup cu?): {e}.png")

print(f"{'anh':<32}{'kich thuoc':>13}{'KB':>7}{'do lech':>9}{'tuoi':>8}  ghi chu")
print("-" * 86)
for n, w, h, kb, sd, age, flags in rows:
    print(f"{n:<32}{f'{w}x{h}':>13}{kb:>7}{sd:>9.1f}{age:>7.0f}s  {', '.join(flags) if flags else 'ok'}")

print("-" * 86)
if problems:
    print(f"\n{len(problems)} VAN DE:")
    for x in problems:
        print("  !!", x)
    sys.exit(1)
print(f"\n{len(rows)}/{len(names)} anh DAT — khong anh trang, khong trung, khong sot anh cu.")
