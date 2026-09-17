"""DOI CHIEU DON VI GIUA HAI NAM LIEN TIEP — phat hien bao cao bi nhan sai he so don vi.

Rang buoc: bao cao nam N in lai so nam N-1 o cot ky truoc. Neu TONG TAI SAN cot ky truoc cua bao
cao nam N khac TONG TAI SAN cot ky nay cua bao cao nam N-1 dung mot LUY THUA CUA 1000, thi mot
trong hai bao cao da bi `detect_unit` nhan sai he so — va ta biet chac ai sai bang cach xem ben nao
cho tong tai san nam trong khoang hop ly.

Day la rang buoc noi tai cua du lieu, khong doan dap an => cung ban chat dang tin voi audit_wrong.py.
Khac voi doi chieu lien nam chung chung (do chinh xac chi 52%), rang buoc "lech dung luy thua 1000"
sac hon nhieu vi trung hop ngau nhien gan nhu khong the.
"""
import os, re, sys, json, collections, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "submission_out", "submission.json"), encoding="utf-8"))

# cac (ma CK, nam, loai BC) that su duoc dung lam nguon dap an
used = set()
for e in E:
    for ev in e.get("evidence", []):
        m = re.match(r"data/(.+?)_financial_statements_(\d{4})_(\w+?)(?:_table_\d+.*)?\.csv$", ev["csv_path"])
        if m:
            used.add((m.group(1), int(m.group(2)), m.group(3)))
        break

_C = {}
def tot_assets(tk, yr, kind):
    """(cur, prev) cua TONG TAI SAN sau chuan hoa, None neu khong co."""
    key = (tk, yr, kind)
    if key in _C:
        return _C[key]
    fr = P.find_report(tk, str(yr), kind)
    out = (None, None)
    if fr:
        rows, uf, src = P.ingest(open(str(fr[0]), encoding="utf-8", errors="replace").read())
        r = next((r for r in rows if r["cur"] is not None
                  and ("tong cong tai san" in P.strip_vn(r["label"]) or r["ma"] == "270")), None)
        if r:
            out = (r["cur"], r["prev"])
    _C[key] = out
    return out


LO, HI = 1e10, 2e16
stat = collections.Counter()
bad = []
for tk, yr, kind in sorted(used):
    cur_n, prev_n = tot_assets(tk, yr, kind)
    cur_p, _ = tot_assets(tk, yr - 1, kind)
    if prev_n is None or cur_p is None or cur_p == 0:
        stat["khong doi chieu duoc"] += 1
        continue
    stat["doi chieu duoc"] += 1
    ratio = abs(prev_n) / abs(cur_p)
    if 0.98 <= ratio <= 1.02:
        stat["khop"] += 1
        continue
    # lech dung mot luy thua cua 1000?
    lg = math.log(ratio, 1000) if ratio > 0 else None
    if lg is not None and abs(lg - round(lg)) < 0.02 and round(lg) != 0:
        stat["LECH luy thua 1000"] += 1
        # ben nao sai: ben cho tong tai san NGOAI khoang hop ly
        who = ("nam N" if not (LO <= abs(prev_n / 1) <= HI) else
               "nam N-1" if not (LO <= abs(cur_p) <= HI) else "khong ro")
        bad.append((tk, yr, kind, prev_n, cur_p, round(lg), who))
    else:
        stat["lech khac (co the do thay doi that)"] += 1

print(f"bo (ma CK, nam, loai BC) dung lam nguon dap an: {len(used)}")
for k, v in stat.most_common():
    print(f"  {k:<38}: {v}")
print()
print(f"=> {len(bad)} cap bao cao LECH DUNG LUY THUA 1000 (mot ben nhan sai don vi):")
for tk, yr, kind, pn, cp, lg, who in bad[:20]:
    print(f"  {tk} {yr} {kind:<12} prev(N)={pn:.4e}  cur(N-1)={cp:.4e}  lech 1000^{lg}  -> nghi {who}")
json.dump([[t, y, k] for t, y, k, *_ in bad], open(os.path.join(HERE, "unit_mismatch.json"), "w"))
