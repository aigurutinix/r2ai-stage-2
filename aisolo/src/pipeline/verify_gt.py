"""Ground-truth (best-effort, hand-specced) cho các câu AGG/COND RÕ RÀNG → đo accuracy agent local.
Chỉ gồm câu logic không cãi được (standard metric). So agent (agent_results.json) vs GT.
LƯU Ý: GT là cách hiểu của MÌNH; câu mơ hồ (median edge, định nghĩa lạ) KHÔNG đưa vào để tránh tự-lừa."""
import os, json, statistics
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

_c = {}
def load(tk, y, dt="consolidated"):
    k = (tk, int(y), dt)
    if k not in _c:
        fr = P.find_report(tk, str(y), dt)
        _c[k] = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0] if fr else None
    return _c[k]
def val(rows, ma, st):
    if rows is None: return None
    ma = str(ma)
    if len(ma) >= 3:
        c = [r["cur"] for r in rows if str(r["ma"]) == ma and r["cur"] is not None]
        return max(c, key=abs) if c else None
    return next((r["cur"] for r in rows if str(r["ma"]) == ma and r["st"] == st and r["cur"] is not None), None)

# ----- metric library (rows 1 năm) -----
def cfo(r): return val(r,20,"CF")
def lnst(r): return val(r,60,"PL")
def rev(r): return val(r,10,"PL")
def gross(r): return val(r,20,"PL")
def lnthuan(r): return val(r,30,"PL")   # LN thuần HĐKD
def lntt(r): return val(r,50,"PL")
def interest(r): return val(r,23,"PL")  # chi phí lãi vay (thường âm)
def D(a,b): return a/b if (a is not None and b) else None
def bien_gop(r): return D(gross(r),rev(r))
def npm(r): return D(lnst(r),rev(r))          # LNST/DTT (tỉ số; ×100 = %)
def quick(r):
    a,h,n=val(r,100,"BS"),val(r,140,"BS"),val(r,310,"BS"); return (a-h)/n if (a is not None and h is not None and n) else None
def current(r): return D(val(r,100,"BS"),val(r,310,"BS"))
def de(r): return D(val(r,300,"BS"),val(r,400,"BS"))
def debt_asset(r): return D(val(r,300,"BS"),val(r,270,"BS"))
def roa(r): return D(lnst(r),val(r,270,"BS"))
def icover(r):
    m50,m23=lntt(r),interest(r); return (m50+m23)/m23 if (m50 is not None and m23) else None
def cfo_debt(r): return D(cfo(r),val(r,310,"BS"))
def cfo_lnst(r): return D(cfo(r),lnst(r))
def cfo_lnthuan(r): return D(cfo(r),lnthuan(r))
def htk_ratio(r): return D(val(r,140,"BS"),val(r,270,"BS"))   # tỷ trọng HTK/TS
def wcap(r):
    a,n=val(r,100,"BS"),val(r,310,"BS"); return a-n if (a is not None and n is not None) else None

def argsel(items, key, d):
    best=None
    for it in items:
        k=key(it)
        if k is None: continue
        if best is None or (k>best[0] if d=="max" else k<best[0]): best=(k,it)
    return best[1] if best else None

# ----- SPECS câu MAX->B (1 năm): filter cond -> argmax/min theo sel -> tính ans của công ty đó / unit -----
# sel/ans nhận rows; cond nhận rows->bool; unit chia (1=ratio/lần; 100 nếu cần ×100 cho %; 1e12 nghìn tỷ...)
S = {
 371: dict(g="BSR PLX PVT",y=2024,cond=lambda r:(cfo(r) or -1)>0,sel=bien_gop,d="max",ans=icover,u=1),
 392: dict(g="MCH QNS OGC",y=2024,sel=cfo_lnst,d="max",ans=quick,u=1),
 394: dict(g="HPX KBC NVL SCR VIC VPI VRE",y=2024,cond=lambda r:(lnst(r) or -1)>0,sel=lnst,d="max",ans=cfo,u=1e12),
 396: dict(g="ASM DBC MSN OGC",y=2024,cond=lambda r:(cfo(r) or -1)>0 and (lnst(r) or -1)>0,sel=cfo_lnst,d="max",ans=quick,u=1),
 403: dict(g="DIG HPX KBC NVL SCR VIC VPI VRE",y=2024,cond=lambda r:(current(r) or 0)>1,sel=quick,d="min",ans=cfo_debt,u=1),
 416: dict(g="BSR PLX PVT",y=2024,sel=cfo_debt,d="min",ans=quick,u=1),
 420: dict(g="VNM MSN DBC ASM MPC OGC",y=2024,cond=lambda r:(wcap(r) or 0)<0,sel=debt_asset,d="min",ans=lambda r:roa(r)*100,u=1),
 461: dict(g="BSR PLX PVT",y=2017,sel=lambda r:D(cfo(r)-lnst(r),rev(r)),d="max",ans=de,u=1),
 465: dict(g="BSR PLX PVT",y=2019,sel=de,d="max",ans=icover,u=1),
 553: dict(g="BSR PLX PVT",y=2019,sel=de,d="max",ans=icover,u=1),
 477: dict(g="BSR PLX PVT",y=2017,sel=cfo_lnthuan,d="min",ans=lambda r:npm(r)*100,u=1),
 492: dict(g="BSR PLX PVT",y=2017,cond=lambda r:(lnthuan(r) or -1)>0,sel=cfo_lnthuan,d="min",ans=lambda r:npm(r)*100,u=1),
 494: dict(g="BSR PLX PVT",y=2019,cond=lambda r:(lnthuan(r) or -1)>0,sel=cfo_lnthuan,d="min",ans=lambda r:npm(r)*100,u=1),
 545: dict(g="BSR PLX PVT",y=2017,cond=lambda r:(lnthuan(r) or -1)>0,sel=cfo_lnthuan,d="min",ans=lambda r:npm(r)*100,u=1),
 376: dict(g="HPX KBC NVL SCR VIC VPI VRE",y=2024,cond=lambda r:(current(r) or 0)>1.5,sel=quick,d="min",ans=lambda r:htk_ratio(r)*100,u=1),
 397: dict(g="DIG HPX KBC NVL SCR VIC VPI VRE",y=2024,cond=lambda r:(current(r) or 0)>1.5,sel=quick,d="min",ans=lambda r:val(r,140,"BS"),u=1e12),
}
def gt_maxb(s):
    g=[t for t in s["g"].split() if (s.get("cond",lambda r:True)(load(t,s["y"])) if load(t,s["y"]) else False)]
    w=argsel(g, lambda t: s["sel"](load(t,s["y"])), s["d"])
    if not w: return None
    v=s["ans"](load(w,s["y"]))
    return round(v/s["u"],3) if v is not None else None

# ----- SPECS temporal (1 công ty): chọn NĂM theo điều kiện -> ans của năm đó/sau -----
def gt_temporal(qid):
    if qid==363:  # KBC 2016-2020, năm max D/E -> icover
        w=argsel(range(2016,2021), lambda y:de(load("KBC",y)),"max"); return round(icover(load("KBC",w)),3)
    if qid==365:  # KBC 2016-2021, năm CFO<0 đầu tiên -> bien gop năm SAU (%)
        yr=next(y for y in range(2016,2022) if (cfo(load("KBC",y)) or 0)<0); return round(bien_gop(load("KBC",yr+1))*100,3)
    if qid==372:  # VRE 2021-2024, năm min quick -> năm SAU cfo_debt
        yr=argsel(range(2021,2025), lambda y:quick(load("VRE",y)),"min"); return round(cfo_debt(load("VRE",yr+1)),3)
    if qid==415:  # HPG 2020-2024, năm max doanh thu -> current
        w=argsel(range(2020,2025), lambda y:rev(load("HPG",y)),"max"); return round(current(load("HPG",w)),3)
    if qid==456:  # KBC 2016-2021, năm sau năm CFO<0 đầu tiên -> biên gộp % (= id365)
        yr=next(y for y in range(2016,2022) if (cfo(load("KBC",y)) or 0)<0); return round(bien_gop(load("KBC",yr+1))*100,3)
    if qid==387:  # HPG 2021-2024, năm LNST>0 có min(CFO/LNST) -> icover
        yrs=[y for y in range(2021,2025) if (lnst(load("HPG",y)) or -1)>0]
        w=argsel(yrs, lambda y:cfo_lnst(load("HPG",y)),"min"); return round(icover(load("HPG",w)),3)
    return None
TEMPORAL=[363,365,372,415,456,387]

def all_gt():
    g={}
    for i in S:
        try: g[i]=gt_maxb(S[i])
        except Exception as e: g[i]=f"ERR:{type(e).__name__}"
    for i in TEMPORAL:
        try: g[i]=gt_temporal(i)
        except Exception as e: g[i]=f"ERR:{type(e).__name__}"
    return g

if __name__=="__main__":
    g=all_gt()
    print("=== GROUND TRUTH ({} câu) ===".format(len(g)))
    for i in sorted(g): print(f"  id{i}: {g[i]}")
    p=os.environ.get("AGENT_OUT","agent_results.json")
    if os.path.exists(p):
        agent={e["id"]:e for e in json.load(open(p,encoding="utf-8"))}
        ok=tot=0; fails=[]
        print("\n=== SO agent vs GT ===")
        for i in sorted(g):
            v=g[i]; a=agent.get(i,{}).get("answer")
            if a is None or isinstance(v,str) or v is None: continue
            tot+=1; m=abs(a-v)/max(1e-9,abs(v))<0.03; ok+=m
            if not m: fails.append(i)
            print(f"  id{i}: agent={round(a,3)} GT={v} {'OK' if m else 'SAI'}")
        print(f"\nACCURACY: {ok}/{tot}" + (f" | SAI: {fails}" if fails else ""))
