"""Audit the computed-ratio / ratio-filter pool against helpers.zip."""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from vifin.answering import compose, ratio as ratio_mod
from vifin.answering import lookup as L
from vifin.query.parse import parse_all

ROOT = Path(__file__).resolve().parents[1]

# Threshold / screen / cohort wording that typically needs a *computed* ratio.
NEED_COMPUTED = re.compile(
    r"\b(?:ROA|ROE|ROS)\b|"
    r"biên lợi nhuận|"
    r"(?:hệ số|tỷ số|tỉ số|tỷ lệ|tỉ lệ)\s+"
    r"(?:thanh toán|nợ|CFO|D/E|khả năng|"
    r"lợi nhuận|sinh lời|vòng quay|đòn bẩy)|"
    r"trên\s+(?:doanh thu|tổng|vốn|nợ|tài sản)|"
    r"CFO\s+trên|LNST\s+trên|"
    r"thấp hơn trung vị|cao hơn trung vị|"
    r"thấp hơn .{0,20}(?:lần|%)|cao hơn .{0,20}(?:lần|%)",
    re.I,
)

NAMED = re.compile(
    r"\b(?:ROA|ROE|ROS)\b|biên lợi nhuận|"
    r"(?:hệ số|tỷ số)\s+thanh toán|"
    r"sinh lời trên",
    re.I,
)


def branch(code: str) -> str:
    c = (code or "").strip()
    if c in ("", "result = 0.0"):
        return "zero"
    if "def num(" in c or "find_row(" in c:
        return "llm"
    if c.count("df") >= 2 and ("iloc" in c or "num(" in c):
        # heuristic: multi-frame often compose/screen/plan
        if " / " in c or "ratio" in c.lower():
            return "ratioish"
        return "multi"
    if "iloc" in c:
        return "lookup"
    return "other"


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    zip_path = ROOT / "submissions" / "helpers.zip"
    with zipfile.ZipFile(zip_path) as z:
        raw = json.loads(z.read("submission.json"))
    preds = {p["id"]: p for p in (raw if isinstance(raw, list) else raw["predictions"])}

    pool = []
    for i, q in parsed.items():
        metric = L.extract_metric(q.question)
        ratio_filter_block = bool(compose.RATIO_FILTER_RE.search(metric or ""))
        # Would be screen except ratio filter, OR question clearly needs computed ratio.
        scr = compose.screen_shape(q)
        elig = compose.eligible(q)
        rshape = ratio_mod.shape(q)
        cshape = ratio_mod.compound_shape(q)
        needs = bool(NEED_COMPUTED.search(q.question)) or ratio_filter_block
        if not needs and scr is None:
            continue
        # Focus: screening/filtering/ranking by ratio OR named formula in question
        # while answer path is weak OR ratio_filter blocks screen.
        if not (ratio_filter_block or NAMED.search(q.question) or (
            needs and scr is None and elig is None and q.target_unit in ("phan_tram", "lan", None)
        )):
            # keep broader: any NEED_COMPUTED with unit ratio OR ratio_filter
            if not (ratio_filter_block or (needs and q.target_unit in ("phan_tram", "lan"))):
                continue

        pred = preds[i]
        code = pred.get("pandas_query") or ""
        ans = pred.get("answer")
        pool.append(
            {
                "id": i,
                "ans": ans,
                "zero": str(code).strip() == "result = 0.0",
                "branch": branch(code),
                "unit": q.target_unit,
                "scr": scr,
                "elig": elig,
                "rshape": rshape,
                "cshape": cshape,
                "ratio_filter": ratio_filter_block,
                "metric": (metric or "")[:70],
                "q": q.question[:140],
            }
        )

    # Also: classic "ratio filter screens left alone" set
    ratio_screens = []
    for i, q in parsed.items():
        metric = L.extract_metric(q.question)
        if not metric or not compose.RATIO_FILTER_RE.search(metric):
            continue
        # mimic screen detection without filter
        axis = None
        if compose.SCREEN_YEAR_RE.search(q.question):
            axis = "year"
        elif compose.SCREEN_TICKER_RE.search(q.question):
            axis = "ticker"
        if axis is None:
            continue
        pred = preds[i]
        code = pred.get("pandas_query") or ""
        ratio_screens.append(
            {
                "id": i,
                "axis": axis,
                "ans": pred.get("answer"),
                "branch": branch(code),
                "zero": str(code).strip() == "result = 0.0",
                "rshape": ratio_mod.shape(q) is not None,
                "cshape": ratio_mod.compound_shape(q) is not None,
                "metric": metric[:60],
                "q": q.question[:120],
            }
        )

    out = ROOT / "artifacts" / "_audit_ratio_pool.txt"
    lines: list[str] = []
    p = lines.append
    p(f"helpers.zip predictions: {len(preds)}")
    p(f"\n=== BROAD NEED_COMPUTED / FILTER POOL: {len(pool)}")
    p(f"  zero: {sum(1 for x in pool if x['zero'])}")
    p(f"  nonzero: {sum(1 for x in pool if not x['zero'])}")
    p(f"  by branch: {Counter(x['branch'] for x in pool)}")
    p(f"  ratio_filter blocked: {sum(1 for x in pool if x['ratio_filter'])}")
    p(f"  has ratio.shape: {sum(1 for x in pool if x['rshape'])}")
    p(f"  has compound_shape: {sum(1 for x in pool if x['cshape'])}")
    p(f"  named ROA/ROE/quick/margin patterns in q: "
      f"{sum(1 for x in pool if NAMED.search(x['q']))}")

    p(f"\n=== RATIO-FILTER SCREENS (classic 49-set): {len(ratio_screens)}")
    p(f"  zero: {sum(1 for x in ratio_screens if x['zero'])}")
    p(f"  by branch: {Counter(x['branch'] for x in ratio_screens)}")
    p(f"  shape-ok (ratio or compound): "
      f"{sum(1 for x in ratio_screens if x['rshape'] or x['cshape'])}")
    p(f"  by axis: {Counter(x['axis'] for x in ratio_screens)}")

    # Solvable probe: how many ratio_screens have full shape rewrite
    p("\n=== SAMPLE ratio-filter screens (id axis rshape cshape branch | q)")
    for x in ratio_screens[:25]:
        p(
            f"  {x['id']} {x['axis']} r={x['rshape']} c={x['cshape']} "
            f"{x['branch']} zero={x['zero']} | {x['q']}"
        )

    # Intersection: ratio_screens that ALSO have compound/named OR rshape
    ripe = [x for x in ratio_screens if x["rshape"] or x["cshape"]]
    p(f"\n=== RIPE FOR SCREEN-ON-RATIO (shape known): {len(ripe)}")
    p(f"  currently zero: {sum(1 for x in ripe if x['zero'])}")
    for x in ripe[:20]:
        p(f"  {x['id']} {x['axis']} c={x['cshape']} r={x['rshape']} | {x['metric']}")

    # Compare helpers vs screen_ratio.zip answers on ripe ids
    sr_path = ROOT / "submissions" / "screen_ratio.zip"
    if sr_path.exists():
        with zipfile.ZipFile(sr_path) as z:
            raw2 = json.loads(z.read("submission.json"))
        sr = {p["id"]: p for p in (raw2 if isinstance(raw2, list) else raw2["predictions"])}
        changed = same0 = gained = lost = 0
        for x in ratio_screens:
            a = preds[x["id"]].get("answer")
            b = sr[x["id"]].get("answer")
            az = str(preds[x["id"]].get("pandas_query") or "").strip() == "result = 0.0"
            bz = str(sr[x["id"]].get("pandas_query") or "").strip() == "result = 0.0"
            if a != b:
                changed += 1
                if az and not bz:
                    gained += 1
                if (not az) and bz:
                    lost += 1
            if az and bz:
                same0 += 1
        p(f"\n=== helpers vs screen_ratio.zip on {len(ratio_screens)} ratio-screens")
        p(f"  answer diffs: {changed}  zero->nonzero: {gained}  nonzero->zero: {lost}  both zero: {same0}")

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
