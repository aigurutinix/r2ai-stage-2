"""EXEC residual audit on screen_ratio_gated.zip."""
from __future__ import annotations

import dataclasses
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from vifin.answering import compose, ratio as R
from vifin.query.parse import parse_all
from vifin.retrieval.lexical import LexicalRetriever
from vifin.store import TableStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    with zipfile.ZipFile(ROOT / "submissions" / "screen_ratio_gated.zip") as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ret = LexicalRetriever(store.frame)

    lines: list[str] = []
    p = lines.append

    def branch(code: str) -> str:
        c = (code or "").strip()
        if c in ("", "result = 0.0"):
            return "zero"
        if "def num(" in c or "find_row(" in c:
            return "llm"
        if c.count("df") >= 2:
            return "multi"
        if "iloc" in c:
            return "lookup"
        return "other"

    bc = Counter(branch(preds[i].get("pandas_query") or "") for i in parsed)
    p(f"branches: {dict(bc)}")

    bad = []
    for i, q in parsed.items():
        if q.target_unit not in ("phan_tram", "lan", "vong"):
            continue
        ans = float(preds[i].get("answer") or 0)
        if abs(ans) <= 1000:
            continue
        bad.append((i, ans, branch(preds[i].get("pandas_query") or ""), q))
    p(f"\nratio-unit |ans|>1000: {len(bad)} by_branch={Counter(b for _,_,b,_ in bad)}")
    p(f"  1-ticker={sum(1 for *_, q in bad if len(q.tickers)==1)} "
      f"multi={sum(1 for *_, q in bad if len(q.tickers)>1)}")

    rows = []
    for i, q in parsed.items():
        m = compose.SCREEN_TICKER_RE.search(q.question)
        if not m:
            continue
        filt = m.group(1).strip(" ,.;:")
        if not compose.RATIO_FILTER_RE.search(filt):
            continue
        prefix = q.question[: m.start()]
        after = q.question[m.end() :]
        dirty = bool(compose._RATIO_SCREEN_DIRTY.search(filt))
        nested = bool(
            compose._RATIO_SCREEN_PREFIX_NESTED.search(prefix)
            or compose._RATIO_SCREEN_AFTER_NESTED.search(after)
        )
        clean = not dirty and not nested
        syn = dataclasses.replace(
            q, question=filt.split(".")[0].split(",")[0].strip()
        )
        shaped = R.shape(syn) is not None or R.compound_shape(syn) is not None
        year = max(q.years) if q.years else None
        ok = 0
        if year and shaped:
            for t in q.tickers:
                hit = compose._ratio_filter_amount(q, t, year, filt, store, ret, 10)
                if hit:
                    ok += 1
        ans = float(preds[i].get("answer") or 0)
        br = branch(preds[i].get("pandas_query") or "")
        rows.append(
            {
                "id": i,
                "dirty": dirty,
                "nested": nested,
                "clean": clean,
                "shaped": shaped,
                "ok": ok,
                "n": len(q.tickers),
                "ans": ans,
                "unit": q.target_unit,
                "br": br,
                "filt": filt[:55],
                "q": q.question[:110],
            }
        )

    p(f"\nratio-filter screens: {len(rows)}")
    p(f"  clean_gate: {sum(r['clean'] for r in rows)}")
    p(f"  nested|dirty: {sum(r['nested'] or r['dirty'] for r in rows)}")
    all_ok = [r for r in rows if r["shaped"] and r["n"] and r["ok"] == r["n"]]
    p(f"  ALL_ok filter resolve: {len(all_ok)} ids={[r['id'] for r in all_ok]}")
    blocked = [r for r in all_ok if not r["clean"]]
    p(f"  ALL_ok blocked by nest/dirty (new headroom): {len(blocked)}")
    for r in blocked:
        p(
            f"    id={r['id']} nested={r['nested']} dirty={r['dirty']} "
            f"{r['br']} ans={r['ans']:.4g} | {r['filt']}"
        )

    arch = Counter()
    for r in rows:
        if not (r["nested"] or r["dirty"]):
            continue
        q = parsed[r["id"]].question
        if re.search(r"trung vị", q, re.I):
            arch["median_cohort"] += 1
        elif re.search(r"lớn hơn|thấp hơn", q, re.I):
            arch["threshold_cohort"] += 1
        else:
            arch["other_nested"] += 1
    p(f"\nnested/dirty arch: {dict(arch)}")

    # Pass-2 feasibility: for blocked ALL_ok, can we resolve asked metric for winner?
    p("\n=== blocked ALL_ok: if we IGNORED nest and ranked by filter only ===")
    for r in blocked:
        q = parsed[r["id"]]
        m = compose.SCREEN_TICKER_RE.search(q.question)
        filt = m.group(1).strip(" ,.;:")
        year = max(q.years)
        want_max = m.group(2).lower() in compose.MAX_WORDS
        amounts = {}
        for t in q.tickers:
            hit = compose._ratio_filter_amount(q, t, year, filt, store, ret, 10)
            amounts[t] = hit[0]
        winner = max(amounts, key=amounts.get) if want_max else min(amounts, key=amounts.get)
        am_s = ", ".join(f"{k}:{v:.3f}" for k, v in amounts.items())
        p(f"  id={r['id']} winner_if_blind={winner} amounts={am_s}")
        p(f"    WARN nested — blind rank may be wrong | {r['q']}")

    p("\n=== EXEC VERDICT ===")
    p("zeros=17 → ceiling tiny if filled")
    p("|ans|>1000 ratio=94 → ship wrong figures; 1-ticker ratio.resolve cannot displace any today")
    p("clean screen_ratio done (+1). Next volume = nested cohort recipes, not more clean gates.")
    p("~133 pool @9% → 15% = +8q (~+0.016 EXEC); 25% = +21q (~+0.042 EXEC)")
    p("Vương gap EXEC 0.3538-0.3281=0.0257 ≈ 13 graded questions")

    out = ROOT / "artifacts" / "_exec_residual_audit.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        if "WARN" in line or line.startswith("    id="):
            continue
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
