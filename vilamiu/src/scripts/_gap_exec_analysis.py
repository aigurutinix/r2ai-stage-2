"""Quantify remaining EXEC gaps after shape_lock. Write UTF-8 report."""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from vifin.answering import compose, lookup as L, ratio as ratio_mod
from vifin.query.parse import parse_all
from vifin.retrieval.lexical import LexicalRetriever
from vifin.store import TableStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "_gap_exec_analysis.txt"


def load_preds(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        raw = json.loads(z.read("submission.json"))
    preds = raw["predictions"] if isinstance(raw, dict) else raw
    return {p["id"]: p for p in preds}


def main() -> None:
    lines: list[str] = []
    p = lines.append

    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    sl = load_preds("shape_lock.zip")
    ep = load_preds("evidence_pad4.zip")

    # --- pool sizes ---
    n_elig = sum(1 for q in parsed.values() if compose.eligible(q))
    n_scr = sum(1 for q in parsed.values() if compose.screen_shape(q))
    n_rat = sum(1 for q in parsed.values() if ratio_mod.eligible(q))
    n_derived = sum(1 for q in parsed.values() if not L.is_single_lookup(q.question))
    n_single = 1012 - n_derived
    n_currency_single = sum(
        1
        for q in parsed.values()
        if q.unit_scale and L.is_single_lookup(q.question)
    )
    p("=== POOL SIZES (live parsers) ===")
    p(f"total questions: 1012")
    p(f"derived (DERIVED_RE): {n_derived}")
    p(f"single-lookup: {n_single}")
    p(f"currency single-lookup: {n_currency_single}")
    p(f"compose.eligible: {n_elig}")
    p(f"compose.screen_shape: {n_scr}")
    p(f"ratio.eligible: {n_rat}")

    # --- shape_lock outcome for claimed shapes ---
    elig_shipped = elig_zero = elig_other = 0
    scr_shipped = scr_zero = scr_other = 0
    elig_zero_ids = []
    scr_zero_ids = []
    for i, q in parsed.items():
        code = (sl[i].get("pandas_query") or "").strip()
        zero = code == "result = 0.0"
        multi = "df1" in code or code.count("iloc") >= 2 or "idxmax" in code or "idxmin" in code
        if compose.eligible(q):
            if zero:
                elig_zero += 1
                elig_zero_ids.append(i)
            elif multi:
                elig_shipped += 1
            else:
                elig_other += 1
        if compose.screen_shape(q):
            if zero:
                scr_zero += 1
                scr_zero_ids.append(i)
            elif multi or "idxmax" in code or "idxmin" in code:
                scr_shipped += 1
            else:
                scr_other += 1
    p("\n=== SHAPE_LOCK OUTCOMES ===")
    p(f"eligible: shipped_multi={elig_shipped} zero={elig_zero} other={elig_other}")
    p(f"screen: shipped={scr_shipped} zero={scr_zero} other={scr_other}")
    p(f"elig_zero ids: {elig_zero_ids}")
    p(f"scr_zero ids: {scr_zero_ids}")

    became_zero = [
        i
        for i in parsed
        if (ep[i].get("pandas_query") or "").strip() != "result = 0.0"
        and (sl[i].get("pandas_query") or "").strip() == "result = 0.0"
    ]
    p(f"pad4→shape became zero: {len(became_zero)}")

    # --- why derived year not eligible ---
    block_reasons: Counter = Counter()
    block_ex: dict[str, list] = defaultdict(list)
    for i, q in parsed.items():
        if L.is_single_lookup(q.question):
            continue
        if not (len(q.tickers) == 1 and len(q.years) >= 2):
            continue
        if compose.eligible(q) is not None or compose.screen_shape(q) is not None:
            continue
        op = compose.classify(q.question)
        if op is None:
            reason = "no_op_classify"
        elif op != "growth" and q.unit_scale is None:
            reason = "no_unit_scale"
        elif op in ("diff", "growth") and len(q.years) != 2:
            reason = f"{op}_n_years_ne_2(n={len(q.years)})"
        else:
            reason = "other"
        block_reasons[reason] += 1
        if len(block_ex[reason]) < 5:
            block_ex[reason].append((i, op, q.years, q.target_unit, q.question[:110]))
    p("\n=== DERIVED 1-TICKER MULTI-YEAR NOT ELIGIBLE/SCREEN ===")
    for k, v in block_reasons.most_common():
        p(f"  {k}: {v}")
        for e in block_ex[k]:
            p(f"    {e}")

    # multi-ticker derived not eligible
    mt_block = Counter()
    mt_ex = defaultdict(list)
    for i, q in parsed.items():
        if len(q.tickers) < 2 or not q.years:
            continue
        if compose.eligible(q) or compose.screen_shape(q):
            continue
        op = compose.classify(q.question)
        if op is None:
            reason = "no_op"
        elif compose.SCREEN_RE.search(q.question):
            reason = "SCREEN_RE_blocks_ticker_axis"
        elif op != "growth" and q.unit_scale is None:
            reason = "no_scale"
        elif op in ("diff", "growth") and len(q.tickers) != 2:
            reason = f"{op}_n_tickers_ne_2"
        else:
            reason = "other"
        mt_block[reason] += 1
        if len(mt_ex[reason]) < 4:
            mt_ex[reason].append((i, op, q.tickers, q.question[:100]))
    p("\n=== MULTI-TICKER WITH YEARS NOT ELIGIBLE/SCREEN ===")
    p(f"total multi-ticker qs: {sum(1 for q in parsed.values() if len(q.tickers)>=2)}")
    for k, v in mt_block.most_common():
        p(f"  {k}: {v}")
        for e in mt_ex[k]:
            p(f"    {e}")

    # count-like
    count_re = re.compile(r"có bao nhiêu (năm|công ty)|số năm|bao nhiêu công ty", re.I)
    p("\n=== COUNT-LIKE ===")
    for i, q in parsed.items():
        if not count_re.search(q.question):
            continue
        code = (sl[i].get("pandas_query") or "").strip()
        p(
            f"  id={i} zero={code=='result = 0.0'} ans={sl[i].get('answer')} "
            f"elig={compose.eligible(q)} single={L.is_single_lookup(q.question)}"
        )
        p(f"    {q.question[:140]}")

    # false sum (Tong in metric)
    false_sum = []
    for i, q in parsed.items():
        el = compose.eligible(q)
        if not el or el[0] != "sum":
            continue
        if re.search(r"tăng|giảm|chênh lệch", q.question, re.I) and not re.search(
            r"tính tổng|cộng lại|tích lũy", q.question, re.I
        ):
            false_sum.append(i)
    p(f"\n=== FALSE SUM (Tong in name + tang/giam): {len(false_sum)} → {false_sum[:20]}")

    # resolve failures on elig_zero
    p("\n=== RESOLVE ON ELIG_ZERO (first 20) ===")
    resolve_fail = Counter()
    for i in elig_zero_ids[:20]:
        q = parsed[i]
        el = compose.eligible(q)
        r = compose.resolve(q, store, retriever)
        if r is None:
            resolve_fail["resolve_None"] += 1
            status = "None"
        else:
            resolve_fail["resolve_ok"] += 1
            status = f"{r.op} score={r.score:.2f} n={len(r.values)}"
        # plan/llm present?
        p(f"  id={i} elig={el} resolve={status}")
        p(f"    {q.question[:120]}")
    p(f"  (sample counts {dict(resolve_fail)})")

    # screen zero resolve
    p("\n=== RESOLVE_SCREEN ON SCR_ZERO ===")
    for i in scr_zero_ids:
        q = parsed[i]
        sc = compose.screen_shape(q)
        s = compose.resolve_screen(q, store, retriever)
        p(f"  id={i} shape={sc} ok={s is not None}")
        p(f"    {q.question[:120]}")

    # weak single failures: empty/header/bare-total labels in shape_lock
    weak_labels = Counter()
    weak_ids = defaultdict(list)
    for i, q in parsed.items():
        if not L.is_single_lookup(q.question):
            continue
        code = sl[i].get("pandas_query") or ""
        m = re.search(r'labels\s*==\s*"([^"]*)"', code)
        if not m:
            continue
        lab = m.group(1)
        bad = None
        if lab == "":
            bad = "empty_label"
        elif re.search(r"tại ngày|cuối năm|đầu năm|năm nay|năm trước", lab, re.I):
            bad = "period_header"
        elif re.fullmatch(r"tổng cộng|tổng cộng tài sản|tổng tài sản", lab, re.I):
            # may be correct for some qs
            if "tổng tài sản" not in q.question.lower() and "tổng cộng" not in q.question.lower():
                bad = "bare_total_mismatch"
        if bad:
            weak_labels[bad] += 1
            if len(weak_ids[bad]) < 8:
                weak_ids[bad].append((i, lab, q.question[:90]))
    p("\n=== WEAK SINGLE-CELL LABELS (heuristic) ===")
    for k, v in weak_labels.most_common():
        p(f"  {k}: {v}")
        for e in weak_ids[k]:
            p(f"    {e}")

    # ratio_screen left alone (48 claimed)
    ratio_screens = []
    for i, q in parsed.items():
        # would be screen except RATIO_FILTER
        if len(q.tickers) == 1 and len(q.years) >= 2:
            m = compose.SCREEN_YEAR_RE.search(q.question)
            axis = "year"
        elif len(q.tickers) >= 2 and q.years:
            m = compose.SCREEN_TICKER_RE.search(q.question)
            axis = "ticker"
        else:
            continue
        if m is None:
            continue
        metric = m.group(1).strip(" ,.;:")
        if compose.RATIO_FILTER_RE.search(metric):
            ratio_screens.append((i, axis, metric[:60], q.question[:100]))
    p(f"\n=== RATIO-FILTER SCREENS LEFT ALONE: {len(ratio_screens)}")
    for e in ratio_screens[:12]:
        p(f"  {e}")

    # ABSENT-style: currency single, not shape locked, still garbage or zero
    # Compare: questions where pad4 and shape same single-cell with suspicious label
    p("\n=== DOCUMENTED EXAMPLE IDS STATUS ===")
    for qid in [938, 823, 608, 239, 102, 222, 37, 681, 116, 15, 2, 514, 922]:
        q = parsed[qid]
        p(
            f"id={qid} elig={compose.eligible(q)} screen={compose.screen_shape(q)} "
            f"classify={compose.classify(q.question)} single={L.is_single_lookup(q.question)}"
        )
        p(f"  tickers={q.tickers} years={q.years} unit={q.target_unit}")
        p(f"  pad4={ep[qid].get('answer')} shape={sl[qid].get('answer')}")
        p(f"  q={q.question[:150]}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
