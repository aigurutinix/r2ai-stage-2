"""Deep EXEC headroom audit across latest submissions + artifacts.

Outputs UTF-8 report to artifacts/_exec_headroom.txt
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import compose, lookup as L, ratio as ratio_mod
from vifin.corpus.numeric import is_correct
from vifin.query.parse import parse_all
from vifin.retrieval.lexical import LexicalRetriever
from vifin.store import TableStore

extract_metric = L.extract_metric

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "_exec_headroom.txt"
lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)


def load_preds(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        raw = json.loads(z.read("submission.json"))
    preds = raw["predictions"] if isinstance(raw, dict) else raw
    return {int(x["id"]): x for x in preds}


def load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["id"])] = row
    return rows


def is_zero(pred: dict) -> bool:
    return (pred.get("pandas_query") or "").strip() == "result = 0.0"


def is_multi(code: str) -> bool:
    c = (code or "").strip()
    return (
        "df1" in c
        or c.count("iloc") >= 2
        or "idxmax" in c
        or "idxmin" in c
        or (" + " in c and "iloc" in c)
        or (" - " in c and "iloc" in c)
    )


def branch_guess(pred: dict) -> str:
    """Infer branch from program shape (submissions have no source field)."""
    code = (pred.get("pandas_query") or "").strip()
    if code == "result = 0.0":
        return "zero"
    if "idxmax" in code or "idxmin" in code:
        return "screen"
    if "df1" in code or (code.count("iloc") >= 2 and ("+" in code or "-" in code or "/" in code or "max(" in code or "min(" in code or "sum(" in code)):
        # compose / ratio_divide / plan multi
        if "/" in code and code.count("iloc") == 2 and "df1" not in code:
            # could be plan or ratio on one df
            pass
        if "df1" in code:
            # plan often uses df1..; compose also does
            if re.search(r"df\d+\s*/\s*df\d+", code) or " / " in code:
                return "compose_or_plan_ratio"
            return "compose_or_plan"
        return "compose_or_multi"
    # single-cell
    if "astype" in code or "replace" in code:
        return "lookup_or_locate"  # synthesized scale dance
    if "iloc" in code:
        return "single_cell"
    return "other"


COUNT_RE = re.compile(
    r"(có\s+bao\s+nhiêu|bao\s+nhiêu\s+năm|bao\s+nhiêu\s+công\s+ty|"
    r"số\s+năm|số\s+công\s+ty)",
    re.I,
)
ABSENT_HINT = re.compile(
    r"(thù\s*lao|thu\s*nhập\s*của|ông|bà|ngành|ghi\s*chú|thuyết\s*minh|"
    r"cam\s*kết|chi\s*tiết|theo\s*khoản)",
    re.I,
)


def main() -> None:
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    zips = {
        "pad4": load_preds("evidence_pad4.zip"),
        "shape": load_preds("shape_lock.zip"),
        "declare": load_preds("label_declare.zip"),
        "reorder": load_preds("label_reorder.zip"),
        "metric_years": load_preds("metric_years.zip"),
        "compose": load_preds("compose.zip"),
        "label_rescan": load_preds("label_rescan.zip"),
        "gen_new": load_preds("gen_new.zip"),
    }

    planned_all = load_jsonl(ROOT / "artifacts" / "planned.jsonl")
    located_all = load_jsonl(ROOT / "artifacts" / "located.jsonl")
    gen_all = load_jsonl(ROOT / "artifacts" / "generated_full.jsonl")

    planned_ok = {i: r for i, r in planned_all.items() if r.get("ok") and r.get("value") is not None}
    located_ok = {i: r for i, r in located_all.items() if r.get("ok") and r.get("value") is not None}
    gen_ok = {i: r for i, r in gen_all.items() if r.get("ok") and r.get("value") is not None}

    # ---- 1. Branch-shape stats across zips ----
    p("=" * 72)
    p("1. BRANCH-SHAPE STATS (inferred from pandas_query)")
    p("=" * 72)
    for name, preds in zips.items():
        ctr = Counter(branch_guess(pr) for pr in preds.values())
        zeros = sum(1 for pr in preds.values() if is_zero(pr))
        nonzero = 1012 - zeros
        multi = sum(1 for pr in preds.values() if is_multi(pr.get("pandas_query") or ""))
        p(f"\n{name}: zeros={zeros} nonzero={nonzero} multi_ilike={multi}")
        for k, v in ctr.most_common():
            p(f"  {k:24s} {v}")

    # answer equality across zips (exec-relevant)
    p("\n--- Answer / code deltas (vs shape_lock best EXEC) ---")
    base = zips["shape"]
    for name, preds in zips.items():
        if name == "shape":
            continue
        ans_diff = sum(
            1
            for i in parsed
            if not is_correct(float(base[i]["answer"] or 0), float(preds[i]["answer"] or 0))
        )
        code_diff = sum(
            1
            for i in parsed
            if (base[i].get("pandas_query") or "") != (preds[i].get("pandas_query") or "")
        )
        both_nz_diff = sum(
            1
            for i in parsed
            if not is_zero(base[i])
            and not is_zero(preds[i])
            and not is_correct(float(base[i]["answer"] or 0), float(preds[i]["answer"] or 0))
        )
        became_z = sum(1 for i in parsed if not is_zero(base[i]) and is_zero(preds[i]))
        left_z = sum(1 for i in parsed if is_zero(base[i]) and not is_zero(preds[i]))
        p(
            f"{name:14s} ans_diff={ans_diff:4d} code_diff={code_diff:4d} "
            f"both_nz_diff={both_nz_diff:4d} became_zero={became_z:3d} left_zero={left_z:3d}"
        )

    # declare vs reorder (TABLES only expect)
    ans_same_dr = sum(
        1
        for i in parsed
        if is_correct(float(zips["declare"][i]["answer"] or 0), float(zips["reorder"][i]["answer"] or 0))
    )
    code_same_dr = sum(
        1
        for i in parsed
        if (zips["declare"][i].get("pandas_query") or "")
        == (zips["reorder"][i].get("pandas_query") or "")
    )
    p(f"\ndeclare vs reorder: ans_same={ans_same_dr}/1012 code_same={code_same_dr}/1012")

    # tables list length / order diffs
    def tbl_sig(pr):
        return tuple(pr.get("relevant_tables") or [])

    pad4_vs_decl = sum(1 for i in parsed if tbl_sig(zips["pad4"][i]) != tbl_sig(zips["declare"][i]))
    pad4_vs_reord = sum(1 for i in parsed if tbl_sig(zips["pad4"][i]) != tbl_sig(zips["reorder"][i]))
    decl_vs_reord = sum(1 for i in parsed if tbl_sig(zips["declare"][i]) != tbl_sig(zips["reorder"][i]))
    avg_len = {
        n: sum(len(tbl_sig(pr)) for pr in preds.values()) / 1012 for n, preds in zips.items()
    }
    p(f"tables list differ: pad4≠declare={pad4_vs_decl} pad4≠reorder={pad4_vs_reord} declare≠reorder={decl_vs_reord}")
    p(f"avg relevant_tables len: { {k: round(v, 3) for k, v in avg_len.items()} }")

    # ---- shape-locked outcomes on latest EXEC-identical packs ----
    p("\n" + "=" * 72)
    p("2. SHAPE_LOCK / COMPOSE / SCREEN (live parsers on shape_lock answers)")
    p("=" * 72)
    elig_ids = [i for i, q in parsed.items() if compose.eligible(q) is not None]
    scr_ids = [i for i, q in parsed.items() if compose.screen_shape(q) is not None]
    rat_ids = [i for i, q in parsed.items() if ratio_mod.eligible(q)]
    p(f"compose.eligible={len(elig_ids)} screen_shape={len(scr_ids)} ratio.eligible={len(rat_ids)}")

    def shape_buckets(preds):
        e_z, e_m, e_o = [], [], []
        s_z, s_m, s_o = [], [], []
        for i in elig_ids:
            code = preds[i].get("pandas_query") or ""
            if is_zero(preds[i]):
                e_z.append(i)
            elif is_multi(code):
                e_m.append(i)
            else:
                e_o.append(i)
        for i in scr_ids:
            code = preds[i].get("pandas_query") or ""
            if is_zero(preds[i]):
                s_z.append(i)
            elif is_multi(code) or "idxmax" in code or "idxmin" in code:
                s_m.append(i)
            else:
                s_o.append(i)
        return e_z, e_m, e_o, s_z, s_m, s_o

    for name in ("pad4", "shape", "declare", "reorder", "metric_years", "compose", "label_rescan"):
        e_z, e_m, e_o, s_z, s_m, s_o = shape_buckets(zips[name])
        p(
            f"{name:14s} elig zero={len(e_z):3d} multi={len(e_m):3d} other(single!)={len(e_o):3d} | "
            f"scr zero={len(s_z):3d} multi={len(s_m):3d} other={len(s_o):3d}"
        )

    e_z, e_m, e_o, s_z, s_m, s_o = shape_buckets(zips["shape"])
    p(f"\nelig_zero ids ({len(e_z)}): {e_z}")
    p(f"elig_other (single shipped under lock) ({len(e_o)}): {e_o}")
    p(f"scr_zero ids ({len(s_z)}): {s_z}")

    # Offline resolve on elig zeros — how many actually resolve today?
    p("\n--- Offline compose.resolve on elig_zero ---")
    resolve_ok = []
    resolve_none = []
    resolve_ok_by_op: Counter = Counter()
    resolve_none_by_op: Counter = Counter()
    metric_broken = []
    for i in e_z:
        q = parsed[i]
        el = compose.eligible(q)
        op, axis = el
        got = compose.resolve(q, store, retriever)
        metric = extract_metric(q.question) or ""
        if got is None:
            resolve_none.append(i)
            resolve_none_by_op[op] += 1
            # crude metric-break signals
            if metric.strip() in ("", "và") or metric.startswith("và ") or re.search(r"\b20\d{2}\b", metric):
                metric_broken.append((i, metric[:60], op, axis))
        else:
            resolve_ok.append((i, got.op, got.value if hasattr(got, "value") else None))
            resolve_ok_by_op[got.op] += 1
    p(f"resolve_ok={len(resolve_ok)} resolve_none={len(resolve_none)}")
    p(f"  ok by op: {dict(resolve_ok_by_op)}")
    p(f"  none by op: {dict(resolve_none_by_op)}")
    if resolve_ok:
        p(f"  resolve_ok but still zero in shape_lock (!): {[x[0] for x in resolve_ok]}")
        p("  (implies submit path differs — filter / impossible / order / top_k)")
    p(f"metric_broken among none ({len(metric_broken)}):")
    for row in metric_broken[:25]:
        p(f"  id={row[0]} op={row[2]}/{row[3]} metric={row[1]!r}")

    # Classify resolve_none by axis + wording
    p("\n--- elig_zero taxonomy (manual heuristics) ---")
    tax = Counter()
    tax_ids = defaultdict(list)
    for i in resolve_none:
        q = parsed[i]
        el = compose.eligible(q)
        op, axis = el
        text = q.question.lower()
        if "tăng" in text and op == "sum":
            key = "false_sum_growth"
        elif "bao nhiêu" in text or "có bao nhiêu" in text:
            key = "count_misrouted"
        elif axis == "ticker" and op == "diff" and len(q.tickers) == 2:
            key = "diff_2ticker"
        elif axis == "year" and op == "diff" and len(q.years) == 2:
            key = "diff_2year"
        elif axis == "year" and op in ("sum", "avg", "max", "min") and len(q.years) >= 3:
            key = "agg_multi_year"
        elif "trong số" in text or "xét các" in text or "có " in text[:40]:
            key = "nested_filter_sounding"
        else:
            key = f"other_{op}_{axis}"
        tax[key] += 1
        tax_ids[key].append(i)
    for k, v in tax.most_common():
        p(f"  {k:28s} {v:3d}  ids={tax_ids[k][:12]}{'...' if len(tax_ids[k])>12 else ''}")

    # ---- 3. Impossible / wrong-shape shipping ----
    p("\n" + "=" * 72)
    p("3. IMPOSSIBLE MAGNITUDE / WRONG SHAPE STILL SHIPPING (shape_lock)")
    p("=" * 72)
    DONG_CEILING = 1e16
    RATIO_UNITS = frozenset({"phan_tram", "lan", "vong"})
    shape = zips["shape"]
    impossible = []
    ratio_bogus = []
    wrong_shape_single = []  # eligible but shipped single cell (should be 0 under lock)
    for i, q in parsed.items():
        pr = shape[i]
        if is_zero(pr):
            continue
        ans = float(pr["answer"] or 0)
        code = pr.get("pandas_query") or ""
        if q.unit_scale and abs(ans) * q.unit_scale > DONG_CEILING:
            impossible.append((i, ans, q.unit_scale, q.target_unit, q.question[:70]))
        if q.target_unit in RATIO_UNITS and abs(ans) > 1000:
            ratio_bogus.append((i, ans, q.target_unit, q.question[:70]))
        if compose.eligible(q) is not None and not is_multi(code) and "idxmax" not in code:
            wrong_shape_single.append(i)
        if compose.screen_shape(q) is not None and "idxmax" not in code and "idxmin" not in code and not is_multi(code):
            wrong_shape_single.append(i)

    p(f"dong-ceiling impossible shipping: {len(impossible)}")
    for row in impossible[:20]:
        p(f"  id={row[0]} ans={row[1]} scale={row[2]} unit={row[3]} | {row[4]}")
    p(f"ratio-unit |ans|>1000 shipping: {len(ratio_bogus)}")
    for row in ratio_bogus[:15]:
        p(f"  id={row[0]} ans={row[1]} unit={row[2]} | {row[3]}")
    p(f"shape-locked but single-cell still: {sorted(set(wrong_shape_single))}")

    # Magnitude heuristics for currency singles: answer vs unit mismatch bands
    # e.g. asking tỷ but answering như đồng đã chia scale sai
    currency_single = [
        i
        for i, q in parsed.items()
        if q.unit_scale and L.is_single_lookup(q.question) and not is_zero(shape[i])
    ]
    huge = []
    for i in currency_single:
        q = parsed[i]
        ans = abs(float(shape[i]["answer"] or 0))
        # after unit_scale division, typical answers for ty are 1..1e6; for trieu 1..1e9
        if q.target_unit in ("ty", "nghin_ty", "tram_ty") and ans > 1e9:
            huge.append((i, ans, q.target_unit, q.question[:60]))
        if q.target_unit == "trieu" and ans > 1e12:
            huge.append((i, ans, q.target_unit, q.question[:60]))
        if q.target_unit in ("dong",) and ans > 1e18:
            huge.append((i, ans, q.target_unit, q.question[:60]))
    p(f"\ncurrency singles with huge post-scale ans: {len(huge)}")
    for row in huge[:20]:
        p(f"  id={row[0]} ans={row[1]:.4g} unit={row[2]} | {row[3]}")

    # ---- 4. planned / generated coverage vs zeros ----
    p("\n" + "=" * 72)
    p("4. ARTIFACT COVERAGE vs ZEROS (shape_lock)")
    p("=" * 72)
    zeros = [i for i in parsed if is_zero(shape[i])]
    p(f"shape_lock zeros: {len(zeros)}")
    p(f"planned.jsonl rows={len(planned_all)} usable={len(planned_ok)}")
    p(f"located.jsonl rows={len(located_all)} usable={len(located_ok)}")
    p(f"generated_full.jsonl rows={len(gen_all)} usable={len(gen_ok)}")

    z_plan = [i for i in zeros if i in planned_ok]
    z_loc = [i for i in zeros if i in located_ok]
    z_gen = [i for i in zeros if i in gen_ok]
    z_any = [i for i in zeros if i in planned_ok or i in located_ok or i in gen_ok]
    z_none = [i for i in zeros if i not in planned_ok and i not in located_ok and i not in gen_ok]
    p(f"zeros WITH usable plan: {len(z_plan)}  locate: {len(z_loc)}  gen: {len(z_gen)}")
    p(f"zeros with ANY model artifact: {len(z_any)}")
    p(f"zeros with NO usable model artifact: {len(z_none)}")

    # Why plan/gen didn't ship: shape_lock blocks locate/fallback; plan/llm should still run
    # Check if zeros that have plan are shape-locked
    z_plan_locked = [i for i in z_plan if compose.eligible(parsed[i]) or compose.screen_shape(parsed[i])]
    z_plan_free = [i for i in z_plan if i not in z_plan_locked]
    z_gen_locked = [i for i in z_gen if compose.eligible(parsed[i]) or compose.screen_shape(parsed[i])]
    z_gen_free = [i for i in z_gen if i not in z_gen_locked]
    p(f"zeros∩plan: shape_locked={len(z_plan_locked)} free={len(z_plan_free)} free_ids={z_plan_free[:30]}")
    p(f"zeros∩gen:  shape_locked={len(z_gen_locked)} free={len(z_gen_free)} free_ids={z_gen_free[:30]}")

    # Compare plan/gen values vs pad4 (pre-lock) — did lock zero out a previously graded-looking multi?
    pad4 = zips["pad4"]
    lock_killed_nonzero = [
        i for i in zeros if not is_zero(pad4[i]) and (compose.eligible(parsed[i]) or compose.screen_shape(parsed[i]))
    ]
    p(f"\npad4 nonzero → shape zero under lock: {len(lock_killed_nonzero)}")
    # among those, how many still have plan/gen that could have filled?
    killed_with_plan = [i for i in lock_killed_nonzero if i in planned_ok]
    killed_with_gen = [i for i in lock_killed_nonzero if i in gen_ok]
    p(f"  of those, have plan={len(killed_with_plan)} gen={len(killed_with_gen)}")
    p(f"  killed_with_plan ids: {killed_with_plan[:40]}")
    p(f"  killed_with_gen ids: {killed_with_gen[:40]}")

    # Sample plan values on free zeros — are they plausible?
    p("\n--- sample: free zeros that have plan but still 0 (should have shipped?) ---")
    for i in z_plan_free[:15]:
        q = parsed[i]
        row = planned_ok[i]
        p(
            f"  id={i} plan_val={row.get('value')} unit={q.target_unit} "
            f"single={L.is_single_lookup(q.question)} | {q.question[:75]}"
        )

    # ---- 5. count questions + ABSENT heuristics ----
    p("\n" + "=" * 72)
    p("5. COUNT QUESTIONS + ABSENT SINGLE-CELL HEURISTICS")
    p("=" * 72)
    count_ids = [i for i, q in parsed.items() if COUNT_RE.search(q.question)]
    p(f"COUNT_RE matches: {len(count_ids)}")
    count_by_state = Counter()
    for i in count_ids:
        q = parsed[i]
        pr = shape[i]
        ans = float(pr["answer"] or 0)
        el = compose.eligible(q) is not None
        sc = compose.screen_shape(q) is not None
        if is_zero(pr):
            st = "zero"
        elif ans == int(ans) and 0 <= ans <= 30:
            st = "small_int_plausible_count"
        elif abs(ans) > 1000:
            st = "money_shaped_wrong"
        else:
            st = "other_nonzero"
        count_by_state[st] += 1
        if st != "small_int_plausible_count":
            p(
                f"  [{st}] id={i} ans={ans} elig={el} scr={sc} single={L.is_single_lookup(q.question)} "
                f"| {q.question[:80]}"
            )
    p(f"count state: {dict(count_by_state)}")

    # ABSENT-style: currency single, not shape locked, weak label / empty / period header
    p("\n--- ABSENT / weak single-cell audit (re-choose live) ---")
    absentish = []
    weak_cats = Counter()
    for i, q in parsed.items():
        if not q.unit_scale or not L.is_single_lookup(q.question):
            continue
        if compose.eligible(q) or compose.screen_shape(q):
            continue
        if is_zero(shape[i]):
            continue
        # live choose top-10
        bm25 = [h.key for h in retriever.search(q, top_k=15)]
        # use corroborator-lite via lookup.find on top keys
        best = None
        for key in bm25[:10]:
            found = L.find(store.rows(key), q)
            if found is None:
                continue
            if best is None or found.score > best[1].score:
                best = (key, found)
        label = best[1].label if best else ""
        score = best[1].score if best else 0.0
        cat = None
        if best is None:
            cat = "no_find_top10"
        elif not (label or "").strip():
            cat = "empty_label"
        elif re.search(r"(đầu năm|cuối năm|tại ngày|số dư cuối|số dư đầu)", label, re.I) and not re.search(
            r"(tiền và tương đương tiền)", label, re.I
        ):
            # period-ish without being the cash line itself when Q asks cash end
            if "cuối năm" in q.question.lower() and "đầu năm" in label.lower():
                cat = "period_flip"
            elif re.fullmatch(r"(tại ngày cuối năm|số dư cuối năm|cuối năm)", label.strip(), re.I):
                cat = "period_header"
        elif re.search(r"^(tổng cộng|tổng tài sản)$", label.strip(), re.I):
            metric = (extract_metric(q.question) or "").lower()
            if "tổng cộng" not in metric and "tổng tài sản" not in metric:
                cat = "bare_total_mismatch"
        elif score < 0.4:
            cat = "score_lt_0.4"
        elif score < 0.75 and ABSENT_HINT.search(q.question):
            cat = "person_or_noteish_lowscore"
        if cat:
            weak_cats[cat] += 1
            absentish.append((i, cat, score, label[:50], float(shape[i]["answer"] or 0), q.question[:70]))

    p(f"weak/ABSENT-ish shipping currency singles: {len(absentish)}")
    for k, v in weak_cats.most_common():
        p(f"  {k}: {v}")
    for cat in ("empty_label", "period_header", "period_flip", "bare_total_mismatch", "no_find_top10"):
        rows = [r for r in absentish if r[1] == cat]
        p(f"\n  detail {cat} ({len(rows)}):")
        for r in rows[:12]:
            p(f"    id={r[0]} score={r[2]:.2f} label={r[3]!r} ans={r[4]} | {r[5]}")

    # ---- 6. Diff vs older leaps: compose / label_rescan mined? ----
    p("\n" + "=" * 72)
    p("6. WHAT OLDER LEAPS ALREADY MINED (compose → label_rescan → shape)")
    p("=" * 72)
    pairs = [
        ("compose", "label_rescan"),
        ("label_rescan", "pad4"),
        ("pad4", "shape"),
        ("shape", "metric_years"),
        ("shape", "declare"),
        ("gen_new", "label_rescan"),
    ]
    for a, b in pairs:
        A, B = zips[a], zips[b]
        changed = [i for i in parsed if not is_correct(float(A[i]["answer"] or 0), float(B[i]["answer"] or 0))]
        # classify changed by shape
        elig_ch = sum(1 for i in changed if compose.eligible(parsed[i]))
        scr_ch = sum(1 for i in changed if compose.screen_shape(parsed[i]))
        single_ch = sum(1 for i in changed if L.is_single_lookup(parsed[i].question))
        to_zero = sum(1 for i in changed if is_zero(B[i]) and not is_zero(A[i]))
        from_zero = sum(1 for i in changed if is_zero(A[i]) and not is_zero(B[i]))
        p(
            f"{a}→{b}: ans_changed={len(changed)} elig={elig_ch} scr={scr_ch} "
            f"single={single_ch} to_zero={to_zero} from_zero={from_zero}"
        )

    # Overlap: questions compose leap won that shape still has
    # Approximate: nonzero multi in both compose and shape
    compose_multi = {i for i in parsed if is_multi(zips["compose"][i].get("pandas_query") or "")}
    shape_multi = {i for i in parsed if is_multi(zips["shape"][i].get("pandas_query") or "")}
    p(f"\nmulti programs: compose_zip={len(compose_multi)} shape={len(shape_multi)} "
      f"intersection={len(compose_multi & shape_multi)} only_compose={len(compose_multi - shape_multi)} "
      f"only_shape={len(shape_multi - compose_multi)}")

    # label_rescan signature: single-cell currency where pad4/shape still nonzero lookup-like
    # Can't fully recover without source tags; compare rescan-era vs gen_new singles that changed
    gen_to_rescan = [
        i
        for i in parsed
        if not is_correct(float(zips["gen_new"][i]["answer"] or 0), float(zips["label_rescan"][i]["answer"] or 0))
    ]
    still_same_as_rescan = sum(
        1
        for i in gen_to_rescan
        if is_correct(float(zips["label_rescan"][i]["answer"] or 0), float(zips["shape"][i]["answer"] or 0))
    )
    p(f"gen_new→label_rescan changed={len(gen_to_rescan)}; of those still same in shape={still_same_as_rescan}")

    # ---- 7. Plan/LLM on elig zeros — potential unlock without resolve fix ----
    p("\n" + "=" * 72)
    p("7. UNLOCK PATHS FOR ELIG_ZERO WITHOUT NEW RESOLVE")
    p("=" * 72)
    for i in e_z:
        has_p = i in planned_ok
        has_g = i in gen_ok
        q = parsed[i]
        el = compose.eligible(q)
        p(
            f"  id={i:4d} elig={el} plan={has_p} gen={has_g} "
            f"tickers={q.tickers} years={q.years} | {q.question[:70]}"
        )

    # ---- 8. Screen zeros ----
    p("\n" + "=" * 72)
    p("8. SCREEN ZEROS — resolve_screen live")
    p("=" * 72)
    for i in s_z:
        q = parsed[i]
        sh = compose.screen_shape(q)
        got = compose.resolve_screen(q, store, retriever)
        p(
            f"  id={i} shape={sh} resolve={'OK' if got else 'None'} "
            f"plan={i in planned_ok} gen={i in gen_ok} | {q.question[:75]}"
        )

    # ---- 9. Graded-pool estimate reminder ----
    p("\n" + "=" * 72)
    p("9. SCORE MATH REMINDER")
    p("=" * 72)
    p("Public ranks by EXECUTION_ACCURACY only; N_graded ≈ 506 (half of 1012).")
    p("1 graded Q ≈ 1/506 ≈ 0.001976 EXEC.")
    p("shape_lock EXEC=0.2846 → ~144 graded correct.")
    p("yyy EXEC≈0.2885 → ~146 graded; gap ≈ 2 graded Q.")
    p("Private: retrieval + answer + execution (weights not published in official README).")

    # ---- 10. Ranked opportunity sizes (computed) ----
    p("\n" + "=" * 72)
    p("10. COMPUTED RESERVOIR SIZES (for ranking)")
    p("=" * 72)
    # count wrong-money
    n_count_money = count_by_state.get("money_shaped_wrong", 0)
    n_count_other = count_by_state.get("other_nonzero", 0) + count_by_state.get("zero", 0)
    n_false_sum = tax.get("false_sum_growth", 0)
    n_diff2 = tax.get("diff_2ticker", 0) + tax.get("diff_2year", 0)
    n_metric_break = len(metric_broken)
    n_plan_on_elig_zero = len([i for i in e_z if i in planned_ok or i in gen_ok])
    n_weak = len(absentish)
    n_free_model_zero = len(z_plan_free) + len([i for i in z_gen_free if i not in z_plan_free])

    p(f"elig_zero reservoir: {len(e_z)} (resolve_none={len(resolve_none)} resolve_ok_bug={len(resolve_ok)})")
    p(f"  false_sum_growth: {n_false_sum}")
    p(f"  diff_2ticker/year: {n_diff2}")
    p(f"  metric_broken among none: {n_metric_break}")
    p(f"  elig_zero with plan|gen artifact: {n_plan_on_elig_zero}")
    p(f"scr_zero: {len(s_z)} (all resolve_screen None historically)")
    p(f"count questions total={len(count_ids)} money_wrong={n_count_money} other_non_plausible={n_count_other}")
    p(f"weak/ABSENT shipping singles: {n_weak}")
    p(f"free (non-shape-lock) zeros with model artifact unused: plan_free={len(z_plan_free)} gen_free={len(z_gen_free)}")
    p(f"ratio-filter screens left alone (do not attack): {len([i for i,q in parsed.items() if compose.screen_shape(q) and compose.screen_shape(q)[2]])}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
