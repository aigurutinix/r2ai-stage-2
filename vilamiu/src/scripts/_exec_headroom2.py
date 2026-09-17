"""Follow-up headroom probes; UTF-8 report."""
from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import compose, lookup as L
from vifin.answering.corroborate import Corroborator
from vifin.query.parse import parse_all
from vifin.retrieval.lexical import LexicalRetriever
from vifin.store import TableStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "_exec_headroom2.txt"
lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)


def load(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        raw = json.loads(z.read("submission.json"))
    preds = raw if isinstance(raw, list) else raw["predictions"]
    return {int(x["id"]): x for x in preds}


def zero(pr: dict) -> bool:
    return (pr.get("pandas_query") or "").strip() == "result = 0.0"


def nframes(code: str | None) -> int:
    return len(set(re.findall(r"\bdf\d*\b", code or "")))


def has_idx(code: str | None) -> bool:
    c = code or ""
    return "idxmax" in c or "idxmin" in c


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
    corr = Corroborator(store)

    shape = load("shape_lock.zip")
    pad4 = load("evidence_pad4.zip")
    declare = load("label_declare.zip")
    reorder = load("label_reorder.zip")
    my = load("metric_years.zip")
    compose_z = load("compose.zip")
    rescan = load("label_rescan.zip")

    p("=== Better program classes ===")
    for name, preds in [
        ("pad4", pad4),
        ("shape", shape),
        ("declare", declare),
        ("reorder", reorder),
        ("metric_years", my),
        ("compose_leap", compose_z),
        ("label_rescan", rescan),
    ]:
        c: Counter[str] = Counter()
        for pr in preds.values():
            code = pr.get("pandas_query") or ""
            if zero(pr):
                c["zero"] += 1
            elif has_idx(code):
                c["screen"] += 1
            elif nframes(code) >= 2:
                c["multi_df"] += 1
            elif "iloc" in code:
                c["single"] += 1
            else:
                c["other"] += 1
        avg = sum(len(pr.get("relevant_tables") or []) for pr in preds.values()) / 1012
        p(f"{name:14s} {dict(c)} avg_tbl={avg:.3f}")

    p("\n=== declare/reorder fill shape zeros (same answers as each other) ===")
    filled = [i for i in parsed if zero(shape[i]) and not zero(declare[i])]
    p(f"n={len(filled)} ids={filled}")
    for i in filled:
        q = parsed[i]
        el = compose.eligible(q)
        sc = compose.screen_shape(q)
        code = declare[i].get("pandas_query") or ""
        p(
            f"id={i} elig={el} scr={sc} ans={declare[i]['answer']} "
            f"nf={nframes(code)} idx={has_idx(code)} | {q.question[:100]}"
        )

    # Are declare fills same as pad4 (fallback restored)?
    same_as_pad4 = sum(
        1
        for i in filled
        if abs(float(declare[i]["answer"] or 0) - float(pad4[i]["answer"] or 0)) < 1e-6
        or (
            float(pad4[i]["answer"] or 0) != 0
            and abs(float(declare[i]["answer"] or 0) - float(pad4[i]["answer"] or 0))
            / abs(float(pad4[i]["answer"] or 0))
            < 2e-4
        )
    )
    p(f"filled answers approx==pad4: {same_as_pad4}/{len(filled)}")

    p("\n=== metric_years fills shape zeros ===")
    filled2 = [i for i in parsed if zero(shape[i]) and not zero(my[i])]
    ops: Counter[str] = Counter()
    for i in filled2:
        el = compose.eligible(parsed[i])
        ops[el[0] if el else "not_elig"] += 1
    p(f"n={len(filled2)} by_elig_op={dict(ops)}")
    p(f"ids={filled2}")

    p("\n=== 541/577 resolve_ok but shape zero ===")
    for i in (541, 577):
        q = parsed[i]
        p(f"--- id={i}")
        p(f"Q: {q.question}")
        p(f"eligible={compose.eligible(q)} screen={compose.screen_shape(q)}")
        got = compose.resolve(q, store, retriever)
        if got is None:
            p("resolve=None")
        else:
            p(
                f"resolve op={got.op} axis={got.axis} score={got.score} "
                f"values={got.values} keys={got.keys}"
            )
            p(f"code: {got.code[:200]}")
        p(f"shape: ans={shape[i]['answer']} code={(shape[i].get('pandas_query') or '')[:120]}")
        p(f"declare: ans={declare[i]['answer']} code={(declare[i].get('pandas_query') or '')[:120]}")
        p(f"metric_years: ans={my[i]['answer']} code={(my[i].get('pandas_query') or '')[:120]}")
        p(f"pad4: ans={pad4[i]['answer']}")

    COUNT_RE = re.compile(
        r"(có\s+bao\s+nhiêu|bao\s+nhiêu\s+năm|bao\s+nhiêu\s+công\s+ty|số\s+năm\b)",
        re.I,
    )
    ez = [
        426, 498, 541, 577, 599, 607, 608, 628, 642, 739, 740, 757, 759, 761,
        789, 808, 809, 816, 833, 840, 847, 872, 875, 887, 888, 891, 893, 894,
        898, 911, 912, 919, 924, 927, 958, 998, 1006,
    ]
    p("\n=== elig_zero true taxonomy ===")
    tax: Counter[str] = Counter()
    ids: dict[str, list[int]] = defaultdict(list)
    for i in ez:
        q = parsed[i]
        el = compose.eligible(q)
        assert el
        op, axis = el
        text = q.question
        nested = bool(
            re.search(
                r"(xét các|trong số các năm có|ở các năm có|có biên |có tỷ lệ |"
                r"thấp hơn trung|lớn hơn trung|có CFO|có LNST)",
                text,
                re.I,
            )
        )
        if COUNT_RE.search(text):
            k = "count"
        elif "tăng" in text.lower() and op == "sum":
            k = "false_sum_growth"
        elif axis == "ticker" and op == "diff" and len(q.tickers) == 2:
            k = "diff_2ticker"
        elif axis == "year" and op == "diff" and len(q.years) == 2:
            k = "diff_2year"
        elif op in ("sum", "avg", "max", "min") and axis == "year":
            k = f"agg_{op}_year_n{len(q.years)}"
        elif op in ("sum", "avg", "max", "min") and axis == "ticker":
            k = f"agg_{op}_ticker_n{len(q.tickers)}"
        elif op == "diff":
            k = f"diff_{axis}_nT{len(q.tickers)}_nY{len(q.years)}"
        else:
            k = f"{op}_{axis}"
        if nested:
            k = "nested_" + k
        tax[k] += 1
        ids[k].append(i)
    for k, v in tax.most_common():
        p(f"{v:2d} {k}: {ids[k]}")

    # metric extract quality on elig_zero
    p("\n=== extract_metric on elig_zero ===")
    for i in ez:
        q = parsed[i]
        m = L.extract_metric(q.question) or ""
        broken = (
            not m.strip()
            or m.strip() in ("và",)
            or m.startswith("và ")
            or bool(re.search(r"\b20\d{2}\b", m))
            or len(m) > 80
        )
        flag = "BROKEN" if broken else "ok"
        p(f"id={i} [{flag}] metric={m[:90]!r}")

    p("\n=== ABSENT via Corroborator.choose (currency single, not shape-locked, nonzero) ===")
    weak: list[tuple] = []
    for i, q in parsed.items():
        if not q.unit_scale or not L.is_single_lookup(q.question):
            continue
        if compose.eligible(q) or compose.screen_shape(q):
            continue
        if zero(shape[i]):
            continue
        searched = [h.key for h in retriever.search(q, top_k=15)]
        picked = corr.choose(q, searched)
        src = "topk"
        if picked is None:
            rescued = corr.keys_matching_label(
                q, retriever.candidate_docs(q), limit=15
            )
            picked = corr.choose(q, rescued) if rescued else None
            src = "rescan" if picked else "absent"
        label = picked.label if picked else ""
        # Corroboration has no score; re-find for score when needed
        score = 0.0
        if picked is not None:
            found = L.find(store.rows(picked.key), q)
            score = found.score if found else 0.0
        cat = None
        if src == "absent":
            cat = "ABSENT_no_label"
        elif not (label or "").strip():
            cat = "empty_label"
        elif re.search(r"^(TỔNG CỘNG|Tổng cộng|Tổng tài sản)$", label.strip()):
            m = (L.extract_metric(q.question) or "").lower()
            if "tổng cộng" not in m and "tổng tài sản" not in m:
                cat = "bare_total"
        elif re.fullmatch(
            r"(Tại ngày cuối năm|Số dư cuối năm|Cuối năm|Tại ngày đầu năm)",
            label.strip(),
            re.I,
        ):
            cat = "period_header"
        elif score < 0.55:
            cat = "low_score_<0.55"
        if cat:
            weak.append((i, cat, score, label[:55], shape[i]["answer"], q.question[:80]))
    wc = Counter(x[1] for x in weak)
    p(f"weak cats={dict(wc)} total={len(weak)}")
    for cat in (
        "ABSENT_no_label",
        "empty_label",
        "bare_total",
        "period_header",
        "low_score_<0.55",
    ):
        rows = [x for x in weak if x[1] == cat]
        p(f"-- {cat} n={len(rows)}")
        for r in rows[:15]:
            p(f"  id={r[0]} sc={r[2]:.2f} lab={r[3]!r} ans={r[4]} | {r[5]}")

    # Coverage of choose vs shipping
    p("\n=== choose coverage on currency singles (shape nonzero / zero) ===")
    cur = [
        i
        for i, q in parsed.items()
        if q.unit_scale
        and L.is_single_lookup(q.question)
        and not (compose.eligible(q) or compose.screen_shape(q))
    ]
    has_choose = no_choose = 0
    nz_no = z_no = 0
    absent_ids = []
    for i in cur:
        q = parsed[i]
        searched = [h.key for h in retriever.search(q, top_k=15)]
        picked = corr.choose(q, searched)
        if picked is None:
            rescued = corr.keys_matching_label(q, retriever.candidate_docs(q), limit=15)
            picked = corr.choose(q, rescued) if rescued else None
        if picked is None:
            no_choose += 1
            absent_ids.append(i)
            if zero(shape[i]):
                z_no += 1
            else:
                nz_no += 1
        else:
            has_choose += 1
    p(f"currency singles not shape-locked: {len(cur)}")
    p(f"  choose hit (top15|rescan): {has_choose}")
    p(f"  ABSENT (no label): {no_choose}  of which shape_nonzero={nz_no} shape_zero={z_no}")
    p(f"  ABSENT ids sample: {absent_ids[:40]}")

    # Gen on elig zeros
    gen: dict[int, dict] = {}
    for line in (ROOT / "artifacts" / "generated_full.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        r = json.loads(line)
        if r.get("ok") and r.get("value") is not None:
            gen[int(r["id"])] = r
    planned: dict[int, dict] = {}
    for line in (ROOT / "artifacts" / "planned.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        r = json.loads(line)
        if r.get("ok") and r.get("value") is not None:
            planned[int(r["id"])] = r

    p("\n=== elig_zero × artifacts (gen should have been allowed under shape_lock) ===")
    for i in ez:
        g = gen.get(i)
        pl = planned.get(i)
        q = parsed[i]
        p(
            f"id={i} gen={'Y:'+str(g['value']) if g else 'N'} "
            f"plan={'Y:'+str(pl['value']) if pl else 'N'} "
            f"pad4={pad4[i]['answer']} shape={shape[i]['answer']} "
            f"unit={q.target_unit}"
        )
        if g and zero(shape[i]):
            # why didn't LLM ship? check reads_no_frame / zero value
            from vifin.submit.validate import reads_no_frame
            from vifin.answering.sandbox import run_query
            from vifin.store import TableKey

            keys = [TableKey(doc, int(tid)) for doc, tid in g["keys"]]
            tables = {name: store.rows(key) for name, key in zip(g["variables"], keys)}
            outcome = run_query(g["code"], tables)
            p(
                f"  gen_reexec ok={outcome.ok} val={getattr(outcome,'value',None)} "
                f"const={reads_no_frame(g['code'])} code={g['code'][:140]!r}"
            )

    # Screen zeros: ratio filter flag
    p("\n=== screen zeros ratio-filter flag ===")
    sz = [380, 413, 422, 475, 496, 499, 500, 506, 522, 543, 575]
    for i in sz:
        q = parsed[i]
        sh = compose.screen_shape(q)
        p(f"id={i} shape={sh} gen={i in gen} plan={i in planned} | {q.question[:100]}")

    # Count pool clean
    p("\n=== COUNT pool (COUNT_RE) shapes ===")
    count_ids = [i for i, q in parsed.items() if COUNT_RE.search(q.question)]
    p(f"n={len(count_ids)}")
    for i in count_ids:
        q = parsed[i]
        ans = float(shape[i]["answer"] or 0)
        if zero(shape[i]):
            st = "zero"
        elif ans == int(ans) and 0 <= ans <= 40:
            st = "plausible_int"
        elif abs(ans) > 100:
            st = "money"
        else:
            st = "other"
        p(
            f"id={i} [{st}] ans={ans} elig={compose.eligible(q)} scr={compose.screen_shape(q)} "
            f"single={L.is_single_lookup(q.question)} tickers={len(q.tickers)} years={q.years} "
            f"| {q.question[:95]}"
        )

    # pad4 vs shape on lock region: how many single->zero vs multi->multi
    p("\n=== pad4→shape lock transition among shape-locked qs ===")
    locked = [
        i
        for i, q in parsed.items()
        if compose.eligible(q) or compose.screen_shape(q)
    ]
    to_zero = keep_multi = keep_single = multi_to_single = single_to_multi = 0
    for i in locked:
        a, b = pad4[i], shape[i]
        ca, cb = a.get("pandas_query") or "", b.get("pandas_query") or ""
        ma = nframes(ca) >= 2 or has_idx(ca)
        mb = nframes(cb) >= 2 or has_idx(cb)
        if zero(b) and not zero(a):
            to_zero += 1
        elif ma and mb:
            keep_multi += 1
        elif (not ma) and (not mb) and not zero(b):
            keep_single += 1
        elif ma and not mb:
            multi_to_single += 1
        elif (not ma) and mb:
            single_to_multi += 1
    p(
        f"locked={len(locked)} to_zero={to_zero} keep_multi={keep_multi} "
        f"keep_single={keep_single} multi→single={multi_to_single} single→multi={single_to_multi}"
    )

    # What fraction of compose_leap multi still in shape?
    p("\n=== compose leap residue ===")
    compose_multi = {
        i for i in parsed if nframes(compose_z[i].get("pandas_query")) >= 2
    }
    shape_multi = {i for i in parsed if nframes(shape[i].get("pandas_query")) >= 2}
    only_c = sorted(compose_multi - shape_multi)
    only_s = sorted(shape_multi - compose_multi)
    p(f"compose_multi={len(compose_multi)} shape_multi={len(shape_multi)}")
    p(f"only_compose (lost/changed): {len(only_c)} sample={only_c[:30]}")
    p(f"only_shape (new since leap): {len(only_s)} sample={only_s[:30]}")

    gen_new = load("gen_new.zip")
    changed = [
        i
        for i in parsed
        if abs(float(gen_new[i]["answer"] or 0) - float(rescan[i]["answer"] or 0)) > 1e-9
    ]
    still = sum(
        1
        for i in changed
        if abs(float(rescan[i]["answer"] or 0) - float(shape[i]["answer"] or 0)) < 1e-9
    )
    p(f"\ngen_new→label_rescan changed={len(changed)}; still identical in shape={still}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        OUT.write_text("\n".join(lines) + f"\n\nCRASH: {e!r}\n", encoding="utf-8")
        raise
