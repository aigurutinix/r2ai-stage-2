"""Fast probe: joint solvability of filter ratios on ratio-filter screens.

Skips BM25 — walks tables for each (ticker, year) and runs label find() directly.
"""
from __future__ import annotations

import dataclasses
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from vifin.answering import compose, ratio as ratio_mod
from vifin.answering import lookup as L
from vifin.query.parse import parse_all
from vifin.store import TableKey, TableStore

ROOT = Path(__file__).resolve().parents[1]


def screen_bits(q):
    if len(q.tickers) == 1 and len(q.years) >= 2:
        match = compose.SCREEN_YEAR_RE.search(q.question)
        axis = "year"
    elif len(q.tickers) >= 2 and q.years:
        match = compose.SCREEN_TICKER_RE.search(q.question)
        axis = "ticker"
    else:
        return None
    if match is None:
        return None
    metric = match.group(1).strip(" ,.;:")
    if not metric:
        return None
    want_max = match.group(2).lower() in compose.MAX_WORDS
    return axis, metric, want_max


def build_index(store: TableStore):
    """ticker -> year -> list[TableKey]"""
    idx: dict[str, dict[str, list[TableKey]]] = defaultdict(lambda: defaultdict(list))
    frame = store.frame
    for doc, tid, ticker, year in frame[
        ["doc_name", "table_id", "ticker", "year"]
    ].itertuples(index=False):
        idx[str(ticker)][str(year)].append(TableKey(doc, int(tid)))
    return idx


def best_hit(store, keys: list[TableKey], probe, variants: list[str]):
    best = None
    for key in keys:
        grid = store.rows(key)
        for variant in variants:
            p = dataclasses.replace(probe, question=variant)
            found = L.find(grid, p)
            if found is not None and (best is None or found.score > best[1].score):
                best = (key, found)
    return best


def resolve_ratio_direct(store, idx, ticker: str, year: int, q_proto, filt: str):
    """Try compound then simple ratio using only tables for (ticker, year)."""
    keys = idx.get(str(ticker), {}).get(str(year), [])
    if not keys:
        return "no_tables", None

    probe = dataclasses.replace(
        q_proto,
        tickers=[ticker],
        years=[year],
        question=filt,
        target_unit="lan",
    )

    # Prefer compound on filter text; fall back to full-q named formulas.
    cshape = ratio_mod.compound_shape(probe) or ratio_mod.compound_shape(
        dataclasses.replace(probe, question=q_proto.question)
    )
    if cshape is not None:
        kind, names = cshape
        hits = []
        for name in names:
            hit = best_hit(store, keys, probe, L.metric_variants(name))
            if hit is None:
                return f"compound_miss:{name[:30]}", None
            hits.append(hit)
        # sanity compute
        if kind == "diff_ratio":
            # need 3 values
            amounts = []
            for key, found in hits:
                meta = store.meta(key)
                scale = L.column_scale(
                    store.rows(key), found.column,
                    f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                )
                amounts.append(abs(found.value) * scale)
            if amounts[2] == 0:
                return "compound_zero_den", None
            val = (amounts[0] - amounts[1]) / amounts[2]
        else:
            # avg_den: num + den close + den open
            top_key, top_hit = hits[0]
            den_key, den_hit = hits[1]
            meta_t = store.meta(top_key)
            scale_t = L.column_scale(
                store.rows(top_key), top_hit.column,
                f"{meta_t.unit_page} {meta_t.unit_doc} {meta_t.caption}",
            )
            cols = L.value_columns(store.rows(den_key))
            if len(cols) < 2:
                return "compound_no_open", None
            meta_d = store.meta(den_key)
            grid_d = store.rows(den_key)
            s0 = L.column_scale(grid_d, cols[0], f"{meta_d.unit_page} {meta_d.unit_doc} {meta_d.caption}")
            s1 = L.column_scale(grid_d, cols[1], f"{meta_d.unit_page} {meta_d.unit_doc} {meta_d.caption}")
            v0 = L._parse_cell(grid_d[den_hit.row][cols[0]]) or 0.0
            v1 = L._parse_cell(grid_d[den_hit.row][cols[1]]) or 0.0
            avg = (abs(v0) * s0 + abs(v1) * s1) / 2.0
            if avg == 0:
                return "compound_zero_avg", None
            val = abs(top_hit.value) * scale_t / avg * 100.0
        if abs(val) > ratio_mod.SANITY_LIMIT:
            return "compound_insane", None
        return "compound", round(val, 4)

    shaped = ratio_mod.shape(probe)
    if shaped is None:
        # try stripping trailing screen noise after first clause
        short = filt.split(".")[0].split(",")[0].strip()
        probe2 = dataclasses.replace(probe, question=short)
        shaped = ratio_mod.shape(probe2)
        if shaped is not None:
            probe = probe2
    if shaped is None:
        return "no_shape", None

    num, den = shaped
    top = best_hit(store, keys, probe, L.metric_variants(num))
    bot = best_hit(store, keys, probe, L.metric_variants(den))
    if top is None and bot is None:
        return "miss_both", None
    if top is None:
        return "miss_num", None
    if bot is None:
        return "miss_den", None
    if top[0] == bot[0] and top[1].row == bot[1].row:
        return "same_cell", None

    amounts = []
    for key, found in (top, bot):
        meta = store.meta(key)
        scale = L.column_scale(
            store.rows(key), found.column,
            f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
        )
        amounts.append(abs(found.value) * scale)
    if amounts[1] == 0:
        return "zero_den", None
    val = amounts[0] / amounts[1]
    # report as lần; if looks like percent phrasing leave as-is
    if abs(val) > ratio_mod.SANITY_LIMIT:
        # maybe should be percent of wrong unit — still a miss for ranking
        return "simple_insane", None
    return "simple", round(val, 4)


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    idx = build_index(store)

    with zipfile.ZipFile(ROOT / "submissions" / "helpers.zip") as z:
        raw = json.loads(z.read("submission.json"))
    preds = {p["id"]: p for p in (raw if isinstance(raw, list) else raw["predictions"])}

    ripe = []
    for i, q in parsed.items():
        bits = screen_bits(q)
        if bits is None:
            continue
        axis, filt, want_max = bits
        if not compose.RATIO_FILTER_RE.search(filt):
            continue
        synthetic = dataclasses.replace(q, question=filt)
        has = (
            ratio_mod.shape(synthetic) is not None
            or ratio_mod.compound_shape(synthetic) is not None
            or ratio_mod.shape(q) is not None
            or ratio_mod.compound_shape(q) is not None
        )
        if has:
            ripe.append((i, q, axis, filt, want_max))

    lines: list[str] = []
    p = lines.append
    p(f"ripe ratio-filter screens with known shape: {len(ripe)}")

    tiers = Counter()
    all_ok = []
    maj = []
    none = []
    why = Counter()

    for i, q, axis, filt, want_max in ripe:
        if axis != "ticker":
            tiers["year_axis_skipped"] += 1
            continue
        year = max(q.years)
        tickers = list(q.tickers)
        kinds = []
        ok = 0
        for t in tickers:
            kind, val = resolve_ratio_direct(store, idx, t, year, q, filt)
            why[kind] += 1
            kinds.append((t, kind, val))
            if kind in ("compound", "simple"):
                ok += 1
        n = len(tickers)
        if ok == n:
            tier = "ALL_ok"
            all_ok.append((i, ok, n, filt, kinds, want_max))
        elif ok == 0:
            tier = "NONE_ok"
            none.append((i, ok, n, filt, kinds))
        elif ok >= max(2, (n + 1) // 2):
            tier = "PARTIAL_majority"
            maj.append((i, ok, n, filt, kinds))
        else:
            tier = "PARTIAL_few"
        tiers[tier] += 1

        # helpers branch hint
        code = str(preds[i].get("pandas_query") or "")
        br = "llm" if "def num(" in code or "find_row(" in code else (
            "zero" if code.strip() == "result = 0.0" else "other"
        )
        if tier == "ALL_ok":
            all_ok[-1] = (i, ok, n, filt, kinds, want_max, br)

    p("\n=== JOINT SOLVABILITY (direct label match, ticker-axis)")
    for k, v in tiers.most_common():
        p(f"  {k}: {v}")
    n_tick = sum(v for k, v in tiers.items() if k != "year_axis_skipped")
    p(f"ticker-axis ripe: {n_tick}")
    p(f"ALL_ok: {len(all_ok)} ({100 * len(all_ok) / max(1, n_tick):.0f}%)")
    p(f"PARTIAL_majority (unsafe without gate): {len(maj)}")
    p(f"NONE_ok: {len(none)}")
    p(f"\n=== per-ticker outcomes: {dict(why)}")

    p(f"\n=== ALL_ok detail ({len(all_ok)})")
    for row in all_ok:
        i, ok, n, filt, kinds, want_max, *rest = row
        br = rest[0] if rest else "?"
        vals = "; ".join(f"{t}:{k}={v}" for t, k, v in kinds)
        p(f"  id={i} helpers={br} want_max={want_max} | {filt[:60]}")
        p(f"    {vals}")

    p(f"\n=== PARTIAL_majority sample")
    for i, ok, n, filt, kinds in maj[:12]:
        miss = [(t, k) for t, k, _ in kinds if k not in ("compound", "simple")]
        p(f"  id={i} {ok}/{n} miss={miss} | {filt[:50]}")

    p(f"\n=== NONE_ok miss reasons (top)")
    miss_why = Counter()
    for i, ok, n, filt, kinds in none:
        for _, k, _ in kinds:
            if k not in ("compound", "simple"):
                miss_why[k] += 1
    for k, v in miss_why.most_common(12):
        p(f"  {v:4d}  {k}")

    p("\n=== VERDICT")
    p(
        f"Safe all-operands gate opens AT MOST {len(all_ok)} screens for "
        "ratio-rank. Pass-2 asked-metric lookup still required — graded wins "
        f"<< {len(all_ok)}. Old trap (<2 if loose joint) still applies to "
        f"{len(maj)} majority-only items — keep the gate."
    )
    # How many ALL_ok currently on llm (displace candidates)
    llm_n = sum(1 for row in all_ok if len(row) > 6 and row[6] == "llm")
    p(f"ALL_ok currently answered by LLM branch in helpers: {llm_n}/{len(all_ok)}")

    out = ROOT / "artifacts" / "_probe_ripe_ratio_screen.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for line in lines:
        if line.startswith("===") or (
            line.strip()
            and "id=" not in line
            and not line.strip().startswith("HPG")
            and "miss=" not in line
        ):
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
