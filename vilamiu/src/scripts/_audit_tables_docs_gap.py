"""Offline audit: TABLES/DOCS packaging gaps across recent submission zips."""

from __future__ import annotations

import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"

ZIPS = [
    "evidence_pad4.zip",
    "label_declare.zip",
    "label_reorder.zip",
    "shape_lock.zip",
    "evidence_refs.zip",
    "label_rescan.zip",
]


def load(name: str) -> list[dict]:
    with zipfile.ZipFile(SUB / name) as z:
        return json.loads(z.read("submission.json"))


def fmt_stats(recs: list[dict], label: str) -> None:
    lens = [len(r["relevant_tables"]) for r in recs]
    dlen = [len(r["relevant_docs"]) for r in recs]
    formats: Counter[str] = Counter()
    line_nums: list[int] = []
    for r in recs:
        for t in r["relevant_tables"]:
            if "|table_" in t:
                formats["prefixed"] += 1
            elif "|" in t:
                part = t.split("|", 1)[1]
                if part.isdigit():
                    formats["bare_line"] += 1
                    line_nums.append(int(part))
                else:
                    formats["other"] += 1
            else:
                formats["no_pipe"] += 1
    print(f"\n=== {label} ===")
    print(
        f"n={len(recs)}  tables: mean={statistics.mean(lens):.3f} "
        f"median={statistics.median(lens)} min={min(lens)} max={max(lens)}"
    )
    print(f"  table hist: {dict(sorted(Counter(lens).items()))}")
    print(
        f"docs: mean={statistics.mean(dlen):.3f} median={statistics.median(dlen)} "
        f"min={min(dlen)} max={max(dlen)}"
    )
    print(f"  doc hist (top): {dict(sorted(Counter(dlen).items())[:20])}")
    print(f"formats: {dict(formats)}")
    if line_nums:
        print(
            f"line nums: min={min(line_nums)} max={max(line_nums)} "
            f"mean={statistics.mean(line_nums):.1f} "
            f"pct<=100={sum(x<=100 for x in line_nums)/len(line_nums):.1%} "
            f"pct>=1000={sum(x>=1000 for x in line_nums)/len(line_nums):.1%}"
        )
    # position of evidence csv vs declared
    evid_in_refs = []
    for r in recs:
        evid_docs = []
        for e in r.get("evidence") or []:
            # data/DOC_table_N.csv
            name = e["csv_path"].split("/")[-1]
            if name.endswith(".csv") and "_table_" in name:
                base = name[: -len(".csv")]
                doc, tid = base.rsplit("_table_", 1)
                evid_docs.append((doc, tid))
        refs = r["relevant_tables"]
        # we can only check doc membership of evidence
        ref_docs = [t.split("|", 1)[0] for t in refs]
        if evid_docs:
            ok = all(d in ref_docs for d, _ in evid_docs)
            evid_in_refs.append(ok)
    if evid_in_refs:
        print(
            f"evidence docs covered by relevant_tables: "
            f"{sum(evid_in_refs)}/{len(evid_in_refs)} "
            f"({sum(evid_in_refs)/len(evid_in_refs):.1%})"
        )


def pair(a: str, b: str, data: dict[str, list[dict]]) -> None:
    bya = {r["id"]: r for r in data[a]}
    byb = {r["id"]: r for r in data[b]}
    same_ans = sum(
        1
        for i in bya
        if bya[i]["answer"] == byb[i]["answer"]
        and bya[i]["pandas_query"] == byb[i]["pandas_query"]
    )
    same_list = sum(1 for i in bya if bya[i]["relevant_tables"] == byb[i]["relevant_tables"])
    same_set = sum(
        1 for i in bya if set(bya[i]["relevant_tables"]) == set(byb[i]["relevant_tables"])
    )
    order_only = sum(
        1
        for i in bya
        if set(bya[i]["relevant_tables"]) == set(byb[i]["relevant_tables"])
        and bya[i]["relevant_tables"] != byb[i]["relevant_tables"]
    )
    set_diff = sum(
        1 for i in bya if set(bya[i]["relevant_tables"]) != set(byb[i]["relevant_tables"])
    )
    same_docs = sum(1 for i in bya if bya[i]["relevant_docs"] == byb[i]["relevant_docs"])
    jacs = []
    first_diff = 0
    for i in bya:
        sa, sb = set(bya[i]["relevant_tables"]), set(byb[i]["relevant_tables"])
        jacs.append(len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0)
        ta, tb = bya[i]["relevant_tables"], byb[i]["relevant_tables"]
        if ta and tb and ta[0] != tb[0]:
            first_diff += 1
    # length deltas
    denser = sum(
        1 for i in bya if len(byb[i]["relevant_tables"]) > len(bya[i]["relevant_tables"])
    )
    shorter = sum(
        1 for i in bya if len(byb[i]["relevant_tables"]) < len(bya[i]["relevant_tables"])
    )
    print(
        f"{a} vs {b}: same_ans+query={same_ans}/{len(bya)} "
        f"same_list={same_list} same_set={same_set} order_only={order_only} "
        f"set_diff={set_diff} first_diff={first_diff} "
        f"mean_jaccard={statistics.mean(jacs):.4f} same_docs={same_docs} "
        f"b_longer={denser} b_shorter={shorter}"
    )


def f2_proxy_math() -> None:
    """What P/R moves close the nguyen gap under F2."""

    print("\n=== F2 arithmetic (macro intuition) ===")
    # Current measured points
    points = [
        ("pad4", 0.3012, 0.5972, 0.4546),
        ("shape_lock", 0.3079, 0.6053, 0.4635),
        ("label_declare", 0.3295, 0.5730, 0.4657),
        ("nguyen", 0.3623, 0.6386, 0.5395),
    ]
    for name, p, r, f2 in points:
        # verify F2
        calc = (5 * p * r) / (4 * p + r) if (p or r) else 0
        print(f"  {name}: P={p:.4f} R={r:.4f} F2_reported={f2:.4f} F2_calc={calc:.4f}")

    # Required P at our R~0.60 to hit 0.54; required R at our P
    target = 0.54
    for r in (0.60, 0.61, 0.64):
        # target = 5pr/(4p+r) => target*(4p+r)=5pr => 4*target*p = 5pr - target*r
        # p*(4*target - 5r) = -target*r => p = target*r / (5r - 4*target)
        denom = 5 * r - 4 * target
        p_need = (target * r) / denom if denom > 0 else float("nan")
        print(f"  to hit F2={target} at R={r:.2f} need P≈{p_need:.3f}")
    for p in (0.31, 0.33, 0.36, 0.40):
        denom = 5 * p - 4 * target
        # target = 5pr/(4p+r) => target*4p + target*r = 5pr => target*4p = r(5p-target)
        # r = 4*target*p / (5p - target)
        denom = 5 * p - target
        r_need = (4 * target * p) / denom if denom > 0 else float("nan")
        print(f"  to hit F2={target} at P={p:.2f} need R≈{r_need:.3f}")

    # Effect of removing noise: if gold avg G tables, declare k, R fixed when k>=G
    # precision ≈ (R * G) / k  in expectation for perfect recall of gold subset
    print("\n=== Length vs precision bound (perfect retrieval of gold subset) ===")
    for g in (1.5, 2.0, 2.5, 3.0):
        for k in (2, 3, 4, 5):
            # if we retrieve all G gold and pad to k with noise: P=G/k, R=1
            # more realistic: R=0.60 means we hit 0.6G gold on avg, declare k
            for r in (0.60, 0.64):
                tp = r * g
                p = tp / k
                f2 = (5 * p * r) / (4 * p + r) if (p or r) else 0
                print(f"  gold≈{g} declare_k={k} R={r:.2f} => P≈{p:.3f} F2≈{f2:.3f}")


def estimate_headroom_docs(data: dict[str, list[dict]]) -> None:
    print("\n=== DOCS pool headroom ===")
    # DOCS_F2 ~0.95; compare lengths and uniqueness across zips
    for name in ["label_reorder.zip", "shape_lock.zip", "evidence_pad4.zip"]:
        recs = data[name]
        dlen = [len(r["relevant_docs"]) for r in recs]
        # docs implied by tables only
        table_docs = []
        for r in recs:
            td = list(dict.fromkeys(t.split("|", 1)[0] for t in r["relevant_tables"]))
            table_docs.append(len(td))
        extras = [len(r["relevant_docs"]) - td for r, td in zip(recs, table_docs)]
        print(
            f"{name}: docs mean={statistics.mean(dlen):.2f} "
            f"from_tables mean={statistics.mean(table_docs):.2f} "
            f"extra_docs mean={statistics.mean(extras):.2f} "
            f"(pct with extras={sum(e>0 for e in extras)/len(extras):.1%})"
        )


def sample_label_prefer(data: dict[str, list[dict]]) -> None:
    """Where label_reorder differs from pad4: first-slot and membership."""
    a, b = data["evidence_pad4.zip"], data["label_reorder.zip"]
    bya = {r["id"]: r for r in a}
    byb = {r["id"]: r for r in b}
    promoted = 0  # b's first table was in a but not first
    new_first = 0
    added_labelish = 0
    for i in bya:
        ta, tb = bya[i]["relevant_tables"], byb[i]["relevant_tables"]
        if not ta or not tb or ta == tb:
            continue
        if tb[0] != ta[0]:
            if tb[0] in ta:
                promoted += 1
            else:
                new_first += 1
        if set(tb) - set(ta):
            added_labelish += 1
    print("\n=== pad4 -> label_reorder structural diffs ===")
    print(f"lists differ: {sum(1 for i in bya if bya[i]['relevant_tables']!=byb[i]['relevant_tables'])}")
    print(f"first promoted within same set: {promoted}")
    print(f"first is newly introduced: {new_first}")
    print(f"sets have new members: {added_labelish}")


def main() -> None:
    data = {z: load(z) for z in ZIPS}
    for z in ZIPS:
        fmt_stats(data[z], z)
    print("\n=== Pairwise ===")
    for a, b in [
        ("evidence_pad4.zip", "label_declare.zip"),
        ("evidence_pad4.zip", "label_reorder.zip"),
        ("label_declare.zip", "label_reorder.zip"),
        ("shape_lock.zip", "label_reorder.zip"),
        ("shape_lock.zip", "evidence_pad4.zip"),
        ("shape_lock.zip", "label_declare.zip"),
    ]:
        pair(a, b, data)
    sample_label_prefer(data)
    estimate_headroom_docs(data)
    f2_proxy_math()


if __name__ == "__main__":
    main()
