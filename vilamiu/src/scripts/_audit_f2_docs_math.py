"""F2 arithmetic + DOCS headroom proxies (ASCII-only prints)."""

from __future__ import annotations

import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def f2(p: float, r: float) -> float:
    return 0.0 if p == 0 and r == 0 else (5 * p * r) / (4 * p + r)


def main() -> None:
    print("=== Required moves to F2~0.54 (F2-of-means, upper-bound intuition) ===")
    target = 0.54
    for r in (0.60, 0.605, 0.64, 0.70):
        p = (target * r) / (5 * r - 4 * target)
        print(f"  R={r:.3f} needs P={p:.3f} (delta from 0.308: {p - 0.308:+.3f})")
    for p in (0.31, 0.33, 0.36, 0.40, 0.45):
        r = (4 * target * p) / (5 * p - target)
        print(f"  P={p:.3f} needs R={r:.3f} (delta from 0.605: {r - 0.605:+.3f})")

    print("\nNote: F2(macroP, macroR) != leaderboard macro-F2.")
    print(f"  pad4 F2(P,R)={f2(0.3012, 0.5972):.4f} reported 0.4546")
    print(f"  shape F2(P,R)={f2(0.3079, 0.6053):.4f} reported 0.4635")
    print(f"  declare F2(P,R)={f2(0.3295, 0.5730):.4f} reported 0.4657")
    print(f"  nguyen F2(P,R)={f2(0.3623, 0.6386):.4f} reported 0.5395")
    print(f"  gap F2(P,R)={f2(0.3623, 0.6386) - f2(0.3079, 0.6053):+.4f}")
    print(f"  gap reported={0.5395 - 0.4635:+.4f}")

    print("\n=== Implied mean gold size g = P*k/R ===")
    for name, p, r, k in [
        ("pad4", 0.3012, 0.5972, 4.157),
        ("shape", 0.3079, 0.6053, 4.214),
        ("declare3", 0.3295, 0.5730, 3.590),
        ("evidence_refs", 0.37, 0.49, 2.623),
        ("nguyen_if_k2.5", 0.3623, 0.6386, 2.5),
        ("nguyen_if_k3", 0.3623, 0.6386, 3.0),
        ("nguyen_if_k4", 0.3623, 0.6386, 4.0),
    ]:
        g = p * k / r
        print(f"  {name}: k={k:.2f} => g~{g:.2f} tp/q~{p * k:.2f}")

    # Scenario: keep R=0.605, cut noise fillers (lower k) -> what F2?
    print("\n=== If R held at 0.605 while shortening k (perfect-knowledge bound) ===")
    # Current: k=4.214, P=0.308 => tp=1.30. If we drop noise keeping all tp:
    tp = 0.3079 * 4.214
    for k in (4.214, 3.5, 3.0, 2.5, 2.0, 1.5):
        p = min(1.0, tp / k)
        print(f"  k={k:.2f} P={p:.3f} F2(P,R)={f2(p, 0.6053):.3f}")

    # Scenario: keep k=4.2, lift R by adding true tables (displacing noise)
    print("\n=== If k fixed 4.214, displace noise with true tables ===")
    k = 4.214
    for add_tp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tp2 = tp + add_tp
        # g unknown; assume g such that R=tp/g=0.605 => g=tp/0.605
        g = tp / 0.6053
        r = min(1.0, tp2 / g)
        p = min(1.0, tp2 / k)
        print(f"  +{add_tp:.1f} tp/q => P={p:.3f} R={r:.3f} F2={f2(p, r):.3f}")

    # Better: gold g from nguyen-compat assumption g~2.0
    print("\n=== With assumed g=2.0 (near nguyen-implied at k~3) ===")
    g = 2.0
    for tp2 in (1.2, 1.28, 1.4, 1.5, 1.6, 1.8):
        for k2 in (2.5, 3.0, 4.0):
            p = tp2 / k2
            r = tp2 / g
            if p > 1 or r > 1:
                continue
            print(f"  tp={tp2:.2f} k={k2:.1f} => P={p:.3f} R={r:.3f} F2={f2(p, r):.3f}")

    print("\n=== DOCS headroom ===")
    with zipfile.ZipFile(ROOT / "submissions" / "label_reorder.zip") as z:
        recs = json.loads(z.read("submission.json"))
    dlen = [len(r["relevant_docs"]) for r in recs]
    print("doc hist:", dict(sorted(Counter(dlen).items())))
    print(f"single-doc fraction: {sum(x == 1 for x in dlen) / len(dlen):.1%}")

    parsed = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    byid = {q.id: q for q in parsed}
    shallow_multi = deep_multi = 0
    for r in recs:
        q = byid[r["id"]]
        if len(q.tickers) >= 2:
            if len(r["relevant_docs"]) < len(q.tickers):
                shallow_multi += 1
            else:
                deep_multi += 1
    print(f"multi-ticker: docs<tickers={shallow_multi} docs>=tickers={deep_multi}")

    ts = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retr = LexicalRetriever(ts.frame)
    cand_sizes = []
    pool_sizes = []
    deltas = []
    year_only_adjacent = 0
    for q in parsed:
        docs = retr.candidate_docs(q)
        bm25 = [h.key for h in retr.search(q, top_k=10)]
        pool: list[str] = []
        for k in bm25:
            if k.doc_name not in pool:
                pool.append(k.doc_name)
        cand_sizes.append(len(docs))
        pool_sizes.append(len(pool))
        deltas.append(len(set(docs) - set(pool)))
        # would adjacent years add docs when year filter already succeeded?
        if q.years and q.tickers:
            tickers = q.tickers
            years = {str(y) for y in q.years}
            adj = {str(y + 1) for y in q.years} | {str(y - 1) for y in q.years}
            pool_t = []
            for t in tickers:
                for d in retr._docs_by_ticker.get(t, []):
                    y = retr._doc_meta[d][1]
                    if y in years or y in adj:
                        pool_t.append(d)
            if set(pool_t) - set(docs):
                year_only_adjacent += 1

    print(
        f"mean candidate_docs={statistics.mean(cand_sizes):.2f} "
        f"mean BM25-doc-pool={statistics.mean(pool_sizes):.2f} "
        f"mean missing_cand_docs={statistics.mean(deltas):.2f} "
        f"pct with missing={sum(d > 0 for d in deltas) / len(deltas):.1%}"
    )
    print(
        f"qs where forcing adjacent years widens cand_docs: {year_only_adjacent}/{len(parsed)}"
    )
    print(
        f"synera DOCS_F2=0.9684 ours~0.9502 gap~0.018; "
        f"declaring full candidate_docs would add mean {statistics.mean(deltas):.2f} docs "
        f"on {sum(d > 0 for d in deltas) / len(deltas):.0%} of questions"
    )


if __name__ == "__main__":
    main()
