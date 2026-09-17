"""The 94 certainly-wrong answers: what would it take to replace them?

A question whose target unit is `phan_tram` / `lan` / `vong` cannot have an answer
in the billions. At a 0.02% relative tolerance those are not near-misses, they are
zeros — which makes the pool a **free roll**: a replacement that is right earns
+1, a replacement that is wrong costs nothing, because the current figure already
scores nothing.

`_exec_residual_audit.py` asserted "1-ticker ratio.resolve cannot displace any
today". This checks that claim and, more usefully, asks *where* the resolution
stops: no shape at all, shape but a missing operand, or shape and operands but a
value the sanity limit rejects.

Usage:  PYTHONPATH=src python scripts/_probe_ratio_unit_pool.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import ratio as R  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}

    bad = []
    for i, q in parsed.items():
        if q.target_unit not in ("phan_tram", "lan", "vong"):
            continue
        try:
            ans = float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            continue
        if abs(ans) <= 1000:
            continue
        bad.append((i, ans, q))

    lines: list[str] = []
    p = lines.append
    p(f"ratio-unit answers over 1000 (certainly wrong): {len(bad)}")
    p(f"  by unit: {dict(Counter(q.target_unit for _, _, q in bad))}")
    p(f"  by ticker count: {dict(Counter(len(q.tickers) for _, _, q in bad))}")
    p(f"  by year count: {dict(Counter(len(q.years) for _, _, q in bad))}")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ret = LexicalRetriever(store.frame)

    single = [(i, a, q) for i, a, q in bad if len(q.tickers) == 1 and q.years]
    p(f"\nsingle company, at least one year: {len(single)}")

    stage: Counter[str] = Counter()
    unshaped: list[tuple[int, str]] = []
    resolved: list[tuple[int, float, float]] = []
    for i, ans, q in single:
        comp = R.compound_shape(q)
        sh = R.shape(q)
        if comp is None and sh is None:
            stage["no_shape"] += 1
            unshaped.append((i, q.question))
            continue
        stage["compound" if comp else "simple"] += 1
        try:
            if comp is not None:
                got = R.resolve_compound(q, store, ret, 10)
            else:
                got = R.resolve(q, store, ret, 10)
        except Exception as exc:  # a probe, not the pipeline
            stage[f"raised:{type(exc).__name__}"] += 1
            continue
        if got is None:
            stage["shape_but_unresolved"] += 1
        else:
            stage["RESOLVED"] += 1
            resolved.append((i, ans, float(got.value)))

    p(f"\nwhere resolution stops on those {len(single)}:")
    for key, count in stage.most_common():
        p(f"  {key:24s} {count}")

    if resolved:
        p(f"\nreplaceable today ({len(resolved)}):")
        for i, old, new in resolved:
            p(f"  id={i:4d} unit={parsed[i].target_unit:9s} {old:.4g} -> {new:.6g}")
            p(f"        {parsed[i].question[:140]}")

    # What concepts are the unshaped ones asking for? A named-ratio table entry is
    # cheap; knowing which names repeat is the whole question.
    p(f"\nunshaped single-company questions ({len(unshaped)}) — leading concept:")
    concept = re.compile(
        r"(?:hệ số|tỷ số|tỉ số|tỷ lệ|tỉ lệ|tỷ trọng|biên|vòng quay|số ngày|chỉ số)"
        r"[^,.?]{0,60}", re.I)
    names: Counter[str] = Counter()
    for i, text in unshaped:
        for m in concept.findall(text):
            names[" ".join(m.split()).lower()[:60]] += 1
    for name, count in names.most_common(30):
        p(f"  {count:3d}  {name}")

    p("\nsample unshaped questions:")
    for i, text in unshaped[:12]:
        p(f"  id={i:4d} {text[:150]}")

    out = ROOT / "artifacts" / "_probe_ratio_unit_pool.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
