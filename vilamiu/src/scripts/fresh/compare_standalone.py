"""Compare standalone_v1 vs vote3 by block, layer, and local EXEC proxy."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import hard_hop as hh  # noqa: E402
from blocks import block_of  # noqa: E402
from method_pipeline import shape_of  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

TOL = 0.011


def load_zip(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        return {r["id"]: r for r in json.loads(z.read("submission.json"))}


def exec_one(row: dict, frames: dict[str, pd.DataFrame]) -> str:
    ev = row.get("evidence") or []
    if not ev:
        return "const"
    ns: dict = {"pd": pd}
    try:
        for item in ev:
            ns[item["variable"]] = frames[item["csv_path"]]
        exec(row["pandas_query"], ns, ns)  # noqa: S102
        got = float(ns["result"])
        if abs(got - float(row["answer"])) <= TOL:
            return "ok"
        return "lech"
    except Exception:
        return "loi"


def exec_by_block(rows: dict, zname: str) -> dict[str, Counter]:
    resolver = TickerResolver()
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(ROOT / "submissions" / zname) as z:
        frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }
    by_shape: dict[str, Counter] = defaultdict(Counter)
    by_block: dict[str, Counter] = defaultdict(Counter)
    for qid in range(1, 1013):
        row = rows[qid]
        question = qs[qid]
        n_co = len(hh.resolve_cohort(question, resolver))
        block = block_of(question, companies=max(1, n_co))
        shape = shape_of(block)
        st = exec_one(row, frames)
        by_shape[shape][st] += 1
        by_block[block][st] += 1
    return {"shape": by_shape, "block": by_block}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v3 = load_zip("vote3.zip")
    st = load_zip("standalone_v1.zip")

    trace: dict[int, str] = {}
    tpath = ROOT / "artifacts/fresh/standalone_trace.jsonl"
    if tpath.exists():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                trace[rec["id"]] = rec.get("chosen") or "?"

    print("=== EXEC proxy (program ≡ answer field) ===")
    for zname, rows in [("vote3.zip", v3), ("standalone_v1.zip", st)]:
        stats = exec_by_block(rows, zname)
        ok = sum(c["ok"] for c in stats["shape"].values())
        loi = sum(c["loi"] for c in stats["shape"].values())
        lech = sum(c["lech"] for c in stats["shape"].values())
        const = sum(c["const"] for c in stats["shape"].values())
        print(f"\n{zname}: ok={ok} ({ok/1012:.4f})  lech={lech}  loi={loi}  const={const}")

    print("\n--- by SHAPE ---")
    v3_stats = exec_by_block(v3, "vote3.zip")
    st_stats = exec_by_block(st, "standalone_v1.zip")
    for shape in ("single", "ratio", "multi_year", "screen", "other"):
        v = v3_stats["shape"].get(shape, Counter())
        s = st_stats["shape"].get(shape, Counter())
        n = sum(v.values()) or 1
        m = sum(s.values()) or 1
        print(
            f"  {shape:12s}  vote3 ok={v.get('ok',0):3d}/{n} ({v.get('ok',0)/n:.3f})  "
            f"standalone ok={s.get('ok',0):3d}/{m} ({s.get('ok',0)/m:.3f})  "
            f"Δok={s.get('ok',0)-v.get('ok',0):+d}  "
            f"st loi={s.get('loi',0)} lech={s.get('lech',0)}"
        )

    print("\n=== Answer diff vs vote3 ===")
    diff_ids = [
        qid for qid in range(1, 1013)
        if abs(float(v3[qid]["answer"]) - float(st[qid]["answer"])) > TOL
    ]
    same_ids = [qid for qid in range(1, 1013) if qid not in diff_ids]
    print(f"  changed={len(diff_ids)}  same={len(same_ids)}")

    resolver = TickerResolver()
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    diff_by_shape: Counter[str] = Counter()
    diff_by_layer: Counter[str] = Counter()
    zero_by_shape: Counter[str] = Counter()
    for qid in diff_ids:
        q = qs[qid]
        shape = shape_of(block_of(q, companies=max(1, len(hh.resolve_cohort(q, resolver)))))
        diff_by_shape[shape] += 1
        diff_by_layer[trace.get(qid, "?")] += 1
        if abs(float(st[qid]["answer"])) < 1e-9:
            zero_by_shape[shape] += 1

    print("\n--- diffs by shape ---")
    for shape, n in diff_by_shape.most_common():
        print(f"  {shape:12s} {n:4d}  (zero answers: {zero_by_shape.get(shape, 0)})")

    print("\n--- diffs by standalone layer ---")
    for layer, n in diff_by_layer.most_common():
        print(f"  {layer:18s} {n:4d}")

    # Where standalone EXEC got worse vs vote3 (lost ok status)
    with zipfile.ZipFile(ROOT / "submissions/vote3.zip") as z:
        v3_frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist() if n.startswith("data/")
        }
    with zipfile.ZipFile(ROOT / "submissions/standalone_v1.zip") as z:
        st_frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist() if n.startswith("data/")
        }

    lost_ok: Counter[str] = Counter()
    gained_ok: Counter[str] = Counter()
    for qid in range(1, 1013):
        q = qs[qid]
        shape = shape_of(block_of(q, companies=max(1, len(hh.resolve_cohort(q, resolver)))))
        v_ok = exec_one(v3[qid], v3_frames) == "ok"
        s_ok = exec_one(st[qid], st_frames) == "ok"
        if v_ok and not s_ok:
            lost_ok[shape] += 1
        if not v_ok and s_ok:
            gained_ok[shape] += 1

    print("\n=== EXEC ok gained/lost vs vote3 (by shape) ===")
    for shape in ("single", "ratio", "multi_year", "screen"):
        print(
            f"  {shape:12s}  lost_ok={lost_ok.get(shape,0):3d}  "
            f"gained_ok={gained_ok.get(shape,0):3d}  "
            f"net={gained_ok.get(shape,0)-lost_ok.get(shape,0):+d}"
        )

    # Same-answer but program changed (EXEC risk without answer diff)
    same_prog_diff = sum(
        1 for qid in same_ids
        if (v3[qid].get("pandas_query") or "").strip()
        != (st[qid].get("pandas_query") or "").strip()
    )
    print(f"\n  same answer but different program: {same_prog_diff}")

    print("\n=== Worst standalone layers (lost ok count) ===")
    lost_by_layer: Counter[str] = Counter()
    for qid in range(1, 1013):
        v_ok = exec_one(v3[qid], v3_frames) == "ok"
        s_ok = exec_one(st[qid], st_frames) == "ok"
        if v_ok and not s_ok:
            lost_by_layer[trace.get(qid, "?")] += 1
    for layer, n in lost_by_layer.most_common(10):
        print(f"  {layer:18s} lost_ok={n}")


if __name__ == "__main__":
    main()
