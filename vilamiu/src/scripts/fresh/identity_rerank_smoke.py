"""Rerank candidate BCTC tables by offline identity score (no model).

Candidates from greedy_plan CSV + all sibling tables in the same doc folder.
Reports how often identity uniquely picks a perfect roll-up table among k≥2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from identity_check import table_identity_score  # noqa: E402


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["id"])] = rec
    return out


def doc_tables(csv_path: Path) -> list[Path]:
    parent = csv_path.parent
    if not parent.is_dir():
        return [csv_path] if csv_path.is_file() else []
    return sorted(parent.glob("table_*.csv"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0=all greedy ids")
    args = parser.parse_args()

    greedy = load_jsonl(ROOT / "artifacts/fresh/greedy_plan.jsonl")
    ids = sorted(greedy)
    if args.limit:
        ids = ids[: args.limit]

    stats: Counter[str] = Counter()
    flips: list[dict] = []

    for qid in ids:
        entry = greedy[qid]
        raw = entry.get("csv") or ""
        base = Path(raw) if Path(raw).is_file() else ROOT / raw
        if not base.is_file():
            stats["missing_csv"] += 1
            continue
        cands = doc_tables(base)
        scored: list[tuple[float, int, Path, dict]] = []
        for p in cands:
            s = table_identity_score(p)
            if s is None or not s.get("applicable"):
                continue
            scored.append((float(s["score"]), int(s["n_codes"]), p, s))
        if not scored:
            stats["no_cdkt_rollup"] += 1
            continue
        stats["has_applicable"] += 1
        if len(scored) < 2:
            stats["only1_applicable"] += 1
            continue
        stats["ge2_applicable"] += 1
        scored.sort(key=lambda t: (-t[0], -t[1]))
        best_score, _, best_path, best_meta = scored[0]
        winners = [t for t in scored if t[0] == best_score]
        if best_score >= 1.0 and len(winners) == 1:
            stats["unique_perfect"] += 1
        elif best_score >= 1.0:
            stats["tied_perfect"] += 1
        elif len(winners) == 1:
            stats["unique_best"] += 1
        else:
            stats["tie"] += 1

        if best_path.resolve() != base.resolve() and best_score > 0:
            stats["differs_from_greedy"] += 1
            if len(flips) < 20:
                gscore = next(
                    (t[0] for t in scored if t[2].resolve() == base.resolve()),
                    None,
                )
                flips.append({
                    "id": qid,
                    "greedy": base.name,
                    "greedy_score": gscore,
                    "id_pick": best_path.name,
                    "id_score": best_score,
                    "n_codes": best_meta["n_codes"],
                    "n_cand": len(scored),
                })

    print(f"greedy_ids={len(ids)}")
    print("stats", dict(stats))
    print("flips (identity≠greedy):", len(flips))
    for f in flips[:12]:
        print(f)


if __name__ == "__main__":
    main()
