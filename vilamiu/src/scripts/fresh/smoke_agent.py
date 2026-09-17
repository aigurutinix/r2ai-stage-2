"""Smoke loop for agent_standalone.py — NO vote3.

Usage:
  python scripts/fresh/smoke_agent.py
  python scripts/fresh/smoke_agent.py --sample 40 --min-pass 0.60 --full
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from blocks import block_of  # noqa: E402
from method_pipeline import shape_of  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402
import hard_hop as hh  # noqa: E402


def pick_sample(n: int, seed: int) -> list[int]:
    """Stratify by shape so screen/single/ratio all appear."""

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    resolver = TickerResolver()
    by_shape: dict[str, list[int]] = defaultdict(list)
    for qid, question in qs.items():
        n_co = len(hh.resolve_cohort(question, resolver))
        shape = shape_of(block_of(question, companies=max(1, n_co)))
        by_shape[shape].append(qid)

    rng = random.Random(seed)
    ids: list[int] = []
    per = max(3, n // max(len(by_shape), 1))
    for shape_ids in by_shape.values():
        shuffled = shape_ids[:]
        rng.shuffle(shuffled)
        ids.extend(shuffled[:per])
    while len(ids) < n:
        cand = rng.randint(1, 1012)
        if cand not in ids:
            ids.append(cand)
    return sorted(ids[:n])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-pass", type=float, default=0.60,
                        help="BCTC grounding rate on sample (stubs may fail)")
    parser.add_argument("--out", default="submissions/standalone_smoke.zip")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    ids = pick_sample(args.sample, args.seed)
    id_str = ",".join(str(i) for i in ids)
    print(f"smoke ids ({len(ids)}): {id_str[:100]}…")

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/fresh/agent_standalone.py"),
        "--dest", args.out,
        "--ids", id_str,
        "--trace", "artifacts/fresh/standalone_trace_smoke.jsonl",
    ], cwd=ROOT, check=True)

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/fresh/agent_review.py"),
        "--trace", "artifacts/fresh/standalone_trace_smoke.jsonl",
        "--zip", args.out,
        "--out", "artifacts/fresh/standalone_review_smoke.txt",
    ], cwd=ROOT, check=True)

    # Layer histogram from trace
    layers: dict[str, int] = defaultdict(int)
    for line in (ROOT / "artifacts/fresh/standalone_trace_smoke.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            layers[json.loads(line).get("chosen") or "?"] += 1
    print(f"layer mix: {dict(layers)}")

    result = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/fresh/validate_bctc.py"),
        "--zip", args.out,
        "--ids", id_str,
        "--min-pass", str(args.min_pass),
    ], cwd=ROOT)

    if result.returncode != 0:
        print("SMOKE FAILED — see artifacts/fresh/standalone_review_smoke.txt")
        sys.exit(1)

    print("SMOKE PASSED — see artifacts/fresh/standalone_review_smoke.txt")
    if args.full:
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/fresh/agent_standalone.py"),
            "--dest", "submissions/standalone_v1.zip",
            "--trace", "artifacts/fresh/standalone_trace.jsonl",
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/fresh/validate_bctc.py"),
            "--zip", "submissions/standalone_v1.zip",
            "--sample", "50",
            "--min-pass", str(max(0.5, args.min_pass * 0.9)),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
