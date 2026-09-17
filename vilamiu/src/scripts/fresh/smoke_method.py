"""Smoke loop: small sample → BCTC validate → report.

Runs method_pipeline on a stratified sample, validates corpus grounding,
exits non-zero if pass rate below threshold (default 85%).

Usage:
  python scripts/fresh/smoke_method.py
  python scripts/fresh/smoke_method.py --sample 40 --min-pass 0.90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-pass", type=float, default=0.85)
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--out", default="submissions/smoke_method.zip")
    parser.add_argument("--full", action="store_true",
                        help="If smoke passes, run full 1012")
    args = parser.parse_args()

    # 1. Pick sample ids from last manifest (or stratified default in validate_bctc)
    manifest = ROOT / "artifacts/fresh/method_manifest.jsonl"
    sample_script = ROOT / "scripts/fresh/validate_bctc.py"

    # Run pipeline on FULL set first to get manifest, OR use --ids from prior manifest
    # For smoke: run pipeline restricted to sample ids only
    import random
    from collections import defaultdict

    by_layer: dict[str, list[int]] = defaultdict(list)
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                by_layer[m["layer"]].append(m["id"])
    rng = random.Random(args.seed)
    ids: list[int] = []
    per = max(2, args.sample // max(len(by_layer), 1))
    for layer, layer_ids in by_layer.items():
        rng.shuffle(layer_ids)
        ids.extend(layer_ids[:per])
    while len(ids) < args.sample:
        cand = rng.randint(1, 1012)
        if cand not in ids:
            ids.append(cand)
    ids = sorted(ids)[:args.sample]
    id_str = ",".join(str(i) for i in ids)
    print(f"smoke ids ({len(ids)}): {id_str[:80]}…")

    # 2. Run pipeline on sample (writes full zip but only processes sample — need --ids)
    cmd_pipe = [
        sys.executable,
        str(ROOT / "scripts/fresh/method_pipeline.py"),
        "--base", args.base,
        "--dest", args.out,
        "--ids", id_str,
        "--no-consistency",
    ]
    print("running pipeline…")
    subprocess.run(cmd_pipe, cwd=ROOT, check=True)

    # 3. Validate BCTC grounding on sample
    cmd_val = [
        sys.executable,
        str(sample_script),
        "--zip", args.out,
        "--ids", id_str,
        "--min-pass", str(args.min_pass),
    ]
    print("validating BCTC…")
    result = subprocess.run(cmd_val, cwd=ROOT)
    if result.returncode != 0:
        print("SMOKE FAILED — fix pipeline before full run")
        sys.exit(1)

    print("SMOKE PASSED")
    if args.full:
        cmd_full = [
            sys.executable,
            str(ROOT / "scripts/fresh/method_pipeline.py"),
            "--base", args.base,
            "--dest", "submissions/method_v3.zip",
            "--no-consistency",
        ]
        print("running full pipeline…")
        subprocess.run(cmd_full, cwd=ROOT, check=True)
        subprocess.run([
            sys.executable,
            str(sample_script),
            "--zip", "submissions/method_v3.zip",
            "--sample", "50",
            "--min-pass", str(args.min_pass * 0.95),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
