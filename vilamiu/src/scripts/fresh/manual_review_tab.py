"""Dump tab_verified OK rows for manual semantic check."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    tab = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/tab_plan.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    aud = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok_tab = [a for a in aud if a["verdict"] == "OK" and a["layer"] == "tab_verified"]
    for a in sorted(ok_tab, key=lambda x: x["id"]):
        qid = a["id"]
        q = qs[qid]["question"]
        t = tab[qid]
        csv_path = ROOT / t["csv"]
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            grid = list(csv.reader(f))
        r, c = t["row"], t["col"]
        label = grid[r][0] if r < len(grid) else "?"
        cell = grid[r + 1][c] if r + 1 < len(grid) and c < len(grid[r + 1]) else "?"
        print(f"Q{qid}: {q[:120]}")
        print(f"  row label: {label[:90]}")
        print(f"  cell: {cell}  scale={t.get('scale')}  quoted={t.get('quoted')}")
        print(f"  {a['old']:.4g} -> {a['new']:.4g}")
        print()


if __name__ == "__main__":
    main()
