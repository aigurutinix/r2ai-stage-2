"""Compare method_v4 vs ~0.42 baseline submissions."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        return {r["id"]: r for r in json.loads(z.read("submission.json"))}


def exec_status(rows: dict, zname: str) -> dict[str, int]:
    with zipfile.ZipFile(ROOT / "submissions" / zname) as z:
        frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }
    st = {"ok": 0, "lech": 0, "loi": 0, "const": 0}
    for qid in range(1, 1013):
        row = rows[qid]
        ev = row.get("evidence") or []
        if not ev:
            st["const"] += 1
            continue
        ns: dict = {"pd": pd}
        try:
            for item in ev:
                ns[item["variable"]] = frames[item["csv_path"]]
            exec(row["pandas_query"], ns, ns)  # noqa: S102
            got = float(ns["result"])
            key = "ok" if abs(got - float(row["answer"])) <= 0.01 else "lech"
            st[key] += 1
        except Exception:
            st["loi"] += 1
    return st


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    zips = [
        "vote3.zip",
        "vote3_hardhop.zip",
        "method_v1.zip",
        "method_v3.zip",
        "method_v4.zip",
    ]
    data = {z: load(z) for z in zips if (ROOT / "submissions" / z).exists()}

    print("=== EXEC proxy (program == answer field) ===")
    for z, rows in data.items():
        st = exec_status(rows, z)
        ok = st["ok"]
        print(f"  {z:22s} ok={ok:4d} ({ok/1012:.4f})  lech={st['lech']} loi={st['loi']}")

    base = "vote3_hardhop.zip"
    if base not in data:
        base = "vote3.zip"
    print(f"\n=== Answer diff vs {base} (public ~0.427 lineage) ===")
    for z, rows in data.items():
        if z == base:
            continue
        diff = sum(
            1 for qid in range(1, 1013)
            if abs(float(data[base][qid]["answer"]) - float(rows[qid]["answer"])) > 0.01
        )
        print(f"  {z:22s} changed={diff}  same={1012-diff}")

    v4 = data.get("method_v4.zip")
    hh = data.get("vote3_hardhop.zip")
    v3 = data.get("method_v3.zip")
    v1 = data.get("method_v1.zip")
    if not v4 or not hh:
        return

    print("\n=== method_v4 vs vote3_hardhop ===")
    changed = [
        qid for qid in range(1, 1013)
        if abs(float(hh[qid]["answer"]) - float(v4[qid]["answer"])) > 0.01
    ]
    print(f"  changed answers: {len(changed)}")

    if v3:
        c43 = sum(
            1 for qid in range(1, 1013)
            if abs(float(v4[qid]["answer"]) - float(v3[qid]["answer"])) > 0.01
        )
        print(f"  v4 vs v3 changed: {c43}")

    if v1:
        c41 = sum(
            1 for qid in range(1, 1013)
            if abs(float(v4[qid]["answer"]) - float(v1[qid]["answer"])) > 0.01
        )
        print(f"  v4 vs v1 changed: {c41}")

    man_path = ROOT / "artifacts/fresh/method_manifest.jsonl"
    manifest = []
    if man_path.exists():
        for line in man_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                manifest.append(json.loads(line))

    print(f"\n  v4 manifest splices: {len(manifest)}")
    same_hh = sum(
        1 for m in manifest
        if abs(float(hh[m["id"]]["answer"]) - float(v4[m["id"]]["answer"])) <= 0.01
    )
    print(f"  splices same answer as hardhop: {same_hh}/{len(manifest)}")
    diff_hh = [m for m in manifest
               if abs(float(hh[m["id"]]["answer"]) - float(v4[m["id"]]["answer"])) > 0.01]
    print(f"  splices NEW vs hardhop: {len(diff_hh)}")
    for m in diff_hh[:12]:
        qid = m["id"]
        print(f"    id={qid} [{m['layer']}] "
              f"hh={hh[qid]['answer']} -> v4={v4[qid]['answer']}")

    # v4-only changes (not in hardhop diff set)
    hh_changed = {
        qid for qid in range(1, 1013)
        if abs(float(data["vote3.zip"][qid]["answer"]) - float(hh[qid]["answer"])) > 0.01
    }
    v4_only = [qid for qid in changed if qid not in hh_changed]
    v4_lost = [qid for qid in hh_changed if qid not in {
        q for q in range(1, 1013)
        if abs(float(hh[q]["answer"]) - float(v4[q]["answer"])) > 0.01
    }]
    print(f"\n  v4 changes beyond hardhop: {len(v4_only)}")
    print(f"  hardhop splices reverted in v4: {len(v4_lost)}")


if __name__ == "__main__":
    main()
