"""Compare LECH ids between two submission zips."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def lech_ids(path: Path) -> dict[int, tuple[float | None, float, str]]:
    with zipfile.ZipFile(path) as z:
        rows = json.loads(z.read("submission.json"))
        frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }
    out: dict[int, tuple[float | None, float, str]] = {}
    for row in rows:
        ev = row.get("evidence") or []
        if not ev:
            continue
        ns: dict = {"pd": pd}
        try:
            for item in ev:
                ns[item["variable"]] = frames[item["csv_path"]]
            exec(row["pandas_query"], ns, ns)  # noqa: S102
            got = float(ns["result"])
            status = "ok" if abs(got - float(row["answer"])) <= 0.01 else "lech"
        except Exception as exc:
            got = None
            status = f"loi:{type(exc).__name__}"
        out[row["id"]] = (got, float(row["answer"]), status)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = lech_ids(ROOT / "submissions/vote3.zip")
    meth = lech_ids(ROOT / "submissions/method_v1.zip")
    manifest = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/method_manifest.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = {m["id"] for m in manifest}

    new_bad = []
    fixed = []
    for qid in sorted(changed):
        b = base.get(qid, (None, 0, "?"))
        m = meth.get(qid, (None, 0, "?"))
        if b[2] == "ok" and m[2] != "ok":
            new_bad.append((qid, b, m))
        if b[2] != "ok" and m[2] == "ok":
            fixed.append((qid, b, m))

    print(f"specialist changed {len(changed)}")
    print(f"  introduced LECH/LOI (vote3 ok -> method bad): {len(new_bad)}")
    for qid, b, m in new_bad[:20]:
        print(f"    {qid} vote3 ok got={b[0]} -> method {m[2]} got={m[0]} ans={m[1]}")
    print(f"  fixed LECH (vote3 bad -> method ok): {len(fixed)}")
    for qid, b, m in fixed[:20]:
        print(f"    {qid} vote3 {b[2]} got={b[0]} -> method ok got={m[0]} ans={m[1]}")


if __name__ == "__main__":
    main()
