"""Per-id program≡answer status across submission zips."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def status(path: Path) -> dict[int, str]:
    with zipfile.ZipFile(path) as z:
        rows = json.loads(z.read("submission.json"))
        frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }
    out: dict[int, str] = {}
    for row in rows:
        qid = row["id"]
        ev = row.get("evidence") or []
        if not ev:
            out[qid] = "const"
            continue
        ns: dict = {"pd": pd}
        try:
            for item in ev:
                ns[item["variable"]] = frames[item["csv_path"]]
            exec(row["pandas_query"], ns, ns)  # noqa: S102
            got = float(ns["result"])
            out[qid] = "ok" if abs(got - float(row["answer"])) <= 0.01 else "lech"
        except Exception:
            out[qid] = "loi"
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    zips = {
        "vote3": ROOT / "submissions/vote3.zip",
        "hardhop": ROOT / "submissions/vote3_hardhop.zip",
        "method_v3": ROOT / "submissions/method_v3.zip",
    }
    st = {name: status(p) for name, p in zips.items()}

    print("summary:")
    for name, s in st.items():
        c = {"ok": 0, "lech": 0, "loi": 0, "const": 0}
        for v in s.values():
            c[v] += 1
        print(f"  {name}: {c}")

    # method worse than hardhop
    worse = []
    better = []
    for qid in range(1, 1013):
        h = st["hardhop"].get(qid, "?")
        m = st["method_v3"].get(qid, "?")
        if h == "ok" and m != "ok":
            worse.append(qid)
        if h != "ok" and m == "ok":
            better.append(qid)
    print(f"\nmethod_v3 worse than hardhop (hard ok, method not): {len(worse)} {worse[:15]}")
    print(f"method_v3 better than hardhop: {len(better)} {better[:15]}")

    # pool where any zip has lech but another ok — borrow opportunity
    borrow = []
    for qid in range(1, 1013):
        if st["method_v3"][qid] != "ok":
            for name in ("hardhop", "vote3"):
                if st[name][qid] == "ok":
                    borrow.append((qid, name))
                    break
    print(f"\nmethod_v3 not ok but vote3/hardhop ok: {len(borrow)}")
    for qid, src in borrow[:25]:
        print(f"  {qid} from {src}")


if __name__ == "__main__":
    main()
