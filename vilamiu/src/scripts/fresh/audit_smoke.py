"""Audit hard_llm smoke batch."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    smoke = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/fresh/hard_llm_smoke3.jsonl"
    with zipfile.ZipFile(ROOT / "submissions/vote3_hardhop.zip") as z:
        hh = {r["id"]: float(r["answer"]) for r in json.loads(z.read("submission.json"))}
    with zipfile.ZipFile(ROOT / "submissions/vote3.zip") as z:
        v3 = {r["id"]: float(r["answer"]) for r in json.loads(z.read("submission.json"))}
    rows = [json.loads(line) for line in smoke.read_text(encoding="utf-8").splitlines() if line.strip()]
    from collections import Counter
    print(f"file: {smoke.name}")
    print("status:", dict(Counter(r.get("status") for r in rows)))
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\nOK {len(ok)}/{len(rows)}:")
    for r in ok:
        qid = r["id"]
        ans = float(r["answer"])
        hit = r["hit"]
        ns: dict = {"pd": pd}
        for ev in hit["evidence"]:
            name = ev["csv_path"].split("/")[-1]
            ns[ev["variable"]] = pd.read_csv(
                io.StringIO(hit["csv_payloads"][name]), dtype=str, keep_default_na=False)
        exec(hit["pandas_query"], ns, ns)  # noqa: S102
        rex = float(ns["result"])
        h, v = hh.get(qid), v3.get(qid)
        mag = max(abs(ans / h), abs(h / ans)) if h and ans else 999
        print(f"  Q{qid} {r.get('op')} ans={ans} reexec={rex} hardhop={h} vote3={v} mag={mag:.2f}")


if __name__ == "__main__":
    main()
