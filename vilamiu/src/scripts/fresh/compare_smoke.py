"""Compare smoke batch vs cached plans."""
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
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(ROOT / "submissions/vote3_hardhop.zip") as z:
        hh = {r["id"]: float(r["answer"]) for r in json.loads(z.read("submission.json"))}
    old = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/hard_llm_plans.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    new = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/hard_llm_smoke2.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    for qid in sorted(new):
        n, o = new[qid], old.get(qid, {})
        print(f"Q{qid} new={n['status']} ans={n.get('answer')} op={n.get('op')}")
        print(f"     old={o.get('status')} ans={o.get('answer')} op={o.get('op')}")
        print(f"     hardhop={hh.get(qid)}")
        print(f"     {qs[qid][:90]}")
        if n.get("status") == "ok":
            hit = n["hit"]
            ns: dict = {"pd": pd}
            for ev in hit["evidence"]:
                name = ev["csv_path"].split("/")[-1]
                ns[ev["variable"]] = pd.read_csv(
                    io.StringIO(hit["csv_payloads"][name]),
                    dtype=str, keep_default_na=False,
                )
            exec(hit["pandas_query"], ns, ns)  # noqa: S102
            print(f"     reexec={ns['result']}")
        print()


if __name__ == "__main__":
    main()
