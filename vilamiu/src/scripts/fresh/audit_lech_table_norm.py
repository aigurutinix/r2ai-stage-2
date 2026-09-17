"""Try table_norm unit gap to explain LECH mismatches."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))
import table_norm as tn  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    with zipfile.ZipFile(ROOT / "submissions/method_v1.zip") as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}

    audit = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/lech_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]

    fixable = []
    for item in audit:
        if item["kind"] in ("LOI",):
            continue
        qid = item["id"]
        row = rows[qid]
        q = qs[qid]
        got = item.get("got")
        want = item.get("want")
        if got is None:
            continue
        ev = row.get("evidence") or []
        if not ev:
            continue
        blob = blobs.get(ev[0]["csv_path"])
        if not blob:
            continue
        eff = tn.effective_divide_unit(q, blob)
        _qn, qu = tn.question_unit(q)
        tab = tn.detect_table_unit_vnd(blob)
        candidates = []
        if eff and eff > 0 and eff != 1.0:
            for op, val in (("div", eff), ("mul", 1 / eff)):
                trial = got / val if op == "div" else got * val
                if abs(trial - want) <= 0.01:
                    candidates.append((f"eff_{op}", eff, trial))
        if qu and tab:
            fac = tab / qu
            for op, val in (("div", fac), ("mul", 1 / fac)):
                if val <= 0:
                    continue
                trial = got / val if op == "div" else got * val
                if abs(trial - want) <= 0.01:
                    candidates.append((f"tabq_{op}", val, trial))
        if candidates:
            fixable.append((qid, item["kind"], got, want, candidates, q[:80]))
            print(f"{qid} {item['kind']} got={got} want={want}")
            for c in candidates:
                print(f"  -> {c[0]} factor={c[1]:.4g} trial={c[2]}")
            print(f"  q={q[:90]}")
            print(f"  tab={tab} qu={qu} eff={eff}")
            print()

    print(f"table_norm fixable: {len(fixable)} / {len(audit)}")


if __name__ == "__main__":
    main()
