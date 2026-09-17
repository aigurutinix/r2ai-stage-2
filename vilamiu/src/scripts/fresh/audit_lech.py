"""Dump and classify program≠answer mismatches in a submission zip."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCALE = (1e3, 1e6, 1e9, 1e11, 1e12)


def classify(got: float, want: float) -> str:
    if abs(want) < 1e-9:
        return "answer_zero"
    if got * want < 0 and abs(abs(got) - abs(want)) <= 0.02:
        return "sign"
    ratio = got / want
    for s in SCALE:
        if abs(ratio / s - 1) < 0.05:
            return f"scale_div_{s:.0e}"
        if abs(ratio - 1 / s) < 0.05:
            return f"scale_mul_{s:.0e}"
    return "other"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/method_v1.zip")
    args = parser.parse_args()

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    with zipfile.ZipFile(ROOT / args.zip) as z:
        rows = json.loads(z.read("submission.json"))
        frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }

    lech = []
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
        except Exception as exc:
            lech.append({
                "id": row["id"],
                "kind": "LOI",
                "err": str(exc)[:100],
                "answer": row.get("answer"),
                "question": qs[row["id"]],
            })
            continue
        want = float(row["answer"])
        if abs(got - want) <= 0.01:
            continue
        lech.append({
            "id": row["id"],
            "kind": classify(got, want),
            "got": got,
            "want": want,
            "ratio": got / want if want else None,
            "question": qs[row["id"]],
        })

    out = ROOT / "artifacts/fresh/lech_audit.jsonl"
    out.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lech) + "\n",
        encoding="utf-8",
    )
    print(f"{len(lech)} mismatches -> {out}")
    for k, v in Counter(x["kind"] for x in lech).most_common():
        print(f"  {k}: {v}")
    for x in sorted(lech, key=lambda r: r["id"]):
        if x["kind"] == "LOI":
            print(f"{x['id']:4d} LOI  ans={x['answer']}  {x['err']}")
        else:
            print(
                f"{x['id']:4d} {x['kind']:16s} "
                f"got={x['got']!s:22s} want={x['want']!s}")


if __name__ == "__main__":
    main()
