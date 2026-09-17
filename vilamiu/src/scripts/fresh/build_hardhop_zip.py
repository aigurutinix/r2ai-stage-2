"""Scan every question; splice every hop that passes the unit/shape gate onto vote3."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from hard_hop import (  # noqa: E402
    CellBook, TickerResolver, confidence_hit, solve_one,
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    base = ROOT / "submissions" / "vote3.zip"
    dest = ROOT / "submissions" / "vote3_hardhop.zip"
    with zipfile.ZipFile(base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    book, resolver = CellBook(), TickerResolver()
    hits = {}
    dropped = 0
    for qid, question in sorted(qs.items()):
        try:
            hit = solve_one(book, question, resolver)
        except Exception as exc:  # noqa: BLE001
            print(qid, "ERROR", exc)
            continue
        if hit is None:
            continue
        if not confidence_hit(question, hit):
            dropped += 1
            continue
        inc = float(rows[qid].get("answer") or 0)
        if abs(hit["answer"] - inc) <= 0.01:
            continue
        hits[qid] = hit
        print(f"{qid} {hit.get('op')} {hit.get('filter')}->{hit.get('target')} "
              f"{hit['answer']} (was {inc})")

    for qid, hit in hits.items():
        rows[qid].update({
            "answer": hit["answer"],
            "pandas_query": hit["pandas_query"],
            "evidence": hit["evidence"],
            "relevant_docs": hit["relevant_docs"],
            "relevant_tables": hit["relevant_tables"],
        })
        for name, text in hit["csv_payloads"].items():
            files[f"data/{name}"] = text.encode("utf-8")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    fail = 0
    with zipfile.ZipFile(dest) as archive:
        for qid, hit in hits.items():
            record = rows[qid]
            namespace: dict = {"pd": pd}
            for item in record["evidence"]:
                namespace[item["variable"]] = pd.read_csv(
                    io.BytesIO(archive.read(item["csv_path"])),
                    dtype=str, keep_default_na=False,
                )
            exec(record["pandas_query"], namespace, namespace)  # noqa: S102
            ok = abs(float(record["answer"]) - float(namespace["result"])) <= 0.01
            if not ok:
                fail += 1
                print(qid, "reexec FAIL", namespace.get("result"))
    print(f"spliced {len(hits)}  dropped_gate {dropped}  reexec_fail {fail} -> {dest}")


if __name__ == "__main__":
    main()
