"""Rule + aggressive rule + cached LLM plans → one submission zip.

Targets the ~300-question screening pool (blocks.json), not just the ~45
deterministic hops.  Layer order:

  1. confidence_hit rule hops (deterministic, tight)
  2. accept_hit aggressive metric-pair tries (broader rule fallback)
  3. cached LLM plans with confidence_hit + reexec

Usage:
  python scripts/fresh/build_mass_hop_zip.py
  python scripts/fresh/build_mass_hop_zip.py --llm artifacts/fresh/hard_llm_plans.jsonl
"""
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
    CellBook, TickerResolver, accept_hit, confidence_hit, solve_one,
)


def reexec_ok(record: dict, csv_blobs: dict[str, bytes]) -> bool:
    namespace: dict = {"pd": pd}
    try:
        for item in record["evidence"]:
            blob = csv_blobs.get(item["csv_path"])
            if blob is None:
                return False
            namespace[item["variable"]] = pd.read_csv(
                io.BytesIO(blob), dtype=str, keep_default_na=False,
            )
        exec(record["pandas_query"], namespace, namespace)  # noqa: S102
        return abs(float(record["answer"]) - float(namespace["result"])) <= 0.01
    except Exception:
        return False


def load_llm(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok" or "hit" not in row:
            continue
        out[row["id"]] = row["hit"]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--dest", default="submissions/vote3_masshop.zip")
    parser.add_argument("--llm", default="artifacts/fresh/hard_llm_screening.jsonl")
    args = parser.parse_args()

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    base_path = ROOT / args.base
    dest = ROOT / args.dest
    with zipfile.ZipFile(base_path) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    book, resolver = CellBook(), TickerResolver()
    llm_hits = load_llm(ROOT / args.llm)

    stats = {"rule_conf": 0, "rule_aggr": 0, "llm": 0}
    hits: dict[int, dict] = {}

    for qid, question in sorted(qs.items()):
        try:
            hit = solve_one(book, question, resolver)
        except Exception as exc:  # noqa: BLE001
            print(qid, "ERROR", exc)
            continue
        if hit is None:
            continue
        inc = float(rows[qid].get("answer") or 0)
        if abs(hit["answer"] - inc) <= 0.01:
            continue
        layer = None
        if confidence_hit(question, hit):
            layer = "rule_conf"
        elif hit.get("aggressive") and accept_hit(question, hit):
            layer = "rule_aggr"
        if layer:
            hits[qid] = hit
            stats[layer] += 1
            print(f"{qid} {layer} {hit.get('op')} {hit.get('filter')}->{hit.get('target')} "
                  f"{hit['answer']} (was {inc})")

    for qid, hit in llm_hits.items():
        if qid in hits:
            continue
        question = qs[qid]
        inc = float(rows[qid].get("answer") or 0)
        if abs(hit["answer"] - inc) <= 0.01:
            continue
        if not confidence_hit(question, hit):
            continue
        hits[qid] = hit
        stats["llm"] += 1
        print(f"{qid} llm {hit.get('op')} {hit.get('filter')}->{hit.get('target')} "
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

    fail = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)
        for qid, hit in hits.items():
            record = rows[qid]
            blobs = {item["csv_path"]: archive.read(item["csv_path"]) for item in record["evidence"]}
            if not reexec_ok(record, blobs):
                fail += 1
                print(qid, "reexec FAIL")

    total = sum(stats.values())
    print(f"spliced {total}  rule_conf={stats['rule_conf']}  "
          f"rule_aggr={stats['rule_aggr']}  llm={stats['llm']}  "
          f"reexec_fail={fail} -> {dest}")


if __name__ == "__main__":
    main()
