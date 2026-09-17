"""Rebuild vote3_hardllm.zip = rule hop zip + clean LLM gap fills."""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import hard_hop as hh  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    plans = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/hard_llm_plans.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = ROOT / "submissions" / "vote3_hardhop.zip"
    dest = ROOT / "submissions" / "vote3_hardllm.zip"
    with zipfile.ZipFile(base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    spliced = 0
    for row in plans:
        if row.get("status") != "ok" or "hit" not in row:
            continue
        qid = row["id"]
        question = qs[qid]
        if hh.HARD_REFUSE.search(question):
            print(qid, "skip refuse")
            continue
        hit = row["hit"]
        if not hh.accept_hit(question, hit):
            print(qid, "skip gate", hit["answer"])
            continue
        if abs(hit["answer"]) > 200 and re_search_pct(question):
            print(qid, "skip huge pct", hit["answer"])
            continue
        if abs(hit["answer"]) > 500 and not re.search(
                r"nghìn tỷ|trăm tỷ|tỷ đồng|triệu đồng", question, re.I):
            print(qid, "skip huge non-money", hit["answer"])
            continue
        inc = float(rows[qid].get("answer") or 0)
        if abs(hit["answer"] - inc) <= 0.01:
            continue
        rows[qid].update({
            "answer": hit["answer"],
            "pandas_query": hit["pandas_query"],
            "evidence": hit["evidence"],
            "relevant_docs": hit["relevant_docs"],
            "relevant_tables": hit["relevant_tables"],
        })
        for name, text in hit["csv_payloads"].items():
            files[f"data/{name}"] = text.encode("utf-8")
        # reexec
        namespace = {"pd": pd}
        for item in hit["evidence"]:
            name = item["csv_path"].split("/", 1)[-1]
            namespace[item["variable"]] = pd.read_csv(
                io.StringIO(hit["csv_payloads"][name]),
                dtype=str, keep_default_na=False,
            )
        exec(hit["pandas_query"], namespace, namespace)  # noqa: S102
        ok = abs(float(hit["answer"]) - float(namespace["result"])) <= 0.01
        print(qid, hit.get("op"), hit["answer"], "reexec", "ok" if ok else "FAIL")
        if not ok:
            continue
        spliced += 1

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)
    print(f"llm spliced {spliced} -> {dest}")


def re_search_pct(q: str) -> bool:
    return bool(re.search(r"phần trăm|%", q, re.I))


if __name__ == "__main__":
    main()
