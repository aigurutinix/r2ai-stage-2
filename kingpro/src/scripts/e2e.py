"""End-to-end 1 câu: retrieval -> answering (model thật) -> đáp án. Chạy vài câu để soi."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve
from kingpro.answering.pandas_answer import answer_question
from kingpro.answering.llm_client import make_llm_from_env

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def tables_for(question, k=2):
    out = []
    for h in retrieve(question, k=k):
        r = CAT.get(h["table_ref"])
        if r:
            out.append({"table_ref": r["table_ref"], "csv_path": "build/tables/" + r["csv_path"]})
    return out


def main():
    ids = [int(x) for x in sys.argv[1:]] or [1, 2, 4]
    qs = {q["id"]: q for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
    llm = make_llm_from_env()
    for i in ids:
        q = qs[i]
        tbls = tables_for(q["question"], k=2)
        print(f"\n=== Q{i}: {q['question']}")
        print("  bảng:", [t["table_ref"] for t in tbls])
        res = answer_question(q["question"], tbls, llm, max_fix=3)
        print(f"  -> ok={res['ok']} answer={res.get('answer')} attempts={res.get('attempts')}")
        print("  pandas:", (res.get("pandas_query") or "").replace("\n", " ")[:200])
        if not res["ok"]:
            print("  err:", res.get("error"))


if __name__ == "__main__":
    main()
