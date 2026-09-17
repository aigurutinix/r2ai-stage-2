"""Test diverse-vote (CHASE-SQL): 2 model x 3 chiến lược -> vote. Xem có tự sửa ca sai không."""
import json
import os
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve
from kingpro.answering.ensemble import diverse_vote
from kingpro.answering.llm_client import chat

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def tables_for(q, k=2):
    return [{"table_ref": h["table_ref"], "csv_path": "build/tables/" + CAT[h["table_ref"]]["csv_path"]}
            for h in retrieve(q, k=k) if h["table_ref"] in CAT]


def main():
    ids = [int(x) for x in sys.argv[1:]] or [4, 5, 9, 11, 21, 24, 2, 13]
    key = os.environ["KINGPRO_LLM_API_KEY"]
    llms = [
        ("coder", lambda s, u: chat(s, u, base_url=os.environ["BASE_CODER"], api_key=key, model=os.environ["MODEL_CODER"], temperature=0.3, max_tokens=1200, timeout=300)),
        ("qwen3", lambda s, u: chat(s, u, base_url=os.environ["BASE_QWEN3"], api_key=key, model=os.environ["MODEL_QWEN3"], temperature=0.3, max_tokens=1200, timeout=300)),
    ]
    qs = {q["id"]: q for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
    hi = 0
    for i in ids:
        tbls = tables_for(qs[i]["question"], k=2)
        r = diverse_vote(qs[i]["question"], tbls, llms, max_fix=2)
        hi += 1 if r["confidence"] == "high" else 0
        print(f"Q{i}: vote={r['answer']} ({r['votes']}/{r['total']}, {r['confidence']})", flush=True)
    print(f"\n=== diverse-vote: {len(ids)} câu | tin CAO (>=60% vote): {hi}")


if __name__ == "__main__":
    main()
