"""Chạy pipeline trên N câu, đo tỉ lệ execution (ra được số), lưu kết quả để soi."""
import json
import sys
import time
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
    import os
    from kingpro.answering.llm_client import chat
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    tag = sys.argv[2] if len(sys.argv) > 2 else "default"   # coder | qwen3
    qs = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")][:n]
    base = os.environ.get(f"BASE_{tag.upper()}") or os.environ.get("KINGPRO_LLM_BASE_URL")
    model = os.environ.get(f"MODEL_{tag.upper()}") or os.environ.get("KINGPRO_LLM_MODEL")
    key = os.environ["KINGPRO_LLM_API_KEY"]
    llm = (lambda s, u: chat(s, u, base_url=base, api_key=key, model=model, max_tokens=1200))
    print(f"MODEL={model} @ {base}")
    out = Path(f"build/run_{tag}.jsonl")
    ok = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            tbls = tables_for(q["question"], k=2)
            try:
                res = answer_question(q["question"], tbls, llm, max_fix=3)
            except Exception as e:
                res = {"ok": False, "answer": None, "error": f"{type(e).__name__}: {e}", "attempts": 0, "pandas_query": ""}
            ok += 1 if res["ok"] else 0
            f.write(json.dumps({"id": q["id"], "question": q["question"], "tables": [t["table_ref"] for t in tbls],
                                "ok": res["ok"], "answer": res.get("answer"), "attempts": res.get("attempts"),
                                "pandas_query": res.get("pandas_query"), "error": res.get("error")}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(qs)}] Q{q['id']} ok={res['ok']} ans={res.get('answer')}", flush=True)
    print(f"\n=== EXECUTION RATE: {ok}/{len(qs)} = {ok/len(qs)*100:.0f}%  ({time.time()-t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
