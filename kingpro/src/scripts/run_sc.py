"""Self-consistency 2 model trên N câu: đo consensus nội bộ + đồng thuận chéo (proxy độ đúng)."""
import json
import os
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve
from kingpro.answering.ensemble import self_consistent

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def tables_for(q, k=2):
    out = []
    for h in retrieve(q, k=k):
        r = CAT.get(h["table_ref"])
        if r:
            out.append({"table_ref": r["table_ref"], "csv_path": "build/tables/" + r["csv_path"]})
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    key = os.environ["KINGPRO_LLM_API_KEY"]
    models = {"coder": (os.environ["BASE_CODER"], os.environ["MODEL_CODER"]),
              "qwen3": (os.environ["BASE_QWEN3"], os.environ["MODEL_QWEN3"])}
    qs = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")][:n]
    res = {"coder": {}, "qwen3": {}}
    tcache = {q["id"]: tables_for(q["question"], k=2) for q in qs}
    for tag, (base, model) in models.items():          # HẾT 1 model rồi mới sang model kia (giữ ấm)
        for q in qs:
            r = self_consistent(q["question"], tcache[q["id"]], base, key, model, n=ns, temp=0.5, max_fix=2)
            res[tag][q["id"]] = r
            print(f"Q{q['id']} {tag}: ans={r['answer']} consensus={r['consensus']:.2f}", flush=True)
    # đồng thuận chéo
    agree = both = 0
    for i in [x["id"] for x in qs]:
        a, b = res["coder"][i]["answer"], res["qwen3"][i]["answer"]
        if a is not None and b is not None:
            both += 1
            if abs(a - b) / max(abs(a), abs(b), 1.0) < 0.01:
                agree += 1
    hi_conf = sum(1 for i in [x["id"] for x in qs]
                  if res["coder"][i]["consensus"] >= 0.99 and res["qwen3"][i]["consensus"] >= 0.99)
    print(f"\n=== SC: cả 2 ra số {both}/{n} | ĐỒNG THUẬN chéo {agree} | mỗi model tự kiên định (consensus=1) cả 2: {hi_conf}")
    json.dump(res, open("build/sc_results.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
