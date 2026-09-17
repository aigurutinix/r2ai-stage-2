"""Đo EXECUTION ACCURACY offline của model fine-tune trên VAL set (proxy điểm khi chưa nộp).
Val answers từ generator đã kiểm đúng (đẳng thức kế toán) -> khớp = model học đúng pattern.
Cần serve model qua vLLM (OpenAI-compat).

  python scripts/eval_finetune.py --val build/sft_chatml_val.jsonl \
      --base https://<ep>-8000.serverless.fptcloud.com/v1 --model kingpro-lora-merged --key kingpro2026
"""
import argparse, json, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from grader_check import run_one
from kingpro.answering.pandas_answer import _with_preamble, _sanitize, _neutralize_exc
from kingpro.answering.llm_client import chat

CAT = {x["table_ref"]: "build/tables/" + x["csv_path"]
       for x in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="build/sft_chatml_val.jsonl")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="kingpro2026")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.val, encoding="utf-8")][: a.n]
    ok = err = wrong = 0
    for i, r in enumerate(rows, 1):
        sysm = r["conversations"][0]["content"]
        user = r["conversations"][1]["content"]
        trs = r.get("table_refs") or r["meta"]["table_refs"]
        try:
            raw = chat(sysm, user, base_url=a.base, api_key=a.key, model=a.model, temperature=0, max_tokens=1200, timeout=120)
            code = _neutralize_exc(_sanitize(raw))
            full = _with_preamble(code, len(trs))
            v = float(run_one(full, {f"df{j+1}": CAT[tr] for j, tr in enumerate(trs)}))
            if abs(v - float(r["answer"])) <= 0.01:
                ok += 1
            else:
                wrong += 1
        except Exception:
            err += 1
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}: đúng {ok} sai {wrong} lỗi {err}", flush=True)
    tot = len(rows)
    print(f"\n=== EXECUTION ACCURACY (val proxy): {ok}/{tot} = {ok*100/tot:.1f}% "
          f"| sai số {wrong} | lỗi/không chạy {err} ===")
    print("(so mốc: baseline 0.36, mình prompt 0.16; fine-tune tốt kỳ vọng 0.4-0.5+)")


if __name__ == "__main__":
    main()
