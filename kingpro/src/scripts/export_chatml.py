"""Xuất bộ SFT -> ChatML (conversations) cho Unsloth, có DEDUP + tách val.
Format: {"conversations":[{role:system},{role:user},{role:assistant}], type, answer}
Chạy: python scripts/export_chatml.py build/sft_data.jsonl build/sft_chatml.jsonl [--val 0.05]
"""
import json, sys, random, collections, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--val", type=float, default=0.05, help="tỉ lệ tách validation (eval theo execution)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    seen, out = set(), []
    for r in rows:
        q = next((l for l in r["user"].split("\n") if l.startswith("Câu hỏi:")), r["user"][:80])
        key = (q.strip(), r["assistant"][:80])
        if key in seen:
            continue
        seen.add(key)
        out.append({"conversations": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["assistant"]},
        ], "type": r["type"], "answer": r["answer"], "table_refs": r["meta"]["table_refs"]})
    random.seed(0); random.shuffle(out)
    n_val = int(len(out) * a.val)
    val, train = out[:n_val], out[n_val:]
    with open(a.dst, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    valpath = a.dst.replace(".jsonl", "_val.jsonl")
    with open(valpath, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["type"] for r in out)
    tot = len(out)
    print(f"xuất {tot} mẫu (dedup từ {len(rows)}) -> train {len(train)} / val {len(val)}")
    print(f"  train: {a.dst}\n  val:   {valpath}")
    print("=== phân bố loại ===")
    for k, v in c.most_common():
        print(f"  {k:14} {v:6} ({v*100//tot}%)")


if __name__ == "__main__":
    main()
