"""Clean emitted picker files to verified-only, print distribution + samples."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

for name in ("picker_train", "picker_heldout"):
    path = ROOT / "artifacts" / "fresh" / f"{name}.jsonl"
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    clean = [p for p in pairs if p["verified"]]
    dropped = len(pairs) - len(clean)
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in clean),
                    encoding="utf-8")
    dist: dict[int, int] = {}
    for p in clean:
        dist[len(p["choices"])] = dist.get(len(p["choices"]), 0) + 1
    print(f"{name}: kept {len(clean)} (dropped {dropped} unverified), "
          f"choices dist {dict(sorted(dist.items()))}")

held = [json.loads(line) for line
        in (ROOT / "artifacts" / "fresh" / "picker_heldout.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()]
print("\n=== 3 held-out samples ===")
for p in held[:3]:
    print(f"\nid={p['id']} gold=r{p['gold_r']},c{p['gold_c']} "
          f"value={p['gold_value']} | {p['question'][:95]}")
    for j, ch in enumerate(p["choices"]):
        mark = "  <= GOLD" if j == p["answer_index"] else ""
        print(f"   [{j}] r{ch['r']:<3d} {ch['label'][:88]}{mark}")
