"""Create auditable formula challengers on top of the V63 private checkpoint."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/codegen_private_v63_final_audited_from_v60.jsonl"
CANDIDATES = ROOT / "artifacts/codegen_private_v61_balanced43_exact_from_v31.jsonl"

CORE_IDS = (111, 494, 512, 520, 521, 529, 677, 725, 726)
EXTENDED_IDS = CORE_IDS + (370, 473, 483)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(ids: tuple[int, ...], name: str) -> None:
    base = load(BASE)
    candidates = {row["id"]: row for row in load(CANDIDATES)}
    by_id = {row["id"]: row for row in base}
    for qid in ids:
        row = dict(candidates[qid])
        row["source"] = "audited_formula_challenger_v64"
        row["detail"] = "private challenger; " + str(row.get("detail", ""))
        row["detail_conf"] = 100.0
        by_id[qid] = row
    out = ROOT / f"artifacts/codegen_private_{name}_from_v63.jsonl"
    out.write_text("".join(json.dumps(by_id[row["id"]], ensure_ascii=False) + "\n"
                           for row in base), encoding="utf-8")
    print(f"{name}: overrides={list(ids)} -> {out}")


if __name__ == "__main__":
    write(CORE_IDS, "v64_formula_core9")
    write(EXTENDED_IDS, "v65_formula_extended12")
