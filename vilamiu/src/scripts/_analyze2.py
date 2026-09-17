"""Phan tich moi tang cua pipeline: dem, so sanh, tim chokepoint."""
import json, sys, collections
sys.path.insert(0, "src")
from pathlib import Path

root = Path(".")

def load_jsonl(p):
    rows = {}
    if not p.exists(): return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok") and r.get("value") is not None:
                rows[r["id"]] = r
    return rows

from vifin.corpus.numeric import is_correct as ic
from vifin.query.parse import parse_all
import vifin.answering.lookup as lk

located = load_jsonl(root / "artifacts" / "located.jsonl")
embed_loc = load_jsonl(root / "artifacts" / "embed_located.jsonl")
planned = load_jsonl(root / "artifacts" / "planned.jsonl")
generated = load_jsonl(root / "artifacts" / "generated_full.jsonl")

parsed = parse_all(root / "data" / "questions" / "questions.jsonl",
                    root / "data" / "code_stock.csv")
parsed_map = {p.id: p for p in parsed}
all_ids = set(parsed_map)

print(f"Total questions: {len(parsed)}")
print()

# 1. Artifact coverage
print("=== ARTIFACT COVERAGE ===")
print(f"  model_located: {len(located)}")
print(f"  embed_located: {len(embed_loc)}")
print(f"  planned:       {len(planned)}")
print(f"  generated:     {len(generated)}")
print()

# 2. Localiser overlap and agreement
both_loc = set(located) & set(embed_loc)
print(f"=== LOCALISER OVERLAP ===")
print(f"  both: {len(both_loc)}  only_model: {len(set(located)-set(embed_loc))}  only_embed: {len(set(embed_loc)-set(located))}")
if both_loc:
    agree = sum(1 for qid in both_loc if ic(located[qid]["value"], embed_loc[qid]["value"]))
    print(f"  agree (is_correct): {agree}/{len(both_loc)} ({agree/len(both_loc):.1%})")
print()

# 3. Plan op distribution
op_counts = collections.Counter()
for qid, row in planned.items():
    op_counts[row.get("op", "?")] += 1
print(f"=== PLAN OPS ({len(planned)} total) ===")
for op, cnt in op_counts.most_common():
    print(f"  {op:20s} {cnt:4d}")
print()

# 4. Gaps: questions with NO artifact
gen_ids = set(generated)
plan_ids = set(planned)
loc_ids = set(located) | set(embed_loc)
has_artifact = gen_ids | plan_ids | loc_ids
no_artifact = all_ids - has_artifact
print(f"=== NO ARTIFACT (fallback/scan candidates) ===")
print(f"  total: {len(no_artifact)}")

# Categorize no-artifact questions
single = sum(1 for qid in no_artifact
            if qid in parsed_map and parsed_map[qid].unit_scale
            and lk.is_single_lookup(parsed_map[qid].question))
derived = sum(1 for qid in no_artifact
              if qid in parsed_map
              and not (parsed_map[qid].unit_scale and lk.is_single_lookup(parsed_map[qid].question)))
no_unit = sum(1 for qid in no_artifact
              if qid in parsed_map and not parsed_map[qid].unit_scale)
print(f"  single-lookup (currency, no locate/plan/gen): {single}")
print(f"  derived/non-currency: {derived}")
print(f"  no unit_scale: {no_unit}")
print()

# 5. Generated vs plan on overlapping questions
gen_and_plan = gen_ids & plan_ids
print(f"=== GENERATED vs PLAN ===")
print(f"  overlap: {len(gen_and_plan)}  gen_only: {len(gen_ids-plan_ids)}  plan_only: {len(plan_ids-gen_ids)}")
if gen_and_plan:
    same = sum(1 for qid in gen_and_plan if ic(generated[qid]["value"], planned[qid]["value"]))
    print(f"  agree: {same}/{len(gen_and_plan)}  disagree: {len(gen_and_plan)-same}")
print()

# 6. Where are the single-lookup questions?
all_single = set(qid for qid, p in parsed_map.items()
                 if p.unit_scale and lk.is_single_lookup(p.question))
print(f"=== SINGLE-LOOKUP QUESTIONS ===")
print(f"  total: {len(all_single)}")
print(f"  in model_located: {len(all_single & set(located))}")
print(f"  in embed_located: {len(all_single & set(embed_loc))}")
print(f"  in either localiser: {len(all_single & loc_ids)}")
print(f"  in neither: {len(all_single - loc_ids)}")
print()

# 7. Derived questions
all_derived = all_ids - all_single
print(f"=== DERIVED QUESTIONS ===")
print(f"  total: {len(all_derived)}")
print(f"  in generated: {len(all_derived & gen_ids)}")
print(f"  in planned: {len(all_derived & plan_ids)}")
print(f"  in either: {len(all_derived & (gen_ids | plan_ids))}")
print(f"  in neither: {len(all_derived - (gen_ids | plan_ids))}")
print()

# 8. Unit distribution for single-lookup
unit_dist = collections.Counter()
for qid in all_single:
    if qid in parsed_map:
        unit_dist[parsed_map[qid].target_unit] += 1
print(f"=== UNIT DISTRIBUTION (single-lookup) ===")
for unit, cnt in unit_dist.most_common(10):
    print(f"  {unit:15s} {cnt:4d}")
print()

# 9. Op distribution for derived questions
print(f"=== DERIVED QUESTION TYPES ===")
ratios = set()
diffs = set()
comparisons = set()
other = set()
for qid in all_derived:
    if qid not in parsed_map: continue
    q = parsed_map[qid].question.lower()
    if "tỷ lệ" in q or "biên" in q or "tỷ số" in q or "hệ số" in q:
        ratios.add(qid)
    elif "chênh lệch" in q or "hiệu số" in q or "thay đổi" in q:
        diffs.add(qid)
    elif "trong các" in q or "trong nhóm" in q or "so với" in q or "cao nhất" in q or "thấp nhất" in q:
        comparisons.add(qid)
    else:
        other.add(qid)
print(f"  ratios: {len(ratios)}  diffs: {len(diffs)}  comparisons: {len(comparisons)}  other: {len(other)}")
print(f"  ratios in plan: {len(ratios & plan_ids)}  in gen: {len(ratios & gen_ids)}")
print(f"  diffs in plan: {len(diffs & plan_ids)}  in gen: {len(diffs & gen_ids)}")
print(f"  comparisons in plan: {len(comparisons & plan_ids)}  in gen: {len(comparisons & gen_ids)}")