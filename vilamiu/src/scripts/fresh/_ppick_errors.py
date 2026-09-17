"""Error analysis of the p_pick baseline run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / "artifacts/fresh/ppick_baseline.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
held = {json.loads(l)["id"]: json.loads(l)
        for l in (ROOT / "artifacts/fresh/picker_heldout.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()}

for arm in ("zero", "fewshot"):
    armrows = {r["id"]: r for r in rows if r["arm"] == arm}
    wrong = [(i, r) for i, r in sorted(armrows.items()) if r["pick"] != r["gold"]]
    print(f"== {arm}: {len(wrong)} sai / {len(armrows)}")
    cats = {"tong_vs_chitiet": 0, "ky_khac": 0, "anh_em_gan_nhau": 0, "tu_choi": 0}
    for i, r in wrong:
        p = held[i]
        gold = p["choices"][p["answer_index"]]["label"]
        if r["pick"] < 0:
            cats["tu_choi"] += 1
            continue
        pick = p["choices"][r["pick"]]["label"]
        gl, pl = gold.lower(), pick.lower()
        g_total = ("cộng" in gl) or ("tổng" in gl)
        p_total = ("cộng" in pl) or ("tổng" in pl)
        if g_total != p_total:
            cats["tong_vs_chitiet"] += 1
        elif any(k in gl for k in ("cuối năm", "đầu năm", "năm nay", "số dư")):
            cats["ky_khac"] += 1
        else:
            cats["anh_em_gan_nhau"] += 1
        print(f"  id{i:4d} [{r['pick']}] {pick[:58]}")
        print(f"        gold [{p['answer_index']}] {gold[:58]}")
    print("  ", cats)

z = {r["id"]: r["pick"] == r["gold"] for r in rows if r["arm"] == "zero"}
f = {r["id"]: r["pick"] == r["gold"] for r in rows if r["arm"] == "fewshot"}
both = sum(1 for k in z if z[k] and f[k])
only_z = sum(1 for k in z if z[k] and not f[k])
only_f = sum(1 for k in z if not z[k] and f[k])
print(f"\nđúng cả hai={both} · chỉ zero={only_z} · chỉ fewshot={only_f}")
