"""Hand-check every pipeline splice before shipping more changes."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps
from answer_gate import verdict
from build_submission import unit_of
from build_dualread_zip import tab_answer, load_jsonl


def mag(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 999.0
    return max(abs(a / b), abs(b / a))


def reexec_row(rec: dict, blobs: dict[str, bytes]) -> tuple[bool, float | None, str]:
    try:
        ns: dict = {"pd": pd}
        for ev in rec.get("evidence") or []:
            ns[ev["variable"]] = pd.read_csv(
                io.BytesIO(blobs[ev["csv_path"]]), dtype=str, keep_default_na=False)
        exec(rec["pandas_query"], ns, ns)  # noqa: S102
        got = float(ns["result"])
        want = float(rec["answer"])
        return abs(got - want) <= 0.01, got, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)[:120]


def verify_tab_quote(manifest_row: dict, tab: dict, qs: dict) -> str | None:
    qid = manifest_row["id"]
    entry = tab.get(qid)
    if entry is None:
        return None
    question = qs[qid]
    computed = tab_answer(entry, question)
    if computed is None:
        return "tab_answer_none"
    if abs(computed - manifest_row["new_answer"]) > 0.01:
        return f"tab_math_mismatch:{computed}"
    # quote vs cell
    from pathlib import Path as P
    csv = ROOT / entry["csv"]
    if not csv.exists():
        return "csv_missing"
    import csv as csv_mod
    with csv.open(encoding="utf-8-sig", newline="") as f:
        grid = list(csv_mod.reader(f))
    row, col = entry["row"], entry["col"]
    if row + 1 >= len(grid) or col >= len(grid[row + 1]):
        return "addr_oob"
    actual = str(grid[row + 1][col]).strip()
    claimed = ps.parse_vn_number(str(entry.get("quoted", "")))
    present = ps.parse_vn_number(actual)
    if claimed is None or present is None or abs(claimed - present) > 0.01:
        return f"quote_mismatch:claimed={entry.get('quoted')} actual={actual}"
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    manifest = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_manifest.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    tab = load_jsonl(ROOT / "artifacts/fresh/tab_plan.jsonl")

    with zipfile.ZipFile(ROOT / "submissions/vote3_pipeline.zip") as z:
        pipe = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}

    verdicts: Counter[str] = Counter()
    details: list[dict] = []

    for m in manifest:
        qid = m["id"]
        rec = pipe[qid]
        q = qs[qid]
        old, new = float(m["old_answer"]), float(m["new_answer"])
        ratio = mag(old, new)
        gate = verdict(q, new)
        ok, got, err = reexec_row(rec, blobs)
        tab_err = None
        if m["layer"].startswith("tab"):
            tab_err = verify_tab_quote(m, tab, qs)

        if gate:
            v = "REJECT_gate"
        elif not ok:
            v = "REJECT_reexec"
        elif tab_err:
            v = "REJECT_tab"
        elif ratio >= 3:
            v = "SUSPECT_mag3x"
        elif old * new < 0 and abs(old) > 0.01:
            v = "SUSPECT_sign"
        elif m["layer"] == "address_plan" and ratio >= 2:
            v = "SUSPECT_address_2x"
        else:
            v = "OK"

        verdicts[v] += 1
        details.append({
            "id": qid,
            "layer": m["layer"],
            "block": m["block"],
            "old": old,
            "new": new,
            "mag": round(ratio, 2),
            "verdict": v,
            "tab_err": tab_err,
            "reexec_got": got,
            "gate": gate,
            "err": err,
            "question": q[:90],
        })

    out = ROOT / "artifacts/fresh/pipeline_audit.jsonl"
    out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details),
                   encoding="utf-8")

    print(f"audited {len(manifest)} splices\n")
    for name, count in verdicts.most_common():
        print(f"  {count:4d}  {name}")

    print("\n=== SUSPECT + REJECT (need human or drop) ===")
    bad = [d for d in details if not d["verdict"].startswith("OK")]
    for d in sorted(bad, key=lambda x: (-x["mag"], x["id"])):
        print(f"{d['id']:4d} [{d['layer']:14s}] {d['verdict']:18s} "
              f"{d['old']:.4g} -> {d['new']:.4g} mag={d['mag']}")
        if d.get("tab_err"):
            print(f"       tab: {d['tab_err']}")
        if d.get("gate"):
            print(f"       gate: {d['gate']}")

    ok_ids = [d["id"] for d in details if d["verdict"] == "OK"]
    print(f"\nOK to keep: {len(ok_ids)} / {len(manifest)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
