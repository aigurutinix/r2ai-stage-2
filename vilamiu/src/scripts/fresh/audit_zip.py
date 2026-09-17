"""Audit a built zip vs hardhop: reexec, magnitude, layer, samples."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from answer_gate import verdict  # noqa: E402
from build_audited_zip import TAB_VERIFIED_REJECT  # noqa: E402


def mag(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 999.0
    return max(abs(a / b), abs(b / a))


def load_sub(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as z:
        rows = json.loads(z.read("submission.json"))
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}
    return {r["id"]: r for r in rows}, blobs


def reexec(rec: dict, blobs: dict[str, bytes]) -> tuple[bool, float | None, str]:
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
        return False, None, str(exc)[:100]


def audit_zip(zip_path: Path, label: str) -> list[dict]:
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    manifest = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_manifest.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    audit = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    base_rows, _ = load_sub(ROOT / "submissions/vote3_hardhop.zip")
    v3_rows, _ = load_sub(ROOT / "submissions/vote3.zip")
    cand_rows, blobs = load_sub(zip_path)

    details: list[dict] = []
    for qid in sorted(cand_rows):
        br = base_rows[qid]
        cr = cand_rows[qid]
        old_hh = float(br["answer"])
        new = float(cr["answer"])
        if abs(old_hh - new) <= 0.01:
            continue
        v3 = float(v3_rows[qid]["answer"])
        m = manifest.get(qid, {})
        a = audit.get(qid, {})
        ratio = mag(old_hh, new)
        ok, got, err = reexec(cr, blobs)
        gate = verdict(qs[qid], new)
        v = a.get("verdict", "?")
        layer = m.get("layer", "?")

        risk = "OK"
        if not ok:
            risk = "REJECT_reexec"
        elif gate:
            risk = "REJECT_gate"
        elif layer == "tab_verified" and qid in TAB_VERIFIED_REJECT:
            risk = "REJECT_bad_tab_row"
        elif ratio >= 3:
            risk = "SUSPECT_mag3x"
        elif old_hh * new < 0 and abs(old_hh) > 0.01:
            risk = "SUSPECT_sign"
        elif layer == "address_plan" and ratio >= 2:
            risk = "SUSPECT_address_2x"
        elif v.startswith("SUSPECT"):
            risk = v

        details.append({
            "id": qid,
            "layer": layer,
            "audit_verdict": v,
            "risk": risk,
            "hardhop": old_hh,
            "vote3": v3,
            "new": new,
            "mag_vs_hardhop": round(ratio, 2),
            "reexec_ok": ok,
            "reexec_got": got,
            "gate": gate,
            "err": err,
            "block": m.get("block", "?"),
            "question": qs[qid][:100],
        })

    verdicts = Counter(d["risk"] for d in details)
    layers = Counter(d["layer"] for d in details)

    print(f"\n{'=' * 72}")
    print(f"{label}: {zip_path.name}")
    print(f"  splices vs hardhop: {len(details)}")
    print(f"  by layer: {dict(layers)}")
    print(f"  by risk: {dict(verdicts.most_common())}")

    bad = [d for d in details if d["risk"] != "OK"]
    if bad:
        print(f"\n  FLAGGED ({len(bad)}):")
        for d in sorted(bad, key=lambda x: (-x["mag_vs_hardhop"], x["id"])):
            print(f"    Q{d['id']:4d} [{d['layer']:14s}] {d['risk']:22s} "
                  f"hh={d['hardhop']:.4g} -> {d['new']:.4g} mag={d['mag_vs_hardhop']}")
            print(f"         {d['question'][:90]}")

    # vote3 alignment: does new match vote3 or diverge from both?
    agree_v3 = sum(1 for d in details if abs(d["new"] - d["vote3"]) <= 0.01)
    vs_v3 = sum(1 for d in details if abs(d["hardhop"] - d["vote3"]) <= 0.01)
    print(f"\n  vs vote3: {agree_v3}/{len(details)} new==vote3, "
          f"{vs_v3}/{len(details)} hardhop was vote3")

    suspect = [d for d in details if d["risk"] != "OK" or d["mag_vs_hardhop"] >= 1.5]
    print(f"\n  TOP suspect (mag>=1.5 or flagged), max 20:")
    for d in sorted(suspect, key=lambda x: (-x["mag_vs_hardhop"], x["id"]))[:20]:
        print(f"    Q{d['id']:4d} {d['risk']:18s} mag={d['mag_vs_hardhop']:5.2f} "
              f"v3={d['vote3']:.4g} hh={d['hardhop']:.4g} -> {d['new']:.4g} [{d['layer']}]")

    return details


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    all_details: dict[str, list[dict]] = {}
    for name, path in [
        ("MEDIUM (67)", ROOT / "submissions/vote3_pipeline_medium.zip"),
        ("LARGE (107)", ROOT / "submissions/vote3_pipeline_large.zip"),
        ("FULL pipeline (167)", ROOT / "submissions/vote3_pipeline.zip"),
    ]:
        if path.exists():
            all_details[name] = audit_zip(path, name)

    # overlap medium vs large
    if "MEDIUM (67)" in all_details and "LARGE (107)" in all_details:
        m = {d["id"] for d in all_details["MEDIUM (67)"]}
        l = {d["id"] for d in all_details["LARGE (107)"]}
        extra = l - m
        print(f"\n{'=' * 72}")
        print(f"LARGE adds {len(extra)} on top of MEDIUM: {sorted(extra)}")

    out = ROOT / "artifacts/fresh/zip_audit_report.jsonl"
    rows = []
    for label, details in all_details.items():
        for d in details:
            rows.append({"zip": label, **d})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
