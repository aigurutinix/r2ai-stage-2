"""Build zip from audit: vote3_hardhop + selected pipeline splices."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

# tab_verified rows that pass reexec but pick the wrong line item (hand review).
TAB_VERIFIED_REJECT = {
    10, 109, 156, 164, 165, 206, 276, 666, 739, 747, 775, 781, 801, 811, 964,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag-max", type=float, default=999.0)
    ap.add_argument("--layers", default="address_plan,tab_prog,tab_verified")
    ap.add_argument(
        "--verdicts",
        default="OK,SUSPECT_sign,SUSPECT_address_2x",
        help="Comma-separated audit verdicts to include",
    )
    ap.add_argument(
        "--exclude-verdicts",
        default="SUSPECT_mag3x",
        help="Always drop these verdicts",
    )
    ap.add_argument(
        "--reject-tab-verified",
        action="store_true",
        default=True,
        help="Drop hand-reviewed bad tab_verified ids",
    )
    ap.add_argument("--no-reject-tab-verified", dest="reject_tab_verified",
                    action="store_false")
    ap.add_argument("--dest", default="submissions/vote3_pipeline_large.zip")
    args = ap.parse_args()
    layers = set(args.layers.split(","))
    verdicts = set(args.verdicts.split(","))
    drop_verdicts = set(args.exclude_verdicts.split(",")) if args.exclude_verdicts else set()

    aud = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    allow = {
        a["id"]
        for a in aud
        if a["verdict"] in verdicts
        and a["verdict"] not in drop_verdicts
        and a["layer"] in layers
        and a["mag"] <= args.mag_max
        and not (args.reject_tab_verified
                 and a["layer"] == "tab_verified"
                 and a["id"] in TAB_VERIFIED_REJECT)
    }

    base_path = ROOT / "submissions/vote3_hardhop.zip"
    pipe_path = ROOT / "submissions/vote3_pipeline.zip"
    dest = ROOT / args.dest

    with zipfile.ZipFile(base_path) as base_z, zipfile.ZipFile(pipe_path) as pipe_z:
        rows = {r["id"]: r for r in json.loads(base_z.read("submission.json"))}
        pipe_rows = {r["id"]: r for r in json.loads(pipe_z.read("submission.json"))}
        files = {n: base_z.read(n) for n in base_z.namelist() if n != "submission.json"}

        applied = []
        for qid in sorted(allow):
            if qid not in pipe_rows:
                continue
            pr, br = pipe_rows[qid], rows[qid]
            if abs(float(pr["answer"]) - float(br["answer"])) <= 0.01:
                continue
            rows[qid] = pr
            for ev in pr.get("evidence") or []:
                p = ev["csv_path"]
                if p in pipe_z.namelist():
                    files[p] = pipe_z.read(p)
            applied.append(qid)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    fail = 0
    with zipfile.ZipFile(dest) as archive:
        for qid in applied:
            rec = rows[qid]
            ns: dict = {"pd": pd}
            for ev in rec.get("evidence") or []:
                ns[ev["variable"]] = pd.read_csv(
                    io.BytesIO(archive.read(ev["csv_path"])),
                    dtype=str, keep_default_na=False,
                )
            exec(rec["pandas_query"], ns, ns)  # noqa: S102
            if abs(float(rec["answer"]) - float(ns["result"])) > 0.01:
                fail += 1
                print(f"REEXEC FAIL {qid}")

    manifest = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_manifest.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_layer: dict[str, int] = {}
    for m in manifest:
        if m["id"] in applied:
            by_layer[m["layer"]] = by_layer.get(m["layer"], 0) + 1

    print(f"base: vote3_hardhop ({45} rule hops baked in)")
    print(f"filter: verdicts={verdicts} drop={drop_verdicts} layers={layers} "
          f"mag<={args.mag_max} reject_tab={args.reject_tab_verified}")
    print(f"candidates: {len(allow)}  applied: {len(applied)}  reexec_fail: {fail}")
    print(f"by layer: {by_layer}")
    print(f"ids: {applied}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
