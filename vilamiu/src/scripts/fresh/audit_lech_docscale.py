"""Audit LECH fixable via document_scale + table_norm."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import table_norm as tn  # noqa: E402
from build_submission import unit_of  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    doc_scale = json.loads(
        (ROOT / "artifacts/fresh/doc_scale.json").read_text(encoding="utf-8"))
    audit = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/lech_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    with zipfile.ZipFile(ROOT / "submissions/method_v1.zip") as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    fix = 0
    for item in audit:
        if item["kind"] in ("LOI", "sign", "scale_div_1e+06", "scale_div_1e+03"):
            continue
        got = item.get("got")
        want = item.get("want")
        if got is None or want is None or abs(want) < 1e-9:
            continue
        row = rows[item["id"]]
        ev = row["evidence"][0]
        blob = blobs[ev["csv_path"]]
        q = qs[item["id"]]
        _, qu = unit_of(q)
        tab = tn.detect_table_unit_vnd(blob)
        stem = ev["csv_path"].replace("data/", "").rsplit("_table_", 1)[0]
        ds = doc_scale.get(stem)
        candidates: set[float] = set()
        if tab and qu:
            candidates.add(qu / tab)
            candidates.add(tab / qu)
        if ds and qu:
            candidates.add(qu / ds)
            candidates.add(ds / qu)
        eff = tn.effective_divide_unit(q, blob)
        if eff and eff > 0:
            candidates.add(eff)
            candidates.add(1 / eff)
        for fac in candidates:
            if fac <= 0 or abs(fac - 1) < 1e-6:
                continue
            for trial in (got / fac, got * fac):
                if abs(trial - want) <= 0.01:
                    print(
                        f"{item['id']} fac={fac:.4g} got={got} want={want} "
                        f"tab={tab} ds={ds} qu={qu}")
                    fix += 1
                    break
            else:
                continue
            break
    print(f"fixable {fix}")


if __name__ == "__main__":
    main()
