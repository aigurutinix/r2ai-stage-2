"""Certify that a patched submission changed only what it was allowed to change.

The patch claims a specific safety property: every question it touches was
provably wrong before, and nothing else moves. That claim is worth checking
mechanically before spending a submission on it, because the failure mode is
silent — a stray edit to a question that was already correct costs a point and
looks like nothing in the diff summary.

Checks, in order of what they would cost if violated:

  1. every changed answer belongs to a question that was provably wrong in the
     base (else the patch can lose points it already had);
  2. `relevant_tables` and `relevant_docs` are byte-identical everywhere (else
     TABLES_F2 moves and the comparison against the base is no longer clean);
  3. every `csv_path` referenced exists in the archive (a missing CSV scores the
     question zero on execution however right the program is);
  4. the record count and id set are unchanged (all 1,012 must ship).

Usage:  PYTHONPATH=src python scripts/_verify_patch.py base.zip patched.zip
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from patch_submission import wrong_for_sure  # noqa: E402


def load(name: str):
    with zipfile.ZipFile(ROOT / "submissions" / name) as z:
        return (
            {r["id"]: r for r in json.loads(z.read("submission.json"))},
            set(z.namelist()),
        )


def main() -> None:
    base_name, patched_name = sys.argv[1], sys.argv[2]
    # `--weak-ok` is for the second kind of patch, which deliberately replaces
    # answers that were merely *likely* wrong (they came from a 5.9% or 7.2%
    # branch) rather than provably wrong. That patch can lose points, so the
    # check becomes a reported count instead of a failure — but every other
    # guarantee still has to hold.
    weak_ok = "--weak-ok" in sys.argv
    base, _ = load(base_name)
    patched, entries = load(patched_name)
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }

    failures: list[str] = []

    if set(base) != set(patched):
        failures.append(f"id set changed: {len(base)} -> {len(patched)}")
    if len(patched) != 1012:
        failures.append(f"submission has {len(patched)} records, not 1012")

    changed = []
    speculative = 0
    for qid in sorted(set(base) & set(patched)):
        if base[qid].get("answer") != patched[qid].get("answer"):
            changed.append(qid)
            why = wrong_for_sure(parsed[qid], base[qid])
            if why is None:
                if weak_ok:
                    speculative += 1
                else:
                    failures.append(
                        f"id={qid} changed but the base answer was not provably "
                        f"wrong ({base[qid].get('answer')!r} -> "
                        f"{patched[qid].get('answer')!r})")
        for field in ("relevant_tables", "relevant_docs"):
            if base[qid][field] != patched[qid][field]:
                failures.append(f"id={qid} {field} changed")

    missing = []
    for qid, record in patched.items():
        for item in record["evidence"]:
            if item["csv_path"] not in entries:
                missing.append((qid, item["csv_path"]))
    if missing:
        failures.append(f"{len(missing)} evidence csv paths absent from the zip, "
                        f"e.g. {missing[:3]}")

    print(f"{base_name} -> {patched_name}")
    print(f"  answers changed : {len(changed)}")
    if weak_ok:
        print(f"  of those, speculative (base not provably wrong): {speculative}"
              f"  <- these can lose points")
    print(f"  records         : {len(patched)}")
    print(f"  archive entries : {len(entries)}")
    if failures:
        print(f"\n  FAILED {len(failures)} check(s):")
        for line in failures[:20]:
            print(f"    - {line}")
        raise SystemExit(1)
    if weak_ok and speculative:
        print(f"\n  OK — declarations are untouched and all evidence resolves, but "
              f"{speculative} of the {len(changed)} replacements are speculative: "
              f"the base answer there was not provably wrong, so this patch can "
              f"lose points as well as gain them.")
    else:
        print("\n  OK — every changed answer was provably wrong in the base, "
              "declarations are untouched, and all evidence resolves.")


if __name__ == "__main__":
    main()
