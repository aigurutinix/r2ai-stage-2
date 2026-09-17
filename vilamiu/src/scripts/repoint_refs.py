"""Declare the tables our programs actually read, ahead of the retrieval filler.

Three submissions in a row moved EXECUTION and left every table metric frozen at
TABLES_F2 0.5641 — by design, because the patches never touched
`relevant_tables`. That axis is a third of the macro average and it is where we
lead, so it was worth protecting while the answer side was being measured. It is
now the only axis left: ANSWER tracks EXECUTION to within 0.002 for every team on
the board, so the two are one axis, and retrieval is the other.

The board says where the loss is. DOCS precision is 0.9705 against TABLES
precision 0.3357, so about 65% of the refs we declare name a gold document at a
non-gold line, and the implied k/g is 2.39 — we declare nearly two and a half
tables for every one the gold program reads.

Something new is available to attack that. Gold `relevant_tables` is defined as
the tables the gold program reads, and we now have 889 generated programs that
each read a specific handful. Where our program is right, the table it read *is*
a gold table by that definition — a signal the BM25 ranking cannot produce,
because it never runs anything.

This keeps k fixed and only changes membership: evidence first, then the existing
refs, truncated to the length the question already declared. Precision's
denominator is therefore unchanged and only the numerator can move, in either
direction.

Usage:
  PYTHONPATH=src python scripts/repoint_refs.py --base sub15.zip --out refs.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--used-frames-only", action="store_true", default=True,
                        help="promote only the frames the program names (default)")
    parser.add_argument("--all-frames", dest="used_frames_only",
                        action="store_false",
                        help="promote every bound table, read or not")
    parser.add_argument("--max-promote", type=int, default=3,
                        help="never move more than this many refs to the front; "
                             "0 disables the cap")
    args = parser.parse_args()

    base_path = ROOT / "submissions" / args.base
    with zipfile.ZipFile(base_path) as z:
        records = json.loads(z.read("submission.json"))
        payloads = {
            name.split("/", 1)[1]: z.read(name)
            for name in z.namelist() if name.startswith("data/")
        }
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    # `evidence` names CSV files; the ref format is `<doc>|<1-based start line>`.
    # Recovering the key from the file name is exact — `_csv_name` builds it as
    # `<doc_name>_table_<table_id>.csv` — but the doc name itself may contain
    # underscores, so split on the last `_table_`.
    def ref_for(csv_path: str) -> str | None:
        name = csv_path.split("/", 1)[-1]
        if not name.endswith(".csv") or "_table_" not in name:
            return None
        doc, _, tail = name[:-4].rpartition("_table_")
        if not tail.isdigit():
            return None
        try:
            meta = store.meta(TableKey(doc, int(tail)))
        except Exception:
            return None
        return f"{doc}|{int(meta.start_line)}"

    stats: Counter[str] = Counter()
    moved = 0
    for record in records:
        refs = list(record["relevant_tables"])
        width = len(refs)
        code = record.get("pandas_query") or ""
        # A generated program is *bound* to every table it was offered — up to
        # fourteen — but it *reads* only the ones it names. Declaring all of them
        # would push k past the question's budget and evict the retrieval refs
        # wholesale, which is the opposite of the intent: gold `relevant_tables`
        # is the set the gold program reads, so only the frames this program
        # actually mentions are candidates for that role.
        used = set(re.findall(r"\bdf\d*\b", code))
        fresh: list[str] = []
        for item in record.get("evidence", []):
            if args.used_frames_only and item.get("variable") not in used:
                stats["frame_bound_but_unread"] += 1
                continue
            ref = ref_for(item.get("csv_path", ""))
            if ref is None:
                stats["unresolvable_evidence"] += 1
            elif ref not in fresh:
                fresh.append(ref)
        if args.max_promote and len(fresh) > args.max_promote:
            stats["promotion_capped"] += 1
            fresh = fresh[: args.max_promote]
        if not fresh:
            stats["no_evidence"] += 1
            continue
        stats["already_first"] += int(refs[:len(fresh)] == fresh)
        inside = sum(1 for r in fresh if r in refs)
        stats["evidence_already_declared"] += int(inside == len(fresh))
        stats["evidence_partly_absent"] += int(0 < inside < len(fresh))
        stats["evidence_wholly_absent"] += int(inside == 0)

        new_refs = (fresh + [r for r in refs if r not in fresh])[:width]
        if new_refs != refs:
            moved += 1
            if not args.dry_run:
                docs: list[str] = []
                for ref in new_refs:
                    doc = ref.split("|", 1)[0]
                    if doc not in docs:
                        docs.append(doc)
                for doc in record["relevant_docs"]:
                    if doc not in docs:
                        docs.append(doc)
                record["relevant_tables"] = new_refs
                record["relevant_docs"] = docs

    print(f"{args.base}: {len(records)} questions")
    for key, count in stats.most_common():
        print(f"  {key:28s} {count:5d}")
    print(f"  questions whose refs move    {moved:5d}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    referenced = {
        item["csv_path"].split("/", 1)[1]
        for record in records for item in record["evidence"]
    }
    missing = referenced - set(payloads)
    if missing:
        raise SystemExit(f"{len(missing)} csv files referenced but absent")

    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(records, ensure_ascii=False, indent=1))
        for name in sorted(referenced):
            archive.writestr(f"data/{name}", payloads[name])
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
