"""Build the corpus index the organisers' generator could actually reach.

Two artefacts come out of one pass over the 1,973 reports.

**The reachable address book.** Their statement parser is the provenance of every
gold answer about the three primary statements, and it discards a table unless it
finds `Mã số` codes, classifies the statement kind, AND resolves a unit scale. A
table it discards produced no question. So the set of (document, kind, ma_so) it can
reach is an upper bound on where those answers live — a constraint derived from the
generator's own limits rather than from guessing.

**The consensus code dictionary.** `Mã số` is fixed by Thông tư 200, so the same code
carries the same concept across every company and year. Each code is therefore
observed with hundreds of independent OCR renderings of its label, and the majority
label is clean where any single one may not be. That inverts the usual problem: a
question's metric name has to be matched against a consensus label for a code, not
against one noisy row of one document.

Writes `artifacts/fresh/statements.jsonl` (one record per parsed statement) and
`artifacts/fresh/ma_so_labels.json` (code -> ranked labels, per statement kind).

Usage:  python scripts/fresh/build_index.py --limit 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_statements import read_document, parse_statement, fold  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC_RE = re.compile(r"^(.+?)_financial_statements_((?:19|20)\d{2})_(.+)$")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = every report")
    parser.add_argument("--out", default="artifacts/fresh")
    args = parser.parse_args()

    files = sorted((ROOT / "data" / "financial_statements").glob("*/*/*/*.txt"))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} bao cao", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / "statements.jsonl").open("w", encoding="utf-8")

    labels: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    stats: Counter[str] = Counter()
    per_doc_kinds: Counter[str] = Counter()
    started = time.time()

    for index, path in enumerate(files, start=1):
        doc_name = path.parent.name
        match = DOC_RE.match(doc_name)
        ticker = match.group(1) if match else ""
        year = match.group(2) if match else ""
        scope = match.group(3) if match else ""

        found = 0
        try:
            tables = read_document(path)
        except Exception as error:  # noqa: BLE001
            stats[f"loi doc: {type(error).__name__}"] += 1
            continue
        stats["bang"] += len(tables)
        for table in tables:
            statement = parse_statement(table)
            if statement is None:
                continue
            found += 1
            stats[f"bao cao {statement.kind}"] += 1
            record = {
                "doc": doc_name,
                "ticker": ticker,
                "year": year,
                "scope": scope,
                "table_id": statement.table_id,
                "page_no": table.page_no,
                "kind": statement.kind,
                "scale": statement.scale,
                "current": {code: [cell.value, cell.label, cell.row_idx, cell.col_idx,
                                   cell.raw]
                            for code, cell in statement.current.items()},
                "prior": {code: [cell.value, cell.label, cell.row_idx, cell.col_idx,
                                 cell.raw]
                          for code, cell in statement.prior.items()},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for code, cell in statement.current.items():
                text = " ".join(str(cell.label).split())
                if text:
                    labels[statement.kind][code][text] += 1
        per_doc_kinds["co bao cao chinh" if found else "khong co bao cao chinh"] += 1
        if index % 200 == 0:
            print(f"  {index}/{len(files)}  {time.time() - started:.0f}s", flush=True)

    handle.close()

    ranked = {
        kind: {code: [text for text, _ in counter.most_common(6)]
               for code, counter in sorted(codes.items())}
        for kind, codes in labels.items()
    }
    (out_dir / "ma_so_labels.json").write_text(
        json.dumps(ranked, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    for name, count in stats.most_common():
        print(f"  {name}: {count}")
    for name, count in per_doc_kinds.most_common():
        print(f"  {name}: {count}")
    for kind, codes in ranked.items():
        print(f"  ma so rieng biet trong {kind}: {len(codes)}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
