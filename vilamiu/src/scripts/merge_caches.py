"""Union two generation caches, keeping whichever model solved each question.

The two models fail differently. On the same 39 questions Qwen3-8B produced 24
executable programs but 7 of them returned a bare 0, while Qwen2.5-Coder-14B
produced 16 with none returning 0. Neither dominates, so the union can cover
more than either alone — the same reasoning that lifted DOCS_F2 from 0.9324 to
0.9504 by unioning two retrieval orderings.

Every candidate is re-executed here, against the local pandas the packager also
uses, so a program that only ran on the remote's pandas 3.0.5 cannot slip in.

Usage:  python scripts/merge_caches.py out.jsonl in1.jsonl in2.jsonl ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402


def usable(row: dict, store: TableStore) -> float | None:
    """The value this program actually produces locally, if it is fit to ship."""

    if not row.get("code") or reads_no_frame(row["code"]):
        return None
    keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
    tables = {name: store.rows(key) for name, key in zip(row["variables"], keys)}
    outcome = run_query(row["code"], tables)
    if not outcome.ok or outcome.value == 0.0:
        return None
    return outcome.value


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out_path = Path(sys.argv[1])
    sources = [Path(p) for p in sys.argv[2:]]

    root = Path(__file__).resolve().parents[1]
    store = TableStore.load(root / "artifacts" / "tables.parquet")

    merged: dict[int, dict] = {}
    credit: dict[str, int] = {}
    for source in sources:
        kept = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in merged:
                continue  # first source listed wins, so order them by trust
            value = usable(row, store)
            if value is None:
                continue
            row["value"], row["ok"] = value, True
            merged[row["id"]] = row
            kept += 1
        credit[source.name] = kept
        print(f"  {source.name:34s} contributed {kept}")

    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged.values()),
        encoding="utf-8",
    )
    print(f"\n{len(merged)} usable programs -> {out_path}")


if __name__ == "__main__":
    main()
