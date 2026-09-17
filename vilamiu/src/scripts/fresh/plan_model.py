"""Turn the model's picks into cell addresses, keeping every check around them.

The model names a source, a code and a period. Everything after that stays
deterministic and verified:

  the code must exist in the block it was shown, or the pick is dropped
  the cell comes from the address book, whose reads the printed identities check at
  97–99.6% and the cross-year comparison at 90.9%
  a note pick is resolved to a row inside that one note — about ten candidates, not a
  corpus-wide search
  the sign rule and the unit conversion are the same measured ones as everywhere else

So a bad pick costs coverage, never a fabricated number. That is the property the
earlier attempt lacked, where the model returned a coordinate straight into the
answer with nothing able to reject it.

Writes a plan in the same shape the other planners use, so `build_submission.py`
consumes it unchanged.

Usage:
  python scripts/fresh/plan_model.py --replies artifacts/fresh/replies_probe.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from plan_notes import PERIOD_LABEL_RE  # noqa: E402
from refine_codes import tokens_exact  # noqa: E402
from render_block import Corpus  # noqa: E402
from score_model import TM_RE, parse  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", required=True)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_probe.jsonl")
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/model_plan.jsonl")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)

    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    counters: Counter[str] = Counter()
    plan, samples = [], []
    for line in (ROOT / args.replies).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = record["id"]
        info = meta.get(qid)
        if info is None:
            continue
        spec = parse(record["reply"])
        if not spec:
            counters["khong doc duoc JSON"] += 1
            continue
        source = str(spec.get("nguon", "")).strip()
        code = str(spec.get("ma", "")).strip()
        period = str(spec.get("ky", "")).strip()
        if period not in ("current", "prior") or not code:
            counters["dau ra khong hop le"] += 1
            continue
        key = (info["ticker"], info["year"], info["scope"])

        if source in ("cdkt", "kqkd", "lctt"):
            slot = corpus.rows.get(key, {}).get((source, code))
            if not slot:
                counters["ma khong co trong khoi"] += 1
                continue
            cell = slot.get(period) or slot.get("current") or slot.get("prior")
            if not cell:
                counters["khong co o cho ky do"] += 1
                continue
            plan.append({"id": qid, "source": "maso", "kind": source, "code": code,
                         "period": period, "label": cell["label"],
                         "row": cell["row"], "col": cell["col"],
                         "csv": cell["csv"], "doc": cell["doc"],
                         "table_id": cell["table_id"],
                         "table_ref": cell["table_ref"], "scale": cell["scale"],
                         "score": 1.0, "picked_by": "model"})
            counters["dia chi tu bao cao chinh"] += 1
            if len(samples) < args.show:
                samples.append(f"  id={qid:<5d} {source}/{code} {period}  "
                               f"{str(cell['label'])[:64]}\n"
                               f"     hoi: {info['question'][:92]}")
            continue

        if source != "TM":
            counters["nguon khong hop le"] += 1
            continue
        match = TM_RE.match(code)
        if not match:
            counters["so TM khong doc duoc"] += 1
            continue
        table_id = int(match.group(1))
        note = next((n for n in corpus.notes.get(key, ())
                     if n["table_id"] == table_id), None)
        if note is None:
            counters["so TM khong co trong muc luc"] += 1
            continue

        # Inside one note, the question's words pick the row among about ten.
        probe = tokens_exact(info["question"])
        try:
            with (ROOT / note["csv"]).open(encoding="utf-8-sig", newline="") as file:
                grid = [row for row in csv.reader(file)]
        except OSError:
            counters["khong doc duoc csv thuyet minh"] += 1
            continue
        scale = Counter(tie["scale"] for tie in note["ties"]).most_common(1)[0][0]
        best = None
        for index, row in enumerate(grid[1:]):
            if not row:
                continue
            label = max((str(c).strip() for c in row
                         if not any(ch.isdigit() for ch in str(c))),
                        key=len, default="")
            if not label or PERIOD_LABEL_RE.match(label.strip()):
                continue
            label_tokens = tokens_exact(label)
            shared = probe & label_tokens
            if not shared:
                continue
            columns = [i for i, cell in enumerate(row)
                       if str(cell).strip()
                       and not ps.BARE_INT_RE.match(str(cell).strip())
                       and ps.parse_vn_number(str(cell)) is not None]
            wanted = 1 if period == "prior" else 0
            if len(columns) <= wanted:
                continue
            score = len(shared) / len(probe | label_tokens)
            if best is None or score > best[0]:
                best = (score, index, label, columns[wanted])
        if best is None:
            counters["mo duoc TM nhung khong khop dong nao"] += 1
            continue
        score, row_index, label, column = best
        plan.append({"id": qid, "source": "note", "kind": "thuyet minh",
                     "code": f"TM{table_id}", "period": period, "label": label,
                     "row": row_index, "col": column, "csv": note["csv"],
                     "doc": note["doc"], "table_id": note["table_id"],
                     "table_ref": note["table_ref"], "scale": scale,
                     "score": round(score, 3), "picked_by": "model",
                     "ref": note["paired"][0] if note["paired"] else ""})
        counters["dia chi tu thuyet minh"] += 1
        if len(samples) < args.show:
            samples.append(f"  id={qid:<5d} TM{table_id} dong={row_index} "
                           f"diem={score:.2f}  {label[:56]}\n"
                           f"     hoi: {info['question'][:92]}")

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\ntong dia chi: {len(plan)}")
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
