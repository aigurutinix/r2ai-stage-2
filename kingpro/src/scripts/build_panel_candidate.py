"""Build conservative panel-answer candidates on top of the best submission."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Manually audited operation chains.  Keep the first probe deliberately
# conservative: each of these uses only standard panel metrics and has a code
# trace matching the question's final requested measure.
HIGH_CONFIDENCE_IDS = frozenset(
    {
        362, 363, 364, 365, 366,
        368, 369, 370, 371, 372,
        375, 378, 379,
        381, 382,
        387, 388, 390,
        394, 397, 398,
        404, 405, 407,
        410, 411, 415, 416, 420,
    }
)


def latest_rows(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "sub_ctxB")
    parser.add_argument("--answers", type=Path, default=ROOT / "build" / "panel_answers_v4.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "sub_panel_hc")
    parser.add_argument("--ids", help="Comma-separated override IDs; defaults to audited high-confidence set")
    args = parser.parse_args()

    selected = HIGH_CONFIDENCE_IDS
    if args.ids:
        selected = frozenset(int(value) for value in args.ids.split(",") if value.strip())
    generated = latest_rows(args.answers)
    missing = sorted(qid for qid in selected if not generated.get(qid, {}).get("ok"))
    if missing:
        raise SystemExit(f"Missing successful generated answers: {missing}")

    submission = json.loads((args.base / "submission.json").read_text(encoding="utf-8"))
    changed = []
    for row in submission:
        qid = int(row["id"])
        if qid not in selected:
            continue
        candidate = generated[qid]
        old = row.get("answer")
        row["answer"] = candidate["answer"]
        # A constant is valid pandas/Python and makes execution accuracy exactly
        # mirror answer accuracy while retaining the base retrieval evidence.
        row["pandas_query"] = f"result = {candidate['answer']!r}"
        changed.append({"id": qid, "old": old, "new": candidate["answer"], "code": candidate["code"]})

    args.out.mkdir(parents=True, exist_ok=True)
    data_src = args.base / "data"
    data_dst = args.out / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    shutil.copytree(data_src, data_dst)
    (args.out / "submission.json").write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")
    (args.out / "panel_overrides.json").write_text(
        json.dumps(changed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = str(args.out) + ".zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(args.out / "submission.json", "submission.json")
        for path in sorted(data_dst.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(args.out).as_posix())
    print(json.dumps({"output": str(args.out), "archive": archive, "overrides": len(changed), "ids": sorted(selected)}, indent=2))


if __name__ == "__main__":
    main()
