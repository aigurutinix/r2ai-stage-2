"""Score a submission's rows by whether the label it read names what was asked.

There is no trustworthy gold here — the local set lies about retrieval and stores
raw cells in 28% of records — so a splice needs a criterion that needs no gold.
This is one: run every program under `sys.settrace`, record the row label of every
cell it reads, and measure token overlap between that label and the question. It
cannot prove an answer right; the organisers' generator deliberately avoids copying
row labels, so a correct answer can score low. But between two builds answering the
same question from two different rows, the row whose label shares more words with
the question is the better bet, and hand inspection of ten disagreements agreed
with that ordering in five of five where the labels differed sharply.

Writes one record per question so a splice can select on it.

Usage:
  PYTHONPATH=src python scripts/_label_agree.py --zip aimed.zip --out artifacts/agree_aimed.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifin.answering.sandbox import frame_from_rows  # noqa: E402

spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "trace_answer.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

STOP = {"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la", "trong",
        "va", "cho", "tai", "theo", "dong", "trieu", "ty", "nghin", "tram", "phan",
        "tram", "ctcp", "tong"}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def tokens(text: str) -> set[str]:
    return {t for t in fold(text).split() if len(t) > 2 and t not in STOP}


def agreement(question: str, label: str) -> float:
    """Share of the label's own words the question also uses.

    Coverage of the label, not of the question: the question carries a company
    name, a year and a unit that no row label will ever contain, so scoring
    against the question's tokens punishes every correct row equally.
    """

    left, right = tokens(question), tokens(label)
    if not right:
        return 0.0
    return len(left & right) / len(right)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = ROOT / "submissions" / args.zip
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))
        grids = {}
        for row in rows:
            for item in row.get("evidence") or []:
                name = item["csv_path"]
                if name not in grids:
                    grids[name] = tr.read_grid(archive, name)

    out = []
    for row in rows:
        frames, raw = {}, {}
        for item in row.get("evidence") or []:
            grid = grids.get(item["csv_path"])
            if grid is None:
                continue
            frames[item["variable"]] = frame_from_rows(grid)
            raw[item["variable"]] = grid
        reads = tr.logged_reads(row.get("pandas_query") or "", frames) if frames else []

        labels, best = [], 0.0
        for variable, r, c, _cell in reads:
            grid = raw.get(variable)
            if grid is None:
                continue
            index = r + 1 if r >= 0 else r
            if not (-len(grid) <= index < len(grid)):
                labels.append("<ngoai bang>")
                continue
            label = str(grid[index][0]) if grid[index] else ""
            labels.append(label)
            best = max(best, agreement(row["question"], label))

        out.append({
            "id": row["id"],
            "answer": row.get("answer"),
            "reads": len(reads),
            "agree": round(best, 3),
            "labels": labels[:4],
        })

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    traced = [r for r in out if r["reads"]]
    strong = sum(1 for r in traced if r["agree"] >= 0.5)
    print(f"{args.zip}: {len(traced)}/{len(out)} dong doc duoc o, "
          f"nhan khop >=0.5: {strong} ({100 * strong / max(1, len(traced)):.1f}%)  -> {args.out}")


if __name__ == "__main__":
    main()
