"""Execute the model's programs the way the organisers execute them, and keep what runs.

The reply is code, so it verifies itself twice over: it either runs or it does not, and
what it produces either is a finite number in a plausible range or it is not. Both gates
are free, and a program that fails either is dropped rather than shipped — the same
discipline the quote check gave the single-cell reader.

Execution uses the organisers' own sandbox from `vifinqa-official`, so `dfs` is keyed and
loaded exactly as it will be when the private round runs these programs: every cell a
raw string, no type inference, empty cells as "". A program that only works against a
differently-loaded frame fails here, which is the point.

The range gate is deliberately loose. It rejects what cannot be an answer — infinities,
NaN, and magnitudes past a quadrillion — and nothing else, because a tight gate would be
me substituting my expectations for the report's contents.

Usage:
  python scripts/fresh/run_programs.py --replies artifacts/fresh/replies_prog.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_gate import verdict  # noqa: E402
from num_helper import SOURCE as NUM_SOURCE  # noqa: E402

FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.M)
BANNED_RE = re.compile(r"\bimport\b|\bopen\s*\(|\b__\w+__\b|\bwhile\b|\beval\s*\(|"
                       r"\bexec\s*\(", re.I)
LIMIT = 1e15


def load_sandbox():
    path = (ROOT / "vifinqa-official" / "src" / "vifinqa" / "answering" /
            "sandbox.py")
    spec = importlib.util.spec_from_file_location("vifinqa_sandbox", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(reply: str) -> str:
    """The code, with reasoning, a markdown fence or a stray prose line removed."""

    # With thinking enabled the reply opens with a <think> block. It is reasoning, not
    # program text, and leaving it in makes every program a syntax error.
    if "</think>" in reply:
        reply = reply.rsplit("</think>", 1)[1]
    body = FENCE_RE.sub("", reply).strip()
    # Some replies open with a sentence before the code; drop leading lines until one
    # looks like Python.
    lines = body.splitlines()
    while lines and not re.match(r"^\s*(#|[A-Za-z_][\w.\[\]\"' ]*\s*=|def |if |for |"
                                 r"result\b|try:)", lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", required=True)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_prog.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--show", type=int, default=4)
    args = parser.parse_args()

    sandbox = load_sandbox()
    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    counters: Counter[str] = Counter()
    results, samples, failures = [], [], Counter()

    for line in (ROOT / args.replies).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        info = meta.get(record["id"])
        if info is None:
            continue
        code = clean(record["reply"])
        if not code:
            counters["tra loi rong"] += 1
            continue
        if BANNED_RE.search(code):
            counters["chuong trinh dung cu phap bi cam"] += 1
            continue
        # A program that never touches a frame is a constant with arithmetic around it.
        # It reads the figure off the rendered table, writes it into the code, and
        # throws away the one guarantee code gives over a picked cell.
        if "dfs[" not in code and not re.search(r"df\d?", code):
            counters["KHONG doc frame nao — la hang so"] += 1
            continue

        paths = {}
        for ref in info["refs"]:
            paths[ref["ref"]] = (
                ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                ref["doc"] / f"{ref['doc']}_extracted_tables"
                / f"table_{ref['table_id']}.csv")
        missing = [r for r, p in paths.items() if not p.exists()]
        for name in missing:
            paths.pop(name)
        if not paths:
            counters["khong con csv nao"] += 1
            continue

        try:
            value = sandbox.run_pandas_code(NUM_SOURCE + code, paths)
        except Exception as error:  # the sandbox wraps everything it raises
            counters["chay LOI"] += 1
            failures[str(error).split(":")[-1].strip()[:60]] += 1
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            counters["result khong phai so"] += 1
            continue
        if not math.isfinite(number) or abs(number) > LIMIT:
            counters["result vo ly (vo cuc/qua lon)"] += 1
            continue
        rejected = verdict(info["question"], number)
        if rejected is not None:
            counters[f"bi loai: {rejected}"] += 1
            continue

        counters["CHAY DUOC"] += 1
        results.append({
            "id": record["id"], "answer": round(number, 2),
            "pandas_query": code,
            "refs": [r["ref"] for r in info["refs"] if r["ref"] in paths],
            "docs": sorted({r["doc"] for r in info["refs"] if r["ref"] in paths}),
            "csvs": {r: str(p.relative_to(ROOT)).replace("\\", "/")
                     for r, p in paths.items()},
        })
        if len(samples) < args.show:
            samples.append(f"  id={record['id']} -> {round(number, 2)}\n"
                           f"     {info['question'][:96]}\n"
                           + "\n".join("     " + l for l in code.splitlines()[:8]))

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    if failures:
        print("\n  loi thuong gap:")
        for name, count in failures.most_common(6):
            print(f"    {count:4d}  {name}")
    print(f"\nchuong trinh chay ra so: {len(results)}")
    for text in samples:
        print(text)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
