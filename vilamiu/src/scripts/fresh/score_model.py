"""Judge the model's picks with the instruments already in place, before any submission.

Three things are measurable here without gold:

  validity      the code the model names has to exist in the block it was shown. This
                is the property the earlier cell-picking attempt could not offer: a
                hallucinated coordinate was indistinguishable from a real one, while a
                code outside the list is rejected on the spot.
  agreement     on the questions the rule-based matcher answers confidently — a set
                that is 52% correct — agreement is a floor. If the model tracks it,
                the model is at least as good; if it diverges, the divergence is where
                the answer lies either way.
  reach         on the questions the matcher refuses, anything the model picks is new
                coverage, because the alternative there is a blank.

What this cannot say is which of the two is right where they differ. That needs the
leaderboard, and the point of measuring first is to spend that slot on a build worth
submitting.

Usage:
  python scripts/fresh/score_model.py --replies artifacts/fresh/replies_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_block import Corpus  # noqa: E402

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
TM_RE = re.compile(r"^\s*(?:TM)?\s*(\d+)\s*$", re.I)


def parse(reply: str) -> dict | None:
    """The outermost JSON object in the reply.

    A regex of `\\{[^{}]*\\}` cannot express this: excluding braces from the body means
    it only ever matches the INNERMOST object. The rate pass returns
    `{"tu": {...}, "mau": {...}}`, so that regex found `{"nguon": …, "ma": …}` and
    every one of 91 replies came back missing its operands. Braces are counted here
    instead.
    """

    text = THINK_RE.sub("", reply or "")
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", required=True)
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_probe.jsonl")
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--plan", default="artifacts/fresh/answer_plan.jsonl")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)

    meta = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            meta[record["id"]] = record["meta"]

    plan = {}
    plan_path = ROOT / args.plan
    if plan_path.exists():
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                plan[record["id"]] = record

    replies = {}
    for line in (ROOT / args.replies).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            replies[record["id"]] = record["reply"]
    print(f"{len(replies)} tra loi tren {len(meta)} prompt")

    counters: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    agree = disagree = 0
    samples = []
    for qid, reply in sorted(replies.items()):
        info = meta.get(qid)
        if info is None:
            continue
        spec = parse(reply)
        if not spec:
            counters["khong doc duoc JSON"] += 1
            continue
        source = str(spec.get("nguon", "")).strip()
        code = str(spec.get("ma", "")).strip()
        period = str(spec.get("ky", "")).strip()
        if source not in ("cdkt", "kqkd", "lctt", "TM") or not code:
            counters["nguon hoac ma khong hop le"] += 1
            continue
        if period not in ("current", "prior"):
            counters["ky khong hop le"] += 1
            continue
        sources[source] += 1

        key = (info["ticker"], info["year"], info["scope"])
        if source == "TM":
            match = TM_RE.match(code)
            table_ids = {n["table_id"] for n in corpus.notes.get(key, ())}
            if not match or int(match.group(1)) not in table_ids:
                counters["so TM khong co trong muc luc"] += 1
                continue
            counters["HOP LE (thuyet minh)"] += 1
        else:
            if (source, code) not in corpus.rows.get(key, {}):
                counters["ma khong co trong khoi"] += 1
                continue
            counters["HOP LE (bao cao chinh)"] += 1

        entry = plan.get(qid)
        if entry is None:
            counters["  bo luat tu choi — do phu MOI"] += 1
            continue
        same = (source == entry["kind"] and code == entry["code"]
                and period == entry["period"])
        if same:
            agree += 1
        else:
            disagree += 1
            if len(samples) < args.show:
                samples.append(
                    f"  id={qid:<5d} bo luat {entry['kind']}/{entry['code']}"
                    f" {entry['period']}  |  model {source}/{code} {period}\n"
                    f"     hoi : {info['question'][:96]}\n"
                    f"     nhan bo luat: {str(entry['label'])[:70]}")

    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\nnguon model chon: {dict(sources)}")
    total = agree + disagree
    if total:
        print(f"\ntren {total} cau bo luat tu tin (bo luat dung ~52%):")
        print(f"  model DONG Y   : {agree} ({100 * agree / total:.0f}%)")
        print(f"  model KHAC     : {disagree} ({100 * disagree / total:.0f}%)")
    print("\nvi du khac nhau:")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
