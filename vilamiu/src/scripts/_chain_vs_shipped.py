"""Does the harness chain reproduce the shipped deterministic answers?

The gold set says the chain scores better with a wider candidate list (14.4% at
search_k=8 -> 17.8% at 40). Acting on that means overwriting shipped answers, and
that is only safe if the chain *is* the thing that produced them.

It may well not be. `run_submit.py` lost a whole flag layer on 10/08 —
`shape_lock`, `label_rescan`, `label_declare`, `consensus`, `credible_gate` and the
entire `ma_so` path — and the harness chain also omits the `located`, `panel`,
`generated` and `planned` branches. So the shipped answer for a question may come
from a richer pipeline than the one measured. Swapping it for a poorer pipeline at a
higher k would trade a real answer for a worse one while the gold numbers say the
opposite, which is exactly the failure mode the project's notes call "a new
mechanism overwritten by an old one" — in reverse.

Agreement at the *same* k is the control. High agreement means the chain is the
pipeline and the k=40 swap inherits the gold measurement. Low agreement means it is
a different beast and the measurement does not transfer.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_gold import shipped_chain  # noqa: E402

from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

TOL = 1e-2


def load_codes(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("code"):
                out[row["id"]] = row["code"].strip()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "noconst.zip"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with zipfile.ZipFile(ROOT / "submissions" / base) as zf:
        shipped = {r["id"]: r for r in json.loads(zf.read("submission.json"))}

    # Which branch produced each shipped answer, by matching the program text
    # against the caches those branches read from.
    program = load_codes(ROOT / "artifacts" / "gen14b_merged.jsonl")
    planned = load_codes(ROOT / "artifacts" / "planned.jsonl")
    located = load_codes(ROOT / "artifacts" / "located.jsonl")

    def branch(qid: int) -> str:
        code = (shipped[qid].get("pandas_query") or "").strip()
        if program.get(qid) == code:
            return "14B program"
        if planned.get(qid) == code:
            return "plan"
        if located.get(qid) == code:
            return "locate"
        return "deterministic"

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    corroborator = Corroborator(store)
    questions = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    if limit:
        questions = questions[:limit]

    branches: Counter = Counter()
    agree: Counter = Counter()
    changed_at_40 = 0
    deterministic = 0

    for question in questions:
        record = shipped.get(question.id)
        if record is None:
            continue
        kind = branch(question.id)
        branches[kind] += 1
        if kind != "deterministic":
            continue
        deterministic += 1
        ship = float(record.get("answer") or 0.0)
        value10, _ = shipped_chain(
            question, store, retriever, corroborator, search_k=10, magnitude=True
        )
        if value10 is not None and abs(value10 - ship) <= TOL:
            agree["chain@10 == shipped"] += 1
        else:
            agree["chain@10 != shipped"] += 1
        value40, _ = shipped_chain(
            question, store, retriever, corroborator, search_k=40, magnitude=True
        )
        if value40 is not None and value10 is not None and abs(value40 - value10) > TOL:
            changed_at_40 += 1

    print(f"base={base}  questions={sum(branches.values())}")
    for name, count in branches.most_common():
        print(f"  {count:4d}  {name}")
    print(f"\ndeterministic subset = {deterministic}")
    for name, count in agree.most_common():
        share = count / max(1, deterministic)
        print(f"  {count:4d}  {name}   ({share:.1%})")
    print(f"\n  {changed_at_40} answers move when the chain widens 10 -> 40 "
          f"({changed_at_40 / max(1, deterministic):.1%} of the subset)")


if __name__ == "__main__":
    main()
