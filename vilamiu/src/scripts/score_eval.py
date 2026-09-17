"""Execute the generated programs and score them against the known answers.

This is the measurement the training metrics cannot give. `eval_token_accuracy`
was 0.9549 on ~28 supervised tokens, but only **16.8% of the target's characters
carry information** — the frame name, the row, the column, the scale. The other
83% is boilerplate identical across all 380 pairs. Reading a cell needs *all three*
of frame, row and column right, so ~1.26 wrong tokens per example is consistent
with anywhere between roughly 30% and 90% of cells being correct. Token accuracy
cannot separate those; running the program can.

Three arms, because two would not settle it:

  * **base** — the same weights with the adapter switched off, which isolates what
    SFT added rather than what the prompt already achieved;
  * **tuned** — the adapter;
  * **matcher** — the lexical label branch, pinned at 42.8% on the leaderboard. It
    is the bar a replacement has to clear to be worth shipping.

Usage:  PYTHONPATH=src python scripts/score_eval.py artifacts/evalgen.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

GEN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "evalgen.jsonl"
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S)
TOL = 2e-4


def extract(text: str) -> str:
    """Strip reasoning and fences; keep the program whole.

    The first version kept only lines containing `result` or starting with
    indentation, on the assumption the target was always `result = num(...)`.
    The model writes label-based lookups instead:

        row = find_row(df1, "Cộng")
        if row < 0:
            row = find_row(df1, "Tổng")
        result = num(df1, row, 1)

    The first two lines match neither condition, so the filter deleted the half
    that defines `row` and every program "crashed" — 57 of 57, scored as the
    model's failure when it was the scorer's. A filter that decides what code
    looks like will delete code that looks different.
    """

    body = re.sub(r"<think>.*?</think>", " ", text, flags=re.S).strip()
    fenced = FENCE_RE.findall(body)
    if fenced:
        body = fenced[-1].strip()
    # Drop only prose: a line with no Python operator and no call parenthesis.
    lines = [
        ln for ln in body.splitlines()
        if not ln.strip() or re.search(r"[=()\[\]:]|^\s*(if|else|elif|for|try|except|raise|pass|break|continue)\b", ln)
    ]
    return "\n".join(lines).strip() or body


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return (scale > 0 and abs(a - b) / scale <= TOL) or abs(a - b) <= 0.01


CELL_RE = re.compile(r"num\(\s*(df\d*)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def cell_of(body: str) -> tuple[str, int, int] | None:
    match = CELL_RE.search(body)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def fault(body: str, target: str) -> str:
    """Which of frame, row or column the program got wrong.

    "Worse than the matcher" does not say what to change. This does. The target
    format makes the model *count rows* to produce an index, which is positional
    arithmetic — the weakest thing a transformer does, and the reason asking for
    `{table, row, column}` measured 13.3% while handing it `find_row()` lifted
    executable programs from 52.4% to 76.9%.

    So if the faults pile up on `row`, the fix is to emit
    `row = find_row(df, "<label>")` and turn counting into copying a string. If
    they pile up on `frame`, the problem is table selection and no change to the
    target format will help.
    """

    got, want = cell_of(body), cell_of(target)
    if want is None:
        return "target_not_a_cell_read"
    if got is None:
        return "no_cell_read_emitted"
    if got[0] != want[0]:
        return "wrong_frame"
    if got[1] != want[1]:
        return "wrong_row"
    if got[2] != want[2]:
        return "wrong_column"
    return "cell_exact"


def main() -> None:
    rows = [
        json.loads(line)
        for line in GEN.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    tally: dict[str, Counter[str]] = {a: Counter() for a in ("base", "tuned", "matcher")}
    faults: dict[str, Counter[str]] = {a: Counter() for a in ("base", "tuned")}

    for row in rows:
        meta = row["meta"]
        want = float(meta["answer"])
        keys = [TableKey(d, int(t)) for d, t in meta["keys"]]
        names = list(meta["variables"])
        grids = {n: store.rows(k) for n, k in zip(names, keys)}
        if len(keys) == 1:
            grids["df"] = grids[names[0]]

        for arm in ("base", "tuned"):
            body = extract(row[arm])
            faults[arm][fault(body, str(meta["target"]))] += 1
            if not body:
                tally[arm]["empty"] += 1
                continue
            outcome = run_query(f"{PRELUDE}\n{body}", grids)
            if not outcome.ok or outcome.value is None:
                tally[arm]["crash"] += 1
                continue
            tally[arm]["runs"] += 1
            if close(float(outcome.value), want):
                tally[arm]["CORRECT"] += 1
            tally[arm]["exact_target"] += int(body.strip() == str(meta["target"]).strip())

        # The 42.8% branch on the same question, given the same gold table.
        question = parse_question(0, meta["question"], roster)
        gold_grid = grids[meta["gold_frame"]]
        found = lookup_mod.find(gold_grid, question)
        if found is None or found.score < lookup_mod.MIN_LABEL_SCORE:
            tally["matcher"]["declined"] += 1
            continue
        meta_row = store.meta(keys[names.index(meta["gold_frame"])])
        scale_in = lookup_mod.column_scale(
            gold_grid, found.column,
            f"{meta_row.unit_page} {meta_row.unit_doc} {meta_row.caption}")
        asked = UNIT_SCALE.get(question.target_unit) or 1.0
        got = abs(found.value) * scale_in / asked
        tally["matcher"]["runs"] += 1
        if close(round(got, 2), want):
            tally["matcher"]["CORRECT"] += 1

    n = len(rows) or 1
    print(f"{GEN.name}: {len(rows)} held-out questions\n")
    print(f"  {'arm':10s} {'correct':>8} {'rate':>7} {'runs':>6} {'crash':>6} "
          f"{'exact':>6}")
    for arm in ("base", "tuned", "matcher"):
        t = tally[arm]
        print(f"  {arm:10s} {t['CORRECT']:8d} {t['CORRECT'] / n:6.1%} "
              f"{t['runs']:6d} {t['crash'] + t['empty']:6d} {t['exact_target']:6d}")
    print(f"\n  matcher declined: {tally['matcher']['declined']}"
          f"  (it answers only where a label matches at >= "
          f"{lookup_mod.MIN_LABEL_SCORE})")

    print("\n  where the cell reference goes wrong:")
    order = ["cell_exact", "wrong_row", "wrong_column", "wrong_frame",
             "no_cell_read_emitted", "target_not_a_cell_read"]
    print(f"    {'fault':24s} {'base':>6} {'tuned':>7}")
    for key in order:
        if faults["base"][key] or faults["tuned"][key]:
            print(f"    {key:24s} {faults['base'][key]:6d} {faults['tuned'][key]:7d}")
    print("\n    Faults on `wrong_row` mean the target format is the problem: the")
    print("    model has to count rows to emit an index. Re-emitting targets as")
    print("    `row = find_row(df, \"<label>\")` turns that into copying a string,")
    print("    which is the change our own numbers argue for — asking for indices")
    print("    measured 13.3%, handing over `find_row` lifted runnable to 76.9%.")
    print("    Faults on `wrong_frame` mean table selection, which no change to")
    print("    the target format can fix.")
    print("\n  The bar is the matcher's 42.8% on the leaderboard. `tuned` above")
    print("  `base` says SFT added something; `tuned` above the matcher says it is")
    print("  worth shipping. `exact` counts byte-identical targets — high exact")
    print("  with low correct would mean it memorised the training shape.")


if __name__ == "__main__":
    main()
