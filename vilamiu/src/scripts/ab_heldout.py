"""Grade a model on the held-out SFT slice by *executing* its program.

The SFT targets are always literal `num(dfK, r, c)` (+ optional unit scale). The
base model usually writes a `find_row` program that reaches the same cell. A
tuple match would score that base program zero and make any LoRA that merely
copied the target format look like a win. Running both programs against the
tables embedded in the prompt puts the arms on the same scale: abs_tol 0.01,
the BTC rule.

Cell-tuple match is still reported when the output happens to be a literal
`num` call — it answers "did the model learn the SFT dialect?", not "is it
right".
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.generate import extract_code  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402

NUM_RE = re.compile(r"num\(\s*df(\d*)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
SCALE_RE = re.compile(r"result\s*=\s*round\(\s*result\s*\*\s*([-\d.eE+]+)\s*,\s*\d+\s*\)")
TABLE_RE = re.compile(
    r'<table variable="(df\d*)">\s*(.*?)\s*</table>',
    re.S,
)
ABS_TOL = 1e-2


def parse_target(code: str) -> tuple[str, int, int, float] | None:
    hit = NUM_RE.search(code)
    if hit is None:
        return None
    scale_hit = SCALE_RE.search(code)
    try:
        scale = float(scale_hit.group(1)) if scale_hit else 1.0
    except ValueError:
        scale = 1.0
    return (hit.group(1) or "", int(hit.group(2)), int(hit.group(3)), scale)


def tables_from_user(user: str) -> dict[str, list[list[str]]]:
    """Rebuild the frames the program is supposed to read from the prompt CSV.

    The SFT renderer joins cells with bare commas and then truncates the block
    (`max_chars=7000`), so rows can be ragged. Pad to a common width; a short
    last row is what truncation looks like, and `num`/`find_row` only touch
    cells that fit.
    """

    tables: dict[str, list[list[str]]] = {}
    for match in TABLE_RE.finditer(user):
        name, body = match.group(1), match.group(2).strip()
        rows = [list(row) for row in csv.reader(io.StringIO(body))]
        if not rows:
            continue
        width = max(len(row) for row in rows)
        tables[name] = [row + [""] * (width - len(row)) for row in rows]
    return tables


def with_prelude(code: str) -> str:
    if not code.strip():
        return code
    if "def num(" in code:
        return code
    return PRELUDE + "\n" + code


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=0.0, abs_tol=ABS_TOL)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:18000/v1")
    parser.add_argument("--data", default=str(ROOT / "artifacts" / "sft_easy.jsonl"))
    parser.add_argument("--limit", type=int, default=57)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    client = ChatClient.local(args.model, base_url=args.base_url, max_tokens=1200)

    records: list[dict] = []
    runnable = correct = cell_ok = scale_ok = 0

    for i, row in enumerate(rows, 1):
        system = next(m["content"] for m in row["messages"] if m["role"] == "system")
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        gold_code = row["messages"][-1]["content"]
        tables = tables_from_user(user)
        gold_run = run_query(with_prelude(gold_code), tables)
        gold_cell = parse_target(gold_code)

        try:
            code = extract_code(client.complete(system, user))
            error = ""
        except Exception as exc:  # keep the rest of the batch if one call dies
            code, error = "", f"{type(exc).__name__}: {exc}"

        pred_run = run_query(with_prelude(code), tables) if code else None
        got_cell = parse_target(code)

        ok = bool(
            pred_run and pred_run.ok and gold_run.ok
            and close(pred_run.value, gold_run.value)
        )
        if pred_run and pred_run.ok:
            runnable += 1
        if ok:
            correct += 1
        if gold_cell and got_cell and got_cell[:3] == gold_cell[:3]:
            cell_ok += 1
        if gold_cell and got_cell and got_cell[3] == gold_cell[3]:
            scale_ok += 1

        records.append({
            "index": i - 1,
            "gold_value": gold_run.value if gold_run.ok else None,
            "gold_error": gold_run.error,
            "pred_value": pred_run.value if pred_run and pred_run.ok else None,
            "pred_error": (pred_run.error if pred_run else "") or error,
            "correct": ok,
            "gold_cell": gold_cell,
            "got_cell": got_cell,
            "code": code,
        })
        mark = "OK" if ok else "MISS"
        print(
            f"  {i:>3}/{len(rows)}  {mark}  gold={gold_run.value}  "
            f"pred={pred_run.value if pred_run and pred_run.ok else pred_run.error if pred_run else error}"
            f"{'  ' + error if error else ''}"
        )

    n = len(rows)
    print(f"\n{args.model} on {n} held-out pairs (abs_tol={ABS_TOL})")
    print(f"  runnable : {runnable:>3}/{n}  {runnable / n:.1%}")
    print(f"  correct  : {correct:>3}/{n}  {correct / n:.1%}   ← the A/B number")
    print(f"  cell-match (SFT dialect): {cell_ok:>3}/{n}  {cell_ok / n:.1%}")
    print(f"  scale-match             : {scale_ok:>3}/{n}  {scale_ok / n:.1%}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
