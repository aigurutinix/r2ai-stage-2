"""Build picker training pairs from execution-verified generator gold.

The picker answers one question only: given a question and a rendered table, WHICH ROW
holds the indicator? That is the skill behind p_pick — the one number that decides
whether Stage C of the pipeline design can exist — and this file turns the organisers'
own gold into supervision for it, with no model in the loop:

  1. execute-shape extraction: each easy-tier gold program is AST-parsed into
     (row-index resolution, column) on the exact frame the grader builds;
  2. verification: the value at the extracted cell must reproduce `answer` under the
     shared `_num` parser (both Vietnamese and English decimal readings accepted) —
     records that fail are counted loudly, never silently kept;
  3. distractors: confusable rows mined from the same table by the same match-specificity
     rule the retriever uses, so negatives are the mistakes the model would actually make
     (131 khách-hàng vs 136 khác), not random rows.

Output: artifacts/fresh/picker_train.jsonl + picker_heldout.jsonl (every 7th record).
Held-out doubles as the p_pick measurement — no separate probe needed.

Usage:
  python scripts/fresh/build_picker_pairs.py            # build + stats
  python scripts/fresh/build_picker_pairs.py --peek 3   # show sample pairs
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_match import fold, score as match_score  # noqa: E402
from num_helper import SOURCE as NUM_SRC  # noqa: E402
from render_v2 import corpus_csv, grader_frame  # noqa: E402

NUM_NS: dict = {}
exec(compile(NUM_SRC, "<num>", "exec"), NUM_NS)
_num = NUM_NS["_num"]

REMOTE_PREFIX = "/workspace/vifin/data/official_corpus/"
LOCAL_PREFIX = "data/official_corpus/"


def read_frame(rec: dict) -> pd.DataFrame | None:
    path = rec["csv_path"].replace(REMOTE_PREFIX, LOCAL_PREFIX)
    path = ROOT / path
    if not path.exists():
        return None
    return grader_frame(path)


def _strip_result_wrappers(node: ast.expr) -> ast.expr:
    """Peel .values[0] / [0] / .iloc[0] / .head(1) / .tolist()[0] off the tail."""
    changed = True
    while changed:
        changed = False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("head", "first") and node.args:
                node, changed = node.args[0], True
                continue
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value == 0:
            node, changed = node.value, True
            continue
        if isinstance(node, ast.Attribute) and node.attr == "values":
            node, changed = node.value, True
            continue
    return node


def _col_index(frame: pd.DataFrame, key_expr: ast.expr,
               iloc_cols: bool) -> int | None:
    """Column key inside .loc/.getitem: 'name', 3, or df.iloc[:, 3]."""
    if isinstance(key_expr, ast.Constant) and isinstance(key_expr.value, str):
        name = key_expr.value
        try:
            return int(frame.columns.get_loc(name))
        except KeyError:
            pass
        folded = {fold(str(c)): i for i, c in enumerate(frame.columns)}
        matches = {i for c, i in folded.items() if c == fold(name)}
        # a normalised name is only trusted when it is unambiguous — a collision
        # means two real columns differ by whitespace, and picking one is a guess
        return matches.pop() if len(matches) == 1 else None
    if isinstance(key_expr, ast.Constant) and isinstance(key_expr.value, int):
        return key_expr.value
    if iloc_cols and isinstance(key_expr, ast.Tuple) and len(key_expr.elts) == 2 \
            and isinstance(key_expr.elts[1], ast.Constant) \
            and isinstance(key_expr.elts[1].value, int) \
            and (isinstance(key_expr.elts[0], ast.Slice)
                 or (isinstance(key_expr.elts[0], ast.Constant)
                     and key_expr.elts[0].value is None)):
        return key_expr.elts[1].value
    return None


def _mask_hits(frame: pd.DataFrame, node: ast.expr) -> list[int] | None:
    """Row indices selected by a boolean mask expression, or None if unrecognised."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        left = _mask_hits(frame, node.left)
        right = _mask_hits(frame, node.right)
        if left is None or right is None:
            return None
        return [r for r in left if r in set(right)]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "isna"):
        col = _col_index(frame, node.func.value, iloc_cols=True)
        if col is None:
            return None
        return frame.index[frame.iloc[:, col].astype(str).str.strip() == ""]
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.Is))):
        return None
    left, right = node.left, node.comparators[0]
    lit = right.value if isinstance(right, ast.Constant) else None
    if lit is None:
        return None
    if isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name):
        col = _col_index(frame, left.slice, iloc_cols=False)
    elif (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Attribute)
            and left.value.attr == "iloc"):
        col = _col_index(frame, left.slice, iloc_cols=True)
    else:
        return None
    if col is None:
        return None
    series_raw = frame.iloc[:, col].astype(str)
    hits = series_raw[series_raw == str(lit)].index.tolist()
    if not hits:  # the exec engine compares raw; stripping is a last resort
        series = series_raw.str.strip()
        literal = str(lit)
        hits = series[series == literal].index.tolist()
        if not hits:
            alt = literal.rstrip(".0") if "." in literal else literal
            hits = series[series == alt].index.tolist()
    return hits


def _const_int(node: ast.expr) -> int | None:
    """A signed integer literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)):
        return -node.operand.value
    return None


def extract_cell(frame: pd.DataFrame, program: str) -> tuple[int, int] | None:
    """The (iloc_row, iloc_col) an easy-tier gold program reads, or None."""
    tree = ast.parse(program.strip())
    assign = next((n for n in tree.body if isinstance(n, ast.Assign)), None)
    if assign is None or len(assign.targets) != 1 \
            or getattr(assign.targets[0], "id", "") != "result":
        return None
    core = _strip_result_wrappers(assign.value)
    n = len(frame)

    # result = df.iloc[r, c]
    if isinstance(core, ast.Subscript) and isinstance(core.value, ast.Attribute) \
            and core.value.attr == "iloc" and isinstance(core.slice, ast.Tuple) \
            and len(core.slice.elts) == 2:
        r = _const_int(core.slice.elts[0])
        c = _const_int(core.slice.elts[1])
        if r is not None and c is not None:
            rr = r + n if r < 0 else r
            cc = c + frame.shape[1] if c < 0 else c
            return (rr, cc) if 0 <= rr < n and 0 <= cc < frame.shape[1] else None

    def resolve(base: ast.expr) -> tuple[int, int] | None:
        hits: list[int] | None = None
        colkey: ast.expr | None = None

        if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Attribute) \
                and base.value.attr == "loc":
            parts = base.slice.elts if isinstance(base.slice, ast.Tuple) \
                else [base.slice]
            if len(parts) != 2:
                return None
            first, colkey = parts
            r = _const_int(first)
            if r is not None:
                rr = r + n if r < 0 else r
            elif (isinstance(first, ast.Subscript)
                    and isinstance(first.value, ast.Attribute)
                    and first.value.attr == "index"):
                idx = _const_int(first.slice)
                rr = n - 1 if idx == -1 else None
            else:
                rr = None
            if rr is not None:
                ci = _col_index(frame, colkey, iloc_cols=True)
                return (rr, ci) if ci is not None and 0 <= rr < n else None
            hits = _mask_hits(frame, first)
        elif (isinstance(base, ast.Subscript)
                and isinstance(base.value, ast.Subscript)
                and isinstance(base.value.value, ast.Name)):  # df[mask][col]
            colkey = base.slice
            hits = _mask_hits(frame, base.value.slice)
        elif (isinstance(base, ast.Subscript)
                and isinstance(base.value, ast.Attribute)
                and base.value.attr == "iloc"
                and isinstance(base.value.value, ast.Subscript)):
            # df[df.iloc[:,0] == v].iloc[:, c]
            inner = base.value.value
            if not (isinstance(inner.value, ast.Name)
                    and isinstance(base.slice, ast.Tuple)
                    and len(base.slice.elts) == 2):
                return None
            colkey = base.slice.elts[1]
            hits = _mask_hits(frame, inner.slice)
        else:
            return None

        if hits is None or colkey is None:
            return None
        ci = _col_index(frame, colkey, iloc_cols=True)
        if ci is None:
            cc = _col_index(frame, colkey, iloc_cols=False)
            ci = cc
        return (hits[0], ci) if hits and ci is not None else None

    got = resolve(core)
    if got is not None:
        return got

    # result = df.iloc[r]['col']
    if isinstance(core, ast.Subscript) and isinstance(core.value, ast.Subscript) \
            and isinstance(core.value.value, ast.Attribute) \
            and core.value.value.attr == "iloc":
        r = _const_int(core.value.slice)
        ci = _col_index(frame, core.slice, iloc_cols=False)
        if r is not None and ci is not None:
            rr = r + n if r < 0 else r
            return (rr, ci) if 0 <= rr < n else None
    return None


def parse_answer(raw) -> list[float]:
    out = []
    if isinstance(raw, (int, float)):
        out.append(float(raw))
    text = str(raw).strip()
    for parser in (_num, lambda t: float(t.replace(".", "").replace(",", ".")
                                         ) if re_dot(t) else None):
        try:
            v = parser(text)
            if v is not None:
                out.append(float(v))
        except (ValueError, TypeError):
            pass
    return out


def re_dot(text: str) -> bool:
    import re as _re
    return bool(_re.search(r"\d", text))


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, 1e-6 * max(abs(a), abs(b)))


def verified_match(cell_value: str, answer) -> bool:
    xs = [_num(cell_value)]
    ys = parse_answer(answer)
    return any(close(x, y) for x in xs for y in ys)


def row_label(frame: pd.DataFrame, r: int, code_col: int | None) -> str:
    """First usable label cell: text, not a bare number or a dash, scanning wider
    than column 0 because OCR sometimes puts the label one cell over."""
    for c in range(min(5, frame.shape[1])):
        if c == code_col:
            continue
        v = " ".join(str(frame.iat[r, c]).split())
        if not v or v in {"-", "–", "—"}:
            continue
        if v.replace(".", "").replace(",", "").replace(" ", "").isdigit():
            continue
        return v
    return "(dòng không nhãn)"


def mine_distractors(frame: pd.DataFrame, gold_r: int, question: str,
                     code_col: int | None, k: int = 4) -> list[tuple[int, float]]:
    out = []
    qf = fold(question)
    for r in range(len(frame)):
        if r == gold_r:
            continue
        label = row_label(frame, r, code_col)
        s = match_score([question], label)
        if s <= 0:
            flat = set(qf.split())
            lab = set(fold(label).split())
            s = min(len(flat & lab), 4) if flat & lab else 0.0
        if s > 0:
            out.append((r, s))
    out.sort(key=lambda t: (-t[1], abs(t[0] - gold_r)))
    return out[:k]


def build(limit: int = 0) -> list[dict]:
    records = [json.loads(line) for line
               in (ROOT / "artifacts" / "easy_416.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    if limit:
        records = records[:limit]

    pairs: list[dict] = []
    stats = {"no_csv": 0, "shape_unknown": 0, "mask_no_hit": 0,
             "unverified": 0, "ok": 0, "verified": 0}
    shapes: dict[str, int] = {}

    for rec in records:
        frame = read_frame(rec)
        if frame is None:
            stats["no_csv"] += 1
            continue
        cell = extract_cell(frame, rec.get("pandas_query") or "")
        if cell is None:
            stats["shape_unknown"] += 1
            continue
        r, c = cell
        value = str(frame.iat[r, c])
        ok = verified_match(value, rec["answer"])
        stats["ok" if ok else "unverified"] += 1
        stats["verified"] += int(ok)

        code_col = None
        for i, name in enumerate(frame.columns[:4]):
            if "mã số" in str(name).lower():
                code_col = i
                break
        dis = mine_distractors(frame, r, rec["question"], code_col)
        choices_rows = sorted({r} | {d[0] for d in dis})
        choices = [{"r": rr, "label": row_label(frame, rr, code_col)[:120]}
                   for rr in choices_rows]
        pairs.append({
            "id": rec["id"],
            "question": rec["question"],
            "table_ref": rec["relevant_tables"][0],
            "unit_hint": "",
            "choices": choices,
            "answer_index": choices_rows.index(r),
            "gold_r": r, "gold_c": c,
            "gold_value": value[:60],
            "gold_answer": rec["answer"],
            "verified": ok,
            "n_candidates_table": len(frame),
        })
    print(f"records           : {len(records)}")
    print(f"  no local CSV    : {stats['no_csv']}")
    print(f"  shape unknown   : {stats['shape_unknown']}")
    print(f"  extracted       : {stats['ok'] + stats['unverified']}")
    print(f"  VERIFIED vs gold: {stats['verified']} "
          f"({stats['verified'] / max(1, stats['ok'] + stats['unverified']):.1%})")
    with_dis = sum(1 for p in pairs if len(p['choices']) > 1)
    print(f"  ≥1 distractor   : {with_dis} ({with_dis / max(1, len(pairs)):.1%})")
    avg = sum(len(p['choices']) for p in pairs) / max(1, len(pairs))
    print(f"  avg choices     : {avg:.2f}")
    return pairs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--peek", type=int, default=0)
    args = parser.parse_args()

    pairs = build(args.limit)
    train = [p for i, p in enumerate(sorted(pairs, key=lambda p: p["id"]))
             if i % 7 != 6]
    held = [p for i, p in enumerate(sorted(pairs, key=lambda p: p["id"]))
            if i % 7 == 6]
    out_train = ROOT / "artifacts/fresh/picker_train.jsonl"
    out_held = ROOT / "artifacts/fresh/picker_heldout.jsonl"
    out_train.write_text("\n".join(json.dumps(p, ensure_ascii=False)
                                   for p in train), encoding="utf-8")
    out_held.write_text("\n".join(json.dumps(p, ensure_ascii=False)
                                  for p in held), encoding="utf-8")
    print(f"\ntrain {len(train)} -> {out_train.relative_to(ROOT)}")
    print(f"heldout {len(held)} -> {out_held.relative_to(ROOT)}")

    for p in (pairs if args.peek else [])[:args.peek]:
        print(f"\n--- id={p['id']} verified={p['verified']} "
              f"gold=r{p['gold_r']},c{p['gold_c']} value={p['gold_value']}")
        print(f"Q: {p['question'][:110]}")
        for j, ch in enumerate(p["choices"]):
            mark = " <= GOLD" if j == p["answer_index"] else ""
            print(f"   [{j}] r{ch['r']:<3d} {ch['label'][:90]}{mark}")


if __name__ == "__main__":
    main()
