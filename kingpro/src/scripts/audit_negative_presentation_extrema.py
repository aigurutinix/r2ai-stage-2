"""Find extrema over accounting values presented as parenthesized negatives.

Expense, provision, loss and adverse-impact rows are often printed in
parentheses.  A program that asks for the highest/lowest *magnitude* but feeds
those signed values directly to ``max``/``min`` reverses the intended order.

The audit maps scalar ``vN`` assignments back to compact source-manifest rows
and inspects only variables that actually reach an extrema call.  Findings are
review signals; wording such as reversal/income may legitimately require the
reported sign.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAR_RE = re.compile(r"v\d+")
EXTREMA_WORDS = (
    "cao nhat", "thap nhat", "lon nhat", "nho nhat", "nhieu nhat", "it nhat"
)
MAGNITUDE_WORDS = (
    "chi phi", "du phong", " lo ", "muc giam", "anh huong", "thiet hai",
)
MAGNITUDE_METRIC_WORDS = (
    "expense", "provision", "loss", "impact", "damage", "decline",
)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _uses_abs(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "abs"
        for item in ast.walk(node)
    )


def _uses_abs_at_call(call: ast.Call, variable: str) -> bool:
    """Return true when every use of ``variable`` in ``call`` is under abs()."""
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(call):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    uses = [
        node
        for node in ast.walk(call)
        if isinstance(node, ast.Name) and node.id == variable
    ]
    if not uses:
        return False
    for use in uses:
        node: ast.AST = use
        protected = False
        while node in parents:
            node = parents[node]
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "abs"
            ):
                protected = True
                break
        if not protected:
            return False
    return True


def _iloc_indices(node: ast.AST) -> set[int]:
    result: set[int] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Subscript):
            continue
        value = item.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "iloc"
            and isinstance(value.value, ast.Name)
        ):
            continue
        slice_node = item.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
            result.add(slice_node.value)
    return result


def _assignments(tree: ast.AST) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not VAR_RE.fullmatch(target.id):
            continue
        result[target.id] = {
            "uses_abs": _uses_abs(node.value),
            "indices": sorted(_iloc_indices(node.value)),
            "expression": ast.unparse(node.value),
        }
    return result


def _manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(submission: Path) -> dict:
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    checked_calls = 0
    for row in rows:
        question = str(row.get("question", ""))
        normalized = fold(question)
        if not any(word in normalized for word in EXTREMA_WORDS):
            continue
        if not any(word in normalized for word in MAGNITUDE_WORDS):
            continue
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        assignments = _assignments(tree)
        if not assignments:
            continue
        qid = int(row["id"])
        manifest = _manifest(submission / "data" / f"q{qid}_source_cells.csv")
        if not manifest:
            continue
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in {"max", "min"}
            ):
                continue
            names = sorted(
                {
                    item.id
                    for item in ast.walk(call)
                    if isinstance(item, ast.Name) and item.id in assignments
                }
            )
            operands: list[dict] = []
            for name in names:
                info = assignments[name]
                indices = info["indices"]
                if len(indices) != 1 or indices[0] >= len(manifest):
                    continue
                source = manifest[indices[0]]
                raw = str(source.get("raw", "")).strip()
                metric_key = str(source.get("metric_key", ""))
                operands.append(
                    {
                        "variable": name,
                        "raw": raw,
                        "parenthesized_negative": raw.startswith("(") and raw.endswith(")"),
                        "uses_abs": bool(info["uses_abs"]) or _uses_abs_at_call(call, name),
                        "expression": info["expression"],
                        "ticker": source.get("ticker"),
                        "year": source.get("year"),
                        "metric_key": metric_key,
                        "magnitude_metric": any(
                            word in metric_key.casefold()
                            for word in MAGNITUDE_METRIC_WORDS
                        ),
                        "source_table": source.get("source_table"),
                    }
                )
            negative = [
                item
                for item in operands
                if item["parenthesized_negative"] and item["magnitude_metric"]
            ]
            if len(negative) < 2:
                continue
            checked_calls += 1
            unsafe = [item for item in negative if not item["uses_abs"]]
            if not unsafe:
                continue
            findings.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": row.get("answer"),
                    "operator": call.func.id,
                    "negative_operand_count": len(negative),
                    "unsafe_negative_operand_count": len(unsafe),
                    "operands": operands,
                }
            )
    # Multiple nested extrema calls may repeat an identical question/operator.
    unique: dict[tuple[int, str], dict] = {}
    for finding in findings:
        key = (finding["id"], finding["operator"])
        previous = unique.get(key)
        if previous is None or finding["negative_operand_count"] > previous["negative_operand_count"]:
            unique[key] = finding
    final = sorted(unique.values(), key=lambda item: item["id"])
    return {
        "submission": str(submission.resolve()),
        "extrema_calls_with_multiple_parenthesized_operands": checked_calls,
        "finding_count": len(final),
        "question_ids": [item["id"] for item in final],
        "findings": final,
        "claim_limit": (
            "Parenthesized presentation plus extrema language is a source-review signal. "
            "Reversal/income semantics can legitimately use reported signs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
