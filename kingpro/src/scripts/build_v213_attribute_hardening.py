"""Build a value-preserving runtime hardening candidate from v212.

The public scorer reported one execution ``AttributeError``.  Pandas can infer
an accounting cell as a numeric scalar, while a number of legacy programs call
``.replace`` directly on that cell.  Wrap only the unsafe receiver at the root
of each replacement chain with ``str``; calculations, answers, evidence and
table selections remain unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v212_difference_magnitude_batch5"
OUTPUT = ROOT / "sub_top123_candidate_v213_attribute_hardened"

TARGET_IDS = {
    4, 41, 65, 67, 83, 148, 158, 207, 225, 226, 230, 246, 264, 287,
    292, 321, 345, 596, 597, 620, 644, 672, 716, 727, 900, 907, 921, 969,
}


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_receiver(node: ast.AST) -> bool:
    """Return whether a replacement-chain receiver is already string-safe."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "str":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "replace":
            return _safe_receiver(node.func.value)
    if isinstance(node, ast.Attribute) and node.attr == "str":
        return True
    return False


class ReplaceReceiverHardener(ast.NodeTransformer):
    def __init__(self) -> None:
        self.wraps = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Visit children first: only the root of a chained .replace sequence is
        # wrapped, and every outer call then inherits a string result.
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
            and not _safe_receiver(node.func.value)
        ):
            node.func.value = ast.Call(
                func=ast.Name(id="str", ctx=ast.Load()),
                args=[node.func.value],
                keywords=[],
            )
            self.wraps += 1
        return node


def _harden(code: str) -> tuple[str, int]:
    tree = ast.parse(code)
    hardener = ReplaceReceiverHardener()
    tree = hardener.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), hardener.wraps


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    per_question_wraps: dict[str, int] = {}

    for row in rows:
        qid = int(row["id"])
        if qid not in TARGET_IDS:
            continue
        hardened, wraps = _harden(str(row.get("pandas_query") or ""))
        if wraps < 1:
            raise AssertionError(f"q{qid}: expected at least one receiver wrap")
        row["pandas_query"] = hardened
        per_question_wraps[str(qid)] = wraps

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(TARGET_IDS):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    for row in rows:
        qid = int(row["id"])
        fields = {
            key for key in row if row.get(key) != source_by_id[qid].get(key)
        }
        if qid in TARGET_IDS:
            if fields != {"pandas_query"}:
                raise AssertionError(f"q{qid} unexpected changed fields: {fields}")
        elif fields:
            raise AssertionError(f"q{qid} unexpectedly changed: {fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(TARGET_IDS),
        "programs_hardened": len(TARGET_IDS),
        "receiver_wraps": sum(per_question_wraps.values()),
        "per_question_wraps": per_question_wraps,
        "changed_fields": ["pandas_query"],
        "answers_changed": 0,
        "evidence_changed": 0,
        "tables_changed": 0,
        "transformation": "unsafe_cell.replace(...) -> str(unsafe_cell).replace(...)",
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": "Runtime hardening; public effect unknown until submitted.",
    }
    _write_json(OUTPUT / "v213_attribute_hardening_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
