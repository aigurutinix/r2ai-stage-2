"""Find direct-lookups whose sibling row matches the question much better.

The audit is intentionally limited to one-cell programs with only display
scaling.  It compares rows inside the exact same physical table and selected
period column, so document, unit and year remain fixed.  Results are review
hints; a candidate row must still be checked in the full table before repair.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

try:
    from audit_exact_value_semantic_alternatives import semantic_bigrams, semantic_tokens
    from audit_legacy_source_scope import fold
    from audit_source_cell_semantics import is_financial_number, parsed_table
except ModuleNotFoundError:
    from scripts.audit_exact_value_semantic_alternatives import semantic_bigrams, semantic_tokens
    from scripts.audit_legacy_source_scope import fold
    from scripts.audit_source_cell_semantics import is_financial_number, parsed_table


ROOT = Path(__file__).resolve().parents[1]


def direct_display_only(code: object) -> bool:
    """Allow one source scalar plus abs/round and power-of-ten scaling."""

    try:
        tree = ast.parse(str(code))
    except SyntaxError:
        return False
    result_nodes = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets)
    ]
    if not result_nodes:
        return False
    rendered = " ".join(ast.unparse(node) for node in result_nodes)
    names = set(re.findall(r"\bv\d+\b", rendered))
    if names != {"v0"}:
        return False
    for node in ast.walk(ast.Module(body=[ast.Expr(result_nodes[-1])], type_ignores=[])):
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.Pow)):
            return False
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name not in {"round", "abs", "float"}:
                return False
    return True


def row_label(frame, row: int, selected_column: int) -> str:
    parts = []
    for column in range(selected_column):
        value = str(frame.iloc[row, column]).strip()
        if value and value.casefold() != "nan" and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def score(question: object, label: object) -> dict:
    q_tokens = set(semantic_tokens(question))
    label_tokens = set(semantic_tokens(label))
    token_overlap = sorted(q_tokens & label_tokens)
    bigram_overlap = sorted(semantic_bigrams(question) & semantic_bigrams(label))
    return {
        "score": len(token_overlap) + 3 * len(bigram_overlap),
        "token_overlap": token_overlap,
        "bigram_overlap": [" ".join(value) for value in bigram_overlap],
    }


def audit(lineage_path: Path, submission_dir: Path) -> dict:
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    submission = {
        int(row["id"]): row
        for row in json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    }
    checked = 0
    findings = []
    for record in lineage.get("records", []):
        qid = int(record["id"])
        cells = record.get("cells", [])
        row = submission.get(qid, {})
        if len(cells) != 1 or not direct_display_only(row.get("pandas_query", "")):
            continue
        checked += 1
        cell = cells[0]
        frame = parsed_table(str(cell["source_table"]))
        selected_row = int(cell["row_idx"])
        selected_column = int(cell["col_idx"])
        current_label = row_label(frame, selected_row, selected_column)
        current = score(record.get("question", ""), current_label)
        alternatives = []
        for candidate_row in range(len(frame)):
            if candidate_row == selected_row:
                continue
            raw = frame.iloc[candidate_row, selected_column]
            if not is_financial_number(raw):
                continue
            label = row_label(frame, candidate_row, selected_column)
            if not label:
                continue
            candidate = score(record.get("question", ""), label)
            gain = candidate["score"] - current["score"]
            if candidate["score"] < 5 or gain < 4 or not candidate["bigram_overlap"]:
                continue
            # Do not replace an explicit total with a component unless the
            # question itself does not ask for a total.
            question_folded = fold(record.get("question", ""))
            current_folded = fold(current_label)
            candidate_folded = fold(label)
            if "tong" in question_folded and any(
                cue in current_folded for cue in ("tong", "cong")
            ) and not any(cue in candidate_folded for cue in ("tong", "cong")):
                continue
            alternatives.append(
                {
                    "row": candidate_row,
                    "column": selected_column,
                    "raw": str(raw),
                    "label": label,
                    "semantic": candidate,
                    "gain": gain,
                }
            )
        if not alternatives:
            continue
        alternatives.sort(key=lambda item: (-item["gain"], item["row"]))
        findings.append(
            {
                "id": qid,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "source_table": cell.get("source_table"),
                "selected": {
                    "row": selected_row,
                    "column": selected_column,
                    "raw": cell.get("raw_physical"),
                    "label": current_label,
                    "semantic": current,
                },
                "alternatives": alternatives[:5],
                "status": "review_required",
            }
        )
    return {
        "kind": "direct_row_sibling_semantic_review_queue",
        "lineage": str(lineage_path),
        "submission": str(submission_dir),
        "eligible_direct_questions_checked": checked,
        "finding_count": len(findings),
        "question_ids": [row["id"] for row in findings],
        "policy": "Read-only lexical row competition within the same table and column.",
        "findings": findings,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lineage",
        nargs="?",
        type=Path,
        default=ROOT / "build" / "v210_source_cell_lineage_v209.json",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=ROOT / "sub_top123_candidate_v209_q15_board_role_r2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "v210_direct_row_sibling_semantics_v209.json",
    )
    args = parser.parse_args()
    payload = audit(args.lineage.resolve(), args.submission_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "findings"}, ensure_ascii=False, indent=2))
    if payload["findings"]:
        print(json.dumps({"findings": payload["findings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
