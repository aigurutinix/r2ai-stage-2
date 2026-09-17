"""Rank unaudited submission programs whose selected row labels poorly match the question.

This is a read-only triage tool.  It does not assume that lexical mismatch proves
an error; note disclosures often use legitimate accounting synonyms.  Its output
is intended to focus manual source-table review on old-model programs that select
an unrelated nearby row.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


STOPWORDS = {
    "bao", "bao nhieu", "bang", "cac", "cua", "cong", "co", "cho", "cuoi",
    "dau", "den", "do", "dong", "duoc", "giai", "gia", "giua", "hay", "la",
    "lam", "lon", "may", "me", "mot", "nam", "nho", "nhat", "nhieu", "o",
    "phan", "qua", "sau", "tai", "theo", "thi", "trong", "tren", "trieu",
    "ty", "vao", "va", "voi", "vnd",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return {
        token for token in fold(text).split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def audited_ids(root: Path) -> set[int]:
    ids: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = root / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if isinstance(row, dict) and "id" in row:
                ids.add(int(row["id"]))
    return ids


def looks_numeric(value: object) -> bool:
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return True
    normalized = text.replace("\u00a0", "").replace(" ", "")
    normalized = normalized.replace("(", "-").replace(")", "")
    normalized = normalized.replace("%", "").replace(".", "").replace(",", "")
    return normalized.lstrip("+-").isdigit()


def source_row_labels(
    submission_dir: Path, tables_root: Path
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Recover row labels for generated source-cell programs.

    Rewritten programs intentionally read a tiny ``q<ID>_source_cells.csv`` at
    grader runtime, so their pandas code no longer contains the source row
    label.  The cumulative audit manifest still records the exact BTC
    document, table line and row coordinate.  Resolve that coordinate against
    ``build/tables`` and collect textual cells preceding the numeric operand.

    The second mapping records resolution problems.  Missing source evidence
    must remain visible rather than silently making a semantic gate look green.
    """

    result: dict[int, list[str]] = {}
    unresolved: dict[int, list[str]] = {}
    for audit_name in ("source_audit.json", "panel_source_audit.json"):
        audit_path = submission_dir / audit_name
        if not audit_path.exists():
            continue
        for audit_row in json.loads(audit_path.read_text(encoding="utf-8")):
            if not isinstance(audit_row, dict) or "id" not in audit_row:
                continue
            qid = int(audit_row["id"])
            labels = result.setdefault(qid, [])
            for source in audit_row.get("sources", []):
                if not isinstance(source, dict):
                    continue
                recorded_row_labels = [
                    str(value).strip()
                    for value in source.get("source_row_labels", [])
                    if str(value).strip()
                ]
                if recorded_row_labels:
                    labels.extend(
                        label for label in recorded_row_labels if label not in labels
                    )
                    continue
                table_ref = str(source.get("table_ref", ""))
                if "|" not in table_ref:
                    unresolved.setdefault(qid, []).append(
                        f"source without table_ref: {table_ref or '<empty>'}"
                    )
                    continue
                document, line_text = table_ref.rsplit("|", 1)
                candidates = sorted((tables_root / document).glob(f"*line{line_text}.csv"))
                if len(candidates) != 1:
                    unresolved.setdefault(qid, []).append(
                        f"{table_ref}: expected one table CSV, found {len(candidates)}"
                    )
                    continue
                try:
                    frame = pd.read_csv(
                        candidates[0], dtype=str, keep_default_na=False, encoding="utf-8-sig"
                    )
                    row_index = int(source["row"])
                    column_index = int(source["column"])
                    row = frame.iloc[row_index]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    unresolved.setdefault(qid, []).append(f"{table_ref}: {exc}")
                    continue
                # Labels normally occupy one of the cells to the left of the
                # selected value; include all textual candidates because many
                # tables split a hierarchy over two columns.
                for value in row.iloc[: max(1, column_index)].tolist():
                    label = str(value).strip()
                    if label and not looks_numeric(label) and len(tokens(label)) and label not in labels:
                        labels.append(label)
    return {qid: labels for qid, labels in result.items() if labels}, unresolved


def string_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    value = node.args[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def selected_labels(query: str) -> list[str]:
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return []
    labels: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"contains", "startswith", "endswith"}:
            continue
        label = string_arg(node)
        if label and len(tokens(label)):
            labels.append(label)
    return list(dict.fromkeys(labels))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-audited", action="store_true")
    parser.add_argument(
        "--tables-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "tables",
    )
    args = parser.parse_args()

    rows = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = set() if args.include_audited else audited_ids(args.submission_dir)
    audit_labels, unresolved_audit_sources = source_row_labels(
        args.submission_dir, args.tables_root
    ) if args.include_audited else ({}, {})
    findings = []
    for row in rows:
        qid = int(row["id"])
        if qid in known:
            continue
        question = str(row.get("question", ""))
        qtokens = tokens(question)
        query_labels = selected_labels(str(row.get("pandas_query", "")))
        labels = list(dict.fromkeys(query_labels + audit_labels.get(qid, [])))
        if not labels or not qtokens:
            continue
        label_tokens = set().union(*(tokens(label) for label in labels))
        shared = qtokens & label_tokens
        # Recall against the compact row-label vocabulary is more useful than
        # whole-question Jaccard, because questions contain company/year boilerplate.
        score = len(shared) / max(1, len(label_tokens))
        findings.append({
            "id": qid,
            "score": round(score, 4),
            "shared": sorted(shared),
            "labels": labels,
            "label_origin": (
                "query+source_audit" if query_labels and qid in audit_labels
                else "source_audit" if qid in audit_labels
                else "query"
            ),
            "question": question,
            "answer": row.get("answer"),
        })

    findings.sort(key=lambda item: (item["score"], item["id"]))
    result = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": len(known),
        "audited_ids_with_resolved_labels": len(audit_labels),
        "unresolved_audit_source_count": sum(
            len(messages) for messages in unresolved_audit_sources.values()
        ),
        "unresolved_audit_sources": unresolved_audit_sources,
        "scored": len(findings),
        "findings": findings[: args.limit],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
