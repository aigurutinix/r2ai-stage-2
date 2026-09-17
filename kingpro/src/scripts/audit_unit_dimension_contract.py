"""Audit operand-level currency units against the final answer expression.

The older unit audits compare a report/table marker with a manifest scale.  A
program may legitimately keep a value in its printed unit and normalize it in
the final expression, so that comparison produces many false positives.  This
read-only audit follows each source operand through the submitted AST instead:

* map ``v0``, ``v1``, ... assignments to their manifest rows;
* resolve the explicit unit marker nearest each cited source table;
* symbolically propagate constant multipliers/divisors into ``result``; and
* require every contributing currency operand to reach the same VND-normalized
  coefficient and, for non-average arithmetic, the requested output unit.

Only linear currency arithmetic with fully resolved markers is claimed.  The
audit deliberately skips ratios, percentages, dataframe reductions, weighted
averages, and programs whose source lineage cannot be proved statically.
Findings remain triage signals and never mutate a submission.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:  # Imported by tests from the repository root.
    from scripts.audit_local_source_units import (
        extracted_text_index,
        nearest_unit_marker,
        parse_table_ref,
    )
except ImportError:  # Executed directly from ``scripts``.
    from audit_local_source_units import (  # type: ignore
        extracted_text_index,
        nearest_unit_marker,
        parse_table_ref,
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "financial_statements"


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in text if not unicodedata.combining(char)).replace("đ", "d")


def requested_currency_factor(question: str) -> float | None:
    """Return the requested VND output factor, excluding non-currency units."""

    text = _fold(question)
    if re.search(r"\b(co phieu|lan|phan tram|%|diem phan tram|he so)\b", text):
        return None
    patterns = (
        # Currency suffixes are mandatory.  In Vietnamese questions the word
        # "công ty" is ubiquitous; accepting a bare "ty" therefore turns the
        # company noun into a false request for output in billions.
        (1e12, r"\bnghin\s+ty\s*(?:vnd|dong)\b"),
        (1e11, r"\btram\s+ty\s*(?:vnd|dong)\b"),
        (1e9, r"\bty\s*(?:vnd|dong)\b"),
        (1e8, r"\btram\s+trieu\s*(?:vnd|dong)\b"),
        (1e6, r"\btrieu\s*(?:vnd|dong)\b"),
        (1e3, r"\b(?:ngan|nghin)\s*(?:vnd|dong)\b"),
        (1.0, r"\b(?:vnd|dong)\b"),
    )
    for factor, pattern in patterns:
        if re.search(pattern, text):
            return factor
    return None


def _literal(node: ast.AST) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if isinstance(value, (int, float)):
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _iloc_access(node: ast.AST) -> tuple[int, str] | None:
    """Recognize ``df.iloc[row]['column']`` anywhere below *node*."""

    for item in ast.walk(node):
        if not isinstance(item, ast.Subscript) or not isinstance(item.slice, ast.Constant):
            continue
        column = item.slice.value
        inner = item.value
        if not (
            isinstance(column, str)
            and isinstance(inner, ast.Subscript)
            and isinstance(inner.value, ast.Attribute)
            and inner.value.attr == "iloc"
        ):
            continue
        row = _literal(inner.slice)
        if isinstance(row, int):
            return row, column
    return None


def _source_row(node: ast.AST) -> int | None:
    """Return the one manifest row read by a scalar source assignment."""

    rows = {
        access[0]
        for item in ast.walk(node)
        if (access := _iloc_access(item)) is not None and access[1] == "raw"
    }
    return next(iter(rows)) if len(rows) == 1 else None


def _assignment_scale(node: ast.AST, row: int, manifest_scale: float) -> float | None:
    """Evaluate the unit multiplier around one ``_btc_number`` source atom."""

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _assignment_scale(node.operand, row, manifest_scale)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _assignment_scale(node.left, row, manifest_scale)
        right = _assignment_scale(node.right, row, manifest_scale)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div) and right != 0:
            return left / right
        return None
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name in {"_btc_number", "parse_btc_number", "_parse"}:
            return 1.0 if _source_row(node) == row else None
        if name == "float" and node.args:
            access = _iloc_access(node.args[0])
            if access == (row, "scale"):
                return manifest_scale
            literal = _literal(node.args[0])
            return float(literal) if isinstance(literal, (int, float)) else None
        if name in {"abs", "round"} and node.args:
            return _assignment_scale(node.args[0], row, manifest_scale)
    return None


@dataclass
class Linear:
    constant: float
    coefficients: dict[int, float]

    def scaled(self, factor: float) -> "Linear":
        return Linear(
            self.constant * factor,
            {key: value * factor for key, value in self.coefficients.items()},
        )


Branches = list[Linear]


def _combine(left: Linear, right: Linear, sign: float) -> Linear:
    coefficients = dict(left.coefficients)
    for key, value in right.coefficients.items():
        coefficients[key] = coefficients.get(key, 0.0) + sign * value
    return Linear(left.constant + sign * right.constant, coefficients)


def _linear(node: ast.AST, env: dict[str, Branches]) -> Branches | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return [Linear(float(node.value), {})]
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp):
        branches = _linear(node.operand, env)
        if branches is None:
            return None
        factor = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return [branch.scaled(factor) for branch in branches]
    if isinstance(node, ast.BinOp):
        left = _linear(node.left, env)
        right = _linear(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            return [_combine(a, b, sign) for a in left for b in right]
        if isinstance(node.op, (ast.Mult, ast.Div)):
            result: Branches = []
            for a in left:
                for b in right:
                    a_scalar = not a.coefficients
                    b_scalar = not b.coefficients
                    if isinstance(node.op, ast.Mult) and a_scalar:
                        result.append(b.scaled(a.constant))
                    elif isinstance(node.op, ast.Mult) and b_scalar:
                        result.append(a.scaled(b.constant))
                    elif isinstance(node.op, ast.Div) and b_scalar and b.constant != 0:
                        result.append(a.scaled(1.0 / b.constant))
                    else:
                        return None
            return result
        return None
    if isinstance(node, ast.IfExp):
        left = _linear(node.body, env)
        right = _linear(node.orelse, env)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        if name in {"abs", "round", "float"} and node.args:
            return _linear(node.args[0], env)
        if name in {"max", "min"}:
            result: Branches = []
            for arg in node.args:
                branches = _linear(arg, env)
                if branches is None:
                    return None
                result.extend(branches)
            return result or None
        if name == "sum" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            total: Branches = [Linear(0.0, {})]
            for item in node.args[0].elts:
                branches = _linear(item, env)
                if branches is None:
                    return None
                total = [_combine(a, b, 1.0) for a in total for b in branches]
            return total
    return None


def _manifest(candidate: Path, qid: int) -> list[dict[str, str]]:
    path = candidate / "data" / "q{}_source_cells.csv".format(qid)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _result_branches(program: str, manifest: list[dict[str, str]]) -> Branches | None:
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None
    env: dict[str, Branches] = {}
    result: Branches | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        row = _source_row(statement.value)
        if row is not None and 0 <= row < len(manifest):
            try:
                manifest_scale = float(manifest[row].get("scale", 1.0) or 1.0)
            except ValueError:
                continue
            factor = _assignment_scale(statement.value, row, manifest_scale)
            if factor is not None:
                env[target.id] = [Linear(0.0, {row: factor})]
                continue
        branches = _linear(statement.value, env)
        if branches is not None:
            env[target.id] = branches
            if target.id == "result":
                result = branches
    return result


def _question_is_mean(question: str) -> bool:
    return bool(re.search(r"\b(trung binh|binh quan)\b", _fold(question)))


def _question_is_weighted_or_ratio(question: str) -> bool:
    text = _fold(question)
    return bool(
        re.search(
            r"\b(ty le|phan tram|bien loi nhuan|he so|tren moi|chia cho|"
            r"trung binh gia quyen|binh quan gia quyen)\b|%",
            text,
        )
    )


def _close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=max(1e-18, abs(right) * tolerance))


def audit_candidate(candidate: Path, data_root: Path, max_distance: int = 60) -> dict:
    candidate = candidate.resolve()
    records = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    index = extracted_text_index(data_root.resolve())
    text_cache: dict[str, list[str]] = {}
    findings: list[dict] = []
    checked = 0
    skipped = {
        "non_currency_or_ratio": 0,
        "no_manifest": 0,
        "ocr_normalized_operand": 0,
        "nonlinear_or_unresolved_ast": 0,
        "missing_local_marker": 0,
    }

    for record in records:
        question = str(record.get("question", ""))
        output_factor = requested_currency_factor(question)
        if output_factor is None or _question_is_weighted_or_ratio(question):
            skipped["non_currency_or_ratio"] += 1
            continue
        qid = int(record["id"])
        manifest = _manifest(candidate, qid)
        if not manifest:
            skipped["no_manifest"] += 1
            continue
        # A metric explicitly tagged ``_ocr`` can carry a decimal correction
        # in the query (for example raw ``3451`` meaning ``34,51``).  The
        # nearby financial-statement unit marker does not encode that OCR
        # repair, so coefficient/unit comparison would manufacture a false
        # mismatch.  Keep this audit high precision and leave OCR-normalized
        # operands to source-field/typed-factor validation.
        if any(str(item.get("metric_key", "")).endswith("_ocr") for item in manifest):
            skipped["ocr_normalized_operand"] += 1
            continue
        branches = _result_branches(str(record.get("pandas_query", "")), manifest)
        if not branches or any(not branch.coefficients for branch in branches):
            skipped["nonlinear_or_unresolved_ast"] += 1
            continue
        used_rows = sorted({row for branch in branches for row in branch.coefficients})
        local_factors: dict[int, float] = {}
        marker_evidence: dict[int, dict] = {}
        marker_failed = False
        for row in used_rows:
            if row >= len(manifest):
                marker_failed = True
                break
            parsed = parse_table_ref(str(manifest[row].get("source_table", "")))
            if parsed is None or parsed[0] not in index:
                marker_failed = True
                break
            document, line = parsed
            if document not in text_cache:
                text_cache[document] = index[document].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            marker = nearest_unit_marker(text_cache[document], line, max_distance=max_distance)
            if marker is None:
                marker_failed = True
                break
            local_factors[row] = float(marker["factor"])
            marker_evidence[row] = marker
        if marker_failed:
            skipped["missing_local_marker"] += 1
            continue

        rates: list[dict] = []
        for branch_index, branch in enumerate(branches):
            for row, coefficient in branch.coefficients.items():
                if abs(coefficient) <= 1e-30:
                    continue
                rates.append(
                    {
                        "branch": branch_index,
                        "row": row,
                        "coefficient": coefficient,
                        "local_unit_factor": local_factors[row],
                        "vnd_normalized_rate": abs(coefficient) / local_factors[row],
                    }
                )
        if not rates:
            continue
        checked += 1
        normalized = [item["vnd_normalized_rate"] for item in rates]
        reference = normalized[0]
        relative_mismatch = any(not _close(value, reference) for value in normalized[1:])

        # For a non-mean sum/difference/direct value, each normalized operand
        # must contribute one requested output unit per VND.  Means add a
        # legitimate cardinality divisor, so only their relative rates are
        # asserted here; arity is covered by audit_mean_aggregation_arity.py.
        output_mismatch = False
        expected_rate = 1.0 / output_factor
        if not _question_is_mean(question):
            output_mismatch = any(not _close(value, expected_rate) for value in normalized)

        if not relative_mismatch and not output_mismatch:
            continue
        findings.append(
            {
                "id": qid,
                "question": question,
                "answer": record.get("answer"),
                "requested_output_factor": output_factor,
                "expected_vnd_normalized_rate": None if _question_is_mean(question) else expected_rate,
                "relative_unit_mismatch": relative_mismatch,
                "output_scale_mismatch": output_mismatch,
                "rates": rates,
                "sources": [
                    {
                        "row": row,
                        "raw": manifest[row].get("raw"),
                        "metric": manifest[row].get("metric_key"),
                        "table_ref": manifest[row].get("source_table"),
                        "marker_factor": local_factors[row],
                        "marker_line": marker_evidence[row]["line"],
                        "marker_text": marker_evidence[row]["text"],
                    }
                    for row in used_rows
                ],
                "reason": "operand coefficients violate the final currency-dimension contract",
            }
        )

    return {
        "submission": str(candidate),
        "question_count": len(records),
        "eligible_programs_checked": checked,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "skipped": skipped,
        "findings": findings,
        "claim_limit": (
            "High-confidence linear currency contract only. Ratios, weighted averages, "
            "dynamic dataframe reductions, unresolved lineage, and missing unit markers are skipped."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-distance", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit_candidate(args.submission, args.data_root, args.max_distance)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if report["finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
