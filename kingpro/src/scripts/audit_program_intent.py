"""Flag executable legacy queries whose arithmetic shape conflicts with the question.

This is a read-only triage tool.  It deliberately does not rewrite answers:
each finding still needs source-table verification before it can be promoted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", value).strip()


def audited_ids(submission_dir: Path) -> set[int]:
    result: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.update(int(row["id"]) for row in payload if "id" in row)
    return result


def _column_dependency_key(node: ast.AST) -> str | None:
    """Return a stable key for pandas column reads and assignments.

    Panel programs build their arithmetic in derived DataFrame columns and
    later read the same column through a filtered row (for example
    ``selected['gross_margin_pct']``).  Tracking the constant column name lets
    this conservative audit follow that dependency without pretending to
    understand all pandas aliasing.
    """
    if isinstance(node, ast.Subscript):
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return f"@column:{slice_node.value}"
    # A derived column assigned as ``df['ratio']`` is often consumed later as
    # ``selected.ratio`` after pandas filtering. Treat both spellings as the
    # same conservative dependency key.
    if isinstance(node, ast.Attribute):
        return f"@column:{node.attr}"
    return None


def assignment_map(tree: ast.AST) -> tuple[dict[str, list[ast.AST]], list[ast.AST]]:
    """Collect every possible assignment, including mutually exclusive branches.

    A static audit must not let the final ``result = None`` fallback hide the
    arithmetic performed in the successful branch.  Keeping all assignments
    intentionally over-approximates the dependency graph; findings are triage,
    not automatic rewrites.
    """
    assignments: dict[str, list[ast.AST]] = {}
    result_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(value)
                if target.id == "result":
                    result_nodes.append(value)
            column_key = _column_dependency_key(target)
            if column_key:
                assignments.setdefault(column_key, []).append(value)
    return assignments, result_nodes


def dependency_shape(result_nodes: list[ast.AST], assignments: dict[str, list[ast.AST]]) -> dict:
    operators: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    sources: set[str] = set()
    numeric_constants: list[float] = []
    visited_names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in visited_names:
            visited_names.add(node.id)
            for value in assignments[node.id]:
                visit(value)
            return
        column_key = _column_dependency_key(node)
        column_keys: list[str] = []
        if column_key:
            column_keys.append(column_key)
            base_key = re.sub(r"_(?:19|20)\d{2}$|_[xy]$", "", column_key)
            if base_key != column_key:
                column_keys.append(base_key)
        # pandas APIs such as ``pivot(values='gross_margin_pct')`` express a
        # column dependency as a string argument rather than a Subscript.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            column_keys.append(f"@column:{node.value}")
        for key in column_keys:
            if key in assignments and key not in visited_names:
                visited_names.add(key)
                for value in assignments[key]:
                    visit(value)
        if isinstance(node, ast.BinOp):
            operators[type(node.op).__name__] += 1
        elif isinstance(node, ast.UnaryOp):
            operators[type(node.op).__name__] += 1
        elif isinstance(node, ast.Compare):
            for operator in node.ops:
                operators[type(operator).__name__] += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls[node.func.id] += 1
            elif isinstance(node.func, ast.Attribute):
                calls[node.func.attr] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            numeric_constants.append(float(node.value))
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            rendered = ast.unparse(node)
            if any(marker in rendered for marker in ("df", "dfs", ".iloc", ".loc", ".values")):
                sources.add(rendered)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for result_node in result_nodes:
        visit(result_node)
    return {
        "operators": dict(operators),
        "calls": dict(calls),
        "source_reads": len(sources),
        "constants": sorted(set(numeric_constants)),
    }


def classify(question: str) -> set[str]:
    text = normalize(question)
    original = question.casefold()
    intents: set[str] = set()
    if any(token in original for token in ("tỷ lệ", "tỉ lệ", "tỷ trọng", "tỉ trọng", "phần trăm", "%")):
        intents.add("ratio")
    if "gap" in text and "bao nhieu lan" in text:
        intents.add("ratio")
    if any(token in text for token in ("bao nhieu lan", "he so")):
        intents.add("ratio")
    if any(token in text for token in (
        "tru di", "be hon", "thap hon bao nhieu",
        "cao hon bao nhieu", "thay doi", "bien dong",
    )) or (
        "vuot" in text and "bao nhieu" in text and "co bao nhieu" not in text
    ) or ("chenh lech" in text and any(token in text for token in ("giua", "so voi"))):
        intents.add("difference")
    # "Lỗ/lãi chênh lệch tỷ giá" and "cổ phiếu bình quân gia quyền"
    # are disclosed line-item names.  They do not, by themselves, request a
    # subtraction or a freshly calculated arithmetic mean.
    if "trung binh" in text or ("binh quan" in text and "binh quan gia quyen" not in text):
        intents.add("mean")
    if "trung vi" in text:
        intents.add("median")
    if "diem phan tram" in text:
        intents.add("difference")
    if any(token in text for token in ("tang truong", "toc do tang", "ty le tang", "ti le tang")):
        intents.add("growth")
    if "co bao nhieu" in text and any(token in text for token in ("cong ty", "doanh nghiep", "nam")):
        intents.add("count")
    extreme_tokens = ("cao nhat", "lon nhat", "thap nhat", "be nhat", "toi da", "toi thieu")
    # "Tiền thuê tối thiểu" is also a named disclosure, not a request to
    # choose a minimum.  Preserve the extreme intent only when another
    # selection phrase is present.
    minimum_lease_disclosure = "tien thue toi thieu" in text
    if any(token in text for token in extreme_tokens) and not minimum_lease_disclosure:
        intents.add("extreme")
    if any(token in text for token in ("tich luy", "qua cac nam")) and "tong" in text:
        intents.add("sum")
    return intents


def score_finding(row: dict, shape: dict) -> tuple[int, list[str]]:
    question = str(row.get("question", ""))
    query = str(row.get("pandas_query", ""))
    text = normalize(question)
    intents = classify(question)
    operators = Counter(shape["operators"])
    calls = Counter(shape["calls"])
    source_reads = max(
        int(shape["source_reads"]),
        calls["_source_value"],
        2 if calls["panel"] else 0,
        calls["persistent"],
    )
    shape["effective_source_reads"] = source_reads
    reasons: list[str] = []
    score = 0

    has_division = operators["Div"] > 0 or calls["mean"] > 0
    has_difference = operators["Sub"] > 0 or operators["USub"] > 0
    has_sum = operators["Add"] > 0 or calls["sum"] > 0
    has_extreme = any(
        calls[name] > 0
        for name in ("max", "min", "idxmax", "idxmin", "nlargest", "nsmallest")
    ) or calls["sort_values"] > 0
    has_comparison = any(
        operators[name] > 0
        for name in ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn")
    )

    if "ratio" in intents and not has_division:
        reasons.append("ratio/percentage question without division in result dependency graph")
        score += 6
    if "difference" in intents and not has_difference:
        reasons.append("difference/change question without subtraction in result dependency graph")
        score += 5
    if "mean" in intents and not has_division and source_reads >= 2:
        reasons.append("mean question with multiple source reads but no division/mean")
        score += 4
    if "extreme" in intents and not has_extreme and source_reads >= 2:
        reasons.append("max/min question with multiple source reads but no max/min")
        score += 4
    if "sum" in intents and not has_sum and source_reads >= 2:
        reasons.append("multi-year cumulative total without addition/sum")
        score += 4
    if "median" in intents and calls["median"] == 0 and source_reads >= 2:
        reasons.append("median selection without median in result dependency graph")
        score += 4
    if "growth" in intents and not (has_division and has_difference) and source_reads >= 2:
        reasons.append("growth request without both division and subtraction")
        score += 5
    if "count" in intents and not (has_comparison and has_sum):
        reasons.append("conditional count without comparison and addition/sum")
        score += 5
    if len(row.get("relevant_docs", [])) >= 2 and source_reads <= 1:
        reasons.append("multiple relevant documents but result depends on <=1 detected source read")
        score += 3
    if (
        "ty le so huu" not in text
        and "so tien" in text
        and re.search(r"tyle_so_huu|ownership|ty_le_so_huu", normalize(query).replace(" ", "_"))
        and re.search(r"1e\+?\d+", query, re.IGNORECASE)
    ):
        reasons.append("money answer synthesized from ownership percentage and an assumed power-of-ten amount")
        score += 9
    if source_reads == 0:
        reasons.append("result dependency graph has no detected dataframe source read")
        score += 2
    return score, reasons


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    rows = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known_audited = audited_ids(args.submission_dir)
    findings: list[dict] = []
    parse_failures: list[dict] = []
    checked = 0
    for row in rows:
        qid = int(row["id"])
        if qid in known_audited and not args.include_audited:
            continue
        checked += 1
        query = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(query)
        except SyntaxError as error:
            parse_failures.append({"id": qid, "error": str(error)})
            continue
        assignments, result_nodes = assignment_map(tree)
        if not result_nodes:
            parse_failures.append({"id": qid, "error": "no result assignment"})
            continue
        shape = dependency_shape(result_nodes, assignments)
        score, reasons = score_finding(row, shape)
        if score <= 0:
            continue
        findings.append({
            "id": qid,
            "score": score,
            "intents": sorted(classify(str(row.get("question", "")))),
            "shape": shape,
            "reasons": reasons,
            "answer": row.get("answer"),
            "question": row.get("question"),
            "relevant_docs": row.get("relevant_docs", []),
            "relevant_tables": row.get("relevant_tables", []),
        })

    findings.sort(key=lambda item: (-item["score"], item["id"]))
    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known_audited),
        "checked": checked,
        "finding_count": len(findings),
        "parse_failure_count": len(parse_failures),
        "findings": findings,
        "parse_failures": parse_failures,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
