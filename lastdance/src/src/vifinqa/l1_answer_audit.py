"""Answer-oriented L1 suspect ranking and independent cell-review gates.

The existing retriever is intentionally high recall.  This module turns its
ranked cells into an answer-blind review task: models see table semantics but
not numeric values and do not know which option is the current baseline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .l1_fact_layer import normalize
from .shadow_ir import extract_json_object


PRIMARY_METRICS = (
    "tong tai san",
    "tong no phai tra",
    "von chu so huu",
    "loi nhuan sau thue",
    "loi nhuan truoc thue",
    "doanh thu thuan",
    "tien va cac khoan tuong duong tien",
    "luu chuyen tien thuan",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cell_ref(row: dict[str, Any]) -> str:
    return f"t{int(row['table_id'])}r{int(row['row_index'])}c{int(row['column_index'])}"


def _metric_coverage(metric: str, *texts: str) -> float:
    wanted = set(normalize(metric).split())
    if not wanted:
        return 0.0
    seen = set(normalize(" ".join(texts)).split())
    return len(wanted & seen) / len(wanted)


def suspect_score(top: dict[str, Any], alternatives: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Prioritize likely semantic cell errors.

    Manual selections remain excluded by default because this function is also
    used by release-time checks.  Broad tuning can explicitly re-audit them in
    :func:`rank_suspects`; a manual override is provenance, not ground truth.
    """

    if top.get("selection_source") == "manual":
        return -1.0, ["manual"]
    score = 10.0
    reasons = ["automatic"]
    confidence = top.get("confidence")
    if confidence == "low":
        score += 40.0
        reasons.append("low_confidence")
    elif confidence == "medium":
        score += 25.0
        reasons.append("medium_confidence")

    gap = float(top.get("runner_up_gap", 0.0))
    if gap <= 2:
        score += 20.0
        reasons.append("gap_le_2")
    elif gap <= 5:
        score += 12.0
        reasons.append("gap_le_5")
    elif gap <= 10:
        score += 6.0
        reasons.append("gap_le_10")

    kind = str(top.get("table_kind", ""))
    if kind in {"other", "governance_or_company_info"}:
        score += 16.0
        reasons.append(f"weak_table_kind:{kind}")
    elif kind == "financial_notes":
        score += 7.0
        reasons.append("financial_notes")

    if top.get("document_scope") != top.get("requested_scope"):
        score += 15.0
        reasons.append("scope_mismatch")
    if float(top.get("period_score", 0.0)) < 10:
        score += 10.0
        reasons.append("weak_period")
    if float(top.get("column_role_score", 0.0)) <= 0:
        score += 7.0
        reasons.append("weak_column_role")

    coverage = _metric_coverage(
        str(top.get("metric_text", "")),
        str(top.get("row_text", "")),
        str(top.get("table_title", "")),
        str(top.get("column_text", "")),
    )
    if coverage < 0.6:
        score += 16.0
        reasons.append("metric_coverage_lt_0.6")
    elif coverage < 0.8:
        score += 8.0
        reasons.append("metric_coverage_lt_0.8")

    metric = normalize(str(top.get("metric_text", "")))
    if any(value in metric for value in PRIMARY_METRICS) and not kind.startswith("primary_"):
        score += 12.0
        reasons.append("primary_metric_nonprimary_table")

    different = [
        row for row in alternatives
        if abs(float(row.get("answer", 0.0)) - float(top.get("answer", 0.0)))
        > max(1e-9, abs(float(top.get("answer", 0.0))) * 1e-9)
    ]
    if different:
        score += min(10.0, 2.0 * len(different))
        reasons.append("answer_disagreement_in_topk")
    if any(
        float(row.get("answer", 0.0)) * float(top.get("answer", 0.0)) < 0
        for row in alternatives
    ):
        score += 6.0
        reasons.append("sign_disagreement")
    return round(score, 4), reasons


def rank_suspects(
    top_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    *,
    include_manual: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[int(row.get("question_id", -1))].append(row)
    result = []
    for top in top_rows:
        qid = int(top.get("question_id", -1))
        alternatives = [row for row in grouped[qid] if int(row.get("candidate_rank", 0)) != 1]
        score, reasons = suspect_score(top, alternatives)
        if score < 0 and include_manual and top.get("selection_source") == "manual":
            score, reasons = suspect_score(
                {**top, "selection_source": "automatic"}, alternatives
            )
            reasons = ["manual_reaudit", *[reason for reason in reasons if reason != "automatic"]]
        if score < 0:
            continue
        result.append({
            "question_id": qid,
            "suspect_score": score,
            "reasons": reasons,
            "baseline_ref": cell_ref(top),
            "baseline_answer": top["answer"],
            "confidence": top.get("confidence"),
            "table_kind": top.get("table_kind"),
            "question": top.get("question"),
            "metric_text": top.get("metric_text"),
        })
    return sorted(result, key=lambda row: (-row["suspect_score"], row["question_id"]))


L1_ILOC_RE_TEMPLATE = r"\b{variable}\.iloc\[\s*(\d+)\s*,\s*(\d+)\s*\]"


def rebase_l1_context_on_submission(
    context: dict[str, Any],
    submission_item: dict[str, Any],
    evidence_root: Path,
    connection: sqlite3.Connection | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve the current L1 baseline from champion evidence and Pandas code.

    Candidate registries are immutable and can lag behind cumulative manual
    fixes.  This binds a masked audit to the cell actually read by the current
    submission, without consulting the submission's saved numeric answer.
    """

    question_id = int(context.get("question_id", -1))
    expected_names = {
        f"q{question_id:04d}_evidence.csv",
        f"q{question_id:04d}_source.csv",
    }
    matches = [
        evidence
        for evidence in submission_item.get("evidence", [])
        if Path(str(evidence.get("csv_path", ""))).name in expected_names
    ]
    audit = {
        "question_id": question_id,
        "original_baseline_ref": context["baseline_ref"],
        "status": "UNRESOLVED",
    }
    if len(matches) != 1:
        audit["reason"] = f"expected one champion evidence path, found {len(matches)}"
        return None, audit
    evidence = matches[0]
    variable = str(evidence["variable"])
    coordinates = {
        (int(row), int(column))
        for row, column in re.findall(
            L1_ILOC_RE_TEMPLATE.format(variable=re.escape(variable)),
            str(submission_item.get("pandas_query", "")),
        )
    }
    if len(coordinates) != 1:
        audit["reason"] = f"expected one iloc for {variable}, found {len(coordinates)}"
        return None, audit
    row_index, column_index = next(iter(coordinates))
    csv_path = evidence_root / str(evidence["csv_path"])
    if not csv_path.is_file():
        audit["reason"] = f"missing champion evidence {csv_path}"
        return None, audit
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.reader(handle))
    data_index = row_index + 1
    if data_index >= len(csv_rows) or column_index >= len(csv_rows[data_index]):
        audit["reason"] = "champion iloc is outside the evidence CSV"
        return None, audit
    raw_value = csv_rows[data_index][column_index]
    option_matches = [
        option
        for option in context["options"]
        if int(option["binding"]["row_index"]) == row_index
        and int(option["binding"]["column_index"]) == column_index
        and str(option["binding"]["raw_value"]) == raw_value
    ]
    relevant_tables = set(map(str, submission_item.get("relevant_tables", [])))
    sourced_matches = [
        option
        for option in option_matches
        if str(option["binding"].get("evidence_key", "")) in relevant_tables
    ]
    if sourced_matches:
        option_matches = sourced_matches
    if len(option_matches) > 1 and connection is not None:
        data_rows = csv_rows[1:]
        exact = []
        for option in option_matches:
            table = connection.execute(
                "SELECT grid_json FROM tables WHERE table_id = ?",
                (int(option["binding"]["table_id"]),),
            ).fetchone()
            if table is None:
                continue
            grid = json.loads(table[0])
            width = max((len(row) for row in grid), default=0)
            padded = [
                [str(value) for value in row] + [""] * (width - len(row))
                for row in grid
            ]
            if padded == data_rows:
                exact.append(option)
        option_matches = exact
    if len(option_matches) != 1:
        audit.update({
            "reason": f"champion cell matches {len(option_matches)} audit options",
            "variable": variable,
            "row_index": row_index,
            "column_index": column_index,
        })
        return None, audit
    selected = option_matches[0]
    rebased = dict(context)
    rebased.update({
        "baseline_ref": selected["cell_ref"],
        "baseline_answer": selected["binding"]["answer"],
        "baseline": selected["binding"],
        "registry_baseline_ref": context["baseline_ref"],
        "baseline_source": "champion_submission",
    })
    audit.update({
        "status": "REBASED",
        "champion_baseline_ref": selected["cell_ref"],
        "changed_from_registry": selected["cell_ref"] != context["baseline_ref"],
        "variable": variable,
        "row_index": row_index,
        "column_index": column_index,
    })
    return rebased, audit


def _neighbors(grid: list[list[Any]], row_index: int) -> list[str]:
    result = []
    for index in range(max(0, row_index - 2), min(len(grid), row_index + 3)):
        cells = [str(value).strip() for value in grid[index] if str(value).strip()]
        # Numeric values are irrelevant to semantic selection and can anchor a judge.
        masked = ["<NUM>" if any(char.isdigit() for char in value) else value for value in cells]
        result.append(f"r{index}: " + " | ".join(masked)[:500])
    return result


NUMERIC_TEXT_RE = re.compile(r"(?<![A-Za-z])\(?-?\d[\d.,]*%?\)?")


def mask_numeric_text(value: Any) -> str:
    """Hide cell-like numbers but retain report years needed for period checks."""

    text = str(value or "")

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) == 4 and 2010 <= int(digits) <= 2030:
            return digits
        return "<NUM>"

    return NUMERIC_TEXT_RE.sub(replace, text)


def enrich_options(
    database: Path, candidates: Iterable[dict[str, Any]], *, limit: int = 10
) -> list[dict[str, Any]]:
    """Attach hierarchy context while retaining exact candidate data for binding."""

    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        unique.setdefault(cell_ref(row), row)
        if len(unique) >= limit:
            break
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        result = []
        for ref, row in unique.items():
            table = connection.execute(
                "SELECT grid_json, context_before, title_hint, unit_text FROM tables WHERE table_id = ?",
                (int(row["table_id"]),),
            ).fetchone()
            if table is None:
                raise ValueError(f"Missing table_id {row['table_id']}")
            grid = json.loads(table["grid_json"])
            result.append({
                "cell_ref": ref,
                "document_id": row["document_id"],
                "document_scope": row["document_scope"],
                "table_id": int(row["table_id"]),
                "evidence_key": row["evidence_key"],
                "table_kind": row["table_kind"],
                "title": row.get("table_title") or table["title_hint"],
                "context": str(table["context_before"] or "")[-900:],
                "unit_text": str(table["unit_text"] or "")[:300],
                "row_index": int(row["row_index"]),
                "column_index": int(row["column_index"]),
                "row_label": row["row_text"],
                "column_header": row["column_text"],
                "neighbors_masked": _neighbors(grid, int(row["row_index"])),
                # Kept out of the prompt; required for deterministic promotion.
                "binding": row,
            })
        return result
    finally:
        connection.close()


def option_order(context: dict[str, Any], reviewer: str) -> list[dict[str, Any]]:
    """Use a stable reviewer-specific shuffle to reduce rank-position agreement."""

    audit_id = context.get("audit_id", context.get("question_id", -1))
    seed = hashlib.sha256(
        f"{reviewer}:{audit_id}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "big"))
    result = list(context["options"])
    rng.shuffle(result)
    return result


def make_review_prompt(context: dict[str, Any], reviewer: str) -> str:
    lines = [
        f"Question ID: {context.get('question_id', -1)}",
        f"Audit target: {context.get('audit_id', context.get('question_id', -1))}",
        f"Câu hỏi: {context['question']}",
        f"Metric đã parse: {context['metric_text']}",
        f"Ticker={context['ticker']}; report_year={context['year']}; "
        f"scope={context['requested_scope']}; period={context['period_kind']}; "
        f"target_unit={context['target_unit']}",
        "",
        "Các lựa chọn ô (số đã bị che; thứ tự đã được tráo ngẫu nhiên):",
    ]
    for index, option in enumerate(option_order(context, reviewer), 1):
        lines.extend([
            f"OPTION {index} cell_ref={option['cell_ref']}",
            f"  document={option['document_id']} scope={option['document_scope']} kind={option['table_kind']}",
            f"  title={mask_numeric_text(option['title'])!r}",
            f"  context={mask_numeric_text(option['context'])!r}",
            f"  unit={mask_numeric_text(option['unit_text'])!r}",
            f"  row={mask_numeric_text(option['row_label'])!r}",
            f"  column={mask_numeric_text(option['column_header'])!r}",
            "  neighborhood=" + " || ".join(option["neighbors_masked"]),
        ])
    lines.extend([
        "",
        "Chọn đúng một ô chứa chính metric được hỏi tại đúng kỳ và scope. "
        "Phân biệt số dư với dòng biến động, tổng với thành phần, và income-statement "
        "expense với cash-flow/payment/accrual. Nếu không lựa chọn nào đúng, chọn NONE.",
        "Trả về JSON ngay lập tức, không viết phân tích trước JSON:",
        f'{{"question_id":{context.get("question_id", -1)},"selected_ref":"t1r2c3|NONE",'
        '"confidence":"HIGH|MEDIUM|LOW","reason":"...",'
        '"rejected_traps":["..."]}',
    ])
    return "\n".join(lines)


def validate_review(raw: str, context: dict[str, Any], model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": int(context.get("question_id", -1)),
        "model": model,
        "status": "INVALID",
        "raw_response": raw,
    }
    if "audit_id" in context:
        result["audit_id"] = context["audit_id"]
    try:
        try:
            payload = extract_json_object(raw)
        except ValueError:
            # Some instruct models emit a syntactically invalid
            # ``rejected_traps`` list while the three decision fields remain
            # unambiguous. Recover only those explicit fields; never infer a
            # missing selection from prose.
            patterns = {
                "question_id": r'"question_id"\s*:\s*(\d+)',
                "selected_ref": r'"selected_ref"\s*:\s*"([^"]+)"',
                "confidence": r'"confidence"\s*:\s*"([^"]+)"',
                "reason": r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"',
            }
            matches = {key: re.search(pattern, raw) for key, pattern in patterns.items()}
            if not all(matches.values()):
                raise
            payload = {
                "question_id": int(matches["question_id"].group(1)),
                "selected_ref": matches["selected_ref"].group(1),
                "confidence": matches["confidence"].group(1),
                "reason": json.loads(f'"{matches["reason"].group(1)}"'),
                "rejected_traps": [],
                "salvaged": True,
            }
        if int(payload.get("question_id", -1)) != int(context.get("question_id", -1)) and int(context.get("question_id", -1)) != -1:
            raise ValueError("question_id mismatch")
        selected = payload.get("selected_ref")
        allowed = {option["cell_ref"] for option in context["options"]} | {"NONE"}
        if selected not in allowed:
            raise ValueError("invented selected_ref")
        confidence = payload.get("confidence")
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("invalid confidence")
        if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
            raise ValueError("missing reason")
        result.update({"status": "VALID", "review": payload})
    except (TypeError, ValueError) as error:
        result["validation_error"] = str(error)
    return result


def consensus_decision(
    context: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    row = {
        "question_id": int(context.get("question_id", -1)),
        "baseline_ref": context["baseline_ref"],
        "suspect_score": context["suspect_score"],
    }
    if "audit_id" in context:
        row["audit_id"] = context["audit_id"]
    if first.get("status") != "VALID" or second.get("status") != "VALID":
        return {**row, "status": "REJECTED_INVALID_REVIEW"}
    a = first["review"]["selected_ref"]
    b = second["review"]["selected_ref"]
    row.update({
        "first_ref": a,
        "second_ref": b,
        "first_confidence": first["review"]["confidence"],
        "second_confidence": second["review"]["confidence"],
    })
    if a != b:
        row["status"] = "DISAGREEMENT"
    elif a == "NONE":
        row["status"] = "CONSENSUS_NONE"
    elif a == context["baseline_ref"]:
        row["status"] = "CONSENSUS_BASELINE"
    elif "LOW" in {row["first_confidence"], row["second_confidence"]}:
        row["status"] = "REJECTED_LOW_CONFIDENCE"
    else:
        row["status"] = "CONSENSUS_ALTERNATIVE"
        option = next(value for value in context["options"] if value["cell_ref"] == a)
        row["alternative"] = option["binding"]
    return row


def alternative_gate(context: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Apply non-LLM dimension checks before a source-level answer audit."""

    result = {
        "question_id": int(context.get("question_id", -1)),
        "consensus_status": decision.get("status"),
        "status": "REJECTED_NOT_ALTERNATIVE",
        "failures": [],
    }
    if "audit_id" in context:
        result["audit_id"] = context["audit_id"]
    if decision.get("status") not in {"CONSENSUS_ALTERNATIVE", "MAJORITY_ALTERNATIVE"}:
        return result
    alternative = decision["alternative"]
    baseline = context["baseline"]
    failures: list[str] = []
    for field in ("ticker", "year", "requested_scope", "target_unit"):
        if alternative.get(field) != baseline.get(field):
            failures.append(f"dimension_mismatch:{field}")

    allowed_scopes = {context["requested_scope"], "unspecified", "aggregated"}
    if alternative.get("document_scope") not in allowed_scopes:
        failures.append("opposite_document_scope")

    years = {
        int(value)
        for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", str(alternative.get("column_text", "")))
    }
    target_year = int(context["year"])
    if years:
        allowed_years = (
            {target_year - 1, target_year}
            if context["period_kind"] == "start"
            else {target_year}
        )
        if not years & allowed_years:
            failures.append("wrong_column_year")

    coverage = _metric_coverage(
        str(context["metric_text"]),
        str(alternative.get("row_text", "")),
        str(alternative.get("table_title", "")),
        str(alternative.get("column_text", "")),
    )
    if coverage < 0.5:
        failures.append("metric_coverage_lt_0.5")
    if float(alternative.get("period_score", 0.0)) < 0:
        failures.append("negative_period_score")
    if float(alternative.get("semantic_conflict_score", 0.0)) < -10:
        failures.append("semantic_conflict")

    baseline_answer = float(baseline["answer"])
    alternative_answer = float(alternative["answer"])
    answer_changed = abs(alternative_answer - baseline_answer) > max(
        1e-9, abs(baseline_answer) * 1e-9
    )
    if not answer_changed:
        failures.append("same_answer")
    result.update({
        "status": "READY_FOR_SOURCE_AUDIT" if not failures else "REJECTED_GATE",
        "failures": failures,
        "baseline_ref": context["baseline_ref"],
        "alternative_ref": decision["first_ref"],
        "baseline_answer": baseline_answer,
        "alternative_answer": alternative_answer,
        "metric_coverage": round(coverage, 4),
        "alternative": alternative,
    })
    return result


def majority_decision(
    context: dict[str, Any], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require an explicit 2-of-3 cell vote; invalid reviews do not vote."""

    valid = [row for row in reviews if row.get("status") == "VALID"]
    votes = Counter(row["review"]["selected_ref"] for row in valid)
    base = {
        "question_id": int(context.get("question_id", -1)),
        "baseline_ref": context["baseline_ref"],
        "votes": dict(votes),
        "valid_review_count": len(valid),
    }
    if "audit_id" in context:
        base["audit_id"] = context["audit_id"]
    if not votes:
        return {**base, "status": "REJECTED_NO_MAJORITY"}
    selected, count = votes.most_common(1)[0]
    if count < 2:
        return {**base, "status": "REJECTED_NO_MAJORITY"}
    supporters = [row for row in valid if row["review"]["selected_ref"] == selected]
    confidences = [row["review"]["confidence"] for row in supporters]
    base.update({
        "selected_ref": selected,
        "support": count,
        "supporting_models": [row["model"] for row in supporters],
        "supporting_confidences": confidences,
    })
    if selected == "NONE":
        base["status"] = "MAJORITY_NONE"
    elif selected == context["baseline_ref"]:
        base["status"] = "MAJORITY_BASELINE"
    elif confidences.count("LOW") >= 2:
        base["status"] = "REJECTED_LOW_CONFIDENCE"
    else:
        base["status"] = "MAJORITY_ALTERNATIVE"
        option = next(value for value in context["options"] if value["cell_ref"] == selected)
        base["alternative"] = option["binding"]
        # Match the shape consumed by alternative_gate.
        base["first_ref"] = selected
    return base
