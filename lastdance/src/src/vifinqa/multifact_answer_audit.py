"""Rank and materialize answer-blind audits for multi-fact solver components."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .l1_answer_audit import cell_ref, enrich_options, suspect_score
from .l1_fact_layer import normalize


ILOC_RE_TEMPLATE = r"\b{variable}\.iloc\[\s*(\d+)\s*,\s*(\d+)\s*\]"


def component_identity(row: dict[str, Any]) -> tuple[int, int, str]:
    """Return the stable question/item/component identity used by set solvers."""

    return (
        int(row.get("question_id", -1)),
        int(row.get("item", 0)),
        str(row.get("component") or row.get("symbol") or "value"),
    )


def audit_id_for(identity: tuple[int, int, str]) -> str:
    question_id, item, component = identity
    return f"q{question_id:04d}_i{item:02d}_{component}"


def metric_text_for(row: dict[str, Any]) -> str:
    return str(
        row.get("component_metric")
        or row.get("metric_text")
        or row.get("metric")
        or ""
    )


def rank_component_suspects(
    candidate_rows: Iterable[dict[str, Any]],
    *,
    include_manual: bool = False,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[component_identity(row)].append(row)

    ranking = []
    for identity, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("candidate_rank", 999)))
        tops = [row for row in rows if int(row.get("candidate_rank", 0)) == 1]
        if len(tops) != 1:
            raise ValueError(f"{audit_id_for(identity)}: expected one rank-1 fact")
        top = dict(tops[0])
        metric_text = metric_text_for(top)
        top["metric_text"] = metric_text
        alternatives = [row for row in rows if int(row.get("candidate_rank", 0)) != 1]
        score, reasons = suspect_score(top, alternatives)
        if score < 0 and include_manual and top.get("selection_source") == "manual":
            # A manual override records how a source cell was selected; it is
            # not ground truth.  Release audits keep the historical default of
            # skipping these rows, while broad tuning can explicitly re-audit
            # them against the same answer-blind candidate evidence.
            score, reasons = suspect_score(
                {**top, "selection_source": "automatic"}, alternatives
            )
            reasons = ["manual_reaudit", *[r for r in reasons if r != "automatic"]]
        if score < 0:
            continue
        question_id, item, component = identity
        ranking.append({
            "audit_id": audit_id_for(identity),
            "question_id": question_id,
            "item": item,
            "component": component,
            "suspect_score": score,
            "reasons": reasons,
            "baseline_ref": cell_ref(top),
            "baseline_answer": top["answer"],
            "confidence": top.get("confidence"),
            "table_kind": top.get("table_kind"),
            "question": top.get("question"),
            "metric_text": metric_text,
        })
    ranking.sort(key=lambda row: (-row["suspect_score"], row["audit_id"]))
    return ranking, grouped


def build_component_contexts(
    candidate_rows: list[dict[str, Any]],
    database: Path,
    *,
    offset: int = 0,
    limit: int = 96,
    option_limit: int = 5,
    include_manual: bool = False,
    unique_questions: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranking, grouped = rank_component_suspects(
        candidate_rows, include_manual=include_manual
    )
    if unique_questions:
        distinct = []
        seen: set[int] = set()
        for suspect in ranking:
            question_id = int(suspect.get("question_id", -1))
            if question_id in seen:
                continue
            seen.add(question_id)
            distinct.append(suspect)
        selected = distinct[offset:offset + limit]
    else:
        selected = ranking[offset:offset + limit]
    contexts = []
    for audit_rank, suspect in enumerate(selected, offset + 1):
        identity = (
            int(suspect.get("question_id", -1)),
            int(suspect["item"]),
            str(suspect["component"]),
        )
        rows = grouped[identity]
        baseline = next(row for row in rows if int(row.get("candidate_rank", 0)) == 1)
        options = enrich_options(database, rows, limit=option_limit)
        if suspect["baseline_ref"] not in {option["cell_ref"] for option in options}:
            raise ValueError(f"{suspect['audit_id']}: baseline cell absent from options")
        question_norm = normalize(str(suspect["question"]))
        contexts.append({
            **suspect,
            "audit_rank": audit_rank,
            "ticker": baseline["ticker"],
            "year": int(baseline["year"]),
            "requested_scope": baseline["requested_scope"],
            "period_kind": (
                "start" if "dau nam" in question_norm or "dau ky" in question_norm
                else "end_or_flow"
            ),
            "target_unit": baseline["target_unit"],
            "baseline": baseline,
            "options": options,
        })
    return ranking, contexts


def rebase_context_on_submission(
    context: dict[str, Any],
    submission_item: dict[str, Any],
    evidence_root: Path,
    connection: sqlite3.Connection | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve an audit context's baseline from the current champion artifact.

    Historical candidate registries are intentionally immutable, so their
    rank-1 cell can lag behind a cumulative leaderboard champion.  L3 evidence
    paths carry the stable component identity (for example
    ``q0859_i2_a.csv``).  We use that path, the champion Pandas ``iloc`` and
    the actual CSV cell to identify the matching option without consulting the
    saved answer.
    """

    audit_id = str(context["audit_id"])
    audit_match = re.fullmatch(r"q(\d+)_i(\d+)_([A-Za-z0-9]+)", audit_id)
    if audit_match is None:
        raise ValueError(f"Invalid audit_id {audit_id}")
    question_id = int(context.get("question_id", audit_match.group(1) if audit_match else -1))
    item_number = int(context.get("item", audit_match.group(2)))
    component = str(context.get("component") or audit_match.group(3))
    expected_names = {
        f"{audit_id}.csv",
        f"q{question_id:04d}_i{item_number}_{component}.csv",
    }
    if item_number == 0:
        expected_names.add(f"q{question_id:04d}_{component}_evidence.csv")
        # L5 plans use compact symbol evidence names such as q0362_f1.csv,
        # while L2 formulas retain the older q0656_a_evidence.csv form.
        expected_names.add(f"q{question_id:04d}_{component}.csv")
    matches = [
        evidence
        for evidence in submission_item.get("evidence", [])
        if Path(str(evidence.get("csv_path", ""))).name in expected_names
    ]
    audit = {
        "audit_id": audit_id,
        "question_id": int(context.get("question_id", -1)),
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
            ILOC_RE_TEMPLATE.format(variable=re.escape(variable)),
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
    data_index = row_index + 1  # pandas consumes the generated col_* header.
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
    if len(option_matches) > 1 and connection is not None:
        # Same row/column/value can occur in both a current report and its
        # comparative copy.  Match the complete champion evidence table to the
        # immutable corpus grid to disambiguate without using the answer.
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
                [str(value) for value in row]
                + [""] * (width - len(row))
                for row in grid
            ]
            if padded == data_rows:
                exact.append(option)
        option_matches = exact
    if len(option_matches) != 1:
        audit.update(
            {
                "reason": f"champion cell matches {len(option_matches)} audit options",
                "variable": variable,
                "row_index": row_index,
                "column_index": column_index,
            }
        )
        return None, audit
    selected = option_matches[0]
    rebased = dict(context)
    rebased.update(
        {
            "baseline_ref": selected["cell_ref"],
            "baseline_answer": selected["binding"]["answer"],
            "baseline": selected["binding"],
            "registry_baseline_ref": context["baseline_ref"],
            "baseline_source": "champion_submission",
        }
    )
    audit.update(
        {
            "status": "REBASED",
            "champion_baseline_ref": selected["cell_ref"],
            "changed_from_registry": selected["cell_ref"] != context["baseline_ref"],
            "variable": variable,
            "row_index": row_index,
            "column_index": column_index,
        }
    )
    return rebased, audit
