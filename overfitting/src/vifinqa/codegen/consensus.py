"""Strict verification for LLM self-consistency challengers."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

from ..utils.viet_text import norm
from .arbitrate import agree
from .fact_resolver import resolve_requirement
from .generate import QuestionBundle, _run_validated


@dataclass(frozen=True)
class ConsensusDecision:
    accepted: bool
    reason: str


@dataclass(frozen=True)
class EnsembleChoice:
    candidate: dict | None
    reason: str


@dataclass(frozen=True)
class CodeEvidence:
    cells: frozenset[tuple[str, int, int, int]]
    sources: frozenset[tuple[str, int]]


def verify_consensus_candidate(
        candidate: dict, bundle: QuestionBundle, *,
        expected_samples: int = 5, min_votes: int = 4,
        min_confidence: float = 78.0) -> ConsensusDecision:
    """Require consensus, exact evidence coverage and a fresh CSV replay."""
    if candidate.get("status") != "ok":
        return ConsensusDecision(False, "candidate status is not ok")
    if not str(candidate.get("source") or "").startswith("llm_select"):
        return ConsensusDecision(False, "candidate is not selection-mode LLM output")

    votes = int(candidate.get("votes") or 0)
    n_ok = int(candidate.get("n_ok") or 0)
    if votes < min_votes or votes > expected_samples:
        return ConsensusDecision(
            False, f"consensus votes {votes}/{expected_samples} below threshold")
    if n_ok < votes or n_ok > expected_samples:
        return ConsensusDecision(False, f"invalid successful sample count {n_ok}")
    confidence = float(candidate.get("detail_conf") or 0.0)
    if confidence < min_confidence:
        return ConsensusDecision(
            False, f"selection confidence {confidence:.1f} below threshold")

    coverage = candidate.get("selection_evidence") or {}
    if not coverage.get("complete"):
        return ConsensusDecision(False, "canonical selection evidence is incomplete")
    required = int(coverage.get("required") or 0)
    covered = int(coverage.get("covered") or 0)
    if covered != required:
        return ConsensusDecision(
            False, f"selection evidence mismatch {covered}/{required}")

    route = bundle.route or {}
    plan = route.get("plan") or {}
    plan_op = str(plan.get("op") or "lookup")
    requirements = route.get("evidence_requirements") or []
    if plan_op in {"ratio", "margin", "ratio_times"} and len(requirements) < 2:
        return ConsensusDecision(
            False, "ratio lacks canonical evidence for both operands")
    if (plan_op == "difference" and float(candidate.get("answer") or 0.0) < 0
            and _uses_absolute_gap(route.get("question", ""))):
        return ConsensusDecision(
            False, "negative answer for an absolute difference question")

    query = str(candidate.get("pandas_query") or "")
    if not query:
        return ConsensusDecision(False, "candidate query is empty")
    replay = _run_validated(bundle, query)
    if replay.get("status") != "ok":
        return ConsensusDecision(
            False, f"fresh replay rejected: {replay.get('error') or replay.get('status')}")
    answer = float(candidate.get("answer") or 0.0)
    replay_value = float(replay.get("value") or 0.0)
    if not math.isclose(
            round(replay_value, 2), round(answer, 2),
            rel_tol=0.0, abs_tol=1e-9):
        return ConsensusDecision(
            False, f"fresh replay mismatch {replay_value} != {answer}")
    semantic = replay.get("semantic") or {}
    if not semantic.get("ok"):
        return ConsensusDecision(False, "fresh semantic validation failed")
    if semantic.get("warnings"):
        return ConsensusDecision(
            False, f"semantic warnings: {semantic.get('warnings')}")

    expected_vars = {
        (str(item["var"]), str(item["report_id"]), int(item["table_pos"]))
        for item in bundle.used_vars(query)
    }
    candidate_vars = {
        (str(item["var"]), str(item["report_id"]), int(item["table_pos"]))
        for item in candidate.get("used_vars") or []
    }
    if not expected_vars or candidate_vars != expected_vars:
        return ConsensusDecision(
            False, "candidate evidence variables do not match replay variables")

    return ConsensusDecision(
        True,
        f"verified consensus={votes}/{expected_samples} evidence={covered}/{required}",
    )


def verify_code_consensus_candidate(
        candidate: dict, bundle: QuestionBundle, *,
        expected_samples: int = 5, min_votes: int = 4) -> ConsensusDecision:
    """Verify a code-mode challenger before cross-run arbitration.

    This deliberately does not accept a candidate on its own. The final chooser
    also requires an independent run to produce the same answer from the same
    concrete source cells.
    """
    if candidate.get("status") != "ok":
        return ConsensusDecision(False, "candidate status is not ok")
    source = str(candidate.get("source") or "")
    if not source.startswith("llm") or source.startswith("llm_select"):
        return ConsensusDecision(False, "candidate is not code-mode LLM output")

    votes = int(candidate.get("votes") or 0)
    n_ok = int(candidate.get("n_ok") or 0)
    if votes < min_votes or votes > expected_samples:
        return ConsensusDecision(
            False, f"code consensus votes {votes}/{expected_samples} below threshold")
    if n_ok < votes or n_ok > expected_samples:
        return ConsensusDecision(False, f"invalid successful sample count {n_ok}")

    route = bundle.route or {}
    plan_op = str((route.get("plan") or {}).get("op") or "lookup")
    requirements = route.get("evidence_requirements") or []
    if plan_op in {"ratio", "margin", "ratio_times"} and len(requirements) < 2:
        return ConsensusDecision(False, "ratio lacks canonical evidence for both operands")
    if (plan_op == "difference" and float(candidate.get("answer") or 0.0) < 0
            and _uses_absolute_gap(route.get("question", ""))):
        return ConsensusDecision(
            False, "negative answer for an absolute difference question")

    replay_error = _verify_fresh_replay(candidate, bundle)
    if replay_error:
        return ConsensusDecision(False, replay_error)

    evidence = code_evidence(candidate, bundle)
    if not evidence.sources or not evidence.cells:
        return ConsensusDecision(False, "query has no traceable row/column evidence")

    required_ids = {
        str(item.get("requirement_id") or "") for item in requirements
        if item.get("requirement_id")
    }
    covered = set()
    for table in bundle.cands:
        source_key = (str(table["report_id"]), int(table["table_pos"]))
        if source_key in evidence.sources:
            covered.update(str(item) for item in table.get("requirement_hits") or [])
    missing = required_ids - covered
    if missing:
        return ConsensusDecision(
            False, f"code evidence misses canonical requirements: {sorted(missing)}")

    exact_cells = set()
    for requirement in requirements:
        resolved = resolve_requirement(
            requirement, bundle.tables,
            question=str(requirement.get("metric_label") or ""),
        )
        if resolved is None:
            return ConsensusDecision(
                False,
                "canonical requirement does not resolve to one exact cell: "
                f"{requirement.get('requirement_id')}",
            )
        exact_cells.add((
            str(resolved.report_id), int(resolved.table_pos),
            int(resolved.row), int(resolved.col),
        ))
    missing_cells = exact_cells - set(evidence.cells)
    if missing_cells:
        return ConsensusDecision(
            False,
            f"query misses exact canonical cells: {sorted(missing_cells)}",
        )

    return ConsensusDecision(
        True,
        f"verified code consensus={votes}/{expected_samples} "
        f"cells={len(evidence.cells)} exact={len(exact_cells)}",
    )


def choose_code_ensemble_candidate(
        base: dict,
        first: tuple[dict, ConsensusDecision, QuestionBundle],
        second: tuple[dict, ConsensusDecision, QuestionBundle], *,
        expected_samples: int = 5) -> EnsembleChoice:
    """Choose only independently reproduced code using identical source cells."""
    return choose_code_ensemble_candidates(
        base, [first, second], expected_samples=expected_samples,
        min_agreeing_runs=2)


def choose_code_ensemble_candidates(
        base: dict,
        runs: list[tuple[dict, ConsensusDecision, QuestionBundle]], *,
        expected_samples: int = 5,
        min_agreeing_runs: int = 2) -> EnsembleChoice:
    """Choose the strongest group reproduced from the same concrete cells."""
    accepted = []
    for candidate, decision, bundle in runs:
        if not decision.accepted:
            continue
        evidence = code_evidence(candidate, bundle)
        accepted.append((candidate, bundle, evidence.cells))
    if len(accepted) < min_agreeing_runs:
        return EnsembleChoice(None, "not enough verified code challengers")

    groups: list[list[tuple[dict, QuestionBundle, frozenset]]] = []
    for item in accepted:
        candidate, _bundle, cells = item
        for group in groups:
            leader, _leader_bundle, leader_cells = group[0]
            if cells == leader_cells and agree(
                    candidate.get("answer"), leader.get("answer")):
                group.append(item)
                break
        else:
            groups.append([item])

    qualified = [group for group in groups if len(group) >= min_agreeing_runs]
    if not qualified:
        return EnsembleChoice(
            None, "verified code challengers do not reproduce answer and cells")
    qualified.sort(key=len, reverse=True)
    if (len(qualified) > 1
            and len(qualified[0]) == len(qualified[1])):
        return EnsembleChoice(None, "conflicting verified code consensus groups")
    winning_group = qualified[0]

    if any(
            not ((bundle.route or {}).get("evidence_requirements") or [])
            for _candidate, bundle, _cells in winning_group):
        unanimous = all(
            int(candidate.get("votes") or 0) == expected_samples
            and int(candidate.get("n_ok") or 0) == expected_samples
            for candidate, _bundle, _cells in winning_group
        )
        if not unanimous:
            return EnsembleChoice(
                None, "uncanonicalized evidence requires unanimous agreeing runs")

    winner = max(
        (candidate for candidate, _bundle, _cells in winning_group),
        key=lambda row: (
            int(row.get("votes") or 0), int(row.get("n_ok") or 0),
            float(row.get("detail_conf") or 0.0),
        ),
    )
    if agree(base.get("answer"), winner.get("answer")):
        return EnsembleChoice(None, "verified code ensemble agrees with base")
    return EnsembleChoice(
        winner,
        f"{len(winning_group)}/{len(runs)} code runs agree on answer and source cells",
    )


def code_evidence(candidate: dict, bundle: QuestionBundle) -> CodeEvidence:
    """Resolve explicit query selectors to stable source-table cell identities."""
    query = str(candidate.get("pandas_query") or "")
    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(query, mode="exec")
        except SyntaxError:
            return CodeEvidence(frozenset(), frozenset())

    clauses: dict[str, list[tuple[str, str, object]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left, right = node.left, node.comparators[0]
            ref, value = _df_column(left), _constant(right)
            if ref is None or value is None:
                ref, value = _df_column(right), _constant(left)
            if ref is not None and value is not None:
                op = "eq" if isinstance(node.ops[0], ast.Eq) else "other"
                clauses.setdefault(ref[0], []).append((ref[1], op, value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"contains", "eq"} or not node.args:
                continue
            ref, value = _df_column(node.func.value), _constant(node.args[0])
            if ref is not None and value is not None:
                clauses.setdefault(ref[0], []).append(
                    (ref[1], node.func.attr, value))

    table_by_var = {str(table["var"]): table for table in bundle.tables}
    cells = set()
    sources = set()
    for var, selectors in clauses.items():
        table = table_by_var.get(var)
        df = bundle.dfs.get(var)
        if table is None or df is None:
            continue
        identities = [item for item in selectors if item[0] in {"row", "label", "code"}]
        periods = [item for item in selectors if item[0] in {"col", "col_name"}]
        if not identities or not periods:
            continue
        period_mask = None
        for selector in periods:
            mask = _selector_mask(df, selector)
            period_mask = mask if period_mask is None else (period_mask | mask)
        for selector in identities:
            mask = _selector_mask(df, selector)
            if period_mask is not None:
                mask = mask & period_mask
            for row in df.loc[mask, ["row", "col"]].drop_duplicates().itertuples(index=False):
                source = (str(table["report_id"]), int(table["table_pos"]))
                sources.add(source)
                cells.add((source[0], source[1], int(row.row), int(row.col)))
    return CodeEvidence(frozenset(cells), frozenset(sources))


def _verify_fresh_replay(candidate: dict, bundle: QuestionBundle) -> str:
    query = str(candidate.get("pandas_query") or "")
    if not query:
        return "candidate query is empty"
    replay = _run_validated(bundle, query)
    if replay.get("status") != "ok":
        return f"fresh replay rejected: {replay.get('error') or replay.get('status')}"
    answer = float(candidate.get("answer") or 0.0)
    replay_value = float(replay.get("value") or 0.0)
    if not math.isclose(
            round(replay_value, 2), round(answer, 2), rel_tol=0.0, abs_tol=1e-9):
        return f"fresh replay mismatch {replay_value} != {answer}"
    semantic = replay.get("semantic") or {}
    if not semantic.get("ok"):
        return "fresh semantic validation failed"
    if semantic.get("warnings"):
        return f"semantic warnings: {semantic.get('warnings')}"

    expected_vars = {
        (str(item["var"]), str(item["report_id"]), int(item["table_pos"]))
        for item in bundle.used_vars(query)
    }
    candidate_vars = {
        (str(item["var"]), str(item["report_id"]), int(item["table_pos"]))
        for item in candidate.get("used_vars") or []
    }
    if not expected_vars or candidate_vars != expected_vars:
        return "candidate evidence variables do not match replay variables"
    return ""


def _df_column(node: ast.AST) -> tuple[str, str] | None:
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript) or not isinstance(child.value, ast.Name):
            continue
        if not re.fullmatch(r"df\d+", child.value.id):
            continue
        column = _constant(child.slice)
        if isinstance(column, str):
            return child.value.id, column
    return None


def _constant(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return node.value
    return None


def _selector_mask(df, selector: tuple[str, str, object]):
    field, op, value = selector
    if field not in df.columns:
        return df.index.to_series().map(lambda _value: False)
    series = df[field]
    if field in {"row", "col"}:
        numeric = series.astype(float)
        try:
            return numeric == float(value)
        except (TypeError, ValueError):
            return numeric != numeric
    text = series.fillna("").astype(str)
    wanted = str(value)
    if op == "contains":
        return text.str.contains(wanted, case=False, regex=False, na=False)
    return text.map(norm) == norm(wanted)


def _uses_absolute_gap(question: str) -> bool:
    text = norm(question)
    directional = any(marker in text for marker in (
        "muc thay doi", "thay doi tu", "bien dong tu", "tang tu", "giam tu",
    ))
    return "chenh lech" in text and not directional


def choose_ensemble_candidate(
        base: dict, verified: list[tuple[dict, ConsensusDecision]], *,
        expected_samples: int = 5) -> EnsembleChoice:
    """Cross-run agreement wins; a lone challenger must be unanimous."""
    accepted = [candidate for candidate, decision in verified if decision.accepted]
    if not accepted:
        return EnsembleChoice(None, "no verified challenger")
    if len(accepted) >= 2:
        first, second = accepted[:2]
        if not agree(first.get("answer"), second.get("answer")):
            return EnsembleChoice(None, "verified challengers disagree")
        winner = max(
            (first, second),
            key=lambda row: (
                int(row.get("votes") or 0), int(row.get("n_ok") or 0),
                float(row.get("detail_conf") or 0.0),
            ),
        )
        if agree(base.get("answer"), winner.get("answer")):
            return EnsembleChoice(None, "cross-run consensus agrees with base")
        return EnsembleChoice(winner, "two verified runs agree")

    candidate = accepted[0]
    if agree(base.get("answer"), candidate.get("answer")):
        return EnsembleChoice(None, "verified challenger agrees with base")
    votes = int(candidate.get("votes") or 0)
    n_ok = int(candidate.get("n_ok") or 0)
    if votes == expected_samples and n_ok == expected_samples:
        return EnsembleChoice(candidate, "single verified challenger is unanimous")
    return EnsembleChoice(None, "single challenger is not unanimous")
