"""Run a cached, parallel, fail-closed audit factory over question queues.

The factory turns a 100/150-question queue into reproducible review packets:

* exact string and official-typed runtime replay;
* physical source-field audit when a compact ``q<ID>_source_cells.csv`` exists;
* AST result-dataflow checks for dead evidence frames;
* intent-to-source-key risk rules (initially gross-vs-net revenue);
* exact source-bundle/equal-answer clusters for representative adjudication;
* content-addressed cache so unchanged questions are not re-audited.

It never writes verdicts or mutates a submission.  ``ready`` means ready for a
human/source adjudication, never an automatic answer label.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_SUBMISSION = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"
DEFAULT_PUBLIC_QUEUE = ROOT / "build/v227_residual_public100_fused_v217.json"
DEFAULT_BATCH_QUEUE = ROOT / "build/v227_residual_batch150_fused_v217.json"
DEFAULT_OUTPUT = ROOT / "build/v229_question_audit_factory"
FACTORY_SCHEMA_VERSION = "1.0"

sys.path.insert(0, str(SCRIPTS))

from audit_unused_evidence_frames import audit_row as audit_unused_evidence_row  # noqa: E402
from grader_check import run_one  # noqa: E402
from audit_positional_semantics import (  # noqa: E402
    _evaluate_dataframe_aliases,
    _resolve_derived_frame,
    positional_reads,
    tokens as positional_tokens,
)


def fold(value: object) -> str:
    text = str(value or "").lower().replace("đ", "d")
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def evidence_hashes(submission_dir: Path, row: dict) -> list[dict]:
    result: list[dict] = []
    for evidence in row.get("evidence", []):
        relative = evidence.get("csv_path") if isinstance(evidence, dict) else None
        path = submission_dir / str(relative)
        result.append(
            {
                "variable": evidence.get("variable") if isinstance(evidence, dict) else None,
                "csv_path": relative,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return result


class DropDeadQid(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if targets and all(name == "qid" for name in targets):
            return None
        return self.generic_visit(node)


def normalized_ast_hash(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tree = DropDeadQid().visit(tree)
    ast.fix_missing_locations(tree)
    return sha256_bytes(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())


def source_bundle_hash(hashes: list[dict]) -> str | None:
    values = sorted(item["sha256"] for item in hashes if item.get("sha256"))
    return sha256_bytes("\n".join(values).encode()) if values else None


def tool_contract_hash() -> str:
    paths = [
        Path(__file__),
        SCRIPTS / "audit_question_source_fields.py",
        SCRIPTS / "audit_unused_evidence_frames.py",
        SCRIPTS / "grader_check.py",
    ]
    return sha256_bytes("\n".join(sha256_file(path) for path in paths).encode())


def question_fingerprint(
    row: dict, hashes: list[dict], contract_hash: str, mode: str = "deep"
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "schema": FACTORY_SCHEMA_VERSION,
                "row": row,
                "evidence": hashes,
                "contract": contract_hash,
                "mode": mode,
            }
        )
    )


def compact_source_previews(submission_dir: Path, qid: int) -> list[dict]:
    compact = submission_dir / "data" / f"q{qid}_source_cells.csv"
    if not compact.is_file():
        return []
    import csv

    with compact.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    previews: list[dict] = []
    for item in rows:
        table_ref = str(item.get("source_table", ""))
        label = ""
        entity_ancestor = ""
        physical_raw = None
        if "|" in table_ref:
            document, line = table_ref.rsplit("|", 1)
            source_dir = ROOT / "build" / "tables" / document
            source_path = source_dir / Path(str(item.get("source_csv", ""))).name
            if not source_path.is_file():
                alternatives = list(source_dir.glob(f"*line{line}.csv"))
                if len(alternatives) == 1:
                    source_path = alternatives[0]
            if source_path.is_file():
                with source_path.open(encoding="utf-8-sig", newline="") as source_handle:
                    physical = list(csv.reader(source_handle))
                try:
                    selected = physical[int(item["row_idx"]) + 1]
                    selected_index = int(item["row_idx"]) + 1
                    column = int(item["col_idx"])
                    physical_raw = selected[column] if column < len(selected) else None
                    textual = [cell for cell in selected if any(char.isalpha() for char in cell)]
                    label = max(textual, key=len) if textual else ""
                    for prior in reversed(physical[max(1, selected_index - 20) : selected_index]):
                        candidate = str(prior[0]).strip() if prior else ""
                        normalized_candidate = fold(candidate)
                        if "cong ty" in normalized_candidate:
                            entity_ancestor = candidate
                            break
                except (KeyError, TypeError, ValueError, IndexError):
                    pass
        previews.append(
            {
                "metric_key": item.get("metric_key"),
                "source_table": table_ref,
                "label": label,
                "entity_ancestor": entity_ancestor,
                "manifest_raw": item.get("raw"),
                "physical_raw": physical_raw,
            }
        )
    return previews


def source_metric_keys(previews: list[dict]) -> list[str]:
    return sorted(
        {
            str(item.get("metric_key", ""))
            for item in previews
            if item.get("metric_key")
        }
    )


def intent_source_flags(
    question: str, metric_keys: list[str], previews: list[dict] | None = None
) -> list[dict]:
    normalized = fold(question)
    flags: list[dict] = []
    asks_sales = "doanh thu ban hang va cung cap dich vu" in normalized
    asks_net = "doanh thu thuan" in normalized
    if asks_sales and not asks_net and "kqkd:10" in metric_keys:
        flags.append(
            {
                "family": "gross_question_net_source",
                "severity": "answer_change_possible",
                "question_intent": "kqkd:01",
                "declared_source": "kqkd:10",
            }
        )
    if asks_net and "kqkd:01" in metric_keys:
        flags.append(
            {
                "family": "net_question_gross_source",
                "severity": "answer_change_possible",
                "question_intent": "kqkd:10",
                "declared_source": "kqkd:01",
            }
        )
    previews = previews or []
    transaction_named = bool(
        re.search(
            r"\b(?:mua|ban|vay|phai thu|phai tra|giao dich|doanh thu|chi phi)"
            r"[^.?]{0,80}\b(?:voi|tu|cho)\s+cong ty\s+(?!me\b)",
            normalized,
        )
    ) and "doi voi cong ty" not in normalized
    project_named = bool(
        re.search(r"\b(?:gia tri|chi phi)[^.?]{0,35}\bdu an\b", normalized)
        or re.search(r"\bdu an\b[^.?]{0,50}\b(?:gia tri|chi phi)\b", normalized)
    )
    named_role = transaction_named or project_named
    labels = [fold(item.get("label", "")) for item in previews if item.get("label")]
    ancestors = [
        fold(item.get("entity_ancestor", ""))
        for item in previews
        if item.get("entity_ancestor")
    ]
    if named_role and labels:
        identity_bound = any(
            label in normalized and ("cong ty" in label or "du an" in label)
            for label in labels + ancestors
        )
        question_terms = positional_tokens(normalized)
        overlaps = []
        for label in labels:
            label_terms = positional_tokens(label)
            union = question_terms | label_terms
            overlaps.append(len(question_terms & label_terms) / len(union) if union else 1.0)
        if not identity_bound and max(overlaps, default=0.0) < 0.40:
            flags.append(
                {
                    "family": "named_entity_context_missing",
                    "severity": "answer_change_possible",
                    "maximum_label_jaccard": round(max(overlaps, default=0.0), 4),
                    "source_labels": labels,
                    "entity_ancestors": ancestors,
                }
            )
    return flags


def positional_fallback(row: dict, submission_dir: Path) -> dict:
    """Resolve constant iloc reads when no compact source manifest exists."""

    code = str(row.get("pandas_query", ""))
    reads = positional_reads(code)
    if not reads:
        return {"status": "missing_compact_manifest", "reason": "no_constant_iloc_reads"}
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for evidence in row.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        variable = evidence.get("variable")
        relative = evidence.get("csv_path")
        if not isinstance(variable, str) or not isinstance(relative, str):
            continue
        path = submission_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        frames[variable] = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
    if missing:
        return {"status": "error", "missing": missing[:10]}
    env = _evaluate_dataframe_aliases(code, frames)
    question_terms = positional_tokens(row.get("question", ""))
    resolved: list[dict] = []
    unresolved: list[dict] = []
    for alias, root, row_idx, col_idx in reads:
        frame = env.get(alias)
        if frame is None:
            frame = _resolve_derived_frame(alias, env)
        if frame is None:
            frame = env.get(root)
        if not isinstance(frame, pd.DataFrame):
            unresolved.append({"alias": alias, "root": root, "reason": "frame_unresolved"})
            continue
        normalized_row = row_idx if row_idx >= 0 else len(frame) + row_idx
        normalized_col = col_idx if col_idx >= 0 else len(frame.columns) + col_idx
        if not (0 <= normalized_row < len(frame) and 0 <= normalized_col < len(frame.columns)):
            unresolved.append(
                {
                    "alias": alias,
                    "root": root,
                    "row": row_idx,
                    "column": col_idx,
                    "reason": "coordinate_out_of_bounds",
                    "shape": list(frame.shape),
                }
            )
            continue
        physical_row = [str(value) for value in frame.iloc[normalized_row].tolist()]
        textual = [value for value in physical_row if any(char.isalpha() for char in value)]
        label = max(textual, key=len) if textual else ""
        row_terms = positional_tokens(label)
        overlap = len(question_terms & row_terms) / len(question_terms | row_terms) if question_terms | row_terms else 1.0
        resolved.append(
            {
                "alias": alias,
                "root": root,
                "row": row_idx,
                "column": col_idx,
                "label": label,
                "raw": str(frame.iloc[normalized_row, normalized_col]),
                "term_jaccard": round(overlap, 4),
            }
        )
    if unresolved:
        return {"status": "positional_failure", "reads": resolved, "unresolved": unresolved}
    minimum = min((item["term_jaccard"] for item in resolved), default=1.0)
    return {
        "status": "positional_risk" if minimum < 0.08 else "positional_pass",
        "reads": resolved,
        "minimum_term_jaccard": minimum,
        "claim_limit": "Lexical/coordinate fallback only; section and role semantics still need adjudication.",
    }


def numeric_runtime(
    row: dict, submission_dir: Path, *, official: bool
) -> dict:
    paths = {
        item["variable"]: str((submission_dir / item["csv_path"]).resolve())
        for item in row.get("evidence", [])
        if isinstance(item, dict)
        and isinstance(item.get("variable"), str)
        and isinstance(item.get("csv_path"), str)
    }
    missing = [value for value in paths.values() if not Path(value).is_file()]
    if missing:
        return {"status": "missing_evidence", "missing": missing[:10]}
    if not paths:
        return {"status": "no_evidence"}
    try:
        executed = run_one(
            str(row.get("pandas_query", "")),
            paths,
            official=official,
            return_namespace=not official,
        )
        if official:
            result = executed
            namespace = {}
        else:
            result, namespace = executed
        numeric = float(result)
        stored = float(row["answer"])
        delta = numeric - stored
        payload = {
            "status": "pass" if abs(delta) <= 0.01 + 1e-9 else "mismatch",
            "result": numeric,
            "stored_answer": stored,
            "delta": delta,
        }
        if namespace:
            scalars = {}
            for name, value in namespace.items():
                if not re.fullmatch(r"v\d+", name):
                    continue
                try:
                    scalars[name] = float(value)
                except (TypeError, ValueError):
                    continue
            payload["scalars"] = scalars
        return payload
    except Exception as exc:  # audit packet must record, not crash the run
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _simple_scalar_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
    ):
        return _simple_scalar_name(node.args[0])
    return None


def arithmetic_invariant_flags(question: str, code: str, scalars: dict[str, float]) -> list[dict]:
    normalized = fold(question)
    share_intent = "ty trong" in normalized and "tong" in normalized
    if not share_intent or not scalars:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    flags: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        numerator = _simple_scalar_name(node.left)
        denominator = _simple_scalar_name(node.right)
        if not numerator or not denominator or (numerator, denominator) in seen:
            continue
        seen.add((numerator, denominator))
        if numerator not in scalars or denominator not in scalars or scalars[denominator] == 0:
            continue
        ratio = abs(scalars[numerator] / scalars[denominator])
        if ratio > 1.05:
            flags.append(
                {
                    "family": "component_exceeds_total",
                    "severity": "source_error_likely",
                    "expression": f"{numerator}/{denominator}",
                    "ratio": ratio,
                }
            )
    return flags


def run_physical_audit(
    qid: int,
    submission_dir: Path,
    physical_dir: Path,
) -> dict:
    compact = submission_dir / "data" / f"q{qid}_source_cells.csv"
    if not compact.is_file():
        rows = load_json(submission_dir / "submission.json")
        row = next(item for item in rows if int(item["id"]) == qid)
        return positional_fallback(row, submission_dir)
    output = physical_dir / f"q{qid}.json"
    command = [
        sys.executable,
        str(SCRIPTS / "audit_question_source_fields.py"),
        "--qid",
        str(qid),
        "--submission",
        str(submission_dir),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        return {
            "status": "error",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-1000:],
            "stdout": completed.stdout[-1000:],
        }
    payload = load_json(output)
    counts = payload.get("counts", {})
    return {
        "status": "pass" if int(counts.get("field_failures", 0)) == 0 else "failure",
        "counts": counts,
        "missing_program_refs": payload.get("missing_program_refs", []),
        "duplicate_keys": payload.get("duplicate_keys", []),
        "field_failures": payload.get("field_failures", []),
        "artifact": str(output.relative_to(ROOT)),
    }


def priority(record: dict) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if record.get("intent_source_flags"):
        score += 100
        reasons.append("intent_source_mismatch")
    if record.get("arithmetic_invariant_flags"):
        score += 100
        reasons.append("arithmetic_invariant_violation")
    runtime_states = {item.get("status") for item in record.get("runtime", {}).values()}
    if runtime_states & {"error", "mismatch", "missing_evidence", "no_evidence"}:
        score += 95
        reasons.append("runtime_failure")
    physical = record.get("physical_audit", {}).get("status")
    if physical in {"failure", "error", "positional_failure"}:
        score += 90
        reasons.append("physical_failure")
    elif physical == "positional_risk":
        score += 80
        reasons.append("low_overlap_positional_read")
    elif physical == "missing_compact_manifest":
        score += 60
        reasons.append("manual_fragment_review")
    unused = record.get("unused_evidence")
    if unused:
        count = len(unused.get("unused_indices", []))
        score += min(30, 10 + count)
        reasons.append(f"unused_evidence:{count}")
    return score, reasons


def triage(record: dict, mode: str = "deep") -> str:
    if mode == "fast":
        if record["intent_source_flags"]:
            return "intent_source_mismatch"
        if record.get("unused_evidence"):
            return "cleanup_candidate"
        return "fast_screen_clear"
    runtimes = record["runtime"]
    if any(item.get("status") != "pass" for item in runtimes.values()):
        return "runtime_failure"
    if record["physical_audit"].get("status") in {
        "error",
        "failure",
        "positional_failure",
    }:
        return "physical_failure"
    if record["physical_audit"].get("status") == "positional_risk":
        return "positional_risk"
    if record["intent_source_flags"]:
        return "intent_source_mismatch"
    if record.get("arithmetic_invariant_flags"):
        return "arithmetic_invariant_violation"
    if record.get("unused_evidence"):
        return "cleanup_candidate"
    if record["physical_audit"].get("status") == "missing_compact_manifest":
        return "manual_fragment_review"
    return "ready_for_semantic_adjudication"


def audit_one(
    row: dict,
    submission_dir: Path,
    cache_dir: Path,
    physical_dir: Path,
    contract_hash: str,
    use_cache: bool,
    mode: str,
) -> dict:
    started = time.perf_counter()
    qid = int(row["id"])
    hashes = evidence_hashes(submission_dir, row)
    fingerprint = question_fingerprint(row, hashes, contract_hash, mode=mode)
    cache_path = cache_dir / f"q{qid}_{fingerprint}.json"
    if use_cache and cache_path.is_file():
        cached = load_json(cache_path)
        cached["cache_hit"] = True
        return cached

    previews = compact_source_previews(submission_dir, qid)
    metrics = source_metric_keys(previews)
    unused = audit_unused_evidence_row(row)
    runtime = (
        {
            "string": numeric_runtime(row, submission_dir, official=False),
            "official_typed": numeric_runtime(row, submission_dir, official=True),
        }
        if mode == "deep"
        else {"fast": {"status": "skipped_fast"}}
    )
    record = {
        "id": qid,
        "question": row.get("question"),
        "answer": row.get("answer"),
        "fingerprint": fingerprint,
        "cache_hit": False,
        "relevant_docs": row.get("relevant_docs", []),
        "relevant_tables": row.get("relevant_tables", []),
        "evidence": hashes,
        "source_bundle_hash": source_bundle_hash(hashes),
        "normalized_ast_hash": normalized_ast_hash(str(row.get("pandas_query", ""))),
        "source_metric_keys": metrics,
        "compact_source_previews": previews,
        "intent_source_flags": intent_source_flags(
            str(row.get("question", "")), metrics, previews
        ),
        "unused_evidence": unused,
        "runtime": runtime,
        "arithmetic_invariant_flags": arithmetic_invariant_flags(
            str(row.get("question", "")),
            str(row.get("pandas_query", "")),
            runtime.get("string", {}).get("scalars", {}),
        ),
        "physical_audit": (
            run_physical_audit(qid, submission_dir, physical_dir)
            if mode == "deep"
            else {"status": "skipped_fast"}
        ),
    }
    record["triage"] = triage(record, mode=mode)
    record["priority_score"], record["priority_reasons"] = priority(record)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    cache_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def build_clusters(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for record in records:
        bundle = record.get("source_bundle_hash")
        if bundle is None:
            continue
        grouped[(bundle, float(record["answer"]))].append(record)
    clusters: list[dict] = []
    for (bundle, answer), members in grouped.items():
        if len(members) < 2:
            continue
        ast_hashes = {item.get("normalized_ast_hash") for item in members}
        table_sets = {tuple(sorted(item.get("relevant_tables", []))) for item in members}
        clusters.append(
            {
                "source_bundle_hash": bundle,
                "answer": answer,
                "question_ids": sorted(item["id"] for item in members),
                "size": len(members),
                "normalized_ast_equal": len(ast_hashes) == 1,
                "relevant_table_sets_equal": len(table_sets) == 1,
                "claim_limit": "Candidate representative cluster; semantic intent still requires review.",
            }
        )
    return sorted(clusters, key=lambda item: (-item["size"], item["question_ids"]))


def render_dashboard(summary: dict, records: list[dict], clusters: list[dict]) -> str:
    lines = [
        "# Question Audit Factory",
        "",
        f"- Questions: **{summary['question_count']}**",
        f"- Wall time: **{summary['wall_seconds']:.2f}s**",
        f"- Throughput: **{summary['questions_per_second']:.2f} q/s**",
        f"- Cache hits: **{summary['cache_hits']}**",
        f"- Candidate equivalence clusters: **{len(clusters)}**",
        "",
        "## Triage",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(summary["triage_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Queue",
            "",
        "| qid | priority | triage | runtime | physical | unused frames | intent flags |",
        "|---:|---:|---|---|---|---:|---:|",
        ]
    )
    for record in records:
        runtime = "/".join(item["status"] for item in record["runtime"].values())
        unused_count = (
            len(record["unused_evidence"].get("unused_indices", []))
            if record.get("unused_evidence")
            else 0
        )
        lines.append(
            f"| q{record['id']} | {record['priority_score']} | {record['triage']} | {runtime} | "
            f"{record['physical_audit'].get('status')} | {unused_count} | "
            f"{len(record['intent_source_flags'])} |"
        )
    lines.extend(["", "## Exact source-bundle clusters", ""])
    if not clusters:
        lines.append("- None in this run.")
    for cluster in clusters:
        lines.append(
            f"- {cluster['question_ids']} · AST_equal={cluster['normalized_ast_equal']} · "
            f"tables_equal={cluster['relevant_table_sets_equal']}"
        )
    lines.extend(
        [
            "",
            "> This factory never auto-writes verdicts or changes answers. Every mutation remains source-adjudicated and fail-closed.",
            "",
        ]
    )
    return "\n".join(lines)


def queue_rows(path: Path) -> list[dict]:
    payload = load_json(path)
    rows = payload.get("queue")
    if not isinstance(rows, list):
        raise ValueError(f"queue missing from {path}")
    return rows


def parse_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--queue", choices=("public", "batch", "all"), default="public")
    parser.add_argument("--public-queue", type=Path, default=DEFAULT_PUBLIC_QUEUE)
    parser.add_argument("--batch-queue", type=Path, default=DEFAULT_BATCH_QUEUE)
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=("fast", "deep"), default="deep")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-closed", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    submission_dir = args.submission.resolve()
    output_dir = args.out.resolve()
    cache_dir = output_dir / "cache"
    physical_dir = output_dir / "physical"
    for path in (output_dir, cache_dir, physical_dir):
        path.mkdir(parents=True, exist_ok=True)

    submission_rows = load_json(submission_dir / "submission.json")
    by_id = {int(row["id"]): row for row in submission_rows}
    requested_ids = parse_ids(args.ids)
    if requested_ids:
        ordered_ids = requested_ids
    elif args.queue == "all":
        ordered_ids = [int(row["id"]) for row in submission_rows]
    else:
        primary = queue_rows(args.public_queue.resolve())
        if args.queue == "batch":
            primary = queue_rows(args.batch_queue.resolve())
        ordered_ids = [int(item["id"]) for item in primary]

    ledger = load_json(args.ledger.resolve()).get("verdicts", {})
    closed = {int(value) for value in ledger}
    seen: set[int] = set()
    ordered_ids = [
        qid
        for qid in ordered_ids
        if qid in by_id
        and qid not in seen
        and not seen.add(qid)
        and (args.include_closed or qid not in closed)
    ][: max(0, args.limit)]

    contract_hash = tool_contract_hash()
    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                audit_one,
                by_id[qid],
                submission_dir,
                cache_dir,
                physical_dir,
                contract_hash,
                not args.no_cache,
                args.mode,
            ): qid
            for qid in ordered_ids
        }
        completed = {future: future.result() for future in concurrent.futures.as_completed(futures)}
    by_result_id = {record["id"]: record for record in completed.values()}
    records = [by_result_id[qid] for qid in ordered_ids]
    clusters = build_clusters(records)
    cluster_ids = {qid for cluster in clusters for qid in cluster["question_ids"]}
    for record in records:
        if record["id"] in cluster_ids:
            record["priority_score"] += 8
            record["priority_reasons"].append("equivalence_cluster")
    priority_records = sorted(records, key=lambda item: (-item["priority_score"], item["id"]))
    wall = time.perf_counter() - started
    triage_counts = Counter(record["triage"] for record in records)
    summary = {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "submission": str(submission_dir),
        "queue": args.queue,
        "mode": args.mode,
        "question_count": len(records),
        "closed_skipped": len(closed) if not args.include_closed else 0,
        "workers": max(1, args.workers),
        "wall_seconds": round(wall, 4),
        "questions_per_second": round(len(records) / wall, 4) if wall else 0.0,
        "cache_hits": sum(bool(record.get("cache_hit")) for record in records),
        "triage_counts": dict(sorted(triage_counts.items())),
        "cluster_count": len(clusters),
        "contract_hash": contract_hash,
        "claim_limit": "No automatic verdict or mutation authority.",
    }
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    (output_dir / "clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DASHBOARD.md").write_text(
        render_dashboard(summary, records, clusters),
        encoding="utf-8",
    )
    (output_dir / "PRIORITY_QUEUE.json").write_text(
        json.dumps(
            [
                {
                    "id": item["id"],
                    "priority_score": item["priority_score"],
                    "priority_reasons": item["priority_reasons"],
                    "triage": item["triage"],
                    "question": item["question"],
                }
                for item in priority_records
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
