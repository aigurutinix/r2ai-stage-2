"""Combine audit, source-cell and history evidence into one review queue.

The engine deliberately does *not* decide whether a hidden answer is correct.
It makes the expensive manual step smaller and reproducible:

* trace the physical cells (writers) read by a Pandas program;
* inspect the program operations (reader);
* combine independent static-audit signals without treating them as an oracle;
* expose mechanically plausible counterfactual calculations;
* remember source-verified reviews so dead hypotheses do not return next run.

No leaderboard delta is used to label an individual question.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from kingpro.experiments import oracle_trust


AUDIT_WEIGHTS = {
    "direct_units": 4.0,
    "unresolved_source": 5.0,
    "source_cell_semantics": 4.0,
    "audited_period": 3.5,
    "period": 3.0,
    "comparison_sign": 2.0,
    "filter_cardinality": 3.5,
    "semantic_alignment": 3.0,
    "metric_codes": 2.5,
    "positional": 3.0,
    "legacy_terminal": 2.0,
    "missing_panel_operands": 4.0,
    "structural": 2.5,
    "selector_coverage": 1.5,
    "program_intent": 3.0,
    "result_shape": 2.5,
    "cross_task": 1.0,
}

REVIEWED_VERDICTS = {
    "current_answer_confirmed",
    "false_positive",
    "source_confirmed",
    "source_confirmed_keep_baseline",
    "source_confirmed_no_change",
    "source_confirmed_fix",
    "source_confirmed_scope_repair",
    "source_formula_confirmed",
    "hypothesis_rejected",
    "counterfactual_rejected",
    "not_actionable",
    "annotation_ambiguous",
    "ambiguous_keep_baseline",
}


def _canonical_verdict(value: Any) -> str:
    """Normalize durable-review spelling without weakening oracle trust.

    Older log entries used both ``source-confirmed-fix`` and
    ``source_confirmed_fix``.  Treating the hyphenated form as unreviewed made
    already closed cases return to every forensic queue.
    """

    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _fold(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def detect_intents(question: str) -> list[str]:
    """Return conservative semantic intents found in a Vietnamese question."""

    text = _fold(question)
    patterns = {
        "sum": r"\b(tong|cong lai|luy ke)\b",
        "difference": r"\b(chenh lech|khac biet|cao hon|thap hon|giam bao nhieu|tang bao nhieu)\b",
        "ratio": r"\b(ty le|ty trong|he so|tren doanh thu|tren tai san|tren von|bao nhieu lan)\b",
        "growth": r"\b(tang truong|muc tang|muc giam|thay doi.*phan tram)\b",
        "extreme": r"\b(cao nhat|thap nhat|lon nhat|nho nhat|toi da|toi thieu)\b",
        "mean": r"\b(trung binh|binh quan)\b",
        "median": r"\btrung vi\b",
        "count": r"\b(co bao nhieu|so luong)\b",
        "identity": r"\b(nam nao|cong ty nao|doanh nghiep nao|ma nao)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def parse_btc_number(raw: Any) -> Optional[float]:
    """Parse the common ViFinQA/BTC numeric cell formats.

    ``None`` and blank/dash cells stay missing.  Parentheses retain their
    accounting sign.  The parser mirrors the submission runtime closely but is
    intentionally read-only and never substitutes missing with zero.
    """

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    text = str(raw).strip().replace("\u00a0", " ")
    if text in {"", "-", "–", "—", "N/A", "n/a"}:
        return None
    if " " in text:
        text = text.split()[0]
    if ")(" in text:
        text = text.split(")(", 1)[0] + ")"
    negative = text.startswith("(") and text.endswith(")")
    text = (
        text.replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace("$", "")
    )
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    elif "." in text:
        tail = text.rsplit(".", 1)[-1]
        if len(tail) == 3:
            text = text.replace(".", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if negative:
        value = -abs(value)
    return value if math.isfinite(value) else None


def _answer_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _version(path: Path) -> Optional[int]:
    match = re.search(r"_v(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else None


def _audit_kind(path: Path) -> str:
    name = path.stem.casefold()
    patterns = (
        ("unresolved_source", "unresolved_source"),
        ("source_cell_semantics", "source_cell_semantics"),
        ("audited_period", "audited_period"),
        ("comparison_sign", "comparison_sign"),
        ("filter_cardinality", "filter_cardinality"),
        ("semantic", "semantic_alignment"),
        ("metric_codes", "metric_codes"),
        ("positional", "positional"),
        ("legacy_terminal", "legacy_terminal"),
        ("missing_panel_operands", "missing_panel_operands"),
        ("structural", "structural"),
        ("selector_coverage", "selector_coverage"),
        ("full_intent", "program_intent"),
        ("program_intent", "program_intent"),
        ("result_shape", "result_shape"),
        ("cross_task", "cross_task"),
        ("direct_units", "direct_units"),
        ("period", "period"),
    )
    for fragment, kind in patterns:
        if fragment in name:
            return kind
    return "other"


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finding_weight(kind: str, item: Mapping[str, Any]) -> float:
    """Convert report-specific evidence to comparable risk weight.

    Positional and semantic reports are validation scores (1 is good), while
    most other reports emit risk scores (larger is worse).  Treating every
    ``findings`` row as an error reverses those two validators and badly
    distorts the queue.
    """

    base = AUDIT_WEIGHTS.get(kind, 0.2)
    if kind in {"semantic_alignment", "positional"}:
        agreement = min(1.0, max(0.0, _float_or(item.get("score"), 1.0)))
        return base * (1.0 - agreement)
    if kind == "selector_coverage":
        coverage = min(1.0, max(0.0, _float_or(item.get("coverage_ratio"), 1.0)))
        multiplier = 1.5 if item.get("review_priority") == "high" else 0.75
        return base * (1.0 - coverage) * multiplier
    if kind == "filter_cardinality":
        return base * min(1.5, _float_or(item.get("risk"), 0.0) / 5.0)
    if kind == "cross_task":
        return base * min(1.5, _float_or(item.get("risk_score"), 0.0) / 15.0)
    if kind == "program_intent":
        return base * min(1.5, _float_or(item.get("score"), 0.0) / 6.0)
    if kind == "result_shape":
        return base * min(1.5, _float_or(item.get("score"), 0.0) / 5.0)
    if kind == "structural":
        return base * min(1.5, _float_or(item.get("score"), 0.0) / 5.0)
    if item.get("high_priority"):
        return base * 1.5
    return base


def _finding_reasons(kind: str, item: Mapping[str, Any]) -> list[str]:
    reasons = item.get("reasons") or item.get("missing_operands") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    output = [str(value) for value in reasons]
    if kind == "filter_cardinality":
        output.append(
            "filter={!r}; matches={}".format(item.get("pattern"), item.get("match_count"))
        )
    elif kind == "selector_coverage":
        output.append(
            "metric={!r}; coverage={}; missing_groups={}".format(
                item.get("metric_key"), item.get("coverage_ratio"), item.get("missing_groups")
            )
        )
    elif kind == "positional":
        output.append("label={!r}; raw={!r}".format(item.get("label"), item.get("raw")))
    elif kind == "semantic_alignment":
        output.append("semantic_score={}; labels={}".format(item.get("score"), item.get("labels")))
    return output


def _iter_finding_items(payload: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    findings = payload.get("findings")
    if isinstance(findings, list):
        return [item for item in findings if isinstance(item, dict) and "id" in item]
    # Risk-matrix reports use ``top``.  They are weak triage evidence only.
    top = payload.get("top")
    if isinstance(top, list):
        return [item for item in top if isinstance(item, dict) and "id" in item]
    return []


def _query_shape(code: str) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "lines": len(code.splitlines()),
        "source_calls": [],
        "operators": Counter(),
        "calls": Counter(),
        "parse_error": None,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        shape["parse_error"] = str(exc)
        return shape
    op_names = {
        ast.Add: "add",
        ast.Sub: "subtract",
        ast.Mult: "multiply",
        ast.Div: "divide",
        ast.Mod: "modulo",
        ast.Pow: "power",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            shape["operators"][op_names.get(type(node.op), type(node.op).__name__)] += 1
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = "<dynamic>"
            shape["calls"][name] += 1
            if name == "_source_value" and len(node.args) >= 3:
                values = []
                for arg in node.args[:3]:
                    values.append(arg.value if isinstance(arg, ast.Constant) else None)
                shape["source_calls"].append(
                    {"ticker": values[0], "year": values[1], "metric_key": values[2]}
                )
    shape["operators"] = dict(shape["operators"])
    shape["calls"] = dict(shape["calls"])
    return shape


def _manifest_features(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {"exists": False}, []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric_values: list[float] = []
    missing = 0
    negative = 0
    logical_keys: Counter[tuple[str, str, str]] = Counter()
    tables: set[str] = set()
    docs: set[str] = set()
    groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    scales: set[float] = set()
    writers: list[dict[str, Any]] = []
    for row in rows:
        ticker = str(row.get("ticker", ""))
        year = str(row.get("year", ""))
        metric = str(row.get("metric_key", ""))
        raw = row.get("raw")
        parsed = parse_btc_number(raw)
        try:
            scale = float(row.get("scale") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        if parsed is None:
            missing += 1
        else:
            # DictReader preserves the original token as text.  ``typed_factor``
            # only repairs a token that pandas has already inferred as numeric
            # at grader runtime; applying it again here would double-scale
            # values such as ``784.295``.
            value = parsed * scale
            numeric_values.append(value)
            negative += int(value < 0)
        source_table = str(row.get("source_table", ""))
        if source_table:
            tables.add(source_table)
            docs.add(source_table.rsplit("|", 1)[0])
        logical_keys[(ticker, year, metric)] += 1
        groups[(ticker, year)].add(metric)
        scales.add(scale)
        writers.append(
            {
                "ticker": ticker,
                "year": year,
                "metric_key": metric,
                "raw": raw,
                "scale": scale,
                "typed_factor": row.get("typed_factor"),
                "source_table": source_table,
                "source_csv": row.get("source_csv"),
                "row_idx": row.get("row_idx"),
                "col_idx": row.get("col_idx"),
            }
        )
    metric_sets = list(groups.values())
    union = set().union(*metric_sets) if metric_sets else set()
    intersection = set.intersection(*metric_sets) if metric_sets else set()
    coverage = (len(intersection) / len(union)) if union else 1.0
    features = {
        "exists": True,
        "cells": len(rows),
        "numeric_cells": len(numeric_values),
        "missing_cells": missing,
        "negative_cells": negative,
        "tables": len(tables),
        "documents": len(docs),
        "ticker_year_groups": len(groups),
        "metric_union": len(union),
        "metric_intersection": len(intersection),
        "metric_coverage": round(coverage, 6),
        "duplicate_logical_keys": sum(count - 1 for count in logical_keys.values() if count > 1),
        "distinct_scales": sorted(scales),
    }
    return features, writers


def _round_candidate(value: float) -> float:
    return round(float(value), 6)


def _counterfactuals(
    writers: Sequence[Mapping[str, Any]], intents: Sequence[str], current_answer: Any
) -> list[dict[str, Any]]:
    values = []
    for index, row in enumerate(writers):
        parsed = parse_btc_number(row.get("raw"))
        if parsed is None:
            continue
        try:
            scale = float(row.get("scale") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        values.append((index, parsed * scale))
    if not values or len(values) > 32:
        return []
    nums = [value for _, value in values]
    candidates: list[tuple[str, float, list[int]]] = []
    all_indices = [index for index, _ in values]
    if "sum" in intents or len(values) <= 8:
        candidates.extend(
            [
                ("signed_sum_all_cells", sum(nums), all_indices),
                ("absolute_sum_all_cells", sum(abs(value) for value in nums), all_indices),
            ]
        )
    if "mean" in intents:
        candidates.append(("mean_all_cells", statistics.mean(nums), all_indices))
    if "extreme" in intents:
        candidates.extend(
            [
                ("minimum_cell", min(nums), [values[nums.index(min(nums))][0]]),
                ("maximum_cell", max(nums), [values[nums.index(max(nums))][0]]),
            ]
        )
    if len(nums) == 2 or "difference" in intents:
        for left in range(min(len(nums), 8)):
            for right in range(left + 1, min(len(nums), 8)):
                candidates.append(
                    (
                        "absolute_difference_{}_{}".format(left, right),
                        abs(nums[left] - nums[right]),
                        [values[left][0], values[right][0]],
                    )
                )
    if "ratio" in intents or "growth" in intents:
        for left in range(min(len(nums), 6)):
            for right in range(min(len(nums), 6)):
                if left == right or nums[right] == 0:
                    continue
                ratio = nums[left] / nums[right]
                candidates.append(
                    (
                        "ratio_pct_{}_{}".format(left, right),
                        ratio * 100.0,
                        [values[left][0], values[right][0]],
                    )
                )
                if "growth" in intents:
                    candidates.append(
                        (
                            "growth_pct_{}_{}".format(left, right),
                            (ratio - 1.0) * 100.0,
                            [values[left][0], values[right][0]],
                        )
                    )
    try:
        current_numeric = float(current_answer)
    except (TypeError, ValueError):
        current_numeric = None
    unique: dict[tuple[str, float], dict[str, Any]] = {}
    for operation, value, indices in candidates:
        if not math.isfinite(value):
            continue
        rounded = _round_candidate(value)
        matches_current = current_numeric is not None and math.isclose(
            rounded, current_numeric, rel_tol=1e-9, abs_tol=1e-6
        )
        unique[(operation, rounded)] = {
            "operation": operation,
            "value": rounded,
            "writer_indices": indices,
            # A matching reconstruction is positive evidence that the reader
            # and writer agree.  Keeping it in the report also makes the
            # computation auditable; alternatives remain hypotheses only.
            "matches_current": matches_current,
        }
    # Keep the report bounded.  These are hypotheses for review, never answers.
    return list(unique.values())[:40]


@dataclass
class AuditSignal:
    source: str
    kind: str
    weight: float
    stale: bool
    reasons: list[str]


class AnswerForensicsEngine:
    """Build a durable, evidence-first review queue for one candidate."""

    def __init__(
        self,
        candidate: Path,
        audit_paths: Sequence[Path] = (),
        ledger_path: Optional[Path] = None,
        history_root: Optional[Path] = None,
        history_min_version: int = 161,
        review_log_path: Optional[Path] = None,
    ) -> None:
        self.candidate = candidate.resolve()
        self.submission_path = self.candidate / "submission.json"
        self.audit_paths = [path.resolve() for path in audit_paths]
        self.ledger_path = ledger_path.resolve() if ledger_path else None
        self.review_log_path = review_log_path.resolve() if review_log_path else None
        self.history_root = history_root.resolve() if history_root else self.candidate.parent
        self.history_min_version = history_min_version

    def _load_rows(self) -> list[dict[str, Any]]:
        payload = json.loads(self.submission_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("submission.json must contain a list")
        return payload

    def _load_ledger(self) -> dict[str, Any]:
        if not self.ledger_path or not self.ledger_path.exists():
            return {"schema_version": 1, "reviews": []}
        payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("forensics ledger must contain an object")
        return payload

    def _load_review_events(self) -> list[dict[str, Any]]:
        """Load trusted repo-local Vô Thượng reviews, ignoring malformed lines."""
        if not self.review_log_path or not self.review_log_path.exists():
            return []
        reviews: list[dict[str, Any]] = []
        for line in self.review_log_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(item, dict) or item.get("kind") != "review":
                continue
            # Older append-only events retain the trust value calculated when
            # they were written.  Re-evaluate the named oracle as a fallback
            # so newly recognized source aliases become useful retroactively
            # without rewriting history.
            if (
                item.get("oracle_trust") != "that"
                and oracle_trust(str(item.get("oracle", ""))) != "that"
            ):
                continue
            if _canonical_verdict(item.get("verdict")) not in REVIEWED_VERDICTS:
                continue
            reviews.append(item)
        return reviews

    @staticmethod
    def _row_signature(row: Mapping[str, Any]) -> str:
        material = {
            "answer": row.get("answer"),
            "pandas_query": row.get("pandas_query"),
            "relevant_docs": row.get("relevant_docs"),
            "relevant_tables": row.get("relevant_tables"),
            "evidence": row.get("evidence"),
        }
        return json.dumps(material, ensure_ascii=False, sort_keys=True)

    def _declared_rows(self, declared: str) -> Optional[dict[int, dict[str, Any]]]:
        candidates: list[Path] = []
        raw = Path(declared)
        candidates.append(raw if raw.name == "submission.json" else raw / "submission.json")
        match = re.search(r"(sub_top123_candidate_v\d+[^\\/]*)", declared)
        if match:
            candidates.append(self.history_root / match.group(1) / "submission.json")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return {int(row["id"]): row for row in payload if isinstance(row, dict) and "id" in row}
            except (OSError, ValueError, TypeError):
                continue
        return None

    def _audit_signals(self, current_by_id: Mapping[int, Mapping[str, Any]]) -> dict[int, list[AuditSignal]]:
        result: defaultdict[int, list[AuditSignal]] = defaultdict(list)
        candidate_name = self.candidate.name
        declared_cache: dict[str, Optional[dict[int, dict[str, Any]]]] = {}
        for path in self.audit_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            declared = str(payload.get("submission", "")) if isinstance(payload, dict) else ""
            report_is_current = not declared or candidate_name in declared
            if declared not in declared_cache and not report_is_current:
                declared_cache[declared] = self._declared_rows(declared)
            declared_rows = declared_cache.get(declared)
            kind = _audit_kind(path)
            # Some audits emit one finding per missing selector/cell.  Those
            # rows are supporting details, not independent votes.  Collapse
            # them so one report contributes at most one signal per question.
            per_question: dict[int, dict[str, Any]] = {}
            for item in _iter_finding_items(payload):
                try:
                    qid = int(item["id"])
                except (TypeError, ValueError):
                    continue
                weight = _finding_weight(kind, item)
                if weight <= 0.05:
                    continue
                entry = per_question.setdefault(qid, {"weight": 0.0, "reasons": []})
                entry["weight"] = max(float(entry["weight"]), weight)
                for reason in _finding_reasons(kind, item):
                    reason_text = str(reason)
                    if reason_text not in entry["reasons"]:
                        entry["reasons"].append(reason_text)
            for qid, entry in per_question.items():
                stale = not report_is_current
                if stale and declared_rows is not None and qid in declared_rows and qid in current_by_id:
                    stale = self._row_signature(declared_rows[qid]) != self._row_signature(current_by_id[qid])
                weight = float(entry["weight"])
                if stale:
                    weight *= 0.25
                result[qid].append(
                    AuditSignal(path.name, kind, round(weight, 3), stale, entry["reasons"][:8])
                )
        # Aliased/re-run reports from one audit family are corroborating detail,
        # not independent evidence.  Keep the strongest family vote per ID.
        collapsed: dict[int, list[AuditSignal]] = {}
        for qid, signals in result.items():
            by_kind: dict[str, AuditSignal] = {}
            for signal in signals:
                existing = by_kind.get(signal.kind)
                if existing is None:
                    by_kind[signal.kind] = signal
                    continue
                sources = sorted(set(existing.source.split(";") + signal.source.split(";")))
                reasons = list(existing.reasons)
                for reason in signal.reasons:
                    if reason not in reasons:
                        reasons.append(reason)
                if signal.weight > existing.weight:
                    existing.weight = signal.weight
                existing.source = ";".join(sources)
                existing.stale = existing.stale and signal.stale
                existing.reasons = reasons[:8]
            collapsed[qid] = list(by_kind.values())
        return collapsed

    def _history(self, ids: set[int]) -> dict[int, dict[str, Any]]:
        snapshots: list[tuple[int, str, Path]] = []
        for path in self.history_root.glob("sub_top123_candidate_v*"):
            if not path.is_dir() or not (path / "submission.json").exists():
                continue
            version = _version(path)
            if version is None or version < self.history_min_version:
                continue
            snapshots.append((version, path.name, path / "submission.json"))
        snapshots.sort(key=lambda item: (item[0], item[1]))
        seen: defaultdict[int, list[tuple[int, str, Any]]] = defaultdict(list)
        for version, name, path in snapshots:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for row in payload:
                try:
                    qid = int(row["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if qid in ids:
                    seen[qid].append((version, name, row.get("answer")))
        output: dict[int, dict[str, Any]] = {}
        for qid, values in seen.items():
            alternatives: dict[str, dict[str, Any]] = {}
            last_key: Optional[str] = None
            changes: list[dict[str, Any]] = []
            for version, name, answer in values:
                key = _answer_key(answer)
                entry = alternatives.setdefault(key, {"answer": answer, "versions": []})
                entry["versions"].append({"version": version, "candidate": name})
                if last_key is not None and key != last_key:
                    changes.append({"version": version, "candidate": name, "answer": answer})
                last_key = key
            output[qid] = {
                "snapshot_count": len(values),
                "unique_answers": list(alternatives.values()),
                "answer_change_count": len(changes),
                "changes": changes,
            }
        return output

    def run(self, top: int = 100) -> dict[str, Any]:
        rows = self._load_rows()
        candidate_sha = _sha256(self.submission_path)
        ids = {int(row["id"]) for row in rows}
        current_by_id = {int(row["id"]): row for row in rows}
        audit_by_id = self._audit_signals(current_by_id)
        history_by_id = self._history(ids)
        ledger = self._load_ledger()
        reviews_by_id = {
            int(item["id"]): item
            for item in ledger.get("reviews", [])
            if isinstance(item, dict) and "id" in item
        }
        for event in self._load_review_events():
            for raw_id in event.get("question_ids", []):
                try:
                    qid = int(raw_id)
                except (TypeError, ValueError):
                    continue
                reviews_by_id[qid] = {
                    "id": qid,
                    "verdict": event.get("verdict"),
                    "summary": event.get("summary", ""),
                    "evidence": event.get("evidence", []),
                    "source": "vothuong",
                    "event_id": event.get("id"),
                    "timestamp": event.get("timestamp"),
                }

        ranked: list[dict[str, Any]] = []
        reviewed: list[dict[str, Any]] = []
        for row in rows:
            qid = int(row["id"])
            question = str(row.get("question", ""))
            answer = row.get("answer")
            intents = detect_intents(question)
            shape = _query_shape(str(row.get("pandas_query", "")))
            features, writers = _manifest_features(self.candidate / "data" / "q{}_source_cells.csv".format(qid))
            audit_signals = audit_by_id.get(qid, [])
            history = history_by_id.get(qid, {"snapshot_count": 0, "unique_answers": [], "answer_change_count": 0, "changes": []})
            review = reviews_by_id.get(qid)

            reasons: list[str] = []
            priority = sum(signal.weight for signal in audit_signals)
            if len({signal.kind for signal in audit_signals}) >= 2:
                priority += 2.0
                reasons.append("multiple_independent_audit_families")
            if features.get("missing_cells", 0):
                priority += min(3.0, float(features["missing_cells"]))
                reasons.append("manifest_contains_missing_cells")
            if features.get("duplicate_logical_keys", 0):
                priority += 2.0
                reasons.append("duplicate_ticker_year_metric_keys")
            if features.get("ticker_year_groups", 0) >= 3 and features.get("metric_coverage", 1.0) < 0.5:
                priority += 1.0
                reasons.append("asymmetric_metric_coverage_review_only")
            if len(features.get("distinct_scales", [])) >= 2:
                priority += 1.0
                reasons.append("mixed_output_scales")
            if history.get("answer_change_count", 0):
                priority += min(3.0, float(history["answer_change_count"]))
                reasons.append("historically_changed_answer")
            if shape.get("parse_error"):
                priority += 10.0
                reasons.append("query_parse_error")
            requested = {
                (str(call.get("ticker")), str(call.get("year")), str(call.get("metric_key")))
                for call in shape.get("source_calls", [])
                if call.get("ticker") is not None and call.get("year") is not None and call.get("metric_key") is not None
            }
            present = {
                (str(item.get("ticker")), str(item.get("year")), str(item.get("metric_key")))
                for item in writers
            }
            unresolved_calls = sorted(requested - present)
            if unresolved_calls:
                priority += 8.0
                reasons.append("source_value_call_missing_from_manifest")

            record = {
                "id": qid,
                "priority": round(priority, 3),
                "question": question,
                "answer": answer,
                "intents": intents,
                "priority_reasons": reasons,
                "audit_signals": [signal.__dict__ for signal in audit_signals],
                "manifest": features,
                "reader": shape,
                "unresolved_source_calls": [list(value) for value in unresolved_calls],
                "history": history,
                "counterfactuals": _counterfactuals(writers, intents, answer),
                "writers": writers,
                "review": review,
                "claim_limit": "Triage only; verify against original BTC tables before changing an answer.",
            }
            if review and _canonical_verdict(review.get("verdict")) in REVIEWED_VERDICTS:
                reviewed.append(record)
            elif priority > 0:
                ranked.append(record)

        ranked.sort(key=lambda item: (-float(item["priority"]), int(item["id"])))
        reviewed.sort(key=lambda item: int(item["id"]))
        top_records = ranked[: max(0, top)]
        return {
            "schema_version": 1,
            "candidate": str(self.candidate),
            "candidate_submission_sha256": candidate_sha,
            "policy": {
                "purpose": "Prioritize source review, never infer hidden correctness.",
                "leaderboard_per_question_inference": "forbidden",
                "automatic_answer_mutation": False,
                "required_fix_evidence": "Original BTC table cells plus reproducible Pandas calculation.",
            },
            "counts": {
                "questions": len(rows),
                "audit_reports": len(self.audit_paths),
                "prioritized_unreviewed": len(ranked),
                "durably_reviewed": len(reviewed),
                "returned": len(top_records),
            },
            "top_ids": [item["id"] for item in top_records],
            "prioritized": top_records,
            "reviewed": reviewed,
        }


def render_markdown(report: Mapping[str, Any]) -> str:
    counts = report.get("counts", {})
    lines = [
        "# Answer Forensics Review Queue",
        "",
        "Generated from `{}`.".format(report.get("candidate", "")),
        "",
        "> This is a triage queue, not a correctness oracle. No answer may be changed without original BTC source evidence.",
        "",
        "- Questions: `{}`".format(counts.get("questions", 0)),
        "- Audit reports combined: `{}`".format(counts.get("audit_reports", 0)),
        "- Durable reviewed cases excluded: `{}`".format(counts.get("durably_reviewed", 0)),
        "",
        "| Rank | ID | Priority | Intent | Signals | Question |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for rank, item in enumerate(report.get("prioritized", []), 1):
        question = str(item.get("question", "")).replace("|", "\\|")
        if len(question) > 150:
            question = question[:147] + "..."
        lines.append(
            "| {} | {} | {:.3f} | {} | {} | {} |".format(
                rank,
                item.get("id"),
                float(item.get("priority", 0.0)),
                ", ".join(item.get("intents", [])) or "lookup",
                len(item.get("audit_signals", [])),
                question,
            )
        )
    lines.extend(
        [
            "",
            "## Durable reviews",
            "",
            "| ID | Verdict | Evidence summary |",
            "|---:|---|---|",
        ]
    )
    for item in report.get("reviewed", []):
        review = item.get("review") or {}
        summary = str(review.get("summary", "")).replace("|", "\\|")
        lines.append("| {} | {} | {} |".format(item.get("id"), review.get("verdict", ""), summary))
    return "\n".join(lines) + "\n"
