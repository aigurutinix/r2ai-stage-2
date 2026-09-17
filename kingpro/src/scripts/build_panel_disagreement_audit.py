"""Build a durable, source-linked audit for independent panel disagreements.

The panel solver is a useful second implementation, not an oracle.  This tool
therefore separates a numerical disagreement from mutation authority and adds
structural warnings for common reviewer failures (returning an intermediate
metric, changing a difference into a ratio, or using a different source
snapshot).  It never edits a submission.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def query_tail(code: str) -> str:
    marker = "df = pd.DataFrame(_rows)"
    position = code.rfind(marker)
    return code[position:] if position >= 0 else code


def normalized_logic(code: str) -> str:
    lines = []
    for line in query_tail(code).splitlines():
        compact = re.sub(r"\s+", "", line)
        if compact and not compact.startswith("result=float(result)"):
            lines.append(compact)
    return "\n".join(lines)


def plausible(current: float, alternative: float) -> bool:
    if not math.isfinite(current) or not math.isfinite(alternative):
        return False
    if current == alternative:
        return False
    if current == 0.0 or alternative == 0.0:
        return True
    ratio = abs(alternative / current)
    return 0.05 <= ratio <= 20.0


def probability_for(question: dict[str, Any], answer: float, tolerance: float) -> float:
    for candidate in question.get("candidates", []):
        if math.isclose(float(candidate["answer"]), answer, rel_tol=0.0, abs_tol=tolerance):
            return float(candidate["probability"])
    return 0.0


def semantic_alerts(question: str, current_code: str, panel_code: str) -> list[str]:
    text = question.casefold()
    current = normalized_logic(current_code)
    panel = normalized_logic(panel_code)
    alerts: list[str] = []
    similarity = difflib.SequenceMatcher(None, current, panel).ratio()
    if similarity >= 0.92:
        alerts.append("same_logic_different_answer_source_snapshot_or_rounding")
    if "chênh lệch" in text and "/" in panel and "mean" in panel:
        alerts.append("panel_may_replace_difference_with_ratio")
    terminal = "\n".join(line for line in panel_code.splitlines() if "result" in line)
    if "thanh toán nhanh" in text and "operating_cash_flow_ratio" in terminal and "quick_ratio" not in terminal:
        alerts.append("panel_returns_intermediate_instead_of_requested_quick_ratio")
    if ("bao nhiêu doanh nghiệp" in text or "có bao nhiêu" in text) and not any(
        token in panel for token in ("shape[0]", "len(", ".count(", ".nunique(")
    ):
        alerts.append("panel_count_question_without_count_terminal")
    if "bình quân" in text and not any(token in panel for token in ("mean(", "average", "sum(")):
        alerts.append("panel_average_question_without_aggregation")
    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3",
    )
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "build/panel_answers_v4.jsonl"
    )
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=ROOT / "build/v227_public_ensemble_v217_updated.json",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=ROOT / "knowledge/vothuong/panel_disagreement_verdicts.json",
    )
    parser.add_argument("--tolerance", type=float, default=0.011)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/v227_panel_disagreement_audit.json",
    )
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    entries = {int(row["id"]): row for row in load_json(candidate / "submission.json")}
    if len(entries) != 1012:
        raise SystemExit(f"candidate must have 1,012 unique IDs, got {len(entries)}")
    panel = latest_jsonl(args.panel)
    ensemble_payload = load_json(args.ensemble)
    ensemble = {
        int(row["id"]): row for row in ensemble_payload.get("question_predictions", [])
    }
    verdict_payload = load_json(args.verdicts) if args.verdicts.exists() else {}
    verdicts = {
        int(qid): row for qid, row in (verdict_payload.get("verdicts") or {}).items()
    }

    disagreements: list[dict[str, Any]] = []
    panel_ok = 0
    numerical_disagreements = 0
    for qid, reviewed in sorted(panel.items()):
        if not reviewed.get("ok") or qid not in entries:
            continue
        panel_ok += 1
        current_row = entries[qid]
        current = float(current_row["answer"])
        alternative = float(reviewed["answer"])
        scale = max(1.0, abs(current), abs(alternative))
        if abs(current - alternative) <= 0.005 or abs(current - alternative) / scale <= 1e-6:
            continue
        numerical_disagreements += 1
        if not plausible(current, alternative):
            continue
        current_code = str(current_row.get("pandas_query") or "")
        panel_code = str(reviewed.get("code") or "")
        prediction = ensemble.get(qid, {})
        alerts = semantic_alerts(str(current_row.get("question") or ""), current_code, panel_code)
        evidence = []
        for item in current_row.get("evidence") or []:
            csv_path = candidate / str(item.get("csv_path") or "")
            evidence.append(
                {
                    "variable": item.get("variable"),
                    "path": relative(csv_path),
                    "exists": csv_path.exists(),
                    "bytes": csv_path.stat().st_size if csv_path.exists() else None,
                }
            )
        similarity = difflib.SequenceMatcher(
            None, normalized_logic(current_code), normalized_logic(panel_code)
        ).ratio()
        verdict = verdicts.get(qid)
        disagreements.append(
            {
                "id": qid,
                "question": current_row.get("question"),
                "current_answer": current,
                "panel_answer": alternative,
                "answer_ratio": None if current == 0 else alternative / current,
                "logic_similarity": round(similarity, 4),
                "semantic_alerts": alerts,
                "current_ensemble_probability": probability_for(
                    prediction, current, args.tolerance
                ),
                "panel_ensemble_probability": probability_for(
                    prediction, alternative, args.tolerance
                ),
                "none_probability": prediction.get("none_probability"),
                "current_logic": query_tail(current_code),
                "panel_logic": panel_code,
                "relevant_tables": current_row.get("relevant_tables") or [],
                "evidence": evidence,
                "status": verdict.get("verdict") if verdict else "needs_source_adjudication",
                "verdict_reason": verdict.get("reason") if verdict else None,
                "mutation_authority": bool(
                    verdict and verdict.get("verdict") == "use_panel_alternative"
                ),
            }
        )

    disagreements.sort(
        key=lambda row: (
            -len(row["semantic_alerts"]),
            -row["logic_similarity"],
            row["id"],
        )
    )
    payload = {
        "schema_version": 1,
        "kind": "independent_panel_disagreement_audit",
        "candidate": relative(candidate),
        "panel": relative(args.panel),
        "ensemble": relative(args.ensemble),
        "verdicts": relative(args.verdicts),
        "policy": {
            "panel_is_oracle": False,
            "automatic_answer_changes": False,
            "source_adjudication_required": True,
        },
        "counts": {
            "candidate_questions": len(entries),
            "panel_latest_records": len(panel),
            "panel_ok": panel_ok,
            "numerical_disagreements": numerical_disagreements,
            "plausible_disagreements": len(disagreements),
            "with_semantic_alerts": sum(bool(row["semantic_alerts"]) for row in disagreements),
            "adjudicated": sum(row["status"] != "needs_source_adjudication" for row in disagreements),
            "panel_alternatives_authorized": sum(row["mutation_authority"] for row in disagreements),
        },
        "disagreements": disagreements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "ids": [row["id"] for row in disagreements],
                "out": relative(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
