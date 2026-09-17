"""Stage-level diagnostics for the formula-specific offline evaluation set."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..utils.io import read_json, read_jsonl, write_json


def audit_formula_eval(gold_path: Path, retrieval_path: Path, codegen_path: Path,
                       k: int = 15, out_path: Path | None = None) -> dict:
    gold = read_json(gold_path)
    retrieval = {int(row["id"]): row for row in read_jsonl(retrieval_path)}
    codegen = {int(row["id"]): row for row in read_jsonl(codegen_path)}

    classes: dict[str, dict] = {}
    for qid_text, expected in gold.items():
        qid = int(qid_text)
        klass = expected.get("klass", "unknown")
        stats = classes.setdefault(klass, {
            "n": 0, "retrieval_recall_sum": 0.0, "retrieval_complete": 0,
            "solver_ok": 0, "answer_correct": 0,
            "sources": Counter(), "failures": Counter(),
        })
        stats["n"] += 1

        wanted = {
            (str(item["report_id"]), int(item["table_pos"]))
            for item in expected.get("operands", [])
        }
        candidates = retrieval.get(qid, {}).get("candidates", [])[:k]
        found = {
            (str(item["report_id"]), int(item["table_pos"]))
            for item in candidates
        }
        recall = len(wanted & found) / len(wanted) if wanted else 0.0
        stats["retrieval_recall_sum"] += recall
        stats["retrieval_complete"] += int(bool(wanted) and wanted <= found)

        prediction = codegen.get(qid, {})
        source = str(prediction.get("source", "missing"))
        stats["sources"][source] += 1
        if prediction.get("status") == "ok":
            stats["solver_ok"] += 1
            answer = round(float(prediction.get("answer", 0.0)), 2)
            target = round(float(expected["answer"]), 2)
            stats["answer_correct"] += int(abs(answer - target) <= 0.010000001)
        else:
            detail = str(prediction.get("detail", "missing result"))
            stats["failures"][detail] += 1

    report = {"k": k, "n": sum(item["n"] for item in classes.values()),
              "per_class": {}}
    for klass, stats in sorted(classes.items()):
        n = max(stats["n"], 1)
        solver_ok = stats["solver_ok"]
        report["per_class"][klass] = {
            "n": stats["n"],
            "retrieval_recall": round(stats["retrieval_recall_sum"] / n, 4),
            "retrieval_complete": round(stats["retrieval_complete"] / n, 4),
            "solver_coverage": round(solver_ok / n, 4),
            "answer_acc": round(stats["answer_correct"] / n, 4),
            "acc_when_solved": round(
                stats["answer_correct"] / max(solver_ok, 1), 4),
            "sources": dict(stats["sources"]),
            "top_failures": [
                {"detail": detail, "n": count}
                for detail, count in stats["failures"].most_common(3)
            ],
        }

    print(f"  {'class':24} {'n':>4} {'R@k':>7} {'all':>7} "
          f"{'solve':>7} {'answer':>7} {'ans|ok':>7}")
    for klass, stats in report["per_class"].items():
        print(f"  {klass:24} {stats['n']:4} "
              f"{stats['retrieval_recall']:7.3f} "
              f"{stats['retrieval_complete']:7.3f} "
              f"{stats['solver_coverage']:7.3f} "
              f"{stats['answer_acc']:7.3f} "
              f"{stats['acc_when_solved']:7.3f}")
    if out_path:
        write_json(out_path, report)
        print(f"[formula-audit] report -> {out_path}")
    return report
