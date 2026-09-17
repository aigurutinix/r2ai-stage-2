import json

from vifinqa.validation.audit_formula_eval import audit_formula_eval


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def test_formula_audit_separates_retrieval_coverage_and_solver_accuracy(tmp_path):
    gold_path = tmp_path / "gold.json"
    retrieval_path = tmp_path / "retrieval.jsonl"
    codegen_path = tmp_path / "codegen.jsonl"
    gold_path.write_text(json.dumps({
        "1": {"klass": "ratio", "answer": 2.0,
              "operands": [{"report_id": "A", "table_pos": 1}]},
        "2": {"klass": "ratio", "answer": 3.0,
              "operands": [{"report_id": "B", "table_pos": 2}]},
    }), encoding="utf-8")
    _write_jsonl(retrieval_path, [
        {"id": 1, "candidates": [{"report_id": "A", "table_pos": 1}]},
        {"id": 2, "candidates": [{"report_id": "X", "table_pos": 9}]},
    ])
    _write_jsonl(codegen_path, [
        {"id": 1, "status": "ok", "answer": 2.0, "source": "rule_formula"},
        {"id": 2, "status": "failed", "source": "none", "detail": "missing"},
    ])

    report = audit_formula_eval(gold_path, retrieval_path, codegen_path, k=1)
    stats = report["per_class"]["ratio"]

    assert stats["retrieval_recall"] == 0.5
    assert stats["retrieval_complete"] == 0.5
    assert stats["solver_coverage"] == 0.5
    assert stats["answer_acc"] == 0.5
    assert stats["acc_when_solved"] == 1.0
