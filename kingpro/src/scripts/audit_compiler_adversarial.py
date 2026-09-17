"""Audit fail-closed behavior of the deterministic financial compiler.

The registry audit measures precision on questions the compiler accepts. This
companion audit probes deliberately deceptive variants that preserve a valid
company/year and often a valid source cell, but change the requested time basis,
scope, aggregation, unit or business meaning. Every negative must be refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler  # noqa: E402
from kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402


NEGATIVE_CASES: tuple[tuple[str, str], ...] = (
    ("beginning_date_slash", "Số dư tiền và tương đương tiền của DTK tại ngày 01/01/2024 là bao nhiêu tỷ đồng?"),
    ("beginning_date_short", "Số dư tiền và tương đương tiền của DTK tại ngày 1/1/2024 là bao nhiêu tỷ đồng?"),
    ("beginning_date_dash", "Số dư tiền và tương đương tiền của DTK tại ngày 01-01-2024 là bao nhiêu tỷ đồng?"),
    ("beginning_date_words", "Số dư tiền và tương đương tiền của DTK ngày 1 tháng 1 năm 2024 là bao nhiêu tỷ đồng?"),
    ("beginning_period", "Tiền và tương đương tiền đầu kỳ năm 2024 của DTK là bao nhiêu tỷ đồng?"),
    ("beginning_balance", "Số đầu năm 2024 của tiền và tương đương tiền DTK là bao nhiêu tỷ đồng?"),
    ("scenario_assumption", "Giả sử chi phí không đổi, lợi nhuận thuần từ hoạt động tài chính của KBC năm 2015 là bao nhiêu tỷ đồng?"),
    ("scenario_label", "Theo kịch bản bất lợi, lợi nhuận thuần từ hoạt động tài chính của KBC năm 2015 là bao nhiêu tỷ đồng?"),
    ("conditional", "Nếu doanh thu tăng, lợi nhuận thuần từ hoạt động tài chính của KBC năm 2015 là bao nhiêu tỷ đồng?"),
    ("next_year", "Thu nhập khác thuần của HSG vào năm sau năm 2021 là bao nhiêu triệu đồng?"),
    ("next_period", "Thu nhập khác thuần của HSG ở năm kế tiếp sau năm 2021 là bao nhiêu triệu đồng?"),
    ("shareholder_note", "Vốn cổ phần của cổ đông Nguyễn Văn A tại VGT năm 2024 là bao nhiêu tỷ đồng?"),
    ("project_note", "Tiền và tương đương tiền của dự án A tại DTK năm 2024 là bao nhiêu tỷ đồng?"),
    ("related_party_note", "Tiền và tương đương tiền của bên liên quan tại DTK năm 2024 là bao nhiêu tỷ đồng?"),
    ("segment_note", "Lợi nhuận tài chính ròng của bộ phận điện tại KBC năm 2015 là bao nhiêu tỷ đồng?"),
    ("industry_note", "Tiền và tương đương tiền theo ngành tại DTK năm 2024 là bao nhiêu tỷ đồng?"),
    ("subsidiary_qualifier", "Vốn cổ phần thuộc công ty con A của VGT năm 2024 là bao nhiêu tỷ đồng?"),
    ("raw_metric_ratio", "Tỷ lệ vốn cổ phần trên tổng tài sản của VGT năm 2024 là bao nhiêu phần trăm?"),
    ("raw_cash_ratio", "Tỷ trọng tiền và tương đương tiền trên tổng tài sản của DTK năm 2024 là bao nhiêu phần trăm?"),
    ("ambiguous_thousand_vnd", "Tiền và tương đương tiền của DTK năm 2024 là bao nhiêu nghìn đồng?"),
    ("wrong_share_unit", "Vốn cổ phần của VGT năm 2024 là bao nhiêu cổ phiếu?"),
    ("comparison_hidden_entity", "Vốn cổ phần VGT năm 2024 chênh lệch bao nhiêu so với doanh nghiệp đối thủ?"),
    ("comparison_explicit", "So sánh vốn cổ phần VGT năm 2024 với doanh nghiệp đối thủ."),
    ("between_entities", "Chênh lệch vốn cổ phần giữa VGT và doanh nghiệp đối thủ năm 2024 là bao nhiêu?"),
    ("average_request", "Vốn cổ phần trung bình của VGT năm 2024 là bao nhiêu tỷ đồng?"),
    ("extreme_request", "Vốn cổ phần cao nhất của VGT năm 2024 là bao nhiêu tỷ đồng?"),
    ("deferred_tax", "Chi phí thuế thu nhập hoãn lại của NLG năm 2019 là bao nhiêu tỷ đồng?"),
    ("undistributed_profit", "Lợi nhuận sau thuế chưa phân phối của VIF năm 2021 là bao nhiêu tỷ đồng?"),
    ("sensitivity_table", "Theo bảng độ nhạy, tiền và tương đương tiền của DTK năm 2024 là bao nhiêu tỷ đồng?"),
)

POSITIVE_CONTROL_IDS = (12, 84, 236, 363, 666, 682, 715)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit(*, root: Path, registry: Path) -> dict:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    compiler = DeterministicFinancialCompiler(root)

    positive_failures: list[dict] = []
    for question_id in POSITIVE_CONTROL_IDS:
        row = by_id.get(question_id)
        if row is None:
            positive_failures.append({"id": question_id, "reason": "missing_registry_row"})
            continue
        question = str(row.get("question", ""))
        compiled = compiler.compile(question, extract_all_facets(question))
        if compiled is None:
            positive_failures.append({"id": question_id, "reason": "unexpected_refusal"})

    negative_acceptances: list[dict] = []
    category_counts: dict[str, int] = {}
    for category, question in NEGATIVE_CASES:
        category_counts[category] = category_counts.get(category, 0) + 1
        compiled = compiler.compile(question, extract_all_facets(question))
        if compiled is not None:
            negative_acceptances.append(
                {
                    "category": category,
                    "question": question,
                    "metric": compiled.metric,
                    "source_cells": len(compiled.source_cells),
                }
            )

    passed = not positive_failures and not negative_acceptances
    return {
        "passed": passed,
        "input_hashes": {
            "registry_sha256": _sha256(registry),
            "statement_cube_sha256": _sha256(root / "build" / "statement_cube.jsonl"),
            "compiler_sha256": _sha256(
                root / "src" / "kingpro" / "product" / "deterministic_compiler.py"
            ),
        },
        "positive_controls": len(POSITIVE_CONTROL_IDS),
        "positive_failures": positive_failures,
        "adversarial_cases": len(NEGATIVE_CASES),
        "adversarial_rejected": len(NEGATIVE_CASES) - len(negative_acceptances),
        "negative_acceptances": negative_acceptances,
        "category_counts": category_counts,
        "scope_note": (
            "Finite local adversarial suite for known compiler boundary classes; "
            "not a proof against every possible natural-language attack."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio" / "submission.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args()
    report = audit(root=args.root.resolve(), registry=args.registry.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] or not args.require_perfect else 2


if __name__ == "__main__":
    raise SystemExit(main())
