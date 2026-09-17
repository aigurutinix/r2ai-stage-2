from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("build_compliant_standard_candidate_receivable_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q35_remains_gross_two_row_formula_for_reproducibility() -> None:
    formula = module.FORMULAS[35]
    assert len(formula.operands) == 2
    assert formula.expression == "(v0 + v1) / 1e6"


def test_summary_variant_uses_direct_counterparty_balance() -> None:
    formula = module.FORMULA_VARIANTS["receivable-summary"][35]
    assert len(formula.operands) == 1
    operand = formula.operands[0]
    cell = module.EXPLICIT_CELLS[(operand.ticker, operand.year, operand.metric_key, operand.scope)]

    assert cell.raw == "222.575.005.778"
    assert cell.table_ref == "BVH_financial_statements_2015_separate|1232"
    assert cell.row_idx == 2
    assert cell.col_idx == 1
    assert formula.expression == "v0 / 1e6"
