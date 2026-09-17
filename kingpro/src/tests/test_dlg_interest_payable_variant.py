from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_dlg_interest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q26_remains_legacy() -> None:
    assert 26 not in module.FORMULAS


def test_dlg_interest_variant_reads_ending_balance_not_related_transaction() -> None:
    formula = module.FORMULA_VARIANTS["dlg-interest-payable-ending"][26]
    assert formula.expression == "v0 / 1e6"
    assert len(formula.operands) == 1
    operand = formula.operands[0]
    cell = module.EXPLICIT_CELLS[
        (operand.ticker, operand.year, operand.metric_key, operand.scope)
    ]

    assert cell.raw == "350.187.565.073"
    assert cell.table_ref == "DLG_financial_statements_2023_consolidated|1455"
    assert (cell.row_idx, cell.col_idx) == (1, 1)
    assert round(cell.value / 1e6, 2) == 350_187.57
