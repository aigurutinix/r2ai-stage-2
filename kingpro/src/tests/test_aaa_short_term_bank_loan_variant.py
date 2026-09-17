from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_aaa_short_bank", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q354_remains_reproducible() -> None:
    # q354 remains a legacy row unless this opt-in variant is requested.
    assert 354 not in module.FORMULAS


def test_aaa_short_bank_variant_reads_direct_short_term_balance() -> None:
    formula = module.FORMULA_VARIANTS["aaa-short-bank-2021"][354]
    assert len(formula.operands) == 1
    operand = formula.operands[0]
    cell = module.EXPLICIT_CELLS[
        (operand.ticker, operand.year, operand.metric_key, operand.scope)
    ]

    assert cell.raw == "1.401.195.977.583"
    assert cell.table_ref == "AAA_financial_statements_2021_separate|1058"
    assert cell.row_idx == 3
    assert cell.col_idx == 7
    assert formula.expression == "v0 / 1e11"
    assert round(cell.value / 1e11, 2) == 14.01
