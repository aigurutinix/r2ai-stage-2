from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_vic_cip", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q244_remains_legacy() -> None:
    assert 244 not in module.FORMULAS


def test_vic_cip_variant_reads_consolidated_ending_balance() -> None:
    formula = module.FORMULA_VARIANTS["vic-ending-long-term-cip"][244]
    assert formula.expression == "v0 / 1e11"
    assert len(formula.operands) == 1
    operand = formula.operands[0]
    cell = module.EXPLICIT_CELLS[
        (operand.ticker, operand.year, operand.metric_key, operand.scope)
    ]

    assert cell.raw == "33.991.567.265.462"
    assert cell.table_ref == "VIC_financial_statements_2016_consolidated|209"
    assert (cell.row_idx, cell.col_idx) == (16, 3)
    assert round(cell.value / 1e11, 2) == 339.92
