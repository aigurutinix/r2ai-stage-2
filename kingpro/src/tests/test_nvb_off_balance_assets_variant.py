from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_nvb_q717", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q717_remains_legacy() -> None:
    assert 717 not in module.FORMULAS


def test_nvb_q717_variant_uses_same_period_primary_total_assets() -> None:
    formula = module.FORMULA_VARIANTS["nvb-off-balance-assets-2019"][717]
    assert formula.expression == "v0 / v1 * 100"
    assert len(formula.operands) == 2
    cells = [
        module.EXPLICIT_CELLS[
            (operand.ticker, operand.year, operand.metric_key, operand.scope)
        ]
        for operand in formula.operands
    ]

    assert cells[0].raw == "12.053.691"
    assert cells[0].table_ref == "NVB_financial_statements_2019_separate|1423"
    assert (cells[0].row_idx, cells[0].col_idx) == (12, 1)
    assert cells[1].raw == "80.405.111"
    assert cells[1].table_ref == "NVB_financial_statements_2019_separate|268"
    assert (cells[1].row_idx, cells[1].col_idx) == (21, 2)
    assert round(cells[0].value / cells[1].value * 100, 2) == 14.99
